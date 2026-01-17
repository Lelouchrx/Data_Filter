import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from collections import deque, Counter
import time
from transformers import pipeline
from PIL import Image

class DepthEstimator:
    def __init__(self, device='cuda'):
        """
        初始化深度估计模型 (使用 Depth Anything V2 Small)
        """
        print(f"[Init] Loading Depth Anything V2 (Small) on {device}...")
        # 使用 Hugging Face 的 pipeline，自动下载并加载模型
        # depth-anything/Depth-Anything-V2-Small-hf 是目前最快的小模型
        self.pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
        
    def infer(self, frame_bgr):
        """
        输入 BGR 图像，输出归一化的深度图 (0.0 - 1.0, 越近越亮/值越大)
        """
        # OpenCV BGR -> PIL RGB
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        
        # 推理
        depth_output = self.pipe(image)
        depth_map = depth_output["depth"] # 获取 PIL Image 格式的深度图
        
        # 转回 numpy 数组并归一化
        depth_np = np.array(depth_map)
        
        # 归一化到 0.0 - 1.0 (Min-Max Normalization)
        # 注意：Depth Anything 输出的是相对深度（Disparity），值越大代表越近
        d_min = depth_np.min()
        d_max = depth_np.max()
        norm_depth = (depth_np - d_min) / (d_max - d_min + 1e-6)
        
        return norm_depth

class HandObjectInteractionSystem:
    def __init__(self, model_size='yolov8s.pt', history_len=6, visualize=True):
        """
        初始化 HOI 检测系统
        :param model_size: YOLO 模型大小 (建议 'yolov8s.pt' 平衡速度与精度)
        :param history_len: 时序平滑的窗口大小 (报告 7.1 节建议)
        :param visualize: 是否开启可视化
        """
        print(f"[Init] Loading YOLO model: {model_size}...")
        self.yolo_model = YOLO(model_size)
        self.visualize = visualize
        
        # MediaPipe 初始化
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # 时序平滑缓冲池 (Key: Hand_ID, Value: Deque of states)
        # 用来存储每只手过去 N 帧的状态
        self.history_len = history_len
        self.state_buffers = {0: deque(maxlen=history_len), 1: deque(maxlen=history_len)}
        
        # 预定义参数 (报告 9.1 节)
        self.GRASP_SPREAD_THRESHOLD = 0.40  # 手指张开度阈值 (区分 Open/Closed)
        self.OBJ_PADDING = 0.10            # 物体框扩充比例 (解决边缘接触问题)
        
        # === 【新增】初始化深度模型 ===
        # 如果你是 M1/M2/M3 Mac，这里可以用 'mps' 加速；如果是 N卡用 'cuda'；否则用 'cpu'
        self.depth_model = DepthEstimator(device='cuda') 
        
        # === 【修改】因为指尖取样更准了，我们可以收紧阈值，让判定更严格、更精准 ===
        self.DEPTH_THRESHOLD = 0.25

    def _get_expanded_box(self, box, frame_w, frame_h):
        """
        扩充物体边界框，增加对边缘抓取的容忍度 (报告 2.1 非凸几何失配)
        """
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        pad_w = w * self.OBJ_PADDING
        pad_h = h * self.OBJ_PADDING
        
        nx1 = max(0, x1 - pad_w)
        ny1 = max(0, y1 - pad_h)
        nx2 = min(frame_w, x2 + pad_w)
        ny2 = min(frame_h, y2 + pad_h)
        
        return int(nx1), int(ny1), int(nx2), int(ny2)

    def _is_point_in_box(self, point, box):
        """
        核心几何判定：点是否在框内 (Keypoint Gating)
        point: (x, y) 像素坐标
        box: (x1, y1, x2, y2)
        """
        px, py = point
        x1, y1, x2, y2 = box
        return x1 <= px <= x2 and y1 <= py <= y2

    def _get_region_depth(self, depth_map, box):
        """
        获取指定区域的深度值（取中位数，更鲁棒）
        box: (x1, y1, x2, y2) 像素坐标
        """
        x1, y1, x2, y2 = box
        # 确保坐标在有效范围内
        h, w = depth_map.shape
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        # 提取区域深度
        region = depth_map[y1:y2+1, x1:x2+1]
        if region.size == 0:
            return 0.5  # 默认值
        
        # 返回中位数（比平均值更鲁棒，不受异常值影响）
        return np.median(region)

    def _analyze_hand_geometry(self, landmarks):
        """
        分析手部几何特征：返回关键点坐标和张开度
        """
        # 提取关键点 (归一化坐标 -> 像素坐标转换在外部做)
        # 关注点：拇指指尖(4), 食指指尖(8) -> 用于接触检测
        # 关注点：所有指尖 -> 用于计算张开度
        
        keypoints = {
            'wrist': landmarks[0],
            'thumb_tip': landmarks[4],
            'index_tip': landmarks[8],
            'middle_tip': landmarks[12],
            'ring_tip': landmarks[16],
            'pinky_tip': landmarks[20]
        }
        
        # 计算张开度 (所有指尖到手腕的平均归一化距离)
        wrist_vec = np.array([keypoints['wrist'].x, keypoints['wrist'].y])
        tips_vecs = [
            np.array([lm.x, lm.y]) for lm in 
            [keypoints['index_tip'], keypoints['middle_tip'], keypoints['ring_tip'], keypoints['pinky_tip']]
        ]
        avg_spread = np.mean([np.linalg.norm(t - wrist_vec) for t in tips_vecs])
        
        is_open = avg_spread > self.GRASP_SPREAD_THRESHOLD
        
        return keypoints, is_open, avg_spread

    def _smooth_state(self, hand_id, current_state):
        """
        时序平滑逻辑 (报告 7.1)
        使用投票机制滤除单帧抖动
        """
        self.state_buffers[hand_id].append(current_state)
        
        # 只有当缓冲区填满一半以上时才开始平滑
        if len(self.state_buffers[hand_id]) < 2:
            return current_state
            
        # 统计出现次数最多的状态
        counter = Counter(self.state_buffers[hand_id])
        most_common_state, count = counter.most_common(1)[0]
        
        return most_common_state

    # === 【修改 1】增加一个日志写入辅助函数 ===
    def _log_data(self, frame_idx, hand_id, obj_name, hand_z, obj_z, diff, result, reason):
        """
        把关键数据写入 CSV 文件
        """
        with open("hoi_debug_log.csv", "a") as f:
            # 格式: 帧号, 手ID, 物体名, 手深度, 物体深度, 深度差, 最终判定, 拒绝原因
            f.write(f"{frame_idx},{hand_id},{obj_name},{hand_z:.4f},{obj_z:.4f},{diff:.4f},{result},{reason}\n")

    # === 【修改 2】重写 process_frame 以支持深度可视化和日志 ===
    def process_frame(self, frame, frame_idx=0): # 增加 frame_idx 参数
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display_frame = frame.copy() 

        # 1. 计算深度并生成可视化图 (Heatmap)
        norm_depth_map = self.depth_model.infer(frame)
        
        # 将深度图转为彩色热力图 (方便肉眼观察)
        # 0 (远/黑) -> 255 (近/白)
        depth_uint8 = (norm_depth_map * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)

        # 2. 运行 MediaPipe 和 YOLO
        mp_results = self.hands.process(frame_rgb)
        yolo_results = self.yolo_model(frame, verbose=False, conf=0.15)
        
        # ... (YOLO 解析代码，提取 detected_objects 保持不变) ...
        detected_objects = []
        for box in yolo_results[0].boxes:
            cls_id = int(box.cls[0]); 
            if cls_id == 0: continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = self.yolo_model.names[cls_id]
            ex_box = self._get_expanded_box((x1, y1, x2, y2), w, h)
            # 保存原始框用于提取深度，保存扩充框用于几何判定
            detected_objects.append({'box': ex_box, 'label': label, 'raw_box': (x1, y1, x2, y2)})
            if self.visualize:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # 3. HOI 逻辑核心
        if mp_results.multi_hand_landmarks:
            for i, hand_lms in enumerate(mp_results.multi_hand_landmarks):
                # 几何分析
                kpts, is_open, spread_val = self._analyze_hand_geometry(hand_lms.landmark)
                thumb_px = (int(kpts['thumb_tip'].x * w), int(kpts['thumb_tip'].y * h))
                index_px = (int(kpts['index_tip'].x * w), int(kpts['index_tip'].y * h))
                wrist_px = (int(kpts['wrist'].x * w), int(kpts['wrist'].y * h))

                # === 【新代码 (替换为手指中点)】 ===
                # 1. 计算交互中心 (拇指和食指的中点)
                # 理论上，抓取时这个点应该紧贴物体表面
                inter_x = int((thumb_px[0] + index_px[0]) / 2)
                inter_y = int((thumb_px[1] + index_px[1]) / 2)
                interaction_point = (inter_x, inter_y)

                # 2. 在这个中点周围取样深度
                # 框可以稍微小一点，因为指尖区域比较窄
                hand_depth_box = (inter_x-10, inter_y-10, inter_x+10, inter_y+10)
                hand_z = self._get_region_depth(norm_depth_map, hand_depth_box)

                # 3. 可视化：画个青色的圈，让你看到它现在是在"盯着"手指看
                cv2.circle(depth_colormap, interaction_point, 8, (255, 255, 0), 2) 
                # 同时也画在左边的原图上，方便确认位置
                cv2.circle(display_frame, interaction_point, 5, (0, 255, 255), -1)

                # 判定逻辑
                contact_obj = None
                reject_reason = "None" # 用于日志
                depth_diff_log = 0.0
                obj_z_log = 0.0  # 用于记录物体的深度值

                for obj in detected_objects:
                    obox = obj['box']
                    # === 第一关：2D 几何检查 ===
                    if self._is_point_in_box(thumb_px, obox) or self._is_point_in_box(index_px, obox):
                        # === 第二关：深度一致性检查 ===
                        # 缩小一点物体的深度取样框 (Center Cropping)，防止取到边缘背景
                        ox1, oy1, ox2, oy2 = obj['raw_box']
                        ow, oh = ox2-ox1, oy2-oy1
                        # 只取中间 50% 的区域算深度，这样更准
                        center_crop = (ox1+int(ow*0.25), oy1+int(oh*0.25), ox2-int(ow*0.25), oy2-int(oh*0.25))
                        
                        obj_z = self._get_region_depth(norm_depth_map, center_crop)
                        obj_z_log = obj_z  # 保存物体的深度值用于日志
                        depth_diff = abs(hand_z - obj_z)
                        depth_diff_log = depth_diff # 记录下来

                        # 画出物体的取样框
                        cv2.rectangle(depth_colormap, (center_crop[0], center_crop[1]), (center_crop[2], center_crop[3]), (0, 255, 255), 1)

                        if depth_diff < self.DEPTH_THRESHOLD:
                            contact_obj = obj['label']
                            reject_reason = "Pass"
                            break 
                        else:
                            # !!! 这里就是案发现场：2D过了，3D被拒 !!!
                            reject_reason = "Depth_Reject" # 深度拒绝
                            # 在屏幕上用红色大字写出 dZ，方便你肉眼看
                            cv2.putText(display_frame, f"REJECT dZ:{depth_diff:.2f}", (wrist_px[0], wrist_px[1]-40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # 状态决策
                raw_state = "Free"
                if contact_obj:
                    if is_open:
                        raw_state = f"Hovering: {contact_obj}"
                    else:
                        raw_state = f"Grasping: {contact_obj}"
                
                # 记录每一帧的关键数据到 CSV
                # 重点关注那些 reject_reason == "Depth_Reject" 的行
                if contact_obj or reject_reason == "Depth_Reject":
                     # 如果没有接触物体，名字就记为 Near_Obj
                     obj_name_log = contact_obj if contact_obj else "Nearby_Obj"
                     # === 【修改】 把原来写死的 0.0 改成记录下来的 obj_z_log ===
                     self._log_data(frame_idx, i, obj_name_log, hand_z, obj_z_log, depth_diff_log, raw_state, reject_reason)

                # 平滑处理
                final_state = self._smooth_state(i % 2, raw_state)

                # 可视化
                if self.visualize:
                    color = (0, 255, 0)
                    if "Hovering" in final_state: color = (0, 255, 255)
                    if "Grasping" in final_state: color = (0, 0, 255)
                    self.mp_drawing.draw_landmarks(display_frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    cv2.putText(display_frame, final_state, (wrist_px[0], wrist_px[1] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # 也在深度图上写状态，方便对比
                    cv2.putText(depth_colormap, final_state, (wrist_px[0], wrist_px[1] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # === 拼合图像：左边 RGB，右边 深度热力图 ===
        # 这样你可以一眼看出深度对不对
        combined_view = np.hstack((display_frame, depth_colormap))
        
        return combined_view

    # === 【修改 3】run_on_video 增加 CSV 初始化和帧号计数 ===
    def run_on_video(self, video_path, output_path="diagnostic_view.mp4"):
        # 初始化 CSV 文件头
        with open("hoi_debug_log.csv", "w") as f:
            f.write("Frame,Hand_ID,Obj_Name,Hand_Z,Obj_Z,Delta_Z,Result,Reason\n")
            
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 注意：因为我们拼合了图像，宽度要 x2
        # 确保服务器端代码使用 mp4v 编码器（Linux 兼容性更好）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))
        
        frame_idx = 0
        print(f"Starting Diagnostic Run on {video_path}...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 传入 frame_idx
            processed_frame = self.process_frame(frame, frame_idx)
            out.write(processed_frame)
            frame_idx += 1
            
        cap.release()
        out.release()
        print(f"诊断完成！请查看生成的文件：\n1. 视频: {output_path} (检查右侧深度图是否清晰)\n2. 日志: hoi_debug_log.csv (查看 Reason='Depth_Reject' 的行)")

# --- 运行入口 ---
if __name__ == "__main__":
    # 实例化系统 (使用 Small 模型更稳)
    hoi_sys = HandObjectInteractionSystem(model_size='yolov8s.pt')
    
    # 运行
    hoi_sys.run_on_video("test_video.mp4")


