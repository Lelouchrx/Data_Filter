import cv2
import mediapipe as mp
import sys
import time
import numpy as np
import os
# 【新增】导入 YOLO
from ultralytics import YOLO

class HandDataFilter:
    def __init__(self, min_conf=0.5, missing_tolerance=10, check_border=True):
        self.mp_hands = mp.solutions.hands
        # 提高阈值到 0.7，保证数据质量
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_conf,
            min_tracking_confidence=min_conf
        )
        self.tolerance = missing_tolerance # 允许连续丢失的最大帧数
        self.check_border = check_border   # 是否开启边缘检查
        self.margin = 0.05                 # 边缘留白 5%
        self.mp_drawing = mp.solutions.drawing_utils

        # 【新增】加载 YOLOv8 Small 模型
        # 原代码：
        # self.yolo_model = YOLO('yolov8n.pt') 
        
        # 修改后：换成 yolov8s.pt (小) 或 yolov8m.pt (中)
        # 第一次运行会自动下载，大概 20MB - 50MB
        print("正在加载 YOLOv8 Small 模型...")
        self.yolo_model = YOLO('yolov8s.pt') 
        # 设置不想检测的类别 (比如把'人'屏蔽掉，只检测物体)
        # COCO数据集: 0 is person. 我们只需要物体。
        self.ignored_classes = [0]

    # --- 【新增】交互状态分析函数 ---
    def analyze_interaction_state(self, hand_landmarks):
        """
        判断手部状态: Open (张开), Fist (空拳), Grasping (抓取)
        """
        # 1. 获取关键点坐标 (x, y)
        # 拇指(4), 食指(8), 中指(12), 无名指(16), 小指(20), 手腕(0)
        points = {}
        for idx in [0, 4, 8, 12, 16, 20]:
            points[idx] = np.array([hand_landmarks.landmark[idx].x, hand_landmarks.landmark[idx].y])
        
        # 2. 计算 拇指-食指 距离 (Pinch Check)
        pinch_dist = np.linalg.norm(points[4] - points[8])
        is_pinching = pinch_dist < 0.08  # 阈值可微调
        
        if not is_pinching:
            return "Open", (0, 255, 0) # 绿色
        
        # 3. 计算 掌心拥挤度 (指尖到手腕的距离)
        # 用 中指、无名指、小指 到 手腕的平均距离
        finger_to_wrist_dists = [
            np.linalg.norm(points[i] - points[0]) for i in [12, 16, 20]
        ]
        avg_curl_dist = np.mean(finger_to_wrist_dists)
        
        # 4. 区分 空拳 vs 抓取
        # 经验阈值：< 0.25 说明手指缩得很紧（空拳），> 0.25 说明有体积撑着（抓取）
        if avg_curl_dist < 0.25: 
            return "Fist", (0, 255, 255)    # 黄色 (空握)
        else:
            return "Grasping", (0, 0, 255)  # 红色 (抓取中)

    # --- 【新增】获取手的边界框 (Bounding Box) ---
    def get_hand_bbox(self, landmarks, frame_w, frame_h):
        """将归一化的关键点转换为像素坐标的边界框 [x1, y1, x2, y2]"""
        x_list = [lm.x for lm in landmarks.landmark]
        y_list = [lm.y for lm in landmarks.landmark]
        
        x1, x2 = min(x_list), max(x_list)
        y1, y2 = min(y_list), max(y_list)
        
        # 稍微给手部框加一点 padding (扩充 10%)，让重叠检测更灵敏
        padding = 0.05
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(1, x2 + padding)
        y2 = min(1, y2 + padding)

        return [int(x1 * frame_w), int(y1 * frame_h), int(x2 * frame_w), int(y2 * frame_h)]

    # --- 【新增】计算 IoU 重叠并判断交互 ---
    def check_interaction_with_yolo(self, frame, hand_bbox):
        """
        运行 YOLO，看手部框是否与任何物体框重叠
        返回: (是否交互, 物体名称, 物体框)
        """
        # 建议在这里也转一下，确保 YOLO 拿到的是标准格式
        # 如果传入的是 RGB (MediaPipe的frame)，YOLO 能处理
        # 如果传入的是 BGR (OpenCV的frame)，YOLO 也能处理
        # 关键是：不要传 flag.writeable=False 的只读内存进去
        
        # 运行推理
        # 原代码：
        # results = self.yolo_model(frame, verbose=False, conf=0.3)
        
        # 修改后：降低到 0.15，只要有一点像杯子就认
        # 现在的 conf=0.3 可能太高了。在遮挡情况下，YOLO 对这个杯子的置信度可能只有 0.15 左右。
        results = self.yolo_model(frame, verbose=False, conf=0.15)
        
        detected_objects = results[0].boxes
        
        hx1, hy1, hx2, hy2 = hand_bbox
        hand_area = (hx2 - hx1) * (hy2 - hy1)

        best_iou = 0
        interaction_obj = None
        obj_box_coords = None

        for box in detected_objects:
            cls_id = int(box.cls[0])
            
            # 过滤掉 "人" (class 0)，我们不把自己的身体当物体
            if cls_id in self.ignored_classes:
                continue

            # 获取物体坐标
            ox1, oy1, ox2, oy2 = map(int, box.xyxy[0])
            obj_name = self.yolo_model.names[cls_id]

            # === 计算重叠面积 (Intersection) ===
            ix1 = max(hx1, ox1)
            iy1 = max(hy1, oy1)
            ix2 = min(hx2, ox2)
            iy2 = min(hy2, oy2)

            inter_width = max(0, ix2 - ix1)
            inter_height = max(0, iy2 - iy1)
            inter_area = inter_width * inter_height

            # 如果重叠面积 > 0，说明碰到了
            if inter_area > 0:
                # 这是一个简单的 IoU 变体，我们看重叠部分占手部面积的比例
                # 或者只要有重叠就算交互
                interaction_obj = obj_name
                obj_box_coords = (ox1, oy1, ox2, oy2)
                return True, obj_name, obj_box_coords

        return False, None, None

    def is_hand_valid(self, hand_landmarks):
        """
        检查单只手是否有效：
        1. 存在
        2. 手腕 (Wrist, index 0) 在安全区域内
        """
        if not self.check_border:
            return True
        
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        
        # 检查 x, y 是否在 [0.05, 0.95] 区间
        if (self.margin < wrist.x < 1 - self.margin) and \
           (self.margin < wrist.y < 1 - self.margin):
            return True
        return False

    def process_video(self, input_path, output_path=None, visualize=False):
        print(f"🔄 Processing: {input_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return {"status": "Error", "reason": "Cannot open video"}

        # 视频参数
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 仅当需要可视化时才初始化 Writer
        out = None
        if visualize and output_path:
            # Mac 用户强烈建议使用 'avc1' 而不是 'mp4v'
            # 'mp4v' 在 Mac 上经常导致绿屏或马赛克
            fourcc = cv2.VideoWriter_fourcc(*'avc1') 
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            mp_drawing = mp.solutions.drawing_utils

        # --- 状态追踪器 ---
        frame_idx = 0
        consecutive_missing = 0  # 当前连续丢失帧数
        max_missing_streak = 0   # 记录整个视频中最严重的连续丢失
        pass_frames = 0
        
        is_rejected = False
        reject_reason = ""

        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # 性能优化：只在可视化开启时才做深拷贝，否则直接用只读引用
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False 
            
            results = self.hands.process(image_rgb)
            
            # --- 核心判定逻辑 ---
            valid_hands_count = 0
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    if self.is_hand_valid(hand_lms):
                        valid_hands_count += 1
            
            # 判定标准：假设要求必须双手都在
            # 如果你的任务允许单手，把这里改成 valid_hands_count >= 1
            is_frame_good = (valid_hands_count == 2)

            # --- 修改后的代码 (Warm-up Masking) ---
            if is_frame_good:
                consecutive_missing = 0
                pass_frames += 1
            else:
                # 只有当视频播放超过 60 帧 (约 2 秒) 后，才开始计较错误
                # 这样可以过滤掉模型起步时的"犹豫"阶段
                if frame_idx > 60:
                    consecutive_missing += 1
                    max_missing_streak = max(max_missing_streak, consecutive_missing)

            # ================= 关键修改开始 =================
            # 1. MediaPipe 用的是 RGB，OpenCV 画图和保存视频需要 BGR
            # 2. .copy() 非常重要！它能解决内存不连续导致的"马赛克/雪花"问题
            frame_bgr = None
            if visualize and out is not None:
                image_rgb.flags.writeable = True # 确保可写
                frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
            # ================= 关键修改结束 =================

            # --- 可视化 (可选) ---
            if visualize and out is not None:
                # 注意：以后所有的画图操作 (cv2.rectangle, putText) 
                # 都要画在 【frame_bgr】 上，而不是原来的 frame 上！
                
                # 绘制手部
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # 画骨架 (需要把原来的 frame 换成 frame_bgr)
                        self.mp_drawing.draw_landmarks(
                            frame_bgr, # <--- 改这里
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )
                        
                        # === 【新增】YOLO 交互检测逻辑 ===
                        # 1. 获取手的框
                        h_box = self.get_hand_bbox(hand_landmarks, width, height)
                        
                        # 2. 调用 YOLO 检查是否拿着东西
                        # 传给 YOLO 的可以用原来的 frame (RGB) 或者 frame_bgr 都可以，YOLO很聪明
                        # 但画图一定要画在 frame_bgr 上
                        is_grasping_something, grasp_obj_name, o_box = self.check_interaction_with_yolo(frame_bgr, h_box)
                        
                        wrist_x = int(hand_landmarks.landmark[0].x * width)
                        wrist_y = int(hand_landmarks.landmark[0].y * height)
                        
                        # === 【新增】几何姿态计算 (用来过滤"悬停"误判) ===
                        # 计算所有指尖到手腕的平均距离
                        # 手腕(0), 食指(8), 中指(12), 无名指(16), 小指(20)
                        wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
                        finger_tips = [
                            np.array([hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y])
                            for i in [8, 12, 16, 20]
                        ]
                        # 计算平均张开距离
                        tips_to_wrist_dists = [np.linalg.norm(tip - wrist) for tip in finger_tips]
                        avg_spread = np.mean(tips_to_wrist_dists)
                        
                        # 设定一个"张开阈值"
                        # > 0.40 说明手掌完全张开（在悬停）
                        IS_HAND_OPEN = avg_spread > 0.40 
                        # ===============================================

                        # === 最终判定逻辑 (YOLO + Geometry) ===
                        final_state = "Free Hand"
                        color = (0, 255, 0) # Green

                        if is_grasping_something:
                            if IS_HAND_OPEN:
                                # 1. 框重叠了，但是手是张开的 -> 悬停 (Hovering)
                                final_state = f"Hovering: {grasp_obj_name}"
                                color = (255, 255, 0) # Cyan/Yellow (黄色警告)
                            else:
                                # 2. 框重叠了，且手是弯曲的 -> 真抓取 (Grasping)
                                final_state = f"Grasping: {grasp_obj_name}"
                                color = (0, 0, 255) # Red
                            
                            # 画出物体的框 (可选，为了 Debug 看得更清楚)
                            if o_box:
                                cv2.rectangle(frame_bgr, (o_box[0], o_box[1]), (o_box[2], o_box[3]), color, 2)
                                cv2.putText(frame_bgr, grasp_obj_name, (o_box[0], o_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                        # === 画图 ===
                        # 画手部框
                        cv2.rectangle(frame_bgr, (h_box[0], h_box[1]), (h_box[2], h_box[3]), color, 2)
                        
                        # 写文字
                        cv2.putText(frame_bgr, final_state, (wrist_x, wrist_y - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        # ====================================
                
                # 绘制状态信息
                color = (0, 255, 0) if is_frame_good else (0, 0, 255)
                cv2.putText(frame_bgr, f"Hands: {valid_hands_count} | MissStreak: {consecutive_missing}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # 绘制安全框 (Safe Zone)
                h, w, _ = frame_bgr.shape
                p1 = (int(w * self.margin), int(h * self.margin))
                p2 = (int(w * (1-self.margin)), int(h * (1-self.margin)))
                cv2.rectangle(frame_bgr, p1, p2, (255, 255, 0), 1)
                
                # ================= 写入视频 =================
                # 确保写入的是 frame_bgr
                out.write(frame_bgr)

            # --- Early Stopping (可选：如果只想过滤掉坏数据，发现太差直接退出) ---
            # if consecutive_missing > self.tolerance * 5: 
            #     is_rejected = True
            #     reject_reason = "Too many missing frames"
            #     break 

        # --- 在 cap.release() 之前加入这个判断 ---
        if consecutive_missing > self.tolerance:
            loss_start_time = (frame_idx - consecutive_missing) / fps
            loss_end_time = frame_idx / fps
            print(f"⚠️ 发现结尾丢帧: {loss_start_time:.2f}秒 -> {loss_end_time:.2f}秒 (持续 {consecutive_missing} 帧)")

        cap.release()
        if out: out.release()
        
        duration = time.time() - start_time
        fps_process = frame_idx / duration if duration > 0 else 0

        # --- 最终判定 ---
        # 规则：最大连续丢失不能超过 Tolerance (例如 10帧)
        if max_missing_streak > self.tolerance:
            final_status = "REJECT"
            reject_reason = f"Continuous missing frames exceeded limit ({max_missing_streak} > {self.tolerance})"
        else:
            final_status = "PASS"
            reject_reason = "None"

        return {
            "video": input_path,
            "status": final_status,
            "reason": reject_reason,
            "max_missing_streak": max_missing_streak,
            "pass_ratio": pass_frames / frame_idx if frame_idx > 0 else 0,
            "process_fps": f"{fps_process:.1f}",
            "duration_seconds": f"{duration:.2f}"
        }

# --- 使用示例 ---
if __name__ == "__main__":
    # --- 原代码 ---
    # filter_tool = HandDataFilter(min_conf=0.7, missing_tolerance=10, check_border=True)
    
    # --- 修改后：将 min_conf 改为 0.5 ---
    # 0.5 是 MediaPipe 官方推荐的默认值，足以过滤掉明显的背景误检，但不会误杀真手
    filter_tool = HandDataFilter(min_conf=0.5, missing_tolerance=10, check_border=True)
    
    # 模式 A: 快速过滤 (不生成视频，速度快)
    result = filter_tool.process_video("test_video.mp4", visualize=False)
    print("📊 快速扫描结果:", result)
    
    # 模式 B: Debug 模式 (生成视频，查看哪里出了问题)
    if result["status"] == "REJECT":
        print("🔍 正在生成 Debug 视频...")
        filter_tool.process_video("test_video.mp4", "debug_output.mp4", visualize=True)

