import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import torch
from transformers import pipeline
from PIL import Image

class DepthEstimator:
    def __init__(self, device='cuda'):
        print(f"[Init] Loading Depth Anything V2 (Small) on {device}...")
        self.pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
        self.infer_size = 518

    def infer(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = 1.0
        process_w, process_h = w, h
        
        if min(w, h) > self.infer_size:
            scale = self.infer_size / min(w, h)
            process_w = int(w * scale)
            process_h = int(h * scale)
            frame_input = cv2.resize(frame_bgr, (process_w, process_h))
        else:
            frame_input = frame_bgr

        image = Image.fromarray(cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB))
        depth_output = self.pipe(image)
        depth_map_pil = depth_output["depth"]
        depth_np = np.array(depth_map_pil)
        
        if scale != 1.0:
            depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)

        d_min = depth_np.min()
        d_max = depth_np.max()
        norm_depth = (depth_np - d_min) / (d_max - d_min + 1e-6)
        
        return norm_depth

class HandObjectInteractionSystem:
    def __init__(self, model_size='yolov8s.pt', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"[Init] Device set to use {self.device}")
        
        # YOLO (最快，用来做前哨)
        print(f"[Init] Loading YOLO model: {model_size}...")
        self.yolo_model = YOLO(model_size)
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.OBJ_PADDING_PX = 20
        self.DEPTH_THRESHOLD = 0.15
        # 原来是 5 (配合 Stride=15 会导致太长)
        # 建议改为 2 或 3
        self.BUFFER_LIMIT = 2
        self.interaction_buffer = 0
        
        # 深度模型 (最慢，最后调用)
        self.depth_model = DepthEstimator(device=self.device)
        
    def _get_region_depth(self, depth_map, box):
        x1, y1, x2, y2 = map(int, box)
        h, w = depth_map.shape
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1: return 0.0
        region = depth_map[y1:y2, x1:x2]
        return np.median(region) if region.size > 0 else 0.0

    def _rescue_missing_hand(self, depth_map, known_hand_box, known_hand_depth):
        h, w = depth_map.shape
        depth_threshold = 0.05 
        center_val = float(known_hand_depth)
        lower = max(0.0, center_val - depth_threshold)
        upper = min(1.0, center_val + depth_threshold)
        mask = cv2.inRange(depth_map, lower, upper)
        
        kx1, ky1, kx2, ky2 = known_hand_box
        padding = 30
        cv2.rectangle(mask, (max(0, kx1-padding), max(0, ky1-padding)), 
                            (min(w, kx2+padding), min(h, ky2+padding)), 0, -1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found_rescue = False
        rescue_box = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000: 
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                aspect_ratio = float(w_box)/h_box
                if 0.3 < aspect_ratio < 3.0:
                    found_rescue = True
                    rescue_box = [x, y, x+w_box, y+h_box]
                    break
        return found_rescue, rescue_box

    def process_frame(self, frame, frame_idx=0):
        h, w, _ = frame.shape
        display_frame = frame.copy() 
        
        # ===========================
        # 1. 极速前哨：YOLO 检测
        # ===========================
        # YOLOv8 很快，先跑它
        yolo_results = self.yolo_model(frame, verbose=False, conf=0.15)
        objects = []
        for r in yolo_results:
            boxes = r.boxes
            for box in boxes:
                if int(box.cls[0]) == 0: continue # 跳过人
                b = box.xyxy[0].cpu().numpy().astype(int)
                label = self.yolo_model.names[int(box.cls[0])]
                objects.append({'bbox': b, 'label': label})
                cv2.rectangle(display_frame, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
                cv2.putText(display_frame, label, (b[0], b[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 🚀【短路逻辑】如果没东西，HOI 肯定不存在，直接跳过深度和手部计算！
        # 但要注意：为了保持 buffer 逻辑，我们不能直接 return，要走一遍空的逻辑
        skip_heavy_compute = (len(objects) == 0)

        norm_depth_map = None
        depth_colormap = np.zeros_like(frame) # 占位符
        hands_list = []
        is_rescued = False

        if not skip_heavy_compute:
            # ===========================
            # 2. 如果有物体，再算手 (MediaPipe)
            # ===========================
            # 缩放加速 MediaPipe
            mp_scale = 1.0
            mp_input_w = 640
            if w > mp_input_w:
                mp_scale = mp_input_w / w
                mp_h = int(h * mp_scale)
                frame_small = cv2.resize(frame, (mp_input_w, mp_h))
                frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_results = self.hands.process(frame_rgb)
            
            # 🚀【短路逻辑 2】如果有物体但没手，而且 buffer 也空了，是不是也可以跳过深度？
            # 不行，因为要考虑 "Ghost Hand Rescue" (有物体，MediaPipe漏检，但深度能救)
            # 所以只要有物体，我们就最好算一下深度
            
            # ===========================
            # 3. 如果有物体，最后算深度 (最重)
            # ===========================
            norm_depth_map = self.depth_model.infer(frame)
            depth_uint8 = (norm_depth_map * 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)

            if hand_results.multi_hand_landmarks:
                for hand_lms in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(display_frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    x_min, y_min = w, h
                    x_max, y_max = 0, 0
                    for lm in hand_lms.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x_max, x)
                        y_max = max(y_max, y)
                    
                    box = [x_min, y_min, x_max, y_max]
                    d = self._get_region_depth(norm_depth_map, box)
                    hands_list.append({'box': box, 'depth': d, 'type': 'MediaPipe'})

            # 救援逻辑
            if len(hands_list) == 1:
                rescued, rescue_box = self._rescue_missing_hand(norm_depth_map, hands_list[0]['box'], hands_list[0]['depth'])
                if rescued:
                    rescue_depth = self._get_region_depth(norm_depth_map, rescue_box)
                    hands_list.append({'box': rescue_box, 'depth': rescue_depth, 'type': 'Rescued'})
                    is_rescued = True
                    cv2.rectangle(display_frame, (rescue_box[0], rescue_box[1]), (rescue_box[2], rescue_box[3]), (0, 255, 255), 2)
                    cv2.putText(display_frame, "RESCUED", (rescue_box[0], rescue_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ===========================
        # 4. 决策与缓冲 (所有情况都要跑)
        # ===========================
        current_frame_interaction = False
        interact_obj_label = None
        
        if not skip_heavy_compute and len(hands_list) > 0:
            for hand in hands_list:
                pad = self.OBJ_PADDING_PX
                hx1, hy1, hx2, hy2 = hand['box']
                hand_box_padded = [hx1-pad, hy1-pad, hx2+pad, hy2+pad]
                
                for obj in objects:
                    ox1, oy1, ox2, oy2 = obj['bbox']
                    ix1 = max(hand_box_padded[0], ox1)
                    iy1 = max(hand_box_padded[1], oy1)
                    ix2 = min(hand_box_padded[2], ox2)
                    iy2 = min(hand_box_padded[3], oy2)
                    
                    if ix2 > ix1 and iy2 > iy1:
                        obj_depth = self._get_region_depth(norm_depth_map, obj['bbox'])
                        diff = abs(hand['depth'] - obj_depth)
                        if diff < self.DEPTH_THRESHOLD:
                            current_frame_interaction = True
                            interact_obj_label = obj['label']
                            # 画线
                            cx_hand, cy_hand = int((hx1+hx2)/2), int((hy1+hy2)/2)
                            cx_obj, cy_obj = int((ox1+ox2)/2), int((oy1+oy2)/2)
                            cv2.line(display_frame, (cx_hand, cy_hand), (cx_obj, cy_obj), (0, 255, 0), 3)
                            break
                if current_frame_interaction: break

        final_state = "Hovering"
        if current_frame_interaction:
            self.interaction_buffer = self.BUFFER_LIMIT
            final_state = f"Interacting: {interact_obj_label}"
        else:
            if self.interaction_buffer > 0:
                self.interaction_buffer -= 1
                final_state = "Interacting (Buffered)"

        # 可视化文本
        color = (0, 255, 0) if "Interacting" in final_state else (0, 255, 255)
        cv2.putText(display_frame, f"State: {final_state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if is_rescued:
             cv2.putText(display_frame, "[Depth Rescue Active]", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if skip_heavy_compute:
             cv2.putText(display_frame, "[Lazy Skip: No Objects]", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        # 缓冲条
        bar_w = 200
        fill = int(bar_w * (self.interaction_buffer / self.BUFFER_LIMIT))
        cv2.rectangle(display_frame, (w-220, 20), (w-20, 40), (100,100,100), 2)
        cv2.rectangle(display_frame, (w-220, 20), (w-220+fill, 40), (0, 255, 0), -1)
        cv2.putText(display_frame, "Stability", (w-220, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        combined_view = np.hstack((display_frame, depth_colormap))
        
        return combined_view, final_state, {
            "hands_detected": len(hands_list),
            "objects_detected": len(objects),
            "interaction_buffer": self.interaction_buffer,
            "is_rescued": is_rescued
        }

    def run_on_video(self, video_path, output_path="final_diagnostic.mp4"):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w*2, h))
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            res, _, _ = self.process_frame(frame, frame_idx)
            out.write(res)
            frame_idx += 1
        cap.release()
        out.release()

if __name__ == "__main__":
    pass