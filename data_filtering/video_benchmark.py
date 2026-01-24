import cv2
import yaml
import time
import os
from tqdm import tqdm
from .hoi_system import HandObjectInteractionSystem  # 更新类名

# 配置
VIDEO_PATH = "test_video.mp4"  # 你的视频路径
OUTPUT_YAML = "clean_list.yaml"
FRAME_STRIDE = 15  
# 如果视频是 60fps，意味着每秒只看 4 帧。
# 对于判断“这视频能不能用”来说，每秒 4 帧的信息量绝对够了。

def analyze_video(video_path, hoi_sys=None):
    if not os.path.exists(video_path):
        print(f"❌ 找不到视频: {video_path}")
        return None

    print(f"🚀 启动基准测试 (Benchmarks): {video_path}")
    
    # 增加判断逻辑：
    # 如果外部传进来了模型，就直接用；
    # 如果没传（比如你单独运行脚本测试时），才在内部加载。
    if hoi_sys is None:
        print("⚠️ 未检测到预加载模型，正在初始化新模型...")
        hoi_sys = HandObjectInteractionSystem(model_size='yolov8s.pt')
    else:
        print("✅ 使用预加载的全局模型")
    
    cap = cv2.VideoCapture(video_path)
    if not os.path.exists(video_path):
        print(f"❌ 找不到视频: {video_path}")
        return None

    print(f"🚀 启动基准测试 (Benchmarks): {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 统计数据
    stats = {
        "total_samples": 0,
        "interacting_frames": [],
        "missing_hand_frames": [],
        "interaction_count": 0,
        "missing_hand_count": 0
    }
    
    start_time = time.time()
    
    # 使用 tqdm 显示进度条
    for frame_idx in tqdm(range(0, total_frames, FRAME_STRIDE), desc="Scanning"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: break
        
        # === 核心调用 (接收 3 个返回值) ===
        # visual_img: 可视化图 (benchmark 不需要用，忽略)
        # state: 状态文本 (Interacting / Hovering)
        # info: 详细数据字典
        visual_img, state, info = hoi_sys.process_frame(frame, frame_idx)
        
        stats["total_samples"] += 1
        
        # 1. 统计交互
        if "Interacting" in state:
            stats["interaction_count"] += 1
            stats["interacting_frames"].append(frame_idx)
            
        # 2. 统计无手 (hands_detected == 0)
        # 注意：info['hands_detected'] 包含了被 Rescue 救回来的手
        if info['hands_detected'] == 0:
            stats["missing_hand_count"] += 1
            stats["missing_hand_frames"].append(frame_idx)

    cap.release()
    end_time = time.time()
    
    # === 计算最终指标 ===
    total = max(1, stats["total_samples"])
    interaction_ratio = stats["interaction_count"] / total
    missing_ratio = stats["missing_hand_count"] / total
    
    # 决策逻辑 (阈值可调)
    # 规则：有交互 (>0) 且 脏数据没那么多 (<90%)
    keep = (interaction_ratio > 0.0) and (missing_ratio < 0.9)

    result = {
        "video_path": os.path.abspath(video_path),
        "keep": keep,
        "is_interaction": stats["interaction_count"] > 0,
        "interaction_ratio": round(interaction_ratio, 4),
        "missing_hand_ratio": round(missing_ratio, 4),
        "total_samples": total,
        "processing_time": round(end_time - start_time, 2),
        "interaction_frames_sample": stats["interacting_frames"][:20], # 只存前20个省空间
        "missing_hand_frames_sample": stats["missing_hand_frames"][:20]
    }
    
    return result

if __name__ == "__main__":
    hoi_sys = None  # 全局变量，存放预加载模型
    print("⚡️ 正在加载 YOLO 模型... ")
    result_data = analyze_video(VIDEO_PATH, hoi_sys=hoi_sys)
    
    if result_data:
        # 打印到控制台
        print("\n" + "="*40)
        print(f"RESULT: Keep = {result_data['keep']}")
        print(f"Interaction Ratio: {result_data['interaction_ratio']*100:.2f}%")
        print(f"Missing Hand Ratio: {result_data['missing_hand_ratio']*100:.2f}%")
        print("="*40)
        
        # 保存为 YAML
        with open(OUTPUT_YAML, 'w') as f:
            yaml.dump(result_data, f, sort_keys=False)
        print(f"📄 报告已保存: {OUTPUT_YAML}")