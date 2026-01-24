import cv2
import os
import time
from tqdm import tqdm
from hoi_system import HandObjectInteractionSystem

# =================配置区域=================
INPUT_VIDEO = "test_video.mp4"       # 输入视频
OUTPUT_VIDEO = "debug_visual_pro.mp4" # 输出的诊断视频
MODEL_SIZE = "yolov8s.pt"            # 模型大小
# =========================================

def generate_demo():
    if not os.path.exists(INPUT_VIDEO):
        print(f"❌ 错误: 找不到输入视频 {INPUT_VIDEO}")
        return

    print(f"🚀 初始化 HOI 引擎 (用于生成可视化)...")
    # 初始化引擎
    hoi_sys = HandObjectInteractionSystem(model_size=MODEL_SIZE)
    
    # 打开视频读取
    cap = cv2.VideoCapture(INPUT_VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 设置视频写入器
    # 注意：因为 HOI 引擎返回的是 "左RGB + 右深度" 的拼接图，所以宽度要 x2
    out_width = width * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (out_width, height))

    print(f"🎬 开始处理视频: {INPUT_VIDEO}")
    print(f"💾 输出路径: {OUTPUT_VIDEO}")
    print(f"📊 总帧数: {total_frames}")

    start_time = time.time()

    # 使用 tqdm 显示进度条
    for frame_idx in tqdm(range(total_frames), desc="Rendering"):
        ret, frame = cap.read()
        if not ret:
            break

        # === 核心调用 ===
        # 我们这里只需要第一个返回值 (visual_img)
        # state 和 info 在生成视频时可以忽略，因为它们已经画在图上了
        visual_img, state, info = hoi_sys.process_frame(frame, frame_idx)

        # 写入视频帧
        out.write(visual_img)

    # 资源释放
    cap.release()
    out.release()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*40)
    print(f"✅ 可视化视频生成完毕!")
    print(f"耗时: {duration:.2f}秒")
    print(f"FPS: {total_frames/duration:.2f}")
    print(f"文件已保存至: {os.path.abspath(OUTPUT_VIDEO)}")
    print("="*40)

if __name__ == "__main__":
    generate_demo()