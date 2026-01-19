from pathlib import Path


OUTPUT_DIR = 'frames_output'  # 输出目录
IMG_EXTENSIONS = ['.jpg', '.png', '.jpeg']  # 图片扩展名
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']  # 视频扩展名

# 模糊检测配置
THRESHOLD = 20.0  # 模糊阈值
FIX_SIZE = True  # 是否固定图片尺寸
TARGET_SIZE = (500, 500)  # 固定尺寸的目标大小

# 视频处理配置
VIDEO_BLUR_RATIO = 0.3  # 视频模糊帧比例阈值
SAMPLE_RATE = 10.0  # 采样率（每秒帧数）
MOTION_METHOD = 'lk'  # 运动估计方法: 'farneback' 或 'lk'
MOTION_SKIP_FRAMES = 5  # 运动估计跳帧数（越高越快）

# 视频质量判断配置
JITTER_THRESHOLD = 6.0  # 抖动阈值
VALID_RATIO_THRESHOLD = 0.7  # 有效率阈值

# 日志配置
VERBOSE = False  # 是否显示详细日志
DISPLAY = False  # 是否显示图片

