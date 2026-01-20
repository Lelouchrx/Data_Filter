from pathlib import Path


OUTPUT_DIR = 'frames_output'
IMG_EXTENSIONS = ['.jpg', '.png', '.jpeg']
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']

THRESHOLD = 20.0 
FIX_SIZE = True
TARGET_SIZE = (500, 500)

VIDEO_BLUR_RATIO = 0.3
SAMPLE_RATE = 10.0
MOTION_METHOD = 'lk'
MOTION_SKIP_FRAMES = 5

JITTER_THRESHOLD = 3.0
VALID_RATIO_THRESHOLD = 0.7

VERBOSE = False
DISPLAY = False

