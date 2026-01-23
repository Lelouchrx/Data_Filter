# Data_Filter: Hand-Object Interaction (HOI) Filtering System

A high-efficiency, multi-modal computer vision pipeline designed to filter first-person video data. It identifies valid **Hand-Object Interaction** segments by combining 2D object detection, hand tracking, and 3D depth consistency checks.

## 🚀 Key Features

* **⚡ YOLO-First "Lazy" Execution**:
    * Uses **YOLOv8** as a high-speed gatekeeper.
    * Automatically skips expensive computation (Hand/Depth) for empty frames, achieving **10x speedup** on sparse videos.
* **🖐️ Ghost Hand Rescue (Depth-Based)**:
    * Recovers missed hand detections (e.g., due to motion blur or occlusion) using **Depth Anything V2** and adaptive depth slicing.
* **🔋 Temporal Stability Buffer**:
    * Implements an energy-buffer mechanism to prevent flickering and maintain state continuity during brief detection losses.
* **📏 Physics-Grounded Interaction**:
    * Validates interactions not just by 2D overlap (IoU), but by **3D depth consistency** (Z-axis alignment) between the hand and the object.
* **🏎️ Resolution Optimized**:
    * Smart downsampling (518px for depth, 640px for hands) ensures high inference speed without compromising decision accuracy.

## 🛠️ Tech Stack

* **Object Detection**: [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)
* **Hand Tracking**: [MediaPipe Hands](https://developers.google.com/mediapipe)
* **Depth Estimation**: [Depth Anything V2 (Small)](https://huggingface.co/depth-anything/Depth-Anything-V2-Small)
* **Core**: PyTorch, OpenCV, NumPy

## 📦 Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Lelouchrx/Data_Filter.git
    cd Data_Filter
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Models will be automatically downloaded on the first run.)*

## 🏃 Usage

### 1. Data Filtering (Benchmark)
Use this script to process videos and generate a YAML report. It decides whether to keep a video based on interaction ratio and data quality.

```bash
python video_benchmark.py
```
**Input**: Edit `VIDEO_PATH` in the script.  
**Output**: Generates `clean_list.yaml` containing metrics (Keep/Drop, Interaction Ratio, etc.).  
**Config**: Adjust `FRAME_STRIDE` (default: 15) to balance speed and sampling density.

### 2. Visualization (Debug Mode)
Generate a diagnostic video with split-screen view (RGB + Depth Heatmap) to verify the algorithm's logic.

```bash
python make_demo_video.py
```
**Output**: Saves a video file (e.g., `debug_visual_pro.mp4`) showing the "Rescue" boxes, interaction lines, and buffer status.

## ⚙️ Configuration

Key parameters in `hoi_system.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OBJ_PADDING_PX` | 20 | Expands object bbox to catch edge interactions (e.g., grasping a cup handle). |
| `DEPTH_THRESHOLD` | 0.15 | Max allowed Z-distance (normalized) between hand and object to count as interaction. |
| `BUFFER_LIMIT` | 2 | Frames to hold state after detection loss. (2 = ~0.5s stability at 60fps/stride15). |

## 📂 Project Structure

```
.
├── hoi_system.py         # Core Engine: YOLO + MediaPipe + Depth logic
├── video_benchmark.py    # Main script for batch processing & metrics
├── make_demo_video.py    # Visualization tool for debugging
├── requirements.txt      # Dependency list
└── README.md             # Documentation
```

## 📝 License

This project is intended for research and data cleaning purposes.
