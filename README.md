# Automated Video Quality & Content Filtering System

This project is a client-server based pipeline designed to automatically filter video datasets for robotics learning or computer vision tasks. It utilizes a two-stage screening process to filter out low-quality videos (blur, shake, bad exposure) and videos lacking specific content (hand-object interaction).

## ✨ Key Features

* **Two-Stage Analysis Pipeline**:
1. **Quality Filter**: fast pre-screening for blur, camera jitter/shake, and exposure issues.
2. **Content Filter**: AI-based detection (YOLO + Depth Anything) to verify valid hand-object interactions.


* **Automatic Sorting**: processed videos are automatically moved to `server_data/accepted_videos` or `server_data/rejected_videos`. **No data is deleted.**
* **Comprehensive Logging**: detailed JSON analysis reports are generated for every single video.
* **Smart Batch Processing**: the client supports recursive directory scanning, automatic error handling, and batch summary statistics.
* **High Performance**: server-side models are pre-loaded into GPU memory for fast inference.

## 📂 Project Structure

```text
project_root/
├── blur/                   # [Core] Quality detection algorithms (blur/jitter/exposure)
├── data_filtering/         # [Core] Content detection algorithms (YOLO/HOI System)
├── server_data/            # [Auto-Generated] Server storage
│   ├── temp_uploads/       # Temporary staging area
│   ├── accepted_videos/    # ✅ High-quality videos with interaction
│   ├── rejected_videos/    # 🚫 Low-quality or non-interactive videos
│   └── processing_logs/    # 📝 JSON result logs
├── server.py               # FastAPI Server script
├── client.py               # Client script (English CLI)
└── requirements.txt        # Dependencies

```

## 🛠️ Prerequisites & Installation

Recommended: Python 3.10+ and a CUDA-capable GPU.

1. **Install Dependencies**:
```bash
pip install fastapi uvicorn python-multipart requests opencv-python numpy ultralytics torch

```


*(Note: Ensure you have the necessary libraries for the `blur` and `data_filtering` modules as well, such as `tqdm`, `matplotlib`, `pyyaml`, etc.)*

## 🚀 Usage Guide

### 1. Start the Server

The server handles model loading, inference, and file management.

```bash
python server.py

```

* **Startup**: Wait until you see `✅ [Server] model loaded successfully` (Model loaded) and `Uvicorn running on http://0.0.0.0:8000`.
* **First Run**: The system may download YOLO/Depth weights automatically on the first launch.

### 2. Run the Client

The client uploads videos and displays real-time analysis results.

**Option A: Process a Single File**

```bash
python client.py /path/to/video.mp4
python client.py "client files"
```

**Option B: Batch Process a Folder (Recommended)**
Recursively scans the directory for video files (`.mp4`, `.mov`, `.avi`, etc.).

```bash
python client.py /path/to/dataset_folder/

```

## 📊 Outputs & Logs

### Client Console

After processing, the client displays a summary dashboard:

```text
========================================
📊 Batch Processing Completed (45.2s)
========================================
Total Videos:       10
✅ Accepted:         3
🚫 Quality Fail:     5
⚠️ Content Fail:     2
❌ Errors:           0
========================================

```

### Server Storage (`server_data/`)

* **`accepted_videos/`**: Contains the "clean" dataset ready for training.
* **`rejected_videos/`**: Contains videos that failed checks (kept for auditing).
* **`processing_logs/`**: JSON files containing detailed metrics. Example:
```json
{
    "filename": "demo.mp4",
    "pipeline_status": "ACCEPTED",
    "quality_data": {
        "blur_score": 214.5,
        "is_shake": false,
        "jitter_disp_pct": 2.1
    },
    "content_data": {
        "interaction_ratio": 0.95,
        "missing_hand_ratio": 0.0
    }
}

```



## ⚙️ Configuration

* **Server Port**: Modify the `uvicorn.run(..., port=8000)` line in `server.py`.
* **Thresholds**: Adjust quality or interaction thresholds inside the `analyze_video_endpoint` function in `server.py` (e.g., `threshold=100.0`, `video_blur_ratio=0.3`).

## ⚠️ Notes

1. **GPU Memory**: The server keeps models loaded in VRAM. Ensure you have at least 4GB VRAM available (depending on model size).
2. **Concurrency**: The current implementation handles requests sequentially to ensure GPU stability.
3. **Temporary Files**: The `temp_uploads` folder is self-cleaning. Files are moved out immediately after processing.