import requests
import os
import sys
import time
from datetime import datetime

# === Configuration ===
# SERVER_URL = "http://localhost:8000/analyze"# local server URL
SERVER_URL = "http://123.45.67.89:8000/analyze" # remote server URL
VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.mkv', '.webm') 

def upload_video(file_path, current_idx=1, total_count=1):
    """Upload a single video, print result, and return status."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return "ERROR"

    filename = os.path.basename(file_path)
    print(f"\n[{current_idx}/{total_count}] 📤 Processing: {filename} ...")
    start_time = time.time()

    try:
        with open(file_path, 'rb') as f:
            files = {'file': (filename, f, 'video/mp4')}
            # Send POST request
            resp = requests.post(SERVER_URL, files=files)

        if resp.status_code == 200:
            data = resp.json()
            print_report(data)
            return data.get('pipeline_status', 'UNKNOWN')
        else:
            print(f"❌ Server Error ({resp.status_code}): {resp.text}")
            return "SERVER_ERROR"

    except requests.exceptions.ConnectionError:
        print("❌ Connection Refused: Is the server running?")
        return "CONNECTION_ERROR"
    except Exception as e:
        print(f"❌ Unknown Error: {e}")
        return "CLIENT_ERROR"
    finally:
        elapsed = time.time() - start_time
        print(f"⏱️ Time taken: {elapsed:.2f}s")

def print_report(data):
    """Print a detailed report in English."""
    status = data.get('pipeline_status', 'UNKNOWN')
    
    # Status Icons
    icons = {
        "ACCEPTED": "✅",
        "REJECTED_QUALITY": "🚫",
        "REJECTED_CONTENT": "⚠️",
        "ERROR": "❌"
    }
    icon = icons.get(status, "❓")
    
    print("-" * 50)
    print(f"🏁 Final Status: {icon} {status}")
    
    # 1. Quality Report
    q = data.get('quality_data')
    if q:
        pass_str = "PASS" if q.get('passed') else "FAIL"
        print(f"   [Quality Check]: {pass_str}")
        print(f"     - Blur Score:       {q.get('blur_score', 0):.1f}")
        print(f"     - Blur Ratio:       {q.get('blur_ratio', 0)*100:.1f}%")
        
        is_shake = "YES" if q.get('is_shake') else "NO"
        disp_pct = q.get('jitter_disp_pct', 0)
        print(f"     - Jitter/Shake:     {is_shake} (Disp: {disp_pct:.1f}%)")
    
    # 2. Content Report
    c = data.get('content_data')
    if c:
        pass_str = "PASS" if c.get('passed') else "FAIL"
        print(f"   [Content Check]: {pass_str}")
        print(f"     - Interaction:      {c.get('interaction_ratio', 0)*100:.1f}%")
        print(f"     - Missing Hand:     {c.get('missing_hand_ratio', 0)*100:.1f}%")
    elif status == "REJECTED_QUALITY":
        print("   [Content Check]: SKIPPED (Low Quality)")
    
    # Reject Reason
    if data.get('reject_reason'):
        print(f"   [Reason]: {data['reject_reason']}")
    print("-" * 50)

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file: python client.py path/to/video.mov")
        print("  Folder:      python client.py path/to/video_folder/")
        return

    input_path = sys.argv[1]
    
    # Collect tasks
    tasks = []
    if os.path.isfile(input_path):
        tasks.append(input_path)
    elif os.path.isdir(input_path):
        print(f"📂 Scanning directory: {input_path} ...")
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(VIDEO_EXTS):
                    tasks.append(os.path.join(root, file))
        tasks.sort()
    else:
        print(f"❌ Invalid path: {input_path}")
        return

    total = len(tasks)
    if total == 0:
        print("⚠️ No valid video files found.")
        return

    print(f"🚀 Ready to process {total} video(s)...\n")
    
    # Statistics
    stats = {
        "ACCEPTED": 0,
        "REJECTED_QUALITY": 0,
        "REJECTED_CONTENT": 0,
        "ERRORS": 0
    }

    start_all = time.time()

    # === Main Loop ===
    for i, file_path in enumerate(tasks):
        status = upload_video(file_path, i + 1, total)
        
        # Update stats
        if status == "ACCEPTED":
            stats["ACCEPTED"] += 1
        elif status == "REJECTED_QUALITY":
            stats["REJECTED_QUALITY"] += 1
        elif status == "REJECTED_CONTENT":
            stats["REJECTED_CONTENT"] += 1
        else:
            stats["ERRORS"] += 1

    # === Final Summary ===
    duration = time.time() - start_all
    print("\n" + "="*40)
    print(f"📊 Batch Processing Completed ({duration:.1f}s)")
    print("="*40)
    print(f"Total Videos:       {total}")
    print(f"✅ Accepted:         {stats['ACCEPTED']}")
    print(f"🚫 Quality Fail:     {stats['REJECTED_QUALITY']}")
    print(f"⚠️ Content Fail:     {stats['REJECTED_CONTENT']}")
    if stats['ERRORS'] > 0:
        print(f"❌ Errors:           {stats['ERRORS']}")
    print("="*40)

if __name__ == "__main__":
    main()