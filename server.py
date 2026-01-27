import os
import shutil
import uvicorn
import json
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager

# === imports ===
from blur.process import process_media
from data_filtering.video_benchmark import analyze_video
from data_filtering.hoi_system import HandObjectInteractionSystem

GLOBAL_HOI_SYS = None

# === config ===
DIRS = {
    "TEMP": "server_data/temp_uploads",
    "ACCEPTED": "server_data/accepted_videos",
    "REJECTED": "server_data/rejected_videos",
    "LOGS": "server_data/processing_logs"
}

for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global GLOBAL_HOI_SYS
    print("🚀 [Server] loading (YOLO + Depth)...")
    try:
        GLOBAL_HOI_SYS = HandObjectInteractionSystem(model_size='yolov8s.pt')
        print("✅ [Server] loaded successfully.")
    except Exception as e:
        print(f"❌ [Server] load failed: {e}")
    yield
    print("🛑 [Server] server shutdown.")

app = FastAPI(lifespan=lifespan)

def save_server_log(data, filename):
    """save JSON log on server side"""
    json_name = f"{os.path.splitext(filename)[0]}_result.json"
    log_path = os.path.join(DIRS["LOGS"], json_name)
    data["processed_at"] = datetime.now().isoformat()
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"📝 [Server Log] Saved: {json_name}")

@app.post("/analyze")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    temp_file_path = os.path.join(DIRS["TEMP"], file.filename)
    
    # initialize response structure
    final_response = {
        "filename": file.filename,
        "pipeline_status": "ERROR",
        "reject_reason": None,
        "processed_at": None,  # will be filled later
        "quality_data": None,
        "content_data": None
    }

    try:
        # 1. save uploaded file temporarily
        print(f"📥 [Recv] receive: {file.filename}")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 1. quality check
        print(f"🔍 [Step 1] running quality check...")
        quality_results_list = process_media(
            inputs=[temp_file_path],
            threshold=100.0,
            video_blur_ratio=0.3,
            enable_blur_test=True,
            enable_jitter_test=True,
            enable_exposure_test=True,
            verbose=False
        )

        if not quality_results_list:
            raise ValueError("Process media returned no results")

        q_res = quality_results_list[0]
        
        # 填充质量数据
        final_response["quality_data"] = {
            "passed": bool(q_res['keep']),
            "blur_score": float(q_res['blur_video']),
            "blur_ratio": float(q_res.get('blur_ratio', 0.0)),
            "jitter_disp_pct": float(q_res.get('displacement_percentage', 0.0)),
            "jitter_hf_energy": float(q_res.get('hf_energy_ratio', 0.0)),
            "is_shake": bool(q_res.get('is_shake', False)),
            "exposure_ratio": float(q_res.get('exposure_ratio', 0.0)),
            "max_consecutive_bad_exp": int(q_res.get('max_consecutive_bad_exposure', 0))
        }

        if not q_res['keep']:
            final_response["pipeline_status"] = "REJECTED_QUALITY"
            final_response["reject_reason"] = "Video quality too low"
            print(f"❌ [Result] Quality check failed")
        else:
            # 2. 内容检测
            print(f"🧠 [Step 2] running content analysis...")
            content_res = analyze_video(temp_file_path, hoi_sys=GLOBAL_HOI_SYS)
            
            if content_res:
                final_response["content_data"] = {
                    "passed": bool(content_res['keep']),
                    "interaction_ratio": float(content_res['interaction_ratio']),
                    "missing_hand_ratio": float(content_res['missing_hand_ratio']),
                    "processing_time": float(content_res['processing_time'])
                }

                if content_res['keep']:
                    final_response["pipeline_status"] = "ACCEPTED"
                    print(f"✅ [Result] Perfect pass!")
                else:
                    final_response["pipeline_status"] = "REJECTED_CONTENT"
                    final_response["reject_reason"] = "No valid interaction"
                    print(f"⚠️ [Result] Content mismatch")
            else:
                final_response["pipeline_status"] = "ERROR_CONTENT"
                final_response["reject_reason"] = "Content analysis failed"

    except Exception as e:
        import traceback
        traceback.print_exc()
        final_response["pipeline_status"] = "SERVER_ERROR"
        final_response["reject_reason"] = str(e)
    
    finally:
        # === set timestamp ===
        # make sure Server log and Client received JSON have the same timestamp
        final_response["processed_at"] = datetime.now().isoformat()

        # 1. 服务端保存日志
        try:
            save_server_log(final_response, file.filename)
        except Exception as e:
            print(f"❌ log save failed: {e}")

        # 2. 文件归档
        if os.path.exists(temp_file_path):
            if final_response["pipeline_status"] == "ACCEPTED":
                target_folder = DIRS["ACCEPTED"]
                icon = "✅"
            else:
                target_folder = DIRS["REJECTED"]
                icon = "🚫"
            
            dest_path = os.path.join(target_folder, file.filename)
            try:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                shutil.move(temp_file_path, dest_path)
                print(f"{icon} [Storage] video moved to: {dest_path}")
            except Exception as e:
                print(f"❌ failed to move file: {e}")

    return final_response  # return final JSON response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)