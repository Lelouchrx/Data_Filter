import os
import shutil
import uvicorn
import json
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager

# === 导入你的处理模块 ===
from blur.process import process_media
from data_filtering.video_benchmark import analyze_video
from data_filtering.hoi_system import HandObjectInteractionSystem

# === 全局变量 ===
GLOBAL_HOI_SYS = None

# === 目录配置 ===
DIRS = {
    "TEMP": "server_data/temp_uploads",        # 1. 临时接收
    "ACCEPTED": "server_data/accepted_videos", # 2. 通过的视频
    "REJECTED": "server_data/rejected_videos", # 3. 被拒绝的视频
    "LOGS": "server_data/processing_logs"      # 4. JSON 结果日志
}

for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

# === 生命周期管理 ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    global GLOBAL_HOI_SYS
    print("🚀 [Server] 正在初始化模型 (YOLO + Depth)...")
    try:
        GLOBAL_HOI_SYS = HandObjectInteractionSystem(model_size='yolov8s.pt')
        print("✅ [Server] 模型加载完成。")
    except Exception as e:
        print(f"❌ [Server] 模型加载失败: {e}")
    yield
    print("🛑 [Server] 服务关闭。")

app = FastAPI(lifespan=lifespan)

def save_log(data, filename):
    json_name = f"{os.path.splitext(filename)[0]}_result.json"
    log_path = os.path.join(DIRS["LOGS"], json_name)
    data["processed_at"] = datetime.now().isoformat()
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"📝 [Log] 结果日志已保存: {json_name}")

@app.post("/analyze")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    temp_file_path = os.path.join(DIRS["TEMP"], file.filename)
    
    final_response = {
        "filename": file.filename,
        "pipeline_status": "ERROR",
        "reject_reason": None,
        "quality_data": None,
        "content_data": None
    }

    try:
        # 1. 保存文件
        print(f"📥 [Recv] 接收文件: {file.filename}")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. 质量检测
        print(f"🔍 [Step 1] 运行质量检测...")
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
        
        # === 修复点：确保所有字段都存在 ===
        final_response["quality_data"] = {
            "passed": bool(q_res['keep']),
            "blur_score": float(q_res['blur_video']),
            "blur_ratio": float(q_res.get('blur_ratio', 0.0)), # <--- 之前漏了这行
            "jitter_disp_pct": float(q_res.get('displacement_percentage', 0.0)),
            "jitter_hf_energy": float(q_res.get('hf_energy_ratio', 0.0)), # <--- 客户端可能也需要这个
            "is_shake": bool(q_res.get('is_shake', False)),
            "exposure_ratio": float(q_res.get('exposure_ratio', 0.0)),
            "max_consecutive_bad_exp": int(q_res.get('max_consecutive_bad_exposure', 0))
        }

        if not q_res['keep']:
            final_response["pipeline_status"] = "REJECTED_QUALITY"
            final_response["reject_reason"] = "Video quality too low"
            print(f"❌ [Result] 质量检测未通过")
        else:
            # 3. 内容检测
            print(f"🧠 [Step 2] 运行内容分析...")
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
                    print(f"✅ [Result] 完美通过！")
                else:
                    final_response["pipeline_status"] = "REJECTED_CONTENT"
                    final_response["reject_reason"] = "No valid interaction"
                    print(f"⚠️ [Result] 内容不符")
            else:
                final_response["pipeline_status"] = "ERROR_CONTENT"
                final_response["reject_reason"] = "Content analysis failed"

    except Exception as e:
        import traceback
        traceback.print_exc()
        final_response["pipeline_status"] = "SERVER_ERROR"
        final_response["reject_reason"] = str(e)
    
    finally:
        # 保存日志
        try:
            save_log(final_response, file.filename)
        except:
            pass

        # 移动文件
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
                print(f"{icon} [Storage] 视频已移动至: {dest_path}")
            except Exception as e:
                print(f"❌ 移动文件失败: {e}")

    return final_response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)