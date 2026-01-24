from contextlib import asynccontextmanager
import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uvicorn

from blur.process import process_media

from data_filtering.hoi_system import HandObjectInteractionSystem
from data_filtering.video_benchmark import analyze_video

app = FastAPI(title="Video Quality & Content API")

# === 配置路径 ===
UPLOAD_DIR = "./uploads"       # 临时上传区
APPROVED_DIR = "./uploads_approved" # 合格视频存档区
REJECTED_DIR = "./uploads_rejected" # (可选) 垃圾桶

# === 全局变量 ===
GLOBAL_HOI_SYS = None

# === 3. 定义生命周期管理器 ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时运行：加载模型
    global GLOBAL_HOI_SYS
    print("⚡️ 正在加载 YOLO 模型... (这只需要一次)")
    GLOBAL_HOI_SYS = HandObjectInteractionSystem(model_size='yolov8s.pt')
    print("✅ 模型加载完毕，服务已就绪！")
    
    yield  # 服务运行中...
    
    # 关闭时运行：清理资源
    print("正在清理资源...")
    GLOBAL_HOI_SYS = None
# 确保文件夹存在
for path in [UPLOAD_DIR, APPROVED_DIR, REJECTED_DIR]:
    os.makedirs(path, exist_ok=True)

class VideoResponse(BaseModel):
    filename: str
    status: str  # "approved", "rejected", "processing_error"
    quality_score: float = 0.0
    interaction_score: float = 0.0
    details: dict = {}

@app.post("/upload_video/", response_model=VideoResponse)
def process_video_endpoint(file: UploadFile = File(...)):
    """
    上传视频 -> 质量检测 -> 内容检测 -> 归档
    """
    # 1. 生成唯一文件名，防止覆盖
    file_ext = file.filename.split('.')[-1]
    unique_name = f"{uuid.uuid4()}.{file_ext}"
    temp_path = os.path.join(UPLOAD_DIR, unique_name)

    # 2. 保存上传的文件到临时区
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

    # 3. 第一步：运行质量清洗 (process_media)
    # 注意：process_media 接收 list，返回 list
    try:
        # 开启 jitter_test 对 VISOR 很重要
        quality_results = process_media(
            inputs=[temp_path],
            enable_jitter_test=True,
            enable_blur_test=True,
            record=False # API模式下不要生成大量中间图片
        )
        
        if not quality_results:
            raise ValueError("质量检测未返回结果")
            
        q_res = quality_results[0] # 取第一个结果
        
    except Exception as e:
        # 清理坏文件
        if os.path.exists(temp_path): os.remove(temp_path)
        return {
            "filename": file.filename, 
            "status": "processing_error", 
            "details": {"error": f"质量检测崩溃: {str(e)}"}
        }

    # 4. 判断质量是否合格
    if not q_res.get('keep', False):
        # --- 质量不合格 ---
        # 移动到拒绝文件夹 (或直接 os.remove 删除)
        shutil.move(temp_path, os.path.join(REJECTED_DIR, unique_name))
        
        return {
            "filename": file.filename,
            "status": "rejected",
            "quality_score": q_res.get('blur_video', 0),
            "details": {
                "reason": "Quality Check Failed",
                "is_shake": q_res.get('is_shake'),
                "blur_score": q_res.get('blur_video')
            }
        }

    # 5. 第二步：运行内容分析 (analyze_video)
    # 只有质量合格才跑这一步，节省算力
    try:
        # 🟢 关键修改：把全局模型传进去
        c_res = analyze_video(temp_path, hoi_sys=GLOBAL_HOI_SYS)
    except Exception as e:
        return {
            "filename": file.filename, 
            "status": "processing_error", 
            "details": {"error": f"内容分析崩溃: {str(e)}"}
        }
    # 6. 判断内容是否合格 (双重验证)
    final_approved = c_res.get('keep', False)
    
    if final_approved:
        # --- ✅ 完全合格 ---
        final_path = os.path.join(APPROVED_DIR, unique_name)
        shutil.move(temp_path, final_path)
        status = "approved"
    else:
        # --- ❌ 内容不符 (虽然画质好) ---
        shutil.move(temp_path, os.path.join(REJECTED_DIR, unique_name))
        status = "rejected"

    # 7. 返回综合报告
    return {
        "filename": file.filename,
        "status": status,
        "quality_score": q_res.get('blur_video', 0),
        "interaction_score": c_res.get('interaction_ratio', 0),
        "details": {
            "quality_metrics": {
                "shake": q_res.get('is_shake'),
                "blur": q_res.get('blur_video')
            },
            "content_metrics": {
                "has_interaction": c_res.get('is_interaction'),
                "interaction_ratio": c_res.get('interaction_ratio'),
                "missing_hand_ratio": c_res.get('missing_hand_ratio')
            },
            "server_path": unique_name
        }
    }

if __name__ == "__main__":
    # 启动服务器，端口 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)