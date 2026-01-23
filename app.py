#!/usr/bin/env python3
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import tempfile
from typing import List
import uuid  # 新增：用于生成唯一文件名
from datetime import datetime
# internal imports
from blur.process import process_media, load_config

app = FastAPI(
    title="Media Process API",
    description="upload video files, run blur/jitter/exposure tests on media files via API",
    version="1.0"
)

# 指定固定上传目录（改成你想要的路径，推荐相对路径）
UPLOAD_DIR = "uploads"  # 项目根目录下的 uploads 文件夹
os.makedirs(UPLOAD_DIR, exist_ok=True)  # 自动创建如果不存在

@app.post("/process")
async def process_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="至少上传一个文件")

    input_paths = []

    try:
        for file in files:
            if not file.content_type.startswith(("video/", "image/")):
                raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file.filename}")

            # 生成唯一文件名（原名 + 时间戳 + 随机UUID，防止覆盖）
            original_name = file.filename
            unique_suffix = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
            file_extension = os.path.splitext(original_name)[1]
            unique_filename = os.path.splitext(original_name)[0] + "_" + unique_suffix + file_extension

            file_path = os.path.join(UPLOAD_DIR, unique_filename)

            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            input_paths.append(file_path)

            # 可选：打印日志方便调试
            print(f"文件已保存: {file_path}")
        # 加载配置（保持和你原脚本一致）
        config = load_config()

        # 调用原函数处理（固定参数和你原脚本保持一致）
        results = process_media(
            input_paths,
            threshold=config['blur_thresh'],
            fix_size=config['fix_size'],
            output_dir=config['output_dir'],  # 如果会生成输出文件，会保存在这里
            video_blur_ratio=config['blur_ratio'],
            sample_rate=config['sample'],
            motion_method=config['motion'],
            verbose=config['verbose'],
            display=config['display'],
            record=False,
            enable_blur_test=config['blur_test'],
            enable_jitter_test=config['jitter_test'],
            enable_exposure_test=config['exposure_test'],
            show_jitter_plot=False
        )

        # 返回结果时，顺便告诉客户端文件保存路径
        return JSONResponse(content={
            "status": "success",
            "processed_count": len(results),
            "results": results,
            "saved_files": input_paths,  # 新增：返回保存路径，方便你后续读取
            "message": f"处理完成，共处理 {len(results)} 个文件，文件已保存到 {UPLOAD_DIR}"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.get("/")
def root():
    return {"message": "Media Process API has started. POST /process to upload files, visit /docs for interactive documentation"}