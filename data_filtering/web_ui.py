import gradio as gr
import os
import sys

# 确保可以导入同目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hand_filter_demo import HandDataFilter

def process_video_wrapper(video_path):
    """
    Gradio 包装函数：接收视频路径，调用算法处理，返回结果和处理后的视频路径
    """
    if not video_path:
        return {"status": "Error", "reason": "No video uploaded"}, None

    print(f"📥 收到视频: {video_path}")
    
    # 生成输出视频路径
    # Gradio 的输入通常在临时目录，我们把输出也放在旁边
    output_path = os.path.splitext(video_path)[0] + "_processed.mp4"
    
    # 初始化过滤器 (使用默认推荐参数)
    # min_conf=0.5: 保证召回率
    # missing_tolerance=10: 允许短暂遮挡
    try:
        filter_tool = HandDataFilter(min_conf=0.5, missing_tolerance=10, check_border=True)
        
        # 运行处理逻辑 (visualize=True 会生成带画图的视频)
        result = filter_tool.process_video(video_path, output_path=output_path, visualize=True)
        
        # 返回: (JSON结果, 视频文件路径)
        return result, output_path
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "Critical Error", "reason": str(e)}, None

# 构建 Gradio 界面
iface = gr.Interface(
    fn=process_video_wrapper,
    inputs=gr.Video(label="上传视频 (Upload Video)", interactive=True),
    outputs=[
        gr.JSON(label="算法分析结果 (Analysis Result)"),
        gr.Video(label="可视化输出 (Visualized Output)")
    ],
    title="🖐️ 手部交互数据过滤器 (Hand Interaction Data Filter)",
    description="""
    ### 使用说明
    1. 点击下方上传包含手部操作的视频。
    2. 点击 **Submit** 按钮开始处理。
    3. 等待算法运行完毕后，右侧将显示每一帧的检测结果统计，底部将显示画好框的视频。
    
    **功能**:
    - 自动检测双手是否存在
    - 结合 YOLOv8 检测手部是否与物体交互 (Grasping/Hovering)
    - 过滤掉严重丢帧或手部出画的数据
    """,
    theme="default",
    allow_flagging="never"
)

if __name__ == "__main__":
    print("🚀 正在启动网页服务...")
    print("请在浏览器打开显示的链接 (通常是 http://127.0.0.1:7860)")
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
