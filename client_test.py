import requests
import os
# 服务端地址（替换成你的服务器内网IP）
url = "http://192.168.x.x:8000/process"  # 示例：http://10.0.0.123:8000/process

# 要上传的文件路径（支持多个）
file_paths = [
    "/path/to/video1.mp4",
    "/path/to/image1.jpg",
    # "/path/to/video2.mp4",
]

files = [('files', (os.path.basename(p), open(p, 'rb'))) for p in file_paths]

try:
    response = requests.post(url, files=files)
    response.raise_for_status()
    print("返回结果：")
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")