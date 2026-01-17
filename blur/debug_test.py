#!/usr/bin/env python3
"""
调试测试脚本
"""

import subprocess
import sys
import os
import pathlib

def main():
    print("开始调试测试")

    test_data_path = "/media/cwr/新加卷/Detection_Data/vedio/original_data/test"

    # 查找视频文件
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []

    path = pathlib.Path(test_data_path)
    if path.exists():
        print(f"路径存在: {test_data_path}")
        for ext in video_extensions:
            files = list(path.rglob(f'*{ext}'))
            print(f"找到 {len(files)} 个 {ext} 文件")
            video_files.extend(files)
    else:
        print(f"路径不存在: {test_data_path}")
        return

    video_files = sorted(video_files)
    print(f"总共找到 {len(video_files)} 个视频文件")

    if not video_files:
        print("没有找到视频文件")
        return

    # 测试前3个视频
    for i, video_path in enumerate(video_files[:3], 1):
        print(f"\n测试视频 {i}: {video_path.name}")

        cmd = [
            sys.executable, "process.py",
            "-i", str(video_path),
            "--sample-rate", "5.0",
            "-t", "50.0",
            "--motion-method", "lk",
            "--motion-skip-frames", "1"
        ]

        print(f"执行命令: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
                timeout=60  # 1分钟超时
            )

            print(f"返回码: {result.returncode}")

            if result.returncode == 0:
                # 查找结果行
                for line in result.stdout.split('\n'):
                    if 'Video:' in line and 'Blur:' in line:
                        print(f"结果: {line.strip()}")
                        break
                else:
                    print("未找到结果行")
                    print("STDOUT:", result.stdout[-500:])  # 最后500字符
            else:
                print("处理失败")
                print("STDERR:", result.stderr[:500])  # 前500字符

        except subprocess.TimeoutExpired:
            print("超时")
        except Exception as e:
            print(f"异常: {e}")

if __name__ == "__main__":
    main()

