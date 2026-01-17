#!/usr/bin/env python3
"""
多线程批量视频质量检测脚本
"""

import subprocess
import sys
import os
import pathlib
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

def find_video_files(test_data_path):
    """查找所有视频文件"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
    video_files = []

    path = pathlib.Path(test_data_path)
    if path.exists():
        for ext in video_extensions:
            video_files.extend(list(path.rglob(f'*{ext}')))

    return sorted(video_files)

def process_single_video(video_path, output_dir, sample_rate=10.0, threshold=50.0):
    """处理单个视频"""
    cmd = [
        sys.executable, "process.py",
        "-i", str(video_path),
        "-o", output_dir,
        "--sample-rate", str(sample_rate),
        "-t", str(threshold),
        "--motion-method", "lk",
        "--motion-skip-frames", "5"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            timeout=300  # 5分钟超时
        )

        # 解析结果
        video_result = None
        if result.returncode == 0:
            # 从输出中提取结果
            for line in result.stdout.split('\n'):
                if 'Video:' in line and 'Blur:' in line:
                    parts = line.split('|')
                    video_name = parts[0].split(':')[1].strip()
                    blur = float(parts[1].split(':')[1].strip())
                    jitter = float(parts[2].split(':')[1].strip())
                    valid = float(parts[3].split(':')[1].strip())
                    keep = parts[4].split(':')[1].strip().lower() == 'true'

                    video_result = {
                        'video_path': str(video_path),
                        'video_name': video_name,
                        'blur': blur,
                        'jitter': jitter,
                        'valid': valid,
                        'keep': keep,
                        'success': True
                    }
                    break

        if video_result is None:
            video_result = {
                'video_path': str(video_path),
                'video_name': video_path.name,
                'success': False,
                'error': result.stderr or 'Unknown error'
            }

        return video_result

    except subprocess.TimeoutExpired:
        return {
            'video_path': str(video_path),
            'video_name': video_path.name,
            'success': False,
            'error': 'Timeout (5 minutes)'
        }
    except Exception as e:
        return {
            'video_path': str(video_path),
            'video_name': video_path.name,
            'success': False,
            'error': str(e)
        }

def main():
    print("🚀 多线程批量视频质量检测")
    print("=" * 60)

    # 配置参数
    test_data_path = "/media/cwr/新加卷/Detection_Data/vedio/original_data/RealSource-World"
    output_dir = "batch_output"
    max_workers = 4  # 同时处理4个视频
    sample_rate = 10.0
    threshold = 50.0

    # 检查路径
    if not os.path.exists(test_data_path):
        print(f"❌ 错误：测试数据路径不存在: {test_data_path}")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 查找视频文件
    print(f"🔍 扫描视频文件: {test_data_path}")
    video_files = find_video_files(test_data_path)
    print(f"📋 找到 {len(video_files)} 个视频文件")

    if not video_files:
        print("❌ 未找到任何视频文件")
        return

    # 显示待处理视频
    print("\n📝 待处理的视频文件:")
    for i, video_file in enumerate(video_files, 1):
        print("3d")
    print()

    # 开始批量处理
    print(f"⚡ 开始多线程处理 (最大并发: {max_workers})")
    print(f"📁 输出目录: {output_dir}")
    print("-" * 60)

    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_video = {
            executor.submit(process_single_video, video_path, output_dir, sample_rate, threshold): video_path
            for video_path in video_files
        }

        # 处理结果
        completed = 0
        for future in as_completed(future_to_video):
            video_path = future_to_video[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1

                # 显示进度
                status = "✅" if result.get('success', False) else "❌"
                keep_status = "保留" if result.get('keep', False) else "丢弃"
                if result.get('success', False):
                    print(f"[{completed:2d}/{len(video_files):2d}] {status} {video_path.name}: {keep_status}")
                else:
                    print(f"[{completed:2d}/{len(video_files):2d}] {status} {video_path.name}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"❌ 处理 {video_path.name} 时发生异常: {e}")
                completed += 1

    # 计算总时间
    end_time = time.time()
    total_time = end_time - start_time

    print("-" * 60)
    print("📊 处理完成！")
    print(".2f")
    print(".2f")

    # 统计结果
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    kept = [r for r in successful if r.get('keep', False)]
    discarded = [r for r in successful if not r.get('keep', False)]

    print(f"\n📈 详细统计:")
    print(f"   成功处理: {len(successful)}/{len(results)}")
    print(f"   处理失败: {len(failed)}/{len(results)}")
    print(f"   保留视频: {len(kept)}")
    print(f"   丢弃视频: {len(discarded)}")

    # 保存详细结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"batch_results_{timestamp}.json"

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_videos': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'kept': len(kept),
                'discarded': len(discarded),
                'total_time_seconds': total_time,
                'avg_time_per_video': total_time / len(results) if results else 0
            },
            'config': {
                'test_data_path': test_data_path,
                'output_dir': output_dir,
                'sample_rate': sample_rate,
                'threshold': threshold,
                'max_workers': max_workers
            },
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 详细结果已保存到: {results_file}")
    print("=" * 60)

    # 显示前几个失败的视频
    if failed:
        print("\n❌ 处理失败的视频:")
        for fail in failed[:5]:  # 只显示前5个
            print(f"   {fail['video_name']}: {fail.get('error', 'Unknown error')}")
        if len(failed) > 5:
            print(f"   ... 还有 {len(failed) - 5} 个")

    # 显示一些保留/丢弃的示例
    if kept:
        print("\n✅ 保留的视频示例:")
        for video in kept[:3]:
            print(f"   {video['video_name']}: Blur={video['blur']:.1f}, Jitter={video['jitter']:.1f}")

    if discarded:
        print("\n❌ 丢弃的视频示例:")
        for video in discarded[:3]:
            print(f"   {video['video_name']}: Blur={video['blur']:.1f}, Jitter={video['jitter']:.1f}")

if __name__ == "__main__":
    main()
