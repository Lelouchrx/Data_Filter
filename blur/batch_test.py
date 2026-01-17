#!/usr/bin/env python3
"""
批量测试脚本 - 对测试文件夹中的所有视频进行模糊检测处理
"""

import os
import sys
import subprocess
import argparse
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


def setup_logging(log_file=None):
    """设置日志记录"""
    level = logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def find_video_files(test_dir):
    """查找测试目录中的所有视频文件"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
    video_files = []

    test_path = Path(test_dir)
    if not test_path.exists():
        logging.error(f"测试目录不存在: {test_dir}")
        return video_files

    for file_path in test_path.rglob('*'):
        if file_path.is_file() and file_path.suffix in video_extensions:
            video_files.append(file_path)

    logging.info(f"找到 {len(video_files)} 个视频文件")
    return sorted(video_files)


def process_single_video(video_path, sample_rate=5.0, threshold=50.0, motion_method='farneback', motion_skip_frames=5, output_dir=None):
    """处理单个视频文件"""
    try:
        cmd = [
            sys.executable, 'process.py',
            '-i', str(video_path),
            '--sample-rate', str(sample_rate),
            '-t', str(threshold),
            '--motion-method', motion_method,
            '--motion-skip-frames', str(motion_skip_frames)
        ]

        if output_dir:
            cmd.extend(['-o', output_dir])

        logging.info(f"开始处理视频: {video_path.name}")
        logging.debug(f"执行命令: {' '.join(cmd)}")

        # 执行命令
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        end_time = time.time()

        processing_time = end_time - start_time

        if result.returncode == 0:
            logging.info(f"成功处理 {video_path.name}，耗时 {processing_time:.2f} 秒")
            # 解析process.py的JSON输出
            import json
            blur_score = jitter_score = valid_ratio = keep_decision = None

            if result.stdout:
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if line.startswith('RESULT_JSON:'):
                        try:
                            json_str = line.replace('RESULT_JSON:', '').strip()
                            result_data = json.loads(json_str)
                            blur_score = result_data.get('blur_video')
                            jitter_score = result_data.get('jitter_video')
                            valid_ratio = result_data.get('valid_ratio')
                            keep_decision = result_data.get('keep', False)
                            break
                        except json.JSONDecodeError as e:
                            logging.warning(f"解析JSON失败: {e}")
                            break

            # 如果JSON解析失败，尝试从日志中解析（备用方案）
            if blur_score is None and result.stdout:
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if 'Video:' in line and 'Blur:' in line:
                        # 解析视频统计信息
                        parts = line.split('|')
                        for part in parts:
                            if 'Blur:' in part:
                                try:
                                    blur_score = float(part.split(':')[1].strip())
                                except:
                                    pass
                            elif 'Jitter:' in part:
                                try:
                                    jitter_score = float(part.split(':')[1].strip())
                                except:
                                    pass
                            elif 'Valid:' in part:
                                try:
                                    valid_ratio = float(part.split(':')[1].strip().rstrip('%'))
                                except:
                                    pass
                            elif 'Keep:' in part:
                                keep_decision = 'True' in part or 'keep' in part.lower()

            return {
                'video_path': str(video_path),
                'success': True,
                'processing_time': processing_time,
                'blur_score': blur_score,
                'jitter_score': jitter_score,
                'valid_ratio': valid_ratio,
                'keep_decision': keep_decision
            }
        else:
            logging.error(f"处理失败 {video_path.name}: {result.stderr}")
            return {
                'video_path': str(video_path),
                'success': False,
                'error': result.stderr,
                'processing_time': processing_time
            }

    except Exception as e:
        logging.error(f"处理视频 {video_path.name} 时发生异常: {str(e)}")
        return {
            'video_path': str(video_path),
            'success': False,
            'error': str(e),
            'processing_time': 0
        }


def batch_process_videos(test_dir, max_workers=4, **kwargs):
    """批量处理视频文件"""
    video_files = find_video_files(test_dir)
    if not video_files:
        logging.warning("没有找到视频文件")
        return []

    results = []
    total_files = len(video_files)

    logging.info(f"开始批量处理，共 {total_files} 个视频文件，使用 {max_workers} 个并行线程")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_video = {
            executor.submit(process_single_video, video_path, **kwargs): video_path
            for video_path in video_files
        }

        # 收集结果
        completed = 0
        for future in as_completed(future_to_video):
            video_path = future_to_video[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                logging.info(f"进度: {completed}/{total_files} ({completed/total_files*100:.1f}%)")

                # 显示当前视频的处理结果
                video_name = Path(result['video_path']).name
                if result.get('success', False):
                    blur = result.get('blur_score', 'N/A')
                    jitter = result.get('jitter_score', 'N/A')
                    valid = result.get('valid_ratio', 'N/A')
                    keep = "保留" if result.get('keep_decision', False) else "丢弃"
                    time_taken = result.get('processing_time', 0)
                    
                    # 检查抖动检测状态
                    jitter_status = ""
                    if jitter == 'N/A' or jitter is None:
                        jitter_status = " [抖动未检测]"
                    elif isinstance(jitter, (int, float)):
                        if jitter == 0.0:
                            jitter_status = " [抖动未检测到]"
                        elif jitter < 1.0:
                            jitter_status = " [抖动轻微]"
                        elif jitter >= 6.0:
                            jitter_status = " [抖动严重]"
                    
                    # 格式化有效率
                    valid_str = f"{valid:.1%}" if isinstance(valid, (int, float)) else str(valid)
                    
                    print(f"✓ {video_name}")
                    print(f"  模糊度: {blur} | 抖动: {jitter} | 有效率: {valid_str} | 决策: {keep} | 耗时: {time_taken:.2f}秒{jitter_status}")
                    print()
                else:
                    print(f"✗ {video_name}: 处理失败 - {result.get('error', '未知错误')}")
                    print()

            except Exception as e:
                logging.error(f"处理 {video_path.name} 时发生异常: {str(e)}")
                results.append({
                    'video_path': str(video_path),
                    'success': False,
                    'error': str(e),
                    'processing_time': 0
                })
                completed += 1

    return results


def save_results(results, output_file):
    """保存处理结果到文件"""
    import json

    # 添加统计信息
    successful = sum(1 for r in results if r.get('success', False))
    total = len(results)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_videos': total,
        'successful': successful,
        'failed': total - successful,
        'success_rate': successful / total * 100 if total > 0 else 0,
        'results': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info(f"结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='批量测试视频模糊检测')
    parser.add_argument('-d', '--test-dir', required=True, help='测试视频文件夹路径')
    parser.add_argument('-o', '--output-dir', help='输出目录（可选）')
    parser.add_argument('-w', '--max-workers', type=int, default=2, help='最大并行处理数（默认: 2）')
    parser.add_argument('-l', '--log-file', help='日志文件路径（可选）')
    parser.add_argument('-r', '--result-file', help='结果文件路径（可选，默认: batch_results_YYYYMMDD_HHMMSS.json）')
    parser.add_argument('--sample-rate', type=float, default=5.0, help='采样率（默认: 5.0）')
    parser.add_argument('-t', '--threshold', type=float, default=50.0, help='模糊阈值（默认: 50.0）')
    parser.add_argument('--motion-method', default='farneback', choices=['farneback', 'lk'], help='运动检测方法（默认: farneback）')
    parser.add_argument('--motion-skip-frames', type=int, default=5, help='运动检测跳帧数（默认: 5）')

    args = parser.parse_args()

    # 设置日志
    if args.log_file:
        setup_logging(args.log_file)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'batch_test_{timestamp}.log'
        setup_logging(log_file)
        logging.info(f"日志将保存到: {log_file}")

    # 设置结果文件
    if not args.result_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.result_file = f'batch_results_{timestamp}.json'

    logging.info("="*60)
    logging.info("批量视频模糊检测测试开始")
    logging.info("="*60)
    logging.info(f"测试目录: {args.test_dir}")
    logging.info(f"采样率: {args.sample_rate}")
    logging.info(f"模糊阈值: {args.threshold}")
    logging.info(f"运动检测方法: {args.motion_method}")
    logging.info(f"运动跳帧数: {args.motion_skip_frames}")
    logging.info(f"并行处理数: {args.max_workers}")
    if args.output_dir:
        logging.info(f"输出目录: {args.output_dir}")
    logging.info(f"结果文件: {args.result_file}")
    logging.info("="*60)

    # 开始批量处理
    start_time = time.time()
    results = batch_process_videos(
        test_dir=args.test_dir,
        max_workers=args.max_workers,
        sample_rate=args.sample_rate,
        threshold=args.threshold,
        motion_method=args.motion_method,
        motion_skip_frames=args.motion_skip_frames,
        output_dir=args.output_dir
    )
    end_time = time.time()

    total_time = end_time - start_time

    # 保存结果
    save_results(results, args.result_file)

    # 输出统计信息
    successful = sum(1 for r in results if r.get('success', False))
    total = len(results)

    logging.info("="*60)
    logging.info("批量处理完成")
    logging.info("="*60)
    logging.info(f"总视频数: {total}")
    logging.info(f"成功处理: {successful}")
    logging.info(f"处理失败: {total - successful}")
    logging.info(f"成功率: {successful/total*100:.1f}%" if total > 0 else "成功率: N/A")
    logging.info(f"总处理时间: {total_time:.2f} 秒")
    logging.info(f"平均处理时间: {total_time/total:.2f} 秒/视频" if total > 0 else "平均处理时间: N/A")
    logging.info("="*60)


if __name__ == '__main__':
    main()
