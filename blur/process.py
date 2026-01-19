import sys
import argparse
import logging
import pathlib
import time
import json

import cv2
import numpy as np

from detection import estimate_blur
from detection import fix_image_size
from detection import pretty_blur_map
from detection import estimate_motion
from detection import calc_jitter
from config import (
    OUTPUT_DIR, THRESHOLD, FIX_SIZE, VIDEO_BLUR_RATIO, SAMPLE_RATE,
    MOTION_METHOD, MOTION_SKIP_FRAMES, JITTER_THRESHOLD, VALID_RATIO_THRESHOLD,
    IMG_EXTENSIONS, VIDEO_EXTENSIONS, VERBOSE, DISPLAY
)


def parse_args():
    parser = argparse.ArgumentParser(description='run blur detection on images and videos')
    parser.add_argument('-i', '--inputs', type=str, nargs='+', required=True, help='directory of images/videos')
    parser.add_argument('-o', '--output-dir', type=str, default='frames_output', help='directory to save frame images')

    parser.add_argument('-t', '--threshold', type=float, default=20.0, help='blurry threshold')
    parser.add_argument('-f', '--variable-size', action='store_true', help='fix the image size')

    parser.add_argument('-v', '--verbose', action='store_true', help='set logging level to debug')
    parser.add_argument('-d', '--display', action='store_true', help='display images')
    parser.add_argument('--video-blur-ratio', type=float, default=0.3, help='ratio of blurry frames to mark video as unusable')
    parser.add_argument('--sample-rate', type=float, default=10.0, help='sample rate per second (frames per second)')
    parser.add_argument('--motion-method', type=str, default='lk', choices=['farneback', 'lk'], help='motion estimation method (lk is faster)')
    parser.add_argument('--motion-skip-frames', type=int, default=5, help='skip frames for motion estimation (higher = faster)')

    return parser.parse_args()


def find_media(media_paths, img_extensions=None, video_extensions=None):
    # 使用传入参数或从config读取默认值
    img_extensions = img_extensions if img_extensions is not None else IMG_EXTENSIONS
    video_extensions = video_extensions if video_extensions is not None else VIDEO_EXTENSIONS
    
    img_extensions += [i.upper() for i in img_extensions]
    video_extensions += [i.upper() for i in video_extensions]
    all_extensions = img_extensions + video_extensions

    for path in media_paths:
        path = pathlib.Path(path)

        if path.is_file():
            if path.suffix not in all_extensions:
                logging.info(f'{path.suffix} is not a supported extension! skipping {path}')
                continue
            else:
                yield path

        if path.is_dir():
            for ext in all_extensions:
                yield from path.rglob(f'*{ext}')


def process_video(video_path, threshold=None, fix_size=None, blur_ratio_threshold=None, output_dir=None, sample_rate=None,
                 motion_method=None, motion_skip_frames=None, jitter_threshold=None, valid_ratio_threshold=None):
    """Process video with optimized motion detection.

    Args:
        video_path: 视频文件路径
        threshold: 模糊阈值（默认从config读取）
        fix_size: 是否固定尺寸（默认从config读取）
        blur_ratio_threshold: 视频模糊帧比例阈值（默认从config读取）
        output_dir: 输出目录（默认从config读取）
        sample_rate: 采样率（默认从config读取）
        motion_method: 运动估计方法 'farneback' 或 'lk'（默认从config读取）
        motion_skip_frames: 运动估计跳帧数（默认从config读取）
        jitter_threshold: 抖动阈值（默认从config读取）
        valid_ratio_threshold: 有效率阈值（默认从config读取）
    """

    threshold = threshold if threshold is not None else THRESHOLD
    fix_size = fix_size if fix_size is not None else FIX_SIZE
    blur_ratio_threshold = blur_ratio_threshold if blur_ratio_threshold is not None else VIDEO_BLUR_RATIO
    output_dir = output_dir if output_dir is not None else OUTPUT_DIR
    sample_rate = sample_rate if sample_rate is not None else SAMPLE_RATE
    motion_method = motion_method if motion_method is not None else MOTION_METHOD
    motion_skip_frames = motion_skip_frames if motion_skip_frames is not None else MOTION_SKIP_FRAMES
    jitter_threshold = jitter_threshold if jitter_threshold is not None else JITTER_THRESHOLD
    valid_ratio_threshold = valid_ratio_threshold if valid_ratio_threshold is not None else VALID_RATIO_THRESHOLD
    
    start_time = time.time()
    logging.info(f"开始处理视频: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        end_time = time.time()
        logging.warning(f'failed to open video {video_path} (耗时: {end_time - start_time:.2f}秒)')
        return None, True

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logging.info(f"视频信息: 分辨率={video_width}x{video_height}, 帧率={fps:.2f}fps, 总帧数={frame_count}")

    if frame_count == 0:
        cap.release()
        return None, True

    if output_dir:
        video_name = video_path.stem
        frames_dir = pathlib.Path(output_dir) / video_name
        frames_dir.mkdir(parents=True, exist_ok=True)
    else:
        frames_dir = None

    # Sample frames based on fps and sample_rate
    frame_interval = max(1, int(fps / sample_rate))
    blurry_frames = 0
    total_samples = 0

    # Motion detection variables
    blur_scores = []
    valid_frames = []
    dx_seq, dy_seq = [], []
    prev_frame = None
    motion_counter = 0  # 单独的运动估计计数器

    for i in range(0, frame_count, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        original_frame = frame.copy()
        if fix_size:
            frame = fix_image_size(frame)

        blur_map, score, blurry = estimate_blur(frame, threshold=threshold)
        blur_scores.append(score)
        valid_frames.append(not blurry)

        # Optimized motion estimation with frame skipping
        if prev_frame is not None:
            # 每次循环都计算运动，但使用单独的计数器来控制跳帧
            if motion_counter % motion_skip_frames == 0:
                dx, dy = estimate_motion(prev_frame, original_frame, method=motion_method)
                dx_seq.append(dx)
                dy_seq.append(dy)
            else:
                # 对于跳过的帧，使用上一次的运动值（或0）
                if len(dx_seq) > 0:
                    dx_seq.append(dx_seq[-1])
                    dy_seq.append(dy_seq[-1])
                else:
                    dx_seq.append(0.0)
                    dy_seq.append(0.0)
            motion_counter += 1

        prev_frame = original_frame

        # Add text overlay with blur score
        display_frame = original_frame.copy()
        status = "BLUR" if blurry else "CLEAR"
        text = f"{status} {score:.1f}"
        cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if not blurry else (0, 0, 255), 2)

        # Save frame with simple naming
        if frames_dir:
            frame_filename = f"{i:04d}_{status}_{score:.1f}.jpg"
            frame_path = frames_dir / frame_filename
            cv2.imwrite(str(frame_path), display_frame)

        if blurry:
            blurry_frames += 1
        total_samples += 1

    cap.release()

    if not blur_scores:
        return None, True

    # Video-level statistics
    blur_video = np.percentile(blur_scores, 10)
    valid_ratio = sum(valid_frames) / len(valid_frames)
    
    # 抖动检测
    if len(dx_seq) < 2:
        logging.warning(f"抖动检测数据不足: dx_seq长度={len(dx_seq)}, dy_seq长度={len(dy_seq)}, 总样本数={len(blur_scores)}")
        jitter_video = 0.0
    else:
        jitter_video = calc_jitter(dx_seq, dy_seq)
        if jitter_video == 0.0 and len(dx_seq) >= 2:
            # 检查是否所有运动值都是0
            total_motion = np.sum(np.abs(dx_seq)) + np.sum(np.abs(dy_seq))
            if total_motion == 0.0:
                logging.warning(f"所有运动值都为0，可能未正确检测到运动")
            else:
                logging.debug(f"抖动检测: dx_seq长度={len(dx_seq)}, dy_seq长度={len(dy_seq)}, 抖动值={jitter_video:.3f}")
    
    # Decision rules - adjusted for better accuracy
    keep = True
    rejection_reasons = []
    
    if blur_video < threshold:
        keep = False
        rejection_reasons.append(f"模糊度过低({blur_video:.1f} < {threshold})")
    
    if jitter_video > jitter_threshold:
        keep = False
        rejection_reasons.append(f"抖动过大({jitter_video:.3f} > {jitter_threshold})")
    
    if valid_ratio < valid_ratio_threshold:
        keep = False
        rejection_reasons.append(f"有效率过低({valid_ratio:.3f} < {valid_ratio_threshold})")
    
    # 获取视频名称（在使用之前定义）
    video_name = video_path.stem
    
    # 记录拒绝原因
    if rejection_reasons:
        logging.info(f"视频 {video_name} 被拒绝，原因: {', '.join(rejection_reasons)}")

    end_time = time.time()
    processing_time = end_time - start_time
    logging.info(f"Video: {video_name} | Resolution: {video_width}x{video_height} | Blur: {blur_video:.1f} | Jitter: {jitter_video:.3f} | Valid: {valid_ratio:.3f} | Keep: {keep} | Time: {processing_time:.2f}s")

    # 输出JSON格式的结果到stdout，方便批量脚本解析
    result_json = {
        'video_name': video_name,
        'blur_video': float(blur_video),
        'jitter_video': float(jitter_video),
        'valid_ratio': float(valid_ratio),
        'keep': bool(keep),
        'total_samples': int(len(blur_scores)),
        'processing_time': float(processing_time)
    }
    print(f"RESULT_JSON:{json.dumps(result_json)}", flush=True)

    return {
        'blur_video': blur_video,
        'jitter_video': jitter_video,
        'valid_ratio': valid_ratio,
        'keep': keep,
        'total_samples': len(blur_scores),
        'processing_time': processing_time
    }, not keep


def process_image(image_path, threshold=None, fix_size=None, output_dir=None, display=None):
    """处理单张图片
    
    Args:
        image_path: 图片文件路径
        threshold: 模糊阈值（默认从config读取）
        fix_size: 是否固定尺寸（默认从config读取）
        output_dir: 输出目录（默认从config读取）
        display: 是否显示图片（默认从config读取）
    
    Returns:
        dict: 包含处理结果的字典，格式为 {'score': float, 'blurry': bool}
    """
    # 使用传入参数或从config读取默认值
    threshold = threshold if threshold is not None else THRESHOLD
    fix_size = fix_size if fix_size is not None else FIX_SIZE
    output_dir = output_dir if output_dir is not None else OUTPUT_DIR
    display = display if display is not None else DISPLAY
    
    image = cv2.imread(str(image_path))
    if image is None:
        logging.warning(f'warning! failed to read image from {image_path}; skipping!')
        return None

    logging.info(f'processing image {image_path}')

    if fix_size:
        image = fix_image_size(image)
    else:
        logging.warning('not normalizing image size for consistent scoring!')

    blur_map, score, blurry = estimate_blur(image, threshold=threshold)

    logging.info(f'image_path: {image_path} score: {score:.1f} blurry: {blurry}')

    # Save image with overlay
    if output_dir:
        img_dir = pathlib.Path(output_dir) / image_path.stem
        img_dir.mkdir(parents=True, exist_ok=True)

        status = "BLUR" if blurry else "CLEAR"
        display_image = image.copy()
        cv2.putText(display_image, f"{status} {score:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                   (0, 255, 0) if not blurry else (0, 0, 255), 2)

        img_filename = f"{image_path.stem}_{status}_{score:.1f}.jpg"
        img_path = img_dir / img_filename
        cv2.imwrite(str(img_path), display_image)

    if display:
        cv2.imshow('input', image)
        cv2.imshow('result', pretty_blur_map(blur_map))

        if cv2.waitKey(0) == ord('q'):
            logging.info('exiting...')
            return {'score': score, 'blurry': blurry}
    
    return {'score': score, 'blurry': blurry}


def process_media(inputs, **kwargs):
    """处理媒体文件（图片或视频）的主函数
    
    Args:
        inputs: 输入路径（字符串、路径对象或列表）
        **kwargs: 可选参数，会覆盖config中的默认值
            - threshold: 模糊阈值
            - fix_size: 是否固定尺寸
            - output_dir: 输出目录
            - video_blur_ratio: 视频模糊帧比例阈值
            - sample_rate: 采样率
            - motion_method: 运动估计方法
            - motion_skip_frames: 运动估计跳帧数
            - jitter_threshold: 抖动阈值
            - valid_ratio_threshold: 有效率阈值
            - verbose: 是否显示详细日志
            - display: 是否显示图片
            - img_extensions: 图片扩展名列表
            - video_extensions: 视频扩展名列表
    
    Returns:
        list: 处理结果列表，每个元素是处理结果字典
    """
    # 从kwargs中提取参数，如果没有则使用config默认值
    threshold = kwargs.get('threshold', THRESHOLD)
    fix_size = kwargs.get('fix_size', FIX_SIZE)
    output_dir = kwargs.get('output_dir', OUTPUT_DIR)
    video_blur_ratio = kwargs.get('video_blur_ratio', VIDEO_BLUR_RATIO)
    sample_rate = kwargs.get('sample_rate', SAMPLE_RATE)
    motion_method = kwargs.get('motion_method', MOTION_METHOD)
    motion_skip_frames = kwargs.get('motion_skip_frames', MOTION_SKIP_FRAMES)
    jitter_threshold = kwargs.get('jitter_threshold', JITTER_THRESHOLD)
    valid_ratio_threshold = kwargs.get('valid_ratio_threshold', VALID_RATIO_THRESHOLD)
    verbose = kwargs.get('verbose', VERBOSE)
    display = kwargs.get('display', DISPLAY)
    img_extensions = kwargs.get('img_extensions', IMG_EXTENSIONS)
    video_extensions = kwargs.get('video_extensions', VIDEO_EXTENSIONS)
    
    # 设置日志级别
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 确保inputs是列表
    if isinstance(inputs, (str, pathlib.Path)):
        inputs = [inputs]
    
    results = []
    video_ext_list = video_extensions + [ext.upper() for ext in video_extensions]
    
    for media_path in find_media(inputs, img_extensions=img_extensions, video_extensions=video_extensions):
        media_path = pathlib.Path(media_path)
        is_video = media_path.suffix in video_ext_list

        if is_video:
            # Process video
            logging.info(f'processing video {media_path}')
            video_result, unusable = process_video(
                media_path,
                threshold=threshold,
                fix_size=fix_size,
                blur_ratio_threshold=video_blur_ratio,
                output_dir=output_dir,
                sample_rate=sample_rate,
                motion_method=motion_method,
                motion_skip_frames=motion_skip_frames,
                jitter_threshold=jitter_threshold,
                valid_ratio_threshold=valid_ratio_threshold
            )

            if video_result is None:
                logging.warning(f'failed to process video {media_path}; skipping!')
                continue
            
            video_result['media_path'] = str(media_path)
            results.append(video_result)
        else:
            # Process image
            image_result = process_image(
                media_path,
                threshold=threshold,
                fix_size=fix_size,
                output_dir=output_dir,
                display=display
            )
            
            if image_result is not None:
                image_result['media_path'] = str(media_path)
                results.append(image_result)
    
    return results


if __name__ == '__main__':
    # 命令行接口（保留原有功能）
    assert sys.version_info >= (3, 6), sys.version_info
    args = parse_args()

    # 使用命令行参数调用主函数
    results = process_media(
        args.inputs,
        threshold=args.threshold,
        fix_size=not args.variable_size,
        output_dir=args.output_dir,
        video_blur_ratio=args.video_blur_ratio,
        sample_rate=args.sample_rate,
        motion_method=args.motion_method,
        motion_skip_frames=args.motion_skip_frames,
        verbose=args.verbose,
        display=args.display
    )
    
    # 输出结果摘要
    logging.info(f'处理完成，共处理 {len(results)} 个文件')