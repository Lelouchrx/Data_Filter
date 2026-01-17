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


def find_media(media_paths, img_extensions=['.jpg', '.png', '.jpeg'], video_extensions=['.mp4', '.avi', '.mov', '.mkv']):
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


def process_video(video_path, threshold=100.0, fix_size=True, blur_ratio_threshold=0.3, output_dir=None, sample_rate=1.0,
                 motion_method='lk', motion_skip_frames=3):
    """Process video with optimized motion detection.

    Args:
        motion_method: 'farneback' or 'lk' (faster)
        motion_skip_frames: Skip frames for motion estimation (e.g., 3 = every 3rd frame)
    """
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

    # Create output directory
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
    
    if jitter_video > 6.0:  # Further increased jitter threshold
        keep = False
        rejection_reasons.append(f"抖动过大({jitter_video:.3f} > 6.0)")
    
    if valid_ratio < 0.7:
        keep = False
        rejection_reasons.append(f"有效率过低({valid_ratio:.3f} < 0.7)")
    
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


if __name__ == '__main__':
    assert sys.version_info >= (3, 6), sys.version_info
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level)

    fix_size = not args.variable_size
    logging.info(f'fix_size: {fix_size}')

    for media_path in find_media(args.inputs):
        # Check if it's a video file
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
        is_video = media_path.suffix in video_extensions

        if is_video:
            # Process video
            logging.info(f'processing video {media_path}')
            video_result, unusable = process_video(media_path, threshold=args.threshold,
                                                 fix_size=fix_size,
                                                 blur_ratio_threshold=args.video_blur_ratio,
                                                 output_dir=args.output_dir,
                                                 sample_rate=args.sample_rate,
                                                 motion_method=args.motion_method,
                                                 motion_skip_frames=args.motion_skip_frames)

            if video_result is None:
                logging.warning(f'failed to process video {media_path}; skipping!')
                continue
        else:
            # Process image
            image = cv2.imread(str(media_path))
            if image is None:
                logging.warning(f'warning! failed to read image from {media_path}; skipping!')
                continue

            logging.info(f'processing image {media_path}')

            if fix_size:
                image = fix_image_size(image)
            else:
                logging.warning('not normalizing image size for consistent scoring!')

            blur_map, score, blurry = estimate_blur(image, threshold=args.threshold)

            logging.info(f'image_path: {media_path} score: {score:.1f} blurry: {blurry}')

            # Save image with overlay
            if args.output_dir:
                img_dir = pathlib.Path(args.output_dir) / media_path.stem
                img_dir.mkdir(parents=True, exist_ok=True)

                status = "BLUR" if blurry else "CLEAR"
                display_image = image.copy()
                cv2.putText(display_image, f"{status} {score:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                           (0, 255, 0) if not blurry else (0, 0, 255), 2)

                img_filename = f"{media_path.stem}_{status}_{score:.1f}.jpg"
                img_path = img_dir / img_filename
                cv2.imwrite(str(img_path), display_image)

            if args.display:
                cv2.imshow('input', image)
                cv2.imshow('result', pretty_blur_map(blur_map))

                if cv2.waitKey(0) == ord('q'):
                    logging.info('exiting...')
                    exit()