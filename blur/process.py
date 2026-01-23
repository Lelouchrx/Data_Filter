import sys
import argparse
import logging
import pathlib
import time
import json
import yaml

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .detection import estimate_blur

def create_jitter_analysis_plot(jitter_stats, dx_seq, dy_seq, theta_seq=None, exposure_scores=None,
                               video_name="Video", save_path=None):
    """
    Create a comprehensive jitter analysis plot showing various motion metrics over time.

    Args:
        jitter_stats: Detailed jitter statistics from calc_jitter
        dx_seq: X-axis displacement sequence
        dy_seq: Y-axis displacement sequence
        theta_seq: Rotation angle sequence (optional)
        exposure_scores: Exposure scores sequence (optional)
        video_name: Name of the video for plot title
        save_path: Path to save the plot (optional)
    """
    try:
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'Jitter Analysis: {video_name}', fontsize=16, fontweight='bold')

        gs = GridSpec(4, 2, figure=fig, height_ratios=[2, 2, 2, 1])

        # Convert sequences to numpy arrays for easier processing
        dx_seq = np.array(dx_seq)
        dy_seq = np.array(dy_seq)
        frames = np.arange(len(dx_seq))

        # 1. Displacement over time (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(frames, dx_seq, 'b-', alpha=0.7, label='X displacement', linewidth=1)
        ax1.plot(frames, dy_seq, 'r-', alpha=0.7, label='Y displacement', linewidth=1)
        ax1.set_title('Frame Displacement Over Time', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Displacement (pixels)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        # 2. Motion magnitude and direction (top-right)
        ax2 = fig.add_subplot(gs[0, 1])
        motion_magnitude = np.sqrt(dx_seq**2 + dy_seq**2)
        motion_angle = np.arctan2(dy_seq, dx_seq)

        ax2.plot(frames, motion_magnitude, 'g-', linewidth=1.5, label='Motion Magnitude')
        ax2.set_title('Motion Magnitude Over Time', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Motion Magnitude (pixels)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 3. Rotation analysis (middle-left)
        ax3 = fig.add_subplot(gs[1, 0])
        if theta_seq is not None and len(theta_seq) > 0:
            theta_seq = np.array(theta_seq)
            # Convert to degrees for better readability
            theta_deg = np.degrees(theta_seq)
            ax3.plot(frames[:len(theta_deg)], theta_deg, 'purple', linewidth=1, label='Rotation Angle')
            ax3.set_title('Rotation Angle Over Time', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Frame Number')
            ax3.set_ylabel('Rotation Angle (degrees)')
        else:
            ax3.text(0.5, 0.5, 'Rotation data not available',
                    transform=ax3.transAxes, ha='center', va='center', fontsize=12)
            ax3.set_title('Rotation Analysis (N/A)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 4. High-frequency ratio analysis (middle-right)
        ax4 = fig.add_subplot(gs[1, 1])
        hf_energy_ratio = jitter_stats.get('hf_energy_ratio', 0.0)
        hf_high = CONFIG['hf_high']
        hf_low = CONFIG['hf_low']

        ax4.axhline(y=hf_low, color='orange', linestyle=':',
                   alpha=0.7, label=f'HF Low: {hf_low:.1f}%')
        ax4.axhline(y=hf_high, color='red', linestyle='--',
                   alpha=0.7, label=f'HF High: {hf_high:.1f}%')
        ax4.axhline(y=hf_energy_ratio, color='b', linestyle='-', alpha=0.7,
                   label=f'HF Ratio: {hf_energy_ratio:.1f}%')
        ax4.fill_between([0, 1], [hf_energy_ratio, hf_energy_ratio],
                       alpha=0.3, color='blue', label='HF Energy Ratio')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, max(hf_energy_ratio * 1.2, hf_high * 1.2, 100))
        ax4.set_title('High-Frequency Energy Analysis', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Analysis Summary')
        ax4.set_ylabel('HF Energy Mean')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        # 5. Exposure scores (bottom-left)
        ax5 = fig.add_subplot(gs[2, 0])
        if exposure_scores is not None and len(exposure_scores) > 0:
            exposure_scores = np.array(exposure_scores)
            ax5.plot(frames[:len(exposure_scores)], exposure_scores, 'orange', linewidth=1, label='Exposure Score')
            ax5.axhline(y=CONFIG['exp_ratio'], color='r', linestyle='--', alpha=0.7, label=f'Threshold: {CONFIG["exp_ratio"]}')
            ax5.fill_between(frames[:len(exposure_scores)],
                           np.where(exposure_scores > 0.15, exposure_scores, 0),
                           0, alpha=0.3, color='red', label='Bad Exposure')
            ax5.set_title('Exposure Score Over Time', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Frame Number')
            ax5.set_ylabel('Exposure Score')
        else:
            ax5.text(0.5, 0.5, 'Exposure data not available',
                    transform=ax5.transAxes, ha='center', va='center', fontsize=12)
            ax5.set_title('Exposure Analysis (N/A)', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend()

        # 6. Summary statistics (bottom-right)
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')

        displacement_percentage = jitter_stats.get('displacement_percentage', 0.0)
        hf_energy_mean = jitter_stats.get('hf_energy_mean', 0.0)
        is_shake = jitter_stats.get('is_shake', False)
        total_frames = jitter_stats.get('total_frames', 0)

        summary_text = f"""
        Jitter Analysis Summary:

        Displacement > 15px: {displacement_percentage:.1f}%
        High-Frequency Energy Mean: {hf_energy_mean:.1f}
        HF Energy Threshold: 20.0
        Is Shaking: {'Yes' if is_shake else 'No'}

        Detection Logic:
        • Displacement > 10% AND
        • HF Energy Mean > 20.0

        Total Frames: {total_frames}
        """

        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

        # 7. Conclusion (bottom full width)
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        is_shake = jitter_stats.get('is_shake', False)
        displacement_percentage = jitter_stats.get('displacement_percentage', 0.0)
        hf_energy_mean = jitter_stats.get('hf_energy_mean', 0.0)

        if is_shake:
            conclusion = f"抖动检测: 位移比例 {displacement_percentage:.1f}%, 高频能量平均值 {hf_energy_mean:.1f}"
            color = 'red'
        else:
            conclusion = f"视频稳定: 位移比例 {displacement_percentage:.1f}%, 高频能量平均值 {hf_energy_mean:.1f}"
            color = 'green'

        ax7.text(0.5, 0.5, conclusion,
                transform=ax7.transAxes, ha='center', va='center',
                fontsize=14, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Jitter analysis plot saved to: {save_path}")
        else:
            plt.show()

        plt.close(fig)

    except Exception as e:
        print(f"Error creating jitter analysis plot: {e}")
        import traceback
        traceback.print_exc()
from .detection import fix_image_size
from .detection import pretty_blur_map
from .detection import estimate_motion
from .detection import judge_exposure
from .detection import optical_flow_method
from .detection import calc_jitter

# Load configuration from config.yaml
def load_config():
    config_path = pathlib.Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Convert lists to tuples where needed
    config['target_size'] = tuple(config['target_size'])

    return config

CONFIG = load_config()


def parse_args():
    parser = argparse.ArgumentParser(description='run blur detection on images and videos')
    parser.add_argument('-i', '--inputs', type=str, nargs='+', required=True, help='directory of images/videos')
    parser.add_argument('-o', '--output-dir', type=str, default=CONFIG['output_dir'], help='directory to save frame images')

    parser.add_argument('-t', '--threshold', type=float, default=CONFIG['blur_thresh'], help='blurry threshold')
    parser.add_argument('-f', '--variable-size', action='store_true', help='fix the image size')

    parser.add_argument('-v', '--verbose', action='store_true', help='set logging level to debug')
    parser.add_argument('-d', '--display', action='store_true', help='display images')
    parser.add_argument('-r', '--record', action='store_true', help='save frame images and output JSON log')
    parser.add_argument('--video-blur-ratio', type=float, default=CONFIG['blur_ratio'], help='ratio of blurry frames to mark video as unusable')
    parser.add_argument('--jitter-plot', action='store_true', help='show jitter analysis plot')
    parser.add_argument('--sample-rate', type=float, default=CONFIG['sample'], help='sample rate per second (frames per second)')
    parser.add_argument('--motion-method', type=str, default=CONFIG['motion'], choices=['farneback', 'lk'], help='motion estimation method (lk is faster)')

    return parser.parse_args()


def find_media(media_paths, img_extensions=None, video_extensions=None):
    img_extensions = img_extensions if img_extensions is not None else CONFIG['img_extensions']
    video_extensions = video_extensions if video_extensions is not None else CONFIG['video_extensions']
    
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
                 motion_method=None, record=False,
                 enable_blur_test=None, enable_jitter_test=None, enable_exposure_test=None, show_jitter_plot=False):
    """Process video with optimized motion detection.

    Args:
        video_path: 视频文件路径
        threshold: 模糊阈值
        fix_size: 是否固定尺寸
        blur_ratio_threshold: 视频模糊帧比例阈值
        output_dir: 输出目录
        sample_rate: 采样率
        motion_method: 运动估计方法 'farneback' 或 'lk'
        record: 是否保存frame输出和JSON日志
        enable_blur_test: 是否启用模糊检测（默认从config读取）
        enable_jitter_test: 是否启用抖动检测（默认从config读取）
        enable_exposure_test: 是否启用曝光检测（默认从config读取）
        show_jitter_plot: 是否显示抖动分析图表
    """

    threshold = threshold if threshold is not None else CONFIG['blur_thresh']
    fix_size = fix_size if fix_size is not None else CONFIG['fix_size']
    blur_ratio_threshold = blur_ratio_threshold if blur_ratio_threshold is not None else CONFIG['blur_ratio']
    output_dir = output_dir if output_dir is not None else CONFIG['output_dir']
    sample_rate = sample_rate if sample_rate is not None else CONFIG['sample']
    motion_method = motion_method if motion_method is not None else CONFIG['motion']
    enable_blur_test = enable_blur_test if enable_blur_test is not None else CONFIG['blur_test']
    enable_jitter_test = enable_jitter_test if enable_jitter_test is not None else CONFIG['jitter_test']
    enable_exposure_test = enable_exposure_test if enable_exposure_test is not None else CONFIG['exposure_test']
    
    start_time = time.time()
    video_name = video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        end_time = time.time()
        logging.warning(f'failed to open video {video_path}')
        return None, True

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count == 0:
        cap.release()
        return None, True

    if record and output_dir:
        frames_dir = pathlib.Path(output_dir) / video_name
        frames_dir.mkdir(parents=True, exist_ok=True)
    else:
        frames_dir = None

    if sample_rate <= fps:
        frame_interval = int(sample_rate)
    else:
        frame_interval = max(1, int(fps / sample_rate))
    
    blurry_frames = 0
    total_samples = 0

    blur_scores = []
    valid_frames = []
    blur_frames = []  # 模糊帧索引列表

    dx_seq, dy_seq = [], []
    theta_seq = []
    inlier_ratio_seq = []
    shake_frames = []  # 抖动帧索引列表

    prev_frame = None
    prev_gray = None

    exposure_scores = []
    exposure_frames = []  # 曝光不良帧索引列表
    consecutive_bad_exposure = 0
    max_consecutive_bad_exposure = 0
    bad_exposure_count = 0

    # 初始化抖动检测变量
    p0 = None
    frame_idx = 0

    # 计算预期处理帧数用于进度条
    expected_frames = frame_count // frame_interval
    processed_frames = 0

    with tqdm(total=expected_frames, desc=f"处理 {video_name}", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 按采样间隔处理帧
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            original_frame = frame.copy()
            if fix_size:
                frame = fix_image_size(frame)

            # 灰度 & resize 只做一次，共享给所有检测
            gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, None, fx=0.5, fy=0.5)

            if enable_blur_test:
                blur_map, score, blurry = estimate_blur(frame, threshold=threshold)
                blur_scores.append(score)
                valid_frames.append(not blurry)
                if blurry:
                    blur_frames.append(frame_idx)
            else:
                blur_scores.append(100.0)
                valid_frames.append(True)
                blurry = False
                score = 100.0

            if enable_exposure_test:
                is_overexposed, exposure_stats = judge_exposure(original_frame,
                    mean_threshold=CONFIG['exp_mean'],
                    std_threshold=CONFIG['exp_std'])
                exposure_scores.append(exposure_stats["mean_luminance"])  # 保存亮度均值用于统计
                if is_overexposed:
                    consecutive_bad_exposure += 1
                    max_consecutive_bad_exposure = max(max_consecutive_bad_exposure, consecutive_bad_exposure)
                    bad_exposure_count += 1
                    exposure_frames.append(frame_idx)
                else:
                    consecutive_bad_exposure = 0
            else:
                exposure_scores.append(0.0)

            # 抖动检测 - 只用estimate_motion()
            if enable_jitter_test:
                dx, dy, theta, p0_next, inlier_ratio = estimate_motion(
                    prev_frame=prev_frame if frame_idx > 0 else original_frame,
                    curr_frame=original_frame,
                    p0=p0,
                    frame_idx=frame_idx,
                    ransac_interval=CONFIG['ransac_interval']
                )
                dx_seq.append(dx)
                dy_seq.append(dy)
                theta_seq.append(theta)
                inlier_ratio_seq.append(inlier_ratio)
                p0 = p0_next
                prev_frame = original_frame

            if record and frames_dir:
                display_frame = original_frame.copy()
                status = "BLUR" if blurry else "CLEAR"
                text = f"{status} {score:.1f}"
                cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if not blurry else (0, 0, 255), 2)
                frame_filename = f"{frame_idx:04d}_{status}_{score:.1f}.jpg"
                frame_path = frames_dir / frame_filename
                cv2.imwrite(str(frame_path), display_frame)

            if blurry:
                blurry_frames += 1
            total_samples += 1
            processed_frames += 1
            pbar.update(1)
            frame_idx += 1

    cap.release()

    if not blur_scores:
        return None, True

    # Video-level statistics
    blur_video = np.percentile(blur_scores, 10) if enable_blur_test else 100.0
    valid_ratio = sum(valid_frames) / len(valid_frames) if enable_blur_test else 1.0
    
    jitter_video = 0.0
    jitter_stats = None

    if enable_jitter_test:
        if len(dx_seq) < 3:
            displacement_percentage = 0.0
            hf_energy_value = 0.0
            is_shake = False
            shake_frames = []
            jitter_stats = {
                'displacement_percentage': 0.0,
                'hf_energy': 0.0,
                'hf_energy_threshold': CONFIG['hf_energy_threshold'],
                'is_shake': False,
                'total_frames': 0,
                'frame_shake_results': []
            }
        else:
            displacement_percentage, jitter_stats = calc_jitter(
                dx_seq, dy_seq, theta_seq, inlier_ratio_seq,
                hf_high=CONFIG['hf_high'],
                hf_low=CONFIG['hf_low'],
                hf_threshold=CONFIG['hf_threshold'],
                disp_px=CONFIG['disp_px'],
                disp_high=CONFIG['disp_high'],
                disp_low=CONFIG['disp_low'],
                return_detailed_stats=True
            )
            hf_energy_ratio = jitter_stats.get('hf_energy_ratio', 0.0)
            is_shake = jitter_stats.get('is_shake', False)
            # 获取逐帧抖动结果
            frame_shake_results = jitter_stats.get('frame_shake_results', [])
            shake_frames = [i for i, is_shake_frame in enumerate(frame_shake_results) if is_shake_frame]
    
    # 曝光检测
    if enable_exposure_test:
        total_frames = len(exposure_scores)
        bad_exposure_ratio = bad_exposure_count / total_frames if total_frames > 0 else 0.0
        exposure_bad = bad_exposure_ratio > CONFIG['exp_ratio']
    else:
        exposure_bad = False
        bad_exposure_ratio = 0.0
        exposure_frames = []
    
    # Decision rules
    keep = True
    rejection_reasons = []
    
    if enable_blur_test:
        if blur_video < threshold:
            keep = False
            rejection_reasons.append(f"模糊度过低({blur_video:.1f} < {threshold})")
    
    
    if enable_jitter_test:
        if is_shake:
            keep = False
            rejection_reasons.append(f"抖动检测({displacement_percentage:.1f}%位移, {hf_energy_ratio:.1f}%高频比例)")

    if enable_exposure_test:
        if exposure_bad:
            keep = False
            rejection_reasons.append(f"曝光不合格({bad_exposure_count}/{len(exposure_scores)}={bad_exposure_ratio:.1%} > 10%)")
    
    end_time = time.time()
    processing_time = end_time - start_time
    avg_exposure = np.mean(exposure_scores) if exposure_scores else 0.0
    
    
    jitter_info = f"{displacement_percentage:.1f}%/{hf_energy_ratio:.1f}%"
    exposure_info = f"{bad_exposure_ratio:.1%}" if enable_exposure_test else "N/A"
    logging.info(f"{video_name} | {video_width}x{video_height} | Blur: {blur_video:.1f} | Jitter: {jitter_info} | Exposure: {exposure_info} | Keep: {keep} | {processing_time:.1f}s")
    
    if rejection_reasons:
        logging.info(f"  拒绝原因: {', '.join(rejection_reasons)}")

    avg_exposure = np.mean(exposure_scores) if exposure_scores else 0.0

    # 显示抖动分析图表
    if show_jitter_plot and enable_jitter_test:
        try:
            # 总是保存图表，无论是否指定record参数
            plot_save_path = pathlib.Path(output_dir) / f"{video_name}_jitter_analysis.png"
            print(f"Generating jitter analysis plot and saving to: {plot_save_path}")

            create_jitter_analysis_plot(
                jitter_stats=jitter_stats,
                dx_seq=dx_seq,
                dy_seq=dy_seq,
                theta_seq=theta_seq,
                exposure_scores=exposure_scores,
                video_name=video_name,
                save_path=plot_save_path
            )

            print(f"✓ Jitter analysis plot saved successfully to: {plot_save_path}")

        except Exception as e:
            logging.warning(f"Failed to create jitter analysis plot: {e}")
            import traceback
            traceback.print_exc()
    
    if record:
        result_json = {
            'video_name': video_name,
            'blur_video': float(blur_video),
            'displacement_percentage': float(displacement_percentage),
            'hf_energy_mean': float(hf_energy_mean),
            'is_shake': bool(is_shake),
            'exposure_score': float(avg_exposure),
            'max_consecutive_bad_exposure': int(max_consecutive_bad_exposure),
            'keep': bool(keep),
            'total_samples': int(len(blur_scores)),
            'processing_time': float(processing_time)
        }
        print(f"RESULT_JSON:{json.dumps(result_json)}", flush=True)

    result = {
        'video_path': str(video_path),
        'blur_video': blur_video,
        'blur_ratio': 1.0 - (sum(valid_frames) / len(valid_frames)) if enable_blur_test else 0.0,
        'blur_frames': blur_frames,
        'displacement_percentage': displacement_percentage,
        'hf_energy_ratio': hf_energy_ratio,
        'is_shake': is_shake,
        'shake_frames': shake_frames,
        'exposure_score': avg_exposure,
        'exposure_ratio': bad_exposure_ratio,
        'exposure_frames': exposure_frames,
        'max_consecutive_bad_exposure': max_consecutive_bad_exposure,
        'keep': keep,
        'total_samples': len(blur_scores),
        'processing_time': processing_time
    }

    return result


def process_image(image_path, threshold=None, fix_size=None, output_dir=None, display=None, record=False):
    """处理单张图片
    
    Args:
        image_path: 图片文件路径
        threshold: 模糊阈值
        fix_size: 是否固定尺寸
        output_dir: 输出目录
        display: 是否显示图片
        record: 是否保存输出
    """
    # 使用传入参数或从config读取默认值
    threshold = threshold if threshold is not None else CONFIG['blur_thresh']
    fix_size = fix_size if fix_size is not None else CONFIG['fix_size']
    output_dir = output_dir if output_dir is not None else CONFIG['output_dir']
    display = display if display is not None else CONFIG['display']
    
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

    if record and output_dir:
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
        **kwargs: 可选参数
            - threshold: 模糊阈值
            - fix_size: 是否固定尺寸
            - output_dir: 输出目录
            - video_blur_ratio: 视频模糊帧比例阈值
            - sample_rate: 采样率
            - motion_method: 运动估计方法
            - verbose: 是否显示详细日志
            - display: 是否显示图片
            - record: 是否保存frame输出和JSON日志
            - img_extensions: 图片扩展名列表
            - video_extensions: 视频扩展名列表
            - enable_blur_test: 是否启用模糊检测
            - enable_jitter_test: 是否启用抖动检测
            - enable_exposure_test: 是否启用曝光检测
    """
    threshold = kwargs.get('threshold', CONFIG['blur_thresh'])
    fix_size = kwargs.get('fix_size', CONFIG['fix_size'])
    output_dir = kwargs.get('output_dir', CONFIG['output_dir'])
    video_blur_ratio = kwargs.get('video_blur_ratio', CONFIG['blur_ratio'])
    sample_rate = kwargs.get('sample_rate', CONFIG['sample'])
    motion_method = kwargs.get('motion_method', CONFIG['motion'])
    verbose = kwargs.get('verbose', CONFIG['verbose'])
    display = kwargs.get('display', CONFIG['display'])
    record = kwargs.get('record', False)
    img_extensions = kwargs.get('img_extensions', CONFIG['img_extensions'])
    video_extensions = kwargs.get('video_extensions', CONFIG['video_extensions'])
    enable_blur_test = kwargs.get('enable_blur_test', CONFIG['blur_test'])
    enable_jitter_test = kwargs.get('enable_jitter_test', CONFIG['jitter_test'])
    enable_exposure_test = kwargs.get('enable_exposure_test', CONFIG['exposure_test'])
    
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
            video_result = process_video(
                media_path,
                threshold=threshold,
                fix_size=fix_size,
                blur_ratio_threshold=video_blur_ratio,
                output_dir=output_dir,
                sample_rate=sample_rate,
                motion_method=motion_method,
                record=record,
                enable_blur_test=enable_blur_test,
                enable_jitter_test=enable_jitter_test,
                enable_exposure_test=enable_exposure_test,
                show_jitter_plot=kwargs.get('show_jitter_plot', False)
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
                display=display,
                record=record
            )
            
            if image_result is not None:
                image_result['media_path'] = str(media_path)
                results.append(image_result)
    
    return results


if __name__ == '__main__':
    assert sys.version_info >= (3, 6), sys.version_info
    args = parse_args()

    results = process_media(
            args.inputs,
            threshold=args.threshold,
            fix_size=not args.variable_size,
            output_dir=args.output_dir,
            video_blur_ratio=args.video_blur_ratio,
            sample_rate=args.sample_rate,
            motion_method=args.motion_method,
            verbose=args.verbose,
            display=args.display,
            record=args.record,
            show_jitter_plot=args.jitter_plot
        )

    