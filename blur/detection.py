import cv2
import numpy as np


def optical_flow_method(prev_gray, curr_gray, scale=0.5, max_corners=60, quality_level=0.02, min_distance=25, block_size=3):
    """Compute optical flow between two frames using Shi-Tomasi corners and Lucas-Kanade tracking."""
    # Downscale resolution for efficiency
    prev_small = cv2.resize(prev_gray, None, fx=scale, fy=scale)
    curr_small = cv2.resize(curr_gray, None, fx=scale, fy=scale)

    prev_pts = cv2.goodFeaturesToTrack(prev_small, maxCorners=max_corners, qualityLevel=quality_level,
                                       minDistance=min_distance, blockSize=block_size)
    if prev_pts is None or len(prev_pts) < 4:
        return None, None, None

    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_small, curr_small, prev_pts, None)
    if curr_pts is None or status is None or prev_pts.shape != curr_pts.shape:
        return None, None, None

    status_mask = status.ravel() == 1
    if status_mask.sum() < 4:
        return None, None, None

    prev_pts = prev_pts[status_mask]
    curr_pts = curr_pts[status_mask]

    if prev_pts.ndim == 3:
        prev_pts = prev_pts.reshape(-1, 2)
        curr_pts = curr_pts.reshape(-1, 2)

    # Scale coordinates back to original resolution
    prev_pts /= scale
    curr_pts /= scale

    return prev_pts, curr_pts, None


def estimate_motion(prev_frame, curr_frame, resize_to=None, p0=None, frame_idx=0, reinit_interval=10, ransac_interval=5, max_corners=60, quality_level=0.02, min_distance=25, block_size=3):
    """Estimate motion between frames using optical flow and RANSAC for robustness."""
    if resize_to is not None:
        prev_frame = cv2.resize(prev_frame, resize_to, interpolation=cv2.INTER_NEAREST)
        curr_frame = cv2.resize(curr_frame, resize_to, interpolation=cv2.INTER_NEAREST)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if prev_frame.ndim == 3 else prev_frame
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if curr_frame.ndim == 3 else curr_frame

    need_reinit = p0 is None or len(p0) < 10 or frame_idx % reinit_interval == 0

    if need_reinit:
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance, blockSize=block_size)
        if p0 is not None:
            p0 = p0.astype(np.float32)

    if p0 is None or len(p0) < 3:
        return 0.0, 0.0, 0.0, None, 0.0

    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, winSize=(15, 15), maxLevel=2,
                                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    if st is None:
        return 0.0, 0.0, 0.0, None, 0.0

    good_mask = st.ravel() == 1
    good_count = np.sum(good_mask)
    
    if good_count < 3:
        return 0.0, 0.0, 0.0, None, 0.0

    good_old = p0[good_mask]
    good_new = p1[good_mask]

    # RANSAC runs infrequently for performance
    if frame_idx % ransac_interval == 0:
        try:
            M, mask = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC,
                                                  ransacReprojThreshold=3.0, confidence=0.99, maxIters=200)
        except:
            try:
                M = cv2.estimateRigidTransform(good_old, good_new, fullAffine=False)
                mask = None
            except:
                return 0.0, 0.0, 0.0, None, 0.0

        if M is None:
            return 0.0, 0.0, 0.0, None, 0.0

        if mask is not None:
            inlier_mask = mask.ravel() == 1
            inlier_count = np.sum(inlier_mask)
            inlier_ratio = inlier_count / len(mask) if len(mask) > 0 else 0.0
        else:
            inlier_mask = np.ones(len(good_old), dtype=bool)
            inlier_count = len(good_old)
            inlier_ratio = 1.0

        dx = float(M[0, 2])
        dy = float(M[1, 2])
        theta = float(np.arctan2(M[1, 0], M[0, 0]))
    else:
        # High-frequency optical flow: translation only
        if good_old.ndim == 3:
            good_old = good_old.reshape(-1, 2)
            good_new = good_new.reshape(-1, 2)
        dx = np.mean(good_new[:, 0] - good_old[:, 0])
        dy = np.mean(good_new[:, 1] - good_old[:, 1])
        theta = 0.0
        inlier_count = len(good_old)
        inlier_ratio = 1.0
        inlier_mask = np.ones(len(good_old), dtype=bool)

    p0_next = good_new[inlier_mask] if inlier_count >= 5 else None
    return dx, dy, theta, p0_next, inlier_ratio


def moving_average(data, window_size):
    """Compute moving average with specified window size."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def calc_jitter(dx_seq, dy_seq, theta_seq=None, inlier_ratio_seq=None,
                alpha=1.0, beta=1.0, min_inlier_ratio=0.6,
                hf_high=40.0, hf_low=10.0, hf_threshold=50.0, disp_px=7.0,
                disp_high=0.10, disp_low=0.03,
                return_frame_results=False, return_detailed_stats=False):
    """Analyze jitter in motion sequences using high-frequency energy and displacement thresholds."""
    
    dx_seq = np.asarray(dx_seq, dtype=np.float32)
    dy_seq = np.asarray(dy_seq, dtype=np.float32)

    if len(dx_seq) < 3:
        empty_result = {'jitter_score': 0.0, 'is_shake': False, 'shake_percentage': 0.0,
                       'total_frames': len(dx_seq), 'shake_frames': 0, 'conclusion': '视频稳定，未检测到抖动',
                       'translations': {'max': 0.0, 'mean': 0.0}, 'rotations': {'max': 0.0, 'mean': 0.0},
                       'hf_energy': {'mean': 0.0, 'max': 0.0}, 'features': {}}
        if return_frame_results and return_detailed_stats:
            return 0.0, [False] * len(dx_seq), empty_result
        elif return_frame_results:
            return 0.0, [False] * len(dx_seq)
        elif return_detailed_stats:
            return 0.0, empty_result
        return 0.0

    valid_mask = np.ones(len(dx_seq), dtype=bool)
    if inlier_ratio_seq is not None:
        valid_mask = np.asarray(inlier_ratio_seq, dtype=np.float32) >= min_inlier_ratio

    dx_valid = dx_seq[valid_mask]
    dy_valid = dy_seq[valid_mask]

    if len(dx_valid) < 3:
        empty_result = {'jitter_score': 0.0, 'is_shake': False, 'shake_percentage': 0.0,
                       'total_frames': len(dx_seq), 'shake_frames': 0, 'conclusion': '视频稳定，未检测到抖动',
                       'translations': {'max': 0.0, 'mean': 0.0}, 'rotations': {'max': 0.0, 'mean': 0.0},
                       'hf_energy': {'mean': 0.0, 'max': 0.0}, 'features': {}}
        if return_frame_results and return_detailed_stats:
            return 0.0, [False] * len(dx_seq), empty_result
        elif return_frame_results:
            return 0.0, [False] * len(dx_seq)
        elif return_detailed_stats:
            return 0.0, empty_result
        return 0.0

    window_size = min(5, len(dx_valid) // 3)
    if window_size < 3:
        window_size = 3

    dx_low = moving_average(dx_valid, window_size)
    dy_low = moving_average(dy_valid, window_size)

    pad_len = len(dx_valid) - len(dx_low)
    if pad_len > 0:
        pad_width = ((pad_len // 2, pad_len - pad_len // 2),)
        dx_low = np.pad(dx_low, pad_width, mode='edge')
        dy_low = np.pad(dy_low, pad_width, mode='edge')

    hf_magnitude = np.sqrt((dx_valid[:len(dx_low)] - dx_low)**2 + (dy_valid[:len(dy_low)] - dy_low)**2)
    hf_energy_values = hf_magnitude**2
    hf_energy_ratio = np.mean(hf_energy_values > hf_threshold) * 100 if len(hf_energy_values) > 0 else 0.0

    displacement_mask = (np.abs(dx_valid) > disp_px) | (np.abs(dy_valid) > disp_px)
    displacement_percentage = np.mean(displacement_mask) * 100 if len(dx_valid) > 0 else 0.0

    # Calculate per-frame high-frequency energy
    hf_frame_mask = hf_energy_values > hf_threshold
    hf_frame_percentage = np.mean(hf_frame_mask) * 100 if len(hf_energy_values) > 0 else 0.0

    # Per-frame shake detection (local conditions)
    frame_shake_mask = np.zeros(len(dx_valid), dtype=bool)
    if hf_frame_percentage >= hf_low:
        frame_shake_mask = (displacement_mask & hf_frame_mask)
    # Shake detection logic: HF energy + displacement thresholds
    if hf_energy_ratio < hf_low:
        is_shake = False
        frame_shake_mask = np.zeros(len(dx_valid), dtype=bool)
    elif displacement_percentage < disp_low * 100:
        is_shake = False
        frame_shake_mask = np.zeros(len(dx_valid), dtype=bool)
    elif displacement_percentage > disp_high * 100 and hf_energy_ratio >= hf_low:
        is_shake = True
        frame_shake_mask = np.ones(len(dx_valid), dtype=bool)
    elif displacement_percentage >= disp_low * 100 and hf_energy_ratio >= hf_high:
        is_shake = True
        frame_shake_mask = np.ones(len(dx_valid), dtype=bool)
    else:
        is_shake = False
        frame_shake_mask = np.zeros(len(dx_valid), dtype=bool)

    # Map frame indices from valid to original
    valid_indices = np.where(valid_mask)[0]
    frame_shake_results = [False] * len(dx_seq)
    for i, valid_idx in enumerate(valid_indices):
        if i < len(frame_shake_mask):
            frame_shake_results[valid_idx] = frame_shake_mask[i]

    if return_detailed_stats:
        stats_dict = {
            'displacement_percentage': float(displacement_percentage),
            'hf_energy_ratio': float(hf_energy_ratio),
            'hf_high': hf_high,
            'hf_low': hf_low,
            'disp_high': disp_high * 100,
            'disp_low': disp_low * 100,
            'is_shake': bool(is_shake),
            'total_frames': len(dx_valid),
            'frame_shake_results': frame_shake_results
        }

        if return_frame_results:
            return displacement_percentage, frame_shake_results, stats_dict
        else:
            return displacement_percentage, stats_dict

    if return_frame_results:
        return displacement_percentage, frame_shake_results

    return displacement_percentage


def fix_image_size(image, target_size=(500, 500)):
    """Resize image to target dimensions."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def estimate_blur(image, threshold=100.0, resize_to=None):
    """Estimate image blur using Laplacian variance."""
    if resize_to:
        image = cv2.resize(image, resize_to, interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    return cv2.convertScaleAbs(laplacian), variance, variance < threshold


def judge_exposure(img, resize_to=None, mean_threshold=220, std_threshold=40):
    """Check if image is overexposed using Lab color space luminance."""
    if resize_to:
        img = cv2.resize(img, resize_to, interpolation=cv2.INTER_LINEAR)

    L = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, 0]
    mean_luminance = L.mean()
    std_luminance = L.std()
    is_overexposed = (mean_luminance > mean_threshold) and (std_luminance < std_threshold)

    return is_overexposed, {
        "mean_luminance": float(mean_luminance),
        "std_luminance": float(std_luminance),
        "is_overexposed": bool(is_overexposed)
    }


def pretty_blur_map(blur_map, sigma=5, min_abs=0.5):
    """Create visual blur map for display."""
    abs_image = np.abs(blur_map).astype(np.float32)
    abs_image[abs_image < min_abs] = min_abs
    abs_image = np.log(abs_image)
    abs_image = cv2.blur(abs_image, (sigma, sigma))
    return cv2.medianBlur(abs_image, sigma)


def apply_gamma_correction(image, gamma=0.7):
    """Apply gamma correction to adjust image brightness."""
    inv_gamma = 1.0 / gamma
    table = np.power(np.arange(256, dtype=np.float32) / 255.0, inv_gamma) * 255
    return cv2.LUT(image, table.astype(np.uint8))
