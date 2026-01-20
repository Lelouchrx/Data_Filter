import cv2
import numpy as np


def estimate_motion(prev_frame, curr_frame, method='lk', resize_to=(320, 180), 
                    p0=None, frame_idx=0, reinit_interval=10):
    """Estimate camera motion using optical flow.
    
    Args:
        prev_frame: Previous frame
        curr_frame: Current frame
        method: 'farneback' or 'lk'
        resize_to: Resize frames to this resolution
        p0: Previous feature points
        frame_idx: Current frame index
        reinit_interval: Reinitialize feature detection every N frames
    """
    if resize_to:
        prev_frame = cv2.resize(prev_frame, resize_to, interpolation=cv2.INTER_LINEAR)
        curr_frame = cv2.resize(curr_frame, resize_to, interpolation=cv2.INTER_LINEAR)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    if method == 'farneback':
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx = np.mean(flow[..., 0])
        dy = np.mean(flow[..., 1])
        return dx, dy, None
    elif method == 'lk':
        # 每N帧重新检测特征点，中间帧只做LK跟踪
        if frame_idx % reinit_interval == 0 or p0 is None or (p0 is not None and len(p0) < 5):
            p0 = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=20,
                qualityLevel=0.1,
                minDistance=10,
                blockSize=7
            )

        if p0 is None or len(p0) == 0:
            return 0.0, 0.0, None

        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)

        if st is None:
            return 0.0, 0.0, None

        good_mask = st.flatten() == 1
        if np.sum(good_mask) == 0:
            return 0.0, 0.0, None

        good_new = p1[good_mask]
        good_old = p0[good_mask]
        
        good_new = good_new.reshape(-1, 2)
        good_old = good_old.reshape(-1, 2)

        dx = np.median(good_new[:, 0] - good_old[:, 0])
        dy = np.median(good_new[:, 1] - good_old[:, 1])
        
        # 返回更新后的特征点用于下一帧
        p0_next = p1[good_mask].reshape(-1, 2) if len(good_new) >= 5 else None
        return dx, dy, p0_next
    else:
        raise ValueError(f"Unknown method: {method}")


def moving_average(data, window_size=5):
    """Calculate moving average."""
    if len(data) < window_size:
        return np.array([np.mean(data)] * len(data))
    
    data = np.asarray(data, dtype=np.float64)
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


def calc_jitter(dx_seq, dy_seq, window_size=5):
    """Calculate high-frequency jitter."""
    if len(dx_seq) < 2:
        return 0.0

    # 转换为numpy数组一次，避免多次转换
    dx_seq = np.asarray(dx_seq, dtype=np.float64)
    dy_seq = np.asarray(dy_seq, dtype=np.float64)

    dx_ma = moving_average(dx_seq, window_size)
    dy_ma = moving_average(dy_seq, window_size)

    min_len = min(len(dx_seq), len(dx_ma))
    dx_hp = dx_seq[:min_len] - dx_ma
    dy_hp = dy_seq[:min_len] - dy_ma

    return np.std(dx_hp) + np.std(dy_hp)


def fix_image_size(image, target_size=(500, 500)):
    """Resize image"""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def estimate_blur(image, threshold=100.0, resize_to=(320, 180)):
    """Estimate blur using Laplacian variance."""
    if resize_to:
        image = cv2.resize(image, resize_to, interpolation=cv2.INTER_LINEAR)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    blur_map = cv2.convertScaleAbs(laplacian)
    blurry = variance < threshold

    return blur_map, variance, blurry

def pretty_blur_map(blur_map, sigma=5, min_abs=0.5):
    abs_image = np.abs(blur_map).astype(np.float32)
    abs_image[abs_image < min_abs] = min_abs
    abs_image = np.log(abs_image)

    abs_image = cv2.blur(abs_image, (sigma, sigma))
    abs_image = cv2.medianBlur(abs_image, sigma)

    return abs_image

