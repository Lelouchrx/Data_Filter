import cv2
import numpy as np


def estimate_motion(prev_frame, curr_frame, method='farneback', resize_to=(320, 180)):
    """Estimate camera motion using optical flow.

    Args:
        prev_frame: Previous frame
        curr_frame: Current frame
        method: 'farneback' (dense) or 'lk' (sparse, faster)
        resize_to: Resize frames to this resolution before processing
    """
    # 统一降分辨率到指定尺寸
    if resize_to:
        prev_frame = cv2.resize(prev_frame, resize_to)
        curr_frame = cv2.resize(curr_frame, resize_to)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    if method == 'farneback':
        # Dense optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx = np.mean(flow[..., 0])
        dy = np.mean(flow[..., 1])
    elif method == 'lk':
        # Sparse optical flow
        p0 = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=20,  # Reduced to 20 for maximum speed
            qualityLevel=0.1,  # Increased for fewer but better points
            minDistance=10,  # Increased for speed
            blockSize=7  # Smaller block size for speed
        )

        if p0 is None or len(p0) == 0:
            return 0.0, 0.0

        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)

        # Filter valid points
        if st is None:
            return 0.0, 0.0

        good_new = p1[st.flatten() == 1]
        good_old = p0[st.flatten() == 1]

        if len(good_new) == 0 or len(good_old) == 0:
            return 0.0, 0.0

        # Calculate median motion (ensure proper shape)
        good_new = good_new.reshape(-1, 2)
        good_old = good_old.reshape(-1, 2)

        dx = np.median(good_new[:, 0] - good_old[:, 0])
        dy = np.median(good_new[:, 1] - good_old[:, 1])
    else:
        raise ValueError(f"Unknown method: {method}")

    return dx, dy


def moving_average(data, window_size=5):
    """Calculate moving average."""
    if len(data) < window_size:
        return np.array([np.mean(data)] * len(data))
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def calc_jitter(dx_seq, dy_seq, window_size=5):
    """Calculate high-frequency jitter."""
    if len(dx_seq) < 2:
        return 0.0

    dx_ma = moving_average(dx_seq, window_size)
    dy_ma = moving_average(dy_seq, window_size)

    # Ensure same length
    min_len = min(len(dx_seq), len(dx_ma), len(dy_seq), len(dy_ma))
    dx_hp = np.array(dx_seq[:min_len]) - dx_ma[:min_len]
    dy_hp = np.array(dy_seq[:min_len]) - dy_ma[:min_len]

    return np.std(dx_hp) + np.std(dy_hp)


def fix_image_size(image, target_size=(500, 500)):
    """Resize image"""
    return cv2.resize(image, target_size)


def estimate_blur(image, threshold=100.0, resize_to=(320, 180)):
    """Estimate blur using Laplacian variance."""
    # 统一降分辨率到指定尺寸
    if resize_to:
        image = cv2.resize(image, resize_to)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    blur_map = cv2.convertScaleAbs(laplacian)
    blurry = variance < threshold

    return blur_map, variance, blurry


def estimate_saturation(image, low_threshold=15, high_threshold=240, resize_to=(320, 180)):
    """Estimate saturation ratio - proportion of pixels that are under/over exposed.
    使用改进的方法检测过曝光和欠曝光。

    Args:
        image: Input image (BGR or grayscale)
        low_threshold: Pixels below this value are considered underexposed (default: 15)
        high_threshold: Pixels above this value are considered overexposed (default: 240)
        resize_to: Resize image to this resolution before processing (default: (320, 180))

    Returns:
        saturation_ratio: Ratio of saturated pixels (0.0 to 1.0)
        under_ratio: Ratio of underexposed pixels
        over_ratio: Ratio of overexposed pixels
    """
    # 统一降分辨率到指定尺寸（与模糊检测保持一致）
    if resize_to:
        image = cv2.resize(image, resize_to)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    total_pixels = gray.size
    
    # 计算平均亮度
    mean_brightness = np.mean(gray)
    
    # 方法1: 统计极端值像素（更严格的阈值）
    # 欠曝光：非常暗的像素（接近0）
    under_pixels_strict = np.sum(gray < low_threshold)
    under_ratio_strict = under_pixels_strict / total_pixels
    
    # 过曝光：非常亮的像素（接近255）
    over_pixels_strict = np.sum(gray > high_threshold)
    over_ratio_strict = over_pixels_strict / total_pixels
    
    # 方法2: 基于直方图分析
    # 计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten()
    hist_normalized = hist / total_pixels
    
    # 统计暗部（0-30）和亮部（225-255）的像素比例
    dark_region = np.sum(hist_normalized[0:31])  # 0-30
    bright_region = np.sum(hist_normalized[225:256])  # 225-255
    
    # 方法3: 基于平均亮度的判断
    # 如果平均亮度太低，可能是欠曝光
    # 如果平均亮度太高，可能是过曝光
    brightness_based_under = 0.0
    brightness_based_over = 0.0
    
    if mean_brightness < 50:  # 平均亮度很低
        # 计算有多少像素低于平均亮度的1.5倍
        brightness_based_under = np.sum(gray < (mean_brightness * 1.5)) / total_pixels
    elif mean_brightness > 200:  # 平均亮度很高
        # 计算有多少像素高于平均亮度的0.8倍
        brightness_based_over = np.sum(gray > (mean_brightness * 0.8)) / total_pixels
    
    # 综合多种方法的结果
    # 欠曝光：取严格阈值和暗部区域的最大值
    under_ratio = max(under_ratio_strict, dark_region * 0.5, brightness_based_under)
    
    # 过曝光：取严格阈值和亮部区域的最大值
    over_ratio = max(over_ratio_strict, bright_region * 0.5, brightness_based_over)
    
    # 如果平均亮度极端，增加相应的曝光问题权重
    if mean_brightness < 30:
        under_ratio = max(under_ratio, 0.15)  # 强制认为有欠曝光问题
    if mean_brightness > 220:
        over_ratio = max(over_ratio, 0.15)  # 强制认为有过曝光问题
    
    # 饱和像素比例（欠曝光 + 过曝光）
    saturation_ratio = under_ratio + over_ratio

    return saturation_ratio, under_ratio, over_ratio


def pretty_blur_map(blur_map, sigma=5, min_abs=0.5):
    abs_image = np.abs(blur_map).astype(np.float32)
    abs_image[abs_image < min_abs] = min_abs
    abs_image = np.log(abs_image)

    abs_image = cv2.blur(abs_image, (sigma, sigma))
    abs_image = cv2.medianBlur(abs_image, sigma)

    return abs_image

