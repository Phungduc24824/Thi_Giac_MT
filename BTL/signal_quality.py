"""Signal quality assessment utilities for realtime inputs (images or 1D signals).

Provides simple, dependency-light heuristics: blur (Laplacian), brightness,
contrast, SNR for 1D signals, and an EWMA tracker for smoothing scores.
"""

from typing import Tuple, Optional
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def compute_blur_score(frame: np.ndarray) -> float:
    """Return blur score normalized 0..1 (higher is sharper).
    Uses Laplacian variance when OpenCV available, otherwise gradient proxy.
    """
    if frame is None:
        return 0.0
    if frame.ndim == 3:
        gray = np.mean(frame, axis=2).astype(np.float32)
    else:
        gray = frame.astype(np.float32)
    if cv2 is not None:
        try:
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            var = float(lap.var())
        except Exception:
            gx, gy = np.gradient(gray)
            var = float((gx**2 + gy**2).var())
    else:
        gx, gy = np.gradient(gray)
        var = float((gx**2 + gy**2).var())
    # normalize with a soft mapping
    score = np.tanh(var / 100.0)
    return float(np.clip(score, 0.0, 1.0))


def compute_brightness_score(frame: np.ndarray) -> float:
    if frame is None:
        return 0.0
    if frame.ndim == 3:
        gray = np.mean(frame, axis=2).astype(np.float32)
    else:
        gray = frame.astype(np.float32)
    mean = float(np.mean(gray))
    # assume 0..255 range; map 30..220 to good band
    score = 1.0 - (abs(mean - 128.0) / 128.0)
    return float(np.clip(score, 0.0, 1.0))


def compute_contrast_score(frame: np.ndarray) -> float:
    if frame is None:
        return 0.0
    if frame.ndim == 3:
        gray = np.mean(frame, axis=2).astype(np.float32)
    else:
        gray = frame.astype(np.float32)
    std = float(np.std(gray))
    score = np.tanh(std / 64.0)
    return float(np.clip(score, 0.0, 1.0))


def assess_image_quality(frame: np.ndarray) -> Tuple[float, float]:
    """Return (score, confidence) for an image.
    score: 0..1 higher is better. confidence: 0..1 reliability of the score.
    """
    b = compute_blur_score(frame)
    br = compute_brightness_score(frame)
    c = compute_contrast_score(frame)
    # weights tuned for quick heuristics
    score = 0.5 * b + 0.25 * c + 0.25 * br
    # confidence based on agreement of metrics
    conf = 1.0 - (abs(b - c) * 0.5 + abs(b - br) * 0.5)
    conf = float(np.clip(conf, 0.2, 1.0))
    return float(np.clip(score, 0.0, 1.0)), conf


def compute_snr(signal: np.ndarray) -> Optional[float]:
    """Compute SNR in dB for 1D signal. Returns None if invalid."""
    if signal is None:
        return None
    sig = np.asarray(signal).astype(np.float32).ravel()
    if sig.size < 4:
        return None
    # detrend with simple mean
    mean = sig.mean()
    signal_power = np.mean((sig - mean) ** 2)
    # estimate noise as small-scale deviations using median filter
    try:
        from scipy.signal import medfilt
        deno = medfilt(sig, kernel_size=3)
    except Exception:
        deno = sig.copy()
        deno[1:-1] = (sig[:-2] + sig[1:-1] + sig[2:]) / 3.0
    noise = sig - deno
    noise_power = np.mean(noise ** 2) + 1e-9
    snr = 10.0 * np.log10(signal_power / noise_power + 1e-9)
    return float(snr)


def assess_signal_quality(signal: np.ndarray) -> Tuple[float, float]:
    """Return (score, confidence) for 1D signal.
    Maps SNR to 0..1 score. Confidence depends on length.
    """
    snr = compute_snr(signal)
    if snr is None:
        return 0.0, 0.2
    # map typical SNR ranges: <0 bad, 0-10 medium, >20 good
    score = (np.tanh((snr - 5.0) / 10.0) + 1.0) / 2.0
    conf = float(np.clip(min(1.0, len(signal) / 256.0), 0.2, 1.0))
    return float(np.clip(score, 0.0, 1.0)), conf


class RealtimeQualityTracker:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.ewma_score = None
        self.ewma_conf = None

    def update(self, score: float, conf: float) -> Tuple[float, float]:
        if self.ewma_score is None:
            self.ewma_score = score
            self.ewma_conf = conf
        else:
            self.ewma_score = self.alpha * score + (1.0 - self.alpha) * self.ewma_score
            self.ewma_conf = self.alpha * conf + (1.0 - self.alpha) * self.ewma_conf
        return float(self.ewma_score), float(self.ewma_conf)


if __name__ == '__main__':
    # quick local test example
    import sys
    if len(sys.argv) > 1 and cv2 is not None:
        path = sys.argv[1]
        img = cv2.imread(path)
        s, c = assess_image_quality(img)
        print('Image score:', s, 'conf:', c)
    else:
        print('Usage: python signal_quality.py <image_path>')
