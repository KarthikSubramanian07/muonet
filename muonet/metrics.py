"""Reconstruction quality metrics: PSNR, SSIM, IoU, and material classification.

All metrics compare a reconstructed volume against ground truth on the shared
[0, 1] scattering-density scale. Predictions are normalized by a high
percentile (robust to a few hot voxels) before comparison, and the same policy
is applied to every method so the numbers are directly comparable. That policy
is stated openly because an honest benchmark lives or dies on it.
"""

from __future__ import annotations

import numpy as np

# Voxels above this normalized density are called high-Z for IoU / accuracy.
DEFAULT_HIGH_Z_THRESHOLD = 0.5
# Percentile used to normalize a raw reconstruction into [0, 1].
DEFAULT_NORM_PERCENTILE = 99.5


def normalize_volume(
    volume: np.ndarray, percentile: float = DEFAULT_NORM_PERCENTILE
) -> np.ndarray:
    """Scale a volume into [0, 1] by a high percentile, clipping the tail."""
    volume = np.asarray(volume, dtype=np.float64)
    volume = np.clip(volume, 0.0, None)
    scale = np.percentile(volume, percentile)
    if scale <= 0.0:
        return np.zeros_like(volume)
    return np.clip(volume / scale, 0.0, 1.0)


def _gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x**2) / (2.0 * sigma**2))
    return k / k.sum()


def gaussian_blur_3d(volume: np.ndarray, sigma: float = 1.5, radius: int = 3) -> np.ndarray:
    """Separable 3D Gaussian blur with reflect padding (no SciPy dependency).

    Vectorized as a slide-and-sum over the (short) kernel taps rather than a
    Python loop over every 1-D fiber. This is on the hot path for PoCA and SSIM.
    """
    k = _gaussian_kernel_1d(sigma, radius)
    out = volume.astype(np.float64)
    for axis in range(volume.ndim):
        pad = [(0, 0)] * volume.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="reflect")
        length = out.shape[axis]
        acc = np.zeros_like(out)
        for tap, weight in enumerate(k):
            sl = [slice(None)] * volume.ndim
            sl[axis] = slice(tap, tap + length)
            acc += weight * padded[tuple(sl)]
        out = acc
    return out


# Cap for a perfect (zero-MSE) match. A finite value keeps results JSON-safe
# (the browser viewer parses these; JSON has no Infinity) while still reading as
# "as good as it gets".
PSNR_PERFECT_DB = 100.0


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB. Higher is better."""
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 0.0:
        return PSNR_PERFECT_DB
    return float(min(10.0 * np.log10(data_range**2 / mse), PSNR_PERFECT_DB))


def ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Mean structural similarity over a Gaussian window. Range roughly [-1, 1]."""
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = gaussian_blur_3d(pred)
    mu_y = gaussian_blur_3d(target)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    var_x = gaussian_blur_3d(pred * pred) - mu_x2
    var_y = gaussian_blur_3d(target * target) - mu_y2
    cov = gaussian_blur_3d(pred * target) - mu_xy
    numerator = (2 * mu_xy + c1) * (2 * cov + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (var_x + var_y + c2)
    return float(np.mean(numerator / denominator))


def iou(pred: np.ndarray, target_mask: np.ndarray, threshold: float = DEFAULT_HIGH_Z_THRESHOLD) -> float:
    """Intersection over union of the thresholded prediction and a target mask."""
    pred_mask = pred >= threshold
    target_mask = target_mask.astype(bool)
    intersection = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()
    if union == 0:
        return 1.0  # both empty: perfect agreement
    return float(intersection / union)


def classification_accuracy(
    pred: np.ndarray, target_mask: np.ndarray, threshold: float = DEFAULT_HIGH_Z_THRESHOLD
) -> float:
    """Per-voxel accuracy of the high-Z vs low-Z call."""
    pred_mask = pred >= threshold
    return float(np.mean(pred_mask == target_mask.astype(bool)))


def evaluate(
    pred_raw: np.ndarray,
    gt_density: np.ndarray,
    gt_high_z: np.ndarray,
    threshold: float = DEFAULT_HIGH_Z_THRESHOLD,
) -> dict[str, float]:
    """Normalize a raw reconstruction and score it against ground truth.

    Returns a dict with ``psnr``, ``ssim``, ``iou``, and ``accuracy``.
    """
    pred = normalize_volume(pred_raw)
    gt = np.clip(np.asarray(gt_density, dtype=np.float64), 0.0, 1.0)
    return {
        "psnr": psnr(pred, gt),
        "ssim": ssim(pred, gt),
        "iou": iou(pred, gt_high_z, threshold),
        "accuracy": classification_accuracy(pred, gt_high_z, threshold),
    }
