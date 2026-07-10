"""Reconstruction quality metrics."""

import numpy as np

from muonet.metrics import (
    classification_accuracy,
    evaluate,
    gaussian_blur_3d,
    iou,
    normalize_volume,
    psnr,
    ssim,
)


def test_normalize_bounds():
    v = np.array([0.0, 1.0, 2.0, 100.0])
    n = normalize_volume(v)
    assert n.min() >= 0.0 and n.max() <= 1.0


def test_normalize_all_zero():
    assert np.all(normalize_volume(np.zeros((4, 4, 4))) == 0.0)


def test_psnr_identical_is_infinite():
    v = np.random.default_rng(0).random((8, 8, 8))
    assert psnr(v, v) == float("inf")


def test_psnr_known_mse():
    a = np.zeros((4, 4, 4))
    b = np.full((4, 4, 4), 0.1)
    # mse = 0.01, PSNR = 10 log10(1/0.01) = 20 dB.
    assert np.isclose(psnr(a, b), 20.0, atol=1e-6)


def test_ssim_identical_is_one():
    v = np.random.default_rng(0).random((16, 16, 16))
    assert np.isclose(ssim(v, v), 1.0, atol=1e-6)


def test_ssim_worse_for_noise():
    rng = np.random.default_rng(0)
    v = rng.random((16, 16, 16))
    noisy = np.clip(v + rng.normal(0, 0.3, v.shape), 0, 1)
    assert ssim(v, noisy) < ssim(v, v)


def test_gaussian_blur_shape_and_conservation():
    rng = np.random.default_rng(0)
    v = rng.random((10, 10, 10))
    b = gaussian_blur_3d(v)
    assert b.shape == v.shape
    # Reflect padding conserves total mass to good approximation.
    assert np.isclose(b.sum(), v.sum(), rtol=0.05)


def test_iou_and_accuracy():
    pred = np.zeros((4, 4, 4))
    pred[0, 0, 0] = 1.0
    pred[0, 0, 1] = 1.0
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 0, 0] = True
    # intersection = 1 voxel, union = 3 voxels.
    assert np.isclose(iou(pred, mask), 1 / 3)
    # 62 of 64 voxels agree (both the 2 mispredictions differ).
    assert np.isclose(classification_accuracy(pred, mask), 62 / 64)


def test_iou_both_empty():
    assert iou(np.zeros((4, 4, 4)), np.zeros((4, 4, 4), dtype=bool)) == 1.0


def test_evaluate_keys():
    rng = np.random.default_rng(0)
    pred = rng.random((8, 8, 8))
    gt = rng.random((8, 8, 8))
    hz = gt > 0.5
    out = evaluate(pred, gt, hz)
    assert set(out) == {"psnr", "ssim", "iou", "accuracy"}
