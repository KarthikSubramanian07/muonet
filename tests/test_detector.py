"""Detector realism: track fitting, efficiency, noise, and preset behaviour."""

import numpy as np

from muonet.detector import DETECTOR_PRESETS, apply_detector, get_detector_config
from muonet.detector.effects import fit_line
from muonet.geometry import DetectorGeometry, Primitive, Scene, VoxelGrid
from muonet.seeding import make_rng
from muonet.sim import simulate


def _truth(n=2000, seed=0):
    grid = VoxelGrid(shape=(32, 32, 32), size_cm=(32, 32, 32))
    scene = Scene.from_primitives(
        grid, "air", [Primitive("box", "tungsten", (0, 0, 0), (10, 10, 10))]
    )
    det = DetectorGeometry(object_half_z_cm=16.0)
    return simulate(scene, det, n, make_rng(seed))


def test_fit_line_recovers_known_track():
    """A clean set of collinear hits fits back to the generating line."""
    plane_z = np.array([10.0, 12.0, 14.0, 16.0])
    # Line through (0,0,0) heading down-and-sideways: x = 0.5 z, y = -0.2 z.
    xs = 0.5 * plane_z
    ys = -0.2 * plane_z
    hits = np.stack([xs, ys], axis=-1)[None]  # (1, P, 2)
    mask = np.ones((1, 4), dtype=bool)
    point, direction, valid = fit_line(hits, mask, plane_z)
    assert valid[0]
    assert np.allclose(point[0, :2], 0.0, atol=1e-6)  # crosses origin at z=0
    # Direction should be parallel to (0.5, -0.2, 1) pointing down.
    expected = -np.array([0.5, -0.2, 1.0])
    expected /= np.linalg.norm(expected)
    assert np.allclose(direction[0], expected, atol=1e-6)


def test_fit_line_invalid_with_one_plane():
    plane_z = np.array([10.0, 12.0])
    hits = np.array([[[1.0, 1.0], [np.nan, np.nan]]])
    mask = np.array([[True, False]])
    _, _, valid = fit_line(hits, mask, plane_z)
    assert not valid[0]


def test_ideal_detector_is_lossless():
    truth = _truth()
    meas = apply_detector(truth, get_detector_config("ideal"), make_rng(1))
    assert np.all(meas.valid)
    # No smearing: measured tracks match truth to floating point.
    assert np.allclose(meas.in_dir, truth.in_dir, atol=1e-6)
    assert np.allclose(meas.momentum_gev, truth.momentum_gev)


def test_resolution_degrades_angle_measurement():
    """Coarser strips widen the measured scattering-angle error."""
    truth = _truth()
    fine = apply_detector(truth, get_detector_config("research"), make_rng(2))
    coarse = apply_detector(truth, get_detector_config("battery_limited"), make_rng(2))
    err_fine = np.abs(fine.scatter_angle - truth.scatter_angle)[fine.valid]
    err_coarse = np.abs(coarse.scatter_angle - truth.scatter_angle)[coarse.valid]
    assert err_coarse.mean() > err_fine.mean()


def test_efficiency_drops_muons():
    truth = _truth()
    ideal = apply_detector(truth, get_detector_config("ideal"), make_rng(3))
    battery = apply_detector(truth, get_detector_config("battery_limited"), make_rng(3))
    assert battery.valid.sum() < ideal.valid.sum()


def test_unknown_momentum_uses_assumed():
    truth = _truth()
    cfg = get_detector_config("realistic_sipm")
    meas = apply_detector(truth, cfg, make_rng(4))
    assert not cfg.momentum_known
    assert np.allclose(meas.momentum_gev, cfg.assumed_momentum_gev)


def test_select_valid_filters():
    truth = _truth()
    meas = apply_detector(truth, get_detector_config("battery_limited"), make_rng(5))
    valid = meas.select_valid()
    assert len(valid) == int(meas.valid.sum())
    assert np.all(valid.valid)


def test_presets_present():
    for name in ("ideal", "research", "realistic_sipm", "battery_limited"):
        assert name in DETECTOR_PRESETS
        assert DETECTOR_PRESETS[name].name == name
