"""MLEM reconstruction: localization and stability."""

import numpy as np

from muonet.detector import apply_detector, get_detector_config
from muonet.geometry import DetectorGeometry, Primitive, Scene, VoxelGrid
from muonet.recon.mlem import mlem_reconstruct
from muonet.seeding import make_rng
from muonet.sim import simulate


def _measure(scene, det, dose, cfg_name, seed):
    rng = make_rng(seed)
    truth = simulate(scene, det, dose, rng)
    return apply_detector(truth, get_detector_config(cfg_name), rng)


def test_mlem_localizes_dense_block():
    """MLEM concentrates scattering density inside the block. It is a reference
    implementation, not tuned to beat PoCA, so we check localization, not that
    it wins."""
    grid = VoxelGrid(shape=(24, 24, 24), size_cm=(24, 24, 24))
    scene = Scene.from_primitives(
        grid, "air", [Primitive("box", "tungsten", (0, 0, 0), (8, 8, 8))]
    )
    det = DetectorGeometry(object_half_z_cm=12.0)
    meas = _measure(scene, det, 1500, "ideal", 0)
    vol = mlem_reconstruct(meas, grid, n_iterations=60)

    hz = scene.high_z_mask()
    assert vol.shape == grid.shape
    assert np.all(np.isfinite(vol))
    assert vol[hz].mean() > 1.5 * vol[~hz].mean()


def test_mlem_stays_bounded():
    """The step clamp must keep MLEM from blowing up over many iterations."""
    grid = VoxelGrid(shape=(20, 20, 20), size_cm=(24, 24, 24))
    scene = Scene.from_primitives(
        grid, "air", [Primitive("sphere", "lead", (0, 0, 0), radius_cm=5.0)]
    )
    det = DetectorGeometry(object_half_z_cm=12.0)
    meas = _measure(scene, det, 1000, "realistic_sipm", 1)
    vol = mlem_reconstruct(meas, grid, n_iterations=40)
    assert np.all(np.isfinite(vol))
    assert vol.max() < 1e3  # bounded, not diverging


def test_mlem_empty_measurement():
    grid = VoxelGrid(shape=(16, 16, 16), size_cm=(24, 24, 24))
    scene = Scene.from_primitives(grid, "air", [])
    det = DetectorGeometry(object_half_z_cm=12.0)
    meas = _measure(scene, det, 200, "ideal", 0)
    vol = mlem_reconstruct(meas, grid, n_iterations=5)
    assert vol.shape == grid.shape
    assert np.all(np.isfinite(vol))
