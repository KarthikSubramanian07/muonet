"""PoCA geometry and reconstruction."""

import numpy as np

from muonet.detector import apply_detector, get_detector_config
from muonet.geometry import DetectorGeometry, Primitive, Scene, VoxelGrid
from muonet.recon import poca_points, poca_reconstruct
from muonet.seeding import make_rng
from muonet.sim import simulate


def test_poca_point_of_intersecting_tracks():
    """Two tracks crossing at a point have their PoCA at that point."""
    in_point = np.array([[1.0, 2.0, 0.0]])
    out_point = np.array([[1.0, 2.0, 0.0]])
    in_dir = np.array([[0.1, 0.0, -1.0]])
    out_dir = np.array([[-0.1, 0.0, -1.0]])
    in_dir /= np.linalg.norm(in_dir, axis=1, keepdims=True)
    out_dir /= np.linalg.norm(out_dir, axis=1, keepdims=True)
    poca, ok = poca_points(in_point, in_dir, out_point, out_dir)
    assert ok[0]
    assert np.allclose(poca[0], [1.0, 2.0, 0.0], atol=1e-6)


def test_poca_parallel_tracks_flagged():
    d = np.array([[0.0, 0.0, -1.0]])
    poca, ok = poca_points(np.array([[0.0, 0.0, 5.0]]), d, np.array([[1.0, 0.0, 5.0]]), d)
    assert not ok[0]


def test_poca_localizes_dense_block():
    grid = VoxelGrid(shape=(32, 32, 32), size_cm=(32, 32, 32))
    scene = Scene.from_primitives(
        grid, "air", [Primitive("box", "tungsten", (0, 0, 0), (10, 10, 10))]
    )
    det = DetectorGeometry(object_half_z_cm=16.0)
    rng = make_rng(0)
    truth = simulate(scene, det, 3000, rng)
    meas = apply_detector(truth, get_detector_config("ideal"), rng)
    vol = poca_reconstruct(meas, grid)

    hz = scene.high_z_mask()
    assert vol[hz].mean() > 10.0 * vol[~hz].mean()
    # The densest reconstructed voxels concentrate inside the tungsten cube:
    # most of the top 1% by density should fall on high-Z voxels.
    # (The high-Z cube is ~3% of the volume, so >0.4 here is a large enrichment;
    # Gaussian smoothing bleeds some density one voxel past the cube edge.)
    top = vol >= np.percentile(vol, 99.0)
    assert hz[top].mean() > 0.4


def test_poca_offset_block_localizes_offset():
    grid = VoxelGrid(shape=(32, 32, 32), size_cm=(32, 32, 32))
    scene = Scene.from_primitives(
        grid, "air", [Primitive("box", "tungsten", (8.0, 0, 0), (8, 8, 8))]
    )
    det = DetectorGeometry(object_half_z_cm=16.0)
    rng = make_rng(1)
    truth = simulate(scene, det, 3000, rng)
    meas = apply_detector(truth, get_detector_config("ideal"), rng)
    vol = poca_reconstruct(meas, grid)
    # Center of mass of the reconstruction should sit on the +x side.
    centers = grid.voxel_centers()
    com_x = np.sum(centers[..., 0] * vol) / vol.sum()
    assert com_x > 3.0


def test_poca_improves_with_dose():
    """More muons give a higher-contrast reconstruction of the block."""
    grid = VoxelGrid(shape=(32, 32, 32), size_cm=(32, 32, 32))
    scene = Scene.from_primitives(
        grid, "air", [Primitive("box", "tungsten", (0, 0, 0), (10, 10, 10))]
    )
    det = DetectorGeometry(object_half_z_cm=16.0)
    hz = scene.high_z_mask()

    def contrast(dose):
        rng = make_rng(0)
        truth = simulate(scene, det, dose, rng)
        meas = apply_detector(truth, get_detector_config("ideal"), rng)
        vol = poca_reconstruct(meas, grid)
        vol = vol / vol.max()
        return vol[hz].mean()

    assert contrast(8192) > contrast(256)
