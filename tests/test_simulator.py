"""End-to-end simulator behaviour and determinism."""

import numpy as np

from muonet.geometry import DetectorGeometry, Primitive, Scene, VoxelGrid
from muonet.seeding import make_rng
from muonet.sim import simulate


def _scene():
    grid = VoxelGrid(shape=(32, 32, 32), size_cm=(32, 32, 32))
    return Scene.from_primitives(
        grid, "air", [Primitive("box", "tungsten", (0, 0, 0), (10, 10, 10))]
    ), DetectorGeometry(object_half_z_cm=16.0)


def test_generates_exact_count():
    scene, det = _scene()
    mu = simulate(scene, det, 500, make_rng(0))
    assert len(mu) == 500
    assert mu.in_dir.shape == (500, 3)
    assert mu.top_hits.shape[0] == 500
    assert mu.top_hits.shape[2] == 2


def test_deterministic_from_seed():
    scene, det = _scene()
    a = simulate(scene, det, 300, make_rng(42))
    b = simulate(scene, det, 300, make_rng(42))
    assert np.array_equal(a.scatter_angle, b.scatter_angle)
    assert np.array_equal(a.momentum_gev, b.momentum_gev)


def test_different_seeds_differ():
    scene, det = _scene()
    a = simulate(scene, det, 300, make_rng(1))
    b = simulate(scene, det, 300, make_rng(2))
    assert not np.array_equal(a.scatter_angle, b.scatter_angle)


def test_dense_material_scatters_more():
    """Muons crossing the tungsten scatter far harder than those missing it."""
    scene, det = _scene()
    mu = simulate(scene, det, 4000, make_rng(0))
    hit_dense = mu.path_radiation_lengths > 5.0
    assert hit_dense.sum() > 50
    assert (
        mu.scatter_angle[hit_dense].mean() > 10.0 * mu.scatter_angle[~hit_dense].mean()
    )


def test_all_muons_in_acceptance():
    scene, det = _scene()
    mu = simulate(scene, det, 500, make_rng(0))
    assert np.all(det.in_acceptance(mu.top_hits[:, 0]))
    assert np.all(det.in_acceptance(mu.bottom_hits[:, -1]))


def test_outgoing_direction_is_unit():
    scene, det = _scene()
    mu = simulate(scene, det, 500, make_rng(0))
    assert np.allclose(np.linalg.norm(mu.out_dir, axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(mu.in_dir, axis=1), 1.0, atol=1e-6)
