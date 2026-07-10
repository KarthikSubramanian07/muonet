"""Ray transport: AABB intersection and radiation-thickness integration."""

import numpy as np

from muonet.geometry import Primitive, Scene, VoxelGrid
from muonet.materials import MATERIALS
from muonet.sim.transport import integrate_path, ray_aabb, transverse_basis


def test_ray_aabb_hit_and_miss():
    lower = np.array([-1.0, -1.0, -1.0])
    upper = np.array([1.0, 1.0, 1.0])
    origins = np.array([[0.0, 0.0, 5.0], [5.0, 5.0, 5.0]])
    directions = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    t_enter, t_exit, hit = ray_aabb(origins, directions, lower, upper)
    assert hit[0] and not hit[1]
    assert np.isclose(t_enter[0], 4.0)
    assert np.isclose(t_exit[0], 6.0)


def test_integrate_full_block_matches_analytic():
    """A vertical muon through a solid tungsten grid sees size_z / X0."""
    grid = VoxelGrid(shape=(16, 16, 16), size_cm=(32, 32, 32))
    scene = Scene.from_primitives(
        grid, "tungsten", [Primitive("box", "tungsten", (0, 0, 0), (32, 32, 32))]
    )
    inv_x0 = scene.inv_x0_field()
    origins = np.array([[0.0, 0.0, 20.0]])
    directions = np.array([[0.0, 0.0, -1.0]])
    x_over_x0, vertex, dense = integrate_path(grid, inv_x0, origins, directions, 128)
    expected = 32.0 / MATERIALS["tungsten"].radiation_length_cm
    assert np.isclose(x_over_x0[0], expected, rtol=0.01)
    assert np.isclose(dense[0], 32.0, rtol=0.02)


def test_vertex_pulled_into_dense_object():
    """The effective vertex sits inside an off-center dense block, not at z=0."""
    grid = VoxelGrid(shape=(32, 32, 32), size_cm=(32, 32, 32))
    scene = Scene.from_primitives(
        grid, "air", [Primitive("box", "tungsten", (0, 0, 8.0), (8, 8, 8))]
    )
    inv_x0 = scene.inv_x0_field()
    origins = np.array([[0.0, 0.0, 20.0]])
    directions = np.array([[0.0, 0.0, -1.0]])
    _, vertex, _ = integrate_path(grid, inv_x0, origins, directions, 128)
    assert np.isclose(vertex[0, 2], 8.0, atol=1.0)


def test_miss_gives_zero_thickness():
    grid = VoxelGrid(shape=(8, 8, 8), size_cm=(8, 8, 8))
    scene = Scene.from_primitives(grid, "air", [])
    inv_x0 = scene.inv_x0_field()
    origins = np.array([[100.0, 100.0, 20.0]])
    directions = np.array([[0.0, 0.0, -1.0]])
    x_over_x0, _, dense = integrate_path(grid, inv_x0, origins, directions, 32)
    assert x_over_x0[0] == 0.0
    assert dense[0] == 0.0


def test_transverse_basis_orthonormal():
    rng = np.random.default_rng(0)
    d = rng.standard_normal((100, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    u, v = transverse_basis(d)
    assert np.allclose(np.sum(u * d, axis=1), 0.0, atol=1e-6)
    assert np.allclose(np.sum(v * d, axis=1), 0.0, atol=1e-6)
    assert np.allclose(np.sum(u * v, axis=1), 0.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-6)
