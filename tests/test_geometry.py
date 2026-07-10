"""Voxel grid, scene rasterization, and detector geometry."""

import numpy as np

from muonet.geometry import DetectorGeometry, Primitive, Scene, VoxelGrid


def test_voxel_grid_bounds_and_mapping():
    grid = VoxelGrid(shape=(10, 10, 10), size_cm=(20, 20, 20))
    assert np.allclose(grid.lower, -10.0)
    assert np.allclose(grid.upper, 10.0)
    assert np.allclose(grid.voxel_size_cm, 2.0)
    # Center of the grid maps to the middle voxel coordinate.
    v = grid.world_to_voxel(np.array([[0.0, 0.0, 0.0]]))
    assert np.allclose(v, 5.0)


def test_voxel_centers_shape_and_extent():
    grid = VoxelGrid(shape=(4, 5, 6), size_cm=(4, 5, 6))
    centers = grid.voxel_centers()
    assert centers.shape == (4, 5, 6, 3)
    assert np.isclose(centers[0, 0, 0, 0], -1.5)  # first voxel center
    assert np.isclose(centers[-1, 0, 0, 0], 1.5)


def test_scene_box_rasterizes():
    grid = VoxelGrid(shape=(20, 20, 20), size_cm=(20, 20, 20))
    scene = Scene.from_primitives(
        grid, "air", [Primitive("box", "lead", (0, 0, 0), (10, 10, 10))]
    )
    frac_lead = np.mean(scene.material_ids == scene.registry.index("lead"))
    # A 10 cm cube in a 20 cm box is 1/8 of the volume.
    assert np.isclose(frac_lead, 0.125, atol=0.02)


def test_scene_sphere_and_density_fields():
    grid = VoxelGrid(shape=(24, 24, 24), size_cm=(24, 24, 24))
    scene = Scene.from_primitives(
        grid, "air", [Primitive("sphere", "tungsten", (0, 0, 0), radius_cm=6.0)]
    )
    density = scene.normalized_density()
    assert density.max() <= 1.0 + 1e-9
    assert np.isclose(density.max(), 1.0)  # tungsten present -> reaches 1
    assert scene.high_z_mask().sum() > 0


def test_later_primitive_overwrites_earlier():
    grid = VoxelGrid(shape=(10, 10, 10), size_cm=(10, 10, 10))
    scene = Scene.from_primitives(
        grid,
        "air",
        [
            Primitive("box", "iron", (0, 0, 0), (10, 10, 10)),
            Primitive("box", "tungsten", (0, 0, 0), (4, 4, 4)),
        ],
    )
    center_id = scene.material_ids[5, 5, 5]
    assert scene.registry[center_id] == "tungsten"


def test_detector_planes_and_acceptance():
    det = DetectorGeometry(object_half_z_cm=10.0, n_planes=3, plane_pitch_cm=2.0)
    top = det.top_plane_z()
    bottom = det.bottom_plane_z()
    assert len(top) == 3 and len(bottom) == 3
    assert np.all(top > 0) and np.all(bottom < 0)
    assert det.top_z == top.max()
    xy = np.array([[0.0, 0.0], [1000.0, 0.0]])
    acc = det.in_acceptance(xy)
    assert acc[0] and not acc[1]
