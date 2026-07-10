"""Built-in scenes and a randomized scene generator.

The named scenes are the fixed targets the benchmarks and the viewer use. The
randomized generator produces the diverse training set the learned model needs:
one to three primitives of random material, shape, size, and position dropped
into air. Keeping every scene in a single 32^3 grid means one model handles
them all and the viewer speaks one format.
"""

from __future__ import annotations

import numpy as np

from .geometry import DetectorGeometry, Primitive, Scene, VoxelGrid

# One grid, one detector, everywhere. 32 cm cube at 1 cm resolution.
GRID_SHAPE = (32, 32, 32)
GRID_SIZE_CM = (32.0, 32.0, 32.0)


def default_grid() -> VoxelGrid:
    return VoxelGrid(shape=GRID_SHAPE, size_cm=GRID_SIZE_CM)


def default_detector() -> DetectorGeometry:
    return DetectorGeometry(object_half_z_cm=16.0, n_planes=4, plane_pitch_cm=4.0)


# Fixed benchmark and viewer targets, each a list of primitives painted on air.
BUILTIN_SCENES: dict[str, list[Primitive]] = {
    # The headline demo: a single dense cube.
    "tungsten_cube": [Primitive("box", "tungsten", (0, 0, 0), (10, 10, 10))],
    # A rounded target, harder to pin down at the edges.
    "lead_sphere": [Primitive("sphere", "lead", (0, 0, 0), radius_cm=6.0)],
    # Z discrimination and an honest hard case: iron scatters far less than
    # tungsten, so the low-Z block is where reconstruction struggles.
    "iron_vs_tungsten": [
        Primitive("box", "iron", (-7, 0, 0), (8, 8, 12)),
        Primitive("box", "tungsten", (7, 0, 0), (8, 8, 12)),
    ],
    # A concealment scenario: a small dense core inside a light container.
    "concealed_core": [
        Primitive("box", "aluminum", (0, 0, 0), (18, 18, 14)),
        Primitive("box", "tungsten", (0, 0, 0), (5, 5, 5)),
    ],
}

# Materials the randomized training scenes draw from, spanning low to high Z.
_TRAIN_MATERIALS = ["water", "concrete", "aluminum", "iron", "lead", "tungsten"]


def build_scene(name: str, grid: VoxelGrid | None = None) -> Scene:
    """Instantiate a named built-in scene."""
    grid = grid or default_grid()
    if name not in BUILTIN_SCENES:
        known = ", ".join(BUILTIN_SCENES)
        raise KeyError(f"unknown scene {name!r}; known scenes: {known}")
    return Scene.from_primitives(grid, "air", BUILTIN_SCENES[name])


def random_scene(rng: np.random.Generator, grid: VoxelGrid | None = None) -> Scene:
    """A random arrangement of 1-3 material blobs in air, for training."""
    grid = grid or default_grid()
    half = 0.5 * np.asarray(grid.size_cm)
    n_objects = int(rng.integers(1, 4))
    primitives: list[Primitive] = []
    for _ in range(n_objects):
        material = str(rng.choice(_TRAIN_MATERIALS))
        # Keep objects inside the inner 60% so they stay well inside the grid.
        center = tuple(rng.uniform(-0.3 * half, 0.3 * half))
        if rng.random() < 0.5:
            size = tuple(rng.uniform(4.0, 12.0, size=3))
            primitives.append(Primitive("box", material, center, tuple(size)))
        else:
            radius = float(rng.uniform(3.0, 7.0))
            primitives.append(
                Primitive("sphere", material, center, radius_cm=radius)
            )
    return Scene.from_primitives(grid, "air", primitives)


def list_scenes() -> list[str]:
    return list(BUILTIN_SCENES)
