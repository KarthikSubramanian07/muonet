"""Config loading, scene building, and the scene library."""

import numpy as np
import pytest

from muonet.config import RunConfig, load_run_config, scene_from_spec
from muonet.geometry import VoxelGrid
from muonet.scenes import BUILTIN_SCENES, build_scene, list_scenes, random_scene


def test_builtin_scenes_build():
    grid = VoxelGrid(shape=(16, 16, 16), size_cm=(32, 32, 32))
    for name in list_scenes():
        scene = build_scene(name, grid)
        assert scene.material_ids.shape == (16, 16, 16)
        assert (scene.normalized_density() > 0).any()


def test_unknown_scene_raises():
    with pytest.raises(KeyError):
        build_scene("no_such_scene")


def test_random_scene_has_material():
    grid = VoxelGrid(shape=(16, 16, 16), size_cm=(32, 32, 32))
    rng = np.random.default_rng(0)
    scene = random_scene(rng, grid)
    assert scene.normalized_density().max() > 0.0


def test_scene_from_inline_spec():
    grid = VoxelGrid(shape=(16, 16, 16), size_cm=(32, 32, 32))
    spec = {
        "background": "air",
        "primitives": [
            {"shape": "box", "material": "lead", "center_cm": [0, 0, 0], "size_cm": [8, 8, 8]},
            {"shape": "sphere", "material": "tungsten", "center_cm": [4, 0, 0], "radius_cm": 3.0},
        ],
    }
    scene = scene_from_spec(spec, grid)
    assert "lead" in scene.registry and "tungsten" in scene.registry


def test_run_config_defaults_and_build():
    cfg = RunConfig()
    assert cfg.scene == "tungsten_cube"
    scene = cfg.build_scene()
    assert scene.grid.shape == (32, 32, 32)


def test_load_run_config(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("scene: lead_sphere\ndetector: ideal\ndosage: 512\nseed: 3\n")
    cfg = load_run_config(str(path))
    assert cfg.scene == "lead_sphere"
    assert cfg.detector == "ideal"
    assert cfg.dosage == 512
    assert cfg.seed == 3


def test_oversized_grid_rejected():
    cfg = RunConfig(grid_shape=(200, 200, 200))
    with pytest.raises(ValueError):
        cfg.grid()


def test_bad_scene_spec_type_rejected():
    grid = VoxelGrid(shape=(8, 8, 8), size_cm=(8, 8, 8))
    with pytest.raises(ValueError):
        scene_from_spec([1, 2, 3], grid)  # neither a name nor a mapping


def test_builtin_scene_count():
    assert set(BUILTIN_SCENES) == {
        "tungsten_cube",
        "lead_sphere",
        "iron_vs_tungsten",
        "concealed_core",
    }
