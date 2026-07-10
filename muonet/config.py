"""YAML-driven run configuration.

A run config pins everything a result depends on: the scene, the detector
preset, the dosage, and the seed. Point the CLI at one of these and the number
it prints is reproducible. Scenes may be a built-in name or a full inline
specification of primitives, so users can describe their own geometry without
touching Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .geometry import Primitive, Scene, VoxelGrid
from .scenes import BUILTIN_SCENES

# Guard against a config that would allocate an absurd voxel grid (local OOM).
MAX_VOXELS = 128**3


@dataclass
class RunConfig:
    scene: str | dict[str, Any] = "tungsten_cube"
    detector: str = "realistic_sipm"
    dosage: int = 1024
    seed: int = 0
    grid_shape: tuple[int, int, int] = (32, 32, 32)
    grid_size_cm: tuple[float, float, float] = (32.0, 32.0, 32.0)
    extras: dict[str, Any] = field(default_factory=dict)

    def grid(self) -> VoxelGrid:
        return build_grid(self.grid_shape, self.grid_size_cm)

    def build_scene(self) -> Scene:
        return scene_from_spec(self.scene, self.grid())


def build_grid(shape: Any, size_cm: Any) -> VoxelGrid:
    """Build a voxel grid from config values, rejecting absurd sizes."""
    shape = tuple(int(s) for s in shape)
    if len(shape) != 3 or any(s < 1 for s in shape):
        raise ValueError(f"grid_shape must be three positive integers, got {shape}")
    if int(np.prod(shape)) > MAX_VOXELS:
        raise ValueError(
            f"grid_shape {shape} exceeds {MAX_VOXELS} voxels; refusing to allocate. "
            f"Keep the grid at or below 128^3."
        )
    return VoxelGrid(shape=shape, size_cm=tuple(float(s) for s in size_cm))


def scene_from_spec(spec: str | dict[str, Any], grid: VoxelGrid) -> Scene:
    """Build a scene from a built-in name or an inline primitive specification."""
    if isinstance(spec, str):
        if spec not in BUILTIN_SCENES:
            known = ", ".join(BUILTIN_SCENES)
            raise KeyError(f"unknown scene {spec!r}; known: {known}")
        return Scene.from_primitives(grid, "air", BUILTIN_SCENES[spec])
    if not isinstance(spec, dict):
        raise ValueError(
            f"scene must be a built-in name or a mapping with 'primitives', got "
            f"{type(spec).__name__}"
        )
    background = spec.get("background", "air")
    primitives = [_primitive_from_dict(p) for p in spec.get("primitives", [])]
    return Scene.from_primitives(grid, background, primitives)


def _primitive_from_dict(data: dict[str, Any]) -> Primitive:
    kwargs: dict[str, Any] = {"shape": data["shape"], "material": data["material"]}
    for key in ("center_cm", "size_cm"):
        if key in data:
            kwargs[key] = tuple(data[key])
    for key in ("radius_cm", "height_cm"):
        if key in data:
            kwargs[key] = float(data[key])
    return Primitive(**kwargs)


def load_run_config(path: str) -> RunConfig:
    """Load a run config from a YAML file."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    known = {f for f in RunConfig.__dataclass_fields__ if f != "extras"}
    fields = {k: v for k, v in data.items() if k in known}
    extras = {k: v for k, v in data.items() if k not in known}
    if "grid_shape" in fields:
        fields["grid_shape"] = tuple(fields["grid_shape"])
    if "grid_size_cm" in fields:
        fields["grid_size_cm"] = tuple(fields["grid_size_cm"])
    return RunConfig(**fields, extras=extras)
