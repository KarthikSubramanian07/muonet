"""Voxel grids, scenes, and detector geometry.

Everything lives in a right-handed frame with +z pointing up. The object under
inspection sits in a voxel grid centered on the origin. Two tracking stations
sandwich it: one above (measures the incoming muon), one below (measures the
outgoing muon). Cosmic muons rain down through the stack in the -z direction.

All lengths are centimeters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .materials import MATERIALS, get_material


@dataclass(frozen=True)
class VoxelGrid:
    """An axis-aligned grid of voxels, centered on ``center_cm``."""

    shape: tuple[int, int, int]
    size_cm: tuple[float, float, float]
    center_cm: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def voxel_size_cm(self) -> np.ndarray:
        return np.asarray(self.size_cm) / np.asarray(self.shape)

    @property
    def lower(self) -> np.ndarray:
        return np.asarray(self.center_cm) - 0.5 * np.asarray(self.size_cm)

    @property
    def upper(self) -> np.ndarray:
        return np.asarray(self.center_cm) + 0.5 * np.asarray(self.size_cm)

    def voxel_centers(self) -> np.ndarray:
        """(nx, ny, nz, 3) array of voxel-center world coordinates."""
        axes = [
            self.lower[d] + (np.arange(self.shape[d]) + 0.5) * self.voxel_size_cm[d]
            for d in range(3)
        ]
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        return np.stack([gx, gy, gz], axis=-1)

    def world_to_voxel(self, points: np.ndarray) -> np.ndarray:
        """Map world points (..., 3) to continuous voxel coordinates (..., 3)."""
        return (points - self.lower) / self.voxel_size_cm

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask of which world points (..., 3) fall inside the grid."""
        v = self.world_to_voxel(points)
        return np.all((v >= 0.0) & (v < np.asarray(self.shape)), axis=-1)


@dataclass
class Primitive:
    """A shape painted into a scene. Later primitives overwrite earlier ones."""

    shape: str  # "box" | "sphere" | "cylinder"
    material: str
    center_cm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size_cm: tuple[float, float, float] = (10.0, 10.0, 10.0)  # box full extents
    radius_cm: float = 5.0  # sphere / cylinder radius
    height_cm: float = 10.0  # cylinder height (along z)

    def mask(self, centers: np.ndarray) -> np.ndarray:
        """Boolean mask over voxel centers (nx, ny, nz, 3) inside this shape."""
        rel = centers - np.asarray(self.center_cm)
        if self.shape == "box":
            half = 0.5 * np.asarray(self.size_cm)
            return np.all(np.abs(rel) <= half, axis=-1)
        if self.shape == "sphere":
            return np.sum(rel**2, axis=-1) <= self.radius_cm**2
        if self.shape == "cylinder":
            radial = rel[..., 0] ** 2 + rel[..., 1] ** 2 <= self.radius_cm**2
            axial = np.abs(rel[..., 2]) <= 0.5 * self.height_cm
            return radial & axial
        raise ValueError(f"unknown primitive shape {self.shape!r}")


@dataclass
class Scene:
    """A voxelized world: a material id per voxel plus the material registry."""

    grid: VoxelGrid
    material_ids: np.ndarray  # int (nx, ny, nz)
    registry: list[str]  # registry[id] -> material name

    @classmethod
    def from_primitives(
        cls,
        grid: VoxelGrid,
        background: str,
        primitives: list[Primitive],
    ) -> Scene:
        registry: list[str] = [background]
        index_of = {background: 0}

        def material_id(name: str) -> int:
            get_material(name)  # validate
            if name not in index_of:
                index_of[name] = len(registry)
                registry.append(name)
            return index_of[name]

        centers = grid.voxel_centers()
        material_ids = np.zeros(grid.shape, dtype=np.int16)
        for prim in primitives:
            material_ids[prim.mask(centers)] = material_id(prim.material)
        return cls(grid=grid, material_ids=material_ids, registry=registry)

    def inv_x0_field(self) -> np.ndarray:
        """Per-voxel inverse radiation length (cm^-1)."""
        lookup = np.array([MATERIALS[name].inv_x0 for name in self.registry])
        return lookup[self.material_ids]

    def z_eff_field(self) -> np.ndarray:
        lookup = np.array([MATERIALS[name].z_eff for name in self.registry])
        return lookup[self.material_ids]

    def normalized_density(self) -> np.ndarray:
        """Ground-truth scattering density on the shared [0, 1] scale."""
        from .materials import REFERENCE_INV_X0

        return self.inv_x0_field() / REFERENCE_INV_X0

    def high_z_mask(self) -> np.ndarray:
        """Voxels whose material is high-Z (nuclear-security regime)."""
        from .materials import HIGH_Z_THRESHOLD

        return self.z_eff_field() >= HIGH_Z_THRESHOLD


@dataclass
class DetectorGeometry:
    """Two tracking stations sandwiching the object volume.

    Each station has ``n_planes`` parallel measuring planes spaced
    ``plane_gap_cm`` apart. The top station sits above ``+object_half_z`` and
    the bottom below ``-object_half_z``. Planes measure (x, y) at a known z.
    """

    object_half_z_cm: float = 16.0
    station_gap_cm: float = 6.0  # gap from object surface to nearest plane
    n_planes: int = 4
    plane_pitch_cm: float = 4.0  # spacing between planes within a station
    plane_half_extent_cm: float = 30.0  # sensitive half-width in x and y

    def top_plane_z(self) -> np.ndarray:
        base = self.object_half_z_cm + self.station_gap_cm
        return base + self.plane_pitch_cm * np.arange(self.n_planes)

    def bottom_plane_z(self) -> np.ndarray:
        base = -(self.object_half_z_cm + self.station_gap_cm)
        return base - self.plane_pitch_cm * np.arange(self.n_planes)

    @property
    def top_z(self) -> float:
        return float(self.top_plane_z().max())

    def in_acceptance(self, xy: np.ndarray) -> np.ndarray:
        """Whether (x, y) points (..., 2) land on the sensitive area."""
        return np.all(np.abs(xy) <= self.plane_half_extent_cm, axis=-1)
