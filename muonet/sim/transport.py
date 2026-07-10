"""Straight-line muon transport through a voxel grid.

Cosmic muons scatter by small angles, so to a good approximation each muon
travels in a straight line through the object while accumulating radiation
thickness ``x / X0``. We integrate that thickness with fixed-step ray marching
(fully vectorized over all muons), locate an effective scattering vertex at the
inverse-radiation-length-weighted centroid of the path, and hand both to the
scattering model. The vertex is what Point-of-Closest-Approach later tries to
recover, so anchoring it in the dense material is what makes reconstruction
work.
"""

from __future__ import annotations

import numpy as np

from ..geometry import VoxelGrid
from ..materials import MATERIALS

_AIR_INV_X0 = MATERIALS["air"].inv_x0


def ray_aabb(
    origins: np.ndarray,
    directions: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slab-method intersection of forward rays with an axis-aligned box.

    Returns ``(t_enter, t_exit, hit)`` where distances are along the (unit)
    direction and ``hit`` marks rays that cross the box ahead of the origin.
    """
    d = np.where(np.abs(directions) < 1e-12, 1e-12, directions)
    t1 = (lower - origins) / d
    t2 = (upper - origins) / d
    t_near = np.minimum(t1, t2)
    t_far = np.maximum(t1, t2)
    t_enter = np.maximum(np.max(t_near, axis=-1), 0.0)
    t_exit = np.min(t_far, axis=-1)
    hit = t_exit > t_enter
    return t_enter, t_exit, hit


def integrate_path(
    grid: VoxelGrid,
    inv_x0_field: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    n_steps: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate radiation thickness along each ray through the grid.

    Args:
        grid: The voxel grid.
        inv_x0_field: ``(nx, ny, nz)`` inverse radiation length per voxel.
        origins: ``(n, 3)`` ray starting points (above the grid).
        directions: ``(n, 3)`` unit directions (pointing down through the grid).
        n_steps: Samples per ray. 64 is plenty for centimeter-scale objects.

    Returns:
        ``(x_over_x0, vertex, dense_path_cm)``:
          - ``x_over_x0``: ``(n,)`` traversed thickness in radiation lengths.
          - ``vertex``: ``(n, 3)`` effective scattering point.
          - ``dense_path_cm``: ``(n,)`` path length through non-air material,
            used to scale the lateral scattering offset.
    """
    t_enter, t_exit, hit = ray_aabb(origins, directions, grid.lower, grid.upper)

    # Fallback vertex: where the ray crosses the object mid-plane z = center_z.
    dz = np.where(np.abs(directions[:, 2]) < 1e-12, 1e-12, directions[:, 2])
    t_mid = (grid.center_cm[2] - origins[:, 2]) / dz
    fallback_vertex = origins + t_mid[:, None] * directions

    seg = np.clip(t_exit - t_enter, 0.0, None)
    step_len = seg / n_steps  # (n,)
    frac = (np.arange(n_steps) + 0.5) / n_steps  # (n_steps,)
    t_samples = t_enter[:, None] + frac[None, :] * seg[:, None]  # (n, n_steps)
    points = origins[:, None, :] + t_samples[..., None] * directions[:, None, :]

    voxel = grid.world_to_voxel(points)
    idx = np.floor(voxel).astype(np.int64)
    for axis in range(3):
        idx[..., axis] = np.clip(idx[..., axis], 0, grid.shape[axis] - 1)
    samples = inv_x0_field[idx[..., 0], idx[..., 1], idx[..., 2]]  # (n, n_steps)

    x_over_x0 = samples.sum(axis=1) * step_len

    weights = samples
    wsum = weights.sum(axis=1)
    weighted = (points * weights[..., None]).sum(axis=1)
    has_weight = wsum > 0.0
    vertex = np.where(
        has_weight[:, None],
        weighted / np.where(has_weight[:, None], wsum[:, None], 1.0),
        fallback_vertex,
    )

    dense = samples > (_AIR_INV_X0 * 5.0)
    dense_path_cm = dense.sum(axis=1) * step_len

    missed = ~hit
    x_over_x0 = np.where(missed, 0.0, x_over_x0)
    dense_path_cm = np.where(missed, 0.0, dense_path_cm)
    vertex = np.where(missed[:, None], fallback_vertex, vertex)
    return x_over_x0, vertex, dense_path_cm


def transverse_basis(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane transverse to each direction."""
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (directions.shape[0], 1))
    near_z = np.abs(directions[:, 2]) > 0.9
    ref[near_z] = np.array([1.0, 0.0, 0.0])
    u = np.cross(directions, ref)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(directions, u)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return u, v
