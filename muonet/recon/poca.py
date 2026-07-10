"""Point of Closest Approach: the classical geometric baseline.

Each muon is assumed to scatter once, at the single point where its measured
incoming and outgoing tracks come closest (the midpoint of their mutual
perpendicular). That point is dropped into the voxel grid carrying the muon's
momentum-weighted squared scattering angle, which is proportional to the local
inverse radiation length. Averaging per voxel gives a scattering-density image.

Fast, transparent, and the honest thing to beat. References: Borozdin et al.,
Nature 422:277 (2003); Schultz et al., NIM A 519:687 (2004).
"""

from __future__ import annotations

import numpy as np

from ..detector.effects import Measurement
from ..geometry import VoxelGrid
from ..metrics import gaussian_blur_3d

# Reference momentum (GeV/c) that momentum weighting is scaled to.
DEFAULT_MOMENTUM_REF = 3.0
# Kernel-density smoothing width (in voxels) applied to the PoCA signal.
DEFAULT_SMOOTHING_VOXELS = 1.0


def poca_points(
    in_point: np.ndarray,
    in_dir: np.ndarray,
    out_point: np.ndarray,
    out_dir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Closest-approach point of each incoming/outgoing track pair.

    Returns ``(poca, well_defined)`` where ``poca`` is ``(n, 3)`` and
    ``well_defined`` is ``False`` for near-parallel (unscattered) tracks.
    """
    w0 = in_point - out_point
    b = np.sum(in_dir * out_dir, axis=1)
    d = np.sum(in_dir * w0, axis=1)
    e = np.sum(out_dir * w0, axis=1)
    denom = 1.0 - b**2
    well_defined = denom > 1e-9
    safe = np.where(well_defined, denom, 1.0)
    t1 = (b * e - d) / safe
    t2 = (e - b * d) / safe
    p1 = in_point + t1[:, None] * in_dir
    p2 = out_point + t2[:, None] * out_dir
    return 0.5 * (p1 + p2), well_defined


def poca_reconstruct(
    measurement: Measurement,
    grid: VoxelGrid,
    momentum_ref: float = DEFAULT_MOMENTUM_REF,
    smoothing_voxels: float = DEFAULT_SMOOTHING_VOXELS,
) -> np.ndarray:
    """Reconstruct a scattering-density volume from a measurement via PoCA.

    Each valid muon deposits its momentum-weighted squared scattering angle at
    its closest-approach voxel. The per-voxel sum of that signal is an
    accumulated "scattering dose" image: dense regions grow faster than air as
    more muons arrive, which is exactly why the picture sharpens with exposure.
    Because cosmic flux is low the point cloud is sparse, so we apply the
    Gaussian kernel-density smoothing that is standard for PoCA images. This is
    what makes PoCA a fair baseline rather than a strawman. Returns an
    ``(nx, ny, nz)`` float array proportional to inverse radiation length.
    """
    m = measurement.select_valid()
    poca, well_defined = poca_points(m.in_point, m.in_dir, m.out_point, m.out_dir)

    voxel = grid.world_to_voxel(poca)
    idx = np.floor(voxel).astype(np.int64)
    inside = np.all((idx >= 0) & (idx < np.asarray(grid.shape)), axis=1)
    keep = well_defined & inside

    idx = idx[keep]
    momentum_scale = (m.momentum_gev[keep] / momentum_ref) ** 2
    signal = (m.scatter_angle[keep] ** 2) * momentum_scale

    flat = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), grid.shape)
    n_voxels = int(np.prod(grid.shape))
    density = np.bincount(flat, weights=signal, minlength=n_voxels).reshape(grid.shape)

    if smoothing_voxels > 0.0:
        radius = max(1, int(round(3.0 * smoothing_voxels)))
        density = gaussian_blur_3d(density, sigma=smoothing_voxels, radius=radius)
    return density
