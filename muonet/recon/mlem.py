"""Maximum Likelihood Expectation Maximization: the strong classical competitor.

PoCA throws away most of the information in each muon by assigning all of its
scattering to one point. MLEM keeps it: it models each muon's scattering angle
and lateral displacement as jointly Gaussian, with a 2x2 covariance that sums
per-voxel contributions along the muon path, and iterates an EM update on the
per-voxel scattering density. It is slower than PoCA and noticeably better,
which is the point: the learned model should have to beat a real competitor,
not a strawman.

Reference: Schultz et al., "Statistical Reconstruction for Cosmic Ray Muon
Tomography," IEEE Trans. Image Processing 16(8):1985 (2007). The per-voxel
weight matrix (L, L^2/2 + LT, L^3/3 + L^2 T + L T^2) and the EM update follow
that paper.
"""

from __future__ import annotations

import numpy as np

from ..detector.effects import Measurement
from ..geometry import VoxelGrid
from ..sim.transport import ray_aabb

DEFAULT_MOMENTUM_REF = 3.0
DEFAULT_ITERATIONS = 50
DEFAULT_STEPS = 96
# Diagonal measurement-error floor that keeps the per-muon covariance invertible
# (angular variance in rad^2, displacement variance in cm^2).
ANGLE_ERROR_VAR = (5.0e-4) ** 2
DISPLACEMENT_ERROR_VAR = (0.05) ** 2


def _path_entries(
    measurement: Measurement, grid: VoxelGrid, n_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per (muon, voxel) path lengths along each incoming track through the grid.

    Returns ``(muon_id, voxel_flat, L, T)`` where ``L`` is the path length inside
    the voxel and ``T`` the remaining path length from the voxel exit to the grid
    exit. Samples on muons that miss the grid are dropped.
    """
    origins = measurement.in_point
    directions = measurement.in_dir
    t_enter, t_exit, hit = ray_aabb(origins, directions, grid.lower, grid.upper)
    seg = np.clip(t_exit - t_enter, 0.0, None)
    step_len = seg / n_steps

    frac = (np.arange(n_steps) + 0.5) / n_steps
    t_samples = t_enter[:, None] + frac[None, :] * seg[:, None]  # (n, S)
    points = origins[:, None, :] + t_samples[..., None] * directions[:, None, :]
    remaining = (t_exit[:, None] - t_samples)  # length from sample to exit

    idx = np.floor(grid.world_to_voxel(points)).astype(np.int64)
    for axis in range(3):
        idx[..., axis] = np.clip(idx[..., axis], 0, grid.shape[axis] - 1)
    voxel_flat = np.ravel_multi_index(
        (idx[..., 0], idx[..., 1], idx[..., 2]), grid.shape
    )

    n = origins.shape[0]
    muon_ids = np.repeat(np.arange(n), n_steps)
    sample_voxel = voxel_flat.reshape(-1)
    sample_len = np.repeat(step_len, n_steps)
    sample_rem = remaining.reshape(-1)
    sample_hit = np.repeat(hit, n_steps)

    keep = sample_hit & (sample_len > 0)
    muon_ids, sample_voxel = muon_ids[keep], sample_voxel[keep]
    sample_len, sample_rem = sample_len[keep], sample_rem[keep]

    # Group samples sharing a (muon, voxel): sum lengths, take exit-side T.
    n_vox = int(np.prod(grid.shape))
    key = muon_ids * n_vox + sample_voxel
    uniq, inverse = np.unique(key, return_inverse=True)
    length = np.bincount(inverse, weights=sample_len)
    exit_t = np.full(uniq.shape, np.inf)
    np.minimum.at(exit_t, inverse, sample_rem)

    e_muon = (uniq // n_vox).astype(np.int64)
    e_voxel = (uniq % n_vox).astype(np.int64)
    return e_muon, e_voxel, length, exit_t


def _muon_data(measurement: Measurement, grid: VoxelGrid) -> np.ndarray:
    """Scattering angle and lateral displacement per muon, per projection.

    Returns ``(n, 2, 2)``: for each muon, the ``[delta_theta, delta_pos]`` pair
    in the x and y projections.
    """
    z_exit = grid.lower[2]
    ip, idir = measurement.in_point, measurement.in_dir
    op, odir = measurement.out_point, measurement.out_dir

    def slope(direction, axis):
        return direction[:, axis] / direction[:, 2]

    def cross(point, direction, axis):
        t = (z_exit - point[:, 2]) / direction[:, 2]
        return point[:, axis] + t * direction[:, axis]

    data = np.zeros((len(measurement), 2, 2))
    for j, axis in enumerate((0, 1)):
        data[:, j, 0] = slope(odir, axis) - slope(idir, axis)
        data[:, j, 1] = cross(op, odir, axis) - cross(ip, idir, axis)
    return data


def mlem_reconstruct(
    measurement: Measurement,
    grid: VoxelGrid,
    n_iterations: int = DEFAULT_ITERATIONS,
    n_steps: int = DEFAULT_STEPS,
    momentum_ref: float = DEFAULT_MOMENTUM_REF,
    max_muons: int = 4096,
) -> np.ndarray:
    """Reconstruct a scattering-density volume by iterative MLEM."""
    m = measurement.select_valid()
    if len(m) > max_muons:
        m = m.head(max_muons)

    e_muon, e_voxel, e_L, e_T = _path_entries(m, grid, n_steps)
    if e_muon.size == 0:
        return np.zeros(grid.shape)

    data = _muon_data(m, grid)  # (n, 2, 2)
    p_ratio2 = (momentum_ref / m.momentum_gev) ** 2

    # Geometry-only weight-matrix components per entry.
    a_tt = e_L
    a_tx = e_L**2 / 2.0 + e_L * e_T
    a_xx = e_L**3 / 3.0 + e_L**2 * e_T + e_L * e_T**2

    n_muons = len(m)
    n_vox = int(np.prod(grid.shape))
    lam = np.zeros(n_vox)
    lam[e_voxel] = 1.0e-2  # seed crossed voxels; the rest stay empty

    d = data  # (n, 2, 2): [:, proj, (theta, pos)]

    for _ in range(n_iterations):
        lam_e = lam[e_voxel]
        s_tt = np.bincount(e_muon, weights=lam_e * a_tt, minlength=n_muons)
        s_tx = np.bincount(e_muon, weights=lam_e * a_tx, minlength=n_muons)
        s_xx = np.bincount(e_muon, weights=lam_e * a_xx, minlength=n_muons)
        s_tt = s_tt * p_ratio2 + ANGLE_ERROR_VAR
        s_tx = s_tx * p_ratio2
        s_xx = s_xx * p_ratio2 + DISPLACEMENT_ERROR_VAR

        det = s_tt * s_xx - s_tx**2
        det = np.where(det > 1e-12, det, 1e-12)
        # Sigma inverse (2x2) per muon.
        m_tt = s_xx / det
        m_tx = -s_tx / det
        m_xx = s_tt / det

        # Gather muon quantities onto entries.
        gtt, gtx, gxx = m_tt[e_muon], m_tx[e_muon], m_xx[e_muon]
        pr2 = p_ratio2[e_muon]

        # A = Sigma^-1 W Sigma^-1 (symmetric 2x2) per entry.
        # First MW = Sigma^-1 W, then A = MW Sigma^-1.
        mw_tt = gtt * a_tt + gtx * a_tx
        mw_tx = gtt * a_tx + gtx * a_xx
        mw_xt = gtx * a_tt + gxx * a_tx
        mw_xx = gtx * a_tx + gxx * a_xx
        a_ap_tt = mw_tt * gtt + mw_tx * gtx
        a_ap_tx = mw_tt * gtx + mw_tx * gxx
        a_ap_xx = mw_xt * gtx + mw_xx * gxx
        trace_mw = mw_tt + mw_xx  # Tr(Sigma^-1 W)

        # D^T A D summed over both projections.
        term1 = np.zeros(e_muon.shape)
        for proj in (0, 1):
            dth = d[e_muon, proj, 0]
            dps = d[e_muon, proj, 1]
            term1 += a_ap_tt * dth**2 + 2 * a_ap_tx * dth * dps + a_ap_xx * dps**2
        c = (term1 - 2.0 * trace_mw) * pr2

        # M-step: per-voxel mean correction, clipped for tail robustness.
        sum_c = np.bincount(e_voxel, weights=c, minlength=n_vox)
        cnt = np.bincount(e_voxel, minlength=n_vox)
        mean_c = np.divide(sum_c, cnt, out=np.zeros_like(sum_c), where=cnt > 0)
        update = lam**2 * mean_c / 2.0
        # Bound the step so a few large-angle muons cannot blow a voxel up, but
        # leave room to grow toward the true density over the iteration budget.
        update = np.clip(update, -lam * 0.8, lam * 2.0)
        lam = np.clip(lam + update, 0.0, None)

    return lam.reshape(grid.shape)
