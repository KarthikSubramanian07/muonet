"""Turn noise-free muon truth into measured hits and refit tracks.

The pipeline per muon:
  1. smear each plane hit by the detector's position resolution, and, at the
     configured rate, replace a hit with a large-offset accidental;
  2. drop hits on planes that did not fire (efficiency);
  3. optionally drop the whole muon if its station times fail to coincide
     within the readout window (timing jitter);
  4. refit straight in/out tracks from the surviving hits by least squares;
  5. hand reconstruction either the true momentum or, when the detector is
     blind to it, a single assumed value.

Everything is vectorized over all muons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sim.scatter import space_angle
from ..sim.simulator import MuonTruth
from .config import DetectorConfig

# Coincidence window against which timing jitter is judged (ns). A muon whose
# station hits scatter beyond this are lost as a failed coincidence.
COINCIDENCE_WINDOW_NS = 20.0
# Scale of an accidental / cross-talk hit relative to the true position (cm).
ACCIDENTAL_SIGMA_CM = 1.5


@dataclass
class Measurement:
    """Measured hits and refit tracks for a batch of muons.

    Shapes with ``N`` muons and ``P`` planes per station:
      - ``top_hits``, ``bottom_hits``: (N, P, 2) measured (x, y), NaN if the
        plane did not fire.
      - ``top_mask``, ``bottom_mask``: (N, P) which planes fired.
      - ``in_point``, ``in_dir``, ``out_point``, ``out_dir``: (N, 3) refit tracks.
      - ``scatter_angle``: (N,) measured space angle between in and out tracks.
      - ``momentum_gev``: (N,) momentum available to reconstruction.
      - ``valid``: (N,) muons with enough surviving hits to reconstruct.
    """

    top_hits: np.ndarray
    bottom_hits: np.ndarray
    top_mask: np.ndarray
    bottom_mask: np.ndarray
    in_point: np.ndarray
    in_dir: np.ndarray
    out_point: np.ndarray
    out_dir: np.ndarray
    scatter_angle: np.ndarray
    momentum_gev: np.ndarray
    valid: np.ndarray
    config: DetectorConfig

    def __len__(self) -> int:
        return self.in_point.shape[0]

    def select_valid(self) -> Measurement:
        """Return a copy keeping only reconstructable muons."""
        keep = self.valid
        return Measurement(
            top_hits=self.top_hits[keep],
            bottom_hits=self.bottom_hits[keep],
            top_mask=self.top_mask[keep],
            bottom_mask=self.bottom_mask[keep],
            in_point=self.in_point[keep],
            in_dir=self.in_dir[keep],
            out_point=self.out_point[keep],
            out_dir=self.out_dir[keep],
            scatter_angle=self.scatter_angle[keep],
            momentum_gev=self.momentum_gev[keep],
            valid=self.valid[keep],
            config=self.config,
        )


def fit_line(
    hits: np.ndarray, mask: np.ndarray, plane_z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares straight-line fit through per-plane (x, y) hits.

    Fits ``x = ax + bx * z`` and ``y = ay + by * z`` using only fired planes.

    Returns ``(point, direction, valid)`` where ``point`` is the track's (x, y)
    at z = 0, ``direction`` is a downward unit vector, and ``valid`` marks
    tracks with at least two independent planes.
    """
    z = plane_z[None, :]
    w = mask.astype(np.float64)
    x = np.nan_to_num(hits[..., 0])
    y = np.nan_to_num(hits[..., 1])

    s0 = w.sum(axis=1)
    sz = (w * z).sum(axis=1)
    szz = (w * z * z).sum(axis=1)
    denom = s0 * szz - sz**2
    valid = denom > 1e-9

    safe = np.where(valid, denom, 1.0)

    def slope_intercept(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sv = (w * v).sum(axis=1)
        szv = (w * z * v).sum(axis=1)
        b = (s0 * szv - sz * sv) / safe
        a = (sv - b * sz) / np.where(s0 > 0, s0, 1.0)
        return a, b

    ax, bx = slope_intercept(x)
    ay, by = slope_intercept(y)

    point = np.stack([ax, ay, np.zeros_like(ax)], axis=1)
    direction = np.stack([bx, by, np.ones_like(bx)], axis=1)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    direction *= -1.0  # muons travel downward (negative z)
    return point, direction, valid


def apply_detector(
    truth: MuonTruth, config: DetectorConfig, rng: np.random.Generator
) -> Measurement:
    """Apply a detector configuration to simulated truth."""
    n = len(truth)
    p = truth.top_plane_z.shape[0]

    def measure(true_hits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        smeared = true_hits + rng.normal(0.0, config.spatial_resolution_cm, (n, p, 2))
        if config.noise_rate > 0.0:
            is_outlier = rng.random((n, p)) < config.noise_rate
            outliers = rng.normal(0.0, ACCIDENTAL_SIGMA_CM, (n, p, 2))
            smeared = np.where(is_outlier[..., None], true_hits + outliers, smeared)
        fired = rng.random((n, p)) < config.efficiency
        hits = np.where(fired[..., None], smeared, np.nan)
        return hits, fired

    top_hits, top_fired = measure(truth.top_hits)
    bottom_hits, bottom_fired = measure(truth.bottom_hits)

    in_point, in_dir, top_valid = fit_line(top_hits, top_fired, truth.top_plane_z)
    out_point, out_dir, bottom_valid = fit_line(
        bottom_hits, bottom_fired, truth.bottom_plane_z
    )

    enough = (top_fired.sum(1) >= config.min_planes_per_station) & (
        bottom_fired.sum(1) >= config.min_planes_per_station
    )

    # Timing jitter costs coincidences: wider jitter, more muons lost.
    if config.timing_jitter_ns > 0.0:
        p_coinc = np.exp(-config.timing_jitter_ns / COINCIDENCE_WINDOW_NS)
        coincident = rng.random(n) < p_coinc
    else:
        coincident = np.ones(n, dtype=bool)

    valid = top_valid & bottom_valid & enough & coincident

    if config.momentum_known:
        momentum = truth.momentum_gev.copy()
    else:
        momentum = np.full(n, config.assumed_momentum_gev)

    return Measurement(
        top_hits=top_hits,
        bottom_hits=bottom_hits,
        top_mask=top_fired,
        bottom_mask=bottom_fired,
        in_point=in_point,
        in_dir=in_dir,
        out_point=out_point,
        out_dir=out_dir,
        scatter_angle=space_angle(in_dir, out_dir),
        momentum_gev=momentum,
        valid=valid,
        config=config,
    )
