"""Top-level simulator: scene + detector geometry in, muon tracks out.

The simulator generates cosmic muons, propagates each one through the scene,
applies multiple scattering at an effective vertex, and records the truth: the
incoming and outgoing tracks, the scattering angle, the momentum, and the
ideal (noise-free) crossing points at every detector plane. The detector
module turns those ideal crossings into realistic measured hits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import DetectorGeometry, Scene
from . import flux, scatter, transport


@dataclass
class SimConfig:
    """Knobs for the physics generation (not the detector)."""

    n_steps: int = 64
    max_zenith_deg: float = 45.0
    momentum_median_gev: float = 3.0
    momentum_sigma_log: float = 0.6
    origin_margin_cm: float = 5.0
    batch_multiplier: int = 6  # oversample candidates to hit the accepted target


@dataclass
class MuonTruth:
    """Ground-truth muon tracks and per-plane ideal crossings.

    Array shapes, with ``N`` muons and ``P`` planes per station:
      - ``in_point``, ``in_dir``, ``out_point``, ``out_dir``, ``vertex``: (N, 3)
      - ``scatter_angle``, ``momentum_gev``, ``path_radiation_lengths``: (N,)
      - ``top_hits``, ``bottom_hits``: (N, P, 2)  ideal (x, y) at each plane
      - ``top_plane_z``, ``bottom_plane_z``: (P,)
    """

    in_point: np.ndarray
    in_dir: np.ndarray
    out_point: np.ndarray
    out_dir: np.ndarray
    vertex: np.ndarray
    scatter_angle: np.ndarray
    momentum_gev: np.ndarray
    path_radiation_lengths: np.ndarray
    top_hits: np.ndarray
    bottom_hits: np.ndarray
    top_plane_z: np.ndarray
    bottom_plane_z: np.ndarray

    def __len__(self) -> int:
        return self.in_point.shape[0]


class Simulator:
    """Generate scattered cosmic muons for a fixed scene and detector."""

    def __init__(
        self,
        scene: Scene,
        detector: DetectorGeometry,
        config: SimConfig | None = None,
    ) -> None:
        self.scene = scene
        self.detector = detector
        self.config = config or SimConfig()
        self._inv_x0 = scene.inv_x0_field()

    def _plane_crossings(
        self, point: np.ndarray, direction: np.ndarray, plane_z: np.ndarray
    ) -> np.ndarray:
        """(N, P, 2) x-y crossings of each ray with a set of plane z-values."""
        t = (plane_z[None, :] - point[:, 2:3]) / direction[:, 2:3]
        xy = point[:, None, :2] + t[..., None] * direction[:, None, :2]
        return xy

    def _generate_candidates(self, rng: np.random.Generator, m: int) -> dict:
        cfg = self.config
        det = self.detector
        directions = flux.sample_directions(rng, m, np.deg2rad(cfg.max_zenith_deg))
        momenta = flux.sample_momenta(
            rng, m, cfg.momentum_median_gev, cfg.momentum_sigma_log
        )
        ext = det.plane_half_extent_cm + cfg.origin_margin_cm
        origin_xy = rng.uniform(-ext, ext, size=(m, 2))
        origin_z = np.full((m, 1), det.top_z)
        origins = np.concatenate([origin_xy, origin_z], axis=1)

        x_over_x0, vertex, dense_path = transport.integrate_path(
            self.scene.grid, self._inv_x0, origins, directions, cfg.n_steps
        )
        theta0 = scatter.highland_theta0(x_over_x0, momenta)
        angles, offsets = scatter.sample_scatter(rng, theta0, dense_path)

        u, v = transport.transverse_basis(directions)
        out_dir = directions + angles[:, 0:1] * u + angles[:, 1:2] * v
        out_dir /= np.linalg.norm(out_dir, axis=1, keepdims=True)
        out_point = vertex + offsets[:, 0:1] * u + offsets[:, 1:2] * v

        top_z = det.top_plane_z()
        bottom_z = det.bottom_plane_z()
        top_hits = self._plane_crossings(origins, directions, top_z)
        bottom_hits = self._plane_crossings(out_point, out_dir, bottom_z)

        return {
            "in_point": origins,
            "in_dir": directions,
            "out_point": out_point,
            "out_dir": out_dir,
            "vertex": vertex,
            "scatter_angle": scatter.space_angle(directions, out_dir),
            "momentum_gev": momenta,
            "path_radiation_lengths": x_over_x0,
            "top_hits": top_hits,
            "bottom_hits": bottom_hits,
            "top_plane_z": top_z,
            "bottom_plane_z": bottom_z,
        }

    def _accepted(self, cand: dict) -> np.ndarray:
        """Muons whose innermost and outermost planes both land in acceptance."""
        det = self.detector
        top = cand["top_hits"]
        bot = cand["bottom_hits"]
        keep = (
            det.in_acceptance(top[:, 0])
            & det.in_acceptance(top[:, -1])
            & det.in_acceptance(bot[:, 0])
            & det.in_acceptance(bot[:, -1])
        )
        return keep

    def generate(self, n: int, rng: np.random.Generator) -> MuonTruth:
        """Generate exactly ``n`` accepted muons."""
        cfg = self.config
        collected: dict[str, list[np.ndarray]] = {}
        total = 0
        scalar_keys = {"top_plane_z", "bottom_plane_z"}
        plane_z = {}
        while total < n:
            m = max(int((n - total) * cfg.batch_multiplier), 1024)
            cand = self._generate_candidates(rng, m)
            keep = self._accepted(cand)
            for key, value in cand.items():
                if key in scalar_keys:
                    plane_z[key] = value
                    continue
                collected.setdefault(key, []).append(value[keep])
            total += int(keep.sum())

        result = {key: np.concatenate(parts)[:n] for key, parts in collected.items()}
        result.update(plane_z)
        return MuonTruth(**result)


def simulate(
    scene: Scene,
    detector: DetectorGeometry,
    n: int,
    rng: np.random.Generator,
    config: SimConfig | None = None,
) -> MuonTruth:
    """Convenience wrapper: build a Simulator and generate ``n`` muons."""
    return Simulator(scene, detector, config).generate(n, rng)
