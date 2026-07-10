"""Cosmic-ray muon flux at sea level: directions and momenta.

Two facts drive everything downstream:

1. Muons arrive from above with a zenith distribution close to ``cos^2(theta)``
   per unit solid angle (the classic sea-level result). The *rate through a
   horizontal detector plane* picks up an extra ``cos(theta)`` from projected
   area, so muons crossing our top station follow ``cos^3(theta) sin(theta)``
   in zenith. We sample that directly by inverting its CDF.

2. Their momenta peak at a few GeV/c with a long high-momentum tail. We model
   this with a log-normal whose median is ~3 GeV/c, which reproduces the
   commonly quoted sea-level mean of ~3-4 GeV/c (PDG Cosmic Ray review). It is
   an approximation, and we say so: the point is a realistic momentum *spread*,
   since real scanners rarely measure per-muon momentum.
"""

from __future__ import annotations

import numpy as np

# Muon rest mass, MeV/c^2 (PDG).
MUON_MASS_MEV = 105.6583745


def zenith_cdf_max(max_zenith_rad: float) -> float:
    """Normalization constant 1 - cos^4(theta_max) for the zenith sampler."""
    return 1.0 - np.cos(max_zenith_rad) ** 4


def sample_directions(
    rng: np.random.Generator,
    n: int,
    max_zenith_rad: float = np.deg2rad(70.0),
) -> np.ndarray:
    """Sample ``n`` downward unit directions crossing a horizontal plane.

    Zenith is drawn from ``cos^3(theta) sin(theta)`` via inverse-CDF sampling;
    azimuth is uniform. Returns an ``(n, 3)`` array of unit vectors pointing
    down (negative z).
    """
    u = rng.random(n)
    norm = zenith_cdf_max(max_zenith_rad)
    cos_theta = (1.0 - u * norm) ** 0.25
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta**2, 0.0, 1.0))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return np.stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), -cos_theta],
        axis=-1,
    )


def sample_momenta(
    rng: np.random.Generator,
    n: int,
    median_gev: float = 3.0,
    sigma_log: float = 0.6,
    min_gev: float = 0.3,
    max_gev: float = 100.0,
) -> np.ndarray:
    """Sample ``n`` muon momenta (GeV/c) from a clipped log-normal spectrum."""
    mu = np.log(median_gev)
    p = rng.lognormal(mean=mu, sigma=sigma_log, size=n)
    return np.clip(p, min_gev, max_gev)


def beta_from_momentum(momentum_mev: np.ndarray) -> np.ndarray:
    """Relativistic beta = p / E for muons of a given momentum (MeV/c)."""
    energy = np.sqrt(momentum_mev**2 + MUON_MASS_MEV**2)
    return momentum_mev / energy
