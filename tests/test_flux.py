"""Cosmic-ray flux: zenith distribution and momentum spectrum."""

import numpy as np

from muonet.sim import sample_directions, sample_momenta
from muonet.sim.flux import beta_from_momentum


def test_directions_point_down_and_unit():
    rng = np.random.default_rng(0)
    d = sample_directions(rng, 10_000)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-6)
    assert np.all(d[:, 2] < 0.0)  # every muon is downward-going


def test_zenith_respects_max():
    rng = np.random.default_rng(0)
    max_zen = np.deg2rad(40.0)
    d = sample_directions(rng, 50_000, max_zenith_rad=max_zen)
    zenith = np.arccos(-d[:, 2])
    assert zenith.max() <= max_zen + 1e-6


def test_zenith_intensity_peaks_at_vertical():
    """Per unit solid angle the flux peaks at vertical: density in cos(theta)
    goes as cos^3(theta), so an equal-width cos band near vertical holds far
    more muons than one lower down. (The count per unit *angle* peaks near 30
    degrees because of the sin(theta) solid-angle factor, which is expected.)"""
    rng = np.random.default_rng(0)
    d = sample_directions(rng, 400_000, max_zenith_rad=np.deg2rad(70.0))
    cos_theta = -d[:, 2]
    near_vertical = np.mean((cos_theta >= 0.9) & (cos_theta <= 1.0))
    lower = np.mean((cos_theta >= 0.5) & (cos_theta <= 0.6))
    assert near_vertical > 3.0 * lower  # ~ (0.95/0.55)^3 = 5x expected


def test_momentum_spectrum_mean():
    """Sea-level mean momentum lands in the published ~3-4 GeV/c band."""
    rng = np.random.default_rng(0)
    p = sample_momenta(rng, 500_000)
    assert 3.0 < p.mean() < 4.5
    assert np.isclose(np.median(p), 3.0, rtol=0.05)
    assert p.min() >= 0.3


def test_beta_relativistic():
    """A few-GeV muon is ultra-relativistic (beta close to 1)."""
    beta = beta_from_momentum(np.array([3000.0]))  # 3 GeV/c in MeV/c
    assert 0.999 < beta[0] < 1.0
