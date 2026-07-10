"""Multiple-scattering formula checks against hand calculation and PDG behaviour."""

import numpy as np

from muonet.sim import highland_theta0
from muonet.sim.scatter import sample_scatter, space_angle


def test_highland_known_value():
    """Hand calculation: x/X0 = 1, p = 1 GeV/c gives ~13.7 mrad plane RMS."""
    theta0 = highland_theta0(np.array([1.0]), np.array([1.0]))
    assert np.isclose(theta0[0], 0.01368, rtol=0.02)


def test_highland_scales_with_thickness_and_momentum():
    """theta_0 grows like sqrt(x/X0) and falls like 1/p."""
    # sqrt(4) = 2, nudged above 2 by the growing log correction term.
    t = highland_theta0(np.array([1.0, 4.0]), np.array([3.0, 3.0]))
    assert 2.0 < t[1] / t[0] < 2.2

    # Halving momentum doubles the scattering angle.
    p = highland_theta0(np.array([2.0, 2.0]), np.array([1.0, 2.0]))
    assert np.isclose(p[0] / p[1], 2.0, rtol=0.05)


def test_highland_zero_for_no_material():
    assert highland_theta0(np.array([0.0]), np.array([3.0]))[0] == 0.0


def test_sample_scatter_rms_matches_theta0():
    """Sampled projected angles must have RMS close to theta_0."""
    rng = np.random.default_rng(0)
    n = 200_000
    theta0 = np.full(n, 0.02)
    path = np.full(n, 5.0)
    angles, offsets = sample_scatter(rng, theta0, path)
    assert np.isclose(angles.std(), 0.02, rtol=0.02)
    assert angles.shape == (n, 2)
    assert offsets.shape == (n, 2)


def test_sample_scatter_angle_offset_correlation():
    """PDG angle/offset correlation is sqrt(3)/2 ~ 0.866."""
    rng = np.random.default_rng(1)
    n = 500_000
    theta0 = np.full(n, 0.01)
    path = np.full(n, 3.0)
    angles, offsets = sample_scatter(rng, theta0, path)
    corr = np.corrcoef(angles[:, 0], offsets[:, 0])[0, 1]
    assert np.isclose(corr, np.sqrt(3) / 2, atol=0.02)


def test_space_angle():
    a = np.array([[0.0, 0.0, -1.0]])
    b = np.array([[0.0, 0.0, -1.0]])
    assert np.isclose(space_angle(a, b)[0], 0.0)
    c = np.array([[1.0, 0.0, 0.0]])
    assert np.isclose(space_angle(a, c)[0], np.pi / 2)
