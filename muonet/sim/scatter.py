"""Multiple Coulomb scattering: the Highland / Lynch-Dahl formula.

When a charged particle crosses material of thickness ``x`` and radiation
length ``X0``, its direction fuzzes out by a roughly Gaussian core with plane
RMS angle (PDG "Passage of particles through matter"):

    theta_0 = (13.6 MeV / (beta c p)) * z * sqrt(x/X0) * [1 + 0.038 * ln(x z^2 / (X0 beta^2))]

For cosmic muons, charge ``z = 1`` and ``beta ~ 1`` at the few-GeV momenta that
dominate the flux. We keep the full beta and log terms anyway so the model
stays honest at low momentum, where scattering is strongest and reconstruction
is easiest.

We use the current PDG 2024 log argument ``x z^2 / (X0 beta^2)`` (Eq. 34.16),
which supersedes the older ``x / X0`` argument. The Gaussian core it describes
covers the central ~98% of the projected angular distribution; older editions
(and Lynch & Dahl 1991) quote ~11% accuracy for 1e-3 < x/X0 < 100. Reference:
Lynch & Dahl, Nucl. Instrum. Meth. B58, 6 (1991); PDG Review of Particle
Physics 2024, "Passage of particles through matter".
"""

from __future__ import annotations

import numpy as np

from .flux import beta_from_momentum

HIGHLAND_MEV = 13.6  # MeV, the Highland constant


def highland_theta0(
    x_over_x0: np.ndarray,
    momentum_gev: np.ndarray,
    charge: float = 1.0,
) -> np.ndarray:
    """Plane RMS scattering angle (radians) from thickness in radiation lengths.

    Args:
        x_over_x0: Dimensionless thickness x / X0 (>= 0).
        momentum_gev: Muon momentum in GeV/c.
        charge: Particle charge in units of e (1 for muons).

    Returns:
        theta_0 in radians, zero wherever ``x_over_x0`` is non-positive.
    """
    x_over_x0 = np.asarray(x_over_x0, dtype=np.float64)
    p_mev = np.asarray(momentum_gev, dtype=np.float64) * 1.0e3
    beta = beta_from_momentum(p_mev)

    safe = x_over_x0 > 0.0
    t = np.where(safe, x_over_x0, 1.0)  # dummy value where unsafe, masked below
    log_arg = t * charge**2 / (beta**2)
    bracket = 1.0 + 0.038 * np.log(log_arg)
    bracket = np.clip(bracket, 0.0, None)  # never let the correction go negative
    theta0 = (HIGHLAND_MEV / (beta * p_mev)) * abs(charge) * np.sqrt(t) * bracket
    return np.where(safe, theta0, 0.0)


def sample_scatter(
    rng: np.random.Generator,
    theta0: np.ndarray,
    path_length_cm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample projected scattering angles and lateral offsets (PDG correlated form).

    For each of the two transverse axes, PDG gives correlated samples from two
    independent standard normals ``z1, z2`` (correlation sqrt(3)/2):

        angle  = z2 * theta_0
        offset = z1 * x * theta_0 / sqrt(12) + z2 * x * theta_0 / 2

    where ``x`` is the traversed thickness. Returns ``(angles, offsets)`` each
    of shape ``(n, 2)`` for the two transverse axes.
    """
    n = theta0.shape[0]
    z1 = rng.standard_normal((n, 2))
    z2 = rng.standard_normal((n, 2))
    theta0 = theta0[:, None]
    x = path_length_cm[:, None]
    angles = z2 * theta0
    offsets = z1 * x * theta0 / np.sqrt(12.0) + z2 * x * theta0 / 2.0
    return angles, offsets


def space_angle(in_dir: np.ndarray, out_dir: np.ndarray) -> np.ndarray:
    """Total (space) scattering angle in radians between two unit directions."""
    dots = np.clip(np.sum(in_dir * out_dir, axis=-1), -1.0, 1.0)
    return np.arccos(dots)


__all__ = ["highland_theta0", "sample_scatter", "space_angle"]
