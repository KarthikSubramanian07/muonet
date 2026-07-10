"""Analytic cosmic-muon scattering simulator.

Pure NumPy, pip-installable, deterministic from a seed. This is the default
data source for the whole toolkit and the only backend CI depends on. A heavy
Geant4 backend is intentionally out of scope for the core pipeline.
"""

from .flux import sample_directions, sample_momenta, zenith_cdf_max
from .scatter import highland_theta0
from .simulator import MuonTruth, SimConfig, Simulator, simulate

__all__ = [
    "MuonTruth",
    "Simulator",
    "SimConfig",
    "simulate",
    "sample_directions",
    "sample_momenta",
    "zenith_cdf_max",
    "highland_theta0",
]
