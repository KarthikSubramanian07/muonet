"""Detector realism: what actually reaches a readout, not idealized physics.

Every academic ML-muon repo assumes a perfect detector. Real scanners do not
have one. This layer turns the simulator's noise-free truth into measured hits
and refit tracks under configurable, field-grounded imperfections:

  - finite spatial resolution (strip/pixel pitch)
  - detection efficiency below 100%
  - SiPM readout noise and timing jitter
  - the unknown-momentum regime (most scanners cannot measure per-muon p)

The degradation these cause, reconstructed and plotted, is one of Muonet's
headline results: it tells a practitioner what resolution they actually need.
"""

from .config import DETECTOR_PRESETS, DetectorConfig, get_detector_config
from .effects import Measurement, apply_detector

__all__ = [
    "DetectorConfig",
    "DETECTOR_PRESETS",
    "get_detector_config",
    "Measurement",
    "apply_detector",
]
