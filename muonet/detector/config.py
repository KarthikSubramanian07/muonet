"""Detector configurations, from a perfect idealization to a battery-limited field unit.

The presets are illustrative but grounded in the values real scattering
tomographs run at: plastic-scintillator or GEM tracking planes with ~0.5-2 mm
position resolution, per-plane efficiencies of 85-99%, SiPM timing jitter of a
few nanoseconds, and, critically, no per-muon momentum measurement (a
spectrometer is expensive and heavy, so most deployed scanners simply assume a
spectrum). Every knob here degrades reconstruction in a way you can measure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorConfig:
    """Everything that stands between true muon tracks and what a readout sees.

    Attributes:
        name: Preset label, surfaced in outputs and plots.
        spatial_resolution_cm: Gaussian position smearing per plane hit (sigma).
        efficiency: Probability an individual plane registers a hit (0-1).
        timing_jitter_ns: SiPM timing spread, recorded per hit (informational;
            does not move the geometry, but models the readout honestly).
        noise_rate: Probability a registered hit is a large-offset outlier
            (cross-talk / accidental), which the track fit must survive.
        momentum_known: Whether reconstruction is given the true per-muon
            momentum. Most field scanners cannot measure it, so this is False
            for realistic presets and they fall back to ``assumed_momentum_gev``.
        assumed_momentum_gev: Momentum reconstruction assumes when it is blind
            to the truth.
        min_planes_per_station: Planes that must fire in a station for its track
            to be fittable. Two is the geometric minimum for a straight line.
    """

    name: str
    spatial_resolution_cm: float
    efficiency: float
    timing_jitter_ns: float
    noise_rate: float
    momentum_known: bool
    assumed_momentum_gev: float
    min_planes_per_station: int = 2


DETECTOR_PRESETS: dict[str, DetectorConfig] = {
    # A perfect detector: the upper bound no hardware reaches. The gap between
    # this and the realistic presets is the whole point of the ablation.
    "ideal": DetectorConfig(
        name="ideal",
        spatial_resolution_cm=0.0,
        efficiency=1.0,
        timing_jitter_ns=0.0,
        noise_rate=0.0,
        momentum_known=True,
        assumed_momentum_gev=3.0,
    ),
    # A good laboratory tracker: sub-millimetre resolution, near-perfect
    # efficiency, momentum still known (e.g. a tagged test-beam or a setup with
    # a spectrometer arm).
    "research": DetectorConfig(
        name="research",
        spatial_resolution_cm=0.05,
        efficiency=0.99,
        timing_jitter_ns=1.0,
        noise_rate=0.001,
        momentum_known=True,
        assumed_momentum_gev=3.0,
    ),
    # A deployed SiPM-readout scanner: millimetre strips, high-but-imperfect
    # efficiency, real readout noise, and blind to per-muon momentum.
    "realistic_sipm": DetectorConfig(
        name="realistic_sipm",
        spatial_resolution_cm=0.1,
        efficiency=0.95,
        timing_jitter_ns=2.0,
        noise_rate=0.01,
        momentum_known=False,
        assumed_momentum_gev=3.0,
    ),
    # A passive, battery-limited field unit: coarse strips, planes that drop
    # out, more accidentals, momentum unknown. The constrained regime real
    # deployments actually run in.
    "battery_limited": DetectorConfig(
        name="battery_limited",
        spatial_resolution_cm=0.2,
        efficiency=0.85,
        timing_jitter_ns=5.0,
        noise_rate=0.02,
        momentum_known=False,
        assumed_momentum_gev=3.0,
    ),
}


def get_detector_config(name: str) -> DetectorConfig:
    """Look up a detector preset by name."""
    try:
        return DETECTOR_PRESETS[name]
    except KeyError as exc:
        known = ", ".join(DETECTOR_PRESETS)
        raise KeyError(f"unknown detector preset {name!r}; known: {known}") from exc
