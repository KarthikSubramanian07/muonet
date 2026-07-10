"""Material properties that muon scattering actually cares about.

Muon multiple scattering is governed by the radiation length ``X0`` of the
material it passes through, not by mass density. A short radiation length
(tungsten, lead, uranium) scatters muons hard; a long one (air, water) barely
touches them. That is the whole physical basis of scattering tomography, and
it is why the reconstructed quantity in Muonet is the inverse radiation length
``1/X0`` rather than g/cm^3.

Radiation-length values are from the Particle Data Group material tables
(pdg.lbl.gov, "Atomic and nuclear properties of materials"). They are quoted
in centimeters so every length in the pipeline stays in one unit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """A material and the one number muons respond to: its radiation length.

    Attributes:
        name: Human-readable label used in configs and plots.
        z_eff: Effective atomic number, kept for context and high-Z labelling.
        radiation_length_cm: X0 in centimeters (PDG values).
        density_g_cm3: Mass density, informational only.
    """

    name: str
    z_eff: float
    radiation_length_cm: float
    density_g_cm3: float

    @property
    def inv_x0(self) -> float:
        """Inverse radiation length in cm^-1. This is what the tomography sees."""
        return 1.0 / self.radiation_length_cm


# PDG radiation lengths (cm). Air is effectively transparent; tungsten is the
# hardest scatterer we ship as a default target.
MATERIALS: dict[str, Material] = {
    "vacuum": Material("vacuum", 0.0, 1.0e12, 0.0),
    "air": Material("air", 7.3, 3.039e4, 1.205e-3),
    "water": Material("water", 7.4, 36.08, 1.000),
    "concrete": Material("concrete", 11.0, 11.55, 2.300),
    "aluminum": Material("aluminum", 13.0, 8.897, 2.699),
    "iron": Material("iron", 26.0, 1.757, 7.874),
    "lead": Material("lead", 82.0, 0.5612, 11.35),
    "tungsten": Material("tungsten", 74.0, 0.3504, 19.30),
    "uranium": Material("uranium", 92.0, 0.3166, 18.95),
}

# Reference scatterer for normalizing scattering density into [0, 1]. Tungsten
# is the densest common target, so 1/X0_tungsten maps to 1.0 and everything
# else falls below it. Fixed across scenes so numbers stay comparable.
REFERENCE_MATERIAL = "tungsten"
REFERENCE_INV_X0 = MATERIALS[REFERENCE_MATERIAL].inv_x0

# A material is "high-Z" (the security-relevant regime) above this effective Z.
HIGH_Z_THRESHOLD = 72.0


def get_material(name: str) -> Material:
    """Look up a material by name, with a helpful error if it is unknown."""
    try:
        return MATERIALS[name]
    except KeyError as exc:
        known = ", ".join(sorted(MATERIALS))
        raise KeyError(f"unknown material {name!r}; known materials: {known}") from exc


def normalized_inv_x0(name: str) -> float:
    """Scattering density of a material on the shared [0, 1] scale."""
    return get_material(name).inv_x0 / REFERENCE_INV_X0


def is_high_z(name: str) -> bool:
    """Whether a material sits in the high-Z regime nuclear security cares about."""
    return get_material(name).z_eff >= HIGH_Z_THRESHOLD
