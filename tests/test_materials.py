"""Material physics: radiation lengths and the high-Z labelling."""

import numpy as np

from muonet.materials import (
    MATERIALS,
    REFERENCE_INV_X0,
    get_material,
    is_high_z,
    normalized_inv_x0,
)


def test_radiation_length_ordering():
    """Denser scatterers must have shorter radiation lengths."""
    x0 = {name: m.radiation_length_cm for name, m in MATERIALS.items()}
    assert x0["tungsten"] < x0["lead"] < x0["iron"] < x0["aluminum"]
    assert x0["aluminum"] < x0["water"] < x0["air"]


def test_pdg_values_are_sane():
    """A few PDG radiation lengths, to a couple percent."""
    assert np.isclose(MATERIALS["lead"].radiation_length_cm, 0.5612, rtol=0.02)
    assert np.isclose(MATERIALS["iron"].radiation_length_cm, 1.757, rtol=0.02)
    assert np.isclose(MATERIALS["water"].radiation_length_cm, 36.08, rtol=0.02)


def test_normalized_density_reference():
    """Tungsten sits at 1.0; air is essentially transparent."""
    assert np.isclose(normalized_inv_x0("tungsten"), 1.0)
    assert normalized_inv_x0("air") < 1e-3
    assert 0.0 < normalized_inv_x0("iron") < normalized_inv_x0("lead") < 1.0


def test_high_z_labelling():
    assert is_high_z("tungsten")
    assert is_high_z("lead")
    assert not is_high_z("iron")
    assert not is_high_z("aluminum")


def test_reference_inv_x0_matches_tungsten():
    assert np.isclose(REFERENCE_INV_X0, get_material("tungsten").inv_x0)
