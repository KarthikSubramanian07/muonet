"""Muonet: cosmic rays in, 3D interiors out.

A reproducible, detector-realistic muon scattering tomography toolkit. Simulate
cosmic muons passing through dense objects, reconstruct the 3D material
distribution from how they scatter, and inspect the result in the browser.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .geometry import DetectorGeometry, Primitive, Scene, VoxelGrid
from .materials import MATERIALS, Material, get_material

__all__ = [
    "__version__",
    "MATERIALS",
    "Material",
    "get_material",
    "VoxelGrid",
    "Scene",
    "Primitive",
    "DetectorGeometry",
]
