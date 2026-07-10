"""Export reconstructions as compact static files the browser viewer can fetch.

Each volume becomes an 8-bit raw binary (one byte per voxel, x fastest so it
drops straight into a Three.js Data3DTexture) plus a JSON sidecar. No server,
no live inference: the viewer just downloads these static files. A manifest
ties together the dosage levels and methods the dosage slider swaps between.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..metrics import normalize_volume


def export_volume(volume: np.ndarray, path: str, normalize: bool = True) -> dict:
    """Write a volume as ``<path>.bin`` (uint8) plus ``<path>.json`` header.

    The binary is laid out x-fastest, then y, then z, which is the order a
    Three.js ``Data3DTexture`` expects. Returns the header dict.
    """
    vol = normalize_volume(volume) if normalize else np.clip(volume, 0.0, 1.0)
    nx, ny, nz = vol.shape
    # x-fastest ordering: C-order ravel of the (z, y, x) transpose.
    flat = np.transpose(vol, (2, 1, 0)).ravel(order="C")
    data = np.clip(np.round(flat * 255.0), 0, 255).astype(np.uint8)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    bin_path = out.with_suffix(".bin")
    json_path = out.with_suffix(".json")
    bin_path.write_bytes(data.tobytes())

    header = {
        "shape": [nx, ny, nz],
        "dtype": "uint8",
        "layout": "x-fastest",
        "value_range": [0.0, 1.0],
        "bin": bin_path.name,
    }
    json_path.write_text(json.dumps(header, indent=2))
    return header


def export_manifest(manifest: dict, path: str) -> str:
    """Write the viewer manifest (scenes, dosage levels, method volumes)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    return str(out)
