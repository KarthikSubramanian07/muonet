"""Static PNG cross-sections of reconstructed volumes.

Clean central slices for the README and papers. Uses the Agg backend so it
renders headless on CI and Colab with no display.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..metrics import normalize_volume  # noqa: E402

# Muonet's default colormap. "inferno" reads as "denser = hotter", which is the
# right intuition for scattering density.
DEFAULT_CMAP = "inferno"


def _central_slices(volume: np.ndarray) -> list[tuple[str, np.ndarray]]:
    nx, ny, nz = volume.shape
    return [
        ("z", volume[:, :, nz // 2].T),
        ("y", volume[:, ny // 2, :].T),
        ("x", volume[nx // 2, :, :].T),
    ]


def save_cross_sections(
    volume: np.ndarray,
    path: str,
    title: str | None = None,
    cmap: str = DEFAULT_CMAP,
    normalize: bool = True,
) -> str:
    """Save three orthogonal central slices of a volume to ``path``."""
    vol = normalize_volume(volume) if normalize else volume
    slices = _central_slices(vol)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, (label, plane) in zip(axes, slices, strict=True):
        ax.imshow(plane, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(f"{label}-normal slice", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def save_comparison(
    volumes: dict[str, np.ndarray],
    path: str,
    axis: str = "z",
    cmap: str = DEFAULT_CMAP,
    title: str | None = None,
) -> str:
    """Save a row of central slices comparing several named reconstructions."""
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    fig, axes = plt.subplots(1, len(volumes), figsize=(3.0 * len(volumes), 3.2))
    if len(volumes) == 1:
        axes = [axes]
    for ax, (name, volume) in zip(axes, volumes.items(), strict=True):
        vol = normalize_volume(volume)
        mid = vol.shape[axis_index] // 2
        plane = np.take(vol, mid, axis=axis_index).T
        ax.imshow(plane, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(name, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
