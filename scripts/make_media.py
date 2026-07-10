"""Generate the README dosage-slider GIF and the social share (OG) image.

    python scripts/make_media.py

Renders the tungsten-cube reconstruction at increasing muon counts and stitches
the frames into a looping GIF that shows a blur resolving into a cube, then
composes a 1200x630 Open Graph image. Uses the committed checkpoint if present
(learned model), otherwise falls back to PoCA.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running as a plain script (python scripts/make_media.py) by putting the
# repo root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from muonet.data import collect_measurement
from muonet.detector import get_detector_config
from muonet.metrics import normalize_volume
from muonet.recon import poca_reconstruct
from muonet.scenes import build_scene, default_detector, default_grid

ROOT = Path(__file__).resolve().parent.parent
DOSAGES = [128, 256, 512, 1024, 4096, 8192]
CKPT = ROOT / "checkpoints" / "muonet.pt"


def _reconstructor():
    if CKPT.exists():
        from muonet.train import load_checkpoint

        rec = load_checkpoint(str(CKPT), device="cpu")
        return lambda meas, grid: rec.reconstruct(meas, grid), "learned"
    return lambda meas, grid: poca_reconstruct(meas, grid), "poca"


def _slice_frame(volume: np.ndarray, dosage: int, method: str) -> Image.Image:
    vol = normalize_volume(volume)
    plane = vol[:, :, vol.shape[2] // 2].T
    fig, ax = plt.subplots(figsize=(4, 4), dpi=110)
    ax.imshow(plane, origin="lower", cmap="inferno", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.5, 1.02, f"{dosage} muons", transform=ax.transAxes, ha="center",
        color="#eee", fontsize=13, fontweight="bold",
    )
    fig.patch.set_facecolor("#0a0a12")
    fig.tight_layout()
    fig.canvas.draw()
    img = Image.frombytes("RGBA", fig.canvas.get_width_height(), bytes(fig.canvas.buffer_rgba()))
    plt.close(fig)
    return img.convert("RGB")


def main() -> None:
    grid = default_grid()
    detector = default_detector()
    scene = build_scene("tungsten_cube", grid)
    reconstruct, method = _reconstructor()
    config = get_detector_config("realistic_sipm")

    frames = []
    for dosage in DOSAGES:
        rng = np.random.default_rng(0)
        meas = collect_measurement(scene, detector, config, dosage, rng)
        vol = reconstruct(meas, grid)
        frames.append(_slice_frame(vol, dosage, method))
        print(f"rendered {dosage} muons")

    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    gif = assets / "dosage-slider.gif"
    # Hold each frame, pause longer on the sharpest one, then loop.
    durations = [700] * (len(frames) - 1) + [1600]
    frames[0].save(
        gif, save_all=True, append_images=frames[1:], duration=durations, loop=0
    )
    print(f"wrote {gif}")

    # Open Graph image: the sharpest reconstruction beside a title.
    og = assets / "og.png"
    _make_og(frames[-1], og, method)
    # The viewer serves it from public/.
    viewer_og = ROOT / "viewer" / "public" / "og.png"
    viewer_og.parent.mkdir(parents=True, exist_ok=True)
    Image.open(og).save(viewer_og)
    print(f"wrote {og} and {viewer_og}")


def _make_og(recon: Image.Image, path: Path, method: str) -> None:
    W, H = 1200, 630
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("#0a0a12")
    ax_img = fig.add_axes([0.55, 0.12, 0.4, 0.76])
    ax_img.imshow(recon.resize((420, 420)))
    ax_img.axis("off")
    fig.text(0.06, 0.66, "Muonet", color="#ffffff", fontsize=58, fontweight="bold")
    fig.text(0.065, 0.55, "Cosmic rays in, 3D interiors out.", color="#b197fc", fontsize=24)
    fig.text(
        0.065, 0.30,
        "Open-source muon scattering tomography:\nsimulate, reconstruct, and inspect dense\nobjects with real detector physics.",
        color="#9a9cc0", fontsize=17, linespacing=1.5,
    )
    fig.savefig(path, facecolor="#0a0a12")
    plt.close(fig)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
