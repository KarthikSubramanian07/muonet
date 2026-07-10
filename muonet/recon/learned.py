"""The learned two-stage reconstructor: trajectory MLP, then 3D U-Net.

This architecture family is prior work, not a Muonet invention (see mu-Net,
arXiv:2312.17265, and the trajectory-estimation and CNN-voxeliser literature).
What Muonet adds is the clean, reproducible, detector-realistic packaging
around it. We say so plainly.

  - Stage 1 (TrajectoryMLP): a small MLP that refits the incoming and outgoing
    tracks from raw hits. Least squares is already optimal for clean Gaussian
    noise, so the MLP earns its keep specifically when accidental outlier hits
    break that assumption, where it behaves like a learned robust estimator.
  - Stage 2 (VoxelUNet3D): a compact 3D U-Net that maps the sparse, noisy PoCA
    volume to a clean density grid. This is what turns the dosage slider from
    a blob into an object.

Torch is imported only here, so the classical pipeline never depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..detector.effects import Measurement
from ..geometry import VoxelGrid
from ..metrics import normalize_volume
from ..recon.poca import poca_reconstruct
from ..sim.scatter import space_angle

# Fixed scales used to normalize MLP inputs/outputs into a friendly range.
_POS_SCALE_CM = 35.0
_SLOPE_SCALE = 1.5


def resolve_device(device: str | None) -> torch.device:
    """Pick a device, defaulting to CPU for reproducible benchmark numbers."""
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Track parameter conversions
# --------------------------------------------------------------------------- #
def tracks_to_params(point: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Convert (point, downward direction) to [x0, y0, slope_x, slope_y] at z=0."""
    dz = np.where(np.abs(direction[:, 2]) < 1e-9, -1e-9, direction[:, 2])
    t = -point[:, 2] / dz
    x0 = point[:, 0] + t * direction[:, 0]
    y0 = point[:, 1] + t * direction[:, 1]
    slope_x = direction[:, 0] / dz
    slope_y = direction[:, 1] / dz
    return np.stack([x0, y0, slope_x, slope_y], axis=1)


def params_to_tracks(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`tracks_to_params`."""
    x0, y0, sx, sy = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    point = np.stack([x0, y0, np.zeros_like(x0)], axis=1)
    direction = np.stack([sx, sy, np.ones_like(sx)], axis=1)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return point, -direction  # downward


def measurement_features(measurement: Measurement) -> np.ndarray:
    """Flatten raw hits and masks into the MLP input feature vector."""
    top = np.nan_to_num(measurement.top_hits) / _POS_SCALE_CM
    bottom = np.nan_to_num(measurement.bottom_hits) / _POS_SCALE_CM
    n = len(measurement)
    return np.concatenate(
        [
            top.reshape(n, -1),
            measurement.top_mask.astype(np.float32),
            bottom.reshape(n, -1),
            measurement.bottom_mask.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


# --------------------------------------------------------------------------- #
# Stage 1: trajectory MLP
# --------------------------------------------------------------------------- #
class TrajectoryMLP(nn.Module):
    """Predicts refined in/out track parameters from raw detector hits."""

    def __init__(self, n_planes: int, hidden: int = 128):
        super().__init__()
        in_dim = 6 * n_planes  # top xy + top mask + bottom xy + bottom mask
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 8),  # in [x0,y0,sx,sy], out [x0,y0,sx,sy]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _scale_params(params: np.ndarray) -> np.ndarray:
    scale = np.array(
        [_POS_SCALE_CM, _POS_SCALE_CM, _SLOPE_SCALE, _SLOPE_SCALE] * 2, dtype=np.float32
    )
    return params / scale


def _unscale_params(scaled: np.ndarray) -> np.ndarray:
    scale = np.array(
        [_POS_SCALE_CM, _POS_SCALE_CM, _SLOPE_SCALE, _SLOPE_SCALE] * 2, dtype=np.float32
    )
    return scaled * scale


def refine_tracks(model: TrajectoryMLP, measurement: Measurement) -> Measurement:
    """Replace a measurement's fitted tracks with the MLP's predictions."""
    model.eval()
    device = next(model.parameters()).device
    feats = torch.from_numpy(measurement_features(measurement)).to(device)
    with torch.no_grad():
        pred = model(feats).cpu().numpy()
    params = _unscale_params(pred)
    in_point, in_dir = params_to_tracks(params[:, :4])
    out_point, out_dir = params_to_tracks(params[:, 4:])
    refined = Measurement(
        top_hits=measurement.top_hits,
        bottom_hits=measurement.bottom_hits,
        top_mask=measurement.top_mask,
        bottom_mask=measurement.bottom_mask,
        in_point=in_point,
        in_dir=in_dir,
        out_point=out_point,
        out_dir=out_dir,
        scatter_angle=space_angle(in_dir, out_dir),
        momentum_gev=measurement.momentum_gev,
        valid=measurement.valid,
        config=measurement.config,
    )
    return refined


# --------------------------------------------------------------------------- #
# Stage 2: compact 3D U-Net
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = min(8, out_ch)
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VoxelUNet3D(nn.Module):
    """A two-level 3D U-Net mapping a PoCA volume to a density volume.

    Downsampling uses strided convolutions and upsampling uses nearest-neighbor
    interpolation. Both are supported on CPU, CUDA, and Apple MPS (3D max-pool
    is not implemented on MPS), so the same model trains on any of them.
    """

    def __init__(self, base: int = 8):
        super().__init__()
        self.base = base
        self.enc1 = ConvBlock(1, base)
        self.down1 = nn.Conv3d(base, base, 3, stride=2, padding=1)
        self.enc2 = ConvBlock(base, base * 2)
        self.down2 = nn.Conv3d(base * 2, base * 2, 3, stride=2, padding=1)
        self.bottleneck = ConvBlock(base * 2, base * 4)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.dec1 = ConvBlock(base * 2 + base, base)
        self.head = nn.Conv3d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))
        d2 = self.dec2(torch.cat([self.up(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return torch.sigmoid(self.head(d1))


@dataclass
class LearnedReconstructor:
    """The full learned pipeline: optional MLP track refinement, then U-Net."""

    unet: VoxelUNet3D
    trajectory: TrajectoryMLP | None = None
    device: torch.device | None = None

    def reconstruct(self, measurement: Measurement, grid: VoxelGrid) -> np.ndarray:
        device = self.device or next(self.unet.parameters()).device
        if self.trajectory is not None:
            measurement = refine_tracks(self.trajectory, measurement)
        volume = normalize_volume(poca_reconstruct(measurement, grid))
        x = torch.from_numpy(volume.astype(np.float32))[None, None].to(device)
        self.unet.eval()
        with torch.no_grad():
            pred = self.unet(x).cpu().numpy()[0, 0]
        return pred
