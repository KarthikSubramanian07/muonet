"""Training loops for the learned two-stage model.

Everything here is sized to finish on a laptop CPU or a free Colab T4 in
minutes, on a fixed seed. There is no required run that needs paid compute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .data import build_training_set
from .detector import get_detector_config
from .recon.learned import (
    LearnedReconstructor,
    TrajectoryMLP,
    VoxelUNet3D,
    _scale_params,
    _unscale_params,
    measurement_features,
    params_to_tracks,
    refine_tracks,
    resolve_device,
    tracks_to_params,
)
from .scenes import default_detector, default_grid, random_scene
from .seeding import seed_everything
from .sim import simulate
from .sim.scatter import space_angle


@dataclass
class TrainConfig:
    """Sizes for the full learned pipeline. Defaults reproduce the benchmarks."""

    mlp_scenes: int = 30
    mlp_epochs: int = 40
    unet_scenes: int = 96
    unet_epochs: int = 20
    unet_base: int = 8
    batch_size: int = 4
    lr: float = 1.5e-3
    detector: str = "realistic_sipm"
    device: str = "cpu"
    # Train on a range of exposures up to this many muons. Evaluation may use
    # more (a higher-dosage, cleaner input is only easier for the model).
    train_dosages: tuple[int, ...] = (128, 256, 512, 1024, 2048)


# --------------------------------------------------------------------------- #
# Stage 1: trajectory MLP
# --------------------------------------------------------------------------- #
def collect_mlp_dataset(
    n_scenes: int, rng: np.random.Generator, detector_name: str, dosage: int = 2048
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather per-muon (features, target params, least-squares params)."""
    grid = default_grid()
    geom = default_detector()
    config = get_detector_config(detector_name)
    feats, targets, lsq = [], [], []
    for _ in range(n_scenes):
        scene = random_scene(rng, grid)
        truth = simulate(scene, geom, dosage, rng)
        from .detector import apply_detector

        meas = apply_detector(truth, config, rng)
        v = meas.valid
        feats.append(measurement_features(meas)[v])
        in_p = tracks_to_params(truth.in_point[v], truth.in_dir[v])
        out_p = tracks_to_params(truth.out_point[v], truth.out_dir[v])
        targets.append(np.concatenate([in_p, out_p], axis=1).astype(np.float32))
        lin = tracks_to_params(meas.in_point[v], meas.in_dir[v])
        lout = tracks_to_params(meas.out_point[v], meas.out_dir[v])
        lsq.append(np.concatenate([lin, lout], axis=1).astype(np.float32))
    return (
        np.concatenate(feats),
        np.concatenate(targets),
        np.concatenate(lsq),
    )


def train_trajectory_mlp(
    config: TrainConfig, rng: np.random.Generator, verbose: bool = True
) -> tuple[TrajectoryMLP, dict[str, float]]:
    """Train the stage-1 MLP and report its track-angle error against LSQ."""
    device = resolve_device(config.device)
    feats, targets, lsq = collect_mlp_dataset(config.mlp_scenes, rng, config.detector)
    n = len(feats)
    split = int(0.85 * n)
    perm = rng.permutation(n)
    train_idx, val_idx = perm[:split], perm[split:]

    x = torch.from_numpy(feats).to(device)
    y = torch.from_numpy(_scale_params(targets)).to(device)

    model = TrajectoryMLP(n_planes=default_detector().n_planes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()

    idx = torch.from_numpy(train_idx).to(device)
    iterator = tqdm(range(config.mlp_epochs), disable=not verbose, desc="stage1 mlp")
    for _ in iterator:
        model.train()
        perm_t = idx[torch.randperm(len(idx), device=device)]
        for start in range(0, len(perm_t), 4096):
            b = perm_t[start : start + 4096]
            opt.zero_grad()
            loss = loss_fn(model(x[b]), y[b])
            loss.backward()
            opt.step()

    stats = _mlp_track_error(model, feats[val_idx], targets[val_idx], lsq[val_idx], device)
    return model, stats


def _track_angle_error(params: np.ndarray, truth_params: np.ndarray) -> float:
    """Mean in+out track angular error (mrad) between two parameter sets."""
    _, d_in = params_to_tracks(params[:, :4])
    _, d_out = params_to_tracks(params[:, 4:])
    _, t_in = params_to_tracks(truth_params[:, :4])
    _, t_out = params_to_tracks(truth_params[:, 4:])
    err = 0.5 * (space_angle(d_in, t_in) + space_angle(d_out, t_out))
    return float(np.mean(err) * 1e3)


def _mlp_track_error(model, feats, targets, lsq, device) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(feats).to(device)).cpu().numpy()
    mlp_params = _unscale_params(pred)
    return {
        "mlp_track_error_mrad": _track_angle_error(mlp_params, targets),
        "lsq_track_error_mrad": _track_angle_error(lsq, targets),
    }


# --------------------------------------------------------------------------- #
# Stage 2: 3D U-Net
# --------------------------------------------------------------------------- #
def train_unet(
    inputs: np.ndarray,
    targets: np.ndarray,
    config: TrainConfig,
    seed: int = 0,
    verbose: bool = True,
) -> VoxelUNet3D:
    """Train the stage-2 U-Net with a foreground-weighted MSE loss."""
    device = resolve_device(config.device)
    seed_everything(seed)
    x = torch.from_numpy(inputs)
    y = torch.from_numpy(targets)

    model = VoxelUNet3D(base=config.unet_base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    n = x.shape[0]

    iterator = tqdm(range(config.unet_epochs), disable=not verbose, desc="stage2 unet")
    for _ in iterator:
        model.train()
        order = torch.randperm(n)
        for start in range(0, n, config.batch_size):
            b = order[start : start + config.batch_size]
            xb = x[b].to(device)
            yb = y[b].to(device)
            opt.zero_grad()
            pred = model(xb)
            # Weight dense voxels so the model cannot win by predicting all air.
            weight = 1.0 + 5.0 * yb
            loss = torch.mean(weight * (pred - yb) ** 2)
            loss.backward()
            opt.step()
    return model


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def build_learned(
    config: TrainConfig | None = None,
    seed: int = 0,
    verbose: bool = True,
    mlp_in_pipeline: bool = False,
) -> tuple[LearnedReconstructor, dict[str, float]]:
    """Train both stages and return the reconstructor plus training stats.

    By default the U-Net headline path runs on robust least-squares PoCA, and
    the stage-1 MLP is trained and reported on the side (least squares is
    already optimal for clean Gaussian noise, so the MLP is an honest, modest,
    outlier-robustness stage, not a free win). Set ``mlp_in_pipeline`` to route
    MLP-refined tracks into the U-Net for the full two-stage system.
    """
    config = config or TrainConfig()
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    device = resolve_device(config.device)

    mlp, stats = train_trajectory_mlp(config, rng, verbose=verbose)
    pipeline_mlp = mlp if mlp_in_pipeline else None
    refine = (lambda m: refine_tracks(mlp, m)) if mlp_in_pipeline else None  # noqa: E731
    inputs, targets = build_training_set(
        config.unet_scenes,
        rng,
        dosages=list(config.train_dosages),
        detector_name=config.detector,
        refine=refine,
    )
    unet = train_unet(inputs, targets, config, seed=seed, verbose=verbose)

    reconstructor = LearnedReconstructor(
        unet=unet, trajectory=pipeline_mlp, device=device
    )
    return reconstructor, stats


def save_checkpoint(reconstructor: LearnedReconstructor, path: str) -> None:
    torch.save(
        {
            "unet": reconstructor.unet.state_dict(),
            "unet_base": reconstructor.unet.base,
            "trajectory": (
                reconstructor.trajectory.state_dict()
                if reconstructor.trajectory is not None
                else None
            ),
            "n_planes": default_detector().n_planes,
        },
        path,
    )


def load_checkpoint(path: str, device: str = "cpu") -> LearnedReconstructor:
    dev = resolve_device(device)
    ckpt = torch.load(path, map_location=dev, weights_only=True)
    unet = VoxelUNet3D(base=ckpt.get("unet_base", 8)).to(dev)
    unet.load_state_dict(ckpt["unet"])
    trajectory = None
    if ckpt.get("trajectory") is not None:
        trajectory = TrajectoryMLP(n_planes=ckpt["n_planes"]).to(dev)
        trajectory.load_state_dict(ckpt["trajectory"])
    return LearnedReconstructor(unet=unet, trajectory=trajectory, device=dev)
