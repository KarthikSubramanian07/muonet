"""Learned model: shapes, a tiny training step, and checkpoint round-trips.

Kept small so the whole file runs in a few seconds on CPU (this is also what CI
exercises for the learned path).
"""

import numpy as np
import torch

from muonet.detector import apply_detector, get_detector_config
from muonet.geometry import DetectorGeometry, Primitive, Scene, VoxelGrid
from muonet.recon.learned import (
    LearnedReconstructor,
    TrajectoryMLP,
    VoxelUNet3D,
    params_to_tracks,
    refine_tracks,
    tracks_to_params,
)
from muonet.seeding import make_rng
from muonet.sim import simulate
from muonet.train import TrainConfig, load_checkpoint, save_checkpoint, train_unet


def test_unet_forward_shape_and_range():
    model = VoxelUNet3D(base=8)
    x = torch.rand(2, 1, 16, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_mlp_forward_shape():
    model = TrajectoryMLP(n_planes=4)
    x = torch.rand(5, 24)
    assert model(x).shape == (5, 8)


def test_track_param_round_trip():
    rng = np.random.default_rng(0)
    point = rng.uniform(-10, 10, (20, 3))
    direction = rng.normal(size=(20, 3))
    direction[:, 2] = -np.abs(direction[:, 2]) - 0.5  # downward
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    params = tracks_to_params(point, direction)
    _, d2 = params_to_tracks(params)
    # Direction recovers up to the shared normalization.
    assert np.allclose(d2, direction, atol=1e-6)


def test_train_unet_runs_and_reduces_loss():
    rng = np.random.default_rng(0)
    n = 6
    y = np.zeros((n, 1, 16, 16, 16), np.float32)
    y[:, 0, 6:10, 6:10, 6:10] = 1.0  # a little cube target
    x = (y + rng.normal(0, 0.2, y.shape)).clip(0, 1).astype(np.float32)
    cfg = TrainConfig(unet_epochs=3, batch_size=3, device="cpu")
    model = train_unet(x, y, cfg, seed=0, verbose=False)
    with torch.no_grad():
        pred = model(torch.from_numpy(x))
    # The model should put more signal in the cube than outside it.
    cube = pred[:, 0, 6:10, 6:10, 6:10].mean()
    outside = pred[:, 0, :4, :4, :4].mean()
    assert cube > outside


def test_checkpoint_round_trip(tmp_path):
    unet = VoxelUNet3D(base=8)
    mlp = TrajectoryMLP(n_planes=4)
    rec = LearnedReconstructor(unet=unet, trajectory=mlp)
    path = str(tmp_path / "model.pt")
    save_checkpoint(rec, path)
    loaded = load_checkpoint(path, device="cpu")
    assert isinstance(loaded.unet, VoxelUNet3D)
    assert isinstance(loaded.trajectory, TrajectoryMLP)


def test_reconstruct_returns_valid_volume():
    grid = VoxelGrid(shape=(16, 16, 16), size_cm=(32, 32, 32))
    scene = Scene.from_primitives(
        grid, "air", [Primitive("box", "tungsten", (0, 0, 0), (10, 10, 10))]
    )
    det = DetectorGeometry(object_half_z_cm=16.0)
    rng = make_rng(0)
    truth = simulate(scene, det, 300, rng)
    meas = apply_detector(truth, get_detector_config("ideal"), rng)
    rec = LearnedReconstructor(unet=VoxelUNet3D(base=8))
    vol = rec.reconstruct(meas, grid)
    assert vol.shape == grid.shape
    assert np.all(np.isfinite(vol))
    assert vol.min() >= 0.0 and vol.max() <= 1.0


def test_refine_tracks_preserves_count():
    grid = VoxelGrid(shape=(16, 16, 16), size_cm=(32, 32, 32))
    scene = Scene.from_primitives(grid, "air", [Primitive("box", "iron", (0, 0, 0), (8, 8, 8))])
    det = DetectorGeometry(object_half_z_cm=16.0)
    rng = make_rng(0)
    truth = simulate(scene, det, 200, rng)
    meas = apply_detector(truth, get_detector_config("realistic_sipm"), rng)
    refined = refine_tracks(TrajectoryMLP(n_planes=4), meas)
    assert len(refined) == len(meas)
    assert refined.in_dir.shape == meas.in_dir.shape
