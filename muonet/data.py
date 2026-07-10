"""Data generation: from scenes to reconstruction inputs and training pairs.

Everything the toolkit consumes is produced here, deterministically from a
seed. ``collect_measurement`` runs the simulator and detector until it has a
target number of *reconstructable* muons (dosage is counted in usable muons,
not muons fired). ``make_pair`` turns a scene into the normalized PoCA volume a
network sees and the ground-truth density it should predict.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .detector import DetectorConfig, apply_detector, get_detector_config
from .detector.effects import Measurement
from .geometry import DetectorGeometry, Scene, VoxelGrid
from .metrics import normalize_volume
from .recon import poca_reconstruct
from .scenes import default_detector, default_grid, random_scene
from .sim import SimConfig, simulate

# Muon counts the dosage story sweeps over.
DOSAGE_LEVELS = [128, 256, 512, 1024, 4096, 8192]


def collect_measurement(
    scene: Scene,
    detector: DetectorGeometry,
    config: DetectorConfig,
    n_valid: int,
    rng: np.random.Generator,
    sim_config: SimConfig | None = None,
) -> Measurement:
    """Simulate until ``n_valid`` reconstructable muons are collected."""
    parts: list[Measurement] = []
    got = 0
    attempts = 0
    max_attempts = 200
    while got < n_valid:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"only {got} of {n_valid} muons survived the detector after {attempts} "
                f"batches. Detector efficiency may be near zero, or the scene may sit "
                f"outside the acceptance."
            )
        attempts += 1
        need = n_valid - got
        batch = max(int(need * 1.5) + 64, 256)
        truth = simulate(scene, detector, batch, rng, sim_config)
        valid = apply_detector(truth, config, rng).select_valid()
        if len(valid) > 0:
            parts.append(valid)
            got += len(valid)
    return Measurement.concat(parts).head(n_valid)


# A hook that refines a measurement's tracks (e.g. the stage-1 MLP). Kept as a
# plain callable so this module never imports Torch.
RefineFn = Callable[[Measurement], Measurement]


def poca_input(
    measurement: Measurement, grid: VoxelGrid, refine: RefineFn | None = None
) -> np.ndarray:
    """The normalized PoCA volume a learned model receives as input."""
    if refine is not None:
        measurement = refine(measurement)
    return normalize_volume(poca_reconstruct(measurement, grid))


def make_pair(
    scene: Scene,
    detector: DetectorGeometry,
    config: DetectorConfig,
    dosage: int,
    rng: np.random.Generator,
    sim_config: SimConfig | None = None,
    refine: RefineFn | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(poca_volume, gt_density, gt_high_z)`` for one scene at a dosage."""
    meas = collect_measurement(scene, detector, config, dosage, rng, sim_config)
    grid = scene.grid
    return (
        poca_input(meas, grid, refine),
        scene.normalized_density(),
        scene.high_z_mask(),
    )


def build_training_set(
    n_scenes: int,
    rng: np.random.Generator,
    dosages: list[int] | None = None,
    detector_name: str = "realistic_sipm",
    grid: VoxelGrid | None = None,
    detector: DetectorGeometry | None = None,
    refine: RefineFn | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``(X, Y)`` training tensors from random scenes.

    ``X`` has shape ``(n_scenes, 1, nx, ny, nz)`` (normalized PoCA volumes) and
    ``Y`` the same shape (ground-truth density). Each scene is simulated at a
    randomly chosen dosage so the model generalizes across exposure.
    """
    grid = grid or default_grid()
    detector = detector or default_detector()
    config = get_detector_config(detector_name)
    dosages = dosages or DOSAGE_LEVELS

    inputs = np.empty((n_scenes, 1, *grid.shape), dtype=np.float32)
    targets = np.empty((n_scenes, 1, *grid.shape), dtype=np.float32)
    for i in range(n_scenes):
        scene = random_scene(rng, grid)
        dosage = int(rng.choice(dosages))
        x, y, _ = make_pair(scene, detector, config, dosage, rng, refine=refine)
        inputs[i, 0] = x
        targets[i, 0] = y
    return inputs, targets
