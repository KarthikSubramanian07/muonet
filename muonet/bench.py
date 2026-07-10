"""The reproducible benchmark harness.

Every number the README quotes comes out of here, from a fixed seed, with one
command. It runs the dosage curve (how reconstruction sharpens with muon
count), the realism ablation (how it degrades as the detector gets worse), and
per-method timing, for PoCA, MLEM, and the learned model. Results are written
as JSON and CSV so they can be diffed on a fresh clone.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from .data import collect_measurement
from .detector import get_detector_config
from .detector.effects import Measurement
from .geometry import Scene, VoxelGrid
from .metrics import evaluate
from .recon import poca_reconstruct
from .recon.mlem import mlem_reconstruct
from .scenes import build_scene, default_detector, default_grid

DEFAULT_DOSAGES = [128, 256, 512, 1024, 4096, 8192]
DEFAULT_SCENES = ["tungsten_cube", "lead_sphere", "iron_vs_tungsten", "concealed_core"]
REALISM_PRESETS = ["ideal", "research", "realistic_sipm", "battery_limited"]


@dataclass
class MethodResult:
    method: str
    scene: str
    detector: str
    dosage: int
    psnr: float
    ssim: float
    iou: float
    accuracy: float
    runtime_s: float


def _timed(fn):
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def reconstruct_all(
    measurement: Measurement,
    scene: Scene,
    grid: VoxelGrid,
    learned=None,
    do_mlem: bool = True,
) -> dict[str, tuple[np.ndarray, dict[str, float], float]]:
    """Reconstruct with every available method; return (volume, metrics, runtime)."""
    gt = scene.normalized_density()
    hz = scene.high_z_mask()
    out: dict[str, tuple[np.ndarray, dict[str, float], float]] = {}

    vol, rt = _timed(lambda: poca_reconstruct(measurement, grid))
    out["poca"] = (vol, evaluate(vol, gt, hz), rt)

    if do_mlem:
        vol, rt = _timed(lambda: mlem_reconstruct(measurement, grid))
        out["mlem"] = (vol, evaluate(vol, gt, hz), rt)

    if learned is not None:
        vol, rt = _timed(lambda: learned.reconstruct(measurement, grid))
        out["learned"] = (vol, evaluate(vol, gt, hz), rt)

    return out


def run_dosage_curve(
    scenes: list[str],
    dosages: list[int],
    detector_name: str = "realistic_sipm",
    seed: int = 0,
    learned=None,
    do_mlem: bool = True,
    grid: VoxelGrid | None = None,
    detector=None,
) -> list[MethodResult]:
    """Sweep muon count for each scene and method."""
    grid = grid or default_grid()
    detector = detector or default_detector()
    config = get_detector_config(detector_name)
    results: list[MethodResult] = []
    for scene_name in scenes:
        scene = build_scene(scene_name, grid)
        for dosage in dosages:
            rng = np.random.default_rng(seed)
            meas = collect_measurement(scene, detector, config, dosage, rng)
            recon = reconstruct_all(meas, scene, grid, learned, do_mlem)
            for method, (_, metrics, rt) in recon.items():
                results.append(
                    MethodResult(
                        method=method,
                        scene=scene_name,
                        detector=detector_name,
                        dosage=dosage,
                        runtime_s=round(rt, 4),
                        **{k: round(v, 4) for k, v in metrics.items()},
                    )
                )
    return results


def run_realism_ablation(
    scene_name: str,
    dosage: int,
    presets: list[str] = tuple(REALISM_PRESETS),
    seed: int = 0,
    learned=None,
    do_mlem: bool = True,
    grid: VoxelGrid | None = None,
    detector=None,
) -> list[MethodResult]:
    """Hold scene and dosage fixed; degrade the detector preset by preset."""
    grid = grid or default_grid()
    detector = detector or default_detector()
    scene = build_scene(scene_name, grid)
    results: list[MethodResult] = []
    for preset in presets:
        config = get_detector_config(preset)
        rng = np.random.default_rng(seed)
        meas = collect_measurement(scene, detector, config, dosage, rng)
        recon = reconstruct_all(meas, scene, grid, learned, do_mlem)
        for method, (_, metrics, rt) in recon.items():
            results.append(
                MethodResult(
                    method=method,
                    scene=scene_name,
                    detector=preset,
                    dosage=dosage,
                    runtime_s=round(rt, 4),
                    **{k: round(v, 4) for k, v in metrics.items()},
                )
            )
    return results


def results_to_rows(results: list[MethodResult]) -> list[dict]:
    return [asdict(r) for r in results]
