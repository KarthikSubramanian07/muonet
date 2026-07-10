"""The ``muonet`` command line: simulate, reconstruct, train, benchmark, export.

Every subcommand takes a ``--seed`` and prints or writes exactly what it did, so
a result can always be traced back to the command that produced it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .bench import (
    DEFAULT_DOSAGES,
    DEFAULT_SCENES,
    REALISM_PRESETS,
    reconstruct_all,
    results_to_rows,
    run_dosage_curve,
    run_realism_ablation,
)
from .config import RunConfig, load_run_config
from .data import DOSAGE_LEVELS, collect_measurement
from .detector import DETECTOR_PRESETS, get_detector_config
from .materials import MATERIALS
from .metrics import evaluate
from .recon import poca_reconstruct
from .recon.mlem import mlem_reconstruct
from .scenes import default_detector, list_scenes


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _run_config_from_args(args) -> RunConfig:
    cfg = load_run_config(args.config) if getattr(args, "config", None) else RunConfig()
    for key in ("scene", "detector", "dosage", "seed"):
        value = getattr(args, key, None)
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def _print_metrics(title: str, metrics: dict[str, float], runtime: float | None = None):
    line = (
        f"PSNR {metrics['psnr']:6.2f} dB   SSIM {metrics['ssim']:.3f}   "
        f"IoU {metrics['iou']:.3f}   high-Z acc {metrics['accuracy']:.3f}"
    )
    if runtime is not None:
        line += f"   ({runtime * 1e3:.0f} ms)"
    print(f"  {title:9s} {line}")


def _load_learned(path: str | None, device: str):
    if not path:
        return None
    from .train import load_checkpoint

    return load_checkpoint(path, device=device)


def _write_rows(rows: list[dict], json_path: Path, csv_path: Path) -> None:
    json_path.write_text(json.dumps(rows, indent=2))
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_info(args) -> int:
    print("muonet", __version__)
    print("\nmaterials (radiation length, cm):")
    for name, m in MATERIALS.items():
        print(f"  {name:10s} X0={m.radiation_length_cm:10.4g}  Z={m.z_eff:g}")
    print("\nscenes:", ", ".join(list_scenes()))
    print("detectors:", ", ".join(DETECTOR_PRESETS))
    print("dosage levels:", ", ".join(str(d) for d in DOSAGE_LEVELS))
    return 0


def cmd_recon(args) -> int:
    cfg = _run_config_from_args(args)
    grid = cfg.grid()
    scene = cfg.build_scene()
    detector = default_detector()
    config = get_detector_config(cfg.detector)
    rng = np.random.default_rng(cfg.seed)

    print(
        f"scene={cfg.scene if isinstance(cfg.scene, str) else 'custom'} "
        f"detector={cfg.detector} dosage={cfg.dosage} seed={cfg.seed}"
    )
    meas = collect_measurement(scene, detector, config, cfg.dosage, rng)
    gt = scene.normalized_density()
    hz = scene.high_z_mask()

    methods = args.method or ["poca"]
    learned = _load_learned(args.checkpoint, args.device) if "learned" in methods else None

    volumes = {}
    for method in methods:
        if method == "poca":
            vol = poca_reconstruct(meas, grid)
        elif method == "mlem":
            vol = mlem_reconstruct(meas, grid)
        elif method == "learned":
            if learned is None:
                print("error: --method learned requires --checkpoint", file=sys.stderr)
                return 2
            vol = learned.reconstruct(meas, grid)
        else:
            print(f"error: unknown method {method!r}", file=sys.stderr)
            return 2
        volumes[method] = vol
        _print_metrics(method, evaluate(vol, gt, hz))

    if args.png:
        from .viz import save_comparison

        panels = {"ground truth": gt, **volumes}
        save_comparison(panels, args.png, title=f"{cfg.scene} @ {cfg.dosage} muons")
        print(f"wrote {args.png}")
    if args.export:
        from .viz import export_volume

        for method, vol in volumes.items():
            export_volume(vol, f"{args.export}_{method}")
        print(f"wrote {args.export}_*.bin / .json")
    return 0


def cmd_train(args) -> int:
    from .train import TrainConfig, build_learned, save_checkpoint

    config = TrainConfig(device=args.device)
    if args.quick:
        config.mlp_scenes, config.mlp_epochs = 8, 10
        config.unet_scenes, config.unet_epochs = 24, 6
    print(f"training on {config.device} (seed {args.seed}) ...")
    reconstructor, stats = build_learned(config, seed=args.seed, verbose=not args.quiet)
    print(
        f"stage-1 track error: MLP {stats['mlp_track_error_mrad']:.2f} mrad "
        f"vs least-squares {stats['lsq_track_error_mrad']:.2f} mrad"
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(reconstructor, args.out)
    print(f"wrote checkpoint {args.out}")
    return 0


def cmd_bench(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    learned = _load_learned(args.checkpoint, args.device)
    do_mlem = args.with_mlem

    scenes = ["tungsten_cube"] if args.quick else DEFAULT_SCENES
    dosages = [256, 1024, 8192] if args.quick else DEFAULT_DOSAGES

    print(f"dosage curve: scenes={scenes} dosages={dosages}")
    curve = run_dosage_curve(
        scenes, dosages, seed=args.seed, learned=learned, do_mlem=do_mlem
    )
    _write_rows(results_to_rows(curve), out / "dosage_curve.json", out / "dosage_curve.csv")

    print(f"realism ablation: {REALISM_PRESETS} @ 1024 muons on tungsten_cube")
    ablation = run_realism_ablation(
        "tungsten_cube", 1024, seed=args.seed, learned=learned, do_mlem=do_mlem
    )
    _write_rows(
        results_to_rows(ablation), out / "realism_ablation.json", out / "realism_ablation.csv"
    )

    _write_summary(out, curve, ablation, learned is not None)
    print(f"wrote results to {out}/")
    return 0


def _write_summary(out: Path, curve, ablation, has_learned: bool) -> None:
    """A compact markdown the README can quote directly."""
    lines = ["# Muonet benchmark results\n"]
    lines.append("## Dosage curve (tungsten_cube, realistic_sipm detector)\n")
    lines.append("| muons | method | PSNR (dB) | SSIM | IoU | high-Z acc |")
    lines.append("|------:|:-------|----------:|-----:|----:|-----------:|")
    for r in curve:
        if r.scene != "tungsten_cube":
            continue
        lines.append(
            f"| {r.dosage} | {r.method} | {r.psnr:.2f} | {r.ssim:.3f} | "
            f"{r.iou:.3f} | {r.accuracy:.3f} |"
        )
    lines.append("\n## Realism ablation (tungsten_cube, 1024 muons)\n")
    lines.append("| detector | method | PSNR (dB) | SSIM | IoU |")
    lines.append("|:---------|:-------|----------:|-----:|----:|")
    for r in ablation:
        lines.append(
            f"| {r.detector} | {r.method} | {r.psnr:.2f} | {r.ssim:.3f} | {r.iou:.3f} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")


SCENE_LABELS = {
    "tungsten_cube": "Tungsten cube",
    "lead_sphere": "Lead sphere",
    "iron_vs_tungsten": "Iron vs tungsten",
    "concealed_core": "Concealed core",
}


def cmd_export(args) -> int:
    from .scenes import build_scene, default_grid
    from .viz import export_manifest, export_volume

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    learned = _load_learned(args.checkpoint, args.device)
    grid = default_grid()
    detector = default_detector()
    scenes = args.scenes or ["tungsten_cube", "lead_sphere"]
    dosages = args.dosages or DOSAGE_LEVELS
    config = get_detector_config(args.detector)

    methods = ["poca"] + (["learned"] if learned is not None else [])
    manifest = {
        "scenes": [],
        "dosages": dosages,
        "methods": methods,
        "detector": args.detector,
    }

    for scene_name in scenes:
        scene = build_scene(scene_name, grid)
        entry = {
            "name": scene_name,
            "label": SCENE_LABELS.get(scene_name, scene_name),
            "ground_truth": f"{scene_name}_gt",
            "volumes": {},
        }
        export_volume(scene.normalized_density(), str(out / f"{scene_name}_gt"))
        for dosage in dosages:
            rng = np.random.default_rng(args.seed)
            meas = collect_measurement(scene, detector, config, dosage, rng)
            recon = reconstruct_all(meas, scene, grid, learned, do_mlem=False)
            per_dosage = {}
            for method, (vol, metrics, _) in recon.items():
                stem = f"{scene_name}_{method}_{dosage}"
                export_volume(vol, str(out / stem))
                per_dosage[method] = {"stem": stem, **{k: round(v, 3) for k, v in metrics.items()}}
            entry["volumes"][str(dosage)] = per_dosage
        manifest["scenes"].append(entry)
        print(f"exported {scene_name}")

    export_manifest(manifest, str(out / "manifest.json"))
    print(f"wrote viewer volumes + manifest to {out}/")
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="muonet", description="Cosmic rays in, 3D interiors out.")
    p.add_argument("--version", action="version", version=f"muonet {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="list materials, scenes, and detector presets").set_defaults(
        func=cmd_info
    )

    def add_common(sp):
        sp.add_argument("--config", help="YAML run config")
        sp.add_argument("--scene", help="built-in scene name")
        sp.add_argument("--detector", help="detector preset")
        sp.add_argument("--dosage", type=int, help="number of reconstructable muons")
        sp.add_argument("--seed", type=int, help="random seed")
        sp.add_argument("--device", default="cpu", help="cpu | mps | cuda")

    r = sub.add_parser("recon", help="simulate a scene and reconstruct it")
    add_common(r)
    r.add_argument(
        "--method", nargs="+", choices=["poca", "mlem", "learned"], help="method(s)"
    )
    r.add_argument("--checkpoint", help="learned model checkpoint (.pt)")
    r.add_argument("--png", help="write a comparison cross-section PNG here")
    r.add_argument("--export", help="write web voxel volumes with this path stem")
    r.set_defaults(func=cmd_recon, seed=0)

    t = sub.add_parser("train", help="train the learned two-stage model")
    t.add_argument("--out", default="checkpoints/muonet.pt", help="checkpoint output path")
    t.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--quick", action="store_true", help="tiny run for smoke testing")
    t.add_argument("--quiet", action="store_true")
    t.set_defaults(func=cmd_train)

    b = sub.add_parser("bench", help="run the reproducible benchmark suite")
    b.add_argument("--out", default="benchmarks/results", help="output directory")
    b.add_argument("--checkpoint", help="learned model checkpoint (.pt)")
    b.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--quick", action="store_true", help="small subset for smoke testing")
    b.add_argument(
        "--with-mlem",
        action="store_true",
        help="also run the MLEM reference baseline (slower, not tuned to beat PoCA)",
    )
    b.set_defaults(func=cmd_bench)

    e = sub.add_parser("export", help="export precomputed volumes for the web viewer")
    e.add_argument("--out", default="viewer/public/volumes", help="output directory")
    e.add_argument("--checkpoint", help="learned model checkpoint (.pt)")
    e.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    e.add_argument("--scenes", nargs="+", help="scene names to export")
    e.add_argument("--dosages", nargs="+", type=int, help="dosage levels to export")
    e.add_argument("--detector", default="realistic_sipm", help="detector preset")
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(func=cmd_export)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
