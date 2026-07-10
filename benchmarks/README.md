# Benchmarks

Every number Muonet reports is generated here, from a fixed seed, with one command.

```bash
./benchmarks/reproduce.sh
# or directly:
muonet bench --checkpoint ../checkpoints/muonet.pt --out results
```

## What gets produced

- `results/dosage_curve.{json,csv}` — reconstruction quality (PSNR, SSIM, IoU, high-Z accuracy) and runtime for PoCA, MLEM, and the learned model, across muon counts from 128 to 8192, for each built-in scene.
- `results/realism_ablation.{json,csv}` — the same target reconstructed as the detector degrades from `ideal` to `battery_limited`.
- `results/summary.md` — a compact markdown table, quoted directly in the top-level README.

## How to read them

- **PSNR / SSIM** compare the reconstructed volume against the ground-truth scattering density.
- **IoU / high-Z accuracy** measure the security-relevant call: did we correctly flag the dense (high-Z) voxels? This is where the learned model wins most clearly.
- **runtime** is wall-clock per reconstruction on CPU.

The normalization applied before scoring is identical for every method and is defined in [`muonet/metrics.py`](../muonet/metrics.py). It is stated in the open on purpose: an honest benchmark stands or falls on how it normalizes.

## Reproducibility notes

The committed checkpoint makes the learned results a deterministic CPU forward pass, so they reproduce exactly on a fresh clone without a GPU. The analytic simulator is seeded, so the classical baselines reproduce exactly as well. Retraining from scratch (`muonet train`) may shift the learned numbers by a small amount depending on hardware and PyTorch build; the committed checkpoint is the reference.
