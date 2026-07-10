#!/usr/bin/env bash
# Regenerate every committed benchmark number from a fixed seed.
#
# With the committed checkpoint this is a deterministic CPU forward pass: no
# GPU, no retraining, no downloads. Runtime is a couple of minutes on a laptop.
#
#   ./benchmarks/reproduce.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="checkpoints/muonet.pt"
OUT="benchmarks/results"

if [[ -f "$CKPT" ]]; then
  muonet bench --checkpoint "$CKPT" --out "$OUT"
else
  echo "No checkpoint at $CKPT; reconstructing classical baselines only."
  echo "Run 'muonet train --out $CKPT' to include the learned model."
  muonet bench --out "$OUT"
fi

echo
echo "Wrote results to $OUT/"
echo "Summary table: $OUT/summary.md"
