# Contributing to Muonet

Thanks for looking. Muonet is meant to be small enough that one person can hold the whole thing in their head, and that is a feature worth protecting.

## Ground rules

1. **Reproducibility is the point.** Anything that produces a number takes a seed and reproduces exactly on a fresh clone. If a change makes a benchmark non-deterministic, it needs a very good reason.
2. **No new required heavy dependencies.** The core pipeline is NumPy plus a small amount of Torch. Geant4, ROOT, and friends stay optional and off the critical path. CI must never need a GPU or Geant4.
3. **Honest benchmarks only.** If a change improves the learned model, show the number. If it regresses somewhere, say so in the PR. Cherry-picking is the one unforgivable sin here.

## Getting set up

```bash
git clone https://github.com/KarthikSubramanian07/muonet.git
cd muonet
pip install -e ".[dev]"
pytest            # should be green
ruff check muonet tests
```

The test suite is CPU-only and runs in under a minute. The learned-model tests use tiny networks on purpose so they stay fast.

## The viewer

```bash
cd viewer
npm install
npm run dev       # http://localhost:3000
```

The viewer reads precomputed volumes from `viewer/public/volumes/`. Generate them with `muonet export`.

## Sending a change

- Keep commits small and logically scoped.
- Run `ruff check` and `pytest` before opening a PR.
- If you touched physics, add or update a test that pins the behavior against a hand calculation or a published value.
- If you touched a benchmark path, include the before/after numbers.

That is it. Open an issue first if you are planning something large, so we can make sure it fits the "small enough to hold in your head" constraint.
