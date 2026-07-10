"""One place to seed everything, so every number reproduces from a single int."""

from __future__ import annotations

import os
import random

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Return a NumPy generator. All simulation randomness flows through this."""
    return np.random.default_rng(seed)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and (if importable) Torch for reproducible runs.

    Torch is imported lazily so the physics simulator never hard-depends on it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
