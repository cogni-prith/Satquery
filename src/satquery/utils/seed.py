"""Deterministic seeding for every entry script.

Seeds the standard library, NumPy and -- when installed -- PyTorch. Torch is an
optional GPU-group dependency, so the import is guarded and its absence is not an
error on a CPU-only laptop.
"""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["DEFAULT_SEED", "seed_everything"]

DEFAULT_SEED = 1337


def seed_everything(seed: int = DEFAULT_SEED, *, deterministic_torch: bool = True) -> int:
    """Seed all random number generators in play and return the seed used.

    Args:
        seed: Value applied to `random`, NumPy and (if importable) PyTorch.
        deterministic_torch: Request deterministic cuDNN kernels. Slower, but
            required for a reproducible before/after adaptation table.

    Returns:
        The seed that was applied, so callers can log it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
