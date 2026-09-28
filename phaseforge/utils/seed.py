"""Deterministic seeding for torch, numpy, random, and CUDA."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducible results.

    Args:
        seed: Integer seed value. Same seed on same hardware yields identical results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU

    # Make CUDA operations deterministic (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
