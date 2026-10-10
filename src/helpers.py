"""Small utilities used across the project: choosing the device (GPU/CPU) and fixing random seeds."""
import random

import numpy as np
import torch


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def device_info() -> str:
    if torch.cuda.is_available():
        return f"cuda: {torch.cuda.get_device_name(0)}"
    return "cpu"


def set_seed(seed: int) -> None:
    """Fix all random seeds so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
