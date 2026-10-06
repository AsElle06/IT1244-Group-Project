import torch.nn as nn


def build(device="cpu"):
    return nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    ).to(device)
