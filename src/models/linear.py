import torch.nn as nn


def build(device="cpu"):
    return nn.Linear(1, 1).to(device)
