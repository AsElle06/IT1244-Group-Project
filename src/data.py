import torch


def make_data(n=1000, device="cpu"):
    x = torch.randn(n, 1, device=device)
    y = x**2 + 0.1 * torch.randn(n, 1, device=device)
    return x, y
