import torch


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def device_info():
    if torch.cuda.is_available():
        return f"cuda: {torch.cuda.get_device_name(0)}"
    return "cpu"
