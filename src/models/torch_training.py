"""Shared training code for the two PyTorch models (BiLSTM and transformer), as plain functions.

Each model's fit() calls these step by step, so both models are trained the same way:
    split_validation           hold out ~10% of the training reports to decide when to stop
    class_weights              rare classes count more in the loss
    make_loader                batches of (model inputs, label ids)
    train_with_early_stopping  the training loop, keeping the best epoch
    predict_proba              class probabilities for every report
    plot_training_curves       loss and validation macro-F1 per epoch, one line per turn

Mixed precision (transformer only, GPU only): the heavy calculations run in 16-bit numbers, which
the T4 GPU computes several times faster; the weights themselves stay 32-bit. Off by default, so
the BiLSTM is trained exactly as before.

Owner: Suling. Status: done.
"""
import contextlib
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src import config
from src.data.splitting import assign_folds

# The training reports are cut into this many group-safe pieces; one piece is the validation set
VALIDATION_PIECES = 10


def split_validation(texts: list[str], labels: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    """Hold out about 1 / VALIDATION_PIECES of the training reports for early stopping.

    Uses the same rule as the test folds (splitting.assign_folds): all copies of a text land on
    the same side, and each class has about the same share in validation. Without this, copies of
    a template would be in both parts, and validation would only test memory.

    Returns (train_texts, train_labels, val_texts, val_labels).
    """
    df = pd.DataFrame({config.TEXT_COL: texts, config.LABEL_COL: labels})
    is_val = (assign_folds(df, n_folds=VALIDATION_PIECES, seed=config.SEED) == 0).to_numpy()
    texts, labels = np.array(texts, dtype=object), np.array(labels, dtype=object)
    return (list(texts[~is_val]), list(labels[~is_val]), list(texts[is_val]), list(labels[is_val]))


def class_weights(label_ids: np.ndarray, n_classes: int) -> torch.Tensor:
    """Weight per class = n_reports / (n_classes * class_count), so rare classes count more."""
    counts = np.bincount(label_ids, minlength=n_classes)
    return torch.tensor(len(label_ids) / (n_classes * np.maximum(counts, 1)), dtype=torch.float)


def make_loader(inputs: list, label_ids: np.ndarray, batch_size: int,
                collate: Callable[[list], dict[str, torch.Tensor]], shuffle: bool = False) -> DataLoader:
    """Batches of (model inputs, label ids).

    inputs: one item per report, in whatever form the model's collate function expects
        (word ids for the BiLSTM, sub-word token ids for the transformer).
    collate: turns a list of items into the dict of tensors the network takes as net(**inputs).
    """
    pairs = list(zip(inputs, label_ids))

    def collate_pairs(batch):
        items = [item for item, _ in batch]
        labels = torch.tensor([label for _, label in batch], dtype=torch.long)
        return collate(items), labels

    # A seeded generator, so the shuffled order is the same in every run
    generator = torch.Generator().manual_seed(config.SEED) if shuffle else None
    return DataLoader(pairs, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_pairs,
                      generator=generator)


def _uses_16_bit(mixed_precision: bool, device: str) -> bool:
    """Mixed precision only runs on a GPU; on a CPU everything stays 32-bit."""
    return mixed_precision and str(device).startswith("cuda")


def _autocast(enabled: bool):
    """Inside this block the GPU does the heavy calculations in 16-bit; does nothing when disabled."""
    return torch.autocast("cuda", dtype=torch.float16) if enabled else contextlib.nullcontext()


def predict_proba(net: torch.nn.Module, loader: DataLoader, device: str,
                  mixed_precision: bool = False) -> np.ndarray:
    """Class probabilities for every report in the loader, shape (n, n_classes)."""
    use_16_bit = _uses_16_bit(mixed_precision, device)
    net.eval()
    batches = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            with _autocast(use_16_bit):
                scores = net(**inputs)
            batches.append(torch.softmax(scores.float(), dim=1).cpu())
    return torch.cat(batches).numpy()


def train_with_early_stopping(net: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                              weights: torch.Tensor, epochs: int, lr: float, patience: int,
                              device: str, weight_decay: float = 0.0,
                              make_scheduler: Callable | None = None,
                              mixed_precision: bool = False) -> list[dict]:
    """Train with class-weighted cross-entropy; stop early on validation macro-F1.

    After every epoch the model is scored on the validation set. Training stops after `patience`
    epochs without improvement, and the network is reset to its best epoch.
    make_scheduler(optimizer, total_steps): optional learning-rate schedule (for the transformer).
    mixed_precision: 16-bit calculations on a GPU (for the transformer); the weights stay 32-bit.

    Returns one dict per epoch: {"epoch", "train_loss", "val_macro_f1", "seconds"}.
    """
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, epochs * len(train_loader)) if make_scheduler else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    val_labels = np.concatenate([labels.numpy() for _, labels in val_loader])

    # With 16-bit calculations, small gradients could round to 0. The scaler multiplies the loss by
    # a large number before the gradients are computed and divides them back before the update.
    # When disabled it does nothing: scale() returns the loss unchanged and step() is optimizer.step().
    use_16_bit = _uses_16_bit(mixed_precision, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_16_bit)

    history = []
    best_f1, best_weights, epochs_without_improvement = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        start = time.time()

        # Train for one epoch
        net.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            with _autocast(use_16_bit):
                loss = loss_fn(net(**inputs), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # true gradient sizes again, before clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # avoid exploding steps
            scaler.step(optimizer)  # with 16-bit: skips the step if a gradient overflowed
            scaler.update()
            if scheduler:
                scheduler.step()
            total_loss += loss.item() * len(labels)

        # Score on the validation set
        val_pred = predict_proba(net, val_loader, device, mixed_precision).argmax(axis=1)
        val_f1 = float(f1_score(val_labels, val_pred, average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_loader.dataset),
                        "val_macro_f1": val_f1, "seconds": round(time.time() - start, 1)})
        print(f"  epoch {epoch}: train loss {history[-1]['train_loss']:.4f}, "
              f"val macro-F1 {val_f1:.4f} ({history[-1]['seconds']:.0f}s)")

        # Early stopping: remember the best epoch's weights
        if val_f1 > best_f1:
            best_f1, epochs_without_improvement = val_f1, 0
            best_weights = {name: w.detach().cpu().clone() for name, w in net.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  early stopping: best val macro-F1 {best_f1:.4f}")
                break

    net.load_state_dict(best_weights)
    return history


# One fixed colour per turn of the rotation (colour-blind-safe order), plus chart surface and ink
TURN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, INK, MUTED_INK, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"


def plot_training_curves(histories: list[pd.DataFrame], model_name: str,
                         save_as: str | None = None) -> None:
    """Training loss and validation macro-F1 per epoch, one line per turn of the rotation.

    histories: one DataFrame per turn with columns epoch, train_loss, val_macro_f1
        (the *_history.csv files saved next to the weights).
    The dot on each validation line marks the best epoch: the weights that were kept.
    If save_as is given, the figure is also saved to config.FIGURES_DIR.
    """
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
    for k, history in enumerate(histories):
        color = TURN_COLORS[k]
        ax_loss.plot(history["epoch"], history["train_loss"], color=color, linewidth=2, label=f"turn {k}")
        ax_f1.plot(history["epoch"], history["val_macro_f1"], color=color, linewidth=2)
        best = history.loc[history["val_macro_f1"].idxmax()]
        ax_f1.scatter(best["epoch"], best["val_macro_f1"], s=70, color=color, edgecolors=SURFACE,
                      linewidths=2, zorder=3)

    for ax, title in [(ax_loss, "Training loss"), (ax_f1, "Validation macro-F1 (dot = epoch kept)")]:
        ax.set_facecolor(SURFACE)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xlabel("epoch", color=MUTED_INK)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED_INK, length=0)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    fig.suptitle(f"{model_name}: training curves of the 5 turns", x=0.01, ha="left", color=INK,
                 fontsize=13)
    fig.legend(loc="upper right", ncol=len(histories), frameon=False, labelcolor=INK,
               bbox_to_anchor=(0.99, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if save_as is not None:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(config.FIGURES_DIR / save_as, dpi=200, bbox_inches="tight")
    plt.show()
