"""Shared training code for the two PyTorch models (BiLSTM and transformer), as plain functions.

Each model's fit() calls these step by step, so both models are trained the same way:
    split_validation           hold out ~10% of the training reports to decide when to stop
    class_weights              rare classes count more in the loss
    make_loader                batches of (model inputs, label ids)
    train_with_early_stopping  the training loop, keeping the best epoch
    predict_proba              class probabilities for every report

Owner: Suling. Status: in progress.
"""
import time
from typing import Callable

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
        (word ids for the BiLSTM, texts for the transformer).
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


def predict_proba(net: torch.nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """Class probabilities for every report in the loader, shape (n, n_classes)."""
    net.eval()
    batches = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            batches.append(torch.softmax(net(**inputs).float(), dim=1).cpu())
    return torch.cat(batches).numpy()


def train_with_early_stopping(net: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                              weights: torch.Tensor, epochs: int, lr: float, patience: int,
                              device: str, weight_decay: float = 0.0,
                              make_scheduler: Callable | None = None) -> list[dict]:
    """Train with class-weighted cross-entropy; stop early on validation macro-F1.

    After every epoch the model is scored on the validation set. Training stops after `patience`
    epochs without improvement, and the network is reset to its best epoch.
    make_scheduler(optimizer, total_steps): optional learning-rate schedule (for the transformer).

    Returns one dict per epoch: {"epoch", "train_loss", "val_macro_f1", "seconds"}.
    """
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, epochs * len(train_loader)) if make_scheduler else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    val_labels = np.concatenate([labels.numpy() for _, labels in val_loader])

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
            loss = loss_fn(net(**inputs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # avoid exploding steps
            optimizer.step()
            if scheduler:
                scheduler.step()
            total_loss += loss.item() * len(labels)

        # Score on the validation set
        val_pred = predict_proba(net, val_loader, device).argmax(axis=1)
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
