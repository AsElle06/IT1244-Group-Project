"""PyTorch training loop shared by the BiLSTM and the transformer.

Class-weighted cross-entropy (so rare classes are not ignored), gradient clipping, and early
stopping on validation macro-F1 (the best epoch's weights are kept).

Owner: Suling. Status: editing.
"""
import copy
import time
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


def balanced_class_weights(label_ids: np.ndarray, n_classes: int) -> torch.Tensor:
    """Weight per class = n_samples / (n_classes * class_count), so rare classes count more."""
    counts = np.bincount(label_ids, minlength=n_classes)
    weights = len(label_ids) / (n_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float)


def predict_logits(model: torch.nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """Run the model on every batch and return the raw scores, shape (n, n_classes)."""
    model.eval()
    all_logits = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            all_logits.append(model(**inputs).float().cpu())
    return torch.cat(all_logits).numpy()


def train_classifier(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                     class_weights: torch.Tensor, epochs: int, lr: float, patience: int,
                     device: str, weight_decay: float = 0.0,
                     make_scheduler: Callable | None = None) -> list[dict]:
    """Train with class-weighted cross-entropy and early stopping on validation macro-F1.

    Each batch from the loaders is (inputs, labels), where inputs is a dict of tensors that is
    passed to the model as model(**inputs). At the end the model is reset to its best epoch.

    Returns one dict per epoch: {"epoch", "train_loss", "val_macro_f1"}.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, epochs * len(train_loader)) if make_scheduler else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    val_labels = np.concatenate([labels.numpy() for _, labels in val_loader])

    history = []
    best_f1, best_weights, epochs_without_improvement = -1.0, None, 0

    for epoch in range(1, epochs + 1):
        start = time.time()

        # Train for one epoch
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(**inputs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            total_loss += loss.item() * len(labels)

        # Check on the validation set
        val_pred = predict_logits(model, val_loader, device).argmax(axis=1)
        val_f1 = f1_score(val_labels, val_pred, average="macro", zero_division=0)
        train_loss = total_loss / len(train_loader.dataset)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": val_f1})
        print(f"  epoch {epoch}: train loss {train_loss:.4f}, val macro-F1 {val_f1:.4f} "
              f"({time.time() - start:.0f}s)")

        # Early stopping: keep the best weights, stop after `patience` epochs without improvement
        if val_f1 > best_f1:
            best_f1, best_weights = val_f1, copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  early stopping, best val macro-F1 {best_f1:.4f}")
                break

    model.load_state_dict(best_weights)
    return history
