"""Shared fit/predict code for the two PyTorch models (BiLSTM and transformer).

Handles the parts that are the same for both: holding out a validation set for early stopping,
turning labels into ids, batching, training (torch_training.py) and predicting.

A subclass only has to define:
    build_network(train_texts)  create self.net (and the vocabulary / tokenizer it needs)
    make_batch(texts)           turn a list of texts into the dict of tensors self.net expects

Owner: Suling. Status: editing.
"""
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src import config
from src.helpers import get_device, set_seed
from src.interfaces import Labels, Texts
from src.models.torch_training import balanced_class_weights, predict_logits, train_classifier

VALIDATION_SHARE = 0.1  # part of the training data held out for early stopping


class TorchTextClassifier:
    def __init__(self, batch_size: int, lr: float, epochs: int, patience: int,
                 weight_decay: float = 0.0, device: str | None = None):
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.weight_decay = weight_decay
        self.device = device or get_device()

    # To be defined by each model --------------------------------------------------

    def build_network(self, train_texts: list[str]) -> None:
        raise NotImplementedError

    def make_batch(self, texts: list[str]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def make_scheduler(self, optimizer, total_steps: int):
        """Optional learning-rate schedule. None = constant learning rate."""
        return None

    # Shared --------------------------------------------------------------------------

    def fit(self, texts: Texts, labels: Labels) -> "TorchTextClassifier":
        """Train the model. 10% of the given data is held out (stratified) for early stopping,
        so during cross-validation the scored fold is never seen."""
        set_seed(config.SEED)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            list(texts), list(labels), test_size=VALIDATION_SHARE, stratify=list(labels),
            random_state=config.SEED)

        self.classes_ = np.array(sorted(set(train_labels)), dtype=object)
        self.build_network(train_texts)

        train_ids = self._to_label_ids(train_labels)
        self.history_ = train_classifier(
            self.net,
            train_loader=self._make_loader(train_texts, train_ids, shuffle=True),
            val_loader=self._make_loader(val_texts, self._to_label_ids(val_labels)),
            class_weights=balanced_class_weights(train_ids, len(self.classes_)),
            epochs=self.epochs, lr=self.lr, patience=self.patience, device=self.device,
            weight_decay=self.weight_decay, make_scheduler=self.make_scheduler,
        )
        return self

    def predict_proba(self, texts: Texts) -> np.ndarray:
        texts = list(texts)
        dummy_labels = np.zeros(len(texts), dtype=int)
        logits = predict_logits(self.net, self._make_loader(texts, dummy_labels), self.device)
        return torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    def predict(self, texts: Texts) -> np.ndarray:
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]

    # Helpers -------------------------------------------------------------------------

    def _to_label_ids(self, labels: list[str]) -> np.ndarray:
        position = {label: i for i, label in enumerate(self.classes_)}
        return np.array([position[label] for label in labels])

    def _make_loader(self, texts: list[str], label_ids: np.ndarray,
                     shuffle: bool = False) -> DataLoader:
        pairs = list(zip(texts, label_ids))
        return DataLoader(pairs, batch_size=self.batch_size, shuffle=shuffle,
                          collate_fn=self._collate)

    def _collate(self, pairs: list[tuple[str, int]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        texts = [text if isinstance(text, str) else "" for text, _ in pairs]
        label_ids = torch.tensor([label_id for _, label_id in pairs], dtype=torch.long)
        return self.make_batch(texts), label_ids
