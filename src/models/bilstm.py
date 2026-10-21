"""Model 2: Bidirectional LSTM with attention.

    words -> word ids -> embedding (128 numbers per word, learned from our reports)
    -> BiLSTM (one pass forwards, one backwards: 128 + 128 = 256 numbers per word)
    -> attention pooling (a learned weight per word, then the weighted average: one 256-vector
       per report, as in Li et al. 2024) -> dropout -> linear layer -> 12 class scores

Unlike TF-IDF it keeps word order and context, e.g. "alarm did not sound". The embeddings are
learned from scratch, so any gain from pretraining is left to the transformer to show.
Class-weighted loss, dropout and early stopping (src/models/torch_training.py) limit overfitting.

Owner: Suling. Status: done. Implements the TextClassifier interface (src/interfaces.py).
"""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from src import config
from src.helpers import get_device, set_seed
from src.interfaces import Labels, Texts
from src.models import torch_training

PAD_ID = 0  # padding, so all reports in a batch have the same length
UNK_ID = 1  # words not in the vocabulary
WORD_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")  # keeps "didn't" as one word


def tokenize(text: str) -> list[str]:
    """Split a cleaned report into words; punctuation is dropped."""
    return WORD_PATTERN.findall(text.lower())


def build_vocab(texts: list[str], min_freq: int) -> list[str]:
    """Words appearing at least min_freq times, most common first.
    Position in the list = word id; ids 0 and 1 are reserved for padding and unknown words."""
    counts = Counter(word for text in texts for word in tokenize(text))
    return ["<pad>", "<unk>"] + [word for word, n in counts.most_common() if n >= min_freq]


class BiLSTMNet(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int, embed_dim: int, hidden_dim: int,
                 dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(2 * hidden_dim, 1)  # one score per word
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_dim, n_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """ids: (batch, max_len) word ids. lengths: (batch,) real lengths. Returns (batch, n_classes)."""
        x = self.dropout(self.embedding(ids))  # (batch, max_len, embed_dim)

        # Packing makes the LSTM skip the padding at the end of short reports
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        states, _ = self.lstm(packed)
        states, _ = pad_packed_sequence(states, batch_first=True, total_length=ids.size(1))
        # states: (batch, max_len, 2 * hidden_dim), one vector per word

        # Attention pooling: a score per word -> weights that sum to 1 (padding gets 0)
        # -> weighted average of the word vectors = one vector per report
        is_word = (ids != PAD_ID).unsqueeze(-1)
        scores = self.attention(states).masked_fill(~is_word, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        report_vector = (weights * states).sum(dim=1)  # (batch, 2 * hidden_dim)

        return self.classifier(self.dropout(report_vector))


class BiLSTMClassifier:
    def __init__(self, max_len: int = 256, min_freq: int = 2, embed_dim: int = 128,
                 hidden_dim: int = 128, dropout: float = 0.3, batch_size: int = 64,
                 lr: float = 1e-3, epochs: int = 10, patience: int = 2, device: str | None = None):
        # All fixed settings (no tuning), kept so save()/load() can rebuild the same model
        self.settings = dict(max_len=max_len, min_freq=min_freq, embed_dim=embed_dim,
                             hidden_dim=hidden_dim, dropout=dropout, batch_size=batch_size,
                             lr=lr, epochs=epochs, patience=patience)
        self.device = device or get_device()
        self.classes_: np.ndarray = np.array([])
        self.history_: list[dict] = []

    def fit(self, texts: Texts, labels: Labels) -> "BiLSTMClassifier":
        """Train on the given reports (in the rotation: the four training folds)."""
        s = self.settings
        set_seed(config.SEED)
        texts, labels = [t if isinstance(t, str) else "" for t in texts], list(labels)

        # 1. Hold out ~10% for early stopping (all copies of a text on one side)
        train_texts, train_labels, val_texts, val_labels = torch_training.split_validation(texts, labels)

        # 2. Classes and vocabulary, from the training part only
        self.classes_ = np.array(sorted(set(labels)), dtype=object)
        self.vocab = build_vocab(train_texts, s["min_freq"])
        self._build_net()

        # 3. Train, keeping the best epoch on validation macro-F1
        train_ids, val_ids = self._label_ids(train_labels), self._label_ids(val_labels)
        self.history_ = torch_training.train_with_early_stopping(
            self.net,
            train_loader=self._loader(train_texts, train_ids, shuffle=True),
            val_loader=self._loader(val_texts, val_ids),
            weights=torch_training.class_weights(train_ids, len(self.classes_)),
            epochs=s["epochs"], lr=s["lr"], patience=s["patience"], device=self.device)
        return self

    def predict_proba(self, texts: Texts) -> np.ndarray:
        texts = [t if isinstance(t, str) else "" for t in texts]
        dummy_labels = np.zeros(len(texts), dtype=int)
        return torch_training.predict_proba(self.net, self._loader(texts, dummy_labels), self.device)

    def predict(self, texts: Texts) -> np.ndarray:
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]

    def save(self, path: Path) -> None:
        """Save settings, classes, vocabulary, weights and training history to one .pt file,
        and the history (loss and validation macro-F1 per epoch) to <name>_history.csv."""
        path = Path(path)
        torch.save({"settings": self.settings, "classes": list(self.classes_), "vocab": self.vocab,
                    "weights": self.net.state_dict(), "history": self.history_}, path)
        pd.DataFrame(self.history_).to_csv(path.with_name(f"{path.stem}_history.csv"), index=False)

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "BiLSTMClassifier":
        saved = torch.load(path, map_location="cpu")
        model = cls(**saved["settings"], device=device)
        model.classes_ = np.array(saved["classes"], dtype=object)
        model.vocab = saved["vocab"]
        model.history_ = saved["history"]
        model._build_net()
        model.net.load_state_dict(saved["weights"])
        model.net.to(model.device)
        return model

    # Helpers -------------------------------------------------------------------------

    def _build_net(self) -> None:
        s = self.settings
        self.word_to_id = {word: i for i, word in enumerate(self.vocab)}
        self.net = BiLSTMNet(len(self.vocab), len(self.classes_), embed_dim=s["embed_dim"],
                             hidden_dim=s["hidden_dim"], dropout=s["dropout"])

    def _label_ids(self, labels: list[str]) -> np.ndarray:
        position = {label: i for i, label in enumerate(self.classes_)}
        return np.array([position[label] for label in labels])

    def _encode(self, text: str) -> torch.Tensor:
        """Word ids of the first max_len words; an empty report still gets one <unk>."""
        words = tokenize(text)[: self.settings["max_len"]]
        return torch.tensor([self.word_to_id.get(w, UNK_ID) for w in words] or [UNK_ID])

    def _loader(self, texts: list[str], label_ids: np.ndarray, shuffle: bool = False):
        # Each report is turned into word ids once, not again in every epoch
        return torch_training.make_loader([self._encode(t) for t in texts], label_ids,
                                          self.settings["batch_size"], self._collate, shuffle)

    @staticmethod
    def _collate(id_lists: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Pad a batch of word-id lists to the longest one in the batch."""
        return {"ids": pad_sequence(id_lists, batch_first=True, padding_value=PAD_ID),
                "lengths": torch.tensor([len(ids) for ids in id_lists])}
