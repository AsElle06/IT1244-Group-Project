"""Model 2: Bidirectional LSTM with word embeddings.

Pipeline: words -> learned embeddings -> BiLSTM (reads the report forwards and backwards)
-> attention pooling (as in Li et al. 2024) -> linear layer -> class scores.
Unlike TF-IDF it keeps word order and context, e.g. "alarm did not sound".
Dropout, class-weighted loss and early stopping limit overfitting.

Owner: Suling. Status: done. Implements the TextClassifier interface (src/interfaces.py).
"""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from src.models.torch_base import TorchTextClassifier

PAD_ID = 0  # padding, so all reports in a batch have the same length
UNK_ID = 1  # words not in the vocabulary
WORD_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into words. Texts are already cleaned by src/data/cleaning.py."""
    return WORD_PATTERN.findall(text.lower())


def build_vocab(texts: list[str], min_freq: int, max_size: int) -> list[str]:
    """The max_size most common words appearing at least min_freq times.
    Position in the list = word id; ids 0 and 1 are reserved for padding and unknown words."""
    counts = Counter(word for text in texts for word in tokenize(text))
    words = [word for word, n in counts.most_common(max_size) if n >= min_freq]
    return ["<pad>", "<unk>"] + words


class BiLSTMNet(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int, embed_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float, attention: bool):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attention = nn.Linear(2 * hidden_dim, 1) if attention else None
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_dim, n_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """ids: (batch, max_len) word ids. lengths: (batch,) real lengths. Returns (batch, n_classes)."""
        x = self.dropout(self.embedding(ids))

        # Packing makes the LSTM skip the padding at the end of short reports
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        states, _ = self.lstm(packed)
        states, _ = pad_packed_sequence(states, batch_first=True, total_length=ids.size(1))
        is_word = (ids != PAD_ID).unsqueeze(-1)  # (batch, max_len, 1)

        # Combine the word states into one vector per report
        if self.attention is not None:
            # Attention: learned weight per word (padding gets weight 0), then weighted average
            scores = self.attention(states).masked_fill(~is_word, float("-inf"))
            weights = torch.softmax(scores, dim=1)
            report_vector = (weights * states).sum(dim=1)
        else:
            # Max pooling over the real words
            report_vector = states.masked_fill(~is_word, float("-inf")).max(dim=1).values

        return self.classifier(self.dropout(report_vector))


class BiLSTMClassifier(TorchTextClassifier):
    def __init__(self, max_len: int = 256, min_freq: int = 2, max_vocab: int = 50_000,
                 embed_dim: int = 128, hidden_dim: int = 128, num_layers: int = 1,
                 dropout: float = 0.3, attention: bool = True, batch_size: int = 64,
                 lr: float = 1e-3, epochs: int = 10, patience: int = 2, device: str | None = None):
        super().__init__(batch_size=batch_size, lr=lr, epochs=epochs, patience=patience,
                         device=device)
        # Remember all settings so save()/load() can rebuild the same model
        self.settings = dict(max_len=max_len, min_freq=min_freq, max_vocab=max_vocab,
                             embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                             dropout=dropout, attention=attention, batch_size=batch_size, lr=lr,
                             epochs=epochs, patience=patience)

    def build_network(self, train_texts: list[str]) -> None:
        self.vocab = build_vocab(train_texts, self.settings["min_freq"], self.settings["max_vocab"])
        self._create_net()

    def _create_net(self) -> None:
        self.word_to_id = {word: i for i, word in enumerate(self.vocab)}
        s = self.settings
        self.net = BiLSTMNet(len(self.vocab), len(self.classes_), embed_dim=s["embed_dim"],
                             hidden_dim=s["hidden_dim"], num_layers=s["num_layers"],
                             dropout=s["dropout"], attention=s["attention"])

    def make_batch(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Word ids padded to the longest report in the batch (cut at max_len words)."""
        id_lists = []
        for text in texts:
            words = tokenize(text)[: self.settings["max_len"]]
            ids = [self.word_to_id.get(word, UNK_ID) for word in words]
            id_lists.append(torch.tensor(ids or [UNK_ID]))  # an empty report still needs one token
        lengths = torch.tensor([len(ids) for ids in id_lists])
        padded = pad_sequence(id_lists, batch_first=True, padding_value=PAD_ID)
        return {"ids": padded, "lengths": lengths}

    def save(self, path: Path) -> None:
        torch.save({"settings": self.settings, "classes": list(self.classes_),
                    "vocab": self.vocab, "weights": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "BiLSTMClassifier":
        saved = torch.load(path, map_location="cpu")
        model = cls(**saved["settings"], device=device)
        model.classes_ = np.array(saved["classes"], dtype=object)
        model.vocab = saved["vocab"]
        model._create_net()
        model.net.load_state_dict(saved["weights"])
        model.net.to(model.device)
        return model
