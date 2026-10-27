"""Model 3: fine-tuned DistilBERT.

    words -> sub-word tokens (the first 384, e.g. "allegation" -> "all", "##ega", "##tion")
    -> pretrained DistilBERT (6 layers, 66M weights, already trained on English text): one vector
       per token, each read in the context of the whole report
    -> the vector of the first token ([CLS]) -> new classification layers -> 12 class scores

Unlike the BiLSTM, it starts from a model that already knows English, which may help rare classes
and one-off reports. Trained with the same shared code as the BiLSTM (src/models/torch_training.py):
same validation split, class-weighted loss and early stopping. Settings that differ, because the
model is pretrained (learning rate, warm-up, weight decay, epochs), are explained in main.ipynb 5.3.

Owner: Suling. Status: done. Implements the TextClassifier interface (src/interfaces.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from src import config
from src.helpers import get_device, set_seed
from src.interfaces import Labels, Texts
from src.models import torch_training

DISTILBERT = "distilbert-base-uncased"


class TransformerNet(nn.Module):
    """The Hugging Face model, returning plain class scores (the shared training loop expects
    net(**inputs) to give a (batch, n_classes) tensor)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """input_ids: (batch, max_len) token ids. attention_mask: 1 for real tokens, 0 for padding."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


class TransformerClassifier:
    def __init__(self, model_name: str = DISTILBERT, max_len: int = 384, batch_size: int = 32,
                 lr: float = 2e-5, weight_decay: float = 0.01, warmup: float = 0.1, epochs: int = 4,
                 patience: int = 2, mixed_precision: bool = True, device: str | None = None):
        # All fixed settings (no tuning), kept so save()/load() can rebuild the same model
        self.settings = dict(model_name=model_name, max_len=max_len, batch_size=batch_size, lr=lr,
                             weight_decay=weight_decay, warmup=warmup, epochs=epochs,
                             patience=patience, mixed_precision=mixed_precision)
        self.device = device or get_device()
        self.classes_: np.ndarray = np.array([])
        self.history_: list[dict] = []
        # The tokenizer is not trained: it is the pretrained model's, downloaded by name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def fit(self, texts: Texts, labels: Labels) -> "TransformerClassifier":
        """Train on the given reports (in the rotation: the four training folds)."""
        s = self.settings
        set_seed(config.SEED)
        texts, labels = [t if isinstance(t, str) else "" for t in texts], list(labels)

        # 1. Hold out ~10% for early stopping (the same split as the BiLSTM's)
        train_texts, train_labels, val_texts, val_labels = torch_training.split_validation(texts, labels)

        # 2. Classes from the training part; pretrained DistilBERT with a new 12-class layer
        self.classes_ = np.array(sorted(set(labels)), dtype=object)
        self._build_net(pretrained=True)

        # 3. Train, keeping the best epoch on validation macro-F1
        train_ids, val_ids = self._label_ids(train_labels), self._label_ids(val_labels)
        self.history_ = torch_training.train_with_early_stopping(
            self.net,
            train_loader=self._loader(train_texts, train_ids, shuffle=True),
            val_loader=self._loader(val_texts, val_ids),
            weights=torch_training.class_weights(train_ids, len(self.classes_)),
            epochs=s["epochs"], lr=s["lr"], patience=s["patience"], device=self.device,
            weight_decay=s["weight_decay"], make_scheduler=self._warmup_schedule,
            mixed_precision=s["mixed_precision"])
        return self

    def predict_proba(self, texts: Texts) -> np.ndarray:
        texts = [t if isinstance(t, str) else "" for t in texts]
        dummy_labels = np.zeros(len(texts), dtype=int)
        return torch_training.predict_proba(self.net, self._loader(texts, dummy_labels), self.device,
                                            self.settings["mixed_precision"])

    def predict(self, texts: Texts) -> np.ndarray:
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]

    def save(self, path: Path) -> None:
        """Save settings, classes, weights (32-bit) and training history to one .pt file (~268 MB),
        and the history (loss and validation macro-F1 per epoch) to <name>_history.csv."""
        path = Path(path)
        torch.save({"settings": self.settings, "classes": list(self.classes_),
                    "weights": self.net.state_dict(), "history": self.history_}, path)
        pd.DataFrame(self.history_).to_csv(path.with_name(f"{path.stem}_history.csv"), index=False)

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "TransformerClassifier":
        saved = torch.load(path, map_location="cpu")
        model = cls(**saved["settings"], device=device)
        model.classes_ = np.array(saved["classes"], dtype=object)
        model.history_ = saved["history"]
        model._build_net(pretrained=False)  # our trained weights replace the pretrained ones
        model.net.load_state_dict(saved["weights"])
        model.net.to(model.device)
        return model

    # Helpers -------------------------------------------------------------------------

    def _build_net(self, pretrained: bool) -> None:
        """pretrained=True: download DistilBERT's pretrained weights (for training).
        pretrained=False: same network with random weights, only the small config file is
        downloaded (for load(), which then fills in our trained weights)."""
        name, n_classes = self.settings["model_name"], len(self.classes_)
        if pretrained:
            model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=n_classes)
        else:
            model_config = AutoConfig.from_pretrained(name, num_labels=n_classes)
            model = AutoModelForSequenceClassification.from_config(model_config)
        self.net = TransformerNet(model)

    def _warmup_schedule(self, optimizer: torch.optim.Optimizer, total_steps: int):
        """Learning rate rises linearly from 0 to lr over the first `warmup` share of the steps,
        then falls linearly to 0 at the last step of the last epoch."""
        return get_linear_schedule_with_warmup(optimizer, int(self.settings["warmup"] * total_steps),
                                               total_steps)

    def _label_ids(self, labels: list[str]) -> np.ndarray:
        position = {label: i for i, label in enumerate(self.classes_)}
        return np.array([position[label] for label in labels])

    def _loader(self, texts: list[str], label_ids: np.ndarray, shuffle: bool = False):
        # Each report is turned into token ids once, not again in every epoch:
        # the first max_len tokens, counting the [CLS] and [SEP] markers the model adds
        token_ids = self.tokenizer(texts, truncation=True, max_length=self.settings["max_len"])["input_ids"]
        return torch_training.make_loader([torch.tensor(ids) for ids in token_ids], label_ids,
                                          self.settings["batch_size"], self._collate, shuffle)

    def _collate(self, id_lists: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Pad a batch of token-id lists to the longest one in the batch; the attention mask tells
        the model which positions are padding."""
        ids = pad_sequence(id_lists, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        lengths = torch.tensor([len(token_ids) for token_ids in id_lists])
        attention_mask = (torch.arange(ids.size(1)).unsqueeze(0) < lengths.unsqueeze(1)).long()
        return {"input_ids": ids, "attention_mask": attention_mask}
