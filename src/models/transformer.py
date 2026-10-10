"""Model 3: fine-tuned pretrained transformer (DistilBERT or Bio_ClinicalBERT).

A pretrained language model reads each word in the context of the whole report, so it can
tell "alarm did not sound" from "alarm sounded", and brings knowledge from pretraining that
may help rare classes and free-text reports. We add a classification layer on top and
fine-tune all the weights on our reports (as Luschi et al. 2023 did with ClinicalBERT).

Uses the Hugging Face `transformers` library. Needs a GPU (e.g. Colab) for the full data.

Owner: Suling. Status: editing. Implements the TextClassifier interface (src/interfaces.py).
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from src.models.torch_base import TorchTextClassifier

DISTILBERT = "distilbert-base-uncased"
CLINICAL_BERT = "emilyalsentzer/Bio_ClinicalBERT"


class LogitsOnly(nn.Module):
    """Hugging Face models return an output object; the training loop needs just the scores."""

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, **inputs) -> torch.Tensor:
        return self.hf_model(**inputs).logits


class TransformerClassifier(TorchTextClassifier):
    def __init__(self, model_name: str = DISTILBERT, max_len: int = 256, batch_size: int = 16,
                 lr: float = 2e-5, epochs: int = 3, patience: int = 1, weight_decay: float = 0.01,
                 device: str | None = None):
        super().__init__(batch_size=batch_size, lr=lr, epochs=epochs, patience=patience,
                         weight_decay=weight_decay, device=device)
        self.model_name = model_name
        self.max_len = max_len  # in sub-word tokens (max 512)
        # Remember all settings so save()/load() can rebuild the same model
        self.settings = dict(model_name=model_name, max_len=max_len, batch_size=batch_size,
                             lr=lr, epochs=epochs, patience=patience, weight_decay=weight_decay)

    def build_network(self, train_texts: list[str]) -> None:
        """Download the pretrained model and put a new classification layer on top."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        hf_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.classes_),
            id2label={i: label for i, label in enumerate(self.classes_)},
            label2id={label: i for i, label in enumerate(self.classes_)},
            ignore_mismatched_sizes=True,  # replace any existing classification layer
        )
        self.net = LogitsOnly(hf_model)

    def make_batch(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Sub-word token ids and attention mask, padded to the longest report in the batch."""
        tokens = self.tokenizer(texts, truncation=True, max_length=self.max_len, padding=True,
                                return_tensors="pt")
        return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}

    def make_scheduler(self, optimizer, total_steps: int):
        """Standard BERT fine-tuning schedule: warm up over the first 10% of steps, then decay."""
        return get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    def save(self, path: Path) -> None:
        """Save to a folder: Hugging Face weights + tokenizer + our settings and class list."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.net.hf_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        info = {"settings": self.settings, "classes": list(self.classes_)}
        (path / "classifier.json").write_text(json.dumps(info, indent=2))

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "TransformerClassifier":
        path = Path(path)
        info = json.loads((path / "classifier.json").read_text())
        model = cls(**info["settings"], device=device)
        model.classes_ = np.array(info["classes"], dtype=object)
        model.tokenizer = AutoTokenizer.from_pretrained(path)
        model.net = LogitsOnly(AutoModelForSequenceClassification.from_pretrained(path))
        model.net.to(model.device)
        return model
