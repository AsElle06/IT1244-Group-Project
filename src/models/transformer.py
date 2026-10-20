"""Model 3: fine-tuned pretrained transformer (DistilBERT or Bio_ClinicalBERT).

A pretrained language model reads each word in the context of the whole report and brings
knowledge from pretraining, which may help rare classes and free-text reports. Trained with the
same shared code as the BiLSTM (src/models/torch_training.py), so both are trained the same way.

Owner: Suling. Status: placeholder, to be planned after the BiLSTM.
Implements the TextClassifier interface (src/interfaces.py).
"""
from pathlib import Path

import numpy as np

from src.interfaces import Labels, Texts

TODO = "TODO (Suling): planned after the BiLSTM"

DISTILBERT = "distilbert-base-uncased"
CLINICAL_BERT = "emilyalsentzer/Bio_ClinicalBERT"


class TransformerClassifier:
    def __init__(self, model_name: str = DISTILBERT, max_len: int = 256, device: str | None = None):
        self.model_name = model_name
        self.max_len = max_len  # in sub-word tokens (max 512); decided when the transformer is planned
        self.device = device
        self.classes_: np.ndarray = np.array([])

    def fit(self, texts: Texts, labels: Labels) -> "TransformerClassifier":
        raise NotImplementedError(TODO)

    def predict(self, texts: Texts) -> np.ndarray:
        raise NotImplementedError(TODO)

    def predict_proba(self, texts: Texts) -> np.ndarray:
        raise NotImplementedError(TODO)

    def save(self, path: Path) -> None:
        raise NotImplementedError(TODO)

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "TransformerClassifier":
        raise NotImplementedError(TODO)
