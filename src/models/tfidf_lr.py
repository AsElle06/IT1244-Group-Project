"""Model 1 (baseline): TF-IDF + Logistic Regression.

Turns each report into weighted word/phrase counts and learns which n-grams point to which
category, with class_weight="balanced" for the rare classes. Strong on template reports where
one phrase (e.g. "signal loss") decides the label, but ignores word order.

Owner: James. Status: placeholder. Implements the TextClassifier interface (src/interfaces.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.interfaces import Labels, Texts

TODO = "TODO (James)"


class TfidfLR:
    def __init__(self, ngram_range: tuple[int, int] = (1, 2), min_df: int = 2, C: float = 1.0):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.C = C
        self.classes_: np.ndarray = np.array([])

    def fit(self, texts: Texts, labels: Labels) -> "TfidfLR":
        raise NotImplementedError(TODO)

    def predict(self, texts: Texts) -> np.ndarray:
        raise NotImplementedError(TODO)

    def predict_proba(self, texts: Texts) -> np.ndarray:
        raise NotImplementedError(TODO)

    def top_features(self, k: int = 10) -> pd.DataFrame:
        """One column per class, k rows: its highest-weighted n-grams (interpretability check)."""
        raise NotImplementedError(TODO)

    def save(self, path: Path) -> None:
        raise NotImplementedError(TODO)

    @classmethod
    def load(cls, path: Path) -> "TfidfLR":
        raise NotImplementedError(TODO)
