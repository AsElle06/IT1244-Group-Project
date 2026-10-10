"""Majority-class baseline: a reference point for the evaluation, not a real model.

What it does
    Ignores the report text completely and always predicts the most common category in the
    training set (e.g. A09 Output Problem). The evaluation plan compares every model with it
    ("a simple baseline that always predicts the largest category").

Why we need it
    1. It is the minimum bar. A model that doesn't clearly beat it has learned nothing useful
       from the text.
    2. It shows why macro-F1 is our main metric. If A09 is 40% of reports and there are 15
       classes, this baseline gets 40% accuracy, which sounds acceptable, but a macro-F1 of
       about 0.04: it never predicts the other 14 classes, so each of them scores F1 = 0
       (A09's own F1 is about 0.57, averaged over 15 classes). High accuracy can hide a model
       that misses every rare failure type.

Not to be confused with TF-IDF + LR, which the proposal calls our "baseline model": that is
the simplest real classifier. This file is the trivial lower bound below all three models.

It has the same methods as the models (TextClassifier in src/interfaces.py), so the evaluation
in main.ipynb section 6 can loop over it together with them; it is the first row of the
results table.

Owner: Lucas. Status: placeholder (about 5 lines: remember the most frequent label in fit,
return it for every text in predict).
"""
from pathlib import Path

import numpy as np

from src.interfaces import Labels, Texts

TODO = "TODO (Lucas)"


class MajorityBaseline:
    def __init__(self):
        self.classes_: np.ndarray = np.array([])

    def fit(self, texts: Texts, labels: Labels) -> "MajorityBaseline":
        """Remember the most frequent label in `labels` (texts are ignored)."""
        raise NotImplementedError(TODO)

    def predict(self, texts: Texts) -> np.ndarray:
        """The most frequent training label for every text. Shape (n,)."""
        raise NotImplementedError(TODO)

    def predict_proba(self, texts: Texts) -> np.ndarray:
        """Probability 1 for the most frequent class, 0 for the rest. Shape (n, len(classes_))."""
        raise NotImplementedError(TODO)

    def save(self, path: Path) -> None:
        raise NotImplementedError(TODO)
