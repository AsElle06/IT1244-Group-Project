"""Shared types that connect the parts of the pipeline.

Every module codes against these, so data -> models -> evaluation plug together:
the model interface (TextClassifier), the score dictionaries and the fold type.
If you need to change one, tell the team first.
"""
from pathlib import Path
from typing import Protocol, Sequence, TypedDict, Union

import numpy as np
import pandas as pd

# A collection of report texts / labels (a list or a pandas column both work)
Texts = Union[Sequence[str], pd.Series]
Labels = Union[Sequence[str], pd.Series]

# One turn of the 5-fold rotation: (training row positions, test row positions)
Fold = tuple[np.ndarray, np.ndarray]


class Scores(TypedDict):
    """Multi-class metrics for one model on one dataset."""
    macro_f1: float      # main metric
    accuracy: float
    weighted_f1: float


class TextClassifier(Protocol):
    """What every model in src/models/ provides (plus a `load(path)` class method)."""

    classes_: np.ndarray  # the class labels, in the column order of predict_proba

    def fit(self, texts: Texts, labels: Labels) -> "TextClassifier":
        """Train on the given reports and return self."""
        ...

    def predict(self, texts: Texts) -> np.ndarray:
        """Predicted label for each text. Shape (n,)."""
        ...

    def predict_proba(self, texts: Texts) -> np.ndarray:
        """Probability of each class for each text. Shape (n, len(classes_)), rows sum to 1."""
        ...

    def save(self, path: Path) -> None:
        """Save the trained model so `load(path)` can restore it."""
        ...
