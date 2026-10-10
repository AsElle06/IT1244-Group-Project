"""Metrics for comparing the models on the test set.

Main metric: macro-F1 (every class counts equally, so rare failure types matter).
Supporting: accuracy, weighted F1, per-class precision/recall/F1, confusion matrix,
results per brand and per report type, and the binary defect-vs-not extension.

Owner: Lucas. Status: placeholder. Keep the signatures and return types.
"""
import numpy as np
import pandas as pd

from src.interfaces import BinaryScores, Labels, Scores

TODO = "TODO (Lucas)"


def scores(y_true: Labels, y_pred: Labels) -> Scores:
    """Macro-F1, accuracy and weighted F1."""
    raise NotImplementedError(TODO)


def per_class_report(y_true: Labels, y_pred: Labels) -> pd.DataFrame:
    """Index: class label. Columns: "precision", "recall", "f1-score" (float), "support" (int)."""
    raise NotImplementedError(TODO)


def plot_confusion_matrix(y_true: Labels, y_pred: Labels, title: str = "",
                          save_as: str | None = None) -> None:
    """Row-normalised confusion matrix: where each true class's reports were predicted."""
    raise NotImplementedError(TODO)


def macro_f1_by_group(y_true: Labels, y_pred: Labels, groups: Labels) -> pd.Series:
    """Macro-F1 within each group (e.g. brand, single- vs multi-category). Index: group."""
    raise NotImplementedError(TODO)


def binary_scores(is_defect: np.ndarray, defect_probability: np.ndarray) -> BinaryScores:
    """Binary extension. is_defect: 0/1 per report; defect_probability: model score per report."""
    raise NotImplementedError(TODO)
