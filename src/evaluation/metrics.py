"""Metrics for comparing the models on their out-of-fold predictions (each fold's test turn).

Main metric: macro-F1 (every class counts equally, so rare failure types matter).
Supporting: accuracy, weighted F1, per-class precision/recall/F1, confusion matrix,
ROC curves and ROC-AUC, results per manufacturer, text frequency and report type.

Owner: Lucas. Status: placeholder. Keep the signatures and return types.
"""
import pandas as pd

from src.interfaces import Labels, Scores

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


def plot_roc_curves(y_true: Labels, proba: pd.DataFrame, title: str = "",
                    save_as: str | None = None) -> pd.Series:
    """One-vs-rest ROC curve of each class from a model's class probabilities, in one figure.

    proba: one row per report, one column per class (the out-of-fold probabilities of section 5).
    Returns ROC-AUC per class, plus "macro average" (the mean over classes). Index: class.
    """
    raise NotImplementedError(TODO)


def macro_f1_by_group(y_true: Labels, y_pred: Labels, groups: Labels) -> pd.Series:
    """Macro-F1 within each group (e.g. turn, manufacturer, text frequency, single- vs
    multi-category). Index: group.

    Within a group, average over the classes that occur in that group's true labels (the
    `labels=` argument of sklearn's f1_score). Otherwise a class that is only predicted in the
    group, never true there, would count as an extra F1 of 0.
    """
    raise NotImplementedError(TODO)
