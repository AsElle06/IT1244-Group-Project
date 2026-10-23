"""Metrics for comparing the models on their out-of-fold predictions (each fold's test turn).

Main metric: macro-F1 (every class counts equally, so rare failure types matter).
Supporting: accuracy, weighted F1, per-class precision/recall/F1, confusion matrix,
ROC curves and ROC-AUC, results per manufacturer, text frequency and report type.

Owner: Lucas. Status: placeholder. Keep the signatures and return types.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from src.interfaces import Labels, Scores


def scores(y_true: Labels, y_pred: Labels) -> Scores:
    """Macro-F1, accuracy and weighted F1."""
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def per_class_report(y_true: Labels, y_pred: Labels) -> pd.DataFrame:
    """Index: class label. Columns: "precision", "recall", "f1-score" (float), "support" (int)."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).T

    # Drop overall summary rows so only individual class labels remain
    df_per_class = df_report.drop(
        index=["accuracy", "macro avg", "weighted avg"], errors="ignore"
    )
    df_per_class["support"] = df_per_class["support"].astype(int)
    return df_per_class[["precision", "recall", "f1-score", "support"]]


def plot_confusion_matrix(
    y_true: Labels,
    y_pred: Labels,
    title: str = "",
    save_as: str | None = None,
) -> None:
    """Row-normalised confusion matrix: where each true class's reports were predicted."""
    classes = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
    )
    plt.title(title or "Normalized Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_as:
        Path(save_as).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_as, dpi=300)
    plt.show()
    plt.close()


def plot_roc_curves(
    y_true: Labels,
    proba: pd.DataFrame,
    title: str = "",
    save_as: str | None = None,
) -> pd.Series:
    """One-vs-rest ROC curve of each class from a model's class probabilities, in one figure.

    proba: one row per report, one column per class (the out-of-fold probabilities of section 5).
    Returns ROC-AUC per class, plus "macro average" (the mean over classes). Index: class.
    """
    y_true_arr = np.asarray(y_true)
    classes = list(proba.columns)
    auc_scores = {}

    plt.figure(figsize=(10, 8))

    for cls in classes:
        # Binarize label for one-vs-rest
        y_binary = (y_true_arr == cls).astype(int)
        y_score = proba[cls].values

        if len(np.unique(y_binary)) > 1:
            auc = roc_auc_score(y_binary, y_score)
            auc_scores[cls] = float(auc)
            fpr, tpr, _ = roc_curve(y_binary, y_score)
            plt.plot(fpr, tpr, label=f"{cls} (AUC = {auc:.3f})")
        else:
            auc_scores[cls] = np.nan

    # Compute macro-average AUC over classes present in y_true
    valid_aucs = [v for v in auc_scores.values() if not np.isnan(v)]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else np.nan
    auc_scores["macro average"] = macro_auc

    plt.plot([0, 1], [0, 1], "k--", label=f"Chance (Macro AUC = {macro_auc:.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title or "One-vs-Rest ROC Curves")
    plt.legend(loc="lower right", bbox_to_anchor=(1.35, 0.0), frameon=True)
    plt.tight_layout()

    if save_as:
        Path(save_as).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_as, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()

    return pd.Series(auc_scores, name="roc_auc")


def macro_f1_by_group(y_true: Labels, y_pred: Labels, groups: Labels) -> pd.Series:
    """Macro-F1 within each group (e.g. turn, manufacturer, text frequency, single- vs
    multi-category). Index: group.

    Within a group, average over the classes that occur in that group's true labels (the
    `labels=` argument of sklearn's f1_score). Otherwise a class that is only predicted in the
    group, never true there, would count as an extra F1 of 0.
    """
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred), "group": np.asarray(groups)})

    group_f1 = {}
    for group_name, group_df in df.groupby("group", observed=True):
        group_classes = np.unique(group_df["y_true"])
        score = f1_score(
            group_df["y_true"],
            group_df["y_pred"],
            labels=group_classes,
            average="macro",
            zero_division=0,
        )
        group_f1[group_name] = float(score)

    return pd.Series(group_f1, name="macro_f1")