"""Cleaning the report texts and labels before they go into the models.

Steps decided by the dataset analysis (main.ipynb section 2):
- Drop reports labelled NONE: they have no IMDRF code and no problem term, so the label is
  missing. Drop the reports with an empty description.
- Merge A24-A27 into one class, as the course README's label table does
  ("Unknown / No device problem").
- Merge classes too small for a reliable test F1 into "OTHER" (config.MIN_CLASS_COUNT).
- Mask redaction markers such as "(B)(6)", dates, years and IDs in the text.
The functions below are the candidate steps from the proposal; delete any we don't need.
The same steps are applied to the training and test sets.

Owner: Suling. Status: steps decided, functions still placeholders (template_key is done). Keep the signatures and return types.
"""
import pandas as pd

from src import config

TODO = "TODO: decide after the dataset analysis"

REDACTION_MARKER = r"\(\s*[Bb]\s*\)\s*\(\s*\d\s*\)"  # e.g. "(B)(6)": text removed by the FDA


def template_key(texts: pd.Series) -> pd.Series:
    """Each report text reduced to its template, for finding near-identical reports.

    Upper-cases the text and removes redaction markers, every token containing a digit (dates,
    years, IDs, glucose values) and punctuation. Reports with the same key differ only in those
    details, so the train/test split and the CV folds must keep them on one side, and the
    dataset analysis counts them as one description. Same index as texts; missing values give "".
    """
    return (
        texts.fillna("").astype(str).str.upper()
        .str.replace(REDACTION_MARKER, " ", regex=True)
        .str.replace(r"\S*\d\S*", " ", regex=True)
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.split().str.join(" ")
    )


def clean_text(text: str) -> str:
    """Normalise one report text (e.g. lowercase, expand abbreviations, replace IDs and dates).

    Must return "" for missing values (NaN / None).
    """
    raise NotImplementedError(TODO)


def clean_reports(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all chosen cleaning steps to a train or test DataFrame.

    Returns the same columns as df with config.TEXT_COL cleaned (clean_text), possibly fewer
    rows (e.g. reports with no usable text removed), index reset to 0..n-1.
    """
    raise NotImplementedError(TODO)


def find_rare_classes(train_labels: pd.Series, min_count: int = config.MIN_CLASS_COUNT) -> list[str]:
    """Classes with fewer than min_count reports in the training set (e.g. A08, A14)."""
    raise NotImplementedError(TODO)


def merge_classes(labels: pd.Series, classes_to_merge: list[str], new_label: str = "OTHER") -> pd.Series:
    """Replace every label in classes_to_merge by new_label. Same index and length as labels."""
    raise NotImplementedError(TODO)
