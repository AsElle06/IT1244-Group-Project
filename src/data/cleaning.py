"""Cleaning the report texts and labels before they go into the models.

Which steps we use is decided after the dataset analysis (src/analysis/), e.g. what to do with
empty reports, which abbreviations/IDs/dates to normalise, which rare classes to merge.
The functions below are the candidate steps from the proposal; delete any we don't need.
The same steps are applied to the training and test sets.

Owner: Suling, James. Status: placeholder. Keep the signatures and return types.
"""
import pandas as pd

from src import config

TODO = "TODO: decide after the dataset analysis"


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
    """Classes with fewer than min_count reports in the training set (e.g. A24-A27)."""
    raise NotImplementedError(TODO)


def merge_classes(labels: pd.Series, classes_to_merge: list[str], new_label: str = "OTHER") -> pd.Series:
    """Replace every label in classes_to_merge by new_label. Same index and length as labels."""
    raise NotImplementedError(TODO)
