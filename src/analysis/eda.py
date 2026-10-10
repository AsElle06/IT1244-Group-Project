"""Exploratory analysis of the training set: distributions and associations.

One function per planned analysis in the proposal; each comment says which decision it informs.
Plot functions show the figure and, if save_as is given, also save it to config.FIGURES_DIR
for the report and slides.

Owner: Leanne. Status: placeholder. Keep the signatures and return types.
"""
import pandas as pd

TODO = "TODO (Leanne)"


# 1. Class counts -> class weights, merging rare classes, why macro-F1

def class_counts(labels: pd.Series) -> pd.DataFrame:
    """Index: class label, most frequent first. Columns: "count" (int), "share_%" (float)."""
    raise NotImplementedError(TODO)


def plot_class_counts(labels: pd.Series, save_as: str | None = None) -> None:
    """Bar chart of reports per class."""
    raise NotImplementedError(TODO)


# 2. Text length -> max input length for the BiLSTM and transformer

def length_percentiles(texts: pd.Series, percentiles: tuple[int, ...] = (50, 90, 95, 99)) -> pd.Series:
    """Words per description at each percentile. Index: "p50", "p90", ... Values: float."""
    raise NotImplementedError(TODO)


def plot_length_hist(texts: pd.Series, save_as: str | None = None) -> None:
    """Histogram of words per description."""
    raise NotImplementedError(TODO)


def plot_length_by_class(df: pd.DataFrame, save_as: str | None = None) -> None:
    """Description length per class (e.g. one box per class)."""
    raise NotImplementedError(TODO)


# 3. Empty / very short descriptions -> cleaning: drop them or keep them

def short_description_summary(texts: pd.Series) -> pd.DataFrame:
    """Index: thresholds such as "<= 0 words", "<= 3 words". Columns: "reports" (int), "share_%" (float)."""
    raise NotImplementedError(TODO)


# 4. Duplicate (template) descriptions -> how the split keeps them on one side

def duplicate_summary(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Returns (stats, top_templates).

    stats: named counts, e.g. "reports", "unique descriptions", "duplicate texts with >1 label".
    top_templates: the most repeated descriptions. Index: text. Columns: "count" (int),
    "labels" (dict of label -> count).
    """
    raise NotImplementedError(TODO)


# 5. Reports by brand -> per-brand results, templated vs free text

def brand_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Index: brand. Columns: "reports" (int), "duplicate_%" (float), "median_words" (float)."""
    raise NotImplementedError(TODO)


def plot_brand_by_class(df: pd.DataFrame, save_as: str | None = None) -> None:
    """Heatmap of each brand's share of reports in each class."""
    raise NotImplementedError(TODO)


# 6. Reports with more than one IMDRF category -> evaluate single vs multi separately

def count_imdrf_codes(imdrf_categories: pd.Series) -> pd.Series:
    """Number of distinct IMDRF codes listed in each report. Same index as the input, int values."""
    raise NotImplementedError(TODO)


def plot_imdrf_code_counts(imdrf_categories: pd.Series, save_as: str | None = None) -> None:
    """Bar chart: how many reports list 1, 2, 3, ... IMDRF categories."""
    raise NotImplementedError(TODO)


# 7. Most frequent words / phrases per class -> cleaning steps, TF-IDF n-gram range

def top_ngrams_per_class(texts: pd.Series, labels: pd.Series, k: int = 15) -> pd.DataFrame:
    """One column per class, k rows: that class's most frequent words/phrases (str)."""
    raise NotImplementedError(TODO)
