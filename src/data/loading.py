"""Reads the course CGM MAUDE dataset from data/ into a DataFrame.

Owner: Suling, James. Status: placeholder. Keep the signature and return type.
"""
from pathlib import Path

import pandas as pd


def load_reports(path: Path | None = None) -> pd.DataFrame:
    """Load the dataset file (default: config.DATA_FILE, or the first .csv/.parquet/.xlsx in data/).

    Returns one row per report, unchanged, with columns config.TEXT_COL, config.LABEL_COL
    and whichever of config.ANALYSIS_COLS exist in the file.
    """
    raise NotImplementedError("TODO (Suling, James)")
