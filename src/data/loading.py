"""Reads the course CGM MAUDE dataset from data/ into a DataFrame.

If data/ has no dataset file yet, it is first downloaded from the Google Drive folder
config.DATA_URL.

Owner: Suling, James. Status: done.
"""
from pathlib import Path

import pandas as pd

from src import config

READERS = {".csv": pd.read_csv, ".parquet": pd.read_parquet, ".xlsx": pd.read_excel}


def find_data_file() -> Path | None:
    """Return config.DATA_FILE in data/, or the first .csv/.parquet/.xlsx there; None if missing."""
    if config.DATA_FILE is not None:
        path = config.DATA_DIR / config.DATA_FILE
        return path if path.exists() else None
    if not config.DATA_DIR.exists():
        return None
    for path in sorted(config.DATA_DIR.iterdir()):
        if path.suffix in READERS:
            return path
    return None


def download_data() -> None:
    """Download the Google Drive folder config.DATA_URL into data/."""
    import gdown

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(url=config.DATA_URL, output=str(config.DATA_DIR), quiet=False)


def load_reports(path: Path | None = None) -> pd.DataFrame:
    """Load the dataset file (default: config.DATA_FILE, or the first .csv/.parquet/.xlsx in data/).

    The file is downloaded from config.DATA_URL first if it is not in data/ already.
    Returns one row per report, unchanged, with columns config.TEXT_COL, config.LABEL_COL
    and whichever of config.ANALYSIS_COLS exist in the file.
    """
    if path is None:
        path = find_data_file()
        if path is None:
            download_data()
            path = find_data_file()
        if path is None:
            raise FileNotFoundError(f"No dataset file in {config.DATA_DIR} after downloading {config.DATA_URL}")
    path = Path(path)
    return READERS[path.suffix](path)
