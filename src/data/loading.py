"""Reads the course CGM MAUDE dataset from data/ into a DataFrame, and downloads the trained weights.

If data/ has no dataset file yet, it is first downloaded from the Google Drive folder
config.DATA_URL. download_weights() fetches the trained models from config.WEIGHTS_URL.

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


def download_weights() -> None:
    """Download the trained weights in the Google Drive folder config.WEIGHTS_URL into weights/.

    Only files that are not in weights/ yet are downloaded, so files already there (e.g. weights
    you have just trained) are never overwritten.
    """
    import gdown

    config.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    # List the folder's files first (no download), then fetch only the missing ones
    files = gdown.download_folder(url=config.WEIGHTS_URL, output=str(config.WEIGHTS_DIR),
                                  quiet=True, skip_download=True)
    missing = [f for f in files if not (config.WEIGHTS_DIR / f.path).exists()]
    for f in missing:
        target = config.WEIGHTS_DIR / f.path
        target.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=f.id, output=str(target), quiet=True)
    print(f"weights/: downloaded {len(missing)} files, {len(files) - len(missing)} were already there")


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
