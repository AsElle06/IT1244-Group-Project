# Data

This folder holds the course-provided CGM subset of the FDA MAUDE database
(379,886 adverse event reports). It is not committed to git because of its size.

## Getting the dataset

You do not need to download it by hand. `loading.load_reports()` checks this folder first,
and if there is no dataset file here it downloads `cgm_reports_dataset.csv` (about 300 MB)
from the [Google Drive folder](https://drive.google.com/drive/folders/1HgWnAbksANmolFRrmLkOjy8J3Zhu4V4x)
into this folder. Later runs find the file and skip the download.

- The download needs `gdown` (included in `requirements.txt`).
- To download it manually instead, save `cgm_reports_dataset.csv` from the link above into
  this folder.
- The download link is `DATA_URL` in `src/config.py`.

## File format

- Accepted formats: `.csv`, `.parquet` (needs `pyarrow`) or `.xlsx` (needs `openpyxl`).
- The first such file in this folder is used. To pick a specific file, set `DATA_FILE`
  in `src/config.py`.

## Columns used

| Column | Use |
|---|---|
| `event_description` | model input |
| `primary_category` | label (IMDRF problem code) |
| `brand_name`, `problem_terms`, `imdrf_top_categories`, `manufacturer_narrative` | analysis only |

If the real column names differ, change them in `src/config.py`.
