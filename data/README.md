# Data

Put the course-provided CGM subset of the FDA MAUDE database here
(379,886 adverse event reports). It is not committed to git because of its size.

- Accepted formats: `.csv`, `.parquet` (needs `pyarrow`) or `.xlsx` (needs `openpyxl`).
- The first such file in this folder is used. To pick a specific file, set `DATA_FILE`
  in `src/config.py`.

Columns used:

| Column | Use |
|---|---|
| `event_description` | model input |
| `primary_category` | label (IMDRF problem code) |
| `brand_name`, `problem_terms`, `imdrf_top_categories`, `manufacturer_narrative` | analysis only |

If the real column names differ, change them in `src/config.py`.
