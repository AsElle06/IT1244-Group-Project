"""Project-wide settings: folders, random seed, column names, split sizes.

Every module reads its settings from here, so change values here rather than in the code.
"""
from pathlib import Path

# Folders
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WEIGHTS_DIR = ROOT / "weights"
FIGURES_DIR = ROOT / "figures"

# Dataset file inside data/ (None = use the first .csv/.parquet/.xlsx found there)
DATA_FILE = None
# Google Drive folder the dataset is downloaded from when data/ has no dataset file
DATA_URL = "https://drive.google.com/drive/folders/1HgWnAbksANmolFRrmLkOjy8J3Zhu4V4x"

SEED = 42

# Columns
TEXT_COL = "event_description"   # the only model input
LABEL_COL = "primary_category"   # IMDRF problem code, e.g. A07, A09, A13
BRAND_COL = "brand_name"
IMDRF_COL = "imdrf_top_categories"
# Used for analysis only, never as model input
ANALYSIS_COLS = [BRAND_COL, "problem_terms", IMDRF_COL, "manufacturer_narrative"]

# Train/test split and cross-validation
TEST_SIZE = 0.2
N_FOLDS = 5

# Classes with fewer training reports than this are merged into one "OTHER" class.
# 150 keeps A10 and A12 (~190 training, ~48 test reports each) and merges A15 (~98 training,
# ~24 test) and smaller: too few test reports for a reliable F1 (main.ipynb section 2.1).
MIN_CLASS_COUNT = 150
