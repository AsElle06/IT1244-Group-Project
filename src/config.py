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

# Classes with fewer training reports than this are merged into one "OTHER" class
MIN_CLASS_COUNT = 50  # TODO: decide after the dataset analysis (class counts)
