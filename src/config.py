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
# Google Drive folders: the dataset is downloaded from DATA_URL when data/ has no dataset file;
# missing trained weights are downloaded from WEIGHTS_URL when the notebook runs with TRAIN = False
DATA_URL = "https://drive.google.com/drive/folders/1fmwK3pREwZvBy02zrfxeUNDK-461Kqbe"
WEIGHTS_URL = "https://drive.google.com/drive/folders/17BsA7F9oVMgjUuJ_sb79y4z7yqE-fekn"

SEED = 42

# Columns
TEXT_COL = "event_description"   # the only model input
LABEL_COL = "primary_category"   # IMDRF problem code, e.g. A07, A09, A13
BRAND_COL = "brand_name"
IMDRF_COL = "imdrf_top_categories"
# Used for analysis only, never as model input
ANALYSIS_COLS = [BRAND_COL, "problem_terms", IMDRF_COL, "manufacturer_narrative"]

# Folds: every report is in one of N_FOLDS folds; each fold is the test set once (5-fold rotation)
N_FOLDS = 5
TEST_SIZE = 1 / N_FOLDS  # share of the reports in the test fold of each turn

# A cleaned text that appears more than this many times is randomly cut to this many copies,
# so no single template makes one fold much heavier in its class (main.ipynb section 3)
MAX_COPIES_PER_TEXT = 200

# Classes with fewer training reports than this are merged into one "OTHER" class, decided
# once on the whole dataset (150 training reports per turn = 150 / (1 - TEST_SIZE) reports in all).
# 150 keeps A10 and A12 (~190 training, ~48 test reports each) and merges A15 (~98 training,
# ~24 test) and smaller: too few test reports for a reliable F1 (main.ipynb section 2.1).
MIN_CLASS_COUNT = 150
