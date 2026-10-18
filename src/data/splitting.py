"""Train/test split and cross-validation folds (evaluation plan).

Stratified 80/20 split on the label with a fixed seed, grouped so that near-identical reports
(same cleaning.template_key) stay on one side; the test set is used only once, for the final
comparison. Stratified, grouped 5-fold cross-validation on the training set for tuning.

Owner: Suling. Status: placeholder. Keep the signatures and return types.
"""
import pandas as pd

from src import config
from src.interfaces import Fold

TODO = "TODO (Suling)"


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified (on config.LABEL_COL) split with config.TEST_SIZE and config.SEED.

    Near-identical descriptions (same cleaning.template_key) must never end up in both sets:
    exact-text matching misses copies of a template that differ only in dates, IDs or values.

    Returns (train_df, test_df), both with the same columns as df and index reset to 0..n-1.
    """
    raise NotImplementedError(TODO)


def cv_folds(train_df: pd.DataFrame, n_folds: int = config.N_FOLDS) -> list[Fold]:
    """Stratified cross-validation folds on the training set, grouped by cleaning.template_key.

    Returns n_folds (train_positions, val_positions) pairs of integer row positions.
    """
    raise NotImplementedError(TODO)
