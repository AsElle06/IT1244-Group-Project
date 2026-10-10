"""Train/test split and cross-validation folds (evaluation plan).

Stratified 80/20 split on the label with a fixed seed; the test set is used only once, for
the final comparison. Stratified 5-fold cross-validation on the training set for tuning.

Owner: Lucas. Status: placeholder. Keep the signatures and return types.
"""
import pandas as pd

from src import config
from src.interfaces import Fold

TODO = "TODO (Lucas)"


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified (on config.LABEL_COL) split with config.TEST_SIZE and config.SEED.

    Identical descriptions (template reports) must never end up in both sets.

    Returns (train_df, test_df), both with the same columns as df and index reset to 0..n-1.
    """
    raise NotImplementedError(TODO)


def cv_folds(train_df: pd.DataFrame, n_folds: int = config.N_FOLDS) -> list[Fold]:
    """Stratified cross-validation folds on the training set.

    Returns n_folds (train_positions, val_positions) pairs of integer row positions.
    """
    raise NotImplementedError(TODO)
