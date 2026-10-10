"""Cross-validation on the training set, used to choose each model's hyper-parameters.

Every model is trained on each fold's training part and scored (macro-F1) on its validation
part; the test set is never touched here.

Owner: Lucas. Status: placeholder. Keep the signatures and return types.
"""
from typing import Callable

import pandas as pd

from src.interfaces import Fold, Labels, TextClassifier, Texts

TODO = "TODO (Lucas)"


def cross_validate(make_model: Callable[[], TextClassifier], texts: Texts, labels: Labels,
                   folds: list[Fold]) -> list[float]:
    """Train a fresh model (make_model()) on each fold's training part.

    Returns the macro-F1 on each fold's validation part, one float per fold.
    """
    raise NotImplementedError(TODO)


def grid_search(make_model: Callable[..., TextClassifier], grid: list[dict], texts: Texts,
                labels: Labels, folds: list[Fold]) -> tuple[pd.DataFrame, dict]:
    """Cross-validate make_model(**params) for every params dict in grid.

    Returns (table, best_params). table has one row per params dict, with columns
    "params" (dict), "mean_macro_f1" (float), "std" (float).
    """
    raise NotImplementedError(TODO)
