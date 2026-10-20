"""The 5 folds of the train/test rotation (evaluation plan).

Every report goes into one of config.N_FOLDS folds. Each fold is the test set once, while the
models train on the other folds; every report is therefore predicted once, by a model that never
saw it. There is no separate fixed test set.

- Templates (a cleaned text that appears 2+ times): all copies of the text go into the same
  fold, so a model is never tested on a text it trained on. Groups are placed largest first,
  each in the fold that so far has the fewest reports of the group's class.
- Free text (a cleaned text that appears once): spread over the folds at random, stratified by
  class, so every fold has about the same share of each class's free text.

Owner: Suling. Status: done.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src import config
from src.interfaces import Fold


def assign_folds(df: pd.DataFrame, n_folds: int = config.N_FOLDS, seed: int = config.SEED) -> pd.Series:
    """Fold number (0..n_folds-1) of every report. Same index as df.

    Expects cleaned, capped reports (config.TEXT_COL cleaned, config.LABEL_COL final labels).
    """
    texts, labels = df[config.TEXT_COL], df[config.LABEL_COL]
    copies = texts.map(texts.value_counts())
    folds = pd.Series(-1, index=df.index)

    # Templates: whole groups, largest first (random order among equal sizes), each to the fold
    # with the fewest reports of the group's most common class so far
    templates = df[copies > 1]
    sizes = templates.groupby(config.TEXT_COL).size()
    main_class = templates.groupby(config.TEXT_COL)[config.LABEL_COL].agg(lambda s: s.value_counts().index[0])
    order = sizes.sample(frac=1, random_state=seed).sort_values(ascending=False, kind="stable").index
    filled = {label: np.zeros(n_folds) for label in labels.unique()}
    group_fold = {}
    for text in order:
        fold = int(filled[main_class[text]].argmin())
        group_fold[text] = fold
        filled[main_class[text]][fold] += sizes[text]
    folds[templates.index] = templates[config.TEXT_COL].map(group_fold)

    # Free text: stratified random spread
    free = df[copies == 1]
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, positions) in enumerate(splitter.split(free, free[config.LABEL_COL])):
        folds[free.index[positions]] = fold

    folds = folds.astype(int)
    assert (folds >= 0).all(), "every report must be in a fold"
    assert (folds.groupby(texts).nunique() == 1).all(), "a text must never be in two folds"
    return folds


def fold_positions(folds: pd.Series) -> list[Fold]:
    """(training positions, test positions) for each turn: test = one fold, training = the rest."""
    fold_numbers = folds.to_numpy()
    return [(np.flatnonzero(fold_numbers != k), np.flatnonzero(fold_numbers == k))
            for k in sorted(np.unique(fold_numbers))]


def fold_summary(df: pd.DataFrame, folds: pd.Series) -> pd.DataFrame:
    """Share (%) of each class's reports in each fold (ideally 100 / n_folds everywhere).

    Index: class, largest first, plus "ALL". Columns: one per fold, then "reports".
    """
    labels = df[config.LABEL_COL]
    shares = pd.crosstab(labels, folds, normalize="index") * 100
    shares.loc["ALL"] = folds.value_counts(normalize=True).sort_index() * 100
    shares.columns = [f"fold {k}" for k in shares.columns]
    shares["reports"] = pd.concat([labels.value_counts(), pd.Series({"ALL": len(labels)})])
    return shares.sort_values("reports", ascending=False)
