"""The 5-fold rotation: train a model once per fold and predict the fold it never saw.

For each fold, a fresh model is trained on the other folds and predicts the reports of that
fold. Together the five turns give one prediction for every report (out-of-fold predictions),
each made by a model that did not train on that report or on any copy of its text.
Settings are fixed (no hyper-parameter tuning).

Owner: Suling. Status: done.
"""
from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.interfaces import Fold, TextClassifier


def out_of_fold_proba(make_model: Callable[[], TextClassifier], texts: pd.Series, labels: pd.Series,
                      folds: list[Fold], model_path: Callable[[int], Path] | None = None,
                      train: bool = True, load_model: Callable[[Path], TextClassifier] | None = None,
                      max_train_reports: int | None = None) -> pd.DataFrame:
    """Class probabilities for every report, from the model of the turn in which it was tested.

    make_model: returns a fresh, untrained model with fixed settings.
    model_path: k -> where turn k's model is saved (train=True) or loaded from (train=False).
        With train=True, a turn whose file already exists is loaded instead of retrained, so a
        run that stopped (e.g. a Colab disconnect) continues where it left off. Delete the
        files in weights/ to retrain from scratch.
    load_model: path -> trained model; needed when train=False or to continue a stopped run.
    max_train_reports: train on a stratified random sample of this many training reports
        (e.g. to save compute for the transformer); None = all of them.

    Returns a DataFrame with the same index as texts and one column per class.
    The predicted label of each report is the column with the highest probability.
    """
    turns = []
    for k, (train_positions, test_positions) in enumerate(folds):
        path = model_path(k) if model_path else None
        already_saved = path is not None and Path(path).exists() and load_model is not None
        if train and not already_saved:
            print(f"turn {k}: training")
            train_texts, train_labels = texts.iloc[train_positions], labels.iloc[train_positions]
            if max_train_reports and max_train_reports < len(train_texts):
                train_texts, _, train_labels, _ = train_test_split(
                    train_texts, train_labels, train_size=max_train_reports,
                    stratify=train_labels, random_state=config.SEED)
            model = make_model().fit(train_texts, train_labels)
            if path is not None:
                model.save(path)
        else:
            print(f"turn {k}: loading the saved model from {path}")
            model = load_model(path)

        test_texts = texts.iloc[test_positions]
        turns.append(pd.DataFrame(model.predict_proba(test_texts), index=test_texts.index,
                                  columns=model.classes_))

    # A class missing from one turn's training data gets probability 0 in that turn
    return pd.concat(turns).fillna(0).reindex(texts.index)
