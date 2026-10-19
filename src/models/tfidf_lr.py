"""Model 1 (baseline): TF-IDF + Logistic Regression.

Turns each report into weighted word/phrase counts and learns which n-grams point to which
category, with class_weight="balanced" for the rare classes. Strong on template reports where
one phrase (e.g. "signal loss") decides the label, but ignores word order.

Owner: James. Status: In progress. Implements the TextClassifier interface (src/interfaces.py).
"""
from pathlib import Path

import joblib 
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import sys 

ROOT = Path.cwd() 
if str(ROOT) not in sys.path: 
    sys.path.insert(0, str(ROOT))


from src.interfaces import Labels, Texts

TODO = "TODO (James)"

def _as_text_lists(texts: Texts) -> list[str]: 
    """Accept a list or a pandas column; guard against NaN or non-str entries."""
    return pd.Series(list(texts)).fillna("").astype(str).tolist() 

class TfidfLR:
    def __init__(self, ngram_range: tuple[int, int] = (1, 2), min_df: int = 2, C: float = 1.0):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.C = C
        self.classes_: np.ndarray = np.array([])
        self.pipeline_: Pipeline | None = None

    def _check_fitted(self) -> Pipeline: 
        if self.pipeline_ is None: 
            raise RuntimeError("TfidfLR is not fitted yet. Call fit() or load() first.")
        return self.pipeline_

    def fit(self, texts: Texts, labels: Labels) -> "TfidfLR":
        X = _as_text_lists(texts)
        y = np.asarray(list(labels))
        if len(X) != len(y): 
            raise ValueError(f"texts ({len(X)}) and labels ({len(y)}) differ in length.")

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range = self.ngram_range,
                min_df = self.min_df, 
                lowercase = True, 
                sublinear_tf = True, 
            )), 
            ("clf", LogisticRegression(
                C = self.C, 
                class_weight = "balanced", 
                max_iter = 1000,
                random_state = 0, 
            )), 
        ])
        self.pipeline_.fit(X, y)
        self.classes_ = self.pipeline_.named_steps["clf"].classes_ 
        return self

    def predict(self, texts: Texts) -> np.ndarray:
        return self._check_fitted().predict(_as_text_lists(texts))

    def predict_proba(self, texts: Texts) -> np.ndarray:
        raise self._check_fitted.predict_proba(_as_text_lists(texts))

    def top_features(self, k: int = 10) -> pd.DataFrame:
        """One column per class, k rows: its highest-weighted n-grams (interpretability check)."""
        pipe = self._check_fitted()
        vocab = pipe.named_steps["tfidf"].get_feature_names_out()
        coef = pipe.named_steps["clf"].coef_ 

        if coef.shape[0] == 1: 
            coef = np.vstack([-coef[0], coef[0]])

        #Top - k
        k = min(k, len(vocab))
        cols = {}
        for i, cls_name in enumerate(self.classes_): 
            top_idx = np.argsort(coef[i])[:: -1][:k]
            cols[cls_name] = vocab[top_idx]
        return pd.DataFrame(cols)

    def save(self, path: Path) -> None:
        pipe = self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents = True, exist_ok = True)
        joblib.dump(
            {
                "params": {"ngram_range": self.ngram_range, "min_df": self.min_df, "C": self.C}, 
                "pipeline": pipe, 
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "TfidfLR":
        state = joblib.load(Path(path))
        model = cls(**state["params"])
        model.pipeline_ = state["pipeline"]
        model.classes_ = model.pipeline_.named_steps["clf"].classes_
        return model
