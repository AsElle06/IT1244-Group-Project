"""Two baselines: reference points for the evaluation, not real models.

MajorityBaseline
    Ignores the report text completely and always predicts the most common category in the
    training set (A09 Output Problem). The evaluation plan compares every model with it
    ("a simple baseline that always predicts the largest category").
    1. It is the minimum bar. A model that doesn't clearly beat it has learned nothing useful
       from the text.
    2. It shows why macro-F1 is our main metric. After the copy cap A09 is 40% of reports and
       there are 12 classes, so this baseline gets 40% accuracy but a macro-F1 of about 0.05:
       it never predicts the other 11 classes, so each of them scores F1 = 0 (A09's own F1 is
       about 0.57, averaged over 12 classes). High accuracy can hide a model that misses every
       rare failure type.

PhraseLookupBaseline
    The finding of main.ipynb section 2.5 turned into a lookup table, with no machine learning:
    most reports open with "IT WAS REPORTED THAT A <complaint phrase> OCCURRED", and that phrase
    alone predicts the label 96.6% of the time. It remembers each phrase's most common training
    label ("signal loss" -> A13) and predicts the largest class for reports without a known
    phrase. It shows how far template recognition alone gets (expected: accuracy about 0.88,
    macro-F1 about 0.56; F1 = 0 on A02, A10, A12 and OTHER, which have no template), so what
    the models add beyond it is their real value.

Not to be confused with TF-IDF + LR, which the proposal calls our "baseline model": that is
the simplest real classifier. These are reference points below all three models.

Both have the same methods as the models (TextClassifier in src/interfaces.py, except save:
they are recomputed instantly, never saved), so the evaluation in main.ipynb section 6 can loop
over them together with the models; they are the first rows of the results table.

Owner: Lucas. Status: done.
"""
from collections import Counter, defaultdict
import numpy as np
import pandas as pd


from src.analysis import eda
from src.interfaces import Labels, Texts

class MajorityBaseline:
    def __init__(self):
        self.classes_: np.ndarray = np.array([])
        self.majority_label_ = None

    def fit(self, texts: Texts, labels: Labels) -> "MajorityBaseline":
        """Remember the most frequent label in `labels` (texts are ignored)."""
        labels_arr = np.asarray(labels)
        self.classes_ = np.unique(labels_arr)
        
        counts = Counter(labels_arr)
        self.majority_label_ = counts.most_common(1)[0][0]
        return self

    def predict(self, texts: Texts) -> np.ndarray:
        """The most frequent training label for every text. Shape (n,)."""
        n = len(texts)
        return np.full(n, self.majority_label_)

    def predict_proba(self, texts: Texts) -> np.ndarray:
        """Probability 1 for the most frequent class, 0 for the rest. Shape (n, len(classes_))."""
        n = len(texts)
        proba = np.zeros((n, len(self.classes_)), dtype=float)
        majority_idx = int(np.where(self.classes_ == self.majority_label_)[0][0])
        proba[:, majority_idx] = 1.0
        return proba

class PhraseLookupBaseline:
    def __init__(self):
        self.classes_: np.ndarray = np.array([])
        self.majority_label_ = None
        self.phrase_to_label_: dict = {}

    def fit(self, texts: Texts, labels: Labels) -> "PhraseLookupBaseline":
        """Remember each complaint phrase's most common label among the training reports
        (phrases from eda.complaint_phrase, which also works on the cleaned, lowercase text),
        and the most frequent label overall for reports without a known phrase."""
        labels_arr = np.asarray(labels)
        self.classes_ = np.unique(labels_arr)

        counts = Counter(labels_arr)
        self.majority_label_ = counts.most_common(1)[0][0]

        # Extract all phrases at once by passing texts as a pandas Series
        phrases = eda.complaint_phrase(pd.Series(texts))

        # Count label co-occurrences per phrase
        phrase_label_counts = defaultdict(Counter)
        for phrase, label in zip(phrases, labels_arr):
            if pd.notna(phrase):
                phrase_label_counts[phrase][label] += 1

        # Save the most common training label for each observed phrase
        self.phrase_to_label_ = {
            phrase: counter.most_common(1)[0][0]
            for phrase, counter in phrase_label_counts.items()
        }
        return self

    def predict(self, texts: Texts) -> np.ndarray:
        phrases = eda.complaint_phrase(pd.Series(texts))
        preds = [
            self.phrase_to_label_.get(p, self.majority_label_) if pd.notna(p) else self.majority_label_
            for p in phrases
        ]
        return np.asarray(preds)

    def predict_proba(self, texts: Texts) -> np.ndarray:
        """Probability 1 for the predicted class, 0 for the rest. Shape (n, len(classes_))."""
        preds = self.predict(texts)
        return (preds[:, None] == self.classes_[None, :]).astype(float)
