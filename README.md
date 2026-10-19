# IT1244 Group Project: CGM Device Failure Classification

Predict each FDA MAUDE continuous glucose monitor (CGM) report's `primary_category`
(IMDRF problem code, e.g. A07 Electrical, A09 Output, A13 Communication) from its
free-text `event_description`. We compare three models of increasing complexity:

1. TF-IDF + Logistic Regression (baseline)
2. Bidirectional LSTM with word embeddings and attention
3. Fine-tuned transformer (DistilBERT or Bio_ClinicalBERT)

Main metric: macro-F1 on a stratified 80/20 split, with hyper-parameters chosen by
stratified 5-fold cross-validation on the training set. Both are grouped so that
near-identical reports stay on one side.

About 90% of reports are templates where one complaint phrase almost decides the label
(notebook section 2.5), so we judge the models on rare classes and non-template text, and
against a phrase-lookup baseline, not on overall accuracy.

## Setup

```
git clone <repo-url> .
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

1. The dataset downloads into `data/` the first time the notebook loads it
   (see [data/README.md](data/README.md)).
2. Open `main.ipynb`, select the `.venv` kernel, and run all cells.
   - `TRAIN = True` trains every model and saves weights to `weights/`.
   - `TRAIN = False` loads saved weights instead (download them first, see `links.txt`).
   - `RUN_CV = False` skips the cross-validation search to save time.

The transformer needs a GPU for the full training set (e.g. Google Colab).

## Structure

The notebook runs the pipeline in order; the code it calls is in `src/`, one folder per stage.

```
main.ipynb                     the whole pipeline: load -> analyse -> split -> clean -> models -> evaluate
src/
  config.py                    settings: folders, seed, column names, split sizes
  interfaces.py                shared types that connect the parts (read this first)
  helpers.py                   device (GPU/CPU) and random seed
  data/
    loading.py                 read the dataset file
    splitting.py               train/test split and cross-validation folds
    cleaning.py                text and label cleaning (steps decided after the analysis)
  analysis/
    eda.py                     dataset analysis plots and tables
  models/
    tfidf_lr.py                model 1: TF-IDF + Logistic Regression
    bilstm.py                  model 2: BiLSTM
    transformer.py             model 3: fine-tuned transformer
    torch_base.py              shared fit/predict for the two PyTorch models
    torch_training.py          shared PyTorch training loop
  evaluation/
    baseline.py                majority-class baseline (reference point, see below)
    metrics.py                 macro-F1 and other metrics on the test set
    cross_validation.py        cross-validation and hyper-parameter search
data/                          the dataset (not in git)
weights/                       trained model weights (not in git)
figures/                       plots saved for the report and slides
links.txt                      Google Drive links for files over 10 MB
```

## Who owns what

| Part | Owner | Code | Notebook | Status |
|---|---|---|---|---|
| Loading the data | Suling, James | `data/loading.py` | 1 | done |
| Dataset analysis | Leanne | `analysis/eda.py` | 2 | done |
| Train/test split | Suling | `data/splitting.py` | 3 | placeholder |
| Cleaning | Suling | `data/cleaning.py` | 4 | done |
| Evaluation (incl. baselines) | Lucas | `evaluation/` | 6 | placeholders |
| TF-IDF + LR | James | `models/tfidf_lr.py` | 5.1 | placeholder |
| BiLSTM, transformer | Suling | `models/bilstm.py`, `transformer.py`, `torch_base.py`, `torch_training.py` | 5.2, 5.3 | done |

## Why the dataset analysis uses the whole dataset

Section 2 of the notebook analyses all 379,886 reports, before the train/test split, because
the split depends on its findings: which labels are missing or too rare to keep, and which
reports are near-copies of the same template that must stay on one side of the split. The
group-aware split also moves whole templates to one side, so the training set's class and brand
shares can drift from the dataset's.

This does not leak the test set into the models, because the analysis only sets rules (drop
NONE, merge A24-A27, mask dates and IDs, group by normalised text). Everything fitted to the
data (TF-IDF vocabulary, class weights, which classes fall under `MIN_CLASS_COUNT`, `MAX_LEN`)
is computed from the training set only. The decisions are summarised in notebook section 2.10.

## The majority baseline

`src/evaluation/baseline.py` ignores the text and always predicts the most common training
category (e.g. A09). It is the minimum bar every model must beat, and it shows why macro-F1 is
our main metric: with A09 at 30% of reports and 12 classes it gets 30% accuracy but a macro-F1
of about 0.04, because every other class scores F1 = 0. It is not the same as TF-IDF + LR,
which the proposal calls our "baseline model" (the simplest real classifier).

A much stronger reference is a phrase-lookup baseline: the template's complaint phrase ->
its most common training label, otherwise the largest class. Section 2.5 of the notebook
shows why: the phrase predicts the label for 96.6% of the reports that have one.

## How the placeholders work

Unfinished functions already have their final signature, type hints and a docstring that
says exactly what they return, and raise `NotImplementedError`. To implement one:

1. Replace the `raise NotImplementedError(...)` with your code.
2. Return exactly the type the docstring describes (e.g. same columns, same index), so the
   notebook and the other parts keep working.
3. Don't change a signature without telling the team.

Every model follows the `TextClassifier` interface in `src/interfaces.py`:
`fit(texts, labels)`, `predict(texts)`, `predict_proba(texts)`, `save(path)`, `load(path)`.

## Submission checklist (from the handout)

- `IT1244_TeamX_Code.zip`: `main.ipynb`, `src/`, trained weights, `Readme.pdf`
  (step-by-step run instructions, based on this file).
- No file over 10 MB in the zip. Upload the dataset and large weights to Google Drive and
  put viewable and downloadable links in `links.txt`.
- References and how we address their limitations are at the top of `main.ipynb`.
- `IT1244_TeamX_Report.pdf` (AAAI format, 2 pages + up to 2 for references/appendix,
  with the "AI Usage Declaration" appendix) and `IT1244_TeamX_Presentation.pdf`.
