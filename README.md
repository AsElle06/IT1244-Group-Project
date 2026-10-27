# IT1244 Group Project: CGM Device Failure Classification

**Team:** Lim Su Ling, Ho Zi Yun Leanne, James Elliot Toh Zhiheng, Chng Lucas

Predict each FDA MAUDE continuous glucose monitor (CGM) report's `primary_category`
(IMDRF problem code, e.g. A07 Electrical, A09 Output, A13 Communication) from its
free-text `event_description`. We compare three models of increasing complexity:

1. TF-IDF + Logistic Regression
2. Bidirectional LSTM with word embeddings and attention
3. Fine-tuned transformer (DistilBERT)

against two simple baselines (majority class and phrase lookup).

Main metric: macro-F1 over a 5-fold rotation. Every report is in one of 5 folds; each fold is
the test set once while the models train on the other four, so every report is predicted
once by a model that never saw it. All copies of the same cleaned text stay in one fold.
Model settings are fixed (no hyper-parameter tuning).

About 90% of reports are templates where one complaint phrase almost decides the label
(notebook section 2.5), so we judge the models on rare classes and non-template text, and
against a phrase-lookup baseline, not on overall accuracy.

## How to run

Run `main.ipynb` in VS Code on a Google Colab GPU, using the Google Colab extension as the
kernel. Only `main.ipynb` is needed: everything else is downloaded automatically. The
notebook is set to `TRAIN = False`, so it loads our trained weights and reproduces our
results without retraining.

1. Install [VS Code](https://code.visualstudio.com) and, from its Extensions view, the
   **Jupyter** extension and the **Google Colab** extension (by Google).
2. Open `main.ipynb` in VS Code.
3. Click **Select Kernel** (top right of the notebook), choose **Colab**, then
   **New Colab Server**, and pick a **GPU (T4)** server. Sign in with a Google account when
   asked.
4. Click **Run All** and wait for the last cell to finish.

The first code cell downloads our code (`src/`, `requirements.txt`) from our public GitHub
repository, <https://github.com/AsElle06/IT1244-Group-Project>, into the Colab server's
`/content` folder, then installs the packages. After that, the notebook downloads:
- the dataset (about 300 MB) into `data/`;
- our trained weights for all three models (25 files, including 5 DistilBERT files of about
  268 MB each) into `weights/`;
- the DistilBERT tokenizer from Hugging Face.

You do not need to download anything by hand: the trained weights download automatically
from Google Drive when the notebook runs with `TRAIN = False`, and files already in
`weights/` are never overwritten. The Google Drive links are in `links.txt`.

The GPU matters: with `TRAIN = False`, DistilBERT still has to read every report to make its
predictions. That takes a few minutes on the Colab GPU, against about 70 minutes on a CPU.

### What you should see

Section 6.1 prints the macro-F1 of every model on all out-of-fold predictions:

| Model | Macro-F1 | Accuracy |
|---|---|---|
| Majority baseline | 0.048 | 0.399 |
| Phrase-lookup baseline | 0.564 | 0.879 |
| TF-IDF + LR | 0.850 | 0.955 |
| BiLSTM | 0.855 | 0.959 |
| DistilBERT | 0.882 | 0.973 |

The figures are saved to `figures/`.

### Retraining (optional)

Set `TRAIN = True` in the second code cell. This needs a GPU. On a Colab T4, the BiLSTM takes
about 25 minutes for all 5 turns and DistilBERT about 4 hours.

A turn whose model file is already in `weights/` is loaded instead of retrained, so an
interrupted run continues where it stopped. To retrain from scratch, empty `weights/` first.
Colab deletes its files when the runtime disconnects, so copy `weights/` off the server
after training (in VS Code: **Colab: Mount Server To Workspace**).

## Project structure

The notebook runs the pipeline in order: load, analyse, clean, make the folds, train the
models, evaluate. The code it calls is in `src/`, one folder per stage.

| Path | Contents |
|---|---|
| `main.ipynb` | The whole pipeline, with our analysis and results |
| `src/config.py` | Settings: folders, random seed, column names, number of folds, copy cap |
| `src/interfaces.py` | Shared types that connect the parts |
| `src/helpers.py` | Device (GPU/CPU) and random seed |
| `src/data/loading.py` | Read the dataset; download the dataset and weights if missing |
| `src/data/cleaning.py` | Text and label cleaning; cap on copies per text |
| `src/data/splitting.py` | The 5 folds of the train/test rotation |
| `src/analysis/eda.py` | Dataset analysis plots and tables |
| `src/models/tfidf_lr.py` | Model 1: TF-IDF + Logistic Regression |
| `src/models/bilstm.py` | Model 2: BiLSTM with attention |
| `src/models/transformer.py` | Model 3: fine-tuned DistilBERT |
| `src/models/torch_training.py` | Training functions shared by the two PyTorch models |
| `src/evaluation/baseline.py` | Majority and phrase-lookup baselines |
| `src/evaluation/metrics.py` | Macro-F1 and the other metrics on the out-of-fold predictions |
| `src/evaluation/cross_validation.py` | The rotation: train once per fold, predict the fold left out |
| `data/` | The dataset (downloaded automatically, see `data/README.md`) |
| `weights/` | Trained model weights (see `weights/README.md`) |
| `figures/` | Plots saved for the report and slides |
| `links.txt` | Google Drive links for files over 10 MB |
| `requirements.txt` | Python packages |

Every model follows the `TextClassifier` interface in `src/interfaces.py`:
`fit(texts, labels)`, `predict(texts)`, `predict_proba(texts)`, `save(path)` and `load(path)`.
The baselines have the same methods except `save` and `load`.

## Why the dataset analysis uses the whole dataset

Section 2 of the notebook analyses all 379,886 reports, before the folds are made, because
the folds depend on its findings: which labels are missing or too rare to keep, and which
reports are copies of the same template that must stay in one fold.

This does not leak test reports into the models, because the analysis only sets rules (drop
NONE, merge A24-A27, mask dates and IDs, keep copies of a text together, cap copies at 200).
Everything the models learn (TF-IDF vocabulary, class weights, network weights) comes from
each turn's training folds only. The decisions are summarised in notebook section 2.10.

## Why a 5-fold rotation instead of a fixed 80/20 test set

The course project README asks for a fixed 80/20 split with 5-fold cross-validation on the 80% to avoid
overfitting. Our data is about 90% templates, so the main overfitting risk is leakage: copies
of the same text in both training and test. We prevent it directly: all copies of a cleaned
text stay in one fold, and a text that appears more than 200 times is cut to 200 copies so no
template makes one fold much heavier in its class. Each fold is then the test set once, so
every report (including each class's main templates) is tested once by a model that never saw
it, and the results use all the data instead of one 20% sample. Model settings are fixed, so
no choice is tuned on the folds that are reported.

## The baselines

`src/evaluation/baseline.py` has two reference points that are not real models:

- **Majority baseline:** ignores the text and always predicts the most common training
  category (A09). It is the minimum bar every model must beat, and it shows why macro-F1 is
  our main metric. A09 is 40% of reports after the copy cap and there are 12 classes, so it
  gets 40% accuracy but a macro-F1 of 0.05: every other class scores F1 = 0.
- **Phrase-lookup baseline:** the report's complaint phrase -> that phrase's most common
  training label, otherwise the largest class. Section 2.5 of the notebook shows why this is
  strong: the phrase predicts the label for 96.6% of the reports that have one. It reaches
  88% accuracy but a macro-F1 of 0.56, because the classes without a template (A02, A10, A12,
  OTHER) score F1 = 0.

Neither is the same as TF-IDF + LR, which the proposal calls our "baseline model" (the
simplest real classifier).
