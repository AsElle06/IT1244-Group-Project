# Trained model weights

`main.ipynb` saves each trained model here (with `TRAIN = True`) and loads it from here
(with `TRAIN = False`), so results can be reproduced without retraining.

| File | Model |
|---|---|
| `tfidf_lr_fold{k}.joblib` | TF-IDF + Logistic Regression of turn k (vocabulary, IDF weights and coefficients, ~8 MB; saved with scikit-learn 1.9.1) |
| `bilstm_fold{k}.pt` | BiLSTM of turn k (settings, vocabulary and weights, ~5 MB) |
| `bilstm_fold{k}_history.csv` | BiLSTM of turn k: training loss and validation macro-F1 per epoch |
| `transformer_fold{k}.pt` | DistilBERT of turn k (settings, classes and 32-bit weights, ~268 MB). The tokenizer is not saved: it is downloaded from Hugging Face by name |
| `transformer_fold{k}_history.csv` | DistilBERT of turn k: training loss and validation macro-F1 per epoch |

The weights are not committed to git. They are kept in the Google Drive folder
`WEIGHTS_URL` in `src/config.py` (links in `links.txt` at the project root). With
`TRAIN = False`, the notebook downloads any file that is missing here
(`loading.download_weights()`); files already here are never overwritten.
