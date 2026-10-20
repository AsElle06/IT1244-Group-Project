# Trained model weights

`main.ipynb` saves each trained model here (with `TRAIN = True`) and loads it from here
(with `TRAIN = False`), so results can be reproduced without retraining.

| File | Model |
|---|---|
| `tfidf_lr.joblib` | TF-IDF + Logistic Regression |
| `bilstm_fold{k}.pt` | BiLSTM of turn k (settings, vocabulary and weights, ~5 MB) |
| `bilstm_fold{k}_history.csv` | BiLSTM of turn k: training loss and validation macro-F1 per epoch |
| `transformer/` | fine-tuned transformer (Hugging Face weights, tokenizer, `classifier.json`) |

The weights are not committed to git. Files over 10 MB go on Google Drive, with links in
`links.txt` at the project root.
