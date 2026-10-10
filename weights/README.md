# Trained model weights

`main.ipynb` saves each trained model here (with `TRAIN = True`) and loads it from here
(with `TRAIN = False`), so results can be reproduced without retraining.

| File | Model |
|---|---|
| `tfidf_lr.joblib` | TF-IDF + Logistic Regression |
| `bilstm.pt` | BiLSTM (settings, vocabulary and weights) |
| `transformer/` | fine-tuned transformer (Hugging Face weights, tokenizer, `classifier.json`) |

The weights are not committed to git. Files over 10 MB go on Google Drive, with links in
`links.txt` at the project root.
