"""The classifiers. Each one follows the TextClassifier interface in src/interfaces.py.

    tfidf_lr.py        model 1: TF-IDF + Logistic Regression (James)
    bilstm.py          model 2: BiLSTM with attention (Suling)
    transformer.py     model 3: fine-tuned DistilBERT / Bio_ClinicalBERT (Suling)
    torch_base.py      shared fit/predict code for the two PyTorch models
    torch_training.py  shared PyTorch training loop
"""
