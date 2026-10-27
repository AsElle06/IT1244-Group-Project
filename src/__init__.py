"""Source code for the CGM device-failure classification project.

Imported by main.ipynb, which runs: load -> analyse -> clean -> folds -> models -> evaluate.
    data/        loading, cleaning, the 5 folds of the train/test rotation
    analysis/    dataset analysis on the whole dataset, before the folds are made (plots and tables)
    models/      the three classifiers
    evaluation/  majority and phrase-lookup baselines, metrics and the 5-fold rotation
Shared settings and types live in config.py, interfaces.py and helpers.py.
"""
