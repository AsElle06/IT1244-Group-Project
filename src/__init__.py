"""Source code for the CGM device-failure classification project.

Imported by main.ipynb, which runs: load -> analyse -> split -> clean -> models -> evaluate.
    data/        loading, train/test split, cleaning
    analysis/    dataset analysis on the whole dataset, before the split (plots and tables)
    models/      the three classifiers
    evaluation/  majority baseline, metrics and cross-validation
Shared settings and types live in config.py, interfaces.py and helpers.py.
"""
