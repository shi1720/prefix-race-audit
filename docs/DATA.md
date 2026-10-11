# Data provenance

The study uses the bundled scikit-learn 1.5.0 digits, Wisconsin diagnostic breast cancer and wine datasets. Original dataset creators and descriptions are recorded in the official documentation:
https://scikit-learn.org/1.5/datasets/toy_dataset.html

Digits originates in the UCI optical recognition dataset; Wisconsin diagnostic breast cancer originates in the work of W. H. Wolberg, W. N. Street and O. L. Mangasarian; wine originates with Forina and colleagues. This artifact stores row indices, labels and derived classifier predictions, rather than redistributing raw feature tables. Dataset creators retain their rights. Classifier code is supplied by scikit-learn under its BSD license.

Latencies in every saved trace are engineered by scripts/run.py. They are not measurements from commercial providers. Prediction and latency seeds, splits, model hyperparameters, and evaluation procedures are specified in the manuscript appendix and scripts. No model parameters are tuned on the evaluation labels.
