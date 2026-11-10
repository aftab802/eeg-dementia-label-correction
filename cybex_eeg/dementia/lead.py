"""LEAD-EEG: lightweight biomarker + soft-voting ensemble pipeline.

Base biomarkers per channel: signal energy E = sum x^2, signal power P = E/N,
Shannon spectral entropy H (bits) of the normalised periodogram. Feature
expansion, ANOVA SelectKBest (k = 25), SMOTE and the RF + GB soft-voting
ensemble are wrapped in a single imbalanced-learn Pipeline, so that every
fitted step (scaler, selector, SMOTE, classifiers) is fitted inside each
training fold only.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import periodogram


def biomarkers(epoch: np.ndarray, sfreq: float) -> np.ndarray:
    """(channels, samples) -> (channels, 3) array of [E, P, H]."""
    x = np.asarray(epoch, dtype=float)
    E = (x ** 2).sum(axis=1)
    P = E / x.shape[1]
    _, pxx = periodogram(x, fs=sfreq, axis=1)
    p = pxx / (pxx.sum(axis=1, keepdims=True) + 1e-20)
    H = -(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0)).sum(axis=1)
    return np.stack([E, P, H], axis=1)


def subject_vector(raw, epoch_s: float = 4.0, per_channel: bool = True) -> np.ndarray:
    """Average biomarkers over non-overlapping epochs, reading lazily from disk.

    ``raw`` may be an MNE Raw opened with preload=False; each epoch is read with
    ``get_data(start, stop)`` so the full recording is never held in memory.
    """
    sf = raw.info["sfreq"]
    n = int(epoch_s * sf)
    acc, k = None, 0
    for start in range(0, raw.n_times - n + 1, n):
        b = biomarkers(raw.get_data(start=start, stop=start + n), sf)
        acc = b if acc is None else acc + b
        k += 1
    if k == 0:
        raise ValueError("Recording shorter than one epoch")
    m = acc / k
    return m.ravel() if per_channel else m.mean(axis=0)


def build_pipeline(k: int = 25, degree: int = 2, random_state: int = 42, rf_estimators: int = 300,
                   gb_estimators: int = 100, weights=(1, 1)):
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    ens = VotingClassifier(
        estimators=[("rf", RandomForestClassifier(n_estimators=rf_estimators, random_state=random_state, n_jobs=-1)),
                    ("gb", GradientBoostingClassifier(n_estimators=gb_estimators, random_state=random_state))],
        voting="soft", weights=list(weights))
    return Pipeline([
        ("scale", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("select", SelectKBest(f_classif, k=k)),
        ("smote", SMOTE(random_state=random_state)),
        ("ensemble", ens),
    ])


def n_expanded(n_base: int, degree: int) -> int:
    from math import comb
    return comb(n_base + degree, degree) - 1


def evaluate(X: np.ndarray, y: np.ndarray, n_splits: int = 10, random_state: int = 42, **pipe_kwargs) -> dict:
    """10-fold stratified CV of the full pipeline; pooled out-of-fold metrics."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    k = pipe_kwargs.get("k", 25)
    degree = pipe_kwargs.get("degree", 2)
    if n_expanded(X.shape[1], degree) < k:
        raise ValueError(f"{X.shape[1]} base features at degree {degree} give {n_expanded(X.shape[1], degree)}"
                         f" expanded features, fewer than k={k}. Use per-channel biomarkers or a higher degree.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pred = cross_val_predict(build_pipeline(**pipe_kwargs), X, y, cv=cv)
    fold_acc = [accuracy_score(y[te], pred[te]) for _, te in cv.split(X, y)]
    return {"accuracy_pooled": accuracy_score(y, pred), "accuracy_fold_mean": float(np.mean(fold_acc)),
            "accuracy_fold_sd": float(np.std(fold_acc)), "confusion_matrix": confusion_matrix(y, pred).tolist(),
            "report": classification_report(y, pred, output_dict=True, zero_division=0)}
