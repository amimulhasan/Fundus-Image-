"""
Common evaluation metrics used across classification, regression and segmentation.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, mean_absolute_error, mean_squared_error, r2_score,
)


def binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan,
    }


def regression_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def dice_coefficient(gt, pred, eps=1e-7):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    return (2.0 * inter + eps) / (gt.sum() + pred.sum() + eps)


def patient_level_bootstrap(metric_fn, y_true, y_pred, patient_ids, n_boot=1000, seed=42):
    """Simple patient-level bootstrap for 95% CI."""
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(patient_ids)
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        mask = np.isin(patient_ids, sampled)
        try:
            values.append(metric_fn(y_true[mask], y_pred[mask]))
        except Exception:
            continue
    if not values:
        return np.nan, np.nan, np.nan
    values = np.array(values)
    return float(np.mean(values)), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
