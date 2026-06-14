"""Metrics for Random Forest efficacy prediction."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _safe_correlation(correlation_fn, y_true, y_pred) -> float:
    if len(y_true) < 2:
        return np.nan
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return np.nan
    return float(correlation_fn(y_true, y_pred)[0])


def compute_regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return the core regression metrics used across RF experiments."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "pearson": _safe_correlation(pearsonr, y_true, y_pred),
        "spearman": _safe_correlation(spearmanr, y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
