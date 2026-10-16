"""Competition metric implementations."""

from __future__ import annotations

import numpy as np


def weighted_zero_mean_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weight: np.ndarray,
) -> float:
    """Return weighted zero-mean R-squared used by the competition.

    Unlike ordinary R-squared, the denominator is measured around zero rather
    than around the weighted target mean.
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)

    if y_true.shape != y_pred.shape or y_true.shape != weight.shape:
        raise ValueError("y_true, y_pred, and weight must have identical shapes")
    if not (
        np.isfinite(y_true).all()
        and np.isfinite(y_pred).all()
        and np.isfinite(weight).all()
    ):
        raise ValueError("metric inputs must be finite")
    if np.any(weight < 0):
        raise ValueError("weights must be non-negative")

    denominator = np.sum(weight * y_true * y_true)
    if denominator <= 0:
        raise ValueError("weighted target energy must be positive")
    numerator = np.sum(weight * (y_true - y_pred) ** 2)
    return float(1.0 - numerator / denominator)
