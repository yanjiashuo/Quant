"""Prediction-only calibration used by the final public submission."""

from __future__ import annotations

import numpy as np


def mean_preserving_std(prediction: np.ndarray, target_std: float) -> np.ndarray:
    """Rescale a prediction to ``target_std`` without changing its mean."""

    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.ndim != 1 or not np.isfinite(prediction).all():
        raise ValueError("prediction must be a finite one-dimensional array")
    if target_std <= 0 or not np.isfinite(target_std):
        raise ValueError("target_std must be finite and positive")

    current_std = prediction.std(ddof=0)
    if current_std <= 0:
        raise ValueError("prediction standard deviation must be positive")
    mean = prediction.mean()
    return (prediction - mean) * (target_std / current_std) + mean


def component_calibration(
    prediction: np.ndarray,
    time_id: np.ndarray,
    asset_id: np.ndarray,
    *,
    time_level: float = 1.0,
    asset_level: float = 1.0,
    residual: float = 1.0,
    weight: np.ndarray | None = None,
) -> np.ndarray:
    """Calibrate global, time-level, asset-level, and residual components.

    The public-test weight was hidden, so the final public submission used
    uniform weights. Supplying ``weight`` is useful for OOF diagnostics only.
    """

    prediction = np.asarray(prediction, dtype=np.float64)
    time_id = np.asarray(time_id)
    asset_id = np.asarray(asset_id)
    if prediction.ndim != 1:
        raise ValueError("prediction must be one-dimensional")
    if prediction.shape != time_id.shape or prediction.shape != asset_id.shape:
        raise ValueError("prediction, time_id, and asset_id must align")
    if not np.isfinite(prediction).all():
        raise ValueError("prediction must be finite")

    if weight is None:
        weight = np.ones_like(prediction)
    else:
        weight = np.asarray(weight, dtype=np.float64)
        if weight.shape != prediction.shape or np.any(weight < 0):
            raise ValueError("weight must align and be non-negative")

    total_weight = weight.sum()
    if total_weight <= 0:
        raise ValueError("weight sum must be positive")
    global_mean = np.sum(prediction * weight) / total_weight

    _, time_index = np.unique(time_id, return_inverse=True)
    time_weight = np.bincount(time_index, weights=weight)
    time_mean = np.bincount(time_index, weights=prediction * weight) / time_weight
    time_component = time_mean[time_index] - global_mean

    deviation = prediction - global_mean - time_component
    _, asset_index = np.unique(asset_id, return_inverse=True)
    asset_weight = np.bincount(asset_index, weights=weight)
    asset_mean = np.bincount(asset_index, weights=deviation * weight) / asset_weight
    asset_component = asset_mean[asset_index]
    residual_component = deviation - asset_component

    return (
        global_mean
        + time_level * time_component
        + asset_level * asset_component
        + residual * residual_component
    )
