from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import Ridge


def _vectors(*values: np.ndarray) -> list[np.ndarray]:
    arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in values]
    if not arrays or any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("all inputs must have the same one-dimensional shape")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("metric inputs must be finite")
    return arrays


def weighted_zero_mean_r2(
    y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray
) -> float:
    y_true, y_pred, weight = _vectors(y_true, y_pred, weight)
    if np.any(weight < 0):
        raise ValueError("weights must be nonnegative")
    denominator = float(np.dot(weight, y_true * y_true))
    if denominator <= 0.0:
        return 0.0
    error = y_true - y_pred
    return 1.0 - float(np.dot(weight, error * error)) / denominator


def optimal_scale(
    y_true: np.ndarray, raw_prediction: np.ndarray, weight: np.ndarray
) -> float:
    y_true, raw_prediction, weight = _vectors(y_true, raw_prediction, weight)
    if np.any(weight < 0):
        raise ValueError("weights must be nonnegative")
    denominator = float(np.dot(weight, raw_prediction * raw_prediction))
    if denominator <= 1e-18:
        return 0.0
    return float(np.dot(weight, y_true * raw_prediction) / denominator)


def level_deviation(
    prediction: np.ndarray, time_id: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    time_id = np.asarray(time_id).reshape(-1)
    if prediction.shape != time_id.shape:
        raise ValueError("prediction and time_id must have the same shape")
    if prediction.size == 0:
        return prediction.copy(), prediction.copy()
    if not np.isfinite(prediction).all():
        raise ValueError("prediction must be finite")

    order = np.argsort(time_id, kind="stable")
    sorted_time = time_id[order]
    sorted_prediction = prediction[order]
    starts = np.r_[0, np.flatnonzero(sorted_time[1:] != sorted_time[:-1]) + 1]
    ends = np.r_[starts[1:], prediction.size]
    means = np.add.reduceat(sorted_prediction, starts) / (ends - starts)
    sorted_level = np.repeat(means, ends - starts)
    level = np.empty_like(prediction)
    level[order] = sorted_level
    return level, prediction - level


def build_l88(
    final: np.ndarray,
    time_id: np.ndarray,
    level_scale: float = 0.88,
    deviation_scale: float = 1.30,
) -> np.ndarray:
    level, deviation = level_deviation(final, time_id)
    return level_scale * level + deviation_scale * deviation


def fit_positive_stack(
    member_matrix: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    alpha: float = 0.5,
    member_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    matrix = np.asarray(member_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("member_matrix must be a nonempty two-dimensional array")
    y, weight = _vectors(y, weight)
    if matrix.shape[0] != y.size:
        raise ValueError("member_matrix row count must match y")
    if not np.isfinite(matrix).all():
        raise ValueError("member_matrix must be finite")
    if np.any(weight < 0):
        raise ValueError("weights must be nonnegative")
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")

    names = list(member_names or [f"member_{index}" for index in range(matrix.shape[1])])
    if len(names) != matrix.shape[1]:
        raise ValueError("member_names length must match member_matrix columns")
    scales = np.std(matrix, axis=0)
    scales[scales < 1e-12] = 1.0
    model = Ridge(alpha=alpha, fit_intercept=False, positive=True)
    model.fit(matrix / scales, y, sample_weight=weight)
    normalized = np.asarray(model.coef_, dtype=np.float64)
    effective = normalized / scales
    return {
        "alpha": float(alpha),
        "member_names": names,
        "standard_deviations": scales.tolist(),
        "normalized_coefficients": normalized.tolist(),
        "effective_coefficients": effective.tolist(),
        "nonzero_members": int(np.count_nonzero(effective > 1e-15)),
    }


def predict_positive_stack(member_matrix: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(member_matrix, dtype=np.float64)
    coefficients = np.asarray(fit["effective_coefficients"], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != coefficients.size:
        raise ValueError("member_matrix columns do not match stack coefficients")
    return matrix @ coefficients


def prediction_summary(prediction: np.ndarray) -> dict[str, float | int]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    finite = np.isfinite(prediction)
    result: dict[str, float | int] = {
        "rows": int(prediction.size),
        "finite_rows": int(finite.sum()),
        "finite_fraction": float(finite.mean()) if prediction.size else 1.0,
    }
    if finite.any():
        values = prediction[finite]
        q01, q50, q99 = np.quantile(values, [0.01, 0.50, 0.99])
        result.update(
            mean=float(values.mean()),
            std=float(values.std()),
            q01=float(q01),
            q50=float(q50),
            q99=float(q99),
            max_abs=float(np.max(np.abs(values))),
        )
    return result


def correlation_matrix(member_matrix: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    matrix = np.asarray(member_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("matrix columns must match names")
    return {"names": list(names), "pearson": np.corrcoef(matrix, rowvar=False).tolist()}
