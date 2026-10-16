from __future__ import annotations

import numpy as np


def _calibration_arrays(
    target: np.ndarray, prediction: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    z = np.asarray(prediction, dtype=np.float64).reshape(-1)
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    if not (y.size == z.size == w.size) or y.size == 0:
        raise ValueError("calibration arrays must be nonempty and aligned")
    if not np.isfinite(y).all() or not np.isfinite(z).all() or not np.isfinite(w).all():
        raise ValueError("calibration arrays must be finite")
    if np.any(w < 0.0) or float(w.sum()) <= 0.0:
        raise ValueError("calibration weight must be nonnegative with positive sum")
    return y, z, w


def fit_global_scale(
    target: np.ndarray, prediction: np.ndarray, weight: np.ndarray
) -> float:
    y, z, w = _calibration_arrays(target, prediction, weight)
    denominator = float(np.dot(w, z * z))
    if denominator <= 1e-18:
        return 1.0
    return float(np.dot(w, y * z) / denominator)


def fit_shrunk_asset_scales(
    target: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    asset_id: np.ndarray,
    global_scale: float,
    kappa: float,
    clip: tuple[float, float] = (0.5, 1.5),
) -> np.ndarray:
    y, z, w = _calibration_arrays(target, prediction, weight)
    assets = np.asarray(asset_id, dtype=np.int64).reshape(-1)
    if assets.size != y.size or np.any(assets < 0):
        raise ValueError("asset_id must be aligned and nonnegative")
    base = float(global_scale)
    shrink = float(kappa)
    lower, upper = map(float, clip)
    if not np.isfinite(base) or not np.isfinite(shrink) or shrink < 0.0:
        raise ValueError("global_scale and kappa are invalid")
    if not (np.isfinite(lower) and np.isfinite(upper) and lower <= upper):
        raise ValueError("clip bounds are invalid")
    n_assets = int(assets.max()) + 1
    numerator = np.bincount(assets, weights=w * y * z, minlength=n_assets)
    energy = np.bincount(assets, weights=w * z * z, minlength=n_assets)
    positive_energy = energy[energy > 0.0]
    reference_energy = float(np.median(positive_energy)) if positive_energy.size else 0.0
    penalty = shrink * reference_energy
    denominator = energy + penalty
    scales = np.divide(
        numerator + penalty * base,
        denominator,
        out=np.full(n_assets, base, dtype=np.float64),
        where=denominator > 1e-18,
    )
    scales[energy <= 1e-18] = base
    return np.clip(scales, lower, upper)


def apply_asset_scales(
    prediction: np.ndarray, asset_id: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    z = np.asarray(prediction, dtype=np.float64).reshape(-1)
    assets = np.asarray(asset_id, dtype=np.int64).reshape(-1)
    values = np.asarray(scales, dtype=np.float64).reshape(-1)
    if z.size != assets.size or np.any(assets < 0) or np.any(assets >= values.size):
        raise ValueError("prediction, asset_id and scales are not aligned")
    if not np.isfinite(z).all() or not np.isfinite(values).all():
        raise ValueError("prediction and scales must be finite")
    return z * values[assets]
