from __future__ import annotations

"""E2 weighted-Ridge base learner for ``cs_rank_linear``.

The linear member solves a weighted ridge on the *same* 323 within-time rank
features and residual label as the frozen ``cs_rank`` LGBM member; only the base
learner changes. Regularization is chosen on the Fold7 child score alone — the
record-only cross-sectional IC diagnostics (``cs_rank_ic``/``cs_pearson_ic``) are
computed for every ratio but never enter :func:`select_ratio`.
"""

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
from scipy.stats import rankdata


@dataclass(frozen=True)
class WeightedStats:
    """Weight-normalized sufficient statistics ``Sxx = Σw·xxᵀ/Σw`` etc."""

    xx: np.ndarray
    xy: np.ndarray
    weight_sum: float


@dataclass(frozen=True)
class LinearFit:
    coefficients: np.ndarray
    penalty: float
    lambda_ratio: float


def sufficient_statistics(
    batches: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> WeightedStats:
    """Accumulate raw weighted ``xᵀx``/``xᵀy``/``Σw`` then normalize once.

    Streaming one partition at a time keeps a single 323-column matrix resident.
    Summing raw moments and dividing by the total weight at the very end makes
    the result batch-order and batch-count invariant.
    """
    raw_xx: np.ndarray | None = None
    raw_xy: np.ndarray | None = None
    total_w = 0.0
    for x, y, w in batches:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        if x.ndim != 2 or x.shape[0] != y.size or w.size != y.size:
            raise ValueError("linear batch arrays have inconsistent shapes")
        xw = x * w[:, None]
        contribution_xx = xw.T @ x
        contribution_xy = xw.T @ y
        if raw_xx is None:
            raw_xx = contribution_xx
            raw_xy = contribution_xy
        else:
            raw_xx += contribution_xx
            raw_xy += contribution_xy
        total_w += float(w.sum())
    if raw_xx is None or total_w <= 0.0:
        raise ValueError("linear batches must contain positive total weight")
    return WeightedStats(
        xx=raw_xx / total_w,
        xy=raw_xy / total_w,
        weight_sum=float(total_w),
    )


def solve_weighted_ridge(stats: WeightedStats, lambda_ratio: float) -> LinearFit:
    """Solve ``(Sxx + λI)β = Sxy`` with ``λ = ratio·trace(Sxx)/n``."""
    ratio = float(lambda_ratio)
    xx = np.asarray(stats.xx, dtype=np.float64)
    xy = np.asarray(stats.xy, dtype=np.float64).reshape(-1)
    if not np.isfinite(xx).all() or not np.isfinite(xy).all():
        raise ValueError("weighted statistics contain non-finite values")
    if not np.isfinite(ratio) or ratio < 0.0:
        raise ValueError("lambda_ratio must be finite and nonnegative")
    n = xx.shape[0]
    penalty = ratio * float(np.trace(xx)) / n
    beta = np.linalg.solve(xx + penalty * np.eye(n), xy)
    if not np.isfinite(beta).all():
        raise ValueError("weighted ridge produced non-finite coefficients")
    return LinearFit(coefficients=beta, penalty=float(penalty), lambda_ratio=ratio)


def select_ratio(
    score_by_ratio: Mapping[float, float], tolerance: float = 1e-12
) -> float:
    """Highest child score wins; ties within ``tolerance`` pick the larger ratio."""
    if not score_by_ratio:
        raise ValueError("score_by_ratio cannot be empty")
    best = max(float(score) for score in score_by_ratio.values())
    eligible = [
        float(ratio)
        for ratio, score in score_by_ratio.items()
        if float(score) >= best - float(tolerance)
    ]
    return max(eligible)


def _weighted_correlation(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float | None:
    """Weighted Pearson correlation; ``None`` when either side has no variance."""
    total = float(w.sum())
    if total <= 0.0:
        return None
    p = w / total
    ca = a - float(np.sum(p * a))
    cb = b - float(np.sum(p * b))
    var_a = float(np.sum(p * ca * ca))
    var_b = float(np.sum(p * cb * cb))
    if var_a <= 1e-24 or var_b <= 1e-24:
        return None
    return float(np.sum(p * ca * cb) / np.sqrt(var_a * var_b))


def _per_time_ic(
    prediction: np.ndarray,
    label: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
    spearman: bool,
) -> float:
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    lab = np.asarray(label, dtype=np.float64).reshape(-1)
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    times = np.asarray(time_id).reshape(-1)
    correlations: list[float] = []
    for value in np.unique(times):
        mask = times == value
        if int(mask.sum()) < 2:
            continue
        a = pred[mask]
        b = lab[mask]
        gw = w[mask]
        if spearman:
            a = rankdata(a)
            b = rankdata(b)
        corr = _weighted_correlation(a, b, gw)
        if corr is not None:
            correlations.append(corr)
    if not correlations:
        return 0.0
    return float(np.mean(correlations))


def weighted_per_time_pearson(
    prediction: np.ndarray,
    label: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
) -> float:
    """Mean over time of the within-time weighted Pearson correlation."""
    return _per_time_ic(prediction, label, weight, time_id, spearman=False)


def weighted_per_time_spearman(
    prediction: np.ndarray,
    label: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
) -> float:
    """Mean over time of the within-time weighted Spearman (rank) correlation."""
    return _per_time_ic(prediction, label, weight, time_id, spearman=True)


def ratio_diagnostics(
    prediction: np.ndarray,
    label: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
) -> dict[str, float]:
    """Record-only cross-sectional IC — never consumed by :func:`select_ratio`."""
    return {
        "cs_rank_ic": weighted_per_time_spearman(prediction, label, weight, time_id),
        "cs_pearson_ic": weighted_per_time_pearson(prediction, label, weight, time_id),
    }
