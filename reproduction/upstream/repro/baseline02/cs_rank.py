from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _group_boundaries(time_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if time_id.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(time_id[1:] != time_id[:-1]) + 1]
    ends = np.r_[starts[1:], time_id.size]
    return starts.astype(np.int64), ends.astype(np.int64)


def build_cs_rank(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    batch_groups: int = 4_096,
) -> np.ndarray:
    features = tuple(feature_columns)
    missing = [column for column in ("time_id", *features) if column not in frame]
    if missing:
        raise ValueError(f"CS-rank input is missing columns: {missing}")
    if not features:
        raise ValueError("at least one feature column is required")
    if batch_groups <= 0:
        raise ValueError("batch_groups must be positive")
    rows = len(frame)
    if rows == 0:
        return np.empty((0, len(features)), dtype=np.float32)

    time_id = frame["time_id"].to_numpy(dtype=np.int64, copy=False)
    order = np.argsort(time_id, kind="stable")
    inverse = np.empty(rows, dtype=np.int64)
    inverse[order] = np.arange(rows, dtype=np.int64)
    sorted_time = time_id[order]
    sorted_features = frame.loc[:, features].to_numpy(dtype=np.float32, copy=True)[order]
    ranked = np.zeros_like(sorted_features, dtype=np.float32)
    starts, ends = _group_boundaries(sorted_time)
    sizes = ends - starts

    for size in np.unique(sizes):
        group_starts = starts[sizes == size]
        offsets = np.arange(int(size), dtype=np.int64)
        for begin in range(0, group_starts.size, batch_groups):
            current_starts = group_starts[begin : begin + batch_groups]
            row_index = current_starts[:, None] + offsets[None, :]
            values = sorted_features[row_index]
            finite_count = np.isfinite(values).sum(axis=1, keepdims=True)
            ranks = rankdata(values, axis=1, method="average", nan_policy="omit")
            denominator = finite_count - 1
            mapped = np.divide(
                2.0 * (ranks - 1.0),
                denominator,
                out=np.zeros_like(ranks, dtype=np.float64),
                where=denominator > 0,
            ) - np.where(denominator > 0, 1.0, 0.0)
            mapped[~np.isfinite(mapped)] = 0.0
            ranked[row_index] = mapped.astype(np.float32, copy=False)

    return ranked[inverse]
