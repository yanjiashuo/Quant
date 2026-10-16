from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


V2_OPERATIONS = (
    "lag10",
    "lag20",
    "roll_mean20",
    "roll_mean40",
    "roll_std10",
    "roll_std20",
    "delta10",
    "delta20",
)
V2_MAX_HISTORY = 39


def _raw_values(frame: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    values = frame.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True)
    values[np.isinf(values)] = np.nan
    return values


def _finite(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )


def build_rolling_v2(frame: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    features = tuple(feature_columns)
    if not features:
        raise ValueError("feature_columns cannot be empty")
    missing = sorted({"time_id", "asset_id", *features} - set(frame.columns))
    if missing:
        raise ValueError(f"rolling V2 input is missing columns: {missing[:10]}")
    order = np.lexsort(
        (frame["asset_id"].to_numpy(copy=False), frame["time_id"].to_numpy(copy=False))
    )
    if not np.array_equal(order, np.arange(len(frame))):
        raise ValueError("rolling V2 input must be sorted by time_id,asset_id")
    values = _raw_values(frame, features)
    value_frame = pd.DataFrame(values, columns=list(features), index=frame.index)
    grouped = value_frame.groupby(frame["asset_id"].to_numpy(copy=False), sort=False)
    lag10_raw = grouped.shift(10).to_numpy(dtype=np.float32)
    lag20_raw = grouped.shift(20).to_numpy(dtype=np.float32)
    mean20 = grouped.transform(
        lambda column: column.rolling(20, min_periods=1).mean()
    ).to_numpy(dtype=np.float32)
    mean40 = grouped.transform(
        lambda column: column.rolling(40, min_periods=1).mean()
    ).to_numpy(dtype=np.float32)
    std10 = grouped.transform(
        lambda column: column.rolling(10, min_periods=2).std(ddof=0)
    ).to_numpy(dtype=np.float32)
    std20 = grouped.transform(
        lambda column: column.rolling(20, min_periods=2).std(ddof=0)
    ).to_numpy(dtype=np.float32)
    blocks = (
        _finite(lag10_raw),
        _finite(lag20_raw),
        _finite(mean20),
        _finite(mean40),
        _finite(std10),
        _finite(std20),
        _finite(values - lag10_raw),
        _finite(values - lag20_raw),
    )
    return np.column_stack(blocks).astype(np.float32, copy=False)


def _mean_std(sequence: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.vstack(sequence).astype(np.float64, copy=False)
    finite = np.isfinite(stacked)
    count = finite.sum(axis=0)
    total = np.nansum(stacked, axis=0)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    centered = np.where(finite, stacked - mean, 0.0)
    variance = np.divide(
        np.sum(centered * centered, axis=0),
        count,
        out=np.zeros_like(total),
        where=count >= 2,
    )
    standard_deviation = np.sqrt(np.maximum(variance, 0.0))
    standard_deviation[count < 2] = 0.0
    return mean.astype(np.float32), standard_deviation.astype(np.float32)


@dataclass
class RollingV2State:
    feature_columns: tuple[str, ...]
    max_history: int = V2_MAX_HISTORY
    history: dict[int, deque[np.ndarray]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=V2_MAX_HISTORY))
    )
    last_time_id: int | None = None

    def __post_init__(self) -> None:
        self.feature_columns = tuple(self.feature_columns)
        if not self.feature_columns:
            raise ValueError("feature_columns cannot be empty")
        if self.max_history != V2_MAX_HISTORY:
            raise ValueError("Rolling V2 requires 39 prior rows")

    def transform_batch(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.empty((0, len(self.feature_columns) * len(V2_OPERATIONS)), dtype=np.float32)
        unique_times = frame["time_id"].unique()
        if len(unique_times) != 1:
            raise ValueError("streaming batch must contain exactly one time_id")
        time_id = int(unique_times[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("streaming time_id must be strictly increasing")
        if frame["asset_id"].duplicated().any():
            raise ValueError("streaming batch contains duplicate asset_id")
        current = _raw_values(frame, self.feature_columns)
        assets = frame["asset_id"].to_numpy(dtype=np.int64, copy=False)
        rows: list[np.ndarray] = []
        for asset, values in zip(assets, current):
            history = self.history[int(asset)]
            lag10_raw = history[-10] if len(history) >= 10 else np.full_like(values, np.nan)
            lag20_raw = history[-20] if len(history) >= 20 else np.full_like(values, np.nan)
            mean20, _ = _mean_std(list(history)[-19:] + [values])
            mean40, _ = _mean_std(list(history)[-39:] + [values])
            _, std10 = _mean_std(list(history)[-9:] + [values])
            _, std20 = _mean_std(list(history)[-19:] + [values])
            rows.append(
                np.concatenate(
                    (
                        _finite(lag10_raw),
                        _finite(lag20_raw),
                        mean20,
                        mean40,
                        std10,
                        std20,
                        _finite(values - lag10_raw),
                        _finite(values - lag20_raw),
                    )
                )
            )
        for asset, values in zip(assets, current):
            self.history[int(asset)].append(values.copy())
        self.last_time_id = time_id
        return np.vstack(rows).astype(np.float32, copy=False)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int | None]:
        asset_ids = np.asarray(sorted(self.history), dtype=np.int64)
        values = np.full(
            (asset_ids.size, self.max_history, len(self.feature_columns)),
            np.nan,
            dtype=np.float32,
        )
        counts = np.empty(asset_ids.size, dtype=np.int64)
        for index, asset in enumerate(asset_ids):
            block = np.asarray(self.history[int(asset)], dtype=np.float32)
            counts[index] = len(block)
            if len(block):
                values[index, -len(block) :] = block
        return asset_ids, values, counts, self.last_time_id

    def restore(
        self,
        asset_ids: np.ndarray,
        values: np.ndarray,
        counts: np.ndarray,
        last_time_id: int | None,
    ) -> None:
        assets = np.asarray(asset_ids, dtype=np.int64).reshape(-1)
        history_values = np.asarray(values, dtype=np.float32)
        history_counts = np.asarray(counts, dtype=np.int64).reshape(-1)
        expected = (assets.size, self.max_history, len(self.feature_columns))
        if history_values.shape != expected or history_counts.shape != assets.shape:
            raise ValueError("rolling V2 history shapes differ from the frozen contract")
        if np.any((history_counts < 0) | (history_counts > self.max_history)):
            raise ValueError("rolling V2 history count is invalid")
        self.history.clear()
        for asset, block, count in zip(assets, history_values, history_counts):
            for row in block[-int(count) :] if count else ():
                self.history[int(asset)].append(row.copy())
        self.last_time_id = None if last_time_id is None else int(last_time_id)
