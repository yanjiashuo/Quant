from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


V1_OPERATIONS = ("lag1", "lag2", "lag5", "roll_mean5", "roll_mean10", "delta1")


def _raw_values(frame: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    values = frame.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True)
    values[np.isinf(values)] = np.nan
    return values


def rolling_column_names(feature_columns: Sequence[str], operations: Sequence[str]) -> list[str]:
    return [f"{operation}__{feature}" for operation in operations for feature in feature_columns]


def build_rolling_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    operations: Sequence[str] = V1_OPERATIONS,
) -> np.ndarray:
    if not feature_columns:
        raise ValueError("feature_columns cannot be empty")
    if tuple(operations) != V1_OPERATIONS:
        raise ValueError(f"Baseline 00 requires operations {V1_OPERATIONS}")
    required = {"time_id", "asset_id", *feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"rolling input missing columns: {missing[:10]}")

    order = np.lexsort(
        (frame["asset_id"].to_numpy(copy=False), frame["time_id"].to_numpy(copy=False))
    )
    if not np.array_equal(order, np.arange(len(frame))):
        raise ValueError("rolling input must be sorted by time_id, asset_id")

    values = _raw_values(frame, feature_columns)
    value_frame = pd.DataFrame(values, columns=list(feature_columns), index=frame.index)
    grouped = value_frame.groupby(frame["asset_id"].to_numpy(copy=False), sort=False)
    blocks: list[np.ndarray] = []
    lags: dict[int, np.ndarray] = {}
    for lag in (1, 2, 5):
        block = grouped.shift(lag).to_numpy(dtype=np.float32)
        block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
        lags[lag] = block
        blocks.append(block)
    for window in (5, 10):
        block = grouped.transform(
            lambda column: column.rolling(window, min_periods=1).mean()
        ).to_numpy(dtype=np.float32)
        blocks.append(np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0))
    blocks.append(np.nan_to_num(values - lags[1], nan=0.0, posinf=0.0, neginf=0.0))
    return np.column_stack(blocks).astype(np.float32, copy=False)


@dataclass
class RollingFeatureState:
    feature_columns: tuple[str, ...]
    max_history: int = 9
    history: dict[int, deque[np.ndarray]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=9))
    )
    last_time_id: int | None = None

    def __post_init__(self) -> None:
        self.feature_columns = tuple(self.feature_columns)
        if not self.feature_columns:
            raise ValueError("feature_columns cannot be empty")
        if self.max_history != 9:
            raise ValueError("Baseline 00 requires nine prior rows for roll_mean10")

    def transform_batch(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.empty((0, len(self.feature_columns) * 6), dtype=np.float32)
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
            lag_blocks = []
            for lag in (1, 2, 5):
                prior = history[-lag] if len(history) >= lag else np.full_like(values, np.nan)
                lag_blocks.append(np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0))

            rolling_blocks = []
            for window in (5, 10):
                previous_count = min(len(history), window - 1)
                sequence = list(history)[-previous_count:] + [values]
                stacked = np.vstack(sequence)
                valid_count = np.sum(~np.isnan(stacked), axis=0)
                summed = np.nansum(stacked, axis=0)
                mean = np.divide(
                    summed,
                    valid_count,
                    out=np.zeros_like(summed, dtype=np.float32),
                    where=valid_count > 0,
                )
                rolling_blocks.append(mean.astype(np.float32, copy=False))

            delta = np.nan_to_num(
                values - lag_blocks[0], nan=0.0, posinf=0.0, neginf=0.0
            )
            rows.append(np.concatenate([*lag_blocks, *rolling_blocks, delta]))

        for asset, values in zip(assets, current):
            self.history[int(asset)].append(values.copy())
        self.last_time_id = time_id
        return np.vstack(rows).astype(np.float32, copy=False)

    def restore(
        self,
        asset_ids: np.ndarray,
        values: np.ndarray,
        counts: np.ndarray,
        last_time_id: int | None,
    ) -> None:
        """Restore the causal tail saved by production training."""
        asset_ids = np.asarray(asset_ids, dtype=np.int64).reshape(-1)
        values = np.asarray(values, dtype=np.float32)
        counts = np.asarray(counts, dtype=np.int64).reshape(-1)
        expected = (asset_ids.size, self.max_history, len(self.feature_columns))
        if values.shape != expected or counts.shape != asset_ids.shape:
            raise ValueError(
                f"rolling history shapes must be assets={asset_ids.shape}, "
                f"values={expected}, counts={asset_ids.shape}"
            )
        if np.any((counts < 0) | (counts > self.max_history)):
            raise ValueError("rolling history counts are outside the valid range")
        self.history.clear()
        for asset, block, count in zip(asset_ids, values, counts):
            for row in block[-int(count) :] if count else ():
                self.history[int(asset)].append(row.copy())
        self.last_time_id = None if last_time_id is None else int(last_time_id)
