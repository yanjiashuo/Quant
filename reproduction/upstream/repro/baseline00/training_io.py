from __future__ import annotations

import gc
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def validate_time_order(time_id: np.ndarray) -> None:
    time_id = np.asarray(time_id).reshape(-1)
    if time_id.size and np.any(time_id[1:] < time_id[:-1]):
        raise ValueError("time_id must be nondecreasing")


def group_boundaries(time_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time_id = np.asarray(time_id).reshape(-1)
    validate_time_order(time_id)
    if time_id.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(time_id[1:] != time_id[:-1]) + 1]
    ends = np.r_[starts[1:], time_id.size]
    return starts.astype(np.int64), ends.astype(np.int64)


def weighted_time_mean_rows(
    time_id: np.ndarray, values: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    time_id = np.asarray(time_id)
    values = np.asarray(values, dtype=np.float64)
    weight = np.maximum(np.asarray(weight, dtype=np.float64), 0.0)
    if not (time_id.shape == values.shape == weight.shape):
        raise ValueError("time_id, values and weight must have equal shapes")
    starts, ends = group_boundaries(time_id)
    sums = np.add.reduceat(weight * values, starts)
    weight_sums = np.add.reduceat(weight, starts)
    means = np.divide(sums, weight_sums, out=np.zeros_like(sums), where=weight_sums > 0)
    return np.repeat(means, ends - starts)


def load_validation_frame(train_dir: Path, fold: int, feature_columns: Sequence[str]) -> pd.DataFrame:
    columns = ["row_id", "time_id", "asset_id", "weight", "target", *feature_columns]
    frame = pd.read_parquet(
        Path(train_dir) / f"train_partition_{fold:03d}.parquet", columns=columns
    )
    return frame.sort_values(["time_id", "asset_id"], kind="stable", ignore_index=True)


def load_prediction_frame(
    train_dir: Path, fold: int, feature_columns: Sequence[str]
) -> pd.DataFrame:
    columns = ["row_id", "time_id", "asset_id", *feature_columns]
    frame = pd.read_parquet(
        Path(train_dir) / f"train_partition_{fold:03d}.parquet", columns=columns
    )
    return frame.sort_values(["time_id", "asset_id"], kind="stable", ignore_index=True)


def load_training_arrays(
    train_dir: Path,
    partitions: Sequence[int],
    feature_columns: Sequence[str],
    cutoff_time_id: int,
    label_mode: str = "target",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    if label_mode not in {"target", "target_demean", "weight"}:
        raise ValueError(f"unsupported label_mode: {label_mode}")
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    rows_by_partition: dict[str, int] = {}
    columns = ["time_id", "weight", "target", *feature_columns]
    for partition in partitions:
        frame = pd.read_parquet(
            Path(train_dir) / f"train_partition_{partition:03d}.parquet", columns=columns
        )
        if partition == max(partitions):
            frame = frame.loc[frame["time_id"] < cutoff_time_id]
        frame = frame.sort_values("time_id", kind="stable", ignore_index=True)
        x = frame.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        raw_weight = np.maximum(
            frame["weight"].to_numpy(dtype=np.float64, copy=False), 0.0
        )
        target = frame["target"].to_numpy(dtype=np.float64, copy=False)
        if label_mode == "target_demean":
            level = weighted_time_mean_rows(
                frame["time_id"].to_numpy(copy=False), target, raw_weight
            )
            label = target - level
        elif label_mode == "weight":
            label = raw_weight
        else:
            label = target
        x_parts.append(x)
        y_parts.append(np.asarray(label, dtype=np.float64).copy())
        w_parts.append(raw_weight.copy())
        rows_by_partition[str(partition)] = int(len(frame))
        del frame, x, raw_weight, target, label
        gc.collect()
    x_all = np.concatenate(x_parts, axis=0)
    y_all = np.concatenate(y_parts)
    w_all = np.concatenate(w_parts)
    del x_parts, y_parts, w_parts
    gc.collect()
    return x_all, y_all, w_all, rows_by_partition


def aggregate_features_by_time(
    time_id: np.ndarray,
    features: np.ndarray,
    weight: np.ndarray,
    target: np.ndarray | None = None,
    group_chunk: int = 5_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    time_id = np.asarray(time_id)
    features = np.asarray(features, dtype=np.float32)
    weight = np.maximum(np.asarray(weight, dtype=np.float32), 0.0)
    if features.ndim != 2 or features.shape[0] != time_id.size or weight.size != time_id.size:
        raise ValueError("aggregate input shapes do not align")
    if target is not None and np.asarray(target).size != time_id.size:
        raise ValueError("target shape does not align")
    starts, ends = group_boundaries(time_id)
    unique_time = time_id[starts]
    feature_mean = np.empty((starts.size, features.shape[1]), dtype=np.float32)
    target_mean = np.empty(starts.size, dtype=np.float64) if target is not None else None
    weight_sum_all = np.empty(starts.size, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64) if target is not None else None
    for group_start in range(0, starts.size, group_chunk):
        group_end = min(group_start + group_chunk, starts.size)
        row_start = int(starts[group_start])
        row_end = int(ends[group_end - 1])
        relative_starts = starts[group_start:group_end] - row_start
        block_weight = weight[row_start:row_end]
        weight_sums = np.add.reduceat(block_weight.astype(np.float64), relative_starts)
        weighted = features[row_start:row_end] * block_weight[:, None]
        feature_sums = np.add.reduceat(weighted, relative_starts, axis=0)
        feature_mean[group_start:group_end] = np.divide(
            feature_sums,
            weight_sums[:, None],
            out=np.zeros_like(feature_sums, dtype=np.float32),
            where=weight_sums[:, None] > 0,
        )
        weight_sum_all[group_start:group_end] = weight_sums
        if target_mean is not None and target_values is not None:
            target_sums = np.add.reduceat(
                target_values[row_start:row_end] * block_weight, relative_starts
            )
            target_mean[group_start:group_end] = np.divide(
                target_sums,
                weight_sums,
                out=np.zeros_like(target_sums),
                where=weight_sums > 0,
            )
        del weighted, feature_sums
    return unique_time, feature_mean, target_mean, weight_sum_all, ends - starts


def broadcast_group_values(group_values: np.ndarray, group_counts: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(group_values), np.asarray(group_counts, dtype=np.int64))
