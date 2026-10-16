from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class FoldSpec:
    valid_partition: int
    train_partitions: tuple[int, ...]
    gap_time_ids: int
    held_out: bool


def make_fold(valid_partition: int, gap_time_ids: int = 0) -> dict[str, object]:
    if valid_partition not in {5, 6, 7, 8}:
        raise ValueError("valid_partition must be one of 5, 6, 7, 8")
    if gap_time_ids < 0:
        raise ValueError("gap_time_ids must be nonnegative")
    spec = FoldSpec(
        valid_partition=valid_partition,
        train_partitions=tuple(range(valid_partition)),
        gap_time_ids=gap_time_ids,
        held_out=valid_partition == 8,
    )
    result = asdict(spec)
    result["train_partitions"] = list(spec.train_partitions)
    return result


def purged_training_mask(
    train_time_id: np.ndarray, valid_time_id: np.ndarray, gap_time_ids: int
) -> np.ndarray:
    train_time_id = np.asarray(train_time_id).reshape(-1)
    valid_time_id = np.asarray(valid_time_id).reshape(-1)
    if valid_time_id.size == 0:
        raise ValueError("validation time_id cannot be empty")
    if gap_time_ids < 0:
        raise ValueError("gap_time_ids must be nonnegative")
    cutoff = int(valid_time_id.min()) - int(gap_time_ids)
    return train_time_id < cutoff


def validate_split(
    train_time_id: np.ndarray, valid_time_id: np.ndarray, gap_time_ids: int
) -> dict[str, int]:
    train_time_id = np.asarray(train_time_id).reshape(-1)
    valid_time_id = np.asarray(valid_time_id).reshape(-1)
    if train_time_id.size == 0 or valid_time_id.size == 0:
        raise ValueError("train and validation time_id must be nonempty")
    train_unique = np.unique(train_time_id)
    valid_unique = np.unique(valid_time_id)
    if np.intersect1d(train_unique, valid_unique).size:
        raise ValueError("the same time_id appears in train and validation")
    train_max = int(train_unique.max())
    valid_min = int(valid_unique.min())
    if not train_max < valid_min - gap_time_ids:
        raise ValueError("training tail violates the requested purge gap")
    return {
        "train_min_time_id": int(train_unique.min()),
        "train_max_time_id": train_max,
        "valid_min_time_id": valid_min,
        "valid_max_time_id": int(valid_unique.max()),
        "gap_time_ids": int(gap_time_ids),
    }
