from __future__ import annotations

"""Shared CS-rank data pipeline for Baseline05 E1 (LGBM) and E2 (linear).

Every feature/label detail is identical to the frozen Plan02 ``cs_rank`` member;
only the *training window* (E1 full-history vs recent-4) and the *base learner*
(E1 LGBM vs E2 weighted Ridge) change downstream. To guarantee byte-identical
rank features and residual labels, this module reuses the frozen Plan02 helpers
(`build_cs_rank`, `weighted_time_mean_rows`) rather than reimplementing them.
"""

import gc
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from repro.baseline00.training_io import weighted_time_mean_rows
from repro.baseline02.cs_rank import build_cs_rank
from repro.baseline02.train_candidate import (
    _build_cs_rank_training_cache,
    _load_rank_frame,
)
from repro.config import FEATS


def training_partitions(stop: int, mode: str) -> list[int]:
    """Partition indices used to train a CS-rank model for fold ``stop``.

    ``long`` uses every prior partition ``0..stop-1``; ``recent4`` uses the
    frozen recent-four window ``max(0, stop-4)..stop-1``.
    """
    stop = int(stop)
    if mode == "long":
        return list(range(stop))
    if mode == "recent4":
        return list(range(max(0, stop - 4), stop))
    raise ValueError(f"unknown CS mode: {mode}")


def rank_features(frame: pd.DataFrame) -> np.ndarray:
    """Plan02-identical within-time feature ranks mapped to [-1, 1] (float32)."""
    return build_cs_rank(frame, FEATS)


def rank_target(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Residual label ``target - weighted_time_mean(target)`` and clipped weight."""
    weight = np.maximum(frame["weight"].to_numpy(np.float64), 0.0)
    target = frame["target"].to_numpy(np.float64)
    level = weighted_time_mean_rows(
        frame["time_id"].to_numpy(np.int64), target, weight
    )
    return target - level, weight


def load_rank_frame(path: Path, labels: bool) -> pd.DataFrame:
    """Load one partition sorted by (time_id, asset_id) — Plan02 row order."""
    return _load_rank_frame(Path(path), labels)


def build_lgb_cache(
    train_dir: Path, partitions: Sequence[int], cache_dir: Path
) -> tuple[np.memmap, np.memmap, np.memmap, dict[str, int]]:
    """Frozen Plan02 CS-rank LGBM cache (float32 features, float64 target/weight).

    Delegates to the frozen Plan02 builder so E1's full-history cache is produced
    by exactly the same code path as the recent-4 member.
    """
    return _build_cs_rank_training_cache(Path(train_dir), partitions, Path(cache_dir))


def iter_linear_batches(
    train_dir: Path, partitions: Sequence[int]
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Stream one partition at a time as (rank_features, residual_label, weight).

    Used by E2's weighted-Ridge sufficient-statistics accumulation so that only
    one 323-column partition matrix is resident at a time.
    """
    for partition in partitions:
        path = Path(train_dir) / f"train_partition_{int(partition):03d}.parquet"
        frame = load_rank_frame(path, labels=True)
        x = np.asarray(rank_features(frame), dtype=np.float64)
        y, w = rank_target(frame)
        yield x, y, w
        del frame, x, y, w
        gc.collect()
