from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from repro.baseline03.config import CANDIDATE_MEMBERS, PARENT_MEMBERS


@dataclass(frozen=True)
class OOFFold:
    matrix: np.ndarray
    target: np.ndarray
    weight: np.ndarray
    row_id: np.ndarray
    time_id: np.ndarray
    asset_id: np.ndarray
    member_names: tuple[str, ...]
    source: str

    @property
    def rows(self) -> int:
        return int(self.target.size)


def _vector(data: np.lib.npyio.NpzFile, name: str, dtype: np.dtype) -> np.ndarray:
    return np.asarray(data[name], dtype=dtype).reshape(-1)


def load_oof(
    path: Path, member_names: Sequence[str] = PARENT_MEMBERS
) -> OOFFold:
    source = Path(path)
    names = tuple(member_names)
    required = (*names, "y", "w", "row_id", "time_id", "asset_id")
    with np.load(source, allow_pickle=False) as data:
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(f"{source} is missing columns: {missing}")
        matrix = np.column_stack([_vector(data, name, np.float64) for name in names])
        target = _vector(data, "y", np.float64)
        weight = _vector(data, "w", np.float64)
        row_id = _vector(data, "row_id", np.int64)
        time_id = _vector(data, "time_id", np.int64)
        asset_id = _vector(data, "asset_id", np.int64)

    rows = target.size
    if rows == 0 or matrix.shape != (rows, len(names)):
        raise ValueError(f"{source} has an invalid member matrix")
    if any(value.size != rows for value in (weight, row_id, time_id, asset_id)):
        raise ValueError(f"{source} vectors have inconsistent row counts")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError(f"{source} predictions and target must be finite")
    if not np.isfinite(weight).all() or np.any(weight < 0.0) or float(weight.sum()) <= 0.0:
        raise ValueError(f"{source} weight must be finite, nonnegative, and positive in total")
    if np.unique(row_id).size != rows:
        raise ValueError(f"{source} row_id must be unique")
    if np.any(time_id[1:] < time_id[:-1]):
        raise ValueError(f"{source} time_id must be nondecreasing")
    return OOFFold(
        matrix=matrix,
        target=target,
        weight=weight,
        row_id=row_id,
        time_id=time_id,
        asset_id=asset_id,
        member_names=names,
        source=str(source),
    )


def append_candidate(parent: OOFFold, candidate_path: Path) -> OOFFold:
    """Append model_c_long after exact row/time/asset validation."""
    source = Path(candidate_path)
    required = ("model_c_long", "row_id", "time_id", "asset_id")
    with np.load(source, allow_pickle=False) as data:
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(f"{source} is missing columns: {missing}")
        prediction = _vector(data, "model_c_long", np.float64)
        row_id = _vector(data, "row_id", np.int64)
        time_id = _vector(data, "time_id", np.int64)
        asset_id = _vector(data, "asset_id", np.int64)

    if prediction.size != parent.rows:
        raise ValueError("candidate and parent row counts differ")
    for name, candidate, expected in (
        ("row_id", row_id, parent.row_id),
        ("time_id", time_id, parent.time_id),
        ("asset_id", asset_id, parent.asset_id),
    ):
        if not np.array_equal(candidate, expected):
            raise ValueError(f"candidate {name} does not exactly align with parent")
    if not np.isfinite(prediction).all():
        raise ValueError("candidate prediction must be finite")
    return OOFFold(
        matrix=np.column_stack((parent.matrix, prediction)),
        target=parent.target,
        weight=parent.weight,
        row_id=parent.row_id,
        time_id=parent.time_id,
        asset_id=parent.asset_id,
        member_names=CANDIDATE_MEMBERS,
        source=f"{parent.source}+{source}",
    )
