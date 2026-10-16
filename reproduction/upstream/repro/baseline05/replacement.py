from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from repro.baseline02.artifacts import OOFFold


def _validated_prediction(prediction: np.ndarray, rows: int, label: str) -> np.ndarray:
    values = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if values.size != rows or not np.isfinite(values).all():
        raise ValueError(f"{label} prediction is invalid")
    return values


def replace_member(
    parent: OOFFold, old_name: str, new_name: str, prediction: np.ndarray
) -> OOFFold:
    """Replace one member column in place, keeping position and column count."""
    values = _validated_prediction(prediction, parent.rows, "replacement")
    if old_name not in parent.member_names:
        raise ValueError(f"member to replace is missing: {old_name}")
    if new_name != old_name and new_name in parent.member_names:
        raise ValueError(f"replacement name already exists: {new_name}")
    position = parent.member_names.index(old_name)
    matrix = parent.matrix.copy()
    matrix[:, position] = values
    names = list(parent.member_names)
    names[position] = new_name
    return dataclasses.replace(parent, matrix=matrix, member_names=tuple(names))


def append_member(
    parent: OOFFold, after: str, new_name: str, prediction: np.ndarray
) -> OOFFold:
    """Insert a new member column immediately after ``after`` (replaces nothing)."""
    values = _validated_prediction(prediction, parent.rows, "appended")
    if after not in parent.member_names:
        raise ValueError(f"anchor member is missing: {after}")
    if new_name in parent.member_names:
        raise ValueError(f"appended name already exists: {new_name}")
    insert_at = parent.member_names.index(after) + 1
    matrix = np.insert(parent.matrix, insert_at, values, axis=1)
    names = list(parent.member_names)
    names.insert(insert_at, new_name)
    return dataclasses.replace(parent, matrix=matrix, member_names=tuple(names))


def _read_aligned_prediction(
    parent: OOFFold, candidate_path: Path, candidate_name: str
) -> np.ndarray:
    """Read ``candidate_name`` from an archive, enforcing exact index alignment.

    The archive must carry ``row_id``, ``time_id``, ``asset_id`` and the named
    prediction, all matching the parent fold row-for-row.
    """
    path = Path(candidate_path)
    with np.load(path, allow_pickle=False) as data:
        for key in ("row_id", "time_id", "asset_id", candidate_name):
            if key not in data.files:
                raise ValueError(f"{path} must contain {key}")
        row_id = np.asarray(data["row_id"], dtype=np.int64).reshape(-1)
        time_id = np.asarray(data["time_id"], dtype=np.int64).reshape(-1)
        asset_id = np.asarray(data["asset_id"], dtype=np.int64).reshape(-1)
        prediction = np.asarray(data[candidate_name], dtype=np.float64).reshape(-1)
    if not np.array_equal(row_id, parent.row_id):
        raise ValueError(f"{path} row_id differs from parent OOF")
    if not np.array_equal(time_id, parent.time_id):
        raise ValueError(f"{path} time_id differs from parent OOF")
    if not np.array_equal(asset_id, parent.asset_id):
        raise ValueError(f"{path} asset_id differs from parent OOF")
    return prediction


def replace_member_from_archive(
    parent: OOFFold, candidate_path: Path, old_name: str, new_name: str
) -> OOFFold:
    prediction = _read_aligned_prediction(parent, candidate_path, new_name)
    return replace_member(parent, old_name, new_name, prediction)


def append_member_from_archive(
    parent: OOFFold, candidate_path: Path, after: str, new_name: str
) -> OOFFold:
    prediction = _read_aligned_prediction(parent, candidate_path, new_name)
    return append_member(parent, after, new_name, prediction)
