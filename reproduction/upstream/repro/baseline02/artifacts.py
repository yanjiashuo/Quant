from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from repro.baseline02.config import OOF_MD5, PARENT_MEMBERS, oof_paths


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


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _vector(data: np.lib.npyio.NpzFile, key: str, dtype: np.dtype) -> np.ndarray:
    value = np.asarray(data[key], dtype=dtype).reshape(-1)
    return value


def load_oof(
    path: Path,
    members: Sequence[str] = PARENT_MEMBERS,
) -> OOFFold:
    source = Path(path)
    names = tuple(members)
    required = ("row_id", "time_id", "asset_id", "weight", "target", *names)
    with np.load(source, allow_pickle=False) as data:
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(f"{source} is missing columns: {missing}")
        matrix = np.column_stack(
            [_vector(data, name, np.float64) for name in names]
        )
        target = _vector(data, "target", np.float64)
        weight = _vector(data, "weight", np.float64)
        row_id = _vector(data, "row_id", np.int64)
        time_id = _vector(data, "time_id", np.int64)
        asset_id = _vector(data, "asset_id", np.int64)

    rows = target.size
    if rows == 0:
        raise ValueError(f"{source} is empty")
    if matrix.shape != (rows, len(names)):
        raise ValueError(f"{source} member matrix shape is invalid: {matrix.shape}")
    for label, vector in (
        ("weight", weight),
        ("row_id", row_id),
        ("time_id", time_id),
        ("asset_id", asset_id),
    ):
        if vector.size != rows:
            raise ValueError(f"{source} {label} row count differs from target")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError(f"{source} contains non-finite predictions or target")
    if not np.isfinite(weight).all() or np.any(weight < 0.0):
        raise ValueError(f"{source} contains invalid weight")
    if float(weight.sum()) <= 0.0:
        raise ValueError(f"{source} has no positive weight")
    if np.unique(row_id).size != rows:
        raise ValueError("row_id must be unique")
    if np.any(time_id[1:] < time_id[:-1]):
        raise ValueError("time_id must be nondecreasing")
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


def audit_oof_dir(root: Path, verify_hashes: bool = True) -> dict[str, object]:
    report: dict[str, object] = {
        "member_names": list(PARENT_MEMBERS),
        "folds": {},
    }
    fold_report: dict[str, object] = {}
    for fold, path in oof_paths(root).items():
        digest = file_md5(path)
        if verify_hashes and digest != OOF_MD5[fold]:
            raise ValueError(
                f"Fold{fold} MD5 differs: expected {OOF_MD5[fold]}, got {digest}"
            )
        data = load_oof(path)
        fold_report[str(fold)] = {
            "path": str(path),
            "md5": digest,
            "rows": data.rows,
            "time_min": int(data.time_id.min()),
            "time_max": int(data.time_id.max()),
            "assets": np.unique(data.asset_id).astype(int).tolist(),
            "weight_sum": float(data.weight.sum()),
        }
    report["folds"] = fold_report
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-hash-check", action="store_true")
    args = parser.parse_args()
    report = audit_oof_dir(args.oof_dir, verify_hashes=not args.skip_hash_check)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
