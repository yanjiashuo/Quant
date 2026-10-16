from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from repro.baseline01.config import BASELINE
from repro.config import ensure_data2_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_fold_archive(
    parent_archive: Path,
    rolling_dir: Path,
    output: Path,
) -> dict[str, Any]:
    parent_archive = Path(parent_archive)
    rolling_dir = Path(rolling_dir)
    output = Path(output)
    rolling = np.asarray(
        np.load(rolling_dir / "rolling_four.npy", mmap_mode="r"), dtype=np.float64
    ).reshape(-1)
    rolling_row_id = np.asarray(
        np.load(rolling_dir / "row_id.npy", mmap_mode="r"), dtype=np.int64
    ).reshape(-1)
    if rolling.size != rolling_row_id.size:
        raise ValueError("Rolling4 prediction and row_id lengths differ")
    if not np.isfinite(rolling).all():
        raise ValueError("Rolling4 prediction contains non-finite values")

    with np.load(parent_archive, allow_pickle=False) as parent:
        if "row_id" not in parent.files or "rolling_v1" not in parent.files:
            raise ValueError("parent member archive lacks row_id or rolling_v1")
        parent_row_id = np.asarray(parent["row_id"], dtype=np.int64).reshape(-1)
        if np.unique(parent_row_id).size != parent_row_id.size:
            raise ValueError("parent member archive contains duplicate row_id")
        if not np.array_equal(parent_row_id, rolling_row_id):
            raise ValueError("Rolling4 row_id does not exactly match parent archive")
        payload = {name: np.asarray(parent[name]) for name in parent.files}
    payload["rolling_v1"] = rolling
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **payload)

    with np.load(output, allow_pickle=False) as candidate:
        if not np.array_equal(candidate["row_id"], parent_row_id):
            raise AssertionError("written candidate row_id changed")
        for name in payload:
            if name == "rolling_v1":
                continue
            if not np.array_equal(candidate[name], payload[name]):
                raise AssertionError(f"written candidate changed parent array {name}")
        if not np.array_equal(candidate["rolling_v1"], rolling):
            raise AssertionError("written candidate Rolling4 prediction changed")
    return {
        "parent_archive": str(parent_archive.resolve()),
        "rolling_dir": str(rolling_dir.resolve()),
        "output": str(output.resolve()),
        "rows": int(parent_row_id.size),
        "changed_member": "rolling_v1",
        "unchanged_members": [name for name in BASELINE.member_names if name != "rolling_v1"],
        "sha256": _sha256(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-members", type=Path, required=True)
    parser.add_argument("--rolling-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[5, 6, 7, 8])
    args = parser.parse_args()
    output_root = ensure_data2_path(args.out)
    reports = [
        build_fold_archive(
            args.parent_members / f"fold_{fold}.npz",
            args.rolling_root / f"fold_{fold}",
            output_root / f"fold_{fold}.npz",
        )
        for fold in args.folds
    ]
    report_path = output_root / "build_report.json"
    report_path.write_text(
        json.dumps({"folds": reports}, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"folds": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
