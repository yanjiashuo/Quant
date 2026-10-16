from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from repro.baseline02.artifacts import OOFFold, load_oof
from repro.baseline02.stack import augment_fold, evaluate_walk_forward
from repro.baseline04.config import BASE_OOF_FILENAMES, FOLDS, PARENT_MEMBERS, RIDGE_ALPHA
from repro.baseline04.contract import sha256_file, write_json


def _vector(data: np.lib.npyio.NpzFile, name: str, dtype: np.dtype) -> np.ndarray:
    if name not in data.files:
        raise ValueError(f"candidate input is missing {name}")
    return np.asarray(data[name], dtype=dtype).reshape(-1)


def _require_equal(name: str, expected: np.ndarray, actual: np.ndarray) -> None:
    if not np.array_equal(expected, actual):
        raise ValueError(f"{name} differs")


def adapt_model_c_long_fold(
    parent_path: Path,
    cs_rank_path: Path,
    long_path: Path,
    output_path: Path,
) -> dict[str, object]:
    parent = load_oof(parent_path)
    with np.load(cs_rank_path, allow_pickle=False) as cs:
        cs_row = _vector(cs, "row_id", np.int64)
        source_row = _vector(cs, "source_row_id", np.int64)
        cs_time = _vector(cs, "time_id", np.int64)
        cs_asset = _vector(cs, "asset_id", np.int64)
    with np.load(long_path, allow_pickle=False) as long:
        official_row = _vector(long, "row_id", np.int64)
        long_time = _vector(long, "time_id", np.int64)
        long_asset = _vector(long, "asset_id", np.int64)
        prediction = _vector(long, "model_c_long", np.float64)

    _require_equal("CS row_id", parent.row_id, cs_row)
    _require_equal("source row_id", source_row, official_row)
    _require_equal("CS time_id", parent.time_id, cs_time)
    _require_equal("model_c_long time_id", parent.time_id, long_time)
    _require_equal("CS asset_id", parent.asset_id, cs_asset)
    _require_equal("model_c_long asset_id", parent.asset_id, long_asset)
    if np.unique(source_row).size != source_row.size:
        raise ValueError("source row_id contains duplicates")
    if prediction.size != parent.rows or not np.isfinite(prediction).all():
        raise ValueError("model_c_long prediction is misaligned or non-finite")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        row_id=parent.row_id,
        source_row_id=source_row,
        time_id=parent.time_id,
        asset_id=parent.asset_id,
        model_c_long=prediction,
    )
    return {
        "path": str(destination),
        "rows": parent.rows,
        "parent_row_id_min": int(parent.row_id.min()),
        "parent_row_id_max": int(parent.row_id.max()),
        "source_row_id_min": int(source_row.min()),
        "source_row_id_max": int(source_row.max()),
        "sha256": sha256_file(destination),
    }


def build_parent_folds(base_oof_dir: Path, cs_rank_dir: Path) -> dict[int, OOFFold]:
    result: dict[int, OOFFold] = {}
    for fold in FOLDS:
        base = load_oof(Path(base_oof_dir) / BASE_OOF_FILENAMES[fold])
        parent = augment_fold(
            base, Path(cs_rank_dir) / f"fold_{fold}.npz", "cs_rank"
        )
        if parent.member_names != PARENT_MEMBERS:
            raise ValueError(f"Fold{fold} parent member order differs")
        result[fold] = parent
    return result


def _maximum_difference(left: object, right: object) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"reconstruction shapes differ: {a.shape} != {b.shape}")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def verify_parent_report(
    parent_folds: Mapping[int, OOFFold],
    frozen_report: Path,
    tolerance: float = 1e-12,
) -> dict[str, object]:
    rebuilt = evaluate_walk_forward(parent_folds, alpha=RIDGE_ALPHA)
    expected = json.loads(Path(frozen_report).read_text(encoding="utf-8"))
    if tuple(expected["member_names"]) != PARENT_MEMBERS:
        raise ValueError("frozen parent member order differs")
    comparisons = {
        "fold7_score": abs(
            float(rebuilt["fold7"]["raw_score"])
            - float(expected["fold7"]["child_score"])
        ),
        "fold8_score": abs(
            float(rebuilt["fold8"]["raw_score"])
            - float(expected["fold8"]["child_score"])
        ),
        "fold7_coefficients": _maximum_difference(
            rebuilt["fold7"]["fit"]["effective_coefficients"],
            expected["fold7"]["child_fit"]["effective_coefficients"],
        ),
        "fold8_coefficients": _maximum_difference(
            rebuilt["fold8"]["fit"]["effective_coefficients"],
            expected["fold8"]["child_fit"]["effective_coefficients"],
        ),
        "production_coefficients": _maximum_difference(
            rebuilt["production"]["fit"]["effective_coefficients"],
            expected["production"]["fit"]["effective_coefficients"],
        ),
    }
    maximum = max(comparisons.values())
    if maximum > tolerance:
        raise ValueError(
            f"Plan02 E1 parent reconstruction differs by {maximum} > {tolerance}"
        )
    return {
        "passed": True,
        "tolerance": tolerance,
        "maximum_absolute_difference": maximum,
        "comparisons": comparisons,
        "rebuilt": rebuilt,
    }


def run_assembly(
    base_oof_dir: Path,
    cs_rank_dir: Path,
    long_oof_dir: Path,
    frozen_parent_report: Path,
    output_dir: Path,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    parent_folds = build_parent_folds(base_oof_dir, cs_rank_dir)
    parent_rebuild = verify_parent_report(parent_folds, frozen_parent_report)
    adapted = {}
    for fold in FOLDS:
        adapted[str(fold)] = adapt_model_c_long_fold(
            Path(base_oof_dir) / BASE_OOF_FILENAMES[fold],
            Path(cs_rank_dir) / f"fold_{fold}.npz",
            Path(long_oof_dir) / f"fold_{fold}.npz",
            output / f"fold_{fold}.npz",
        )
    report = {
        "parent_rebuild": {
            key: value for key, value in parent_rebuild.items() if key != "rebuilt"
        },
        "adapted_folds": adapted,
    }
    write_json(output / "assembly_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-oof-dir", type=Path, required=True)
    parser.add_argument("--cs-rank-dir", type=Path, required=True)
    parser.add_argument("--long-oof-dir", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run_assembly(
        args.base_oof_dir,
        args.cs_rank_dir,
        args.long_oof_dir,
        args.parent_report,
        args.out,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
