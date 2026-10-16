from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from repro.baseline00.metrics import level_deviation, prediction_summary


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _load_submission(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["row_id", "target"]:
        raise ValueError(f"{path} columns must be row_id,target")
    if frame["row_id"].duplicated().any():
        raise ValueError(f"{path} row_id contains duplicates")
    prediction = frame["target"].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(prediction).all():
        raise ValueError(f"{path} prediction contains non-finite values")
    return frame["row_id"].to_numpy(dtype=np.int64, copy=True), prediction


def _test_index(test_dir: Path) -> pd.DataFrame:
    paths = sorted(Path(test_dir).glob("test_partition_*.parquet"))
    if not paths:
        raise ValueError("no test partitions found")
    frame = pd.concat(
        [pd.read_parquet(path, columns=["row_id", "time_id", "asset_id"]) for path in paths],
        ignore_index=True,
    )
    if frame["row_id"].duplicated().any():
        raise ValueError("test row_id contains duplicates")
    return frame.set_index("row_id")


def build_summary(
    g0_root: Path,
    e1_root: Path,
    test_dir: Path,
    output: Path,
    source_commit: str,
    g0_public_score: float | None = None,
    e1_public_score: float | None = None,
) -> dict[str, Any]:
    g0_root = Path(g0_root)
    e1_root = Path(e1_root)
    g0_path = g0_root / "submission.csv"
    e1_path = e1_root / "submission.csv"
    g0_row, g0 = _load_submission(g0_path)
    e1_row, e1 = _load_submission(e1_path)
    if not np.array_equal(g0_row, e1_row):
        raise ValueError("G0 and E1 row_id order differs")
    index = _test_index(test_dir).reindex(g0_row)
    if index.isna().any().any():
        raise ValueError("test metadata does not cover submission row_id")
    time_id = index["time_id"].to_numpy(dtype=np.int64, copy=False)
    asset_id = index["asset_id"].to_numpy(dtype=np.int64, copy=False)
    difference = e1 - g0
    g0_level, g0_deviation = level_deviation(g0, time_id)
    e1_level, e1_deviation = level_deviation(e1, time_id)
    per_asset: dict[str, Any] = {}
    for asset in np.unique(asset_id):
        selected = asset_id == asset
        per_asset[str(int(asset))] = {
            "rows": int(selected.sum()),
            "pearson": _correlation(g0[selected], e1[selected]),
            "difference": prediction_summary(difference[selected]),
        }
    walk_forward = json.loads((e1_root / "walk_forward.json").read_text(encoding="utf-8"))
    recipe = json.loads((e1_root / "recipe.json").read_text(encoding="utf-8"))
    runner = json.loads((e1_root / "runner_report.json").read_text(encoding="utf-8"))
    payload = {
        "source_commit": source_commit,
        "public_labels_used_for_training_or_gate": False,
        "g0": {
            "variant": "g0_gap0_parent",
            "path": str(g0_path),
            "sha256": _sha256(g0_path),
            "public_score_recorded_after_freeze": g0_public_score,
            "prediction": prediction_summary(g0),
        },
        "e1": {
            "variant": "e1_model_c_long",
            "path": str(e1_path),
            "sha256": _sha256(e1_path),
            "public_score_recorded_after_freeze": e1_public_score,
            "prediction": prediction_summary(e1),
            "verdict": recipe["gate"]["verdict"],
            "recommended_submission": bool(recipe["recommended_submission"]),
            "member_names": recipe["member_names"],
            "effective_coefficients": recipe["effective_coefficients"],
            "amplitude_multiplier": recipe["amplitude_multiplier"],
            "runner_status": runner["status"],
            "runner_rows": int(runner["rows"]),
            "runner_timing": runner["timing"],
        },
        "offline": {
            "fold7_fixed_delta": walk_forward["fold7"]["delta"]["fixed_amplitude"],
            "fold8_fixed_delta": walk_forward["fold8"]["delta"]["fixed_amplitude"],
            "fold7_positive_blocks": walk_forward["fold7"]["fixed_amplitude_blocks"]["positive_blocks"],
            "fold8_positive_blocks": walk_forward["fold8"]["fixed_amplitude_blocks"]["positive_blocks"],
            "gate": walk_forward["gate"],
        },
        "e1_vs_g0": {
            "pearson": _correlation(g0, e1),
            "difference": prediction_summary(difference),
            "level_pearson": _correlation(g0_level, e1_level),
            "deviation_pearson": _correlation(g0_deviation, e1_deviation),
            "per_asset": per_asset,
        },
    }
    _write_json(Path(output), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g0-root", type=Path, required=True)
    parser.add_argument("--e1-root", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--g0-public-score", type=float)
    parser.add_argument("--e1-public-score", type=float)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build_summary(
        args.g0_root,
        args.e1_root,
        args.test_dir,
        args.out,
        args.source_commit,
        args.g0_public_score,
        args.e1_public_score,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
