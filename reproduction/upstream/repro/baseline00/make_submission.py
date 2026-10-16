from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from repro.baseline00.metrics import level_deviation, prediction_summary
from repro.config import TEST_DIR, WORK, ensure_data2_path


EXPECTED_PUBLIC_TEST_ROWS = 3_217_458


def test_index(test_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    times: list[np.ndarray] = []
    for path in sorted(Path(test_dir).glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["row_id", "time_id"])
        rows.append(frame["row_id"].to_numpy(dtype=np.int64, copy=True))
        times.append(frame["time_id"].to_numpy(dtype=np.int64, copy=True))
    if not rows:
        raise ValueError(f"no test parquet files found beneath {test_dir}")
    return np.concatenate(rows), np.concatenate(times)


def validate_submission(
    expected_row_id: np.ndarray, submission_path: Path
) -> tuple[dict[str, Any], np.ndarray]:
    frame = pd.read_csv(submission_path)
    if list(frame.columns) != ["row_id", "target"]:
        raise ValueError(f"submission columns must be row_id,target; got {list(frame.columns)}")
    row_id = frame["row_id"].to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(row_id, np.asarray(expected_row_id, dtype=np.int64)):
        raise ValueError("submission row_id does not exactly match test order")
    if frame["row_id"].duplicated().any():
        raise ValueError("submission row_id contains duplicates")
    prediction = frame["target"].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(prediction).all():
        raise ValueError("submission contains non-finite target values")
    return {
        "path": str(submission_path),
        "rows": int(len(frame)),
        "expected_rows": int(len(expected_row_id)),
        "actual_rows": int(len(frame)),
        "columns": list(frame.columns),
        "unique_row_id": True,
        "prediction": prediction_summary(prediction),
    }, prediction


def aligned_prediction(path: Path, expected_row_id: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path)
    if "row_id" not in frame.columns:
        raise ValueError(f"reference {path} lacks row_id")
    value_columns = [column for column in frame.columns if column != "row_id"]
    if "target" in value_columns:
        value_column = "target"
    elif len(value_columns) == 1:
        value_column = value_columns[0]
    else:
        raise ValueError(
            f"reference {path} must contain target or exactly one prediction column; "
            f"got {value_columns}"
        )
    if frame["row_id"].duplicated().any():
        raise ValueError(f"reference {path} contains duplicate row_id")
    aligned = pd.DataFrame({"row_id": expected_row_id}).merge(
        frame.loc[:, ["row_id", value_column]], on="row_id", how="left", validate="one_to_one"
    )
    prediction = aligned[value_column].to_numpy(dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError(f"reference {path} does not cover all test row_id with finite values")
    return prediction


def comparison(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return {
        "pearson": float(np.corrcoef(left, right)[0, 1]),
        "spearman": float(pd.Series(left).corr(pd.Series(right), method="spearman")),
        "mse": float(np.mean((left - right) ** 2)),
        "mean_abs_difference": float(np.mean(np.abs(left - right))),
    }


def load_strategy(api_dir: Path, strategy_dir: Path):
    sys.path.insert(0, str(api_dir))
    try:
        from runner import load_model

        return load_model(strategy_dir)
    finally:
        sys.path.pop(0)


def replay_members(api_dir: Path, data_root: Path, strategy_dir: Path, output: Path) -> dict:
    sys.path.insert(0, str(api_dir))
    try:
        from runner import iter_test_slices, load_model

        model = load_model(strategy_dir)
        row_parts: list[np.ndarray] = []
        component_parts: dict[str, list[np.ndarray]] = {"model_c": [], "rolling_v1": []}
        final_parts: list[np.ndarray] = []
        for _, test in iter_test_slices(data_root, split="test"):
            final_parts.append(np.asarray(model.predict(test.copy()), dtype=np.float64))
            row_parts.append(test["row_id"].to_numpy(dtype=np.int64, copy=True))
            for name in component_parts:
                component_parts[name].append(np.asarray(model.last_members[name], dtype=np.float64).copy())
    finally:
        sys.path.pop(0)
    payload = {
        "row_id": np.concatenate(row_parts),
        "target": np.concatenate(final_parts),
        **{name: np.concatenate(parts) for name, parts in component_parts.items()},
    }
    np.savez(output, **payload)
    return {
        "path": str(output),
        "rows": int(len(payload["row_id"])),
        "components": list(component_parts),
        "all_finite": bool(all(np.isfinite(value).all() for value in payload.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--api-dir", type=Path)
    parser.add_argument("--strategy-dir", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--public-model-c", type=Path)
    parser.add_argument("--public-rolling", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--out", type=Path, default=WORK / "comparison_report.json")
    args = parser.parse_args()

    output = ensure_data2_path(args.out)
    expected_row_id, time_id = test_index(args.test_dir)
    if len(expected_row_id) != EXPECTED_PUBLIC_TEST_ROWS:
        raise ValueError(
            f"test data rows {len(expected_row_id)} differ from expected "
            f"{EXPECTED_PUBLIC_TEST_ROWS}"
        )
    validation, production = validate_submission(expected_row_id, args.submission)
    report: dict[str, Any] = {"submission": validation}
    if args.replay:
        replay_path = ensure_data2_path(args.replay)
        if not replay_path.exists():
            if not (args.api_dir and args.data_root and args.strategy_dir):
                raise ValueError("creating --replay requires --api-dir, --data-root and --strategy-dir")
            report["member_replay"] = replay_members(
                args.api_dir, args.data_root, args.strategy_dir, replay_path
            )
        with np.load(replay_path) as replay:
            if not np.array_equal(replay["row_id"], expected_row_id):
                raise ValueError("member replay row_id differs from test order")
            report["official_vs_replay"] = comparison(production, replay["target"])
            if args.public_model_c:
                public = aligned_prediction(args.public_model_c, expected_row_id)
                report["model_c_vs_public"] = comparison(replay["model_c"], public)
            if args.public_rolling:
                public = aligned_prediction(args.public_rolling, expected_row_id)
                report["rolling_v1_vs_public"] = comparison(replay["rolling_v1"], public)
    if args.reference:
        reference = aligned_prediction(args.reference, expected_row_id)
        report["production_vs_current_best"] = comparison(production, reference)
        production_level, production_dev = level_deviation(production, time_id)
        reference_level, reference_dev = level_deviation(reference, time_id)
        report["level_deviation_vs_current_best"] = {
            "level_pearson": float(np.corrcoef(production_level, reference_level)[0, 1]),
            "deviation_pearson": float(np.corrcoef(production_dev, reference_dev)[0, 1]),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
