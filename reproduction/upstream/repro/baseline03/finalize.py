from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from repro.baseline00.make_submission import comparison, replay_members, validate_submission


EXPECTED_PUBLIC_TEST_ROWS = 3_217_458


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_row_id(path: Path) -> np.ndarray:
    sample = pd.read_csv(path)
    if list(sample.columns) != ["row_id", "target"]:
        raise ValueError("sample submission columns must be row_id,target")
    if sample["row_id"].duplicated().any():
        raise ValueError("sample submission row_id contains duplicates")
    return sample["row_id"].to_numpy(dtype=np.int64, copy=True)


def run_official_and_replay(
    root: Path,
    data_root: Path,
    api_dir: Path,
    strategy_dir: Path,
) -> dict[str, Any]:
    root = Path(root)
    submission_path = root / "submission.csv"
    replay_path = root / "independent_replay.npz"
    sys.path.insert(0, str(api_dir))
    try:
        from runner import run_strategy

        result = run_strategy(
            data_root=Path(data_root),
            strategy_dir=Path(strategy_dir),
            output_path=submission_path,
            split="test",
        )
    finally:
        sys.path.pop(0)

    replay_report = replay_members(
        Path(api_dir), Path(data_root), Path(strategy_dir), replay_path
    )
    official = pd.read_csv(submission_path)
    with np.load(replay_path) as replay:
        replay_row_id = replay["row_id"]
        replay_prediction = replay["target"]
    official_row_id = official["row_id"].to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(official_row_id, replay_row_id):
        raise ValueError("official runner and independent replay row_id differ")
    official_prediction = official["target"].to_numpy(dtype=np.float64, copy=False)
    difference = np.abs(official_prediction - replay_prediction)
    replay_comparison = comparison(official_prediction, replay_prediction)
    replay_comparison["max_absolute_difference"] = float(difference.max(initial=0.0))

    report = result.as_dict()
    report["audit_scope"] = "fresh_feature_to_model_runner_and_independent_replay"
    report["independent_replay"] = replay_report
    report["official_vs_independent_replay"] = replay_comparison
    if report.get("status") != "ok":
        raise ValueError(f"official runner failed: {report}")
    if replay_comparison["pearson"] < 0.999999999:
        raise ValueError("official runner and independent replay correlation is too low")
    if replay_comparison["max_absolute_difference"] >= 1e-10:
        raise ValueError("official runner and independent replay values differ")
    _write_json(root / "runner_report.json", report)
    return report


def validate_g0_artifacts(
    root: Path,
    sample_submission_path: Path,
    expected_rows: int = EXPECTED_PUBLIC_TEST_ROWS,
) -> dict[str, Any]:
    root = Path(root)
    required = (
        root / "submission.csv",
        root / "recipe.json",
        root / "report.json",
        root / "runner_report.json",
        root / "production" / "production_report.json",
        root / "production" / "api_model" / "production_config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"G0 artifact bundle is incomplete: {missing}")

    expected_row_id = _sample_row_id(sample_submission_path)
    if len(expected_row_id) != expected_rows:
        raise ValueError(
            f"sample submission has {len(expected_row_id)} rows; expected {expected_rows}"
        )
    submission, _ = validate_submission(expected_row_id, root / "submission.csv")
    recipe = json.loads((root / "recipe.json").read_text(encoding="utf-8"))
    if recipe.get("variant_name") != "g0_gap0_parent":
        raise ValueError("G0 recipe variant differs")
    if recipe.get("validation_gap_time_ids") != 0 or recipe.get(
        "production_gap_time_ids"
    ) != 0:
        raise ValueError("G0 recipe is not gap zero")
    runner = json.loads((root / "runner_report.json").read_text(encoding="utf-8"))
    if runner.get("status") != "ok" or int(runner.get("rows", -1)) != expected_rows:
        raise ValueError("G0 official runner report failed")
    replay = runner.get("official_vs_independent_replay", {})
    if float(replay.get("pearson", -1.0)) < 0.999999999:
        raise ValueError("G0 independent replay correlation is too low")
    if float(replay.get("max_absolute_difference", np.inf)) >= 1e-10:
        raise ValueError("G0 independent replay differs from official output")
    return {
        "variant": "g0_gap0_parent",
        "submission": submission,
        "runner_status": runner["status"],
        "runner_rows": int(runner["rows"]),
        "official_vs_independent_replay": replay,
    }


def validate_e1_artifacts(
    root: Path,
    sample_submission_path: Path,
    expected_rows: int = EXPECTED_PUBLIC_TEST_ROWS,
) -> dict[str, Any]:
    root = Path(root)
    required = (
        root / "submission.csv",
        root / "recipe.json",
        root / "report.json",
        root / "walk_forward.json",
        root / "runner_report.json",
        root / "public_model_c_long.parquet",
        root / "production" / "production_report.json",
        root / "production" / "api_model" / "model_c_long.txt",
        root / "production" / "api_model" / "production_config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"E1 artifact bundle is incomplete: {missing}")
    expected_row_id = _sample_row_id(sample_submission_path)
    if len(expected_row_id) != expected_rows:
        raise ValueError(
            f"sample submission has {len(expected_row_id)} rows; expected {expected_rows}"
        )
    submission, _ = validate_submission(expected_row_id, root / "submission.csv")
    recipe = json.loads((root / "recipe.json").read_text(encoding="utf-8"))
    if recipe.get("variant_name") != "e1_model_c_long":
        raise ValueError("E1 recipe variant differs")
    if recipe.get("validation_gap_time_ids") != 0 or recipe.get(
        "production_gap_time_ids"
    ) != 0:
        raise ValueError("E1 recipe is not gap zero")
    if recipe.get("member_names") != [
        "final", "L88", "model_c", "level", "rolling_v1", "model_c_long"
    ]:
        raise ValueError("E1 recipe member order differs")
    runner = json.loads((root / "runner_report.json").read_text(encoding="utf-8"))
    if runner.get("status") != "ok" or int(runner.get("rows", -1)) != expected_rows:
        raise ValueError("E1 official runner report failed")
    replay = runner.get("official_vs_independent_replay", {})
    if float(replay.get("pearson", -1.0)) < 0.999999999:
        raise ValueError("E1 independent replay correlation is too low")
    if float(replay.get("max_absolute_difference", np.inf)) >= 1e-10:
        raise ValueError("E1 independent replay differs from official output")
    return {
        "variant": "e1_model_c_long",
        "submission": submission,
        "verdict": recipe.get("gate", {}).get("verdict"),
        "recommended_submission": bool(recipe.get("recommended_submission", False)),
        "runner_status": runner["status"],
        "runner_rows": int(runner["rows"]),
        "official_vs_independent_replay": replay,
    }


def write_checksums(root: Path) -> dict[str, dict[str, Any]]:
    root = Path(root)
    excluded_parts = {"cache", "__pycache__"}
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "checksums.json"
        and not path.name.endswith(".tmp")
        and not excluded_parts.intersection(path.relative_to(root).parts)
    )
    checksums = {
        path.relative_to(root).as_posix(): {
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in paths
    }
    _write_json(root / "checksums.json", checksums)
    return checksums


def finalize_g0(
    root: Path,
    data_root: Path,
    api_dir: Path,
    sample_submission_path: Path,
) -> dict[str, Any]:
    root = Path(root)
    runner = run_official_and_replay(
        root,
        data_root,
        api_dir,
        root / "production" / "api_model",
    )
    validation = validate_g0_artifacts(root, sample_submission_path)
    checksums = write_checksums(root)
    return {
        "validation": validation,
        "runner": runner,
        "checksums_file": str(root / "checksums.json"),
        "checksummed_files": len(checksums),
    }


def finalize_e1(
    root: Path,
    data_root: Path,
    api_dir: Path,
    sample_submission_path: Path,
) -> dict[str, Any]:
    root = Path(root)
    runner = run_official_and_replay(
        root,
        data_root,
        api_dir,
        root / "production" / "api_model",
    )
    validation = validate_e1_artifacts(root, sample_submission_path)
    checksums = write_checksums(root)
    (root / "FINALIZE_SUCCESS").write_text("ok\n", encoding="ascii")
    return {
        "validation": validation,
        "runner": runner,
        "checksums_file": str(root / "checksums.json"),
        "checksummed_files": len(checksums),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--api-dir", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--variant", choices=("g0", "e1"), default="g0")
    args = parser.parse_args()
    finalize = finalize_e1 if args.variant == "e1" else finalize_g0
    print(
        json.dumps(
            finalize(
                args.root,
                args.data_root,
                args.api_dir,
                args.sample_submission,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
