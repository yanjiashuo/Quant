from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from repro.baseline02.finalize import reconstruct_submission, run_official_replays
from repro.baseline04.config import CHILD_MEMBERS, EXPECTED_ROWS, RIDGE_ALPHA, TARGET_STD
from repro.baseline04.contract import sha256_file, write_json


METHOD = "e1_cs_rank_model_c_long"


def _load_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["row_id", "target"]:
        raise ValueError("submission columns differ")
    if frame["row_id"].duplicated().any():
        raise ValueError("submission row_id contains duplicates")
    if not np.isfinite(frame["target"].to_numpy(dtype=np.float64)).all():
        raise ValueError("submission target is non-finite")
    return frame


def validate_bundle(
    root: Path,
    sample_path: Path,
    expected_rows: int = EXPECTED_ROWS,
    tolerance: float = 1e-12,
) -> dict[str, object]:
    directory = Path(root)
    recipe = json.loads((directory / "recipe.json").read_text(encoding="utf-8"))
    if tuple(recipe.get("member_names", ())) != CHILD_MEMBERS:
        raise ValueError("member order differs")
    if float(recipe.get("ridge_alpha", -1.0)) != RIDGE_ALPHA:
        raise ValueError("Ridge alpha differs")
    calibration = recipe.get("calibration", {})
    if calibration.get("name") != "mean_preserving_std" or float(
        calibration.get("target_std_pop", -1.0)
    ) != TARGET_STD:
        raise ValueError("calibration contract differs")
    if recipe.get("public_labels_used") is not False:
        raise ValueError("public label audit flag differs")

    submission = _load_submission(directory / "submission.csv")
    sample = pd.read_csv(sample_path)
    if list(sample.columns) != ["row_id", "target"]:
        raise ValueError("sample submission columns differ")
    if len(submission) != expected_rows:
        raise ValueError(f"submission rows differ: {len(submission)} != {expected_rows}")
    if not np.array_equal(
        submission["row_id"].to_numpy(dtype=np.int64, copy=False),
        sample["row_id"].to_numpy(dtype=np.int64, copy=False),
    ):
        raise ValueError("submission row_id order differs from sample")
    members = pd.read_parquet(directory / "public_members.parquet")
    if tuple(name for name in CHILD_MEMBERS if name in members) != CHILD_MEMBERS:
        raise ValueError("public member columns differ")
    rebuilt = reconstruct_submission(members, recipe)
    target = submission["target"].to_numpy(dtype=np.float64, copy=False)
    maximum = float(np.max(np.abs(target - rebuilt)))
    if maximum > tolerance:
        raise ValueError(f"recipe reconstruction differs by {maximum}")
    standard_deviation = float(target.std(ddof=0))
    if abs(standard_deviation - TARGET_STD) > tolerance:
        raise ValueError(f"submission std differs: {standard_deviation}")
    return {
        "method": METHOD,
        "rows": int(len(submission)),
        "unique_row_ids": int(submission["row_id"].nunique()),
        "finite_rows": int(np.isfinite(target).sum()),
        "mean": float(target.mean()),
        "std_pop": standard_deviation,
        "min": float(target.min()),
        "max": float(target.max()),
        "recipe_rebuild_max_abs": maximum,
        "submission_sha256": sha256_file(directory / "submission.csv"),
        "offline_verdict": str(
            json.loads((directory / "walk_forward.json").read_text(encoding="utf-8"))[
                "verdict"
            ]
        ),
    }


def write_checksums(root: Path) -> dict[str, object]:
    directory = Path(root)
    names = [
        "submission.csv",
        "recipe.json",
        "report.json",
        "walk_forward.json",
        "public_members.parquet",
        "runner_submission.csv",
        "runner_report.json",
    ]
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"final bundle is incomplete: {missing}")
    payload = {
        name: {
            "bytes": int((directory / name).stat().st_size),
            "sha256": sha256_file(directory / name),
        }
        for name in names
    }
    write_json(directory / "checksums.json", payload)
    return payload


def finalize(
    root: Path,
    sample_path: Path,
    data_root: Path,
    api_dir: Path,
    expected_rows: int = EXPECTED_ROWS,
) -> dict[str, object]:
    experiment_root = Path(root)
    method_root = experiment_root / METHOD
    validation = validate_bundle(method_root, sample_path, expected_rows)
    runner = run_official_replays(
        experiment_root,
        data_root,
        api_dir,
        methods=(METHOD,),
    )[METHOD]
    if runner["status"] != "ok":
        raise ValueError("official public replay failed")
    if int(runner["timing"]["predict_timeout_count"]) != 0:
        raise ValueError("official public replay timed out")
    if float(runner["submission_max_abs_diff"]) > 1e-12:
        raise ValueError("official public replay differs from submission")
    checksums = write_checksums(method_root)
    manifest = {
        "method": METHOD,
        "scope": "public_csv_and_order_format_replay_only",
        "private_deployable": False,
        "validation": validation,
        "runner": runner,
        "checksums": checksums,
        "parent": {
            "method": "plan02_e1_cs_rank",
            "public_score": 0.0034444,
        },
        "public_score": None,
        "public_labels_used_for_fit_or_gate": False,
    }
    write_json(experiment_root / "summary" / "submission_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--api-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    args = parser.parse_args()
    report = finalize(
        args.root,
        args.sample_submission,
        args.data_root,
        args.api_dir,
        args.expected_rows,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
