from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from repro.baseline02.make_submission import calibrate_mean_preserving_std


METHODS = (
    "g0_parent",
    "e1_cs_rank",
    "e2_rolling_v2",
    "e3_combo",
    "e4_level_deviation",
    "e5_ridge",
    "e6_per_asset",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_submission(
    public_members: pd.DataFrame, recipe: dict[str, object]
) -> np.ndarray:
    if "direction_member_names" in recipe:
        names = list(recipe["direction_member_names"])
        coefficients = np.asarray(
            recipe["direction_effective_coefficients"], dtype=np.float64
        )
        raw = public_members.loc[:, names].to_numpy(dtype=np.float64) @ coefficients
        scales = np.asarray(recipe["asset_scales"], dtype=np.float64)
        assets = public_members["asset_id"].to_numpy(dtype=np.int64, copy=False)
        raw = raw * scales[assets]
    else:
        names = list(recipe["member_names"])
        coefficients = np.asarray(recipe["effective_coefficients"], dtype=np.float64)
        raw = public_members.loc[:, names].to_numpy(dtype=np.float64) @ coefficients
    calibration = recipe["calibration"]
    return calibrate_mean_preserving_std(
        raw, float(calibration["target_std_pop"])
    )


def validate_method(
    root: Path,
    method: str,
    sample_row_id: np.ndarray,
    public_members: pd.DataFrame,
) -> dict[str, object]:
    directory = Path(root) / method
    submission = pd.read_csv(directory / "submission.csv")
    if list(submission.columns) != ["row_id", "target"]:
        raise ValueError(f"{method} submission columns differ")
    row_id = submission["row_id"].to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(row_id, sample_row_id):
        raise ValueError(f"{method} row_id differs from sample submission")
    if submission["row_id"].duplicated().any():
        raise ValueError(f"{method} row_id is duplicated")
    prediction = submission["target"].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(prediction).all():
        raise ValueError(f"{method} target is non-finite")
    recipe = json.loads((directory / "recipe.json").read_text(encoding="utf-8"))
    rebuilt = reconstruct_submission(public_members, recipe)
    max_abs = float(np.max(np.abs(prediction - rebuilt)))
    if max_abs > 1e-12:
        raise ValueError(f"{method} recipe rebuild differs by {max_abs}")
    return {
        "method": method,
        "rows": int(len(submission)),
        "mean": float(prediction.mean()),
        "std_pop": float(prediction.std(ddof=0)),
        "min": float(prediction.min()),
        "max": float(prediction.max()),
        "recipe_rebuild_max_abs": max_abs,
        "submission_sha256": sha256_file(directory / "submission.csv"),
    }


def package_replay_model(root: Path, method: str) -> Path:
    strategy = Path(root) / method / "runner_model"
    strategy.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name("replay_main.py"), strategy / "main.py")
    return strategy


def run_official_replays(
    root: Path,
    data_root: Path,
    api_dir: Path,
    methods: Sequence[str],
) -> dict[str, object]:
    sys.path.insert(0, str(api_dir))
    try:
        from runner import run_strategy

        reports: dict[str, object] = {}
        for method in methods:
            strategy = package_replay_model(root, method)
            output = Path(root) / method / "runner_submission.csv"
            result = run_strategy(
                data_root=data_root,
                strategy_dir=strategy,
                output_path=output,
                split="test",
            )
            report = result.as_dict()
            report["audit_scope"] = "public_replay_format_and_order_only"
            source = pd.read_csv(Path(root) / method / "submission.csv")
            replay = pd.read_csv(output)
            if not np.array_equal(source["row_id"].to_numpy(), replay["row_id"].to_numpy()):
                raise ValueError(f"{method} runner row_id differs")
            difference = np.abs(
                source["target"].to_numpy(dtype=np.float64)
                - replay["target"].to_numpy(dtype=np.float64)
            )
            report["submission_max_abs_diff"] = float(difference.max())
            report["runner_submission_sha256"] = sha256_file(output)
            if report["status"] != "ok" or report["submission_max_abs_diff"] > 1e-12:
                raise ValueError(f"{method} official replay failed")
            report_path = Path(root) / method / "runner_report.json"
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
            reports[method] = report
        return reports
    finally:
        sys.path.pop(0)


def update_checksums(root: Path, method: str) -> dict[str, object]:
    directory = Path(root) / method
    files = (
        "submission.csv",
        "recipe.json",
        "report.json",
        "runner_submission.csv",
        "runner_report.json",
    )
    checksums = {
        name: {
            "bytes": int((directory / name).stat().st_size),
            "sha256": sha256_file(directory / name),
        }
        for name in files
    }
    (directory / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True), encoding="utf-8"
    )
    return checksums


def finalize(
    root: Path,
    public_members_path: Path,
    sample_submission_path: Path,
    data_root: Path,
    api_dir: Path,
    methods: Sequence[str] = METHODS,
) -> dict[str, object]:
    public_members = pd.read_parquet(public_members_path)
    sample = pd.read_csv(sample_submission_path)
    if list(sample.columns) != ["row_id", "target"]:
        raise ValueError("sample submission columns differ")
    sample_row_id = sample["row_id"].to_numpy(dtype=np.int64, copy=False)
    validations = {
        method: validate_method(root, method, sample_row_id, public_members)
        for method in methods
    }
    runner = run_official_replays(root, data_root, api_dir, methods)
    manifest_methods: dict[str, object] = {}
    for method in methods:
        report = json.loads(
            (Path(root) / method / "report.json").read_text(encoding="utf-8")
        )
        checksums = update_checksums(root, method)
        manifest_methods[method] = {
            **validations[method],
            "verdict": report["verdict"],
            "recommended": bool(report["recommended"]),
            "runner_status": runner[method]["status"],
            "runner_predict_calls": runner[method]["timing"]["predict_calls"],
            "checksums": checksums,
        }
    manifest = {
        "methods": manifest_methods,
        "recommended_method": next(
            (
                method
                for method in methods
                if manifest_methods[method]["recommended"]
            ),
            None,
        ),
        "public_members": {
            "path": str(public_members_path),
            "rows": int(len(public_members)),
            "sha256": sha256_file(public_members_path),
        },
        "runner_audit_scope": "public_replay_format_and_order_only",
    }
    summary = Path(root) / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    (summary / "submission_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--public-members", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--api-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    args = parser.parse_args()
    print(
        json.dumps(
            finalize(
                args.root,
                args.public_members,
                args.sample_submission,
                args.data_root,
                args.api_dir,
                args.methods,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
