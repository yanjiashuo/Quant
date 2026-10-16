from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from repro.baseline02.finalize import reconstruct_submission
from repro.baseline02.make_submission import build_submission
from repro.baseline04.config import (
    CHILD_MEMBERS,
    EXPECTED_SHA256,
    RIDGE_ALPHA,
    TARGET_STD,
)
from repro.baseline04.contract import sha256_file, write_json


def _require_equal(name: str, expected: np.ndarray, actual: np.ndarray) -> None:
    if not np.array_equal(expected, actual):
        raise ValueError(f"{name} differs")


def verify_parent_submission(
    public_members: pd.DataFrame,
    recipe: Mapping[str, object],
    submission_path: Path,
    tolerance: float = 1e-12,
) -> float:
    expected = pd.read_csv(submission_path)
    if list(expected.columns) != ["row_id", "target"]:
        raise ValueError("parent submission columns differ")
    _require_equal(
        "parent submission row_id",
        public_members["row_id"].to_numpy(dtype=np.int64, copy=False),
        expected["row_id"].to_numpy(dtype=np.int64, copy=False),
    )
    rebuilt = reconstruct_submission(public_members, dict(recipe))
    target = expected["target"].to_numpy(dtype=np.float64, copy=False)
    difference = float(np.max(np.abs(target - rebuilt)))
    if difference > tolerance:
        raise ValueError(
            f"parent submission rebuild differs by {difference} > {tolerance}"
        )
    return difference


def append_public_model_c_long(
    parent: pd.DataFrame, long_prediction: pd.DataFrame
) -> pd.DataFrame:
    required_parent = ("row_id", "time_id", "asset_id", *CHILD_MEMBERS[:-1])
    missing = [column for column in required_parent if column not in parent]
    if missing:
        raise ValueError(f"parent public members are missing: {missing}")
    required_long = ["row_id", "time_id", "asset_id", "model_c_long"]
    if list(long_prediction.columns) != required_long:
        raise ValueError("model_c_long public schema differs")
    if parent["row_id"].duplicated().any() or long_prediction["row_id"].duplicated().any():
        raise ValueError("public row_id contains duplicates")
    aligned = long_prediction.set_index("row_id").reindex(
        parent["row_id"].to_numpy(dtype=np.int64, copy=False)
    )
    if aligned.isna().any().any():
        raise ValueError("model_c_long public rows do not cover parent")
    _require_equal(
        "public time_id",
        parent["time_id"].to_numpy(dtype=np.int64, copy=False),
        aligned["time_id"].to_numpy(dtype=np.int64, copy=False),
    )
    _require_equal(
        "public asset_id",
        parent["asset_id"].to_numpy(dtype=np.int64, copy=False),
        aligned["asset_id"].to_numpy(dtype=np.int64, copy=False),
    )
    prediction = aligned["model_c_long"].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(prediction).all():
        raise ValueError("model_c_long public prediction is non-finite")
    result = parent.loc[:, required_parent].copy()
    result["model_c_long"] = prediction
    return result


def _write_checksums(root: Path) -> dict[str, object]:
    files = (
        "submission.csv",
        "recipe.json",
        "report.json",
        "walk_forward.json",
        "public_members.parquet",
    )
    checksums = {
        name: {
            "bytes": int((root / name).stat().st_size),
            "sha256": sha256_file(root / name),
        }
        for name in files
    }
    write_json(root / "checksums.json", checksums)
    return checksums


def build_baseline04_submission(
    public_members_path: Path,
    public_model_c_long_path: Path,
    parent_recipe_path: Path,
    parent_submission_path: Path,
    walk_forward_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    parent = pd.read_parquet(public_members_path)
    parent_recipe = json.loads(Path(parent_recipe_path).read_text(encoding="utf-8"))
    parent_rebuild_max_abs = verify_parent_submission(
        parent, parent_recipe, parent_submission_path
    )
    long_prediction = pd.read_parquet(public_model_c_long_path)
    child = append_public_model_c_long(parent, long_prediction)
    walk_forward = json.loads(Path(walk_forward_path).read_text(encoding="utf-8"))
    if tuple(walk_forward["member_names"]) != CHILD_MEMBERS:
        raise ValueError("walk-forward member order differs")
    coefficients = np.asarray(
        walk_forward["production"]["fit"]["effective_coefficients"],
        dtype=np.float64,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    child.to_parquet(output / "public_members.parquet", index=False)
    write_json(output / "walk_forward.json", walk_forward)
    verdict = str(walk_forward["verdict"])
    report = build_submission(
        child,
        CHILD_MEMBERS,
        coefficients,
        output,
        method="e1_cs_rank_model_c_long",
        verdict=verdict,
        recommended=verdict == "promoted",
        target_std=TARGET_STD,
        extra_recipe={
            "variant": "baseline04_e1_cs_rank_model_c_long",
            "parent_variant": "plan02_e1_cs_rank",
            "ridge_alpha": RIDGE_ALPHA,
            "fit_folds": [6, 7, 8],
            "parent_submission_sha256": EXPECTED_SHA256["parent_submission"],
            "parent_rebuild_max_abs": parent_rebuild_max_abs,
            "public_labels_used": False,
            "offline_gate": walk_forward["gate"],
        },
    )
    report.update(
        {
            "parent_rebuild_max_abs": parent_rebuild_max_abs,
            "parent_submission_sha256": EXPECTED_SHA256["parent_submission"],
            "public_members_sha256": sha256_file(output / "public_members.parquet"),
            "public_labels_used": False,
        }
    )
    write_json(output / "report.json", report)
    checksums = _write_checksums(output)
    report["checksums"] = checksums
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-members", type=Path, required=True)
    parser.add_argument("--public-model-c-long", type=Path, required=True)
    parser.add_argument("--parent-recipe", type=Path, required=True)
    parser.add_argument("--parent-submission", type=Path, required=True)
    parser.add_argument("--walk-forward", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build_baseline04_submission(
        args.public_members,
        args.public_model_c_long,
        args.parent_recipe,
        args.parent_submission,
        args.walk_forward,
        args.out,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
