from __future__ import annotations

"""Baseline05 public CSV builder for CS-rank replacement (E1/E2) and append (E3).

The frozen Baseline04 public submission is first rebuilt from its recipe with
maximum absolute difference ``<=1e-12`` before any member is touched. Candidates
are aligned by ``row_id`` with exact ``time_id``/``asset_id`` agreement. E1/E2
replace the ``cs_rank`` column in place (seven members); E3 inserts
``cs_rank_long`` after ``cs_rank`` (eight members). The version's production
coefficients are applied, the mean is preserved, and the population std is set to
``0.046``. Public labels are never used.
"""

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from repro.baseline02.finalize import reconstruct_submission
from repro.baseline02.make_submission import build_submission
from repro.baseline04.contract import sha256_file, write_json
from repro.baseline05.config import EXPECTED_SHA256, RIDGE_ALPHA, TARGET_STD


def _require_equal(name: str, expected: np.ndarray, actual: np.ndarray) -> None:
    if not np.array_equal(expected, actual):
        raise ValueError(f"{name} differs")


def verify_parent_submission(
    public_members: pd.DataFrame,
    recipe: Mapping[str, object],
    submission_path: Path,
    tolerance: float = 1e-12,
) -> float:
    """Rebuild the frozen parent submission and confirm it matches within tol."""
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


def _aligned_candidate(
    parent: pd.DataFrame, candidate: pd.DataFrame, name: str
) -> np.ndarray:
    required = ["row_id", "time_id", "asset_id", name]
    if list(candidate.columns) != required:
        raise ValueError(f"candidate columns must be {required}")
    if parent["row_id"].duplicated().any() or candidate["row_id"].duplicated().any():
        raise ValueError("public row_id contains duplicates")
    aligned = candidate.set_index("row_id").reindex(
        parent["row_id"].to_numpy(dtype=np.int64, copy=False)
    )
    if aligned.isna().any().any():
        raise ValueError(f"{name} public rows do not cover parent")
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
    prediction = aligned[name].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(prediction).all():
        raise ValueError(f"{name} public prediction is non-finite")
    return prediction


def replace_public_member(
    parent: pd.DataFrame, candidate: pd.DataFrame, old_name: str, new_name: str
) -> pd.DataFrame:
    """Swap ``old_name`` for the aligned ``new_name`` column (seven members)."""
    if old_name not in parent:
        raise ValueError(f"member to replace is missing: {old_name}")
    if new_name != old_name and new_name in parent:
        raise ValueError(f"replacement name already exists: {new_name}")
    prediction = _aligned_candidate(parent, candidate, new_name)
    result = parent.copy()
    position = result.columns.get_loc(old_name)
    result = result.drop(columns=[old_name])
    result.insert(position, new_name, prediction)
    return result


def append_public_member(
    parent: pd.DataFrame, candidate: pd.DataFrame, after: str, new_name: str
) -> pd.DataFrame:
    """Insert the aligned ``new_name`` column after ``after`` (eight members)."""
    if after not in parent:
        raise ValueError(f"anchor member is missing: {after}")
    if new_name in parent:
        raise ValueError(f"appended name already exists: {new_name}")
    prediction = _aligned_candidate(parent, candidate, new_name)
    result = parent.copy()
    position = result.columns.get_loc(after) + 1
    result.insert(position, new_name, prediction)
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


def _production_coefficients(
    walk_forward: Mapping[str, object], member_names: Sequence[str]
) -> np.ndarray:
    if tuple(walk_forward["member_names"]) != tuple(member_names):
        raise ValueError("walk-forward member order differs")
    coefficients = np.asarray(
        walk_forward["production"]["fit"]["effective_coefficients"], dtype=np.float64
    )
    if coefficients.shape != (len(member_names),):
        raise ValueError("production coefficients do not match member_names")
    return coefficients


def _build_bundle(
    child: pd.DataFrame,
    member_names: Sequence[str],
    coefficients: np.ndarray,
    walk_forward: Mapping[str, object],
    parent_rebuild_max_abs: float,
    output_dir: Path,
    method: str,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    child.to_parquet(output / "public_members.parquet", index=False)
    write_json(output / "walk_forward.json", dict(walk_forward))
    verdict = str(walk_forward["verdict"])
    report = build_submission(
        child,
        tuple(member_names),
        coefficients,
        output,
        method=method,
        verdict=verdict,
        recommended=False,
        target_std=TARGET_STD,
        extra_recipe={
            "variant": method,
            "parent_variant": "baseline04_seven_member",
            "ridge_alpha": RIDGE_ALPHA,
            "fit_folds": [6, 7, 8],
            "parent_submission_sha256": EXPECTED_SHA256["parent_submission"],
            "parent_rebuild_max_abs": parent_rebuild_max_abs,
            "public_labels_used": False,
        },
    )
    report.update(
        {
            "parent_rebuild_max_abs": parent_rebuild_max_abs,
            "parent_submission_sha256": EXPECTED_SHA256["parent_submission"],
            "public_members_sha256": sha256_file(output / "public_members.parquet"),
            "public_labels_used": False,
            "path": str(output / "submission.csv"),
        }
    )
    write_json(output / "report.json", report)
    report["checksums"] = _write_checksums(output)
    return report


def build_replacement_submission(
    public_members_path: Path,
    candidate_public_path: Path,
    parent_recipe_path: Path,
    parent_submission_path: Path,
    walk_forward_path: Path,
    output_dir: Path,
    old_name: str,
    new_name: str,
    member_names: Sequence[str],
    method: str,
) -> dict[str, object]:
    parent = pd.read_parquet(public_members_path)
    parent_recipe = json.loads(Path(parent_recipe_path).read_text(encoding="utf-8"))
    parent_rebuild_max_abs = verify_parent_submission(
        parent, parent_recipe, parent_submission_path
    )
    candidate = pd.read_parquet(candidate_public_path)
    child = replace_public_member(parent, candidate, old_name, new_name)
    walk_forward = json.loads(Path(walk_forward_path).read_text(encoding="utf-8"))
    coefficients = _production_coefficients(walk_forward, member_names)
    return _build_bundle(
        child,
        member_names,
        coefficients,
        walk_forward,
        parent_rebuild_max_abs,
        output_dir,
        method,
    )


def build_append_submission(
    public_members_path: Path,
    candidate_public_path: Path,
    parent_recipe_path: Path,
    parent_submission_path: Path,
    walk_forward_path: Path,
    output_dir: Path,
    after: str,
    new_name: str,
    member_names: Sequence[str],
    method: str,
) -> dict[str, object]:
    parent = pd.read_parquet(public_members_path)
    parent_recipe = json.loads(Path(parent_recipe_path).read_text(encoding="utf-8"))
    parent_rebuild_max_abs = verify_parent_submission(
        parent, parent_recipe, parent_submission_path
    )
    candidate = pd.read_parquet(candidate_public_path)
    child = append_public_member(parent, candidate, after, new_name)
    walk_forward = json.loads(Path(walk_forward_path).read_text(encoding="utf-8"))
    coefficients = _production_coefficients(walk_forward, member_names)
    return _build_bundle(
        child,
        member_names,
        coefficients,
        walk_forward,
        parent_rebuild_max_abs,
        output_dir,
        method,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("replace", "append"), required=True)
    parser.add_argument("--public-members", type=Path, required=True)
    parser.add_argument("--candidate-public", type=Path, required=True)
    parser.add_argument("--parent-recipe", type=Path, required=True)
    parser.add_argument("--parent-submission", type=Path, required=True)
    parser.add_argument("--walk-forward", type=Path, required=True)
    parser.add_argument("--anchor", default="cs_rank")
    parser.add_argument("--new-name", required=True)
    parser.add_argument("--member-names", nargs="+", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    builder = (
        build_replacement_submission if args.mode == "replace" else build_append_submission
    )
    anchor_kwarg = "old_name" if args.mode == "replace" else "after"
    report = builder(
        public_members_path=args.public_members,
        candidate_public_path=args.candidate_public,
        parent_recipe_path=args.parent_recipe,
        parent_submission_path=args.parent_submission,
        walk_forward_path=args.walk_forward,
        output_dir=args.out,
        **{anchor_kwarg: args.anchor},
        new_name=args.new_name,
        member_names=tuple(args.member_names),
        method=args.method,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
