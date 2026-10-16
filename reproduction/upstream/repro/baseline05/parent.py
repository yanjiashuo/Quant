from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from repro.baseline02.artifacts import OOFFold
from repro.baseline02.stack import augment_fold, evaluate_walk_forward
from repro.baseline04.assemble import build_parent_folds
from repro.baseline05.config import FOLDS, PARENT_MEMBERS, RIDGE_ALPHA


def load_baseline04_folds(
    base_oof_dir: Path, cs_rank_dir: Path, long_dir: Path
) -> dict[int, OOFFold]:
    """Reconstruct the frozen Baseline04 seven-member Fold6-8 folds.

    Base five members + Plan02 ``cs_rank`` (six members) then the adapted
    ``model_c_long`` archive appended as the seventh member.
    """
    six = build_parent_folds(base_oof_dir, cs_rank_dir)
    folds: dict[int, OOFFold] = {}
    for fold in FOLDS:
        augmented = augment_fold(
            six[fold], Path(long_dir) / f"fold_{fold}.npz", "model_c_long"
        )
        if augmented.member_names != PARENT_MEMBERS:
            raise ValueError(f"Fold{fold} member order differs from Baseline05 parent")
        folds[fold] = augmented
    return folds


def _maximum_difference(left: object, right: object) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"reconstruction shapes differ: {a.shape} != {b.shape}")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def verify_parent_walk_forward(
    parent_folds: Mapping[int, OOFFold],
    frozen_report: Path,
    tolerance: float = 1e-12,
) -> dict[str, object]:
    """Confirm the reconstructed Baseline04 walk-forward matches its frozen report.

    The frozen report is the Baseline04 ``evaluate_candidate`` seven-member child
    record: its Fold7/Fold8 ``child_score``+``child_fit`` and production ``fit``.
    """
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
            f"Baseline04 parent reconstruction differs by {maximum} > {tolerance}"
        )
    return {
        "passed": True,
        "tolerance": tolerance,
        "maximum_absolute_difference": maximum,
        "comparisons": comparisons,
        "rebuilt": rebuilt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-oof-dir", type=Path, required=True)
    parser.add_argument("--cs-rank-dir", type=Path, required=True)
    parser.add_argument("--long-oof-dir", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    folds = load_baseline04_folds(
        args.base_oof_dir, args.cs_rank_dir, args.long_oof_dir
    )
    result = verify_parent_walk_forward(folds, args.parent_report)
    payload = {key: value for key, value in result.items() if key != "rebuilt"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
