from __future__ import annotations

"""Baseline05 offline walk-forward evaluation for CS-rank replacement/append.

E1/E2 replace the sixth member (``cs_rank``) and keep seven members; E3 appends
``cs_rank_long`` after ``cs_rank`` for eight members. Parent and child are both
scored with the frozen :func:`evaluate_walk_forward`, and the verdict is a purely
offline gate (Fold7/Fold8 deltas, 20-block positivity, positive Ridge
coefficient). Public labels never enter this evaluation. For E3, a positive-Ridge
coefficient of zero on the appended member is the valid finding
``collinear_no_gain`` — the report and CSV are still emitted.
"""

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from repro.baseline00.metrics import prediction_summary, weighted_zero_mean_r2
from repro.baseline02.artifacts import OOFFold
from repro.baseline02.stack import evaluate_walk_forward, promotion_gate
from repro.baseline03.stack import (
    _block_deltas,
    _weighted_correlation,
    _weighted_pearson_matrix,
)
from repro.baseline04.contract import write_json
from repro.baseline05.config import RIDGE_ALPHA
from repro.baseline05.parent import load_baseline04_folds
from repro.baseline05.replacement import (
    append_member_from_archive,
    replace_member_from_archive,
)

_ZERO_COEFFICIENT = 1e-12


def _prediction_from_fit(fold: OOFFold, fit: Mapping[str, object]) -> np.ndarray:
    coefficients = np.asarray(fit["effective_coefficients"], dtype=np.float64)
    if coefficients.shape != (fold.matrix.shape[1],):
        raise ValueError("fit coefficients do not match fold member matrix")
    prediction = fold.matrix @ coefficients
    if not np.isfinite(prediction).all():
        raise ValueError("stack prediction is non-finite")
    return prediction


def _member_coefficients(fit: Mapping[str, object]) -> dict[str, float]:
    names = list(fit["member_names"])
    coefficients = [float(value) for value in fit["effective_coefficients"]]
    return dict(zip(names, coefficients))


def _augment_fit(fit: dict[str, object]) -> dict[str, object]:
    fit["member_coefficients"] = _member_coefficients(fit)
    return fit


def _pooled_correlation(
    folds: Mapping[int, OOFFold], left: int, right: int
) -> float:
    a = np.concatenate([fold.matrix[:, left] for fold in folds.values()])
    b = np.concatenate([fold.matrix[:, right] for fold in folds.values()])
    w = np.concatenate([fold.weight for fold in folds.values()])
    return _weighted_correlation(a, b, w)


def _pooled_pair_correlation(
    parent_folds: Mapping[int, OOFFold],
    child_folds: Mapping[int, OOFFold],
    parent_col: int,
    child_col: int,
) -> float:
    a = np.concatenate([fold.matrix[:, parent_col] for fold in parent_folds.values()])
    b = np.concatenate([fold.matrix[:, child_col] for fold in child_folds.values()])
    w = np.concatenate([fold.weight for fold in child_folds.values()])
    return _weighted_correlation(a, b, w)


def _fold_diagnostics(
    parent_fold: OOFFold,
    child_fold: OOFFold,
    parent_eval: Mapping[str, object],
    child_eval: Mapping[str, object],
    candidate_name: str,
    replaced_column: int | None,
) -> dict[str, object]:
    parent_prediction = _prediction_from_fit(parent_fold, parent_eval["fit"])
    child_prediction = _prediction_from_fit(child_fold, child_eval["fit"])
    candidate_column = list(child_fold.member_names).index(candidate_name)
    candidate = child_fold.matrix[:, candidate_column]
    residual = parent_fold.target - parent_prediction
    difference = child_prediction - parent_prediction
    parent_score = float(parent_eval["raw_score"])
    child_score = float(child_eval["raw_score"])
    coefficient = float(
        child_eval["fit"]["effective_coefficients"][candidate_column]
    )
    diagnostics: dict[str, object] = {
        "parent_score": parent_score,
        "child_score": child_score,
        "delta": child_score - parent_score,
        "candidate_effective_coefficient": coefficient,
        "block_deltas": _block_deltas(parent_fold, parent_prediction, child_prediction),
        "corr_candidate_vs_parent_residual": _weighted_correlation(
            candidate, residual, child_fold.weight
        ),
        "corr_parent_vs_child_prediction": _weighted_correlation(
            parent_prediction, child_prediction, child_fold.weight
        ),
        "weighted_pearson": {
            "member_names": list(child_fold.member_names),
            "matrix": _weighted_pearson_matrix(
                child_fold.matrix, child_fold.weight
            ).tolist(),
        },
        "parent_prediction": prediction_summary(parent_prediction),
        "child_prediction": prediction_summary(child_prediction),
        "difference": prediction_summary(difference),
    }
    if replaced_column is not None:
        replaced = parent_fold.matrix[:, replaced_column]
        diagnostics["corr_candidate_vs_replaced_cs_rank"] = _weighted_correlation(
            candidate, replaced, child_fold.weight
        )
    return diagnostics


def _verdict(
    child_report: Mapping[str, object],
    candidate_name: str,
    is_append: bool,
) -> dict[str, object]:
    fold7 = child_report["fold7"]["diagnostics"]
    fold8 = child_report["fold8"]["diagnostics"]
    gate = promotion_gate(
        delta7=float(fold7["delta"]),
        delta8=float(fold8["delta"]),
        positive_blocks7=int(fold7["block_deltas"]["positive_blocks"]),
        positive_blocks8=int(fold8["block_deltas"]["positive_blocks"]),
        coef7=float(fold7["candidate_effective_coefficient"]),
        coef8=float(fold8["candidate_effective_coefficient"]),
    )
    production_coefficient = float(
        child_report["production"]["fit"]["member_coefficients"][candidate_name]
    )
    if is_append and abs(production_coefficient) <= _ZERO_COEFFICIENT:
        verdict = "collinear_no_gain"
    elif gate["confirmation_pass"]:
        verdict = "accepted"
    else:
        verdict = "rejected"
    return {
        "verdict": verdict,
        "promotion_gate": gate,
        "production_candidate_coefficient": production_coefficient,
    }


def _finish_report(
    parent_folds: Mapping[int, OOFFold],
    child_folds: Mapping[int, OOFFold],
    candidate_name: str,
    replaced_column: int | None,
    is_append: bool,
    output: Path | None,
) -> dict[str, object]:
    parent_report = evaluate_walk_forward(parent_folds, alpha=RIDGE_ALPHA)
    child_report = evaluate_walk_forward(child_folds, alpha=RIDGE_ALPHA)
    for fit_holder in (child_report["fold7"], child_report["fold8"]):
        _augment_fit(fit_holder["fit"])
    _augment_fit(child_report["production"]["fit"])
    for fold in (7, 8):
        child_report[f"fold{fold}"]["diagnostics"] = _fold_diagnostics(
            parent_folds[fold],
            child_folds[fold],
            parent_report[f"fold{fold}"],
            child_report[f"fold{fold}"],
            candidate_name,
            replaced_column,
        )
    child_report["parent_walk_forward"] = {
        "fold7_score": float(parent_report["fold7"]["raw_score"]),
        "fold8_score": float(parent_report["fold8"]["raw_score"]),
    }
    child_report.update(_verdict(child_report, candidate_name, is_append))
    child_report["public_labels_used"] = False
    if is_append:
        left = list(child_folds[6].member_names).index("cs_rank")
        right = list(child_folds[6].member_names).index("cs_rank_long")
        child_report["corr_cs_rank_vs_cs_rank_long"] = _pooled_correlation(
            child_folds, left, right
        )
    if output is not None:
        write_json(Path(output), child_report)
    return child_report


def evaluate_replacement(
    parent_folds: Mapping[int, OOFFold],
    candidate: Mapping[int, Path],
    old_name: str,
    new_name: str,
    output: Path | None = None,
) -> dict[str, object]:
    replaced_column = list(parent_folds[6].member_names).index(old_name)
    child_folds = {
        fold: replace_member_from_archive(
            parent_folds[fold], candidate[fold], old_name, new_name
        )
        for fold in (6, 7, 8)
    }
    return _finish_report(
        parent_folds, child_folds, new_name, replaced_column, False, output
    )


def evaluate_append(
    parent_folds: Mapping[int, OOFFold],
    candidate: Mapping[int, Path],
    after: str,
    new_name: str,
    output: Path | None = None,
) -> dict[str, object]:
    child_folds = {
        fold: append_member_from_archive(
            parent_folds[fold], candidate[fold], after, new_name
        )
        for fold in (6, 7, 8)
    }
    return _finish_report(parent_folds, child_folds, new_name, None, True, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("replace", "append"), required=True)
    # In ``replace`` mode this member is swapped out; in ``append`` mode the new
    # member is inserted immediately after it. Both experiments anchor on cs_rank.
    parser.add_argument("--old-name", default="cs_rank")
    parser.add_argument("--new-name", required=True)
    parser.add_argument("--base-oof-dir", type=Path, required=True)
    parser.add_argument("--cs-rank-dir", type=Path, required=True)
    parser.add_argument("--long-oof-dir", type=Path, required=True)
    parser.add_argument("--candidate-fold-6", type=Path, required=True)
    parser.add_argument("--candidate-fold-7", type=Path, required=True)
    parser.add_argument("--candidate-fold-8", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    parent_folds = load_baseline04_folds(
        args.base_oof_dir, args.cs_rank_dir, args.long_oof_dir
    )
    candidate = {
        6: args.candidate_fold_6,
        7: args.candidate_fold_7,
        8: args.candidate_fold_8,
    }
    if args.mode == "replace":
        report = evaluate_replacement(
            parent_folds, candidate, args.old_name, args.new_name, args.out
        )
    else:
        report = evaluate_append(
            parent_folds, candidate, args.old_name, args.new_name, args.out
        )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "new_name": args.new_name,
                "member_names": report["member_names"],
                "verdict": report["verdict"],
                "production_candidate_coefficient": report[
                    "production_candidate_coefficient"
                ],
                "out": str(args.out),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
