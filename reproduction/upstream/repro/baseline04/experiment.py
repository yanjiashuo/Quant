from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from repro.baseline00.metrics import prediction_summary
from repro.baseline02.artifacts import OOFFold
from repro.baseline02.stack import augment_fold, evaluate_candidate
from repro.baseline03.stack import _weighted_correlation, _weighted_pearson_matrix
from repro.baseline04.assemble import build_parent_folds
from repro.baseline04.config import CHILD_MEMBERS, RIDGE_ALPHA, TARGET_STD
from repro.baseline04.contract import write_json


def _prediction_from_fit(fold: OOFFold, fit: Mapping[str, object]) -> np.ndarray:
    coefficients = np.asarray(fit["effective_coefficients"], dtype=np.float64)
    if coefficients.shape != (fold.matrix.shape[1],):
        raise ValueError("fit coefficients do not match fold member matrix")
    prediction = fold.matrix @ coefficients
    if not np.isfinite(prediction).all():
        raise ValueError("stack prediction is non-finite")
    return prediction


def _fold_diagnostics(
    parent_fold: OOFFold,
    child_fold: OOFFold,
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    parent_prediction = _prediction_from_fit(parent_fold, evaluation["parent_fit"])
    child_prediction = _prediction_from_fit(child_fold, evaluation["child_fit"])
    candidate = child_fold.matrix[:, -1]
    residual = parent_fold.target - parent_prediction
    difference = child_prediction - parent_prediction
    return {
        "weighted_pearson": {
            "member_names": list(CHILD_MEMBERS),
            "matrix": _weighted_pearson_matrix(
                child_fold.matrix, child_fold.weight
            ).tolist(),
        },
        "model_c_long_weighted_correlation_with_parent_residual": _weighted_correlation(
            candidate, residual, child_fold.weight
        ),
        "parent_child_weighted_correlation": _weighted_correlation(
            parent_prediction, child_prediction, child_fold.weight
        ),
        "parent_prediction": prediction_summary(parent_prediction),
        "child_prediction": prediction_summary(child_prediction),
        "difference": prediction_summary(difference),
    }


def run_experiment(
    parent_folds: Mapping[int, OOFFold],
    adapted_candidate_dir: Path,
    output: Path,
) -> dict[str, object]:
    report = evaluate_candidate(
        parent_folds=parent_folds,
        candidate_dir=adapted_candidate_dir,
        candidate_name="model_c_long",
        alpha=RIDGE_ALPHA,
        max_report_fold=8,
    )
    child_folds = {
        fold: augment_fold(
            parent_folds[fold],
            Path(adapted_candidate_dir) / f"fold_{fold}.npz",
            "model_c_long",
        )
        for fold in (6, 7, 8)
    }
    if tuple(report["member_names"]) != CHILD_MEMBERS:
        raise ValueError("seven-member report order differs")
    report["fold7"]["diagnostics"] = _fold_diagnostics(
        parent_folds[7], child_folds[7], report["fold7"]
    )
    report["fold8"]["diagnostics"] = _fold_diagnostics(
        parent_folds[8], child_folds[8], report["fold8"]
    )
    report.update(
        {
            "parent_variant": "plan02_e1_cs_rank",
            "variant": "baseline04_e1_cs_rank_model_c_long",
            "public_labels_used": False,
            "calibration_used_for_csv": {
                "name": "mean_preserving_std",
                "target_std_pop": TARGET_STD,
            },
        }
    )
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-oof-dir", type=Path, required=True)
    parser.add_argument("--cs-rank-dir", type=Path, required=True)
    parser.add_argument("--adapted-candidate-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run_experiment(
        build_parent_folds(args.base_oof_dir, args.cs_rank_dir),
        args.adapted_candidate_dir,
        args.out,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
