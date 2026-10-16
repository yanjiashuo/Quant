from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from repro.baseline00 import MEMBER_NAMES
from repro.baseline00.metrics import (
    fit_positive_stack,
    optimal_scale,
    predict_positive_stack,
    prediction_summary,
    weighted_zero_mean_r2,
)
from repro.baseline00.rolling import V1_OPERATIONS
from repro.baseline00_02.config import EXPECTED_ITERATIONS, VARIANT


@dataclass(frozen=True)
class FoldMembers:
    matrix: np.ndarray
    y: np.ndarray
    weight: np.ndarray
    row_id: np.ndarray
    time_id: np.ndarray
    asset_id: np.ndarray

    def training_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.matrix, self.y, self.weight


def load_fold(path: Path) -> FoldMembers:
    archive_path = Path(path)
    required = (*MEMBER_NAMES, "y", "w", "row_id", "time_id", "asset_id")
    with np.load(archive_path) as data:
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(f"{archive_path} is missing {missing}")
        matrix = np.column_stack([data[name] for name in MEMBER_NAMES]).astype(
            np.float64, copy=False
        )
        y = np.asarray(data["y"], dtype=np.float64).reshape(-1)
        weight = np.asarray(data["w"], dtype=np.float64).reshape(-1)
        row_id = np.asarray(data["row_id"], dtype=np.int64).reshape(-1)
        time_id = np.asarray(data["time_id"]).reshape(-1)
        asset_id = np.asarray(data["asset_id"]).reshape(-1)
    rows = matrix.shape[0]
    vectors = (y, weight, row_id, time_id, asset_id)
    if matrix.ndim != 2 or matrix.shape[1] != len(MEMBER_NAMES):
        raise ValueError(f"{archive_path} has an invalid member matrix")
    if any(vector.size != rows for vector in vectors):
        raise ValueError(f"{archive_path} has inconsistent row counts")
    if rows == 0:
        raise ValueError(f"{archive_path} is empty")
    if not np.isfinite(matrix).all() or not np.isfinite(y).all():
        raise ValueError(f"{archive_path} contains non-finite model values")
    if not np.isfinite(weight).all() or np.any(weight < 0.0):
        raise ValueError(f"{archive_path} contains invalid weights")
    if float(weight.sum()) <= 0.0:
        raise ValueError(f"{archive_path} has no positive weight")
    if np.unique(row_id).size != rows:
        raise ValueError(f"{archive_path} row_id values must be unique")
    if np.any(time_id[1:] < time_id[:-1]):
        raise ValueError(f"{archive_path} is not ordered by increasing time_id")
    return FoldMembers(matrix, y, weight, row_id, time_id, asset_id)


def _fit(folds: Sequence[FoldMembers]) -> dict[str, Any]:
    return fit_positive_stack(
        np.concatenate([fold.matrix for fold in folds]),
        np.concatenate([fold.y for fold in folds]),
        np.concatenate([fold.weight for fold in folds]),
        alpha=VARIANT.ridge_alpha,
        member_names=MEMBER_NAMES,
    )


def _evaluation(
    fold: FoldMembers,
    raw_prediction: np.ndarray,
    prior_scale: float,
    ridge_fit_folds: Sequence[int],
    scale_fit_folds: Sequence[int],
) -> dict[str, Any]:
    scaled = raw_prediction * prior_scale
    return {
        "report_fold": int(scale_fit_folds[-1] + 1),
        "report_rows": int(len(fold.y)),
        "ridge_fit_folds": list(ridge_fit_folds),
        "scale_fit_folds": list(scale_fit_folds),
        "prior_scale": float(prior_scale),
        "raw_score": weighted_zero_mean_r2(fold.y, raw_prediction, fold.weight),
        "prior_scaled_score": weighted_zero_mean_r2(fold.y, scaled, fold.weight),
        "raw_prediction": prediction_summary(raw_prediction),
        "prior_scaled_prediction": prediction_summary(scaled),
    }


def _coefficient_drift(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    difference = np.asarray(right["effective_coefficients"], dtype=np.float64) - np.asarray(
        left["effective_coefficients"], dtype=np.float64
    )
    return {
        "l1": float(np.sum(np.abs(difference))),
        "l2": float(np.linalg.norm(difference)),
        "max_abs": float(np.max(np.abs(difference))),
    }


def _scale_sensitivity(
    fold: FoldMembers, prediction: np.ndarray, scale: float
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for label, multiplier in (("c8_x_0.9", 0.9), ("c8", 1.0), ("c8_x_1.1", 1.1)):
        candidate = float(scale * multiplier)
        result[label] = {
            "scale": candidate,
            "score": weighted_zero_mean_r2(
                fold.y, prediction * candidate, fold.weight
            ),
        }
    return result


def fit_strict_recent_stack(
    members_root: Path, recipe_output: Path, report_output: Path
) -> dict[str, Any]:
    root = Path(members_root)
    fold6 = load_fold(root / "fold_6.npz")
    fold7 = load_fold(root / "fold_7.npz")
    fold8 = load_fold(root / "fold_8.npz")
    folds = (fold6, fold7, fold8)
    for earlier, later in zip(folds, folds[1:]):
        if earlier.time_id.max() >= later.time_id.min():
            raise ValueError("member folds must be strictly chronological and disjoint")
        if np.intersect1d(earlier.row_id, later.row_id).size:
            raise ValueError("member folds must have disjoint row_id values")

    r6 = _fit((fold6,))
    z6 = predict_positive_stack(fold6.matrix, r6)
    c6 = optimal_scale(fold6.y, z6, fold6.weight)
    z7 = predict_positive_stack(fold7.matrix, r6)

    r67 = _fit((fold6, fold7))
    c7 = optimal_scale(fold7.y, z7, fold7.weight)
    z8 = predict_positive_stack(fold8.matrix, r67)

    r678 = _fit((fold6, fold7, fold8))
    c8 = optimal_scale(fold8.y, z8, fold8.weight)
    scales = np.asarray([c6, c7, c8], dtype=np.float64)
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("chronological optimal scales must be positive and finite")

    report: dict[str, Any] = {
        "variant_name": VARIANT.name,
        "validation_protocol": "gap0_chronological_meta",
        "member_names": list(MEMBER_NAMES),
        "rolling_operations": list(V1_OPERATIONS),
        "fold7_evaluation": _evaluation(fold7, z7, c6, (6,), (6,)),
        "fold8_evaluation": _evaluation(fold8, z8, c7, (6, 7), (7,)),
        "meta_prediction_provenance": {
            "z7": "R6 predicts Fold7",
            "z8": "R67 predicts Fold8",
        },
        "production_fit": {
            "ridge_fit_folds": [6, 7, 8],
            "scale_fit_folds": [8],
            "fit_rows": int(len(fold6.y) + len(fold7.y) + len(fold8.y)),
            "c6": float(c6),
            "c7": float(c7),
            "c8": float(c8),
            "stack": r678,
            "scale_transfer": "c8 estimated from R67(Fold8), applied to R678(public)",
            "scale_sensitivity_on_z8": _scale_sensitivity(fold8, z8, c8),
        },
        "ridge_sequence": {"R6": r6, "R67": r67, "R678": r678},
        "coefficient_drift": {
            "R6_to_R67": _coefficient_drift(r6, r67),
            "R67_to_R678": _coefficient_drift(r67, r678),
        },
    }
    recipe = {
        "variant_name": VARIANT.name,
        "member_names": list(MEMBER_NAMES),
        "ridge_alpha": VARIANT.ridge_alpha,
        "effective_coefficients": r678["effective_coefficients"],
        "standard_deviations": r678["standard_deviations"],
        "normalized_coefficients": r678["normalized_coefficients"],
        "amplitude_multiplier": float(c8),
        "purge_gap_time_ids": VARIANT.validation_gap_time_ids,
        "production_gap_time_ids": VARIANT.production_gap_time_ids,
        "fit_folds": list(VARIANT.oof_folds),
        "scale_source_fold": VARIANT.scale_source_fold,
        "base_random_state": VARIANT.base_seed,
        "shared_random_state": VARIANT.shared_seed,
        "rolling_operations": list(V1_OPERATIONS),
        "base_model_iterations": EXPECTED_ITERATIONS,
        "frozen_before_public_test": True,
        "frozen_before_fold8": False,
    }
    recipe_path = Path(recipe_output)
    report_path = Path(report_output)
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path.write_text(json.dumps(recipe, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", type=Path, required=True)
    parser.add_argument("--recipe-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    report = fit_strict_recent_stack(args.members, args.recipe_out, args.report_out)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
