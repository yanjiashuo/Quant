from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from repro.baseline00 import MEMBER_NAMES
from repro.baseline00.frozen_recipe import validate_frozen_recipe
from repro.baseline00.metrics import (
    fit_positive_stack,
    predict_positive_stack,
    prediction_summary,
    weighted_zero_mean_r2,
)
from repro.baseline00.rolling import V1_OPERATIONS
from repro.baseline00_00.config import VARIANT


RIDGE_ALPHA = 0.5
EXPECTED_ITERATIONS = {
    "model_a": 139,
    "model_b": 56,
    "model_c": 201,
    "weight_model": 400,
    "level_model": 60,
    "rolling_v1": 200,
}


def _load_fold(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        missing = [name for name in (*MEMBER_NAMES, "y", "w") if name not in data.files]
        if missing:
            raise ValueError(f"{path} is missing {missing}")
        matrix = np.column_stack([data[name] for name in MEMBER_NAMES]).astype(
            np.float64, copy=False
        )
        y = np.asarray(data["y"], dtype=np.float64)
        weight = np.asarray(data["w"], dtype=np.float64)
    if matrix.shape[0] != y.size or y.shape != weight.shape:
        raise ValueError(f"{path} has inconsistent row counts")
    if not np.isfinite(matrix).all() or not np.isfinite(y).all() or not np.isfinite(weight).all():
        raise ValueError(f"{path} contains non-finite values")
    if np.any(weight < 0):
        raise ValueError(f"{path} contains negative weights")
    return matrix, y, weight


def _fit(members_root: Path, folds: Sequence[int]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    loaded = [_load_fold(members_root / f"fold_{fold}.npz") for fold in folds]
    matrix = np.concatenate([item[0] for item in loaded])
    y = np.concatenate([item[1] for item in loaded])
    weight = np.concatenate([item[2] for item in loaded])
    stack = fit_positive_stack(
        matrix, y, weight, alpha=RIDGE_ALPHA, member_names=MEMBER_NAMES
    )
    return stack, matrix, y, weight


def _chronological_evaluation(
    members_root: Path, fit_folds: Sequence[int], report_fold: int
) -> dict[str, Any]:
    stack, _, _, _ = _fit(members_root, fit_folds)
    matrix, y, weight = _load_fold(members_root / f"fold_{report_fold}.npz")
    prediction = predict_positive_stack(matrix, stack)
    return {
        "fit_folds": list(fit_folds),
        "report_fold": int(report_fold),
        "report_rows": int(len(y)),
        "score": weighted_zero_mean_r2(y, prediction, weight),
        "prediction": prediction_summary(prediction),
        "stack": stack,
    }


def fit_gap0_stack(
    members_root: Path, recipe_output: Path, report_output: Path
) -> dict[str, Any]:
    members_root = Path(members_root)
    chronological = [
        _chronological_evaluation(members_root, (5,), 6),
        _chronological_evaluation(members_root, (5, 6), 7),
    ]
    stack, matrix, y, weight = _fit(members_root, (5, 6, 7))
    raw_prediction = predict_positive_stack(matrix, stack)
    raw_std = float(np.std(raw_prediction))
    if raw_std <= 1e-12:
        raise ValueError("gap0 OOF stack prediction has zero standard deviation")
    amplitude = VARIANT.target_prediction_std / raw_std
    calibrated_prediction = raw_prediction * amplitude
    production_fit = {
        "fit_folds": [5, 6, 7],
        "fit_rows": int(len(y)),
        "stack": stack,
        "raw_score": weighted_zero_mean_r2(y, raw_prediction, weight),
        "calibrated_score": weighted_zero_mean_r2(y, calibrated_prediction, weight),
        "raw_prediction": prediction_summary(raw_prediction),
        "calibrated_prediction": prediction_summary(calibrated_prediction),
        "amplitude_multiplier": amplitude,
    }
    warning = (
        "Gap-zero OOF is a GitHub reproduction protocol. It is not the strict purged "
        "validation estimate and can overlap the target horizon near fold boundaries."
    )
    report = {
        "variant_name": VARIANT.name,
        "validation_protocol": "reproduction_only_gap0",
        "leakage_warning": warning,
        "base_random_state": VARIANT.base_seed,
        "shared_random_state": VARIANT.shared_seed,
        "rolling_operations": list(V1_OPERATIONS),
        "chronological_evaluations": chronological,
        "production_fit": production_fit,
    }
    recipe = {
        "variant_name": VARIANT.name,
        "member_names": list(MEMBER_NAMES),
        "ridge_alpha": RIDGE_ALPHA,
        "effective_coefficients": stack["effective_coefficients"],
        "standard_deviations": stack["standard_deviations"],
        "normalized_coefficients": stack["normalized_coefficients"],
        "amplitude_multiplier": amplitude,
        "target_prediction_std": VARIANT.target_prediction_std,
        "purge_gap_time_ids": VARIANT.validation_gap_time_ids,
        "production_gap_time_ids": VARIANT.production_gap_time_ids,
        "base_random_state": VARIANT.base_seed,
        "shared_random_state": VARIANT.shared_seed,
        "rolling_operations": list(V1_OPERATIONS),
        "validation_protocol": "reproduction_only_gap0",
        "leakage_warning": warning,
        "fit_folds": [5, 6, 7],
        "base_model_iterations": EXPECTED_ITERATIONS,
        "frozen_before_fold8": True,
    }
    recipe_output = Path(recipe_output)
    report_output = Path(report_output)
    recipe_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    recipe_output.write_text(json.dumps(recipe, indent=2, sort_keys=True), encoding="utf-8")
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    validate_frozen_recipe(recipe_output, EXPECTED_ITERATIONS)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", type=Path, required=True)
    parser.add_argument("--recipe-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            fit_gap0_stack(args.members, args.recipe_out, args.report_out),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
