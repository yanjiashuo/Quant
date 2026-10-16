from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from repro.config import RIDGE_ALPHA, TARGET_STD, TRAIN_DIR, WORK, ensure_data2_path
from repro.baseline00 import MEMBER_NAMES
from repro.baseline00.frozen_recipe import validate_frozen_recipe
from repro.baseline00.metrics import (
    fit_positive_stack,
    predict_positive_stack,
    prediction_summary,
    weighted_zero_mean_r2,
)


def load_fold(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    matrix = np.column_stack([data[name] for name in MEMBER_NAMES])
    return matrix, data["y"].astype(np.float64), data["w"].astype(np.float64)


def load_heldout_fold(path: Path, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    if "y" in data.files or "w" in data.files:
        raise ValueError("held-out member archive must not contain labels")
    matrix = np.column_stack([data[name] for name in MEMBER_NAMES])
    labels = pd.read_parquet(
        TRAIN_DIR / f"train_partition_{fold:03d}.parquet",
        columns=["row_id", "time_id", "asset_id", "weight", "target"],
    ).sort_values(["time_id", "asset_id"], kind="stable", ignore_index=True)
    if not np.array_equal(
        data["row_id"], labels["row_id"].to_numpy(dtype=np.int64, copy=False)
    ):
        raise ValueError("held-out labels do not match member row_id order")
    return (
        matrix,
        labels["target"].to_numpy(dtype=np.float64, copy=False),
        np.maximum(labels["weight"].to_numpy(dtype=np.float64, copy=False), 0.0),
    )


def fit_and_report(
    members_root: Path,
    fit_folds: Sequence[int],
    report_fold: int | None,
    output: Path,
    recipe_output: Path | None = None,
    leakage_report: Path | None = None,
    held_out: bool = False,
    frozen_recipe: Path | None = None,
) -> dict[str, Any]:
    members_root = ensure_data2_path(members_root)
    output = ensure_data2_path(output)
    if report_fold is not None and report_fold in fit_folds:
        raise ValueError("report_fold cannot be one of fit_folds")
    if held_out and (tuple(fit_folds) != (5, 6, 7) or report_fold != 8):
        raise ValueError("held-out evaluation requires fit_folds=[5,6,7] and report_fold=8")
    if held_out:
        if frozen_recipe is None:
            raise ValueError("held-out evaluation requires --frozen-recipe")
        frozen = validate_frozen_recipe(
            frozen_recipe,
            {
                "model_a": 139,
                "model_b": 56,
                "model_c": 201,
                "weight_model": 400,
                "level_model": 60,
                "rolling_v1": 200,
            },
        )
        if frozen.get("member_names") != list(MEMBER_NAMES):
            raise ValueError("frozen recipe member order does not match Baseline 00")
        effective = np.asarray(frozen["effective_coefficients"], dtype=np.float64)
        if effective.shape != (len(MEMBER_NAMES),) or np.any(effective < 0):
            raise ValueError("frozen recipe Ridge coefficients must be one nonnegative value per member")
        stack = {
            "alpha": float(frozen["ridge_alpha"]),
            "member_names": list(MEMBER_NAMES),
            "standard_deviations": frozen["standard_deviations"],
            "normalized_coefficients": frozen["normalized_coefficients"],
            "effective_coefficients": effective.tolist(),
            "nonzero_members": int(np.count_nonzero(effective > 1e-15)),
        }
        amplitude_multiplier = float(frozen["amplitude_multiplier"])
        report: dict[str, Any] = {
            "fit_folds": list(fit_folds),
            "fit_source": str(Path(frozen_recipe)),
            "stack": stack,
            "target_prediction_std": float(frozen["target_prediction_std"]),
            "amplitude_multiplier": amplitude_multiplier,
            "held_out": True,
        }
    else:
        matrices, targets, weights = zip(
            *(load_fold(members_root / f"fold_{fold}.npz") for fold in fit_folds)
        )
        fit_matrix = np.concatenate(matrices)
        fit_y = np.concatenate(targets)
        fit_w = np.concatenate(weights)
        stack = fit_positive_stack(
            fit_matrix, fit_y, fit_w, alpha=RIDGE_ALPHA, member_names=MEMBER_NAMES
        )
        fit_prediction = predict_positive_stack(fit_matrix, stack)
        raw_std = float(np.std(fit_prediction))
        amplitude_multiplier = TARGET_STD / raw_std if raw_std > 1e-12 else 1.0
        report = {
            "fit_folds": list(fit_folds),
            "fit_rows": int(len(fit_y)),
            "stack": stack,
            "fit_raw_score": weighted_zero_mean_r2(fit_y, fit_prediction, fit_w),
            "fit_prediction": prediction_summary(fit_prediction),
            "target_prediction_std": TARGET_STD,
            "amplitude_multiplier": amplitude_multiplier,
            "held_out": False,
        }
    if report_fold is not None:
        report_path = members_root / f"fold_{report_fold}.npz"
        report_matrix, report_y, report_w = (
            load_heldout_fold(report_path, report_fold)
            if held_out
            else load_fold(report_path)
        )
        raw_prediction = predict_positive_stack(report_matrix, stack)
        calibrated = raw_prediction * amplitude_multiplier
        report.update(
            report_fold=report_fold,
            report_rows=int(len(report_y)),
            report_raw_score=weighted_zero_mean_r2(report_y, raw_prediction, report_w),
            report_calibrated_score=weighted_zero_mean_r2(
                report_y, calibrated, report_w
            ),
            report_raw_prediction=prediction_summary(raw_prediction),
            report_calibrated_prediction=prediction_summary(calibrated),
            report_member_scores={
                name: weighted_zero_mean_r2(
                    report_y, report_matrix[:, column], report_w
                )
                for column, name in enumerate(MEMBER_NAMES)
            },
            held_out_labels_read_once=held_out,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if recipe_output is not None:
        recipe_output = ensure_data2_path(recipe_output)
        if leakage_report is None:
            raise ValueError("recipe_output requires leakage_report")
        leakage = json.loads(Path(leakage_report).read_text(encoding="utf-8"))
        if leakage.get("fold8_read") is not False:
            raise ValueError("leakage report must prove that Fold 8 was not read")
        purge_gap = int(leakage["selection"]["selected_gap_time_ids"])
        recipe = {
            "member_names": list(MEMBER_NAMES),
            "ridge_alpha": RIDGE_ALPHA,
            "effective_coefficients": stack["effective_coefficients"],
            "standard_deviations": stack["standard_deviations"],
            "normalized_coefficients": stack["normalized_coefficients"],
            "amplitude_multiplier": amplitude_multiplier,
            "target_prediction_std": TARGET_STD,
            "purge_gap_time_ids": purge_gap,
            "leakage_report": str(Path(leakage_report)),
            "fit_folds": list(fit_folds),
            "base_model_iterations": {
                "model_a": 139,
                "model_b": 56,
                "model_c": 201,
                "weight_model": 400,
                "level_model": 60,
                "rolling_v1": 200,
            },
            "frozen_before_fold8": True,
        }
        recipe_output.parent.mkdir(parents=True, exist_ok=True)
        recipe_output.write_text(
            json.dumps(recipe, indent=2, sort_keys=True), encoding="utf-8"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", type=Path, default=WORK / "members")
    parser.add_argument("--fit-folds", type=int, nargs="+", required=True)
    parser.add_argument("--report-fold", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recipe-out", type=Path)
    parser.add_argument("--leakage-report", type=Path)
    parser.add_argument("--held-out", action="store_true")
    parser.add_argument("--frozen-recipe", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            fit_and_report(
                args.members,
                args.fit_folds,
                args.report_fold,
                args.out,
                args.recipe_out,
                args.leakage_report,
                args.held_out,
                args.frozen_recipe,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
