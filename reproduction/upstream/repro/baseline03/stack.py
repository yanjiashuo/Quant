from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from repro.baseline00.metrics import (
    fit_positive_stack,
    level_deviation,
    optimal_scale,
    predict_positive_stack,
    prediction_summary,
    weighted_zero_mean_r2,
)
from repro.baseline03.artifacts import OOFFold, append_candidate, load_oof
from repro.baseline03.config import (
    AMPLITUDE_MULTIPLIER,
    CANDIDATE_MEMBERS,
    MODEL_C_LONG_RECIPE,
    OOF_FOLDS,
    PARENT_MEMBERS,
    RIDGE_ALPHA,
    ROLLING_OPERATIONS,
)


BASE_MODEL_ITERATIONS = {
    "model_a": 139,
    "model_b": 56,
    "model_c": 201,
    "weight_model": 400,
    "level_model": 60,
    "rolling_v1": 200,
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _weighted_pearson_matrix(matrix: np.ndarray, weight: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    weights = np.asarray(weight, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[0] != weights.size:
        raise ValueError("matrix rows must match weight")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("correlation input must be finite")
    if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("correlation weight must be nonnegative and positive in total")
    mean = np.sum(values * weights[:, None], axis=0) / weights.sum()
    centered = values - mean
    covariance = (centered * weights[:, None]).T @ centered
    scale = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(scale, scale)
    return np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0.0,
    )


def _contiguous_time_blocks(time_id: np.ndarray, n_blocks: int = 20) -> np.ndarray:
    time = np.asarray(time_id, dtype=np.int64).reshape(-1)
    if time.size == 0 or n_blocks <= 0:
        raise ValueError("time_id must be nonempty and n_blocks positive")
    if np.any(time[1:] < time[:-1]):
        raise ValueError("time_id must be nondecreasing")
    unique_time = np.unique(time)
    groups = np.array_split(unique_time, min(n_blocks, unique_time.size))
    lookup = {
        int(value): block
        for block, group in enumerate(groups)
        for value in group
    }
    return np.fromiter((lookup[int(value)] for value in time), dtype=np.int64)


def _combine(folds: Sequence[OOFFold]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not folds:
        raise ValueError("at least one fit fold is required")
    names = folds[0].member_names
    if any(fold.member_names != names for fold in folds):
        raise ValueError("fold member order differs")
    return (
        np.concatenate([fold.matrix for fold in folds]),
        np.concatenate([fold.target for fold in folds]),
        np.concatenate([fold.weight for fold in folds]),
    )


def _fit(folds: Sequence[OOFFold]) -> dict[str, Any]:
    matrix, target, weight = _combine(folds)
    return fit_positive_stack(
        matrix,
        target,
        weight,
        alpha=RIDGE_ALPHA,
        member_names=folds[0].member_names,
    )


def _evaluate(
    fit_folds: Sequence[OOFFold], report_fold: OOFFold, fit_numbers: Sequence[int]
) -> dict[str, Any]:
    fit = _fit(fit_folds)
    raw = predict_positive_stack(report_fold.matrix, fit)
    calibrated = raw * AMPLITUDE_MULTIPLIER
    return {
        "fit_folds": list(fit_numbers),
        "report_rows": report_fold.rows,
        "raw_score": weighted_zero_mean_r2(
            report_fold.target, raw, report_fold.weight
        ),
        "fixed_amplitude_score": weighted_zero_mean_r2(
            report_fold.target, calibrated, report_fold.weight
        ),
        "amplitude_multiplier": AMPLITUDE_MULTIPLIER,
        "raw_prediction": prediction_summary(raw),
        "fixed_amplitude_prediction": prediction_summary(calibrated),
        "fit": fit,
    }


def evaluate_parent(
    members_root: Path, report_path: Path, recipe_path: Path
) -> dict[str, Any]:
    folds = {
        fold: load_oof(Path(members_root) / f"fold_{fold}.npz")
        for fold in OOF_FOLDS
    }
    if not (
        folds[6].time_id.max() < folds[7].time_id.min()
        < folds[8].time_id.min()
    ):
        raise ValueError("OOF folds must be strictly chronological")

    fold7 = _evaluate((folds[6],), folds[7], (6,))
    fold8 = _evaluate((folds[6], folds[7]), folds[8], (6, 7))
    production_fit = _fit((folds[6], folds[7], folds[8]))
    report = {
        "variant": "g0_gap0_parent",
        "member_names": list(PARENT_MEMBERS),
        "ridge_alpha": RIDGE_ALPHA,
        "amplitude_multiplier": AMPLITUDE_MULTIPLIER,
        "validation_gap_time_ids": 0,
        "production_gap_time_ids": 0,
        "fold7": fold7,
        "fold8": fold8,
        "production": {
            "fit_folds": list(OOF_FOLDS),
            "fit": production_fit,
        },
    }
    recipe = {
        "variant_name": "g0_gap0_parent",
        "member_names": list(PARENT_MEMBERS),
        "fit_folds": list(OOF_FOLDS),
        "ridge_alpha": RIDGE_ALPHA,
        "standard_deviations": production_fit["standard_deviations"],
        "normalized_coefficients": production_fit["normalized_coefficients"],
        "effective_coefficients": production_fit["effective_coefficients"],
        "amplitude_multiplier": AMPLITUDE_MULTIPLIER,
        "purge_gap_time_ids": 0,
        "validation_gap_time_ids": 0,
        "production_gap_time_ids": 0,
        "rolling_operations": list(ROLLING_OPERATIONS),
        "base_model_iterations": BASE_MODEL_ITERATIONS,
        "base_random_state": 0,
        "shared_random_state": 0,
        "frozen_before_fold8": False,
        "frozen_before_public_test": True,
    }
    _write_json(Path(report_path), report)
    _write_json(Path(recipe_path), recipe)
    return report


def _weighted_correlation(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> float:
    matrix = np.column_stack((a, b))
    return float(_weighted_pearson_matrix(matrix, weight)[0, 1])


def _block_deltas(
    fold: OOFFold,
    parent_prediction: np.ndarray,
    child_prediction: np.ndarray,
) -> dict[str, Any]:
    blocks = _contiguous_time_blocks(fold.time_id, n_blocks=20)
    values: list[dict[str, Any]] = []
    for block in np.unique(blocks):
        selected = blocks == block
        parent_score = weighted_zero_mean_r2(
            fold.target[selected], parent_prediction[selected], fold.weight[selected]
        )
        child_score = weighted_zero_mean_r2(
            fold.target[selected], child_prediction[selected], fold.weight[selected]
        )
        values.append(
            {
                "block": int(block),
                "rows": int(selected.sum()),
                "parent_score": parent_score,
                "candidate_score": child_score,
                "delta": child_score - parent_score,
            }
        )
    return {
        "blocks": values,
        "positive_blocks": int(sum(value["delta"] > 0.0 for value in values)),
        "total_blocks": len(values),
    }


def _candidate_fold_evaluation(
    parent_fit_folds: Sequence[OOFFold],
    child_fit_folds: Sequence[OOFFold],
    parent_report: OOFFold,
    child_report: OOFFold,
    fit_numbers: Sequence[int],
) -> dict[str, Any]:
    parent_fit = _fit(parent_fit_folds)
    child_fit = _fit(child_fit_folds)
    parent_raw = predict_positive_stack(parent_report.matrix, parent_fit)
    child_raw = predict_positive_stack(child_report.matrix, child_fit)
    parent_fixed = parent_raw * AMPLITUDE_MULTIPLIER
    child_fixed = child_raw * AMPLITUDE_MULTIPLIER
    parent_raw_score = weighted_zero_mean_r2(
        parent_report.target, parent_raw, parent_report.weight
    )
    child_raw_score = weighted_zero_mean_r2(
        child_report.target, child_raw, child_report.weight
    )
    parent_fixed_score = weighted_zero_mean_r2(
        parent_report.target, parent_fixed, parent_report.weight
    )
    child_fixed_score = weighted_zero_mean_r2(
        child_report.target, child_fixed, child_report.weight
    )
    candidate = child_report.matrix[:, -1]
    residual = parent_report.target - parent_raw
    difference = child_fixed - parent_fixed
    level, deviation = level_deviation(difference, parent_report.time_id)
    total_variance = float(np.var(difference))
    return {
        "fit_folds": list(fit_numbers),
        "rows": child_report.rows,
        "parent": {
            "raw_score": parent_raw_score,
            "fixed_amplitude_score": parent_fixed_score,
            "fit": parent_fit,
            "raw_prediction": prediction_summary(parent_raw),
            "fixed_prediction": prediction_summary(parent_fixed),
        },
        "candidate": {
            "raw_score": child_raw_score,
            "fixed_amplitude_score": child_fixed_score,
            "fit": child_fit,
            "raw_prediction": prediction_summary(child_raw),
            "fixed_prediction": prediction_summary(child_fixed),
        },
        "delta": {
            "raw": child_raw_score - parent_raw_score,
            "fixed_amplitude": child_fixed_score - parent_fixed_score,
        },
        "raw_blocks": _block_deltas(parent_report, parent_raw, child_raw),
        "fixed_amplitude_blocks": _block_deltas(
            parent_report, parent_fixed, child_fixed
        ),
        "model_c_long": {
            "standalone_score": weighted_zero_mean_r2(
                child_report.target, candidate, child_report.weight
            ),
            "prediction": prediction_summary(candidate),
            "weighted_correlation_with_parent_residual": _weighted_correlation(
                candidate, residual, child_report.weight
            ),
        },
        "weighted_pearson": {
            "member_names": list(CANDIDATE_MEMBERS),
            "matrix": _weighted_pearson_matrix(
                child_report.matrix, child_report.weight
            ).tolist(),
        },
        "parent_candidate_prediction": {
            "weighted_correlation": _weighted_correlation(
                parent_fixed, child_fixed, child_report.weight
            ),
            "difference": prediction_summary(difference),
            "level_variance_fraction": (
                float(np.var(level)) / total_variance if total_variance > 0.0 else 0.0
            ),
            "deviation_variance_fraction": (
                float(np.var(deviation)) / total_variance
                if total_variance > 0.0
                else 0.0
            ),
        },
        "diagnostic_optimal_scale": {
            "parent_raw": optimal_scale(
                parent_report.target, parent_raw, parent_report.weight
            ),
            "candidate_raw": optimal_scale(
                child_report.target, child_raw, child_report.weight
            ),
            "used_for_csv": False,
        },
    }


def _passes_gate(fold7: dict[str, Any], fold8: dict[str, Any], kind: str) -> bool:
    delta7 = float(fold7["delta"][kind])
    delta8 = float(fold8["delta"][kind])
    block_key = "fixed_amplitude_blocks" if kind == "fixed_amplitude" else "raw_blocks"
    coefficient7 = float(
        fold7["candidate"]["fit"]["effective_coefficients"][-1]
    )
    coefficient8 = float(
        fold8["candidate"]["fit"]["effective_coefficients"][-1]
    )
    return bool(
        delta7 >= 0.000020
        and delta8 >= 0.0
        and (delta7 + delta8) / 2.0 >= 0.000020
        and int(fold7[block_key]["positive_blocks"]) >= 12
        and int(fold8[block_key]["positive_blocks"]) >= 10
        and coefficient7 > 0.0
        and coefficient8 > 0.0
    )


def evaluate_candidate(
    parent_root: Path,
    candidate_root: Path,
    report_path: Path,
    recipe_path: Path,
) -> dict[str, Any]:
    parent = {
        fold: load_oof(Path(parent_root) / f"fold_{fold}.npz")
        for fold in OOF_FOLDS
    }
    child = {
        fold: append_candidate(
            parent[fold], Path(candidate_root) / f"fold_{fold}.npz"
        )
        for fold in OOF_FOLDS
    }
    if not parent[6].time_id.max() < parent[7].time_id.min() < parent[8].time_id.min():
        raise ValueError("OOF folds must be strictly chronological")
    fold7 = _candidate_fold_evaluation(
        (parent[6],), (child[6],), parent[7], child[7], (6,)
    )
    fold8 = _candidate_fold_evaluation(
        (parent[6], parent[7]),
        (child[6], child[7]),
        parent[8],
        child[8],
        (6, 7),
    )
    production_fit = _fit((child[6], child[7], child[8]))
    fixed_passed = _passes_gate(fold7, fold8, "fixed_amplitude")
    raw_passed = _passes_gate(fold7, fold8, "raw")
    verdict = (
        "promoted"
        if fixed_passed
        else "direction_candidate_only"
        if raw_passed
        else "rejected"
    )
    hashes = {
        f"parent_fold_{fold}": _sha256(Path(parent_root) / f"fold_{fold}.npz")
        for fold in OOF_FOLDS
    }
    hashes.update(
        {
            f"candidate_fold_{fold}": _sha256(
                Path(candidate_root) / f"fold_{fold}.npz"
            )
            for fold in OOF_FOLDS
        }
    )
    gate = {
        "fixed_amplitude_passed": fixed_passed,
        "raw_passed": raw_passed,
        "verdict": verdict,
        "thresholds": {
            "fold7_delta_min": 0.000020,
            "fold8_delta_min": 0.0,
            "mean_delta_min": 0.000020,
            "fold7_positive_blocks_min": 12,
            "fold8_positive_blocks_min": 10,
            "model_c_long_coefficient": "> 0 in R6 and R67",
        },
    }
    report = {
        "variant": "e1_model_c_long",
        "parent_variant": "g0_gap0_parent",
        "member_names": list(CANDIDATE_MEMBERS),
        "only_structural_change": "append model_c_long trained on all prior partitions",
        "validation_gap_time_ids": 0,
        "production_gap_time_ids": 0,
        "ridge_alpha": RIDGE_ALPHA,
        "amplitude_multiplier": AMPLITUDE_MULTIPLIER,
        "fold7": fold7,
        "fold8": fold8,
        "production": {"fit_folds": list(OOF_FOLDS), "fit": production_fit},
        "gate": gate,
        "input_sha256": hashes,
    }
    recipe = {
        "variant_name": "e1_model_c_long",
        "parent_variant": "g0_gap0_parent",
        "member_names": list(CANDIDATE_MEMBERS),
        "fit_folds": list(OOF_FOLDS),
        "ridge_alpha": RIDGE_ALPHA,
        "standard_deviations": production_fit["standard_deviations"],
        "normalized_coefficients": production_fit["normalized_coefficients"],
        "effective_coefficients": production_fit["effective_coefficients"],
        "amplitude_multiplier": AMPLITUDE_MULTIPLIER,
        "purge_gap_time_ids": 0,
        "validation_gap_time_ids": 0,
        "production_gap_time_ids": 0,
        "rolling_operations": list(ROLLING_OPERATIONS),
        "base_model_iterations": BASE_MODEL_ITERATIONS,
        "base_random_state": 0,
        "shared_random_state": 0,
        "model_c_long_recipe": MODEL_C_LONG_RECIPE,
        "gate": gate,
        "recommended_submission": fixed_passed,
        "input_sha256": hashes,
        "public_labels_used": False,
        "frozen_before_public_test": True,
    }
    _write_json(Path(report_path), report)
    sibling = Path(report_path).with_name("report.json")
    if sibling != Path(report_path):
        _write_json(sibling, report)
    _write_json(Path(recipe_path), recipe)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recipe-out", type=Path, required=True)
    args = parser.parse_args()
    if args.parent or args.candidate:
        if not args.parent or not args.candidate:
            raise ValueError("candidate comparison requires both --parent and --candidate")
        result = evaluate_candidate(
            args.parent, args.candidate, args.out, args.recipe_out
        )
    else:
        if not args.members:
            raise ValueError("parent evaluation requires --members")
        result = evaluate_parent(args.members, args.out, args.recipe_out)
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
