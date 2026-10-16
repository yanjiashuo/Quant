from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from repro.baseline00.metrics import (
    correlation_matrix,
    fit_positive_stack,
    predict_positive_stack,
    prediction_summary,
    weighted_zero_mean_r2,
)
from repro.baseline01.config import (
    AMPLITUDE_POLICY_BY_VARIANT,
    BASELINE,
    E0_REFERENCE_SCORES,
    E0_REFERENCE_TOLERANCE,
    FIT_FOLDS_BY_REPORT,
    GATE,
    GATE_BY_VARIANT,
    REPORT_FOLDS,
    VARIANT_PATCHES,
    ResolvedVariant,
    resolve_variant,
)
from repro.config import RIDGE_ALPHA, TARGET_STD, TRAIN_DIR


def apply_amplitude(prediction: np.ndarray, multiplier: float) -> np.ndarray:
    values = np.asarray(prediction, dtype=np.float64).reshape(-1)
    multiplier = float(multiplier)
    if not np.isfinite(values).all():
        raise ValueError("prediction must be finite")
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("amplitude multiplier must be finite and positive")
    return values * multiplier


def comparison_amplitudes(
    parent: ResolvedVariant,
    candidate: ResolvedVariant,
    parent_fit: dict[str, Any],
) -> tuple[float, float, str]:
    policy = AMPLITUDE_POLICY_BY_VARIANT[candidate.name]
    parent_amplitude = (
        float(parent_fit["legacy_amplitude_multiplier"])
        if parent.name == BASELINE.name
        else float(parent.amplitude_multiplier)
    )
    if policy == "candidate_fixed":
        candidate_amplitude = float(candidate.amplitude_multiplier)
    elif policy == "shared_parent_legacy":
        if not np.isclose(
            parent.amplitude_multiplier,
            candidate.amplitude_multiplier,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("shared-amplitude control changed the resolved amplitude")
        candidate_amplitude = parent_amplitude
    else:
        raise ValueError(f"unknown amplitude policy: {policy}")
    return parent_amplitude, candidate_amplitude, policy


def evaluate_gate(
    deltas: Sequence[float], gate: str = "numeric"
) -> dict[str, float | int | bool | str]:
    values = np.asarray(deltas, dtype=np.float64).reshape(-1)
    if values.size != len(REPORT_FOLDS) or not np.isfinite(values).all():
        raise ValueError(f"gate requires {len(REPORT_FOLDS)} finite fold deltas")
    mean_delta = float(np.mean(values))
    minimum = float(np.min(values))
    nonnegative = int(np.count_nonzero(values >= 0.0))
    if gate == "numeric":
        passed = bool(
            mean_delta >= float(GATE["minimum_mean_delta"])
            and nonnegative >= int(GATE["minimum_nonnegative_folds"])
            and minimum >= float(GATE["minimum_single_fold_delta"])
        )
        thresholds: dict[str, float | int] = dict(GATE)
    elif gate == "correctness":
        passed = bool(minimum >= 0.0)
        thresholds = {"minimum_single_fold_delta": 0.0}
    else:
        raise ValueError(f"unknown gate: {gate}")
    return {
        "mode": gate,
        "passed": passed,
        "mean_delta": mean_delta,
        "minimum_fold_delta": minimum,
        "nonnegative_folds": nonnegative,
        "thresholds": thresholds,
    }


def validate_e0_reference(
    reproduced: dict[int, dict[str, float]],
) -> dict[str, Any]:
    errors: list[dict[str, float | int | str]] = []
    for fold, expected in E0_REFERENCE_SCORES.items():
        if fold not in reproduced:
            raise ValueError(f"E0 reference mismatch: missing Fold {fold}")
        for score_name, expected_value in expected.items():
            actual_value = float(reproduced[fold][score_name])
            absolute_error = abs(actual_value - expected_value)
            if not np.isfinite(actual_value) or absolute_error > E0_REFERENCE_TOLERANCE:
                errors.append(
                    {
                        "fold": fold,
                        "score": score_name,
                        "expected": expected_value,
                        "actual": actual_value,
                        "absolute_error": absolute_error,
                    }
                )
    if errors:
        raise ValueError(f"E0 reference mismatch: {errors}")
    return {
        "passed": True,
        "tolerance": E0_REFERENCE_TOLERANCE,
        "expected": E0_REFERENCE_SCORES,
    }


def _validated_index(data: np.lib.npyio.NpzFile, archive: Path) -> np.ndarray:
    if "row_id" not in data.files:
        raise ValueError(f"member archive lacks row_id: {archive}")
    row_id = np.asarray(data["row_id"], dtype=np.int64).reshape(-1)
    if np.unique(row_id).size != row_id.size:
        raise ValueError(f"member archive has duplicate row_id: {archive}")
    return row_id


def _external_labels(
    train_dir: Path, fold: int, expected_row_id: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(train_dir) / f"train_partition_{fold:03d}.parquet"
    labels = pd.read_parquet(
        path,
        columns=["row_id", "time_id", "asset_id", "weight", "target"],
    ).sort_values(["time_id", "asset_id"], kind="stable", ignore_index=True)
    label_row_id = labels["row_id"].to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(expected_row_id, label_row_id):
        raise ValueError(f"fold {fold} label row_id does not match member archive")
    return (
        labels["target"].to_numpy(dtype=np.float64, copy=True),
        np.maximum(
            labels["weight"].to_numpy(dtype=np.float64, copy=True),
            0.0,
        ),
    )


def load_fold_members(
    archive: Path,
    member_names: tuple[str, ...],
    *,
    fold: int | None = None,
    train_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    archive = Path(archive)
    with np.load(archive, allow_pickle=False) as data:
        missing = [name for name in member_names if name not in data.files]
        if missing:
            raise ValueError(f"member archive lacks columns {missing}: {archive}")
        row_id = _validated_index(data, archive)
        columns = [np.asarray(data[name], dtype=np.float64).reshape(-1) for name in member_names]
        if any(column.size != row_id.size for column in columns):
            raise ValueError(f"member archive arrays have unequal lengths: {archive}")
        matrix = np.column_stack(columns)
        has_y = "y" in data.files
        has_w = "w" in data.files
        if has_y != has_w:
            raise ValueError(f"member archive must contain both y and w or neither: {archive}")
        if has_y:
            target = np.asarray(data["y"], dtype=np.float64).reshape(-1)
            weight = np.asarray(data["w"], dtype=np.float64).reshape(-1)
        else:
            if fold is None or train_dir is None:
                raise ValueError("label-free member archive requires fold and train_dir")
            target, weight = _external_labels(Path(train_dir), fold, row_id)
    if target.size != row_id.size or weight.size != row_id.size:
        raise ValueError(f"labels do not match member archive row count: {archive}")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all() or not np.isfinite(weight).all():
        raise ValueError(f"member archive and labels must be finite: {archive}")
    if np.any(weight < 0.0):
        raise ValueError(f"weights must be nonnegative: {archive}")
    return matrix, target, weight, row_id


def fit_for_report_fold(
    members_root: Path,
    member_names: tuple[str, ...],
    report_fold: int,
    *,
    train_dir: Path = TRAIN_DIR,
) -> dict[str, Any]:
    if report_fold not in FIT_FOLDS_BY_REPORT:
        raise ValueError(f"unsupported report fold: {report_fold}")
    fit_folds = FIT_FOLDS_BY_REPORT[report_fold]
    if report_fold in fit_folds or max(fit_folds) >= report_fold:
        raise ValueError("report fold must be strictly later than every fit fold")
    matrices: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for fold in fit_folds:
        matrix, target, weight, _ = load_fold_members(
            Path(members_root) / f"fold_{fold}.npz",
            member_names,
            fold=fold,
            train_dir=train_dir,
        )
        matrices.append(matrix)
        targets.append(target)
        weights.append(weight)
    fit_matrix = np.concatenate(matrices, axis=0)
    fit_target = np.concatenate(targets)
    fit_weight = np.concatenate(weights)
    stack = fit_positive_stack(
        fit_matrix,
        fit_target,
        fit_weight,
        alpha=RIDGE_ALPHA,
        member_names=member_names,
    )
    fit_prediction = predict_positive_stack(fit_matrix, stack)
    fit_prediction_std = float(np.std(fit_prediction))
    legacy_amplitude_multiplier = (
        TARGET_STD / fit_prediction_std if fit_prediction_std > 1e-12 else 1.0
    )
    report_matrix, report_target, report_weight, row_id = load_fold_members(
        Path(members_root) / f"fold_{report_fold}.npz",
        member_names,
        fold=report_fold,
        train_dir=train_dir,
    )
    raw_prediction = predict_positive_stack(report_matrix, stack)
    return {
        "fit_folds": list(fit_folds),
        "stack": stack,
        "matrix": report_matrix,
        "target": report_target,
        "weight": report_weight,
        "row_id": row_id,
        "raw_prediction": raw_prediction,
        "fit_prediction_std": fit_prediction_std,
        "legacy_amplitude_multiplier": legacy_amplitude_multiplier,
    }


def _member_report(
    matrix: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    names: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "scores": {
            name: weighted_zero_mean_r2(target, matrix[:, index], weight)
            for index, name in enumerate(names)
        },
        "correlation": correlation_matrix(matrix, names),
    }


def evaluate_variant(
    parent: ResolvedVariant,
    candidate: ResolvedVariant,
    parent_members_root: Path,
    candidate_members_root: Path,
    output: Path,
    *,
    train_dir: Path = TRAIN_DIR,
    enforce_e0_reference: bool = False,
) -> dict[str, Any]:
    if candidate.name not in GATE_BY_VARIANT:
        raise ValueError(f"candidate has no declared gate: {candidate.name}")
    folds: list[dict[str, Any]] = []
    deltas: list[float] = []
    production_amplitude_deltas: list[float] = []
    reference_scores: dict[int, dict[str, float]] = {}
    same_raw_inputs = (
        Path(parent_members_root).resolve() == Path(candidate_members_root).resolve()
        and parent.member_names == candidate.member_names
    )
    for report_fold in REPORT_FOLDS:
        parent_fit = fit_for_report_fold(
            parent_members_root,
            parent.member_names,
            report_fold,
            train_dir=train_dir,
        )
        candidate_fit = (
            parent_fit
            if same_raw_inputs
            else fit_for_report_fold(
                candidate_members_root,
                candidate.member_names,
                report_fold,
                train_dir=train_dir,
            )
        )
        row_id_equal = bool(np.array_equal(parent_fit["row_id"], candidate_fit["row_id"]))
        if not row_id_equal:
            raise ValueError(f"fold {report_fold} parent and candidate row_id differ")
        if not np.array_equal(parent_fit["target"], candidate_fit["target"]):
            raise ValueError(f"fold {report_fold} parent and candidate targets differ")
        if not np.array_equal(parent_fit["weight"], candidate_fit["weight"]):
            raise ValueError(f"fold {report_fold} parent and candidate weights differ")
        parent_amplitude, candidate_amplitude, amplitude_policy = comparison_amplitudes(
            parent, candidate, parent_fit
        )
        parent_prediction = apply_amplitude(
            parent_fit["raw_prediction"], parent_amplitude
        )
        candidate_prediction = apply_amplitude(
            candidate_fit["raw_prediction"], candidate_amplitude
        )
        target = candidate_fit["target"]
        weight = candidate_fit["weight"]
        parent_score = weighted_zero_mean_r2(target, parent_prediction, weight)
        candidate_score = weighted_zero_mean_r2(target, candidate_prediction, weight)
        delta = candidate_score - parent_score
        deltas.append(delta)
        parent_coefficients = np.asarray(
            parent_fit["stack"]["effective_coefficients"], dtype=np.float64
        )
        candidate_coefficients = np.asarray(
            candidate_fit["stack"]["effective_coefficients"], dtype=np.float64
        )
        ridge_equal = bool(
            parent_coefficients.shape == candidate_coefficients.shape
            and np.array_equal(parent_coefficients, candidate_coefficients)
        )
        member_predictions_equal = bool(
            parent_fit["matrix"].shape == candidate_fit["matrix"].shape
            and np.array_equal(parent_fit["matrix"], candidate_fit["matrix"])
        )
        raw_predictions_equal = bool(
            np.array_equal(
                parent_fit["raw_prediction"], candidate_fit["raw_prediction"]
            )
        )
        parent_raw_score = weighted_zero_mean_r2(
            target, parent_fit["raw_prediction"], weight
        )
        member_prediction_equality = {
            name: bool(
                name in candidate.member_names
                and np.array_equal(
                    parent_fit["matrix"][:, parent.member_names.index(name)],
                    candidate_fit["matrix"][:, candidate.member_names.index(name)],
                )
            )
            for name in parent.member_names
        }
        if candidate.name == "e2_rolling4":
            unchanged = ("final", "L88", "model_c", "level")
            if any(member_prediction_equality[name] is not True for name in unchanged):
                raise ValueError("E2 changed a non-Rolling member prediction")
            if member_prediction_equality["rolling_v1"] is True:
                raise ValueError("E2 Rolling4 prediction is identical to Rolling6")
        if candidate.name == "e3_drop_model_c":
            if Path(parent_members_root).resolve() != Path(candidate_members_root).resolve():
                raise ValueError("E3 must reuse the Baseline00 member archives")
            unchanged = ("final", "L88", "level", "rolling_v1")
            if any(member_prediction_equality[name] is not True for name in unchanged):
                raise ValueError("E3 changed a retained member prediction")
            if member_prediction_equality["model_c"] is not False:
                raise ValueError("E3 did not remove standalone model_c from Ridge")
        production_amplitude = float(candidate.amplitude_multiplier)
        production_parent_score = weighted_zero_mean_r2(
            target,
            apply_amplitude(parent_fit["raw_prediction"], production_amplitude),
            weight,
        )
        production_candidate_score = weighted_zero_mean_r2(
            target,
            apply_amplitude(candidate_fit["raw_prediction"], production_amplitude),
            weight,
        )
        production_amplitude_delta = production_candidate_score - production_parent_score
        production_amplitude_deltas.append(production_amplitude_delta)
        if parent.name == BASELINE.name:
            reference_scores[report_fold] = {
                "raw_score": parent_raw_score,
                "calibrated_score": parent_score,
            }
        folds.append(
            {
                "report_fold": report_fold,
                "fit_folds": parent_fit["fit_folds"],
                "rows": int(target.size),
                "row_id_equal": row_id_equal,
                "ridge_coefficients_equal": ridge_equal,
                "member_predictions_equal": member_predictions_equal,
                "member_prediction_equality": member_prediction_equality,
                "raw_predictions_equal": raw_predictions_equal,
                "parent_amplitude_multiplier": parent_amplitude,
                "candidate_amplitude_multiplier": candidate_amplitude,
                "amplitude_policy": amplitude_policy,
                "parent_fit_prediction_std": parent_fit["fit_prediction_std"],
                "parent_score": parent_score,
                "candidate_score": candidate_score,
                "delta": delta,
                "parent_raw_score": parent_raw_score,
                "candidate_raw_score": weighted_zero_mean_r2(
                    target, candidate_fit["raw_prediction"], weight
                ),
                "parent_stack": parent_fit["stack"],
                "candidate_stack": candidate_fit["stack"],
                "parent_prediction": prediction_summary(parent_prediction),
                "candidate_prediction": prediction_summary(candidate_prediction),
                "candidate_members": _member_report(
                    candidate_fit["matrix"],
                    target,
                    weight,
                    candidate.member_names,
                ),
                "parent_members": _member_report(
                    parent_fit["matrix"],
                    target,
                    weight,
                    parent.member_names,
                ),
                "rolling_member_delta": (
                    weighted_zero_mean_r2(
                        target,
                        candidate_fit["matrix"][
                            :, candidate.member_names.index("rolling_v1")
                        ],
                        weight,
                    )
                    - weighted_zero_mean_r2(
                        target,
                        parent_fit["matrix"][:, parent.member_names.index("rolling_v1")],
                        weight,
                    )
                    if "rolling_v1" in parent.member_names
                    and "rolling_v1" in candidate.member_names
                    else None
                ),
                "production_amplitude_control": {
                    "multiplier": production_amplitude,
                    "parent_score": production_parent_score,
                    "candidate_score": production_candidate_score,
                    "delta": production_amplitude_delta,
                },
            }
        )
    gate_mode = GATE_BY_VARIANT[candidate.name]
    if enforce_e0_reference:
        if parent.name != BASELINE.name:
            raise ValueError("E0 reference validation requires e0_baseline00 parent")
        reference_check = validate_e0_reference(reference_scores)
    else:
        reference_check = {
            "skipped": True,
            "reason": "reference enforcement disabled for this API call",
        }
    report: dict[str, Any] = {
        "parent": parent.name,
        "candidate": candidate.name,
        "parent_resolved": asdict(parent),
        "candidate_resolved": asdict(candidate),
        "parent_members_root": str(Path(parent_members_root).resolve()),
        "candidate_members_root": str(Path(candidate_members_root).resolve()),
        "report_folds": list(REPORT_FOLDS),
        "amplitude_policy": AMPLITUDE_POLICY_BY_VARIANT[candidate.name],
        "folds": folds,
        "gate": evaluate_gate(deltas, gate=gate_mode),
        "reference_check": reference_check,
        "comparison_anchor": parent.name,
        "mean_delta_vs_parent": float(np.mean(deltas)),
        "fold8_status": "late_development_seen",
        "production_amplitude_control_audit": {
            "multiplier": float(candidate.amplitude_multiplier),
            "fold_deltas": production_amplitude_deltas,
            "mean_delta": float(np.mean(production_amplitude_deltas)),
            "used_for_gate": False,
            "note": "Sensitivity audit only; the gate uses the shared chronological calibration.",
        },
    }
    if candidate.name == "e1_scale1":
        report["test_set_std_audit"] = {
            "baseline00_scaled_approx": 0.0428,
            "e1_scale1_raw_approx": 0.0394,
            "fold8_e1_raw": folds[-1]["candidate_prediction"]["std"],
            "used_for_gate": False,
            "note": "Public-test distribution shift is audit context only.",
        }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _variant_from_log(payload: dict[str, Any]) -> tuple[ResolvedVariant, Path]:
    latest = str(payload["latest_accepted"])
    entry = payload["variants"][latest]
    resolved = entry["resolved"]
    return (
        ResolvedVariant(
            name=latest,
            member_names=tuple(resolved["member_names"]),
            rolling_operations=tuple(resolved["rolling_operations"]),
            amplitude_multiplier=float(resolved["amplitude_multiplier"]),
        ),
        Path(entry["members_root"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--parent")
    source.add_argument("--promotion-log", type=Path)
    parser.add_argument("--candidate", choices=sorted(VARIANT_PATCHES), required=True)
    parser.add_argument("--parent-members", type=Path)
    parser.add_argument("--candidate-members", type=Path)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.promotion_log:
        payload = json.loads(args.promotion_log.read_text(encoding="utf-8"))
        parent, parent_members = _variant_from_log(payload)
        if args.parent_members is not None:
            parser.error("--parent-members cannot be used with --promotion-log")
    else:
        if args.parent != BASELINE.name:
            parser.error(f"explicit parent must be {BASELINE.name}")
        if args.parent_members is None:
            parser.error("--parent requires --parent-members")
        parent = BASELINE
        parent_members = args.parent_members
    candidate = resolve_variant(parent, VARIANT_PATCHES[args.candidate])
    candidate_members = args.candidate_members or parent_members
    report = evaluate_variant(
        parent,
        candidate,
        parent_members,
        candidate_members,
        args.out,
        train_dir=args.train_dir,
        enforce_e0_reference=parent.name == BASELINE.name,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
