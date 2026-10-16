from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from repro.baseline00.metrics import weighted_zero_mean_r2
from repro.baseline02.artifacts import OOFFold, load_oof
from repro.baseline02.calibration import (
    apply_asset_scales,
    fit_global_scale,
    fit_shrunk_asset_scales,
)
from repro.baseline02.config import PARENT_RIDGE_ALPHA, oof_paths
from repro.baseline02.reparameterize import (
    evaluate_reparameterization,
    reparameterize_fold,
)
from repro.baseline02.stack import (
    StackFit,
    augment_fold,
    evaluate_candidate_set,
    fit_stack,
    fit_stack_ratio,
    select_lambda_ratio,
)


RIDGE_RATIO_GRID = (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
KAPPA_GRID = (0.25, 1.0, 4.0, 16.0)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_report(path: Path) -> dict[str, object]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("fold8") is None or report.get("production") is None:
        raise ValueError(f"candidate report is incomplete: {path}")
    return report


def _augment_selected(
    parent_folds: Mapping[int, OOFFold],
    selected: Sequence[str],
    candidate_dirs: Mapping[str, Path],
) -> dict[int, OOFFold]:
    result: dict[int, OOFFold] = {}
    for fold in (6, 7, 8):
        current = parent_folds[fold]
        for name in selected:
            current = augment_fold(
                current, Path(candidate_dirs[name]) / f"fold_{fold}.npz", name
            )
        result[fold] = current
    return result


def _fit_fixed(folds: Sequence[OOFFold], alpha: float = PARENT_RIDGE_ALPHA) -> StackFit:
    return fit_stack(
        np.concatenate([fold.matrix for fold in folds]),
        np.concatenate([fold.target for fold in folds]),
        np.concatenate([fold.weight for fold in folds]),
        alpha=alpha,
        member_names=folds[0].member_names,
    )


def _fit_ratio(folds: Sequence[OOFFold], ratio: float) -> StackFit:
    return fit_stack_ratio(
        np.concatenate([fold.matrix for fold in folds]),
        np.concatenate([fold.target for fold in folds]),
        np.concatenate([fold.weight for fold in folds]),
        ratio,
        folds[0].member_names,
    )


def _drift(left: StackFit, right: StackFit) -> float:
    return float(np.sum(np.abs(left.normalized_coefficients - right.normalized_coefficients)))


def _slice_fold(fold: OOFFold, selected: np.ndarray, suffix: str) -> OOFFold:
    return OOFFold(
        matrix=fold.matrix[selected],
        target=fold.target[selected],
        weight=fold.weight[selected],
        row_id=fold.row_id[selected],
        time_id=fold.time_id[selected],
        asset_id=fold.asset_id[selected],
        member_names=fold.member_names,
        source=f"{fold.source}:{suffix}",
    )


def _choose_approved_candidates(
    e1: Mapping[str, object],
    e2: Mapping[str, object],
    combo: Mapping[str, object],
) -> tuple[list[str], dict[str, object]]:
    reports = {str(e1["candidate_name"]): e1, str(e2["candidate_name"]): e2}
    passed = [
        name
        for name, report in reports.items()
        if report["gate"]["confirmation_pass"] is True
    ]
    decision: dict[str, object] = {"individually_passed": passed}
    if len(passed) < 2:
        decision.update(
            combo_eligible=False,
            reason="rejected_parent",
            approved_candidates=passed,
        )
        return passed, decision
    best_single = max(
        passed,
        key=lambda name: (
            float(reports[name]["fold7"]["delta"])
            + float(reports[name]["fold8"]["delta"])
        )
        / 2.0,
    )
    delta7_vs_best = float(combo["fold7"]["child_score"]) - float(
        reports[best_single]["fold7"]["child_score"]
    )
    delta8_vs_best = float(combo["fold8"]["child_score"]) - float(
        reports[best_single]["fold8"]["child_score"]
    )
    coefficients7 = combo["fold7"]["new_member_effective_coefficients"]
    coefficients8 = combo["fold8"]["new_member_effective_coefficients"]
    coefficients_positive = all(
        float(coefficients7[name]) > 0.0 and float(coefficients8[name]) > 0.0
        for name in passed
    )
    combo_passed = (
        delta7_vs_best >= 0.000010
        and delta8_vs_best >= 0.0
        and coefficients_positive
    )
    approved = passed if combo_passed else [best_single]
    decision.update(
        combo_eligible=True,
        best_single=best_single,
        combo_delta7_vs_best_single=delta7_vs_best,
        combo_delta8_vs_best_single=delta8_vs_best,
        combo_new_coefficients_positive=coefficients_positive,
        combo_passed=combo_passed,
        approved_candidates=approved,
        reason="combo_promoted" if combo_passed else "best_single_only",
    )
    return approved, decision


def _run_ridge_experiment(folds: Mapping[int, OOFFold]) -> dict[str, object]:
    development = select_lambda_ratio(folds[6], folds[7], RIDGE_RATIO_GRID)
    ratio = float(development["selected_ratio"])
    reference_r6 = _fit_fixed((folds[6],))
    reference_r67 = _fit_fixed((folds[6], folds[7]))
    candidate_r6 = _fit_ratio((folds[6],), ratio)
    candidate_r67 = _fit_ratio((folds[6], folds[7]), ratio)
    candidate_r678 = _fit_ratio((folds[6], folds[7], folds[8]), ratio)
    reference_score = weighted_zero_mean_r2(
        folds[8].target, reference_r67.predict(folds[8].matrix), folds[8].weight
    )
    candidate_score = weighted_zero_mean_r2(
        folds[8].target, candidate_r67.predict(folds[8].matrix), folds[8].weight
    )
    reference_drift = _drift(reference_r6, reference_r67)
    candidate_drift = _drift(candidate_r6, candidate_r67)
    passed = (
        candidate_score >= reference_score - 0.000010
        and candidate_drift <= reference_drift + 1e-15
    )
    accepted_r678 = candidate_r678 if passed else _fit_fixed(
        (folds[6], folds[7], folds[8])
    )
    return {
        "grid": list(RIDGE_RATIO_GRID),
        "development": development,
        "selected_ratio": ratio,
        "fold8_reference_score": float(reference_score),
        "fold8_candidate_score": float(candidate_score),
        "fold8_delta": float(candidate_score - reference_score),
        "reference_coefficient_drift_l1": reference_drift,
        "candidate_coefficient_drift_l1": candidate_drift,
        "verdict": "promoted" if passed else "rejected",
        "production": {"fit_folds": [6, 7, 8], "fit": candidate_r678.to_dict()},
        "accepted_direction": {
            "kind": "lambda_ratio" if passed else "fixed_alpha",
            "lambda_ratio": ratio if passed else None,
            "alpha": None if passed else PARENT_RIDGE_ALPHA,
            "fit": accepted_r678.to_dict(),
        },
    }


def _direction_fitter(kind: str, ratio: float | None):
    if kind == "lambda_ratio":
        if ratio is None:
            raise ValueError("accepted lambda-ratio direction lacks ratio")
        return lambda folds: _fit_ratio(folds, ratio)
    if kind == "fixed_alpha":
        return lambda folds: _fit_fixed(folds)
    raise ValueError(f"unknown direction kind: {kind}")


def _asset_score_deltas(
    fold: OOFFold, global_prediction: np.ndarray, asset_prediction: np.ndarray
) -> list[dict[str, float | int]]:
    result: list[dict[str, float | int]] = []
    for asset in np.unique(fold.asset_id):
        selected = fold.asset_id == asset
        global_score = weighted_zero_mean_r2(
            fold.target[selected], global_prediction[selected], fold.weight[selected]
        )
        asset_score = weighted_zero_mean_r2(
            fold.target[selected], asset_prediction[selected], fold.weight[selected]
        )
        result.append(
            {
                "asset_id": int(asset),
                "global_score": float(global_score),
                "asset_score": float(asset_score),
                "delta": float(asset_score - global_score),
            }
        )
    return result


def _run_calibration_experiment(
    folds: Mapping[int, OOFFold], ridge_report: Mapping[str, object]
) -> dict[str, object]:
    accepted = ridge_report["accepted_direction"]
    kind = str(accepted["kind"])
    ratio = (
        None if accepted.get("lambda_ratio") is None else float(accepted["lambda_ratio"])
    )
    fit_direction = _direction_fitter(kind, ratio)
    unique_time = np.unique(folds[6].time_id)
    split_time = unique_time[max(1, int(len(unique_time) * 0.70)) - 1]
    early_mask = folds[6].time_id <= split_time
    late_mask = ~early_mask
    early = _slice_fold(folds[6], early_mask, "calibration_early70")
    late = _slice_fold(folds[6], late_mask, "calibration_late30")
    early_fit = fit_direction((early,))
    late_prediction = early_fit.predict(late.matrix)
    global_late = fit_global_scale(late.target, late_prediction, late.weight)
    r6 = fit_direction((folds[6],))
    fold7_prediction = r6.predict(folds[7].matrix)
    kappa_candidates: list[dict[str, object]] = []
    for kappa in KAPPA_GRID:
        scales = fit_shrunk_asset_scales(
            late.target,
            late_prediction,
            late.weight,
            late.asset_id,
            global_late,
            kappa,
        )
        prediction = apply_asset_scales(fold7_prediction, folds[7].asset_id, scales)
        kappa_candidates.append(
            {
                "kappa": float(kappa),
                "fold7_score": weighted_zero_mean_r2(
                    folds[7].target, prediction, folds[7].weight
                ),
                "calibration_scales": scales.tolist(),
            }
        )
    selected = max(kappa_candidates, key=lambda item: float(item["fold7_score"]))
    kappa = float(selected["kappa"])

    global7 = fit_global_scale(folds[7].target, fold7_prediction, folds[7].weight)
    scales7 = fit_shrunk_asset_scales(
        folds[7].target,
        fold7_prediction,
        folds[7].weight,
        folds[7].asset_id,
        global7,
        kappa,
    )
    r67 = fit_direction((folds[6], folds[7]))
    fold8_prediction = r67.predict(folds[8].matrix)
    fold8_global = fold8_prediction * global7
    fold8_asset = apply_asset_scales(fold8_prediction, folds[8].asset_id, scales7)
    global_score8 = weighted_zero_mean_r2(
        folds[8].target, fold8_global, folds[8].weight
    )
    asset_score8 = weighted_zero_mean_r2(
        folds[8].target, fold8_asset, folds[8].weight
    )
    asset_deltas = _asset_score_deltas(folds[8], fold8_global, fold8_asset)
    stable_assets = sum(float(item["delta"]) >= -0.000020 for item in asset_deltas)
    passed = (
        asset_score8 >= global_score8
        and np.isfinite(scales7).all()
        and np.all((scales7 >= 0.5) & (scales7 <= 1.5))
        and stable_assets >= 10
    )

    global8 = fit_global_scale(folds[8].target, fold8_prediction, folds[8].weight)
    production_scales = fit_shrunk_asset_scales(
        folds[8].target,
        fold8_prediction,
        folds[8].weight,
        folds[8].asset_id,
        global8,
        kappa,
    )
    r678 = fit_direction((folds[6], folds[7], folds[8]))
    return {
        "direction_kind": kind,
        "direction_lambda_ratio": ratio,
        "kappa_grid": list(KAPPA_GRID),
        "kappa_development": kappa_candidates,
        "selected_kappa": kappa,
        "fold7_global_scale": global7,
        "fold7_asset_scales": scales7.tolist(),
        "fold8_global_score": float(global_score8),
        "fold8_per_asset_score": float(asset_score8),
        "fold8_delta": float(asset_score8 - global_score8),
        "fold8_asset_deltas": asset_deltas,
        "stable_assets": int(stable_assets),
        "verdict": "promoted" if passed else "rejected",
        "production": {
            "fit_folds": [6, 7, 8],
            "fit": r678.to_dict(),
            "scale_source_fold": 8,
            "global_scale_fold8": global8,
            "asset_scales": production_scales.tolist(),
        },
    }


def run_meta_experiments(
    oof_dir: Path,
    e1_dir: Path,
    e2_dir: Path,
    output_root: Path,
) -> dict[str, object]:
    parent = {fold: load_oof(path) for fold, path in oof_paths(oof_dir).items()}
    e1_report = _load_report(Path(e1_dir) / "report.json")
    e2_report = _load_report(Path(e2_dir) / "report.json")
    candidate_dirs = {
        str(e1_report["candidate_name"]): Path(e1_dir),
        str(e2_report["candidate_name"]): Path(e2_dir),
    }
    combo = evaluate_candidate_set(parent, candidate_dirs)
    approved, decision = _choose_approved_candidates(e1_report, e2_report, combo)
    combo["prerequisite_decision"] = decision
    combo["verdict"] = (
        "promoted" if decision.get("reason") == "combo_promoted" else "audit_only/rejected_parent"
    )
    _write_json(Path(output_root) / "e3_combo" / "report.json", combo)
    approved_report = {
        "parent_id": "P0_OOF_ROLLING4",
        "parent_members": list(parent[6].member_names),
        "approved_candidates": approved,
        "approved_members": [*parent[6].member_names, *approved],
        "candidate_dirs": {name: str(candidate_dirs[name]) for name in approved},
        "decision": decision,
    }
    _write_json(
        Path(output_root) / "e3_combo" / "approved_members.json", approved_report
    )

    selected_folds = _augment_selected(parent, approved, candidate_dirs)
    reparameterization = evaluate_reparameterization(selected_folds)
    reparameterization["input_members"] = list(selected_folds[6].member_names)
    _write_json(
        Path(output_root) / "e4_level_deviation" / "report.json",
        reparameterization,
    )
    ridge_input = (
        {fold: reparameterize_fold(selected_folds[fold]) for fold in (6, 7, 8)}
        if reparameterization["verdict"] == "promoted"
        else selected_folds
    )
    ridge = _run_ridge_experiment(ridge_input)
    ridge["member_names"] = list(ridge_input[6].member_names)
    ridge["basis_source"] = (
        "e4_level_deviation"
        if reparameterization["verdict"] == "promoted"
        else "e3_approved_members"
    )
    _write_json(Path(output_root) / "e5_ridge" / "report.json", ridge)

    calibration = _run_calibration_experiment(ridge_input, ridge)
    calibration["member_names"] = list(ridge_input[6].member_names)
    _write_json(Path(output_root) / "e6_per_asset" / "report.json", calibration)
    summary = {
        "approved": approved_report,
        "e3_verdict": combo["verdict"],
        "e4_verdict": reparameterization["verdict"],
        "e5_verdict": ridge["verdict"],
        "e6_verdict": calibration["verdict"],
    }
    _write_json(Path(output_root) / "summary" / "meta_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--e1-dir", type=Path, required=True)
    parser.add_argument("--e2-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_meta_experiments(args.oof_dir, args.e1_dir, args.e2_dir, args.out),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
