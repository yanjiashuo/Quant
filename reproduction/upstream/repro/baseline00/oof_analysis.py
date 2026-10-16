from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from repro.baseline00 import MEMBER_NAMES
from repro.baseline00.metrics import (
    fit_positive_stack,
    predict_positive_stack,
    weighted_zero_mean_r2,
)
from repro.config import RIDGE_ALPHA, TARGET_STD, WORK, ensure_data2_path


VARIANTS: dict[str, tuple[str, ...]] = {
    "all_five": MEMBER_NAMES,
    "without_model_c": ("final", "L88", "level", "rolling_v1"),
    "without_level": ("final", "L88", "model_c", "rolling_v1"),
    "without_rolling": ("final", "L88", "model_c", "level"),
    "final_l88_rolling": ("final", "L88", "rolling_v1"),
    "final_l88": ("final", "L88"),
}


def load_members(path: Path, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    matrix = np.column_stack([data[name] for name in names])
    return matrix, data["y"].astype(np.float64), data["w"].astype(np.float64)


def evaluate_forward_variant(
    members_root: Path, names: Sequence[str]
) -> dict[str, Any]:
    fold_reports: list[dict[str, Any]] = []
    for fit_folds, report_fold in [((5,), 6), ((5, 6), 7)]:
        fit_parts = [load_members(members_root / f"fold_{fold}.npz", names) for fold in fit_folds]
        fit_matrix = np.concatenate([part[0] for part in fit_parts])
        fit_y = np.concatenate([part[1] for part in fit_parts])
        fit_w = np.concatenate([part[2] for part in fit_parts])
        stack = fit_positive_stack(
            fit_matrix, fit_y, fit_w, alpha=RIDGE_ALPHA, member_names=names
        )
        fit_prediction = predict_positive_stack(fit_matrix, stack)
        fit_std = float(np.std(fit_prediction))
        multiplier = TARGET_STD / fit_std if fit_std > 1e-12 else 1.0
        report_matrix, report_y, report_w = load_members(
            members_root / f"fold_{report_fold}.npz", names
        )
        raw = predict_positive_stack(report_matrix, stack)
        calibrated = raw * multiplier
        fold_reports.append(
            {
                "fit_folds": list(fit_folds),
                "report_fold": report_fold,
                "member_names": list(names),
                "raw_score": weighted_zero_mean_r2(report_y, raw, report_w),
                "calibrated_score": weighted_zero_mean_r2(
                    report_y, calibrated, report_w
                ),
                "raw_prediction_std": float(np.std(raw)),
                "calibrated_prediction_std": float(np.std(calibrated)),
                "amplitude_multiplier": multiplier,
                "effective_coefficients": stack["effective_coefficients"],
                "nonzero_members": stack["nonzero_members"],
            }
        )
    raw_scores = np.asarray([report["raw_score"] for report in fold_reports])
    calibrated_scores = np.asarray(
        [report["calibrated_score"] for report in fold_reports]
    )
    return {
        "member_names": list(names),
        "folds": fold_reports,
        "raw_mean": float(raw_scores.mean()),
        "raw_min": float(raw_scores.min()),
        "raw_std": float(raw_scores.std()),
        "calibrated_mean": float(calibrated_scores.mean()),
        "calibrated_min": float(calibrated_scores.min()),
        "calibrated_std": float(calibrated_scores.std()),
        "calibration_mean_delta": float((calibrated_scores - raw_scores).mean()),
    }


def build_report(
    members_root: Path,
    heldout_report: Path | None = None,
    rolling_four_summary: Path | None = None,
    stop_threshold: float = 0.00030,
) -> dict[str, Any]:
    members_root = ensure_data2_path(members_root)
    variants = {
        name: evaluate_forward_variant(members_root, member_names)
        for name, member_names in VARIANTS.items()
    }
    report: dict[str, Any] = {
        "analysis_scope": "Fold 5-7 only; no Fold 8 labels read",
        "variants": variants,
        "development_ranking_by_raw_min": sorted(
            VARIANTS, key=lambda name: variants[name]["raw_min"], reverse=True
        ),
    }
    if heldout_report is not None:
        heldout = json.loads(Path(heldout_report).read_text(encoding="utf-8"))
        all_five = variants["all_five"]
        dev_mean = all_five["calibrated_mean"]
        heldout_score = float(heldout["report_calibrated_score"])
        drop = dev_mean - heldout_score
        report["heldout_gate_summary_from_existing_json"] = {
            "labels_reloaded": False,
            "development_calibrated_mean": dev_mean,
            "heldout_calibrated_score": heldout_score,
            "drop": drop,
            "threshold": stop_threshold,
            "passed": bool(heldout_score > 0 and drop <= stop_threshold),
        }
    if rolling_four_summary is not None:
        rolling_diagnostic = json.loads(
            Path(rolling_four_summary).read_text(encoding="utf-8")
        )
        if rolling_diagnostic.get("fold8_read") is not False:
            raise ValueError("rolling four-transform diagnostic must not read Fold 8")
        report["rolling_four_transform_diagnostic"] = rolling_diagnostic
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", type=Path, default=WORK / "members")
    parser.add_argument("--heldout-report", type=Path)
    parser.add_argument("--rolling-four-summary", type=Path)
    parser.add_argument("--out", type=Path, default=WORK / "oof_report.json")
    args = parser.parse_args()
    output = ensure_data2_path(args.out)
    report = build_report(
        args.members, args.heldout_report, args.rolling_four_summary
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
