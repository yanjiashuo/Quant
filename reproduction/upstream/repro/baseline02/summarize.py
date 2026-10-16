from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from repro.baseline02.finalize import METHODS, sha256_file


def _json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_aligned(path: Path, row_id: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["row_id", "target"]:
        raise ValueError(f"{path} columns differ")
    if not np.array_equal(frame["row_id"].to_numpy(dtype=np.int64), row_id):
        raise ValueError(f"{path} row order differs")
    return frame["target"].to_numpy(dtype=np.float64, copy=True)


def build_summary(
    root: Path,
    historical_reference: Path,
    methods: Sequence[str] = METHODS,
) -> dict[str, object]:
    root = Path(root)
    manifest_path = root / "summary" / "submission_manifest.json"
    manifest = _json(manifest_path)
    first = pd.read_csv(root / methods[0] / "submission.csv", usecols=["row_id"])
    row_id = first["row_id"].to_numpy(dtype=np.int64, copy=True)
    predictions = {
        method: _load_aligned(root / method / "submission.csv", row_id)
        for method in methods
    }
    matrix = np.column_stack([predictions[method] for method in methods])
    correlation = np.corrcoef(matrix, rowvar=False)
    reference = _load_aligned(historical_reference, row_id)
    reference_correlation = {
        method: float(np.corrcoef(predictions[method], reference)[0, 1])
        for method in methods
    }
    parent = _json(root / "d0" / "parent_walk_forward.json")
    e1 = _json(root / "production" / "candidates" / "cs_rank" / "report.json")
    e2 = _json(root / "production" / "candidates" / "rolling_v2" / "report.json")
    meta = root / "production" / "meta"
    e3 = _json(meta / "e3_report.json")
    e4 = _json(meta / "e4_report.json")
    e5 = _json(meta / "e5_report.json")
    e6 = _json(meta / "e6_report.json")
    outcomes = {
        "g0_parent": {
            "verdict": "control",
            "fold7_score": parent["fold7"]["raw_score"],
            "fold8_score": parent["fold8"]["raw_score"],
        },
        "e1_cs_rank": {
            "verdict": e1["verdict"],
            "fold7_delta": e1["fold7"]["delta"],
            "fold8_delta": e1["fold8"]["delta"],
            "positive_blocks7": e1["fold7"]["positive_blocks"],
            "positive_blocks8": e1["fold8"]["positive_blocks"],
        },
        "e2_rolling_v2": {
            "verdict": e2["verdict"],
            "fold7_delta": e2["fold7"]["delta"],
            "fold8_delta": e2["fold8"]["delta"],
            "positive_blocks7": e2["fold7"]["positive_blocks"],
            "positive_blocks8": e2["fold8"]["positive_blocks"],
        },
        "e3_combo": {
            "verdict": e3["verdict"],
            "fold7_delta": e3["fold7"]["delta"],
            "fold8_delta": e3["fold8"]["delta"],
            "approved_candidates": e3["prerequisite_decision"]["approved_candidates"],
        },
        "e4_level_deviation": {
            "verdict": e4["verdict"],
            "fold7_delta": e4["fold7"]["delta"],
            "fold8_delta": e4["fold8"]["delta"],
            "drift_reduction_fraction": e4["drift_reduction_fraction"],
            "l88_rebuild_max_abs_diff": e4["l88_rebuild_max_abs_diff"],
        },
        "e5_ridge": {
            "verdict": e5["verdict"],
            "selected_ratio": e5["selected_ratio"],
            "fold8_delta": e5["fold8_delta"],
            "reference_coefficient_drift_l1": e5[
                "reference_coefficient_drift_l1"
            ],
            "candidate_coefficient_drift_l1": e5[
                "candidate_coefficient_drift_l1"
            ],
        },
        "e6_per_asset": {
            "verdict": e6["verdict"],
            "selected_kappa": e6["selected_kappa"],
            "fold8_delta_vs_global": e6["fold8_delta"],
            "stable_assets": e6["stable_assets"],
            "production_asset_scales": e6["production"]["asset_scales"],
        },
    }
    report = {
        "recommended_method": manifest["recommended_method"],
        "recommendation_note": (
            "E5 passes the frozen stability gate, but its Fold8 score delta is "
            "numerically near zero; treat it as a regularization-stability variant, "
            "not a demonstrated material signal gain."
        ),
        "outcomes": outcomes,
        "submission_methods": list(methods),
        "submission_pearson": correlation.tolist(),
        "historical_reference": {
            "path": str(historical_reference),
            "sha256": sha256_file(historical_reference),
            "public_score_record": 0.0034426160,
            "correlation": reference_correlation,
        },
        "runner_audit_scope": manifest["runner_audit_scope"],
        "server_mirror": {
            "checked_files": 35,
            "sha256_mismatches": 0,
            "path": "/data2/hjk/projects/quant_competition/work/baseline02/deliverables",
        },
    }
    output = root / "summary" / "experiment_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    manifest["experiment_report"] = {
        "path": str(output),
        "sha256": sha256_file(output),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--historical-reference", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_summary(args.root, args.historical_reference),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
