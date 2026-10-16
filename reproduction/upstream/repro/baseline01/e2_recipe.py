from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from repro.baseline00.rolling_four_diagnostic import four_column_indices
from repro.baseline01.config import BASELINE, FOUR_OPERATIONS
from repro.config import ROLLING_TOP50, ensure_data2_path


def freeze_e2_recipe(
    report_path: Path,
    parent_recipe_path: Path,
    output: Path,
) -> dict[str, Any]:
    report_path = Path(report_path).resolve()
    parent_recipe_path = Path(parent_recipe_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    parent = json.loads(parent_recipe_path.read_text(encoding="utf-8"))
    if report.get("parent") != BASELINE.name or report.get("candidate") != "e2_rolling4":
        raise ValueError("E2 recipe requires an E0 to E2 report")
    if report.get("reference_check", {}).get("passed") is not True:
        raise ValueError("E2 recipe requires an exact E0 reproduction")
    if report.get("amplitude_policy") != "shared_parent_legacy":
        raise ValueError("E2 report did not use the shared-amplitude control")
    folds = report.get("folds", [])
    if len(folds) != 3:
        raise ValueError("E2 report must contain Fold6-8")
    for fold in folds:
        if fold.get("row_id_equal") is not True:
            raise ValueError("E2 report has a row_id mismatch")
        if not np.isclose(
            fold["parent_amplitude_multiplier"],
            fold["candidate_amplitude_multiplier"],
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError("E2 parent and candidate amplitudes differ")
        equality = fold.get("member_prediction_equality", {})
        if any(equality.get(name) is not True for name in ("final", "L88", "model_c", "level")):
            raise ValueError("E2 changed a non-Rolling member")
        if equality.get("rolling_v1") is not False:
            raise ValueError("E2 did not change the Rolling member")

    fold8 = next(fold for fold in folds if int(fold["report_fold"]) == 8)
    stack = fold8["candidate_stack"]
    if tuple(stack["member_names"]) != BASELINE.member_names:
        raise ValueError("E2 Fold8 stack member order differs from Baseline00")
    if tuple(parent["member_names"]) != BASELINE.member_names:
        raise ValueError("parent recipe member order differs from Baseline00")

    recipe = dict(parent)
    recipe.update(
        {
            "variant": "e2_rolling4",
            "parent_variant": BASELINE.name,
            "comparison_mode": "isolated_vs_e0_baseline00",
            "rolling_operations": list(FOUR_OPERATIONS),
            "rolling_column_indices": four_column_indices(len(ROLLING_TOP50)).tolist(),
            "effective_coefficients": stack["effective_coefficients"],
            "normalized_coefficients": stack["normalized_coefficients"],
            "standard_deviations": stack["standard_deviations"],
            "fit_folds": fold8["fit_folds"],
            "amplitude_multiplier": float(parent["amplitude_multiplier"]),
            "source_report": str(report_path),
            "local_gate": report["gate"],
            "accepted_by_local_gate": bool(report["gate"]["passed"]),
        }
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(recipe, indent=2, sort_keys=True), encoding="utf-8")
    return recipe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--parent-recipe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = ensure_data2_path(args.out)
    recipe = freeze_e2_recipe(args.report, args.parent_recipe, output)
    print(json.dumps(recipe, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
