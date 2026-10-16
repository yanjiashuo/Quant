from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from repro.baseline01.config import BASELINE, SIX_OPERATIONS
from repro.config import ROLLING_TOP50, ensure_data2_path


E3_MEMBERS = ("final", "L88", "level", "rolling_v1")
REQUIRED_BASE_MODELS = ("model_a", "model_b", "model_c")


def freeze_e3_recipe(
    report_path: Path,
    parent_recipe_path: Path,
    output: Path,
) -> dict[str, Any]:
    report_path = Path(report_path).resolve()
    parent_recipe_path = Path(parent_recipe_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    parent = json.loads(parent_recipe_path.read_text(encoding="utf-8"))
    if report.get("parent") != BASELINE.name or report.get("candidate") != "e3_drop_model_c":
        raise ValueError("E3 recipe requires an E0 to E3 report")
    if report.get("reference_check", {}).get("passed") is not True:
        raise ValueError("E3 recipe requires an exact E0 reproduction")
    if report.get("amplitude_policy") != "shared_parent_legacy":
        raise ValueError("E3 report did not use the shared-amplitude control")
    if Path(report["parent_members_root"]).resolve() != Path(
        report["candidate_members_root"]
    ).resolve():
        raise ValueError("E3 must reuse the Baseline00 member archives")
    folds = report.get("folds", [])
    if len(folds) != 3:
        raise ValueError("E3 report must contain Fold6-8")
    for fold in folds:
        if fold.get("row_id_equal") is not True:
            raise ValueError("E3 report has a row_id mismatch")
        if not np.isclose(
            fold["parent_amplitude_multiplier"],
            fold["candidate_amplitude_multiplier"],
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError("E3 parent and candidate amplitudes differ")
        equality = fold.get("member_prediction_equality", {})
        if any(equality.get(name) is not True for name in E3_MEMBERS):
            raise ValueError("E3 changed a retained member")
        if equality.get("model_c") is not False:
            raise ValueError("E3 did not remove standalone model_c")

    fold8 = next(fold for fold in folds if int(fold["report_fold"]) == 8)
    stack = fold8["candidate_stack"]
    if tuple(stack["member_names"]) != E3_MEMBERS:
        raise ValueError("E3 Fold8 stack member order is invalid")
    if tuple(parent["member_names"]) != BASELINE.member_names:
        raise ValueError("parent recipe member order differs from Baseline00")

    recipe = dict(parent)
    recipe.update(
        {
            "variant": "e3_drop_model_c",
            "parent_variant": BASELINE.name,
            "comparison_mode": "isolated_vs_e0_baseline00",
            "member_names": list(E3_MEMBERS),
            "required_base_models": list(REQUIRED_BASE_MODELS),
            "rolling_operations": list(SIX_OPERATIONS),
            "rolling_column_indices": list(range(len(ROLLING_TOP50) * len(SIX_OPERATIONS))),
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
    recipe = freeze_e3_recipe(args.report, args.parent_recipe, output)
    print(json.dumps(recipe, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
