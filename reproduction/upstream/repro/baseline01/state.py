from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from repro.baseline01.config import BASELINE


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_rows(path: Path) -> int:
    lines = 0
    last = b""
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            lines += chunk.count(b"\n")
            last = chunk[-1:]
    if last and last != b"\n":
        lines += 1
    return max(lines - 1, 0)


def _artifact_metadata(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".csv":
        metadata["rows"] = _csv_rows(path)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "row_id" not in archive.files:
                raise ValueError(f"member archive lacks row_id: {path}")
            metadata["rows"] = int(np.asarray(archive["row_id"]).size)
            metadata["member_columns"] = [
                name for name in BASELINE.member_names if name in archive.files
            ]
    return metadata


def create_reference_manifest(
    qroot: Path,
    output: Path,
    *,
    git_commit: str,
) -> dict[str, Any]:
    qroot = Path(qroot).resolve()
    output = Path(output).resolve()
    try:
        output.relative_to(Path("/data2").resolve())
    except ValueError as error:
        raise ValueError("reference manifest output must be under /data2") from error
    baseline = qroot / "work" / "baseline00"
    paths = [
        baseline / "frozen_recipe.json",
        *(baseline / "members" / f"fold_{fold}.npz" for fold in (5, 6, 7, 8)),
        baseline / "production" / "api_model" / "production_config.json",
        baseline / "submission_baseline00.csv",
        qroot / "repo" / "artifacts" / "submissions" / "rolling_stack_std46.csv",
    ]
    artifacts = [_artifact_metadata(path) for path in paths]
    payload = {
        "git_commit": git_commit,
        "semantic_member_order": list(BASELINE.member_names),
        "artifacts": artifacts,
    }
    _atomic_write_json(output, payload)
    return payload


def _resolved_payload() -> dict[str, Any]:
    return {
        "member_names": list(BASELINE.member_names),
        "rolling_operations": list(BASELINE.rolling_operations),
        "amplitude_multiplier": BASELINE.amplitude_multiplier,
    }


def initialize_promotion_log(
    output: Path,
    *,
    members_root: Path,
    recipe: Path,
) -> dict[str, Any]:
    output = Path(output)
    e0 = {
        "passed": True,
        "parent": None,
        "members_root": str(Path(members_root).resolve()),
        "recipe": str(Path(recipe).resolve()),
        "resolved": _resolved_payload(),
    }
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("variants", {}).get(BASELINE.name) != e0:
            raise ValueError("existing promotion log has a different E0 reference")
        return existing
    payload = {
        "latest_accepted": BASELINE.name,
        "ordered_attempts": [BASELINE.name],
        "variants": {BASELINE.name: e0},
    }
    _atomic_write_json(output, payload)
    return payload


def _e0_reference(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": BASELINE.name,
        "reference_check": report["reference_check"],
        "fold8_status": report["fold8_status"],
        "folds": [
            {
                "report_fold": fold["report_fold"],
                "fit_folds": fold["fit_folds"],
                "raw_score": fold["parent_raw_score"],
                "calibrated_score": fold["parent_score"],
                "amplitude_multiplier": fold["parent_amplitude_multiplier"],
                "prediction": fold["parent_prediction"],
            }
            for fold in report["folds"]
        ],
    }


def finalize_e1(
    report_path: Path,
    *,
    parent_recipe: Path,
    promotion_log: Path,
    recipe_output: Path,
    e0_output: Path,
) -> dict[str, Any]:
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("parent") != BASELINE.name or report.get("candidate") != "e1_scale1":
        raise ValueError("finalize_e1 requires an E0 to E1 report")
    if report.get("reference_check", {}).get("passed") is not True:
        raise ValueError("E0 reference was not reproduced")
    folds = report.get("folds", [])
    if len(folds) != 3:
        raise ValueError("E1 report must contain exactly three report folds")
    invariants = (
        "ridge_coefficients_equal",
        "row_id_equal",
        "member_predictions_equal",
        "raw_predictions_equal",
    )
    if any(fold.get(name) is not True for fold in folds for name in invariants):
        raise ValueError("E1 changed data, members, raw predictions, or Ridge coefficients")
    gate_passed = report.get("gate", {}).get("mode") == "correctness" and report.get(
        "gate", {}
    ).get("passed") is True
    if gate_passed != all(float(fold["delta"]) >= 0.0 for fold in folds):
        raise ValueError("E1 gate decision does not match the never-harm rule")

    _atomic_write_json(Path(e0_output), _e0_reference(report))
    parent_recipe = Path(parent_recipe).resolve()
    parent = json.loads(parent_recipe.read_text(encoding="utf-8"))
    recipe_path: str | None = None
    if gate_passed:
        fold8 = next(fold for fold in folds if int(fold["report_fold"]) == 8)
        if parent.get("effective_coefficients") != fold8["candidate_stack"].get(
            "effective_coefficients"
        ):
            raise ValueError("E1 Fold8 Ridge coefficients differ from the frozen parent")
        recipe = dict(parent)
        recipe.update(
            {
                "variant": "e1_scale1",
                "parent_variant": BASELINE.name,
                "rolling_operations": list(BASELINE.rolling_operations),
                "amplitude_multiplier": 1.0,
                "source_report": str(report_path),
                "fold8_status": "late_development_seen",
            }
        )
        _atomic_write_json(Path(recipe_output), recipe)
        recipe_path = str(Path(recipe_output).resolve())

    promotion_path = Path(promotion_log)
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    if promotion.get("latest_accepted") not in {BASELINE.name, "e1_scale1"}:
        raise ValueError("promotion log is not at the E1 stage")
    if "e1_scale1" not in promotion["ordered_attempts"]:
        promotion["ordered_attempts"].append("e1_scale1")
    resolved = dict(report["candidate_resolved"])
    resolved.pop("name", None)
    promotion["variants"]["e1_scale1"] = {
        "passed": gate_passed,
        "parent": BASELINE.name,
        "members_root": report["candidate_members_root"],
        "recipe": recipe_path,
        "report": str(report_path),
        "resolved": resolved,
        "gate": report["gate"],
    }
    promotion["latest_accepted"] = "e1_scale1" if gate_passed else BASELINE.name
    _atomic_write_json(promotion_path, promotion)
    return {
        "accepted": gate_passed,
        "recipe": recipe_path,
        "latest_accepted": promotion["latest_accepted"],
    }


def record_isolated_e2(
    report_path: Path,
    recipe_path: Path,
    promotion_log: Path,
) -> dict[str, Any]:
    report_path = Path(report_path).resolve()
    recipe_path = Path(recipe_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if report.get("parent") != BASELINE.name or report.get("candidate") != "e2_rolling4":
        raise ValueError("record_isolated_e2 requires an E0 to E2 report")
    if report.get("reference_check", {}).get("passed") is not True:
        raise ValueError("E2 did not reproduce the E0 reference")
    if recipe.get("variant") != "e2_rolling4" or recipe.get("parent_variant") != BASELINE.name:
        raise ValueError("E2 recipe has the wrong parent or variant")
    if bool(recipe.get("accepted_by_local_gate")) != bool(report["gate"]["passed"]):
        raise ValueError("E2 recipe and report gate decisions differ")

    promotion_path = Path(promotion_log)
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    latest_before = promotion["latest_accepted"]
    if "e2_rolling4" not in promotion["ordered_attempts"]:
        promotion["ordered_attempts"].append("e2_rolling4")
    resolved = dict(report["candidate_resolved"])
    resolved.pop("name", None)
    promotion["variants"]["e2_rolling4"] = {
        "passed": bool(report["gate"]["passed"]),
        "parent": BASELINE.name,
        "comparison_mode": "isolated_vs_e0_baseline00",
        "members_root": report["candidate_members_root"],
        "recipe": str(recipe_path),
        "report": str(report_path),
        "resolved": resolved,
        "gate": report["gate"],
    }
    promotion["latest_accepted"] = latest_before
    _atomic_write_json(promotion_path, promotion)
    return {
        "accepted": bool(report["gate"]["passed"]),
        "latest_accepted": latest_before,
        "comparison_mode": "isolated_vs_e0_baseline00",
    }


def record_isolated_e3(
    report_path: Path,
    recipe_path: Path,
    promotion_log: Path,
) -> dict[str, Any]:
    report_path = Path(report_path).resolve()
    recipe_path = Path(recipe_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if report.get("parent") != BASELINE.name or report.get("candidate") != "e3_drop_model_c":
        raise ValueError("record_isolated_e3 requires an E0 to E3 report")
    if report.get("reference_check", {}).get("passed") is not True:
        raise ValueError("E3 did not reproduce the E0 reference")
    if recipe.get("variant") != "e3_drop_model_c" or recipe.get("parent_variant") != BASELINE.name:
        raise ValueError("E3 recipe has the wrong parent or variant")
    if bool(recipe.get("accepted_by_local_gate")) != bool(report["gate"]["passed"]):
        raise ValueError("E3 recipe and report gate decisions differ")

    promotion_path = Path(promotion_log)
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    latest_before = promotion["latest_accepted"]
    if "e3_drop_model_c" not in promotion["ordered_attempts"]:
        promotion["ordered_attempts"].append("e3_drop_model_c")
    resolved = dict(report["candidate_resolved"])
    resolved.pop("name", None)
    promotion["variants"]["e3_drop_model_c"] = {
        "passed": bool(report["gate"]["passed"]),
        "parent": BASELINE.name,
        "comparison_mode": "isolated_vs_e0_baseline00",
        "members_root": report["candidate_members_root"],
        "recipe": str(recipe_path),
        "report": str(report_path),
        "resolved": resolved,
        "gate": report["gate"],
    }
    promotion["latest_accepted"] = latest_before
    _atomic_write_json(promotion_path, promotion)
    return {
        "accepted": bool(report["gate"]["passed"]),
        "latest_accepted": latest_before,
        "comparison_mode": "isolated_vs_e0_baseline00",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--qroot", type=Path, required=True)
    initialize.add_argument("--git-commit", required=True)
    finalize = subparsers.add_parser("finalize-e1")
    finalize.add_argument("--report", type=Path, required=True)
    finalize.add_argument("--parent-recipe", type=Path, required=True)
    finalize.add_argument("--promotion-log", type=Path, required=True)
    finalize.add_argument("--recipe-out", type=Path, required=True)
    finalize.add_argument("--e0-out", type=Path, required=True)
    record_e2 = subparsers.add_parser("record-e2")
    record_e2.add_argument("--report", type=Path, required=True)
    record_e2.add_argument("--recipe", type=Path, required=True)
    record_e2.add_argument("--promotion-log", type=Path, required=True)
    record_e3 = subparsers.add_parser("record-e3")
    record_e3.add_argument("--report", type=Path, required=True)
    record_e3.add_argument("--recipe", type=Path, required=True)
    record_e3.add_argument("--promotion-log", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "initialize":
        qroot = args.qroot.resolve()
        baseline = qroot / "work" / "baseline00"
        root = qroot / "work" / "baseline01"
        manifest = create_reference_manifest(
            qroot,
            root / "reference_manifest.json",
            git_commit=args.git_commit,
        )
        promotion = initialize_promotion_log(
            root / "reports" / "promotion_log.json",
            members_root=baseline / "members",
            recipe=baseline / "frozen_recipe.json",
        )
        print(json.dumps({"manifest": manifest, "promotion": promotion}, indent=2))
    elif args.command == "finalize-e1":
        result = finalize_e1(
            args.report,
            parent_recipe=args.parent_recipe,
            promotion_log=args.promotion_log,
            recipe_output=args.recipe_out,
            e0_output=args.e0_out,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "record-e2":
        result = record_isolated_e2(
            args.report,
            args.recipe,
            args.promotion_log,
        )
        print(json.dumps(result, indent=2))
    else:
        result = record_isolated_e3(
            args.report,
            args.recipe,
            args.promotion_log,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
