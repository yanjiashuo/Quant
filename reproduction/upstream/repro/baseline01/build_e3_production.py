from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from repro.baseline01.config import BASELINE, SIX_OPERATIONS
from repro.baseline01.e3_recipe import E3_MEMBERS, REQUIRED_BASE_MODELS
from repro.config import WEIGHT_C, ensure_data2_path


UNCHANGED_PACKAGE_FILES = (
    "model_a.txt",
    "model_b.txt",
    "model_c.txt",
    "weight_model.txt",
    "level_model.txt",
    "rolling_v1.txt",
    "rolling_history.npz",
    "rolling.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_e3_recipe(path: Path) -> dict[str, Any]:
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    if recipe.get("variant") != "e3_drop_model_c" or recipe.get("parent_variant") != BASELINE.name:
        raise ValueError("production requires the isolated E3 recipe")
    if tuple(recipe.get("member_names", ())) != E3_MEMBERS:
        raise ValueError("E3 production member order is invalid")
    if tuple(recipe.get("required_base_models", ())) != REQUIRED_BASE_MODELS:
        raise ValueError("E3 production must retain the model_c base booster")
    if tuple(recipe.get("rolling_operations", ())) != SIX_OPERATIONS:
        raise ValueError("E3 production changed Rolling operations")
    if not np.isclose(
        float(recipe["amplitude_multiplier"]),
        1.0884372894699135,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("E3 production amplitude differs from Baseline00")
    return recipe


def package_e3(
    source_package: Path,
    output: Path,
    recipe_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    source_package = Path(source_package).resolve()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    recipe = load_e3_recipe(recipe_path)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("candidate") != "e3_drop_model_c":
        raise ValueError("production report is not E3")

    unchanged: dict[str, dict[str, str]] = {}
    for name in UNCHANGED_PACKAGE_FILES:
        source = source_package / name
        destination = output / name
        shutil.copy2(source, destination)
        source_hash = sha256(source)
        destination_hash = sha256(destination)
        if source_hash != destination_hash:
            raise AssertionError(f"copied Baseline00 file changed: {name}")
        unchanged[name] = {"source_sha256": source_hash, "e3_sha256": destination_hash}

    source_code = Path(__file__).resolve().parents[1] / "baseline00" / "api_model.py"
    shutil.copy2(source_code, output / "main.py")
    shutil.copy2(recipe_path, output / "frozen_recipe.json")
    config = json.loads((source_package / "production_config.json").read_text(encoding="utf-8"))
    if not np.isclose(float(config["formula"]["weight_c"]), WEIGHT_C, rtol=0.0, atol=1e-15):
        raise ValueError("Baseline00 internal model_c weight changed")
    baseline_gate = config.get("production_gate")
    config.update(
        {
            "version": "e3_drop_model_c_only",
            "member_names": list(E3_MEMBERS),
            "effective_coefficients": recipe["effective_coefficients"],
            "amplitude_multiplier": float(recipe["amplitude_multiplier"]),
            "rolling_operations": list(SIX_OPERATIONS),
            "required_base_models": list(REQUIRED_BASE_MODELS),
            "production_gate": {
                "comparison_anchor": BASELINE.name,
                "e3_local_gate": report["gate"],
                "baseline00_gate": baseline_gate,
                "csv_generation_allowed_when_rejected": True,
            },
        }
    )
    config.pop("rolling_column_indices", None)
    (output / "production_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    required = (*UNCHANGED_PACKAGE_FILES, "main.py", "frozen_recipe.json", "production_config.json")
    manifest = {
        "variant": "e3_drop_model_c",
        "comparison_anchor": BASELINE.name,
        "internal_model_c_retained": True,
        "internal_model_c_weight": float(config["formula"]["weight_c"]),
        "unchanged_from_baseline00": unchanged,
        "files": {
            name: {"bytes": (output / name).stat().st_size, "sha256": sha256(output / name)}
            for name in required
        },
    }
    (output / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = ensure_data2_path(args.out)
    manifest = package_e3(args.source_package, output, args.recipe, args.report)
    report_path = output.parent / "production_report.json"
    report_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
