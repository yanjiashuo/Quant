from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from repro.baseline00 import MEMBER_NAMES
from repro.baseline00.rolling import V1_OPERATIONS
from repro.baseline00.train_production import package_model
from repro.baseline00_02.config import VARIANT
from repro.config import FEATS, ROLLING_TOP50, ensure_data2_path


MODEL_ARTIFACTS = (
    "model_a.txt",
    "model_b.txt",
    "model_c.txt",
    "weight_model.txt",
    "level_model.txt",
    "rolling_v1.txt",
    "rolling_history.npz",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_contract(source_config: dict[str, Any]) -> None:
    if source_config.get("version") != "baseline00_01a_gap0":
        raise ValueError("source version must be baseline00_01a_gap0")
    if source_config.get("production_gap_time_ids") != 0:
        raise ValueError("source production gap must be zero")
    if source_config.get("feature_columns") != list(FEATS):
        raise ValueError("source raw feature order differs from Baseline00")
    if source_config.get("rolling_feature_columns") != list(ROLLING_TOP50):
        raise ValueError("source rolling Top50 order differs from Baseline00")
    if source_config.get("rolling_operations") != list(V1_OPERATIONS):
        raise ValueError("source rolling operations differ from Rolling6")
    if source_config.get("member_names") != list(MEMBER_NAMES):
        raise ValueError("source member order differs from Baseline00")
    if source_config.get("deterministic") is not True:
        raise ValueError("source must use deterministic LightGBM")
    if source_config.get("force_col_wise") is not True:
        raise ValueError("source must use force_col_wise LightGBM")
    seed_profile = source_config.get("seed_profile")
    expected_components = {
        "model_a",
        "model_b",
        "model_c",
        "weight_model",
        "level_model",
        "rolling_v1",
    }
    if not isinstance(seed_profile, dict) or not expected_components.issubset(seed_profile):
        raise ValueError("source seed profile is incomplete")
    if {int(seed_profile[name]) for name in expected_components} != {0}:
        raise ValueError("source seed profile must use seed 0 for every model")


def _load_recent_recipe(path: Path) -> dict[str, Any]:
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    if recipe.get("variant_name") != VARIANT.name:
        raise ValueError("recipe variant does not match Baseline00.02")
    if recipe.get("member_names") != list(MEMBER_NAMES):
        raise ValueError("recipe member order does not match Baseline00.02")
    if recipe.get("fit_folds") != list(VARIANT.oof_folds):
        raise ValueError("recipe fit folds must be [6, 7, 8]")
    if recipe.get("scale_source_fold") != VARIANT.scale_source_fold:
        raise ValueError("recipe scale source fold must be Fold8")
    if recipe.get("purge_gap_time_ids") != 0 or recipe.get("production_gap_time_ids") != 0:
        raise ValueError("recipe validation and production gaps must be zero")
    if recipe.get("base_random_state") != 0 or recipe.get("shared_random_state") != 0:
        raise ValueError("recipe seeds must both be zero")
    if recipe.get("rolling_operations") != list(V1_OPERATIONS):
        raise ValueError("recipe rolling operations differ from Rolling6")
    if recipe.get("frozen_before_public_test") is not True:
        raise ValueError("recipe must be frozen before public-test inference")
    coefficients = np.asarray(recipe.get("effective_coefficients"), dtype=np.float64)
    if coefficients.shape != (len(MEMBER_NAMES),) or not np.isfinite(coefficients).all():
        raise ValueError("recipe must contain five finite Ridge coefficients")
    if np.any(coefficients < 0.0):
        raise ValueError("recipe Ridge coefficients must be nonnegative")
    amplitude = float(recipe.get("amplitude_multiplier", np.nan))
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("recipe c8 amplitude must be positive and finite")
    return recipe


def _copy_with_hash(source: Path, destination: Path) -> dict[str, Any]:
    temporary = destination.parent / f".{destination.name}.tmp"
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    source_hash = _sha256(source)
    destination_hash = _sha256(destination)
    if source_hash != destination_hash:
        raise IOError(f"hash mismatch while copying {source.name}")
    return {
        "source": str(source),
        "destination": str(destination),
        "bytes": int(destination.stat().st_size),
        "source_sha256": source_hash,
        "destination_sha256": destination_hash,
    }


def build_production_package(
    source_dir: Path,
    recipe_path: Path,
    destination_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    source = Path(source_dir).resolve()
    destination = ensure_data2_path(destination_dir).resolve()
    recipe_file = Path(recipe_path).resolve()
    report_file = ensure_data2_path(report_path).resolve()
    if source == destination:
        raise ValueError("source and destination package directories must differ")
    source_config_path = source / "production_config.json"
    if not source_config_path.is_file():
        raise ValueError("source production_config.json is missing")
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    validate_source_contract(source_config)
    recipe = _load_recent_recipe(recipe_file)
    missing = [name for name in MODEL_ARTIFACTS if not (source / name).is_file()]
    if missing:
        raise ValueError(f"source production package is missing {missing}")

    destination.mkdir(parents=True, exist_ok=True)
    reused = {
        name: _copy_with_hash(source / name, destination / name)
        for name in MODEL_ARTIFACTS
    }
    config = dict(source_config)
    config.update(
        version=VARIANT.name,
        effective_coefficients=recipe["effective_coefficients"],
        amplitude_multiplier=recipe["amplitude_multiplier"],
        purge_gap_time_ids=VARIANT.validation_gap_time_ids,
        validation_gap_time_ids=VARIANT.validation_gap_time_ids,
        production_gap_time_ids=VARIANT.production_gap_time_ids,
        ridge_fit_folds=list(VARIANT.oof_folds),
        scale_source_fold=VARIANT.scale_source_fold,
    )
    package_hashes = package_model(destination, recipe_file, config)
    report = {
        "variant_name": VARIANT.name,
        "source_dir": str(source),
        "source_config": str(source_config_path),
        "source_config_sha256": _sha256(source_config_path),
        "recipe": str(recipe_file),
        "recipe_sha256": _sha256(recipe_file),
        "destination_dir": str(destination),
        "reused_artifacts": reused,
        "package_artifacts": package_hashes,
        "resolved_contract": {
            "production_gap_time_ids": 0,
            "seed": 0,
            "member_names": list(MEMBER_NAMES),
            "rolling_operations": list(V1_OPERATIONS),
            "ridge_fit_folds": list(VARIANT.oof_folds),
            "scale_source_fold": VARIANT.scale_source_fold,
            "amplitude_multiplier": float(recipe["amplitude_multiplier"]),
        },
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = build_production_package(args.source, args.recipe, args.out, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
