from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from repro.baseline00.train_production import (
    package_model,
    production_config,
    test_metadata,
    train_base,
    train_level,
    train_rolling,
    validate_saved_history,
    sha256,
)
from repro.baseline03.config import (
    AMPLITUDE_MULTIPLIER,
    CANDIDATE_MEMBERS,
    MODEL_C_LONG_RECIPE,
    OOF_FOLDS,
    PARENT_MEMBERS,
    RIDGE_ALPHA,
    ROLLING_OPERATIONS,
)
from repro.baseline03.stack import BASE_MODEL_ITERATIONS
from repro.config import TEST_DIR, TRAIN_DIR, ensure_data2_path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def load_parent_recipe(path: Path) -> dict[str, Any]:
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    if recipe.get("variant_name") != "g0_gap0_parent":
        raise ValueError("recipe variant must be g0_gap0_parent")
    if tuple(recipe.get("member_names", ())) != PARENT_MEMBERS:
        raise ValueError("recipe member order differs from G0")
    if tuple(int(value) for value in recipe.get("fit_folds", ())) != OOF_FOLDS:
        raise ValueError("recipe fit folds must be [6, 7, 8]")
    if float(recipe.get("ridge_alpha", np.nan)) != RIDGE_ALPHA:
        raise ValueError("recipe ridge alpha differs from G0")
    if float(recipe.get("amplitude_multiplier", np.nan)) != AMPLITUDE_MULTIPLIER:
        raise ValueError("recipe amplitude differs from frozen Baseline00 amplitude")
    if recipe.get("purge_gap_time_ids") != 0 or recipe.get("production_gap_time_ids") != 0:
        raise ValueError("recipe training and production gaps must both be zero")
    if tuple(recipe.get("rolling_operations", ())) != ROLLING_OPERATIONS:
        raise ValueError("recipe Rolling operations differ from Rolling6")
    if recipe.get("base_model_iterations") != BASE_MODEL_ITERATIONS:
        raise ValueError("recipe model iterations differ from G0")
    if recipe.get("base_random_state") != 0 or recipe.get("shared_random_state") != 0:
        raise ValueError("recipe seeds must both be zero")
    coefficients = np.asarray(recipe.get("effective_coefficients"), dtype=np.float64)
    if coefficients.shape != (len(PARENT_MEMBERS),) or not np.isfinite(coefficients).all():
        raise ValueError("recipe must contain five finite effective coefficients")
    if np.any(coefficients < 0.0):
        raise ValueError("recipe effective coefficients must be nonnegative")
    if recipe.get("frozen_before_public_test") is not True:
        raise ValueError("recipe must be frozen before public inference")
    return recipe


def load_candidate_recipe(path: Path) -> dict[str, Any]:
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    if recipe.get("variant_name") != "e1_model_c_long":
        raise ValueError("recipe variant must be e1_model_c_long")
    if tuple(recipe.get("member_names", ())) != CANDIDATE_MEMBERS:
        raise ValueError("recipe member order differs from E1")
    if tuple(int(value) for value in recipe.get("fit_folds", ())) != OOF_FOLDS:
        raise ValueError("recipe fit folds must be [6, 7, 8]")
    if float(recipe.get("ridge_alpha", np.nan)) != RIDGE_ALPHA:
        raise ValueError("recipe ridge alpha differs from E1")
    if float(recipe.get("amplitude_multiplier", np.nan)) != AMPLITUDE_MULTIPLIER:
        raise ValueError("recipe amplitude differs from frozen E1 amplitude")
    if recipe.get("purge_gap_time_ids") != 0 or recipe.get("production_gap_time_ids") != 0:
        raise ValueError("recipe training and production gaps must both be zero")
    if recipe.get("model_c_long_recipe") != MODEL_C_LONG_RECIPE:
        raise ValueError("model_c_long recipe differs from the predeclared configuration")
    coefficients = np.asarray(recipe.get("effective_coefficients"), dtype=np.float64)
    if coefficients.shape != (len(CANDIDATE_MEMBERS),) or not np.isfinite(coefficients).all():
        raise ValueError("recipe must contain six finite effective coefficients")
    if np.any(coefficients < 0.0):
        raise ValueError("recipe effective coefficients must be nonnegative")
    return recipe


def package_candidate(
    parent_package: Path,
    model_c_long_path: Path,
    recipe_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    recipe = load_candidate_recipe(recipe_path)
    parent = Path(parent_package)
    output = ensure_data2_path(Path(output_dir))
    output.mkdir(parents=True, exist_ok=True)
    reused = (
        "model_a.txt",
        "model_b.txt",
        "model_c.txt",
        "weight_model.txt",
        "level_model.txt",
        "rolling_v1.txt",
        "rolling_history.npz",
    )
    missing = [name for name in reused if not (parent / name).is_file()]
    if missing:
        raise ValueError(f"frozen G0 package is incomplete: {missing}")
    if not Path(model_c_long_path).is_file():
        raise ValueError("model_c_long production Booster is missing")
    parent_config = json.loads(
        (parent / "production_config.json").read_text(encoding="utf-8")
    )
    if tuple(parent_config.get("member_names", ())) != PARENT_MEMBERS:
        raise ValueError("parent API package is not the frozen five-member G0")
    copied: dict[str, dict[str, Any]] = {}
    for name in reused:
        source = parent / name
        destination = output / name
        shutil.copy2(source, destination)
        if sha256(source) != sha256(destination):
            raise ValueError(f"copied G0 artifact changed: {name}")
        copied[name] = {
            "source_sha256": sha256(source),
            "destination_sha256": sha256(destination),
            "bytes": int(destination.stat().st_size),
        }
    shutil.copy2(model_c_long_path, output / "model_c_long.txt")
    source_dir = Path(__file__).resolve().parent
    shutil.copy2(source_dir / "api_model.py", output / "main.py")
    shutil.copy2(source_dir.parent / "baseline00" / "rolling.py", output / "rolling.py")
    shutil.copy2(recipe_path, output / "frozen_recipe.json")
    config = dict(parent_config)
    config.update(
        {
            "version": "e1_model_c_long",
            "member_names": list(CANDIDATE_MEMBERS),
            "effective_coefficients": recipe["effective_coefficients"],
            "amplitude_multiplier": AMPLITUDE_MULTIPLIER,
            "purge_gap_time_ids": 0,
            "validation_gap_time_ids": 0,
            "production_gap_time_ids": 0,
            "model_c_long_recipe": MODEL_C_LONG_RECIPE,
            "candidate_gate": recipe.get("gate", {}),
            "recommended_submission": bool(recipe.get("recommended_submission", False)),
        }
    )
    scales = dict(config["prediction_scales"])
    scales["model_c_long"] = float(MODEL_C_LONG_RECIPE["prediction_scale"])
    config["prediction_scales"] = scales
    _write_json(output / "production_config.json", config)
    required = (*reused, "model_c_long.txt", "main.py", "rolling.py", "frozen_recipe.json", "production_config.json")
    package = {
        name: {"bytes": int((output / name).stat().st_size), "sha256": sha256(output / name)}
        for name in required
    }
    report = {
        "variant": "e1_model_c_long",
        "parent_package": str(parent),
        "model_c_long": str(Path(model_c_long_path)),
        "recipe": str(Path(recipe_path)),
        "reused_parent_artifacts": copied,
        "package": package,
        "only_new_booster": "model_c_long.txt",
    }
    _write_json(output.parent / "production_report.json", report)
    return report


def train_parent_production(
    recipe_path: Path,
    train_dir: Path,
    test_dir: Path,
    output_dir: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    recipe = load_parent_recipe(recipe_path)
    model_dir = ensure_data2_path(Path(output_dir))
    cache = ensure_data2_path(Path(cache_dir))
    model_dir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    test_info = test_metadata(Path(test_dir))
    test_min = int(test_info["first_time_id"])
    cutoff = test_min
    base_report = train_base(Path(train_dir), model_dir, cutoff, None, 0)
    level_report = train_level(Path(train_dir), model_dir, cache, cutoff, 0)
    rolling_report = train_rolling(Path(train_dir), model_dir, cache, cutoff, 0)
    history_report = validate_saved_history(
        model_dir / "rolling_history.npz", test_info["asset_ids"], test_min
    )
    gate = {
        "allowed": True,
        "source": "baseline03_g0_contract",
        "fit_folds": list(OOF_FOLDS),
        "public_labels_used": False,
    }
    config = production_config(
        recipe,
        cutoff,
        test_min,
        gate,
        test_info,
        production_gap_time_ids=0,
        random_state=0,
        base_random_state=0,
        shared_random_state=0,
        variant_name="g0_gap0_parent",
    )
    _write_json(model_dir / "production_config.json", config)
    packaged = package_model(model_dir, Path(recipe_path), config)
    report = {
        "variant": "g0_gap0_parent",
        "recipe": str(Path(recipe_path)),
        "test_min_time_id": test_min,
        "label_cutoff_time_id_exclusive": cutoff,
        "validation_gap_time_ids": 0,
        "production_gap_time_ids": 0,
        "random_state": 0,
        "deterministic": True,
        "force_col_wise": True,
        "base": base_report,
        "level": level_report,
        "rolling": rolling_report,
        "history_validation": history_report,
        "package": packaged,
    }
    _write_json(model_dir.parent / "production_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    parent = subparsers.add_parser("parent")
    parent.add_argument("--recipe", type=Path, required=True)
    parent.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parent.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parent.add_argument("--gap", type=int, default=0)
    parent.add_argument("--random-state", type=int, default=0)
    parent.add_argument("--out", type=Path, required=True)
    parent.add_argument("--cache", type=Path)
    candidate = subparsers.add_parser("candidate")
    candidate.add_argument("--parent-package", type=Path, required=True)
    candidate.add_argument("--model-c-long", type=Path, required=True)
    candidate.add_argument("--recipe", type=Path, required=True)
    candidate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "candidate":
        print(
            json.dumps(
                package_candidate(
                    args.parent_package, args.model_c_long, args.recipe, args.out
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.gap != 0 or args.random_state != 0:
        raise ValueError("G0 production requires gap=0 and random_state=0")
    cache = args.cache or args.out.parent / "cache"
    report = train_parent_production(
        args.recipe, args.train_dir, args.test_dir, args.out, cache
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
