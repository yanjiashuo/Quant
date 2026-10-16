from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from repro.baseline01.config import BASELINE, FOUR_OPERATIONS
from repro.config import ROLLING_TOP50, ensure_data2_path


UNCHANGED_PACKAGE_FILES = (
    "model_a.txt",
    "model_b.txt",
    "model_c.txt",
    "weight_model.txt",
    "level_model.txt",
    "rolling_history.npz",
    "rolling.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_e2_recipe(path: Path) -> dict[str, Any]:
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    if recipe.get("variant") != "e2_rolling4" or recipe.get("parent_variant") != BASELINE.name:
        raise ValueError("production requires the isolated E2 recipe")
    if tuple(recipe.get("member_names", ())) != BASELINE.member_names:
        raise ValueError("E2 production member order differs from Baseline00")
    if tuple(recipe.get("rolling_operations", ())) != FOUR_OPERATIONS:
        raise ValueError("E2 production recipe does not select four operations")
    indices = np.asarray(recipe.get("rolling_column_indices"), dtype=np.int64)
    expected_columns = len(ROLLING_TOP50) * len(FOUR_OPERATIONS)
    if indices.shape != (expected_columns,) or np.unique(indices).size != expected_columns:
        raise ValueError("E2 rolling_column_indices are invalid")
    if np.any(indices < 0) or np.any(indices >= len(ROLLING_TOP50) * 6):
        raise ValueError("E2 rolling_column_indices exceed the six-operation cache")
    if not np.isclose(
        float(recipe["amplitude_multiplier"]),
        1.0884372894699135,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("E2 production amplitude differs from Baseline00")
    return recipe


def train_rolling4(cache_dir: Path, output: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    import lightgbm as lgb

    started = time.perf_counter()
    cache_dir = Path(cache_dir)
    source = np.load(cache_dir / "train_features.npy", mmap_mode="r")
    target = np.load(cache_dir / "train_y.npy", mmap_mode="r")
    weight = np.load(cache_dir / "train_w.npy", mmap_mode="r")
    if source.ndim != 2 or source.shape[1] != len(ROLLING_TOP50) * 6:
        raise ValueError(f"production Rolling6 cache has unexpected shape {source.shape}")
    if len(target) != len(source) or len(weight) != len(source):
        raise ValueError("production Rolling cache arrays have unequal rows")
    indices = np.asarray(recipe["rolling_column_indices"], dtype=np.int64)
    selected = np.ascontiguousarray(source[:, indices], dtype=np.float32)
    model = lgb.LGBMRegressor(
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.05,
        min_data_in_leaf=500,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        reg_lambda=1.0,
        n_jobs=64,
        verbosity=-1,
        random_state=0,
        deterministic=True,
        force_col_wise=True,
    )
    model.fit(selected, target, sample_weight=weight)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(output))
    report = {
        "source_cache": str(cache_dir.resolve()),
        "source_shape": list(source.shape),
        "selected_shape": list(selected.shape),
        "rolling_operations": list(FOUR_OPERATIONS),
        "rolling_column_indices": indices.tolist(),
        "model_num_features": int(model.booster_.num_feature()),
        "model_sha256": sha256(output),
        "elapsed_seconds": time.perf_counter() - started,
    }
    del selected, source, target, weight, model
    gc.collect()
    return report


def package_e2(
    source_package: Path,
    output: Path,
    recipe_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    source_package = Path(source_package).resolve()
    output = Path(output)
    recipe = load_e2_recipe(recipe_path)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("candidate") != "e2_rolling4":
        raise ValueError("production report is not E2")
    if not (output / "rolling_v1.txt").is_file():
        raise FileNotFoundError(output / "rolling_v1.txt")

    unchanged: dict[str, dict[str, str]] = {}
    for name in UNCHANGED_PACKAGE_FILES:
        source = source_package / name
        destination = output / name
        shutil.copy2(source, destination)
        source_hash = sha256(source)
        destination_hash = sha256(destination)
        if source_hash != destination_hash:
            raise AssertionError(f"copied Baseline00 file changed: {name}")
        unchanged[name] = {"source_sha256": source_hash, "e2_sha256": destination_hash}

    source_code = Path(__file__).resolve().parents[1] / "baseline00" / "api_model.py"
    shutil.copy2(source_code, output / "main.py")
    shutil.copy2(recipe_path, output / "frozen_recipe.json")
    config = json.loads((source_package / "production_config.json").read_text(encoding="utf-8"))
    baseline_gate = config.get("production_gate")
    config.update(
        {
            "version": "e2_rolling4_only",
            "member_names": list(recipe["member_names"]),
            "effective_coefficients": recipe["effective_coefficients"],
            "amplitude_multiplier": float(recipe["amplitude_multiplier"]),
            "rolling_operations": list(FOUR_OPERATIONS),
            "rolling_column_indices": recipe["rolling_column_indices"],
            "production_gate": {
                "comparison_anchor": BASELINE.name,
                "e2_local_gate": report["gate"],
                "baseline00_gate": baseline_gate,
                "csv_generation_allowed_when_rejected": True,
            },
        }
    )
    (output / "production_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    required = (*UNCHANGED_PACKAGE_FILES, "rolling_v1.txt", "main.py", "frozen_recipe.json", "production_config.json")
    manifest = {
        "variant": "e2_rolling4",
        "comparison_anchor": BASELINE.name,
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
    parser.add_argument("--component", choices=("rolling", "package", "all"), default="all")
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = ensure_data2_path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    recipe = load_e2_recipe(args.recipe)
    result: dict[str, Any] = {}
    if args.component in {"rolling", "all"}:
        result["rolling"] = train_rolling4(
            args.source_cache, output / "rolling_v1.txt", recipe
        )
    if args.component in {"package", "all"}:
        result["package"] = package_e2(
            args.source_package, output, args.recipe, args.report
        )
    report_path = output.parent / "production_report.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
