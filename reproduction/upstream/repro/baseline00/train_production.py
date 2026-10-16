from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from repro.baseline00 import MEMBER_NAMES
from repro.baseline00.frozen_recipe import validate_frozen_recipe
from repro.baseline00.train_base import MODEL_RECIPES
from repro.baseline00.train_level import _level_training_data
from repro.baseline00.train_rolling import _features_with_history, _selected_rows
from repro.baseline00.training_io import load_training_arrays
from repro.baseline00.rolling import V1_OPERATIONS
from repro.baseline00_01.config import (
    BASE_COMPONENTS,
    SHARED_COMPONENTS,
    identify_variant,
    resolve_production_cutoff,
)
from repro.config import (
    ASSET_SCALE,
    ASSET_SCALE_SHRINK,
    CS_DEV_ALPHA,
    CS_MEAN_BETA,
    FEATS,
    L88_DEV,
    L88_LEVEL,
    N_TRAIN_PARTITIONS,
    OUTPUT_SCALE,
    POST_BIAS,
    POST_SCALE,
    ROLLING_TOP50,
    SCALE_LEVEL,
    TEST_DIR,
    TRAIN_DIR,
    WEIGHT_A,
    WEIGHT_B,
    WEIGHT_C,
    WORK,
    W_LEVEL,
    ensure_data2_path,
)


EXPECTED_ITERATIONS = {
    "model_a": 139,
    "model_b": 56,
    "model_c": 201,
    "weight_model": 400,
    "level_model": 60,
    "rolling_v1": 200,
}

PRODUCTION_STOP_THRESHOLD = 0.00030


def evaluate_production_gate(
    heldout_report: Path,
    oof_report: Path,
    *,
    stop_threshold: float = PRODUCTION_STOP_THRESHOLD,
    override: bool = False,
) -> dict[str, Any]:
    """Validate the existing Fold 8 decision without rereading any labels."""
    heldout_path = Path(heldout_report)
    oof_path = Path(oof_report)
    heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
    oof = json.loads(oof_path.read_text(encoding="utf-8"))

    if heldout.get("held_out") is not True or int(heldout.get("report_fold", -1)) != 8:
        raise ValueError("heldout report must describe Fold 8 with held_out=true")
    if heldout.get("held_out_labels_read_once") is not True:
        raise ValueError("heldout report must record held_out_labels_read_once=true")
    summary = oof.get("heldout_gate_summary_from_existing_json")
    if not isinstance(summary, dict):
        raise ValueError("OOF report lacks heldout_gate_summary_from_existing_json")
    if summary.get("labels_reloaded") is not False:
        raise ValueError("OOF gate summary must record labels_reloaded=false")

    heldout_score = float(heldout["report_calibrated_score"])
    summary_score = float(summary["heldout_calibrated_score"])
    development_mean = float(summary["development_calibrated_mean"])
    reported_drop = float(summary["drop"])
    reported_threshold = float(summary["threshold"])
    computed_drop = development_mean - heldout_score
    if not np.isclose(summary_score, heldout_score, rtol=0.0, atol=1e-15):
        raise ValueError("heldout score differs between heldout and OOF reports")
    if not np.isclose(reported_drop, computed_drop, rtol=0.0, atol=1e-15):
        raise ValueError(
            f"reported drop {reported_drop} differs from computed drop {computed_drop}"
        )
    if not np.isclose(reported_threshold, stop_threshold, rtol=0.0, atol=1e-15):
        raise ValueError(
            f"reported threshold {reported_threshold} differs from required {stop_threshold}"
        )

    passed = bool(heldout_score > 0.0 and computed_drop <= stop_threshold)
    if summary.get("passed") is not passed:
        raise ValueError("reported gate decision differs from recomputed decision")
    override_requested = bool(override)
    override_used = bool(override_requested and not passed)
    return {
        "heldout_calibrated_score": heldout_score,
        "development_calibrated_mean": development_mean,
        "drop": computed_drop,
        "threshold": float(stop_threshold),
        "passed": passed,
        "override_requested": override_requested,
        "override_used": override_used,
        "allowed": bool(passed or override_requested),
        "labels_reloaded": False,
        "report_paths": {
            "heldout_report": str(heldout_path),
            "oof_report": str(oof_path),
        },
        "report_sha256": {
            "heldout_report": sha256(heldout_path),
            "oof_report": sha256(oof_path),
        },
    }


def require_production_gate(gate: dict[str, Any]) -> dict[str, Any]:
    if not gate.get("allowed", False):
        raise RuntimeError(
            "Fold 8 production gate failed: "
            f"drop={float(gate['drop']):.12f} exceeds "
            f"threshold={float(gate['threshold']):.12f}; "
            "an explicit --override-heldout-gate is required"
        )
    return gate


def test_metadata(test_dir: Path) -> dict[str, Any]:
    paths = sorted(Path(test_dir).glob("*.parquet"))
    if not paths:
        raise ValueError(f"no test parquet files found beneath {test_dir}")
    first_time_ids: list[int] = []
    asset_parts: list[np.ndarray] = []
    rows = 0
    for path in paths:
        table = pq.read_table(path, columns=["time_id", "asset_id"])
        time_id = table["time_id"].to_numpy()
        asset_id = table["asset_id"].to_numpy()
        first_time_ids.append(int(time_id.min()))
        asset_parts.append(np.unique(asset_id.astype(np.int64, copy=False)))
        rows += len(table)
    return {
        "first_time_id": min(first_time_ids),
        "asset_ids": np.unique(np.concatenate(asset_parts)),
        "rows": int(rows),
        "partitions": int(len(paths)),
    }


def first_test_time_id(test_dir: Path) -> int:
    return int(test_metadata(test_dir)["first_time_id"])


def save_history(history: pd.DataFrame, feature_columns: list[str], path: Path) -> dict[str, Any]:
    ordered = history.sort_values(["asset_id", "time_id"], kind="stable")
    asset_ids = np.sort(ordered["asset_id"].unique().astype(np.int64))
    values = np.full((len(asset_ids), 9, len(feature_columns)), np.nan, dtype=np.float32)
    counts = np.zeros(len(asset_ids), dtype=np.int64)
    for index, asset in enumerate(asset_ids):
        tail = ordered.loc[ordered["asset_id"] == asset].tail(9)
        block = tail.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True)
        block[np.isinf(block)] = np.nan
        counts[index] = len(block)
        if len(block):
            values[index, -len(block) :] = block
    last_time_id = int(ordered["time_id"].max())
    np.savez(
        path,
        asset_ids=asset_ids,
        values=values,
        counts=counts,
        last_time_id=np.array(last_time_id, dtype=np.int64),
    )
    return {
        "assets": int(len(asset_ids)),
        "rows": int(counts.sum()),
        "last_time_id": last_time_id,
        "all_assets_have_nine_rows": bool(np.all(counts == 9)),
    }


def validate_saved_history(
    path: Path, expected_asset_ids: np.ndarray, first_test_time_id: int
) -> dict[str, Any]:
    history_path = Path(path)
    with np.load(history_path) as history:
        asset_ids = np.asarray(history["asset_ids"], dtype=np.int64)
        values = np.asarray(history["values"])
        counts = np.asarray(history["counts"], dtype=np.int64)
        last_time_id = int(history["last_time_id"])
    expected = np.unique(np.asarray(expected_asset_ids, dtype=np.int64))
    if not np.array_equal(asset_ids, expected):
        raise ValueError(
            f"rolling history asset IDs {asset_ids.tolist()} differ from test {expected.tolist()}"
        )
    if values.ndim != 3 or values.shape[:2] != (len(asset_ids), 9):
        raise ValueError(f"rolling history values have invalid shape {values.shape}")
    if counts.shape != (len(asset_ids),) or not np.all(counts == 9):
        raise ValueError("rolling history must contain nine rows for every test asset")
    if last_time_id >= int(first_test_time_id):
        raise ValueError(
            f"rolling history last_time_id={last_time_id} must precede first test "
            f"time_id={int(first_test_time_id)}"
        )
    return {
        "path": str(history_path),
        "assets": int(len(asset_ids)),
        "rows": int(counts.sum()),
        "rows_per_asset": 9,
        "last_time_id": last_time_id,
        "first_test_time_id": int(first_test_time_id),
        "causal_time_order": True,
        "sha256": sha256(history_path),
    }


def _regressor(*, random_state: int = 0, **kwargs: Any):
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        n_jobs=int(os.environ.get("BASELINE00_TRAIN_THREADS", "64")),
        histogram_pool_size=float(os.environ.get("BASELINE00_HISTOGRAM_POOL_MB", "-1")),
        verbosity=-1,
        random_state=random_state,
        deterministic=True,
        force_col_wise=True,
        **kwargs,
    )


def train_base(
    train_dir: Path,
    model_dir: Path,
    cutoff: int,
    model_names: list[str] | None = None,
    random_state: int = 0,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    fold = N_TRAIN_PARTITIONS
    selected_names = model_names or list(MODEL_RECIPES)
    for name in selected_names:
        recipe = MODEL_RECIPES[name]
        started = time.perf_counter()
        if recipe["window"] == "all":
            partitions = list(range(fold))
        else:
            width = int(recipe["window"])
            partitions = list(range(fold - width, fold))
        x, y, weight, rows = load_training_arrays(
            train_dir,
            partitions,
            FEATS,
            cutoff_time_id=cutoff,
            label_mode=str(recipe["label_mode"]),
        )
        model = _regressor(
            random_state=random_state,
            n_estimators=int(recipe["n_estimators"]),
            num_leaves=int(recipe["num_leaves"]),
            learning_rate=float(recipe["learning_rate"]),
            min_data_in_leaf=int(recipe["min_data_in_leaf"]),
        )
        model.fit(x, y, sample_weight=weight)
        model.booster_.save_model(str(model_dir / f"{name}.txt"))
        report[name] = {
            "partitions": partitions,
            "rows_by_partition": rows,
            "train_rows": int(len(y)),
            "label_mode": recipe["label_mode"],
            "n_estimators": int(recipe["n_estimators"]),
            "prediction_scale": float(recipe["prediction_scale"]),
            "seed": int(random_state),
            "deterministic": True,
            "force_col_wise": True,
            "elapsed_seconds": time.perf_counter() - started,
        }
        del x, y, weight, model
        gc.collect()
    return report


def _weight_training_cache(
    train_dir: Path,
    partitions: list[int],
    cutoff: int,
    cache_dir: Path,
) -> tuple[np.memmap, np.memmap, dict[str, int]]:
    paths = [train_dir / f"train_partition_{partition:03d}.parquet" for partition in partitions]
    rows = {
        str(partition): _selected_rows(path, cutoff if partition == max(partitions) else None)
        for partition, path in zip(partitions, paths)
    }
    total_rows = int(sum(rows.values()))
    cache_dir.mkdir(parents=True, exist_ok=True)
    x_cache = np.lib.format.open_memmap(
        cache_dir / "weight_features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, len(FEATS)),
    )
    y_cache = np.lib.format.open_memmap(
        cache_dir / "weight_y.npy", mode="w+", dtype=np.float64, shape=(total_rows,)
    )
    cursor = 0
    for partition, path in zip(partitions, paths):
        frame = pd.read_parquet(path, columns=["time_id", "weight", *FEATS])
        if partition == max(partitions):
            frame = frame.loc[frame["time_id"] < cutoff]
        count = len(frame)
        block = frame.loc[:, FEATS].to_numpy(dtype=np.float32, copy=True)
        block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
        x_cache[cursor : cursor + count] = block
        y_cache[cursor : cursor + count] = np.maximum(
            frame["weight"].to_numpy(dtype=np.float64, copy=False), 0.0
        )
        cursor += count
        del frame, block
        gc.collect()
    if cursor != total_rows:
        raise AssertionError(f"weight cache rows differ: {cursor} != {total_rows}")
    x_cache.flush()
    y_cache.flush()
    return x_cache, y_cache, rows


def train_level(
    train_dir: Path,
    model_dir: Path,
    cache_dir: Path,
    cutoff: int,
    random_state: int = 0,
) -> dict[str, Any]:
    started = time.perf_counter()
    weight_partitions = list(range(N_TRAIN_PARTITIONS - 3, N_TRAIN_PARTITIONS))
    x_weight, y_weight, weight_rows = _weight_training_cache(
        train_dir, weight_partitions, cutoff, cache_dir / "level"
    )
    weight_model = _regressor(
        random_state=random_state,
        n_estimators=400,
        num_leaves=63,
        learning_rate=0.05,
        min_data_in_leaf=2_000,
    )
    weight_model.fit(x_weight, y_weight)
    weight_model.booster_.save_model(str(model_dir / "weight_model.txt"))
    del x_weight, y_weight, weight_model
    gc.collect()

    level_x, level_y, level_weight, level_rows = _level_training_data(
        train_dir, N_TRAIN_PARTITIONS, cutoff
    )
    level_model = _regressor(
        random_state=random_state,
        n_estimators=60,
        num_leaves=31,
        learning_rate=0.03,
        min_data_in_leaf=200,
    )
    level_model.fit(level_x, level_y, sample_weight=level_weight)
    level_model.booster_.save_model(str(model_dir / "level_model.txt"))
    report = {
        "weight_partitions": weight_partitions,
        "weight_rows_by_partition": weight_rows,
        "weight_train_rows": int(sum(weight_rows.values())),
        "level_rows_by_partition": level_rows,
        "level_train_time_rows": int(len(level_y)),
        "seed_profile": {
            "weight_model": int(random_state),
            "level_model": int(random_state),
        },
        "deterministic": True,
        "force_col_wise": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    del level_x, level_y, level_weight, level_model
    gc.collect()
    return report


def train_rolling(
    train_dir: Path,
    model_dir: Path,
    cache_dir: Path,
    cutoff: int,
    random_state: int = 0,
) -> dict[str, Any]:
    started = time.perf_counter()
    feature_columns = list(ROLLING_TOP50)
    columns = ["time_id", "asset_id", "weight", "target", *feature_columns]
    paths = [train_dir / f"train_partition_{p:03d}.parquet" for p in range(N_TRAIN_PARTITIONS)]
    train_rows = sum(
        _selected_rows(path, cutoff if index == N_TRAIN_PARTITIONS - 1 else None)
        for index, path in enumerate(paths)
    )
    n_columns = len(feature_columns) * len(V1_OPERATIONS)
    cache_dir.mkdir(parents=True, exist_ok=True)
    x_cache = np.lib.format.open_memmap(
        cache_dir / "train_features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(train_rows, n_columns),
    )
    y_cache = np.lib.format.open_memmap(
        cache_dir / "train_y.npy", mode="w+", dtype=np.float64, shape=(train_rows,)
    )
    w_cache = np.lib.format.open_memmap(
        cache_dir / "train_w.npy", mode="w+", dtype=np.float64, shape=(train_rows,)
    )
    history: pd.DataFrame | None = None
    cursor = 0
    rows_by_partition: dict[str, int] = {}
    for partition, path in enumerate(paths):
        frame = pd.read_parquet(path, columns=columns).sort_values(
            ["time_id", "asset_id"], kind="stable", ignore_index=True
        )
        matrix, ordered, history = _features_with_history(frame, history, feature_columns)
        selected = (
            ordered["time_id"].to_numpy(copy=False) < cutoff
            if partition == N_TRAIN_PARTITIONS - 1
            else np.ones(len(ordered), dtype=bool)
        )
        count = int(selected.sum())
        x_cache[cursor : cursor + count] = matrix[selected]
        y_cache[cursor : cursor + count] = ordered.loc[selected, "target"].to_numpy(
            dtype=np.float64
        )
        w_cache[cursor : cursor + count] = np.maximum(
            ordered.loc[selected, "weight"].to_numpy(dtype=np.float64), 0.0
        )
        cursor += count
        rows_by_partition[str(partition)] = count
        del frame, matrix, ordered
        gc.collect()
    if cursor != train_rows or history is None:
        raise AssertionError(f"production rolling rows differ: {cursor} != {train_rows}")
    x_cache.flush()
    y_cache.flush()
    w_cache.flush()
    history_report = save_history(
        history, feature_columns, model_dir / "rolling_history.npz"
    )
    model = _regressor(
        random_state=random_state,
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.05,
        min_data_in_leaf=500,
    )
    model.fit(x_cache, y_cache, sample_weight=w_cache)
    model.booster_.save_model(str(model_dir / "rolling_v1.txt"))
    report = {
        "feature_columns": feature_columns,
        "operations": list(V1_OPERATIONS),
        "rolling_columns": n_columns,
        "rows_by_partition": rows_by_partition,
        "train_rows": train_rows,
        "history": history_report,
        "elapsed_seconds": time.perf_counter() - started,
        "causal_note": "purged labels are excluded; all visible train features seed test history",
        "seed": int(random_state),
        "deterministic": True,
        "force_col_wise": True,
    }
    del x_cache, y_cache, w_cache, model, history
    gc.collect()
    return report


def production_config(
    recipe: dict[str, Any],
    cutoff: int,
    test_min: int,
    gate: dict[str, Any],
    test_info: dict[str, Any],
    production_gap_time_ids: int | None = None,
    random_state: int = 0,
    base_random_state: int | None = None,
    shared_random_state: int | None = None,
    variant_name: str | None = None,
) -> dict[str, Any]:
    validation_gap = int(recipe["purge_gap_time_ids"])
    production_gap = (
        validation_gap
        if production_gap_time_ids is None
        else int(production_gap_time_ids)
    )
    base_seed = int(random_state if base_random_state is None else base_random_state)
    shared_seed = int(random_state if shared_random_state is None else shared_random_state)
    seed_profile = {name: base_seed for name in sorted(BASE_COMPONENTS)}
    seed_profile.update({name: shared_seed for name in sorted(SHARED_COMPONENTS)})
    resolved_variant = variant_name
    if resolved_variant is None:
        resolved_variant = (
            identify_variant(production_gap, base_seed)
            if base_seed == shared_seed
            else "custom"
        )
    return {
        "version": resolved_variant,
        "feature_columns": list(FEATS),
        "rolling_feature_columns": list(ROLLING_TOP50),
        "rolling_operations": list(V1_OPERATIONS),
        "member_names": list(MEMBER_NAMES),
        "effective_coefficients": recipe["effective_coefficients"],
        "amplitude_multiplier": recipe["amplitude_multiplier"],
        "purge_gap_time_ids": recipe["purge_gap_time_ids"],
        "validation_gap_time_ids": validation_gap,
        "production_gap_time_ids": production_gap,
        "test_min_time_id": test_min,
        "test_rows": int(test_info["rows"]),
        "test_asset_ids": np.asarray(test_info["asset_ids"], dtype=np.int64).tolist(),
        "label_cutoff_time_id_exclusive": cutoff,
        "production_gate": gate,
        "inference_num_threads": 4,
        "seed_profile": seed_profile,
        "deterministic": True,
        "force_col_wise": True,
        "prediction_scales": {
            name: float(value["prediction_scale"]) for name, value in MODEL_RECIPES.items()
        },
        "formula": {
            "weight_a": WEIGHT_A,
            "weight_b": WEIGHT_B,
            "weight_c": WEIGHT_C,
            "post_bias": POST_BIAS,
            "post_scale": POST_SCALE,
            "asset_scale": list(ASSET_SCALE),
            "asset_scale_shrink": ASSET_SCALE_SHRINK,
            "cs_dev_alpha": CS_DEV_ALPHA,
            "cs_mean_beta": CS_MEAN_BETA,
            "output_scale": OUTPUT_SCALE,
            "w_level": W_LEVEL,
            "scale_level": SCALE_LEVEL,
            "l88_level": L88_LEVEL,
            "l88_deviation": L88_DEV,
        },
    }


def resolve_requested_production_gap(
    recipe: dict[str, Any], requested_gap: int | None
) -> int:
    gap = int(recipe["purge_gap_time_ids"]) if requested_gap is None else int(requested_gap)
    if gap < 0:
        raise ValueError("production gap must be nonnegative")
    return gap


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_model(model_dir: Path, recipe_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    source_dir = Path(__file__).resolve().parent
    shutil.copy2(source_dir / "api_model.py", model_dir / "main.py")
    shutil.copy2(source_dir / "rolling.py", model_dir / "rolling.py")
    shutil.copy2(recipe_path, model_dir / "frozen_recipe.json")
    (model_dir / "production_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    required = [
        "model_a.txt",
        "model_b.txt",
        "model_c.txt",
        "weight_model.txt",
        "level_model.txt",
        "rolling_v1.txt",
        "rolling_history.npz",
        "main.py",
        "rolling.py",
        "frozen_recipe.json",
        "production_config.json",
    ]
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise ValueError(f"cannot package production model; missing {missing}")
    return {
        name: {"bytes": (model_dir / name).stat().st_size, "sha256": sha256(model_dir / name)}
        for name in required
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component", choices=["base", "level", "rolling", "package", "all"], default="all"
    )
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--frozen-recipe", type=Path, default=WORK / "frozen_recipe.json")
    parser.add_argument(
        "--heldout-report", type=Path, default=WORK / "heldout_report.json"
    )
    parser.add_argument("--oof-report", type=Path, default=WORK / "oof_report.json")
    parser.add_argument(
        "--override-heldout-gate",
        action="store_true",
        help="Explicitly continue even when the predeclared Fold 8 stability gate failed.",
    )
    parser.add_argument("--out", type=Path, default=WORK / "production" / "api_model")
    parser.add_argument("--cache", type=Path, default=WORK / "production" / "cache")
    parser.add_argument("--production-gap-time-ids", type=int)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--base-random-state", type=int)
    parser.add_argument("--shared-random-state", type=int)
    parser.add_argument("--variant-name")
    parser.add_argument(
        "--base-models",
        nargs="+",
        choices=list(MODEL_RECIPES),
        default=list(MODEL_RECIPES),
        help="Allow an interrupted base stage to resume one model at a time.",
    )
    args = parser.parse_args()

    model_dir = ensure_data2_path(args.out)
    cache_dir = ensure_data2_path(args.cache)
    gate = require_production_gate(
        evaluate_production_gate(
            args.heldout_report,
            args.oof_report,
            override=args.override_heldout_gate,
        )
    )
    recipe = validate_frozen_recipe(args.frozen_recipe, EXPECTED_ITERATIONS)
    test_info = test_metadata(args.test_dir)
    test_min = int(test_info["first_time_id"])
    production_gap = resolve_requested_production_gap(
        recipe, args.production_gap_time_ids
    )
    base_random_state = (
        args.random_state if args.base_random_state is None else args.base_random_state
    )
    shared_random_state = (
        args.random_state if args.shared_random_state is None else args.shared_random_state
    )
    cutoff = resolve_production_cutoff(test_min, production_gap)
    config = production_config(
        recipe,
        cutoff,
        test_min,
        gate,
        test_info,
        production_gap_time_ids=production_gap,
        random_state=args.random_state,
        base_random_state=base_random_state,
        shared_random_state=shared_random_state,
        variant_name=args.variant_name,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    report_path = model_dir.parent / "production_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    report.update(
        {
            "frozen_recipe": str(args.frozen_recipe),
            "test_min_time_id": test_min,
            "label_cutoff_time_id_exclusive": cutoff,
            "gap_time_ids": production_gap,
            "validation_gap_time_ids": int(recipe["purge_gap_time_ids"]),
            "production_gap_time_ids": production_gap,
            "random_state": int(args.random_state),
            "base_random_state": int(base_random_state),
            "shared_random_state": int(shared_random_state),
            "variant_name": config["version"],
            "seed_profile": config["seed_profile"],
            "deterministic": True,
            "force_col_wise": True,
            "production_gate": gate,
            "frozen_recipe_sha256": sha256(args.frozen_recipe),
        }
    )
    components = ["base", "level", "rolling", "package"] if args.component == "all" else [args.component]
    for component in components:
        started = time.perf_counter()
        if component == "base":
            result = train_base(
                args.train_dir,
                model_dir,
                cutoff,
                args.base_models,
                random_state=base_random_state,
            )
        elif component == "level":
            result = train_level(
                args.train_dir,
                model_dir,
                cache_dir,
                cutoff,
                random_state=shared_random_state,
            )
        elif component == "rolling":
            result = train_rolling(
                args.train_dir,
                model_dir,
                cache_dir,
                cutoff,
                random_state=shared_random_state,
            )
            result["history_validation"] = validate_saved_history(
                model_dir / "rolling_history.npz", test_info["asset_ids"], test_min
            )
        else:
            validate_saved_history(
                model_dir / "rolling_history.npz", test_info["asset_ids"], test_min
            )
            result = package_model(model_dir, args.frozen_recipe, config)
        report[component] = result
        report[f"{component}_elapsed_seconds"] = time.perf_counter() - started
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"component": component, "result": result}, indent=2))


if __name__ == "__main__":
    main()
