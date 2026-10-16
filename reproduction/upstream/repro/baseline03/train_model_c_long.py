from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from repro.baseline00.metrics import prediction_summary
from repro.baseline00.training_io import load_prediction_frame, load_training_arrays
from repro.baseline03.artifacts import load_oof
from repro.baseline03.config import MODEL_C_LONG_RECIPE, OOF_FOLDS
from repro.config import FEATS, N_TRAIN_PARTITIONS, TEST_DIR, TRAIN_DIR, ensure_data2_path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _regressor():
    import lightgbm as lgb

    recipe = MODEL_C_LONG_RECIPE
    return lgb.LGBMRegressor(
        n_estimators=int(recipe["n_estimators"]),
        num_leaves=int(recipe["num_leaves"]),
        learning_rate=float(recipe["learning_rate"]),
        min_data_in_leaf=int(recipe["min_data_in_leaf"]),
        feature_fraction=float(recipe["feature_fraction"]),
        bagging_fraction=float(recipe["bagging_fraction"]),
        bagging_freq=int(recipe["bagging_freq"]),
        reg_alpha=float(recipe["reg_alpha"]),
        reg_lambda=float(recipe["reg_lambda"]),
        random_state=int(recipe["random_state"]),
        deterministic=bool(recipe["deterministic"]),
        force_col_wise=bool(recipe["force_col_wise"]),
        n_jobs=min(64, int(os.environ.get("BASELINE03_TRAIN_THREADS", "64"))),
        histogram_pool_size=float(os.environ.get("BASELINE00_HISTOGRAM_POOL_MB", "-1")),
        verbosity=-1,
    )


def _scaled_prediction(model: Any, matrix: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict(matrix), dtype=np.float64)
    return np.clip(
        raw * float(MODEL_C_LONG_RECIPE["prediction_scale"]),
        -float(MODEL_C_LONG_RECIPE["prediction_clip"]),
        float(MODEL_C_LONG_RECIPE["prediction_clip"]),
    )


def _validate_parent_index(frame: pd.DataFrame, parent_path: Path) -> tuple[Any, dict[str, np.ndarray]]:
    parent = load_oof(parent_path)
    index = {
        "row_id": frame["row_id"].to_numpy(dtype=np.int64, copy=False),
        "time_id": frame["time_id"].to_numpy(dtype=np.int64, copy=False),
        "asset_id": frame["asset_id"].to_numpy(dtype=np.int64, copy=False),
    }
    for name in index:
        if not np.array_equal(index[name], getattr(parent, name)):
            raise ValueError(f"validation {name} does not exactly align with frozen G0")
    return parent, index


def train_model_c_long_fold(
    train_dir: Path,
    parent_members: Path,
    output_root: Path,
    fold: int,
    gap: int = 0,
) -> dict[str, Any]:
    if fold not in OOF_FOLDS or gap != 0:
        raise ValueError("model_c_long OOF requires fold 6-8 and gap=0")
    started = time.perf_counter()
    root = ensure_data2_path(Path(output_root))
    root.mkdir(parents=True, exist_ok=True)
    frame = load_prediction_frame(Path(train_dir), fold, FEATS)
    parent_path = Path(parent_members) / f"fold_{fold}.npz"
    parent, index = _validate_parent_index(frame, parent_path)
    cutoff = int(frame["time_id"].min())
    partitions = list(range(fold))
    x_train, y_train, weight, rows = load_training_arrays(
        Path(train_dir),
        partitions,
        FEATS,
        cutoff_time_id=cutoff,
        label_mode=str(MODEL_C_LONG_RECIPE["label_mode"]),
    )
    x_valid = np.nan_to_num(
        frame.loc[:, FEATS].to_numpy(dtype=np.float32, copy=True),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    model = _regressor()
    model.fit(x_train, y_train, sample_weight=weight)
    prediction = _scaled_prediction(model, x_valid)
    if not np.isfinite(prediction).all():
        raise ValueError("model_c_long OOF prediction is non-finite")
    fold_dir = root / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(fold_dir / "model_c_long.txt"))
    np.savez(root / f"fold_{fold}.npz", model_c_long=prediction, **index)
    report = {
        "variant": "e1_model_c_long",
        "fold": fold,
        "gap_time_ids": 0,
        "partitions": partitions,
        "rows_by_partition": rows,
        "train_rows": int(y_train.size),
        "valid_rows": parent.rows,
        "label_cutoff_time_id_exclusive": cutoff,
        "valid_min_time_id": int(parent.time_id.min()),
        "train_max_time_id_upper_bound": cutoff - 1,
        "parent_oof": str(parent_path),
        "recipe": MODEL_C_LONG_RECIPE,
        "prediction": prediction_summary(prediction),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(fold_dir / "report.json", report)
    del frame, x_train, y_train, weight, x_valid, model, prediction
    gc.collect()
    return report


def train_model_c_long_production(
    train_dir: Path,
    test_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = ensure_data2_path(Path(output_root))
    root.mkdir(parents=True, exist_ok=True)
    test_paths = sorted(Path(test_dir).glob("test_partition_*.parquet"))
    if not test_paths:
        raise ValueError("no test partitions found")
    first_test_time = min(
        int(pd.read_parquet(path, columns=["time_id"])["time_id"].min())
        for path in test_paths
    )
    partitions = list(range(N_TRAIN_PARTITIONS))
    x_train, y_train, weight, rows = load_training_arrays(
        Path(train_dir),
        partitions,
        FEATS,
        cutoff_time_id=first_test_time,
        label_mode=str(MODEL_C_LONG_RECIPE["label_mode"]),
    )
    model = _regressor()
    model.fit(x_train, y_train, sample_weight=weight)
    model_path = root / "model_c_long.txt"
    model.booster_.save_model(str(model_path))
    public_parts: list[pd.DataFrame] = []
    for path in test_paths:
        frame = pd.read_parquet(path, columns=["row_id", "time_id", "asset_id", *FEATS])
        frame = frame.sort_values(["time_id", "asset_id"], kind="stable", ignore_index=True)
        matrix = np.nan_to_num(
            frame.loc[:, FEATS].to_numpy(dtype=np.float32, copy=True),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        public_parts.append(
            pd.DataFrame(
                {
                    "row_id": frame["row_id"].to_numpy(dtype=np.int64, copy=False),
                    "time_id": frame["time_id"].to_numpy(dtype=np.int64, copy=False),
                    "asset_id": frame["asset_id"].to_numpy(dtype=np.int64, copy=False),
                    "model_c_long": _scaled_prediction(model, matrix),
                }
            )
        )
        del frame, matrix
        gc.collect()
    public = pd.concat(public_parts, ignore_index=True)
    public_path = root.parent / "public_model_c_long.parquet"
    public.to_parquet(public_path, index=False)
    report = {
        "variant": "e1_model_c_long",
        "gap_time_ids": 0,
        "partitions": partitions,
        "rows_by_partition": rows,
        "train_rows": int(y_train.size),
        "test_rows": int(len(public)),
        "label_cutoff_time_id_exclusive": first_test_time,
        "recipe": MODEL_C_LONG_RECIPE,
        "public_prediction": prediction_summary(public["model_c_long"].to_numpy()),
        "model_path": str(model_path),
        "public_prediction_path": str(public_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(root / "report.json", report)
    del x_train, y_train, weight, model, public, public_parts
    gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--parent-members", type=Path)
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.gap != 0:
        raise ValueError("E1 requires gap=0")
    if args.production:
        report: object = train_model_c_long_production(
            args.train_dir, args.test_dir, args.out
        )
    else:
        if not args.parent_members or not args.folds:
            raise ValueError("OOF training requires --parent-members and --folds")
        report = {
            str(fold): train_model_c_long_fold(
                args.train_dir, args.parent_members, args.out, fold, args.gap
            )
            for fold in args.folds
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
