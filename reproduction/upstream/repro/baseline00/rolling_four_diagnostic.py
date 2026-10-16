from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from repro.baseline00.metrics import prediction_summary, weighted_zero_mean_r2
from repro.baseline00.rolling import V1_OPERATIONS
from repro.baseline00.train_rolling import _features_with_history
from repro.config import ROLLING_TOP50, TRAIN_DIR, WORK, ensure_data2_path


FOUR_OPERATIONS = ("lag1", "roll_mean5", "roll_mean10", "delta1")


def four_column_indices(feature_count: int) -> np.ndarray:
    blocks = []
    for operation in FOUR_OPERATIONS:
        position = V1_OPERATIONS.index(operation)
        blocks.append(
            np.arange(position * feature_count, (position + 1) * feature_count)
        )
    return np.concatenate(blocks).astype(np.int64)


def train_fold(train_dir: Path, folds_root: Path, output_root: Path, fold: int) -> dict:
    import lightgbm as lgb

    started = time.perf_counter()
    folds_root = ensure_data2_path(folds_root)
    output_root = ensure_data2_path(output_root)
    output_dir = output_root / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    six_dir = folds_root / f"fold_{fold}" / "rolling"
    train_features = np.load(six_dir / "cache" / "train_features.npy", mmap_mode="r")
    train_y = np.load(six_dir / "cache" / "train_y.npy", mmap_mode="r")
    train_w = np.load(six_dir / "cache" / "train_w.npy", mmap_mode="r")
    selected_columns = four_column_indices(len(ROLLING_TOP50))

    history_columns = ["row_id", "time_id", "asset_id", *ROLLING_TOP50]
    history = pd.read_parquet(
        train_dir / f"train_partition_{fold - 1:03d}.parquet",
        columns=history_columns,
    ).sort_values(["time_id", "asset_id"], kind="stable", ignore_index=True)
    history = (
        history.groupby("asset_id", sort=False, group_keys=False)
        .tail(9)
        .sort_values(["time_id", "asset_id"], kind="stable", ignore_index=True)
    )
    valid_columns = [
        "row_id",
        "time_id",
        "asset_id",
        "weight",
        "target",
        *ROLLING_TOP50,
    ]
    valid = pd.read_parquet(
        train_dir / f"train_partition_{fold:03d}.parquet", columns=valid_columns
    ).sort_values(["time_id", "asset_id"], kind="stable", ignore_index=True)
    valid_six, valid_ordered, _ = _features_with_history(
        valid, history, list(ROLLING_TOP50)
    )
    valid_four = np.ascontiguousarray(valid_six[:, selected_columns])
    train_four = np.ascontiguousarray(train_features[:, selected_columns])
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
    model.fit(train_four, train_y, sample_weight=train_w)
    prediction = model.predict(valid_four).astype(np.float64)
    y_valid = valid_ordered["target"].to_numpy(dtype=np.float64, copy=False)
    w_valid = np.maximum(
        valid_ordered["weight"].to_numpy(dtype=np.float64, copy=False), 0.0
    )
    row_id = valid_ordered["row_id"].to_numpy(dtype=np.int64, copy=True)
    six_report = json.loads((six_dir / "report.json").read_text(encoding="utf-8"))
    four_score = weighted_zero_mean_r2(y_valid, prediction, w_valid)
    six_score = six_report.get("valid_score")
    if six_score is None:
        six_prediction = np.load(six_dir / "rolling_v1.npy", mmap_mode="r")
        six_row_id = np.load(six_dir / "row_id.npy", mmap_mode="r")
        if len(six_prediction) != len(y_valid):
            raise ValueError(f"fold {fold} six-operation prediction rows differ")
        if not np.array_equal(six_row_id, row_id):
            raise ValueError(f"fold {fold} six-operation row_id differs")
        six_score = weighted_zero_mean_r2(y_valid, six_prediction, w_valid)
    model.booster_.save_model(str(output_dir / "rolling_four.txt"))
    np.save(output_dir / "rolling_four.npy", prediction)
    np.save(output_dir / "row_id.npy", row_id)
    report = {
        "fold": fold,
        "feature_columns": list(ROLLING_TOP50),
        "operations": list(FOUR_OPERATIONS),
        "rolling_columns": int(len(selected_columns)),
        "train_rows": int(len(train_y)),
        "valid_rows": int(len(y_valid)),
        "four_score": four_score,
        "six_score": float(six_score),
        "four_minus_six": four_score - float(six_score),
        "row_id_first": int(row_id[0]),
        "row_id_last": int(row_id[-1]),
        "row_aligned": True,
        "prediction": prediction_summary(prediction),
        "elapsed_seconds": time.perf_counter() - started,
        "diagnostic_only": True,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    del train_four, valid_four, valid_six, valid, valid_ordered, model
    gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=[5, 6, 7, 8])
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--folds-root", type=Path, default=WORK / "folds")
    parser.add_argument("--out", type=Path, default=WORK / "rolling_four")
    args = parser.parse_args()
    reports = [
        train_fold(args.train_dir, args.folds_root, args.out, fold)
        for fold in args.folds
    ]
    summary = {
        "folds": reports,
        "mean_four_minus_six": float(
            np.mean([report["four_minus_six"] for report in reports])
        ),
        "diagnostic_only": True,
        "fold8_read": 8 in args.folds,
    }
    output = ensure_data2_path(args.out) / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
