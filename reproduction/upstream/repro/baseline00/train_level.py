from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from repro.config import FEATS, TRAIN_DIR, WORK, ensure_data2_path
from repro.baseline00.frozen_recipe import validate_frozen_recipe
from repro.baseline00.metrics import prediction_summary, weighted_zero_mean_r2
from repro.baseline00.training_io import (
    aggregate_features_by_time,
    broadcast_group_values,
    load_prediction_frame,
    load_training_arrays,
    load_validation_frame,
)


def _level_training_data(
    train_dir: Path, fold: int, cutoff: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    feature_means: list[np.ndarray] = []
    target_means: list[np.ndarray] = []
    group_weights: list[np.ndarray] = []
    rows_by_partition: dict[str, int] = {}
    columns = ["time_id", "weight", "target", *FEATS]
    for partition in range(fold):
        frame = pd.read_parquet(
            train_dir / f"train_partition_{partition:03d}.parquet", columns=columns
        ).sort_values("time_id", kind="stable", ignore_index=True)
        if partition == fold - 1:
            frame = frame.loc[frame["time_id"] < cutoff].reset_index(drop=True)
        x = np.nan_to_num(
            frame.loc[:, FEATS].to_numpy(dtype=np.float32, copy=True),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        _, x_mean, y_mean, weight_sum, _ = aggregate_features_by_time(
            frame["time_id"].to_numpy(copy=False),
            x,
            frame["weight"].to_numpy(dtype=np.float32, copy=False),
            frame["target"].to_numpy(dtype=np.float64, copy=False),
        )
        assert y_mean is not None
        feature_means.append(x_mean)
        target_means.append(y_mean)
        group_weights.append(weight_sum)
        rows_by_partition[str(partition)] = int(len(frame))
        del frame, x, x_mean, y_mean, weight_sum
        gc.collect()
    return (
        np.concatenate(feature_means),
        np.concatenate(target_means),
        np.concatenate(group_weights),
        rows_by_partition,
    )


def train_fold(
    train_dir: Path,
    output_root: Path,
    fold: int,
    gap: int = 500,
    frozen_recipe: Path | None = None,
) -> dict:
    import lightgbm as lgb

    started = time.perf_counter()
    output_root = ensure_data2_path(output_root)
    out_dir = output_root / f"fold_{fold}" / "level"
    out_dir.mkdir(parents=True, exist_ok=True)
    held_out = frozen_recipe is not None and fold == 8
    valid = (
        load_prediction_frame(train_dir, fold, FEATS)
        if held_out
        else load_validation_frame(train_dir, fold, FEATS)
    )
    valid_min = int(valid["time_id"].min())
    cutoff = valid_min - gap

    weight_partitions = list(range(max(0, fold - 3), fold))
    x_weight, y_weight, _, weight_rows = load_training_arrays(
        train_dir,
        weight_partitions,
        FEATS,
        cutoff_time_id=cutoff,
        label_mode="weight",
    )
    weight_model = lgb.LGBMRegressor(
        n_estimators=400,
        num_leaves=63,
        learning_rate=0.05,
        min_data_in_leaf=2_000,
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
    weight_model.fit(x_weight, y_weight)
    weight_model.booster_.save_model(str(out_dir / "weight_model.txt"))
    del x_weight, y_weight
    gc.collect()

    level_x, level_y, level_w, level_rows = _level_training_data(train_dir, fold, cutoff)
    level_model = lgb.LGBMRegressor(
        n_estimators=60,
        num_leaves=31,
        learning_rate=0.03,
        min_data_in_leaf=200,
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
    level_model.fit(level_x, level_y, sample_weight=level_w)
    level_model.booster_.save_model(str(out_dir / "level_model.txt"))

    x_valid = np.nan_to_num(
        valid.loc[:, FEATS].to_numpy(dtype=np.float32, copy=True),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    predicted_weight = np.maximum(weight_model.predict(x_valid), 0.0).astype(np.float32)
    _, valid_feature_mean, _, _, counts = aggregate_features_by_time(
        valid["time_id"].to_numpy(copy=False), x_valid, predicted_weight
    )
    group_level = level_model.predict(valid_feature_mean)
    level_prediction = broadcast_group_values(group_level, counts).astype(np.float64)
    np.save(out_dir / "level.npy", level_prediction)
    report = {
        "fold": fold,
        "gap_time_ids": gap,
        "weight_partitions": weight_partitions,
        "weight_rows_by_partition": weight_rows,
        "weight_train_rows": int(sum(weight_rows.values())),
        "level_rows_by_partition": level_rows,
        "level_train_time_rows": int(len(level_y)),
        "valid_rows": int(len(valid)),
        "prediction": prediction_summary(level_prediction),
        "elapsed_seconds": time.perf_counter() - started,
        "recipe_note": "weight window is causal recent-3; original artifact metadata was absent",
        "frozen_recipe": str(frozen_recipe) if frozen_recipe else None,
        "held_out_labels_read": False if held_out else None,
    }
    if not held_out:
        report["valid_score"] = weighted_zero_mean_r2(
            valid["target"].to_numpy(dtype=np.float64, copy=False),
            level_prediction,
            np.maximum(valid["weight"].to_numpy(dtype=np.float64, copy=False), 0.0),
        )
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    del level_x, level_y, level_w, x_valid, valid, weight_model, level_model
    gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--gap", type=int, default=500)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--out", type=Path, default=WORK / "folds")
    parser.add_argument("--frozen-recipe", type=Path)
    args = parser.parse_args()
    if args.frozen_recipe:
        recipe = validate_frozen_recipe(
            args.frozen_recipe, {"weight_model": 400, "level_model": 60}
        )
        recipe_gap = int(recipe["purge_gap_time_ids"])
        if args.gap != recipe_gap:
            raise ValueError(f"--gap {args.gap} conflicts with frozen recipe gap {recipe_gap}")
    for fold in args.folds:
        print(
            json.dumps(
                train_fold(
                    args.train_dir,
                    args.out,
                    fold,
                    args.gap,
                    args.frozen_recipe,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
