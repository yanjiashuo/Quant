from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from repro.config import ROLLING_TOP50, TRAIN_DIR, WORK, ensure_data2_path
from repro.baseline00.frozen_recipe import validate_frozen_recipe
from repro.baseline00.metrics import prediction_summary, weighted_zero_mean_r2
from repro.baseline00.rolling import V1_OPERATIONS, build_rolling_features


def _selected_rows(path: Path, cutoff: int | None) -> int:
    rows = pq.ParquetFile(path).metadata.num_rows
    if cutoff is None:
        return int(rows)
    time_id = pq.read_table(path, columns=["time_id"])["time_id"].to_numpy()
    return int(np.count_nonzero(time_id < cutoff))


def _features_with_history(
    frame: pd.DataFrame, history: pd.DataFrame | None, feature_columns: list[str]
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    current = frame.copy()
    current["_current"] = True
    if history is not None and not history.empty:
        previous = history.copy()
        previous["_current"] = False
        combined = pd.concat([previous, current], ignore_index=True)
    else:
        combined = current
    combined = combined.sort_values(["time_id", "asset_id"], kind="stable", ignore_index=True)
    matrix = build_rolling_features(combined, feature_columns)
    current_mask = combined["_current"].to_numpy(dtype=bool, copy=False)
    current_frame = combined.loc[current_mask].drop(columns=["_current"]).reset_index(drop=True)
    current_matrix = matrix[current_mask]
    next_history = (
        combined.drop(columns=["_current"])
        .groupby("asset_id", sort=False, group_keys=False)
        .tail(9)
        .sort_values(["time_id", "asset_id"], kind="stable", ignore_index=True)
    )
    return current_matrix, current_frame, next_history


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
    out_dir = output_root / f"fold_{fold}" / "rolling"
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = list(ROLLING_TOP50)
    held_out = frozen_recipe is not None and fold == 8
    train_columns = ["row_id", "time_id", "asset_id", "weight", "target", *feature_columns]
    valid_columns = (
        ["row_id", "time_id", "asset_id", *feature_columns]
        if held_out
        else train_columns
    )
    valid_path = train_dir / f"train_partition_{fold:03d}.parquet"
    valid_min = int(pq.read_table(valid_path, columns=["time_id"])["time_id"][0].as_py())
    cutoff = valid_min - gap
    train_rows = sum(
        _selected_rows(
            train_dir / f"train_partition_{partition:03d}.parquet",
            cutoff if partition == fold - 1 else None,
        )
        for partition in range(fold)
    )
    n_columns = len(feature_columns) * len(V1_OPERATIONS)
    train_features = np.lib.format.open_memmap(
        cache_dir / "train_features.npy", mode="w+", dtype=np.float32, shape=(train_rows, n_columns)
    )
    train_y = np.lib.format.open_memmap(
        cache_dir / "train_y.npy", mode="w+", dtype=np.float64, shape=(train_rows,)
    )
    train_w = np.lib.format.open_memmap(
        cache_dir / "train_w.npy", mode="w+", dtype=np.float64, shape=(train_rows,)
    )
    cursor = 0
    history: pd.DataFrame | None = None
    rows_by_partition: dict[str, int] = {}
    for partition in range(fold):
        frame = pd.read_parquet(
            train_dir / f"train_partition_{partition:03d}.parquet", columns=train_columns
        ).sort_values(["time_id", "asset_id"], kind="stable", ignore_index=True)
        matrix, ordered, history = _features_with_history(frame, history, feature_columns)
        selected = ordered["time_id"].to_numpy(copy=False) < cutoff if partition == fold - 1 else np.ones(len(ordered), dtype=bool)
        count = int(selected.sum())
        train_features[cursor : cursor + count] = matrix[selected]
        train_y[cursor : cursor + count] = ordered.loc[selected, "target"].to_numpy(dtype=np.float64)
        train_w[cursor : cursor + count] = np.maximum(
            ordered.loc[selected, "weight"].to_numpy(dtype=np.float64), 0.0
        )
        cursor += count
        rows_by_partition[str(partition)] = count
        del frame, matrix, ordered
        gc.collect()
    if cursor != train_rows:
        raise AssertionError(f"rolling cache rows differ: {cursor} != {train_rows}")
    train_features.flush()
    train_y.flush()
    train_w.flush()

    valid_frame = pd.read_parquet(valid_path, columns=valid_columns).sort_values(
        ["time_id", "asset_id"], kind="stable", ignore_index=True
    )
    valid_features, valid_ordered, _ = _features_with_history(valid_frame, history, feature_columns)
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
    model.fit(train_features, train_y, sample_weight=train_w)
    prediction = model.predict(valid_features).astype(np.float64)
    model.booster_.save_model(str(out_dir / "rolling_v1.txt"))
    np.save(out_dir / "rolling_v1.npy", prediction)
    np.save(out_dir / "row_id.npy", valid_ordered["row_id"].to_numpy(dtype=np.int64))
    report = {
        "fold": fold,
        "gap_time_ids": gap,
        "feature_columns": feature_columns,
        "operations": list(V1_OPERATIONS),
        "rolling_columns": n_columns,
        "rows_by_partition": rows_by_partition,
        "train_rows": train_rows,
        "valid_rows": int(len(valid_ordered)),
        "prediction": prediction_summary(prediction),
        "elapsed_seconds": time.perf_counter() - started,
        "causal_note": "purged labels excluded from fit; their already-observed features remain in validation history",
        "frozen_recipe": str(frozen_recipe) if frozen_recipe else None,
        "held_out_labels_read": False if held_out else None,
    }
    if not held_out:
        report["valid_score"] = weighted_zero_mean_r2(
            valid_ordered["target"].to_numpy(dtype=np.float64),
            prediction,
            np.maximum(valid_ordered["weight"].to_numpy(dtype=np.float64), 0.0),
        )
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    del train_features, train_y, train_w, valid_features, valid_frame, valid_ordered, model
    gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--gap", type=int, default=500)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--out", type=Path, default=WORK / "folds")
    parser.add_argument("--frozen-recipe", type=Path)
    parser.add_argument("--ops", nargs="+", default=list(V1_OPERATIONS))
    args = parser.parse_args()
    if tuple(args.ops) != V1_OPERATIONS:
        raise ValueError(
            f"Baseline 00 rolling operations must be {list(V1_OPERATIONS)}, got {args.ops}"
        )
    if args.frozen_recipe:
        recipe = validate_frozen_recipe(args.frozen_recipe, {"rolling_v1": 200})
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
