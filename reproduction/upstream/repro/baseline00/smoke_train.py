from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from repro.config import FEATS, TRAIN_DIR, WORK, ensure_data2_path
from repro.baseline00.metrics import prediction_summary, weighted_zero_mean_r2
from repro.baseline00.rolling import RollingFeatureState, build_rolling_features


def read_time_prefix(path: Path, max_time_ids: int) -> pd.DataFrame:
    columns = ["row_id", "time_id", "asset_id", "weight", "target", *FEATS]
    batches: list[pd.DataFrame] = []
    first_time: int | None = None
    cutoff: int | None = None
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=100_000):
        part = batch.to_pandas()
        if first_time is None:
            first_time = int(part["time_id"].min())
            cutoff = first_time + max_time_ids
        assert cutoff is not None
        selected = part.loc[part["time_id"] < cutoff]
        if not selected.empty:
            batches.append(selected)
        if int(part["time_id"].max()) >= cutoff:
            break
    if not batches:
        raise ValueError("smoke data prefix is empty")
    return pd.concat(batches, ignore_index=True).sort_values(
        ["time_id", "asset_id"], kind="stable", ignore_index=True
    )


def check_streaming_rolling(frame: pd.DataFrame) -> float:
    feature_columns = FEATS[:3]
    sample_times = frame["time_id"].drop_duplicates().iloc[:20]
    sample = frame.loc[frame["time_id"].isin(sample_times), ["time_id", "asset_id", *feature_columns]]
    sample = sample.sort_values(["time_id", "asset_id"], kind="stable", ignore_index=True)
    offline = build_rolling_features(sample, feature_columns)
    state = RollingFeatureState(tuple(feature_columns))
    streaming = np.vstack(
        [state.transform_batch(group) for _, group in sample.groupby("time_id", sort=True)]
    )
    difference = float(np.max(np.abs(offline - streaming)))
    if difference > 1e-6:
        raise AssertionError(f"offline/streaming rolling mismatch: {difference}")
    return difference


def run_smoke(data_path: Path, output_dir: Path, max_time_ids: int = 5_000) -> dict:
    started = time.perf_counter()
    output_dir = ensure_data2_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = read_time_prefix(data_path, max_time_ids=max_time_ids)
    unique_times = frame["time_id"].drop_duplicates().to_numpy()
    split_at = unique_times[max(1, int(len(unique_times) * 0.8))]
    train = frame["time_id"].to_numpy() < split_at
    valid = ~train
    if not train.any() or not valid.any():
        raise ValueError("smoke split produced an empty train or validation set")

    x = frame.loc[:, FEATS].to_numpy(dtype=np.float32, copy=True)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = frame["target"].to_numpy(dtype=np.float64, copy=False)
    weight = np.maximum(frame["weight"].to_numpy(dtype=np.float64, copy=False), 0.0)
    model = lgb.LGBMRegressor(
        n_estimators=50,
        num_leaves=31,
        learning_rate=0.05,
        min_data_in_leaf=100,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        reg_lambda=1.0,
        n_jobs=16,
        verbosity=-1,
        random_state=0,
    )
    model.fit(x[train], y[train], sample_weight=weight[train])
    prediction = model.predict(x[valid])
    score = weighted_zero_mean_r2(y[valid], prediction, weight[valid])

    model_path = output_dir / "smoke_lgb.txt"
    model.booster_.save_model(str(model_path))
    reloaded = lgb.Booster(model_file=str(model_path))
    reloaded_prediction = reloaded.predict(x[valid])
    reload_max_abs_diff = float(np.max(np.abs(prediction - reloaded_prediction)))
    if reload_max_abs_diff > 1e-10:
        raise AssertionError(f"saved booster prediction changed: {reload_max_abs_diff}")

    rolling_max_abs_diff = check_streaming_rolling(frame)
    peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    report = {
        "status": "ok",
        "data_path": str(data_path),
        "rows": int(len(frame)),
        "train_rows": int(train.sum()),
        "valid_rows": int(valid.sum()),
        "train_max_time_id": int(frame.loc[train, "time_id"].max()),
        "valid_min_time_id": int(frame.loc[valid, "time_id"].min()),
        "score": score,
        "reload_max_abs_diff": reload_max_abs_diff,
        "rolling_max_abs_diff": rolling_max_abs_diff,
        "prediction": prediction_summary(prediction),
        "peak_rss_gib": peak_gib,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if report["prediction"]["finite_fraction"] != 1.0:
        raise AssertionError("smoke predictions contain non-finite values")
    if peak_gib >= 40.0:
        raise AssertionError(f"smoke peak memory exceeded 40 GiB: {peak_gib}")
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=TRAIN_DIR / "train_partition_000.parquet")
    parser.add_argument("--out", type=Path, default=WORK / "smoke")
    parser.add_argument("--max-time-ids", type=int, default=5_000)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.data, args.out, args.max_time_ids), indent=2))


if __name__ == "__main__":
    main()
