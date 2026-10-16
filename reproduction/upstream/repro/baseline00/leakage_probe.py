from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from repro.config import ROLLING_TOP50, TRAIN_DIR, WORK, ensure_data2_path
from repro.baseline00.metrics import weighted_zero_mean_r2


DEFAULT_GAPS = (0, 50, 100, 200, 500)
DEFAULT_ACF_LAGS = (1, 2, 5, 10, 20, 50, 100, 200, 500)


def _finite_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3:
        return 0.0
    left = left[finite] - left[finite].mean()
    right = right[finite] - right[finite].mean()
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    if denominator <= 1e-18:
        return 0.0
    return float(np.dot(left, right) / denominator)


def target_acf_by_asset(
    frame: pd.DataFrame, lags: Sequence[int] = DEFAULT_ACF_LAGS
) -> dict[str, Any]:
    required = {"time_id", "asset_id", "target"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"ACF frame missing columns: {missing}")
    ordered = frame.sort_values(["asset_id", "time_id"], kind="stable")
    per_asset: dict[str, dict[str, float]] = {}
    for asset_id, group in ordered.groupby("asset_id", sort=True):
        target = group["target"].to_numpy(dtype=np.float64, copy=False)
        values: dict[str, float] = {}
        for lag in lags:
            values[str(lag)] = (
                _finite_correlation(target[:-lag], target[lag:]) if target.size > lag else 0.0
            )
        per_asset[str(int(asset_id))] = values
    median_abs = {
        str(lag): float(
            np.median([abs(values[str(lag)]) for values in per_asset.values()])
        )
        for lag in lags
    }
    mean = {
        str(lag): float(np.mean([values[str(lag)] for values in per_asset.values()]))
        for lag in lags
    }
    return {"lags": list(lags), "per_asset": per_asset, "mean": mean, "median_abs": median_abs}


def _read_time_range(
    path: Path,
    columns: Sequence[str],
    min_time_id: int | None = None,
    max_time_id_exclusive: int | None = None,
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for batch in pq.ParquetFile(path).iter_batches(columns=list(columns), batch_size=150_000):
        frame = batch.to_pandas()
        mask = np.ones(len(frame), dtype=bool)
        if min_time_id is not None:
            mask &= frame["time_id"].to_numpy() >= min_time_id
        if max_time_id_exclusive is not None:
            mask &= frame["time_id"].to_numpy() < max_time_id_exclusive
        if mask.any():
            selected.append(frame.loc[mask])
        if (
            max_time_id_exclusive is not None
            and int(frame["time_id"].max()) >= max_time_id_exclusive
        ):
            break
    if not selected:
        raise ValueError(f"no rows selected from {path}")
    return pd.concat(selected, ignore_index=True).sort_values(
        ["time_id", "asset_id"], kind="stable", ignore_index=True
    )


def select_gap(
    fold_results: dict[str, Any],
    acf_results: dict[str, Any],
    gaps: Sequence[int] = DEFAULT_GAPS,
    score_tolerance: float = 0.00010,
    correlation_threshold: float = 0.9995,
    acf_threshold: float = 0.01,
) -> dict[str, Any]:
    gaps = tuple(sorted(int(gap) for gap in gaps))
    plateau_gap = gaps[-1]
    for candidate in gaps:
        stable = True
        for fold in fold_results.values():
            base_score = float(fold["gaps"][str(candidate)]["score"])
            for later in (gap for gap in gaps if gap >= candidate):
                later_score = float(fold["gaps"][str(later)]["score"])
                corr_key = f"{candidate}:{later}"
                correlation = float(fold["prediction_correlations"][corr_key])
                if abs(later_score - base_score) > score_tolerance or correlation < correlation_threshold:
                    stable = False
                    break
            if not stable:
                break
        if stable:
            plateau_gap = candidate
            break

    aggregate_acf = {
        lag: float(
            np.median(
                [fold["median_abs"][str(lag)] for fold in acf_results.values()]
            )
        )
        for lag in DEFAULT_ACF_LAGS
    }
    acf_lag = DEFAULT_ACF_LAGS[-1]
    for lag in DEFAULT_ACF_LAGS:
        later_values = [value for later, value in aggregate_acf.items() if later >= lag]
        if later_values and max(later_values) <= acf_threshold:
            acf_lag = lag
            break
    acf_gap = next((gap for gap in gaps if gap >= acf_lag), gaps[-1])
    selected = max(plateau_gap, acf_gap)
    return {
        "selected_gap_time_ids": int(selected),
        "plateau_gap_time_ids": int(plateau_gap),
        "acf_gap_time_ids": int(acf_gap),
        "acf_decay_lag": int(acf_lag),
        "aggregate_median_abs_acf": {str(k): v for k, v in aggregate_acf.items()},
        "score_tolerance": score_tolerance,
        "prediction_correlation_threshold": correlation_threshold,
        "acf_threshold": acf_threshold,
        "rule": "max(score/prediction plateau lower bound, target ACF decay lower bound)",
    }


def run_probe(
    train_dir: Path,
    output: Path,
    gaps: Sequence[int] = DEFAULT_GAPS,
    train_window_time_ids: int = 50_000,
    valid_window_time_ids: int = 20_000,
    n_estimators: int = 100,
) -> dict[str, Any]:
    import lightgbm as lgb

    started = time.perf_counter()
    output = ensure_data2_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_columns = list(ROLLING_TOP50)
    columns = ["time_id", "asset_id", "weight", "target", *feature_columns]
    fold_results: dict[str, Any] = {}
    acf_results: dict[str, Any] = {}

    for fold in (5, 6, 7):
        valid_min = fold * 100_000
        train_start = valid_min - train_window_time_ids
        valid_end = valid_min + valid_window_time_ids
        train_frame = _read_time_range(
            train_dir / f"train_partition_{fold - 1:03d}.parquet",
            columns,
            min_time_id=train_start,
            max_time_id_exclusive=valid_min,
        )
        valid_frame = _read_time_range(
            train_dir / f"train_partition_{fold:03d}.parquet",
            columns,
            min_time_id=valid_min,
            max_time_id_exclusive=valid_end,
        )
        acf_results[str(fold)] = target_acf_by_asset(valid_frame)

        x_train = np.nan_to_num(
            train_frame.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        y_train = train_frame["target"].to_numpy(dtype=np.float64, copy=False)
        w_train = np.maximum(
            train_frame["weight"].to_numpy(dtype=np.float64, copy=False), 0.0
        )
        train_time = train_frame["time_id"].to_numpy(copy=False)
        x_valid = np.nan_to_num(
            valid_frame.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        y_valid = valid_frame["target"].to_numpy(dtype=np.float64, copy=False)
        w_valid = np.maximum(
            valid_frame["weight"].to_numpy(dtype=np.float64, copy=False), 0.0
        )
        predictions: dict[int, np.ndarray] = {}
        gap_results: dict[str, Any] = {}
        for gap in gaps:
            mask = train_time < valid_min - int(gap)
            model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                num_leaves=31,
                learning_rate=0.05,
                min_data_in_leaf=1_000,
                feature_fraction=0.85,
                bagging_fraction=0.85,
                bagging_freq=1,
                reg_lambda=1.0,
                n_jobs=32,
                verbosity=-1,
                random_state=0,
                deterministic=True,
                force_col_wise=True,
            )
            model.fit(x_train[mask], y_train[mask], sample_weight=w_train[mask])
            prediction = model.predict(x_valid)
            predictions[int(gap)] = prediction
            gap_results[str(gap)] = {
                "train_rows": int(mask.sum()),
                "train_max_time_id": int(train_time[mask].max()),
                "score": weighted_zero_mean_r2(y_valid, prediction, w_valid),
                "prediction_std": float(np.std(prediction)),
            }
            del model
            gc.collect()

        correlations = {
            f"{left}:{right}": _finite_correlation(predictions[left], predictions[right])
            for left in gaps
            for right in gaps
            if right >= left
        }
        fold_results[str(fold)] = {
            "train_partition": fold - 1,
            "valid_partition": fold,
            "train_time_range": [int(train_time.min()), int(train_time.max())],
            "valid_time_range": [
                int(valid_frame["time_id"].min()),
                int(valid_frame["time_id"].max()),
            ],
            "valid_rows": int(len(valid_frame)),
            "gaps": gap_results,
            "prediction_correlations": correlations,
        }
        del train_frame, valid_frame, x_train, x_valid, y_train, y_valid, w_train, w_valid
        del predictions
        gc.collect()

    selection = select_gap(fold_results, acf_results, gaps=gaps)
    report = {
        "status": "ok",
        "purpose": "probe-only; no probe model or score is used by production",
        "fold8_read": False,
        "feature_columns": feature_columns,
        "gaps": list(gaps),
        "train_window_time_ids": train_window_time_ids,
        "valid_window_time_ids": valid_window_time_ids,
        "n_estimators": n_estimators,
        "folds": fold_results,
        "target_acf": acf_results,
        "selection": selection,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--out", type=Path, default=WORK / "leakage_probe.json")
    parser.add_argument("--train-window-time-ids", type=int, default=50_000)
    parser.add_argument("--valid-window-time-ids", type=int, default=20_000)
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()
    report = run_probe(
        args.train_dir,
        args.out,
        train_window_time_ids=args.train_window_time_ids,
        valid_window_time_ids=args.valid_window_time_ids,
        n_estimators=args.n_estimators,
    )
    print(json.dumps(report["selection"], indent=2))


if __name__ == "__main__":
    main()
