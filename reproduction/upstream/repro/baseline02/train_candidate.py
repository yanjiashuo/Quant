from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from repro.baseline00.metrics import prediction_summary
from repro.baseline00.training_io import weighted_time_mean_rows
from repro.baseline02.config import (
    CS_RANK_LGB,
    CS_RANK_WINDOW,
    OOF_FILENAMES,
    ROLLING_V2_LGB,
)
from repro.baseline02.cs_rank import build_cs_rank
from repro.baseline02.rolling_v2 import V2_OPERATIONS, build_rolling_v2
from repro.config import FEATS, ROLLING_TOP50, TEST_DIR, TRAIN_DIR, ensure_data2_path


def recent_partitions(stop: int, window: int) -> list[int]:
    return list(range(max(0, int(stop) - int(window)), int(stop)))


def _partition_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _load_rank_frame(path: Path, labels: bool) -> pd.DataFrame:
    columns = ["row_id", "time_id", "asset_id", *FEATS]
    if labels:
        columns[3:3] = ["weight", "target"]
    return pd.read_parquet(path, columns=columns).sort_values(
        ["time_id", "asset_id"], kind="stable", ignore_index=True
    )


def _build_cs_rank_training_cache(
    train_dir: Path,
    partitions: Sequence[int],
    cache_dir: Path,
) -> tuple[np.memmap, np.memmap, np.memmap, dict[str, int]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = sum(
        _partition_rows(Path(train_dir) / f"train_partition_{partition:03d}.parquet")
        for partition in partitions
    )
    x = np.lib.format.open_memmap(
        cache_dir / "features.npy", mode="w+", dtype=np.float32, shape=(rows, len(FEATS))
    )
    y = np.lib.format.open_memmap(
        cache_dir / "target.npy", mode="w+", dtype=np.float64, shape=(rows,)
    )
    w = np.lib.format.open_memmap(
        cache_dir / "weight.npy", mode="w+", dtype=np.float64, shape=(rows,)
    )
    cursor = 0
    rows_by_partition: dict[str, int] = {}
    for partition in partitions:
        path = Path(train_dir) / f"train_partition_{partition:03d}.parquet"
        frame = _load_rank_frame(path, labels=True)
        count = len(frame)
        raw_weight = np.maximum(
            frame["weight"].to_numpy(dtype=np.float64, copy=False), 0.0
        )
        target = frame["target"].to_numpy(dtype=np.float64, copy=False)
        level = weighted_time_mean_rows(
            frame["time_id"].to_numpy(dtype=np.int64, copy=False), target, raw_weight
        )
        x[cursor : cursor + count] = build_cs_rank(frame, FEATS)
        y[cursor : cursor + count] = target - level
        w[cursor : cursor + count] = raw_weight
        cursor += count
        rows_by_partition[str(partition)] = count
        del frame, raw_weight, target, level
        gc.collect()
    if cursor != rows:
        raise AssertionError(f"CS-rank cache rows differ: {cursor} != {rows}")
    x.flush()
    y.flush()
    w.flush()
    return x, y, w, rows_by_partition


def _fit_cs_rank(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        **CS_RANK_LGB,
        n_jobs=min(64, max(1, __import__("os").cpu_count() or 1)),
        verbosity=-1,
    )
    model.fit(x, y, sample_weight=w)
    return model


def _rolling_v2_with_history(
    frame: pd.DataFrame, history: pd.DataFrame | None
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    current = frame.copy()
    current["_current"] = True
    if history is not None and not history.empty:
        previous = history.copy()
        previous["_current"] = False
        combined = pd.concat((previous, current), ignore_index=True)
    else:
        combined = current
    combined = combined.sort_values(
        ["time_id", "asset_id"], kind="stable", ignore_index=True
    )
    matrix = build_rolling_v2(combined, ROLLING_TOP50)
    selected = combined["_current"].to_numpy(dtype=bool, copy=False)
    ordered = combined.loc[selected].drop(columns="_current").reset_index(drop=True)
    next_history = (
        combined.drop(columns="_current")
        .groupby("asset_id", sort=False, group_keys=False)
        .tail(39)
        .sort_values(["time_id", "asset_id"], kind="stable", ignore_index=True)
    )
    return matrix[selected], ordered, next_history


def _build_rolling_v2_training_cache(
    train_dir: Path,
    partitions: Sequence[int],
    cache_dir: Path,
) -> tuple[np.memmap, np.memmap, np.memmap, pd.DataFrame, dict[str, int]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = sum(
        _partition_rows(Path(train_dir) / f"train_partition_{partition:03d}.parquet")
        for partition in partitions
    )
    columns = len(ROLLING_TOP50) * len(V2_OPERATIONS)
    x = np.lib.format.open_memmap(
        cache_dir / "features.npy", mode="w+", dtype=np.float32, shape=(rows, columns)
    )
    y = np.lib.format.open_memmap(
        cache_dir / "target.npy", mode="w+", dtype=np.float64, shape=(rows,)
    )
    w = np.lib.format.open_memmap(
        cache_dir / "weight.npy", mode="w+", dtype=np.float64, shape=(rows,)
    )
    cursor = 0
    history: pd.DataFrame | None = None
    rows_by_partition: dict[str, int] = {}
    train_columns = [
        "row_id",
        "time_id",
        "asset_id",
        "weight",
        "target",
        *ROLLING_TOP50,
    ]
    for partition in partitions:
        path = Path(train_dir) / f"train_partition_{partition:03d}.parquet"
        frame = pd.read_parquet(path, columns=train_columns).sort_values(
            ["time_id", "asset_id"], kind="stable", ignore_index=True
        )
        matrix, ordered, history = _rolling_v2_with_history(frame, history)
        count = len(ordered)
        x[cursor : cursor + count] = matrix
        y[cursor : cursor + count] = ordered["target"].to_numpy(dtype=np.float64)
        w[cursor : cursor + count] = np.maximum(
            ordered["weight"].to_numpy(dtype=np.float64), 0.0
        )
        cursor += count
        rows_by_partition[str(partition)] = count
        del frame, matrix, ordered
        gc.collect()
    if cursor != rows or history is None:
        raise AssertionError(f"Rolling V2 cache rows differ: {cursor} != {rows}")
    x.flush()
    y.flush()
    w.flush()
    return x, y, w, history, rows_by_partition


def _fit_rolling_v2(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        **ROLLING_V2_LGB,
        n_jobs=min(64, max(1, __import__("os").cpu_count() or 1)),
        verbosity=-1,
    )
    model.fit(x, y, sample_weight=w)
    return model


def _fold_index_payload(
    frame: pd.DataFrame, parent_oof_path: Path | None = None
) -> dict[str, np.ndarray]:
    time_id = frame["time_id"].to_numpy(dtype=np.int64, copy=False)
    asset_id = frame["asset_id"].to_numpy(dtype=np.int64, copy=False)
    row_id = np.arange(len(frame), dtype=np.int64)
    if parent_oof_path is not None:
        with np.load(parent_oof_path, allow_pickle=False) as parent:
            parent_row = np.asarray(parent["row_id"], dtype=np.int64)
            parent_time = np.asarray(parent["time_id"], dtype=np.int64)
            parent_asset = np.asarray(parent["asset_id"], dtype=np.int64)
        if not np.array_equal(time_id, parent_time):
            raise ValueError("candidate time_id differs from frozen parent OOF")
        if not np.array_equal(asset_id, parent_asset):
            raise ValueError("candidate asset_id differs from frozen parent OOF")
        row_id = parent_row
    return {
        "row_id": row_id,
        "source_row_id": frame["row_id"].to_numpy(dtype=np.int64, copy=False),
        "time_id": time_id,
        "asset_id": asset_id,
    }


def normalize_existing_candidate_fold(
    train_dir: Path,
    output_root: Path,
    parent_oof_dir: Path,
    fold: int,
    candidate: str,
) -> dict[str, object]:
    path = Path(output_root) / f"fold_{fold}.npz"
    with np.load(path, allow_pickle=False) as data:
        if candidate not in data.files or "row_id" not in data.files:
            raise ValueError(f"{path} lacks row_id or {candidate}")
        source_row_id = np.asarray(
            data["source_row_id"] if "source_row_id" in data.files else data["row_id"],
            dtype=np.int64,
        )
        prediction = np.asarray(data[candidate], dtype=np.float64)
    index = pd.read_parquet(
        Path(train_dir) / f"train_partition_{fold:03d}.parquet",
        columns=["row_id", "time_id", "asset_id"],
    )
    if np.unique(source_row_id).size != source_row_id.size:
        raise ValueError(f"{path} source row_id is not unique")
    source = pd.Series(prediction, index=source_row_id)
    aligned = source.reindex(index["row_id"].to_numpy(dtype=np.int64))
    if aligned.isna().any():
        raise ValueError(f"{path} does not cover the official Fold{fold} rows")
    payload = _fold_index_payload(
        index, Path(parent_oof_dir) / OOF_FILENAMES[int(fold)]
    )
    payload[candidate] = aligned.to_numpy(dtype=np.float64)
    np.savez_compressed(path, **payload)
    return {
        "path": str(path),
        "fold": int(fold),
        "candidate": candidate,
        "rows": int(len(index)),
        "local_row_id": True,
        "source_row_id_preserved": True,
    }


def train_cs_rank_fold(
    train_dir: Path,
    output_root: Path,
    fold: int,
    parent_oof_dir: Path | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    output = ensure_data2_path(output_root)
    fold_dir = output / f"fold_{fold}"
    cache_dir = fold_dir / "cache"
    fold_dir.mkdir(parents=True, exist_ok=True)
    partitions = recent_partitions(fold, CS_RANK_WINDOW)
    x, y, w, rows_by_partition = _build_cs_rank_training_cache(
        train_dir, partitions, cache_dir
    )
    model = _fit_cs_rank(x, y, w)
    valid = _load_rank_frame(
        Path(train_dir) / f"train_partition_{fold:03d}.parquet", labels=False
    )
    valid_x = build_cs_rank(valid, FEATS)
    prediction = np.asarray(model.predict(valid_x), dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError("CS-rank validation prediction contains non-finite values")
    model.booster_.save_model(str(fold_dir / "cs_rank.txt"))
    np.savez_compressed(
        output / f"fold_{fold}.npz",
        **_fold_index_payload(
            valid,
            None
            if parent_oof_dir is None
            else Path(parent_oof_dir) / OOF_FILENAMES[int(fold)],
        ),
        cs_rank=prediction,
    )
    report: dict[str, object] = {
        "candidate": "cs_rank",
        "fold": int(fold),
        "gap_time_ids": 0,
        "training_partitions": partitions,
        "rows_by_partition": rows_by_partition,
        "train_rows": int(len(y)),
        "valid_rows": int(len(valid)),
        "feature_columns": len(FEATS),
        "label_mode": "target_minus_weighted_time_mean",
        "lgb": dict(CS_RANK_LGB),
        "prediction": prediction_summary(prediction),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (fold_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    del x, y, w, model, valid, valid_x, prediction
    gc.collect()
    return report


def train_cs_rank_production(
    train_dir: Path,
    test_dir: Path,
    output_root: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    output = ensure_data2_path(output_root)
    production = output / "production"
    production.mkdir(parents=True, exist_ok=True)
    partitions = recent_partitions(9, CS_RANK_WINDOW)
    x, y, w, rows_by_partition = _build_cs_rank_training_cache(
        train_dir, partitions, production / "cache"
    )
    model = _fit_cs_rank(x, y, w)
    model.booster_.save_model(str(production / "cs_rank.txt"))
    row_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    for path in sorted(Path(test_dir).glob("test_partition_*.parquet")):
        frame = _load_rank_frame(path, labels=False)
        rank = build_cs_rank(frame, FEATS)
        prediction = np.asarray(model.predict(rank), dtype=np.float64)
        if not np.isfinite(prediction).all():
            raise ValueError(f"CS-rank public prediction contains non-finite values: {path}")
        row_parts.append(frame["row_id"].to_numpy(dtype=np.int64, copy=True))
        prediction_parts.append(prediction)
        del frame, rank, prediction
        gc.collect()
    public = pd.DataFrame(
        {"row_id": np.concatenate(row_parts), "cs_rank": np.concatenate(prediction_parts)}
    )
    public.to_parquet(production / "public_cs_rank.parquet", index=False)
    report: dict[str, object] = {
        "candidate": "cs_rank",
        "phase": "production",
        "gap_time_ids": 0,
        "training_partitions": partitions,
        "rows_by_partition": rows_by_partition,
        "train_rows": int(len(y)),
        "public_rows": int(len(public)),
        "feature_columns": len(FEATS),
        "label_mode": "target_minus_weighted_time_mean",
        "lgb": dict(CS_RANK_LGB),
        "prediction": prediction_summary(public["cs_rank"].to_numpy(dtype=np.float64)),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (production / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def train_rolling_v2_fold(
    train_dir: Path,
    output_root: Path,
    fold: int,
    parent_oof_dir: Path | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    output = ensure_data2_path(output_root)
    fold_dir = output / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    partitions = list(range(fold))
    x, y, w, history, rows_by_partition = _build_rolling_v2_training_cache(
        train_dir, partitions, fold_dir / "cache"
    )
    model = _fit_rolling_v2(x, y, w)
    valid_columns = ["row_id", "time_id", "asset_id", *ROLLING_TOP50]
    valid = pd.read_parquet(
        Path(train_dir) / f"train_partition_{fold:03d}.parquet", columns=valid_columns
    ).sort_values(["time_id", "asset_id"], kind="stable", ignore_index=True)
    valid_x, ordered, _ = _rolling_v2_with_history(valid, history)
    prediction = np.asarray(model.predict(valid_x), dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError("Rolling V2 validation prediction contains non-finite values")
    model.booster_.save_model(str(fold_dir / "rolling_v2.txt"))
    np.savez_compressed(
        output / f"fold_{fold}.npz",
        **_fold_index_payload(
            ordered,
            None
            if parent_oof_dir is None
            else Path(parent_oof_dir) / OOF_FILENAMES[int(fold)],
        ),
        rolling_v2=prediction,
    )
    report: dict[str, object] = {
        "candidate": "rolling_v2",
        "fold": int(fold),
        "gap_time_ids": 0,
        "training_partitions": partitions,
        "rows_by_partition": rows_by_partition,
        "train_rows": int(len(y)),
        "valid_rows": int(len(ordered)),
        "feature_columns": list(ROLLING_TOP50),
        "operations": list(V2_OPERATIONS),
        "rolling_columns": len(ROLLING_TOP50) * len(V2_OPERATIONS),
        "label_mode": "target",
        "lgb": dict(ROLLING_V2_LGB),
        "prediction": prediction_summary(prediction),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (fold_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    del x, y, w, model, valid, valid_x, ordered, prediction, history
    gc.collect()
    return report


def train_rolling_v2_production(
    train_dir: Path,
    test_dir: Path,
    output_root: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    output = ensure_data2_path(output_root)
    production = output / "production"
    production.mkdir(parents=True, exist_ok=True)
    partitions = list(range(9))
    x, y, w, history, rows_by_partition = _build_rolling_v2_training_cache(
        train_dir, partitions, production / "cache"
    )
    model = _fit_rolling_v2(x, y, w)
    model.booster_.save_model(str(production / "rolling_v2.txt"))
    history.to_parquet(production / "rolling_v2_history.parquet", index=False)
    row_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    prediction_columns = ["row_id", "time_id", "asset_id", *ROLLING_TOP50]
    for path in sorted(Path(test_dir).glob("test_partition_*.parquet")):
        frame = pd.read_parquet(path, columns=prediction_columns).sort_values(
            ["time_id", "asset_id"], kind="stable", ignore_index=True
        )
        matrix, ordered, history = _rolling_v2_with_history(frame, history)
        prediction = np.asarray(model.predict(matrix), dtype=np.float64)
        if not np.isfinite(prediction).all():
            raise ValueError(f"Rolling V2 public prediction is non-finite: {path}")
        row_parts.append(ordered["row_id"].to_numpy(dtype=np.int64, copy=True))
        prediction_parts.append(prediction)
        del frame, matrix, ordered, prediction
        gc.collect()
    public = pd.DataFrame(
        {"row_id": np.concatenate(row_parts), "rolling_v2": np.concatenate(prediction_parts)}
    )
    public.to_parquet(production / "public_rolling_v2.parquet", index=False)
    report: dict[str, object] = {
        "candidate": "rolling_v2",
        "phase": "production",
        "gap_time_ids": 0,
        "training_partitions": partitions,
        "rows_by_partition": rows_by_partition,
        "train_rows": int(len(y)),
        "public_rows": int(len(public)),
        "feature_columns": list(ROLLING_TOP50),
        "operations": list(V2_OPERATIONS),
        "rolling_columns": len(ROLLING_TOP50) * len(V2_OPERATIONS),
        "label_mode": "target",
        "lgb": dict(ROLLING_V2_LGB),
        "prediction": prediction_summary(public["rolling_v2"].to_numpy(dtype=np.float64)),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (production / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("cs_rank", "rolling_v2"), required=True)
    parser.add_argument("--folds", type=int, nargs="*")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--normalize-existing", action="store_true")
    parser.add_argument("--parent-oof-dir", type=Path)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.production and not args.folds:
        parser.error("provide --folds or --production")
    if args.normalize_existing:
        if args.parent_oof_dir is None:
            parser.error("--normalize-existing requires --parent-oof-dir")
        for fold in args.folds or ():
            print(
                json.dumps(
                    normalize_existing_candidate_fold(
                        args.train_dir,
                        args.out,
                        args.parent_oof_dir,
                        fold,
                        args.candidate,
                    ),
                    indent=2,
                )
            )
        return
    fold_trainer = train_cs_rank_fold if args.candidate == "cs_rank" else train_rolling_v2_fold
    production_trainer = (
        train_cs_rank_production
        if args.candidate == "cs_rank"
        else train_rolling_v2_production
    )
    for fold in args.folds or ():
        print(
            json.dumps(
                fold_trainer(args.train_dir, args.out, fold, args.parent_oof_dir),
                indent=2,
            )
        )
    if args.production:
        print(
            json.dumps(
                production_trainer(args.train_dir, args.test_dir, args.out), indent=2
            )
        )


if __name__ == "__main__":
    main()
