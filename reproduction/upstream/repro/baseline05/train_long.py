from __future__ import annotations

"""E1 full-history LGBM CS-rank trainer (``cs_rank_long``).

Identical to the frozen Plan02 ``cs_rank`` member in every respect (feature
ranks, residual label, sample weights, LightGBM hyperparameters, seed, gap=0,
feature order) EXCEPT the training window: E1 uses all prior partitions
(``0..fold-1``) instead of the recent-4 window. Only the window changes.
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from repro.baseline00.metrics import prediction_summary
from repro.baseline02.config import CS_RANK_LGB, OOF_FILENAMES
from repro.baseline02.train_candidate import _fold_index_payload
from repro.baseline05.config import FOLDS
from repro.baseline05.rank_data import (
    build_lgb_cache,
    load_rank_frame,
    rank_features,
    training_partitions,
)
from repro.config import FEATS, TEST_DIR, TRAIN_DIR, ensure_data2_path

# Frozen: E1 shares the Plan02 CS-rank LightGBM hyperparameters exactly.
LONG_LGB = dict(CS_RANK_LGB)


def fit_long_model(x: np.ndarray, y: np.ndarray, weight: np.ndarray):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        **LONG_LGB,
        n_jobs=min(64, max(1, os.cpu_count() or 1)),
        verbosity=-1,
    )
    model.fit(x, y, sample_weight=weight)
    return model


def train_long_fold(
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
    partitions = training_partitions(fold, "long")
    x, y, w, rows_by_partition = build_lgb_cache(train_dir, partitions, cache_dir)
    model = fit_long_model(x, y, w)
    valid = load_rank_frame(
        Path(train_dir) / f"train_partition_{fold:03d}.parquet", labels=False
    )
    valid_x = rank_features(valid)
    prediction = np.asarray(model.predict(valid_x), dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError("cs_rank_long validation prediction contains non-finite values")
    model.booster_.save_model(str(fold_dir / "cs_rank_long.txt"))
    np.savez_compressed(
        output / f"fold_{fold}.npz",
        **_fold_index_payload(
            valid,
            None
            if parent_oof_dir is None
            else Path(parent_oof_dir) / OOF_FILENAMES[int(fold)],
        ),
        cs_rank_long=prediction,
    )
    report: dict[str, object] = {
        "candidate": "cs_rank_long",
        "fold": int(fold),
        "gap_time_ids": 0,
        "window": "long",
        "training_partitions": partitions,
        "rows_by_partition": rows_by_partition,
        "train_rows": int(len(y)),
        "valid_rows": int(len(valid)),
        "feature_columns": len(FEATS),
        "label_mode": "target_minus_weighted_time_mean",
        "lgb": dict(LONG_LGB),
        "prediction": prediction_summary(prediction),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (fold_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    del x, y, w, model, valid, valid_x, prediction
    gc.collect()
    return report


def train_long_production(
    train_dir: Path,
    test_dir: Path,
    output_root: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    output = ensure_data2_path(output_root)
    production = output / "production"
    production.mkdir(parents=True, exist_ok=True)
    partitions = training_partitions(9, "long")
    x, y, w, rows_by_partition = build_lgb_cache(
        train_dir, partitions, production / "cache"
    )
    model = fit_long_model(x, y, w)
    model.booster_.save_model(str(production / "cs_rank_long.txt"))
    row_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    asset_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    for path in sorted(Path(test_dir).glob("test_partition_*.parquet")):
        frame = load_rank_frame(path, labels=False)
        rank = rank_features(frame)
        prediction = np.asarray(model.predict(rank), dtype=np.float64)
        if not np.isfinite(prediction).all():
            raise ValueError(f"cs_rank_long public prediction is non-finite: {path}")
        row_parts.append(frame["row_id"].to_numpy(dtype=np.int64, copy=True))
        time_parts.append(frame["time_id"].to_numpy(dtype=np.int64, copy=True))
        asset_parts.append(frame["asset_id"].to_numpy(dtype=np.int64, copy=True))
        prediction_parts.append(prediction)
        del frame, rank, prediction
        gc.collect()
    public = pd.DataFrame(
        {
            "row_id": np.concatenate(row_parts),
            "time_id": np.concatenate(time_parts),
            "asset_id": np.concatenate(asset_parts),
            "cs_rank_long": np.concatenate(prediction_parts),
        }
    )
    public.to_parquet(production / "public_cs_rank_long.parquet", index=False)
    report: dict[str, object] = {
        "candidate": "cs_rank_long",
        "phase": "production",
        "gap_time_ids": 0,
        "window": "long",
        "training_partitions": partitions,
        "rows_by_partition": rows_by_partition,
        "train_rows": int(len(y)),
        "public_rows": int(len(public)),
        "feature_columns": len(FEATS),
        "label_mode": "target_minus_weighted_time_mean",
        "lgb": dict(LONG_LGB),
        "prediction": prediction_summary(
            public["cs_rank_long"].to_numpy(dtype=np.float64)
        ),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (production / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="*")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--parent-oof-dir", type=Path)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.production and not args.folds:
        parser.error("provide --folds or --production")
    for fold in args.folds or ():
        if fold not in FOLDS:
            parser.error(f"fold must be one of {FOLDS}")
        print(
            json.dumps(
                train_long_fold(args.train_dir, args.out, fold, args.parent_oof_dir),
                indent=2,
            )
        )
    if args.production:
        print(
            json.dumps(
                train_long_production(args.train_dir, args.test_dir, args.out), indent=2
            )
        )


if __name__ == "__main__":
    main()
