from __future__ import annotations

"""E2 weighted-linear CS-rank trainer (``cs_rank_linear``).

Same 323 within-time rank features and residual label as the frozen ``cs_rank``
LGBM member and the *same* recent-4 training window; only the base learner
(streamed weighted Ridge) changes. Regularization is chosen by
:func:`select_linear_ratio` on the Fold7 child score alone — Fold8, the public
labels, and the record-only IC diagnostic never enter the selection.
"""

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from repro.baseline00.metrics import prediction_summary, weighted_zero_mean_r2
from repro.baseline02.artifacts import OOFFold
from repro.baseline02.config import OOF_FILENAMES
from repro.baseline02.stack import fit_stack
from repro.baseline02.train_candidate import _fold_index_payload
from repro.baseline05.config import FOLDS, LINEAR_RATIO_GRID, RIDGE_ALPHA
from repro.baseline05.linear import (
    ratio_diagnostics,
    select_ratio,
    solve_weighted_ridge,
    sufficient_statistics,
)
from repro.baseline05.rank_data import (
    iter_linear_batches,
    load_rank_frame,
    rank_features,
    rank_target,
    training_partitions,
)
from repro.baseline05.parent import load_baseline04_folds
from repro.baseline05.replacement import replace_member
from repro.config import FEATS, TEST_DIR, TRAIN_DIR, ensure_data2_path


def _partition_path(train_dir: Path, fold: int) -> Path:
    return Path(train_dir) / f"train_partition_{int(fold):03d}.parquet"


def _fold_linear_inputs(train_dir: Path, fold: int) -> dict[str, object]:
    """Recent-4 sufficient stats + this fold's OOF rank features/label/index."""
    partitions = training_partitions(fold, "recent4")
    stats = sufficient_statistics(iter_linear_batches(train_dir, partitions))
    frame = load_rank_frame(_partition_path(train_dir, fold), labels=True)
    features = np.asarray(rank_features(frame), dtype=np.float64)
    residual, weight = rank_target(frame)
    time_id = frame["time_id"].to_numpy(dtype=np.int64)
    return {
        "partitions": partitions,
        "stats": stats,
        "features": features,
        "residual": residual,
        "weight": weight,
        "time_id": time_id,
    }


def select_linear_ratio(
    train_dir: Path,
    parent_folds: Mapping[int, OOFFold],
    grid: Sequence[float] = LINEAR_RATIO_GRID,
    alpha: float = RIDGE_ALPHA,
) -> dict[str, object]:
    """Pick the ridge ratio by the Fold7 child score only.

    The linear member is fit per fold on its recent-4 window and swapped into the
    parent OOF; the meta-stack is fit on Fold6 and scored on Fold7. Fold8 is never
    loaded. IC diagnostics are recorded per ratio but excluded from selection —
    only ``{ratio: fold7_child_score}`` reaches :func:`select_ratio`.
    """
    fit_inputs = _fold_linear_inputs(train_dir, 6)
    report_inputs = _fold_linear_inputs(train_dir, 7)
    parent6 = parent_folds[6]
    parent7 = parent_folds[7]

    ratios: dict[float, dict[str, object]] = {}
    for raw_ratio in grid:
        ratio = float(raw_ratio)
        beta6 = solve_weighted_ridge(fit_inputs["stats"], ratio)
        beta7 = solve_weighted_ridge(report_inputs["stats"], ratio)
        pred6 = np.asarray(fit_inputs["features"], dtype=np.float64) @ beta6.coefficients
        pred7 = np.asarray(report_inputs["features"], dtype=np.float64) @ beta7.coefficients
        child6 = replace_member(parent6, "cs_rank", "cs_rank_linear", pred6)
        child7 = replace_member(parent7, "cs_rank", "cs_rank_linear", pred7)
        stack = fit_stack(child6.matrix, child6.target, child6.weight, alpha, child6.member_names)
        prediction = stack.predict(child7.matrix)
        child_score = weighted_zero_mean_r2(child7.target, prediction, child7.weight)
        diagnostics = ratio_diagnostics(
            pred7, report_inputs["residual"], report_inputs["weight"], report_inputs["time_id"]
        )
        ratios[ratio] = {
            "fold7_child_score": float(child_score),
            "penalty": float(beta6.penalty),
            **diagnostics,
        }
    # Only the child score reaches the selector — IC diagnostics are excluded.
    selection_inputs = {
        ratio: float(report["fold7_child_score"]) for ratio, report in ratios.items()
    }
    selected = select_ratio(selection_inputs)
    return {
        "grid": [float(value) for value in grid],
        "ratios": ratios,
        "selection_inputs": selection_inputs,
        "selected_ratio": float(selected),
    }


def _save_linear_model(path: Path, fit) -> None:
    """Persist the fitted coefficients + penalty as a plain JSON model."""
    Path(path).write_text(
        json.dumps(
            {
                "member": "cs_rank_linear",
                "lambda_ratio": float(fit.lambda_ratio),
                "penalty": float(fit.penalty),
                "feature_columns": list(FEATS),
                "coefficients": np.asarray(fit.coefficients, dtype=np.float64).tolist(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def train_linear_fold(
    train_dir: Path,
    output_root: Path,
    fold: int,
    lambda_ratio: float,
    parent_oof_dir: Path | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    output = ensure_data2_path(output_root)
    fold_dir = output / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    partitions = training_partitions(fold, "recent4")
    stats = sufficient_statistics(iter_linear_batches(train_dir, partitions))
    fit = solve_weighted_ridge(stats, lambda_ratio)
    valid = load_rank_frame(_partition_path(train_dir, fold), labels=False)
    valid_x = np.asarray(rank_features(valid), dtype=np.float64)
    prediction = valid_x @ fit.coefficients
    if not np.isfinite(prediction).all():
        raise ValueError("cs_rank_linear validation prediction is non-finite")
    _save_linear_model(fold_dir / "cs_rank_linear.json", fit)
    np.savez_compressed(
        output / f"fold_{fold}.npz",
        **_fold_index_payload(
            valid,
            None
            if parent_oof_dir is None
            else Path(parent_oof_dir) / OOF_FILENAMES[int(fold)],
        ),
        cs_rank_linear=prediction,
    )
    report: dict[str, object] = {
        "candidate": "cs_rank_linear",
        "fold": int(fold),
        "gap_time_ids": 0,
        "window": "recent4",
        "learner": "weighted_ridge",
        "lambda_ratio": float(lambda_ratio),
        "penalty": float(fit.penalty),
        "training_partitions": partitions,
        "valid_rows": int(len(valid)),
        "feature_columns": len(FEATS),
        "label_mode": "target_minus_weighted_time_mean",
        "prediction": prediction_summary(prediction),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (fold_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    del stats, fit, valid, valid_x, prediction
    gc.collect()
    return report


def train_linear_production(
    train_dir: Path,
    test_dir: Path,
    output_root: Path,
    lambda_ratio: float,
) -> dict[str, object]:
    started = time.perf_counter()
    output = ensure_data2_path(output_root)
    production = output / "production"
    production.mkdir(parents=True, exist_ok=True)
    partitions = training_partitions(9, "recent4")
    stats = sufficient_statistics(iter_linear_batches(train_dir, partitions))
    fit = solve_weighted_ridge(stats, lambda_ratio)
    _save_linear_model(production / "cs_rank_linear.json", fit)
    row_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    asset_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    for path in sorted(Path(test_dir).glob("test_partition_*.parquet")):
        frame = load_rank_frame(path, labels=False)
        rank = np.asarray(rank_features(frame), dtype=np.float64)
        prediction = rank @ fit.coefficients
        if not np.isfinite(prediction).all():
            raise ValueError(f"cs_rank_linear public prediction is non-finite: {path}")
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
            "cs_rank_linear": np.concatenate(prediction_parts),
        }
    )
    public.to_parquet(production / "public_cs_rank_linear.parquet", index=False)
    report: dict[str, object] = {
        "candidate": "cs_rank_linear",
        "phase": "production",
        "gap_time_ids": 0,
        "window": "recent4",
        "learner": "weighted_ridge",
        "lambda_ratio": float(lambda_ratio),
        "penalty": float(fit.penalty),
        "training_partitions": partitions,
        "public_rows": int(len(public)),
        "feature_columns": len(FEATS),
        "label_mode": "target_minus_weighted_time_mean",
        "prediction": prediction_summary(
            public["cs_rank_linear"].to_numpy(dtype=np.float64)
        ),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    (production / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--folds", type=int, nargs="*")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--lambda-ratio", type=float)
    parser.add_argument("--parent-oof-dir", type=Path)
    parser.add_argument("--base-oof-dir", type=Path)
    parser.add_argument("--cs-rank-dir", type=Path)
    parser.add_argument("--long-oof-dir", type=Path)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.select:
        missing = [
            flag
            for flag, value in (
                ("--base-oof-dir", args.base_oof_dir),
                ("--cs-rank-dir", args.cs_rank_dir),
                ("--long-oof-dir", args.long_oof_dir),
            )
            if value is None
        ]
        if missing:
            parser.error(f"--select requires {', '.join(missing)}")
        parent_folds = load_baseline04_folds(
            args.base_oof_dir, args.cs_rank_dir, args.long_oof_dir
        )
        # Ratio selection only ever touches Fold6 (fit) and Fold7 (score).
        selection_parent = {6: parent_folds[6], 7: parent_folds[7]}
        result = select_linear_ratio(args.train_dir, selection_parent)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if (args.folds or args.production) and args.lambda_ratio is None:
        parser.error("--folds/--production require --lambda-ratio")
    if not args.production and not args.folds:
        parser.error("provide --select, --folds, or --production")
    for fold in args.folds or ():
        if fold not in FOLDS:
            parser.error(f"fold must be one of {FOLDS}")
        print(
            json.dumps(
                train_linear_fold(
                    args.train_dir, args.out, fold, args.lambda_ratio, args.parent_oof_dir
                ),
                indent=2,
            )
        )
    if args.production:
        print(
            json.dumps(
                train_linear_production(
                    args.train_dir, args.test_dir, args.out, args.lambda_ratio
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
