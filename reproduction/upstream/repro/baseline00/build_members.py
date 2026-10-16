from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from repro.config import (
    ASSET_SCALE,
    ASSET_SCALE_SHRINK,
    CS_DEV_ALPHA,
    CS_MEAN_BETA,
    L88_DEV,
    L88_LEVEL,
    OUTPUT_SCALE,
    POST_BIAS,
    POST_SCALE,
    SCALE_LEVEL,
    WEIGHT_A,
    WEIGHT_B,
    WEIGHT_C,
    W_LEVEL,
    WORK,
    ensure_data2_path,
)
from repro.baseline00 import MEMBER_NAMES
from repro.baseline00.metrics import (
    build_l88,
    correlation_matrix,
    prediction_summary,
    weighted_zero_mean_r2,
)


def build_final(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    pred_c: np.ndarray,
    level: np.ndarray,
    time_id: np.ndarray,
    asset_id: np.ndarray,
) -> np.ndarray:
    prediction = WEIGHT_A * pred_a + WEIGHT_B * pred_b
    prediction = (prediction - POST_BIAS) * POST_SCALE
    asset_scale = np.asarray(ASSET_SCALE, dtype=np.float64)
    multiplier = 1.0 + ASSET_SCALE_SHRINK * (asset_scale - 1.0)
    prediction = prediction * multiplier[np.asarray(asset_id, dtype=np.int64)]
    order = np.argsort(time_id, kind="stable")
    sorted_time = np.asarray(time_id)[order]
    sorted_prediction = prediction[order]
    starts = np.r_[0, np.flatnonzero(sorted_time[1:] != sorted_time[:-1]) + 1]
    ends = np.r_[starts[1:], len(prediction)]
    means = np.add.reduceat(sorted_prediction, starts) / (ends - starts)
    mean_rows = np.repeat(means, ends - starts)
    calibrated = CS_DEV_ALPHA * (sorted_prediction - mean_rows) + CS_MEAN_BETA * mean_rows
    cs2 = np.empty_like(prediction)
    cs2[order] = calibrated
    td = cs2 + WEIGHT_C * pred_c
    return OUTPUT_SCALE * (td + W_LEVEL * SCALE_LEVEL * level)


def build_fold(folds_root: Path, members_root: Path, fold: int) -> dict:
    folds_root = ensure_data2_path(folds_root)
    members_root = ensure_data2_path(members_root)
    members_root.mkdir(parents=True, exist_ok=True)
    fold_dir = folds_root / f"fold_{fold}"
    base_dir = fold_dir / "base"
    index = np.load(base_dir / "valid_index.npz")
    row_id = index["row_id"]
    rolling_row_id = np.load(fold_dir / "rolling" / "row_id.npy")
    if not np.array_equal(row_id, rolling_row_id):
        raise ValueError(f"fold {fold} rolling row_id does not match base order")
    pred_a = np.load(base_dir / "model_a.npy")
    pred_b = np.load(base_dir / "model_b.npy")
    pred_c = np.load(base_dir / "model_c.npy")
    level = np.load(fold_dir / "level" / "level.npy")
    rolling = np.load(fold_dir / "rolling" / "rolling_v1.npy")
    final = build_final(
        pred_a, pred_b, pred_c, level, index["time_id"], index["asset_id"]
    )
    l88 = build_l88(final, index["time_id"], L88_LEVEL, L88_DEV)
    matrix = np.column_stack([final, l88, pred_c, level, rolling])
    output = members_root / f"fold_{fold}.npz"
    payload = dict(
        final=final,
        L88=l88,
        model_c=pred_c,
        level=level,
        rolling_v1=rolling,
        row_id=row_id,
        time_id=index["time_id"],
        asset_id=index["asset_id"],
    )
    has_labels = "y" in index.files and "w" in index.files
    member_scores = None
    if has_labels:
        y = index["y"].astype(np.float64)
        w = index["w"].astype(np.float64)
        payload.update(y=y, w=w)
        member_scores = {
            name: weighted_zero_mean_r2(y, matrix[:, column], w)
            for column, name in enumerate(MEMBER_NAMES)
        }
    np.savez(output, **payload)
    report = {
        "fold": fold,
        "rows": int(len(row_id)),
        "labels_present": has_labels,
        "member_names": list(MEMBER_NAMES),
        "member_scores": member_scores,
        "member_prediction": {
            name: prediction_summary(matrix[:, column])
            for column, name in enumerate(MEMBER_NAMES)
        },
        "correlation": correlation_matrix(matrix, MEMBER_NAMES),
        "all_finite": bool(np.isfinite(matrix).all()),
        "l88_rebuild_max_abs_diff": float(
            np.max(np.abs(l88 - build_l88(final, index["time_id"], L88_LEVEL, L88_DEV)))
        ),
    }
    (members_root / f"fold_{fold}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--folds-root", type=Path, default=WORK / "folds")
    parser.add_argument("--out", type=Path, default=WORK / "members")
    args = parser.parse_args()
    for fold in args.folds:
        print(json.dumps(build_fold(args.folds_root, args.out, fold), indent=2))


if __name__ == "__main__":
    main()
