from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from repro.baseline00.metrics import weighted_zero_mean_r2
from repro.baseline02.artifacts import OOFFold, load_oof
from repro.baseline02.config import N_TIME_BLOCKS, PARENT_MEMBERS, oof_paths


def _matrix_weight(matrix: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(matrix, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != w.size:
        raise ValueError("matrix rows must match weight")
    if np.any(w < 0.0) or not np.isfinite(w).all():
        raise ValueError("weight must be finite and nonnegative")
    if float(w.sum()) <= 0.0 or not np.isfinite(x).all():
        raise ValueError("diagnostic input is empty or non-finite")
    return x, w


def weighted_gram(matrix: np.ndarray, weight: np.ndarray) -> np.ndarray:
    x, w = _matrix_weight(matrix, weight)
    return (x * w[:, None]).T @ x


def weighted_cosine_matrix(matrix: np.ndarray, weight: np.ndarray) -> np.ndarray:
    gram = weighted_gram(matrix, weight)
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    denominator = np.outer(norms, norms)
    return np.divide(
        gram,
        denominator,
        out=np.zeros_like(gram),
        where=denominator > 0.0,
    )


def weighted_pearson_matrix(matrix: np.ndarray, weight: np.ndarray) -> np.ndarray:
    x, w = _matrix_weight(matrix, weight)
    mean = np.sum(x * w[:, None], axis=0) / w.sum()
    centered = x - mean
    covariance = (centered * w[:, None]).T @ centered
    scale = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(scale, scale)
    return np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0.0,
    )


def contiguous_time_blocks(time_id: np.ndarray, n_blocks: int = N_TIME_BLOCKS) -> np.ndarray:
    time = np.asarray(time_id, dtype=np.int64).reshape(-1)
    if time.size == 0 or n_blocks <= 0:
        raise ValueError("time_id must be nonempty and n_blocks positive")
    if np.any(time[1:] < time[:-1]):
        raise ValueError("time_id must be nondecreasing")
    unique_time = np.unique(time)
    groups = np.array_split(unique_time, min(n_blocks, unique_time.size))
    lookup = {int(value): block for block, group in enumerate(groups) for value in group}
    return np.fromiter((lookup[int(value)] for value in time), dtype=np.int64)


def diagnose_fold(fold: OOFFold) -> dict[str, object]:
    x = fold.matrix
    w = fold.weight
    target_norm = np.sqrt(float(np.dot(w, fold.target * fold.target)))
    member_norm = np.sqrt(np.diag(weighted_gram(x, w)))
    target_dot = x.T @ (w * fold.target)
    target_cosine = np.divide(
        target_dot,
        member_norm * target_norm,
        out=np.zeros_like(target_dot),
        where=member_norm > 0.0,
    )
    standalone = [
        weighted_zero_mean_r2(fold.target, x[:, column], w)
        for column in range(x.shape[1])
    ]
    blocks = contiguous_time_blocks(fold.time_id)
    block_rows = [int(np.count_nonzero(blocks == block)) for block in np.unique(blocks)]
    return {
        "source": fold.source,
        "rows": fold.rows,
        "member_names": list(fold.member_names),
        "weighted_pearson": weighted_pearson_matrix(x, w).tolist(),
        "weighted_cosine": weighted_cosine_matrix(x, w).tolist(),
        "weighted_gram": weighted_gram(x, w).tolist(),
        "target_cosine": dict(zip(fold.member_names, target_cosine.tolist())),
        "standalone_score": dict(zip(fold.member_names, standalone)),
        "time_blocks": len(block_rows),
        "block_rows": block_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {
        str(fold): diagnose_fold(load_oof(path))
        for fold, path in oof_paths(args.oof_dir).items()
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
