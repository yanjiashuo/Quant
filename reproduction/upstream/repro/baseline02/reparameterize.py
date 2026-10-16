from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Mapping, Sequence

from repro.baseline00.metrics import weighted_zero_mean_r2
from repro.baseline02.artifacts import OOFFold
from repro.baseline02.stack import StackFit, fit_stack


def split_final(final: np.ndarray, time_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(final, dtype=np.float64).reshape(-1)
    times = np.asarray(time_id, dtype=np.int64).reshape(-1)
    if prediction.size != times.size or not np.isfinite(prediction).all():
        raise ValueError("final and time_id must be aligned and finite")
    level = pd.Series(prediction).groupby(times, sort=False).transform("mean").to_numpy()
    return level, prediction - level


def reparameterize_fold(fold: OOFFold) -> OOFFold:
    names = list(fold.member_names)
    if names.count("final") != 1 or names.count("L88") != 1:
        raise ValueError("fold must contain exactly one final and one L88 member")
    final_index = names.index("final")
    l88_index = names.index("L88")
    if final_index == l88_index:
        raise AssertionError("final and L88 columns must differ")
    level, deviation = split_final(fold.matrix[:, final_index], fold.time_id)
    keep = [index for index in range(len(names)) if index not in (final_index, l88_index)]
    matrix = np.column_stack((level, deviation, fold.matrix[:, keep]))
    member_names = ("final_level", "final_deviation", *(names[index] for index in keep))
    rebuilt = 0.88 * level + 1.30 * deviation
    if not np.allclose(rebuilt, fold.matrix[:, l88_index], rtol=0.0, atol=1e-7):
        raise ValueError("L88 does not match the frozen level/deviation definition")
    return OOFFold(
        matrix=matrix,
        target=fold.target,
        weight=fold.weight,
        row_id=fold.row_id,
        time_id=fold.time_id,
        asset_id=fold.asset_id,
        member_names=member_names,
        source=f"{fold.source}:level_deviation",
    )


def _fit_folds(folds: Sequence[OOFFold], alpha: float) -> StackFit:
    return fit_stack(
        np.concatenate([fold.matrix for fold in folds]),
        np.concatenate([fold.target for fold in folds]),
        np.concatenate([fold.weight for fold in folds]),
        alpha=alpha,
        member_names=folds[0].member_names,
    )


def _coefficient_drift(left: StackFit, right: StackFit) -> float:
    if left.member_names != right.member_names:
        raise ValueError("coefficient member order differs")
    return float(np.sum(np.abs(left.normalized_coefficients - right.normalized_coefficients)))


def evaluate_reparameterization(
    parent_folds: Mapping[int, OOFFold], alpha: float = 0.01
) -> dict[str, object]:
    rebuild_error: dict[str, float] = {}
    for fold_number in (6, 7, 8):
        fold = parent_folds[fold_number]
        final_index = fold.member_names.index("final")
        l88_index = fold.member_names.index("L88")
        level, deviation = split_final(fold.matrix[:, final_index], fold.time_id)
        rebuild_error[str(fold_number)] = float(
            np.max(
                np.abs(
                    fold.matrix[:, l88_index]
                    - (0.88 * level + 1.30 * deviation)
                )
            )
        )
    transformed = {fold: reparameterize_fold(parent_folds[fold]) for fold in (6, 7, 8)}
    parent_r6 = _fit_folds((parent_folds[6],), alpha)
    parent_r67 = _fit_folds((parent_folds[6], parent_folds[7]), alpha)
    child_r6 = _fit_folds((transformed[6],), alpha)
    child_r67 = _fit_folds((transformed[6], transformed[7]), alpha)
    child_r678 = _fit_folds((transformed[6], transformed[7], transformed[8]), alpha)
    comparisons: dict[str, dict[str, float]] = {}
    for name, parent_fit, child_fit, parent_fold, child_fold in (
        ("fold7", parent_r6, child_r6, parent_folds[7], transformed[7]),
        ("fold8", parent_r67, child_r67, parent_folds[8], transformed[8]),
    ):
        parent_score = weighted_zero_mean_r2(
            parent_fold.target, parent_fit.predict(parent_fold.matrix), parent_fold.weight
        )
        child_score = weighted_zero_mean_r2(
            child_fold.target, child_fit.predict(child_fold.matrix), child_fold.weight
        )
        comparisons[name] = {
            "parent_score": float(parent_score),
            "child_score": float(child_score),
            "delta": float(child_score - parent_score),
        }
    parent_drift = _coefficient_drift(parent_r6, parent_r67)
    child_drift = _coefficient_drift(child_r6, child_r67)
    reduction = (
        float(1.0 - child_drift / parent_drift) if parent_drift > 1e-18 else 0.0
    )
    delta7 = comparisons["fold7"]["delta"]
    delta8 = comparisons["fold8"]["delta"]
    passed = (
        (delta7 + delta8) / 2.0 >= 0.0
        and min(delta7, delta8) >= -0.000010
        and reduction >= 0.10
    )
    return {
        "member_names": list(transformed[6].member_names),
        "l88_rebuild_max_abs_diff": rebuild_error,
        "fold7": comparisons["fold7"],
        "fold8": comparisons["fold8"],
        "parent_normalized_coefficient_drift_l1": parent_drift,
        "child_normalized_coefficient_drift_l1": child_drift,
        "drift_reduction_fraction": reduction,
        "verdict": "promoted" if passed else "rejected",
        "production": {"fit_folds": [6, 7, 8], "fit": child_r678.to_dict()},
    }
