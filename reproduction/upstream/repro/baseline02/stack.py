from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import Ridge

from repro.baseline00.metrics import prediction_summary, weighted_zero_mean_r2
from repro.baseline02.artifacts import OOFFold, load_oof
from repro.baseline02.config import PARENT_MEMBERS, PARENT_RIDGE_ALPHA, oof_paths
from repro.baseline02.diagnostics import contiguous_time_blocks


@dataclass(frozen=True)
class StackFit:
    alpha: float
    member_names: tuple[str, ...]
    standard_deviations: np.ndarray
    normalized_coefficients: np.ndarray
    effective_coefficients: np.ndarray

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        x = np.asarray(matrix, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.effective_coefficients.size:
            raise ValueError("member matrix columns differ from stack fit")
        return x @ self.effective_coefficients

    def to_dict(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "member_names": list(self.member_names),
            "standard_deviations": self.standard_deviations.tolist(),
            "normalized_coefficients": self.normalized_coefficients.tolist(),
            "effective_coefficients": self.effective_coefficients.tolist(),
        }


def fit_stack(
    matrix: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    alpha: float = PARENT_RIDGE_ALPHA,
    member_names: Sequence[str] = PARENT_MEMBERS,
) -> StackFit:
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    names = tuple(member_names)
    if x.ndim != 2 or x.shape != (y.size, len(names)) or w.size != y.size:
        raise ValueError("stack arrays have inconsistent shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(w).all():
        raise ValueError("stack arrays must be finite")
    if np.any(w < 0.0) or float(w.sum()) <= 0.0:
        raise ValueError("stack weight must be nonnegative with positive sum")
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    ridge = Ridge(alpha=alpha, fit_intercept=False, positive=True)
    ridge.fit(x / scale, y, sample_weight=w)
    normalized = np.asarray(ridge.coef_, dtype=np.float64)
    return StackFit(
        alpha=float(alpha),
        member_names=names,
        standard_deviations=scale,
        normalized_coefficients=normalized,
        effective_coefficients=normalized / scale,
    )


def fit_stack_ratio(
    matrix: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    lambda_ratio: float,
    member_names: Sequence[str],
) -> StackFit:
    ratio = float(lambda_ratio)
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    if not np.isfinite(ratio) or ratio < 0.0:
        raise ValueError("lambda_ratio must be finite and nonnegative")
    if not np.isfinite(w).all() or float(w.sum()) <= 0.0:
        raise ValueError("weight must be finite with positive sum")
    return fit_stack(
        matrix,
        target,
        w,
        alpha=ratio * float(w.sum()),
        member_names=member_names,
    )


def select_lambda_ratio(
    fit_fold: OOFFold,
    report_fold: OOFFold,
    grid: Sequence[float],
    plateau_tolerance: float = 0.000010,
) -> dict[str, object]:
    if fit_fold.member_names != report_fold.member_names:
        raise ValueError("fit and report member order differs")
    ratios = tuple(float(value) for value in grid)
    if not ratios:
        raise ValueError("lambda ratio grid cannot be empty")
    candidates: list[dict[str, object]] = []
    for ratio in ratios:
        fit = fit_stack_ratio(
            fit_fold.matrix,
            fit_fold.target,
            fit_fold.weight,
            ratio,
            fit_fold.member_names,
        )
        prediction = fit.predict(report_fold.matrix)
        candidates.append(
            {
                "lambda_ratio": ratio,
                "score": weighted_zero_mean_r2(
                    report_fold.target, prediction, report_fold.weight
                ),
                "fit": fit.to_dict(),
            }
        )
    best = max(float(candidate["score"]) for candidate in candidates)
    eligible = [
        candidate
        for candidate in candidates
        if float(candidate["score"]) >= best - float(plateau_tolerance)
    ]
    selected = max(eligible, key=lambda candidate: float(candidate["lambda_ratio"]))
    return {
        "grid": list(ratios),
        "plateau_tolerance": float(plateau_tolerance),
        "candidates": candidates,
        "best_score": best,
        "selected_ratio": float(selected["lambda_ratio"]),
        "selected_score": float(selected["score"]),
    }


def _combine(folds: Sequence[OOFFold]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not folds:
        raise ValueError("at least one fit fold is required")
    names = folds[0].member_names
    if any(fold.member_names != names for fold in folds):
        raise ValueError("fold member order differs")
    return (
        np.concatenate([fold.matrix for fold in folds]),
        np.concatenate([fold.target for fold in folds]),
        np.concatenate([fold.weight for fold in folds]),
    )


def _evaluation(fit: StackFit, fold: OOFFold, fit_folds: Sequence[int]) -> dict[str, object]:
    prediction = fit.predict(fold.matrix)
    return {
        "fit_folds": list(fit_folds),
        "report_rows": fold.rows,
        "raw_score": weighted_zero_mean_r2(fold.target, prediction, fold.weight),
        "prediction": prediction_summary(prediction),
        "fit": fit.to_dict(),
    }


def evaluate_walk_forward(
    folds: Mapping[int, OOFFold],
    alpha: float = PARENT_RIDGE_ALPHA,
) -> dict[str, object]:
    missing = [fold for fold in (6, 7, 8) if fold not in folds]
    if missing:
        raise ValueError(f"missing folds: {missing}")
    f6, f7, f8 = folds[6], folds[7], folds[8]
    if not (f6.time_id.max() < f7.time_id.min() < f8.time_id.min()):
        raise ValueError("OOF folds must be strictly chronological")
    x6, y6, w6 = _combine((f6,))
    r6 = fit_stack(x6, y6, w6, alpha, f6.member_names)
    x67, y67, w67 = _combine((f6, f7))
    r67 = fit_stack(x67, y67, w67, alpha, f6.member_names)
    x678, y678, w678 = _combine((f6, f7, f8))
    r678 = fit_stack(x678, y678, w678, alpha, f6.member_names)
    return {
        "member_names": list(f6.member_names),
        "fold7": _evaluation(r6, f7, (6,)),
        "fold8": _evaluation(r67, f8, (6, 7)),
        "production": {"fit_folds": [6, 7, 8], "fit": r678.to_dict()},
    }


def promotion_gate(
    delta7: float,
    delta8: float | None,
    positive_blocks7: int,
    positive_blocks8: int | None,
    coef7: float,
    coef8: float | None,
) -> dict[str, bool | None]:
    development_pass = delta7 >= 0.000020 and positive_blocks7 >= 12 and coef7 > 0.0
    confirmation_pass: bool | None = None
    if delta8 is not None:
        if positive_blocks8 is None or coef8 is None:
            raise ValueError("Fold8 metrics must be complete")
        confirmation_pass = (
            development_pass
            and delta8 >= 0.0
            and (delta7 + delta8) / 2.0 >= 0.000020
            and positive_blocks8 >= 10
            and coef8 > 0.0
        )
    return {
        "development_pass": bool(development_pass),
        "confirmation_pass": None if confirmation_pass is None else bool(confirmation_pass),
    }


def augment_fold(fold: OOFFold, candidate_path: Path, candidate_name: str) -> OOFFold:
    path = Path(candidate_path)
    with np.load(path, allow_pickle=False) as data:
        if "row_id" not in data.files or candidate_name not in data.files:
            raise ValueError(f"{path} must contain row_id and {candidate_name}")
        row_id = np.asarray(data["row_id"], dtype=np.int64).reshape(-1)
        prediction = np.asarray(data[candidate_name], dtype=np.float64).reshape(-1)
        candidate_time = (
            np.asarray(data["time_id"], dtype=np.int64).reshape(-1)
            if "time_id" in data.files
            else None
        )
        candidate_asset = (
            np.asarray(data["asset_id"], dtype=np.int64).reshape(-1)
            if "asset_id" in data.files
            else None
        )
    if not np.array_equal(row_id, fold.row_id):
        raise ValueError(f"{path} row_id differs from parent OOF")
    if prediction.size != fold.rows or not np.isfinite(prediction).all():
        raise ValueError(f"{path} prediction is misaligned or non-finite")
    if candidate_time is not None and not np.array_equal(candidate_time, fold.time_id):
        raise ValueError(f"{path} time_id differs from parent OOF")
    if candidate_asset is not None and not np.array_equal(candidate_asset, fold.asset_id):
        raise ValueError(f"{path} asset_id differs from parent OOF")
    if candidate_name in fold.member_names:
        raise ValueError(f"candidate name already exists: {candidate_name}")
    return OOFFold(
        matrix=np.column_stack((fold.matrix, prediction)),
        target=fold.target,
        weight=fold.weight,
        row_id=fold.row_id,
        time_id=fold.time_id,
        asset_id=fold.asset_id,
        member_names=(*fold.member_names, candidate_name),
        source=f"{fold.source}+{path}",
    )


def _block_deltas(
    fold: OOFFold, parent_prediction: np.ndarray, child_prediction: np.ndarray
) -> list[float]:
    blocks = contiguous_time_blocks(fold.time_id)
    deltas: list[float] = []
    for block in np.unique(blocks):
        selected = blocks == block
        parent_score = weighted_zero_mean_r2(
            fold.target[selected], parent_prediction[selected], fold.weight[selected]
        )
        child_score = weighted_zero_mean_r2(
            fold.target[selected], child_prediction[selected], fold.weight[selected]
        )
        deltas.append(float(child_score - parent_score))
    return deltas


def _candidate_evaluation(
    fit_folds: Sequence[OOFFold],
    report_fold: OOFFold,
    child_fit_folds: Sequence[OOFFold],
    child_report_fold: OOFFold,
    fit_fold_numbers: Sequence[int],
    alpha: float,
) -> dict[str, object]:
    parent_x, parent_y, parent_w = _combine(fit_folds)
    parent_fit = fit_stack(
        parent_x, parent_y, parent_w, alpha, fit_folds[0].member_names
    )
    child_x, child_y, child_w = _combine(child_fit_folds)
    child_fit = fit_stack(
        child_x, child_y, child_w, alpha, child_fit_folds[0].member_names
    )
    parent_prediction = parent_fit.predict(report_fold.matrix)
    child_prediction = child_fit.predict(child_report_fold.matrix)
    parent_score = weighted_zero_mean_r2(
        report_fold.target, parent_prediction, report_fold.weight
    )
    child_score = weighted_zero_mean_r2(
        report_fold.target, child_prediction, report_fold.weight
    )
    block_deltas = _block_deltas(report_fold, parent_prediction, child_prediction)
    return {
        "fit_folds": list(fit_fold_numbers),
        "parent_score": float(parent_score),
        "child_score": float(child_score),
        "delta": float(child_score - parent_score),
        "positive_blocks": int(np.count_nonzero(np.asarray(block_deltas) > 0.0)),
        "block_deltas": block_deltas,
        "new_member_effective_coefficient": float(child_fit.effective_coefficients[-1]),
        "parent_fit": parent_fit.to_dict(),
        "child_fit": child_fit.to_dict(),
    }


def evaluate_candidate(
    parent_folds: Mapping[int, OOFFold],
    candidate_dir: Path,
    candidate_name: str,
    alpha: float = PARENT_RIDGE_ALPHA,
    max_report_fold: int = 8,
) -> dict[str, object]:
    required = (6, 7) if max_report_fold == 7 else (6, 7, 8)
    if max_report_fold not in (7, 8):
        raise ValueError("max_report_fold must be 7 or 8")
    missing = [fold for fold in required if fold not in parent_folds]
    if missing:
        raise ValueError(f"missing parent folds: {missing}")
    child = {
        fold: augment_fold(
            parent_folds[fold], Path(candidate_dir) / f"fold_{fold}.npz", candidate_name
        )
        for fold in required
    }
    fold7 = _candidate_evaluation(
        (parent_folds[6],),
        parent_folds[7],
        (child[6],),
        child[7],
        (6,),
        alpha,
    )
    fold8: dict[str, object] | None = None
    production: dict[str, object] | None = None
    if max_report_fold == 8:
        fold8 = _candidate_evaluation(
            (parent_folds[6], parent_folds[7]),
            parent_folds[8],
            (child[6], child[7]),
            child[8],
            (6, 7),
            alpha,
        )
        x, y, w = _combine((child[6], child[7], child[8]))
        production_fit = fit_stack(x, y, w, alpha, child[6].member_names)
        production = {"fit_folds": [6, 7, 8], "fit": production_fit.to_dict()}
    gate = promotion_gate(
        float(fold7["delta"]),
        None if fold8 is None else float(fold8["delta"]),
        int(fold7["positive_blocks"]),
        None if fold8 is None else int(fold8["positive_blocks"]),
        float(fold7["new_member_effective_coefficient"]),
        None if fold8 is None else float(fold8["new_member_effective_coefficient"]),
    )
    report: dict[str, object] = {
        "candidate_name": candidate_name,
        "alpha": float(alpha),
        "member_names": list(child[6].member_names),
        "fold7": fold7,
        "fold8": fold8,
        "production": production,
        "gate": gate,
        "verdict": (
            "development_pass"
            if max_report_fold == 7 and gate["development_pass"]
            else "development_reject"
            if max_report_fold == 7
            else "promoted"
            if gate["confirmation_pass"]
            else "rejected"
        ),
    }
    return report


def select_members(
    parent_members: Sequence[str], candidate_reports: Sequence[Mapping[str, object]]
) -> tuple[str, ...]:
    selected = list(parent_members)
    for report in candidate_reports:
        name = str(report.get("candidate_name", ""))
        gate = report.get("gate")
        if not name or not isinstance(gate, Mapping):
            raise ValueError("candidate report is incomplete")
        if gate.get("confirmation_pass") is not True:
            raise ValueError(f"candidate did not pass confirmation gate: {name}")
        if name in selected:
            raise ValueError(f"candidate member already exists: {name}")
        selected.append(name)
    return tuple(selected)


def evaluate_candidate_set(
    parent_folds: Mapping[int, OOFFold],
    candidate_dirs: Mapping[str, Path],
    alpha: float = PARENT_RIDGE_ALPHA,
) -> dict[str, object]:
    if not candidate_dirs:
        raise ValueError("candidate set cannot be empty")
    missing = [fold for fold in (6, 7, 8) if fold not in parent_folds]
    if missing:
        raise ValueError(f"missing parent folds: {missing}")
    child: dict[int, OOFFold] = {}
    for fold in (6, 7, 8):
        current = parent_folds[fold]
        for name, directory in candidate_dirs.items():
            current = augment_fold(
                current, Path(directory) / f"fold_{fold}.npz", name
            )
        child[fold] = current
    fold7 = _candidate_evaluation(
        (parent_folds[6],),
        parent_folds[7],
        (child[6],),
        child[7],
        (6,),
        alpha,
    )
    fold8 = _candidate_evaluation(
        (parent_folds[6], parent_folds[7]),
        parent_folds[8],
        (child[6], child[7]),
        child[8],
        (6, 7),
        alpha,
    )
    x, y, w = _combine((child[6], child[7], child[8]))
    production_fit = fit_stack(x, y, w, alpha, child[6].member_names)
    candidate_names = tuple(candidate_dirs)
    for evaluation in (fold7, fold8):
        fit = evaluation["child_fit"]
        names = tuple(fit["member_names"])
        coefficients = tuple(fit["effective_coefficients"])
        evaluation["new_member_effective_coefficients"] = {
            name: float(coefficients[names.index(name)]) for name in candidate_names
        }
    return {
        "candidate_names": list(candidate_names),
        "alpha": float(alpha),
        "member_names": list(child[6].member_names),
        "fold7": fold7,
        "fold8": fold8,
        "production": {"fit_folds": [6, 7, 8], "fit": production_fit.to_dict()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=PARENT_RIDGE_ALPHA)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--candidate-name")
    parser.add_argument(
        "--candidate-set",
        nargs="+",
        help="candidate set entries formatted as name=/path/to/artifacts",
    )
    parser.add_argument("--max-report-fold", type=int, choices=(7, 8), default=8)
    args = parser.parse_args()
    folds = {fold: load_oof(path) for fold, path in oof_paths(args.oof_dir).items()}
    if bool(args.candidate_dir) != bool(args.candidate_name):
        parser.error("--candidate-dir and --candidate-name must be provided together")
    if args.candidate_set and args.candidate_dir:
        parser.error("--candidate-set cannot be combined with --candidate-dir")
    if args.candidate_set:
        entries: dict[str, Path] = {}
        for entry in args.candidate_set:
            if "=" not in entry:
                parser.error("--candidate-set entries must be name=path")
            name, raw_path = entry.split("=", 1)
            if not name or name in entries:
                parser.error("candidate set names must be nonempty and unique")
            entries[name] = Path(raw_path)
        report = evaluate_candidate_set(folds, entries, alpha=args.alpha)
    elif args.candidate_dir:
        report = evaluate_candidate(
            folds,
            args.candidate_dir,
            args.candidate_name,
            alpha=args.alpha,
            max_report_fold=args.max_report_fold,
        )
    else:
        report = evaluate_walk_forward(folds, alpha=args.alpha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
