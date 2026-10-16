from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


TARGET_OUTPUT_STD = 0.046
BASE_PUBLIC_MEMBER_COLUMNS = ("final", "L88", "model_c", "level_pred", "rolling_v1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_l88(final: np.ndarray, time_id: np.ndarray) -> np.ndarray:
    prediction = np.asarray(final, dtype=np.float64).reshape(-1)
    times = np.asarray(time_id, dtype=np.int64).reshape(-1)
    if prediction.size != times.size:
        raise ValueError("final and time_id lengths differ")
    if not np.isfinite(prediction).all():
        raise ValueError("final contains non-finite values")
    level = pd.Series(prediction).groupby(times, sort=False).transform("mean").to_numpy()
    deviation = prediction - level
    return 0.88 * level + 1.30 * deviation


def calibrate_mean_preserving_std(
    prediction: np.ndarray, target_std: float = TARGET_OUTPUT_STD
) -> np.ndarray:
    raw = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if not np.isfinite(raw).all() or raw.size == 0:
        raise ValueError("prediction must be nonempty and finite")
    requested = float(target_std)
    if not np.isfinite(requested) or requested <= 0.0:
        raise ValueError("target_std must be positive and finite")
    mean = float(raw.mean())
    standard_deviation = float(raw.std(ddof=0))
    if standard_deviation <= 1e-12:
        raise ValueError("prediction standard deviation is zero")
    return mean + (requested / standard_deviation) * (raw - mean)


def _read_aligned_prediction(path: Path, row_id: np.ndarray, output_name: str) -> np.ndarray:
    frame = pd.read_csv(path)
    if "row_id" not in frame or len(frame.columns) != 2:
        raise ValueError(f"{path} must contain row_id and one prediction column")
    value_column = next(column for column in frame.columns if column != "row_id")
    if frame["row_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate row_id")
    source_rows = frame["row_id"].to_numpy(dtype=np.int64, copy=False)
    if np.array_equal(source_rows, row_id):
        values = frame[value_column].to_numpy(dtype=np.float64, copy=True)
    else:
        aligned = frame.set_index("row_id")[value_column].reindex(row_id)
        if aligned.isna().any():
            raise ValueError(f"{path} row_id does not cover metadata")
        values = aligned.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError(f"{output_name} contains non-finite values")
    return values


def load_public_members(
    metadata_path: Path, member_paths: Mapping[str, Path]
) -> pd.DataFrame:
    required = {"final", "model_c", "level_pred", "rolling_v1"}
    if set(member_paths) != required:
        raise ValueError(f"member paths must be exactly {sorted(required)}")
    metadata = pd.read_csv(metadata_path)
    if list(metadata.columns) != ["row_id", "time_id", "asset_id"]:
        raise ValueError("metadata columns must be row_id,time_id,asset_id")
    if metadata["row_id"].duplicated().any():
        raise ValueError("metadata row_id must be unique")
    row_id = metadata["row_id"].to_numpy(dtype=np.int64, copy=True)
    result = metadata.astype({"row_id": "int64", "time_id": "int64", "asset_id": "int64"}).copy()
    for name in ("final", "model_c", "level_pred", "rolling_v1"):
        result[name] = _read_aligned_prediction(Path(member_paths[name]), row_id, name)
    final = result["final"].to_numpy(dtype=np.float64, copy=False)
    time_id = result["time_id"].to_numpy(dtype=np.int64, copy=False)
    final_level = pd.Series(final).groupby(time_id, sort=False).transform("mean").to_numpy()
    result["L88"] = 0.88 * final_level + 1.30 * (final - final_level)
    result["final_level"] = final_level
    result["final_deviation"] = final - final_level
    return result[
        [
            "row_id",
            "time_id",
            "asset_id",
            "final",
            "L88",
            "model_c",
            "level_pred",
            "rolling_v1",
            "final_level",
            "final_deviation",
        ]
    ]


def append_public_candidate(
    public_members: pd.DataFrame, candidate_path: Path, candidate_name: str
) -> pd.DataFrame:
    if candidate_name in public_members:
        raise ValueError(f"public members already contains {candidate_name}")
    candidate = pd.read_parquet(candidate_path)
    if list(candidate.columns) != ["row_id", candidate_name]:
        raise ValueError(f"candidate columns must be row_id,{candidate_name}")
    if candidate["row_id"].duplicated().any():
        raise ValueError("candidate row_id must be unique")
    source_row = public_members["row_id"].to_numpy(dtype=np.int64, copy=False)
    candidate_row = candidate["row_id"].to_numpy(dtype=np.int64, copy=False)
    if np.array_equal(source_row, candidate_row):
        values = candidate[candidate_name].to_numpy(dtype=np.float64, copy=True)
    else:
        aligned = candidate.set_index("row_id")[candidate_name].reindex(source_row)
        if aligned.isna().any():
            raise ValueError("candidate row_id does not cover public members")
        values = aligned.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("candidate prediction contains non-finite values")
    result = public_members.copy()
    result[candidate_name] = values
    return result


def _prediction_summary(prediction: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(prediction, dtype=np.float64)
    return {
        "rows": int(values.size),
        "mean": float(values.mean()),
        "std_pop": float(values.std(ddof=0)),
        "std_sample": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_submission(
    public_members: pd.DataFrame,
    member_names: Sequence[str],
    coefficients: np.ndarray,
    output_dir: Path,
    method: str,
    verdict: str,
    recommended: bool,
    target_std: float = TARGET_OUTPUT_STD,
    extra_recipe: Mapping[str, object] | None = None,
) -> dict[str, object]:
    names = tuple(member_names)
    coef = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if coef.shape != (len(names),) or not np.isfinite(coef).all():
        raise ValueError("coefficients must match member_names and be finite")
    missing = [name for name in names if name not in public_members]
    if missing:
        raise ValueError(f"public member columns missing: {missing}")
    if public_members["row_id"].duplicated().any():
        raise ValueError("public row_id must be unique")
    matrix = public_members.loc[:, names].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("public member matrix contains non-finite values")
    raw = matrix @ coef
    calibrated = calibrate_mean_preserving_std(raw, target_std=target_std)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    submission_path = output / "submission.csv"
    pd.DataFrame(
        {
            "row_id": public_members["row_id"].to_numpy(dtype=np.int64, copy=False),
            "target": calibrated,
        }
    ).to_csv(submission_path, index=False)
    recipe: dict[str, object] = {
        "method": method,
        "parent_id": "CURRENT_DEPLOY_PARENT",
        "member_names": list(names),
        "effective_coefficients": coef.tolist(),
        "calibration": {
            "name": "mean_preserving_std",
            "target_std_pop": float(target_std),
            "formula": "mean(raw) + target_std / std_pop(raw) * (raw - mean(raw))",
        },
    }
    if extra_recipe:
        recipe.update(dict(extra_recipe))
    recipe_path = output / "recipe.json"
    _write_json(recipe_path, recipe)
    report: dict[str, object] = {
        "method": method,
        "verdict": verdict,
        "recommended": bool(recommended),
        "rows": int(len(public_members)),
        "raw_prediction": _prediction_summary(raw),
        "submission_prediction": _prediction_summary(calibrated),
    }
    report_path = output / "report.json"
    _write_json(report_path, report)
    checksums = {
        path.name: {"bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
        for path in (submission_path, recipe_path, report_path)
    }
    _write_json(output / "checksums.json", checksums)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    public = subparsers.add_parser("public-members")
    public.add_argument("--metadata", type=Path, required=True)
    public.add_argument("--final", type=Path, required=True)
    public.add_argument("--model-c", type=Path, required=True)
    public.add_argument("--level", type=Path, required=True)
    public.add_argument("--rolling-v1", type=Path, required=True)
    public.add_argument("--out", type=Path, required=True)
    submission = subparsers.add_parser("submission")
    submission.add_argument("--members", type=Path, required=True)
    submission.add_argument("--recipe", type=Path, required=True)
    submission.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "public-members":
        frame = load_public_members(
            args.metadata,
            {
                "final": args.final,
                "model_c": args.model_c,
                "level_pred": args.level,
                "rolling_v1": args.rolling_v1,
            },
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(args.out, index=False)
        print(json.dumps({"rows": len(frame), "path": str(args.out), "sha256": sha256_file(args.out)}))
    else:
        recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
        report = build_submission(
            pd.read_parquet(args.members),
            recipe["member_names"],
            np.asarray(recipe["effective_coefficients"], dtype=np.float64),
            args.out,
            recipe.get("method", args.out.name),
            recipe.get("verdict", "audit_only"),
            bool(recipe.get("recommended", False)),
            float(recipe.get("target_std", TARGET_OUTPUT_STD)),
            extra_recipe=recipe,
        )
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
