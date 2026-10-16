from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from repro.baseline00_02.config import VARIANT
from repro.config import ensure_data2_path


CSV_CHUNK_ROWS = 250_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunk_arrays(frame: pd.DataFrame, path: Path) -> tuple[np.ndarray, np.ndarray]:
    if list(frame.columns) != ["row_id", "target"]:
        raise ValueError(f"{path} must contain exactly row_id,target in that order")
    row_id = frame["row_id"].to_numpy(dtype=np.int64, copy=False)
    target = frame["target"].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(target).all():
        raise ValueError(f"{path} contains non-finite target values")
    return row_id, target


def _source_statistics(path: Path, c8: float) -> dict[str, Any]:
    seen: set[int] = set()
    rows = 0
    raw_sum = 0.0
    raw_sum_sq = 0.0
    raw_min = np.inf
    raw_max = -np.inf
    for frame in pd.read_csv(path, chunksize=CSV_CHUNK_ROWS):
        row_id, cstar = _chunk_arrays(frame, path)
        unique = np.unique(row_id)
        if unique.size != row_id.size or any(int(value) in seen for value in unique):
            raise ValueError(f"{path} contains duplicate row_id values")
        seen.update(int(value) for value in unique)
        raw = cstar / c8
        rows += len(raw)
        raw_sum += float(np.sum(raw, dtype=np.float64))
        raw_sum_sq += float(np.dot(raw, raw))
        raw_min = min(raw_min, float(raw.min()))
        raw_max = max(raw_max, float(raw.max()))
    if rows == 0:
        raise ValueError(f"{path} is empty")
    raw_mean = raw_sum / rows
    raw_variance = max(raw_sum_sq / rows - raw_mean * raw_mean, 0.0)
    raw_std = float(np.sqrt(raw_variance))
    if raw_std <= 1e-12:
        raise ValueError("raw prediction standard deviation is zero")
    return {
        "rows": int(rows),
        "raw_mean": float(raw_mean),
        "raw_std": raw_std,
        "raw_min": float(raw_min),
        "raw_max": float(raw_max),
    }


def _file_summary(path: Path) -> dict[str, Any]:
    rows = 0
    total = 0.0
    total_sq = 0.0
    minimum = np.inf
    maximum = -np.inf
    for frame in pd.read_csv(path, chunksize=CSV_CHUNK_ROWS):
        _, values = _chunk_arrays(frame, path)
        rows += len(values)
        total += float(np.sum(values, dtype=np.float64))
        total_sq += float(np.dot(values, values))
        minimum = min(minimum, float(values.min()))
        maximum = max(maximum, float(values.max()))
    mean = total / rows
    return {
        "path": str(path),
        "rows": int(rows),
        "mean": float(mean),
        "std": float(np.sqrt(max(total_sq / rows - mean * mean, 0.0))),
        "min": float(minimum),
        "max": float(maximum),
        "all_finite": True,
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def derive_audit_submissions(
    cstar_csv: Path, c8: float, output_dir: Path
) -> dict[str, Any]:
    amplitude = float(c8)
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("c8 must be positive and finite")
    source = Path(cstar_csv).resolve()
    output = ensure_data2_path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected_source = (output / "baseline00_02_cstar.csv").resolve()
    if source != expected_source:
        raise ValueError("primary source must be output_dir/baseline00_02_cstar.csv")

    source_stats = _source_statistics(source, amplitude)
    raw_mean = float(source_stats["raw_mean"])
    raw_std = float(source_stats["raw_std"])
    destinations = {
        "raw": output / "baseline00_02_raw.csv",
        "std046_audit": output / "baseline00_02_std046_audit.csv",
    }
    temporary = {
        name: path.parent / f".{path.name}.tmp" for name, path in destinations.items()
    }
    handles = {
        name: path.open("w", encoding="utf-8", newline="")
        for name, path in temporary.items()
    }
    try:
        try:
            first = True
            for frame in pd.read_csv(source, chunksize=CSV_CHUNK_ROWS):
                row_id, cstar = _chunk_arrays(frame, source)
                raw = cstar / amplitude
                values = {
                    "raw": raw,
                    "std046_audit": raw_mean
                    + (0.046 / raw_std) * (raw - raw_mean),
                }
                for name, prediction in values.items():
                    pd.DataFrame({"row_id": row_id, "target": prediction}).to_csv(
                        handles[name], index=False, header=first
                    )
                first = False
        finally:
            for handle in handles.values():
                handle.close()
    except BaseException:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
    for name, path in destinations.items():
        os.replace(temporary[name], path)

    report = {
        "variant_name": VARIANT.name,
        "c8": amplitude,
        "derivation": {
            "raw": "cstar / c8",
            "std046_audit": (
                "mean(raw) + (0.046 / std_pop(raw)) * (raw - mean(raw))"
            ),
        },
        "source_statistics": source_stats,
        "primary": _file_summary(source),
        "raw": _file_summary(destinations["raw"]),
        "std046_audit": _file_summary(destinations["std046_audit"]),
    }
    report_path = output / "baseline00_02_family_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    if recipe.get("variant_name") != VARIANT.name:
        raise ValueError("recipe variant does not match Baseline00.02")
    report = derive_audit_submissions(
        args.source, float(recipe["amplitude_multiplier"]), args.out
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
