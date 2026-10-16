from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from repro.baseline00_01 import BASELINE00_MAIN_AMPLITUDE


CSV_CHUNK_ROWS = 250_000


def derive_predictions(
    main: np.ndarray, frozen_amplitude: float
) -> dict[str, np.ndarray]:
    if not np.isfinite(frozen_amplitude) or frozen_amplitude <= 0.0:
        raise ValueError("frozen amplitude must be positive and finite")
    main_array = np.asarray(main, dtype=np.float64)
    if not np.isfinite(main_array).all():
        raise ValueError("main predictions must all be finite")
    raw = main_array / frozen_amplitude
    raw_std = float(np.std(raw))
    if raw_std <= 1e-12:
        raise ValueError("raw prediction standard deviation is zero")
    return {
        "raw": raw,
        "main": main_array,
        "std045": raw * (0.045 / raw_std),
        "std046": raw * (0.046 / raw_std),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_chunk(frame: pd.DataFrame, path: Path) -> tuple[np.ndarray, np.ndarray]:
    if list(frame.columns) != ["row_id", "target"]:
        raise ValueError(f"{path} must contain exactly row_id,target in that order")
    row_id = frame["row_id"].to_numpy(dtype=np.int64, copy=False)
    target = frame["target"].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(target).all():
        raise ValueError(f"{path} contains non-finite predictions")
    return row_id, target


def _source_pass(path: Path, frozen_amplitude: float) -> dict[str, Any]:
    count = 0
    total = 0.0
    total_sq = 0.0
    previous_row_id: int | None = None
    for frame in pd.read_csv(path, chunksize=CSV_CHUNK_ROWS):
        row_id, main = _validate_chunk(frame, path)
        if len(row_id) and previous_row_id is not None and int(row_id[0]) == previous_row_id:
            raise ValueError(f"{path} repeats a row_id across chunk boundaries")
        raw = main / frozen_amplitude
        count += len(raw)
        total += float(np.sum(raw, dtype=np.float64))
        total_sq += float(np.dot(raw, raw))
        if len(row_id):
            previous_row_id = int(row_id[-1])
    if count == 0:
        raise ValueError(f"{path} is empty")
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    std = float(np.sqrt(variance))
    if std <= 1e-12:
        raise ValueError("raw prediction standard deviation is zero")
    return {"rows": int(count), "raw_mean": mean, "raw_std": std}


def _file_summary(path: Path) -> dict[str, Any]:
    count = 0
    total = 0.0
    total_sq = 0.0
    minimum = np.inf
    maximum = -np.inf
    for frame in pd.read_csv(path, chunksize=CSV_CHUNK_ROWS):
        _, values = _validate_chunk(frame, path)
        count += len(values)
        total += float(np.sum(values, dtype=np.float64))
        total_sq += float(np.dot(values, values))
        minimum = min(minimum, float(values.min()))
        maximum = max(maximum, float(values.max()))
    mean = total / count
    return {
        "path": str(path),
        "rows": int(count),
        "mean": mean,
        "std": float(np.sqrt(max(total_sq / count - mean * mean, 0.0))),
        "min": minimum,
        "max": maximum,
        "all_finite": True,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def derive_amplitude_family(
    source_csv: Path,
    frozen_amplitude: float,
    output_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    source = Path(source_csv).resolve()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(frozen_amplitude) or frozen_amplitude <= 0.0:
        raise ValueError("frozen amplitude must be positive and finite")
    source_stats = _source_pass(source, frozen_amplitude)
    raw_std = float(source_stats["raw_std"])
    paths = {name: output / f"{prefix}_{name}.csv" for name in ("raw", "main", "std045", "std046")}
    if paths["main"].resolve() != source:
        raise ValueError("source CSV must be the declared main output path")

    handles = {name: paths[name].open("w", encoding="utf-8", newline="") for name in ("raw", "std045", "std046")}
    try:
        first = True
        for frame in pd.read_csv(source, chunksize=CSV_CHUNK_ROWS):
            row_id, main = _validate_chunk(frame, source)
            raw = main / frozen_amplitude
            values = {
                "raw": raw,
                "std045": raw * (0.045 / raw_std),
                "std046": raw * (0.046 / raw_std),
            }
            for name, target in values.items():
                pd.DataFrame({"row_id": row_id, "target": target}).to_csv(
                    handles[name], index=False, header=first
                )
            first = False
    finally:
        for handle in handles.values():
            handle.close()

    report = {
        "prefix": prefix,
        "frozen_amplitude_multiplier": float(frozen_amplitude),
        "derivation": {
            "raw": "main / frozen_amplitude_multiplier",
            "std045": "raw * (0.045 / std(raw))",
            "std046": "raw * (0.046 / std(raw))",
        },
        "files": {name: _file_summary(path) for name, path in paths.items()},
    }
    report_path = output / f"{prefix}_family_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["a", "b", "00"], required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frozen-amplitude", type=float)
    args = parser.parse_args()

    if args.variant == "a":
        amplitude = BASELINE00_MAIN_AMPLITUDE
        prefix = "baseline00_01a"
    elif args.variant == "b":
        if args.frozen_amplitude is None:
            raise ValueError("variant b requires --frozen-amplitude")
        amplitude = args.frozen_amplitude
        prefix = "baseline00_01b"
    else:
        if args.frozen_amplitude is None:
            raise ValueError("variant 00 requires --frozen-amplitude")
        amplitude = args.frozen_amplitude
        prefix = "baseline00_00"
    report = derive_amplitude_family(args.source, amplitude, args.out, prefix)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
