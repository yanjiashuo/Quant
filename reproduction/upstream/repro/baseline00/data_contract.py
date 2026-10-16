from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from repro.config import DATA, REPO


N_FEATURES = 323
FEATURE_COLUMNS = [f"feature_{index:03d}" for index in range(N_FEATURES)]


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_prediction_csv(path: Path, chunk_rows: int = 250_000) -> dict[str, Any]:
    path = Path(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if len(header) != 2 or header[0] != "row_id":
        raise ValueError(f"{path} must contain row_id and exactly one prediction column")
    value_column = header[1]
    semantic = hashlib.sha256()
    rows = 0
    total = 0.0
    total_sq = 0.0
    minimum = np.inf
    maximum = -np.inf
    previous_row_id: int | None = None

    row_dtype = np.dtype([("row_id", "<i8"), ("value", "<f8")])
    for chunk in pd.read_csv(path, chunksize=chunk_rows):
        row_id = chunk["row_id"].to_numpy(dtype=np.int64, copy=False)
        value = chunk[value_column].to_numpy(dtype=np.float64, copy=False)
        if row_id.size == 0:
            continue
        if not np.isfinite(value).all():
            raise ValueError(f"{path} contains non-finite predictions")
        if np.any(np.diff(row_id) != 1):
            raise ValueError(f"{path} row_id is not strictly contiguous")
        if previous_row_id is not None and int(row_id[0]) != previous_row_id + 1:
            raise ValueError(f"{path} row_id is not contiguous across chunks")
        previous_row_id = int(row_id[-1])

        records = np.empty(row_id.size, dtype=row_dtype)
        records["row_id"] = row_id
        records["value"] = value
        semantic.update(records.tobytes())
        rows += row_id.size
        total += float(value.sum(dtype=np.float64))
        total_sq += float(np.dot(value, value))
        minimum = min(minimum, float(value.min()))
        maximum = max(maximum, float(value.max()))

    if rows == 0:
        raise ValueError(f"{path} is empty")
    mean = total / rows
    variance = max(total_sq / rows - mean * mean, 0.0)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "columns": header,
        "prediction_column": value_column,
        "rows": int(rows),
        "first_row_id": int(pd.read_csv(path, usecols=["row_id"], nrows=1).iloc[0, 0]),
        "last_row_id": int(previous_row_id),
        "value_mean": mean,
        "value_std": float(np.sqrt(variance)),
        "value_min": minimum,
        "value_max": maximum,
        "md5": file_hash(path, "md5"),
        "sha256": file_hash(path, "sha256"),
        "semantic_sha256": semantic.hexdigest(),
    }


def _versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for distribution, key in (
        ("pyarrow", "pyarrow"),
        ("lightgbm", "lightgbm"),
        ("scikit-learn", "sklearn"),
    ):
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = "not-installed"
    return versions


def build_reference_manifest(repo: Path, output: Path) -> dict[str, Any]:
    repo = Path(repo).resolve()
    paths = [
        repo / "artifacts" / "submissions" / "rolling_stack_std46.csv",
        repo / "artifacts" / "member_predictions" / "public_model_c.csv",
        repo / "artifacts" / "member_predictions" / "public_rolling_v1.csv",
    ]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = os.environ.get("SOURCE_GIT_COMMIT", "unavailable")
    result = {
        "git_commit": commit,
        "platform": platform.platform(),
        "versions": _versions(),
        "artifacts": [audit_prediction_csv(path) for path in paths],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def validate_columns(columns: list[str], dataset: str) -> dict[str, Any]:
    column_set = set(columns)
    required = {"row_id", "time_id", "asset_id", *FEATURE_COLUMNS}
    if dataset == "train":
        required.update({"weight", "target"})
    missing = sorted(required - column_set)
    if missing:
        raise ValueError(f"{dataset} schema missing columns: {missing[:10]}")
    forbidden = {"weight", "target"} if dataset == "test" else set()
    forbidden.update(column for column in columns if dataset == "test" and column.startswith("responder_"))
    present_forbidden = sorted(forbidden & column_set)
    if present_forbidden:
        raise ValueError(f"test schema exposes forbidden columns: {present_forbidden[:10]}")
    responders = [column for column in columns if column.startswith("responder_")]
    if dataset == "train" and not responders:
        raise ValueError("train schema must contain responder columns")
    return {
        "columns": len(columns),
        "feature_columns": sum(column.startswith("feature_") for column in columns),
        "responder_columns": len(responders),
    }


def _partition_audit(path: Path, dataset: str) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    columns = parquet.schema_arrow.names
    schema = validate_columns(columns, dataset)
    time_values = pq.read_table(path, columns=["time_id"])["time_id"].to_numpy()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "rows": int(parquet.metadata.num_rows),
        "row_groups": int(parquet.metadata.num_row_groups),
        "min_time_id": int(time_values.min()),
        "max_time_id": int(time_values.max()),
        **schema,
    }


def audit_data_root(data_root: Path, output: Path | None = None) -> dict[str, Any]:
    data_root = Path(data_root).resolve()
    train_paths = sorted((data_root / "train").glob("train_partition_*.parquet"))
    test_paths = sorted((data_root / "test").glob("test_partition_*.parquet"))
    if len(train_paths) != 9:
        raise ValueError(f"expected 9 train partitions, found {len(train_paths)}")
    if not test_paths:
        raise ValueError("no test partitions found")
    train = [_partition_audit(path, "train") for path in train_paths]
    test = [_partition_audit(path, "test") for path in test_paths]
    for dataset_name, partitions in (("train", train), ("test", test)):
        for left, right in zip(partitions, partitions[1:]):
            if left["max_time_id"] >= right["min_time_id"]:
                raise ValueError(f"{dataset_name} partition time ranges overlap or reverse")
    result = {
        "data_root": str(data_root),
        "train": train,
        "test": test,
        "train_rows": int(sum(item["rows"] for item in train)),
        "test_rows": int(sum(item["rows"] for item in test)),
    }
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    if args.reference:
        result = build_reference_manifest(args.repo, args.out)
    else:
        result = audit_data_root(args.data_root, args.out)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
