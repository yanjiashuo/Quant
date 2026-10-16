from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise ValueError(f"input file does not exist: {source}")
    return {
        "path": str(source),
        "bytes": int(source.stat().st_size),
        "sha256": sha256_file(source),
    }


def build_input_manifest(
    train_dir: Path,
    test_dir: Path,
    sample_submission: Path,
    source_files: Sequence[Path],
    source_commit: str,
    output: Path,
) -> dict[str, Any]:
    train_paths = sorted(Path(train_dir).glob("train_partition_*.parquet"))
    test_paths = sorted(Path(test_dir).glob("test_partition_*.parquet"))
    if len(train_paths) != 9:
        raise ValueError(f"expected 9 train partitions; got {len(train_paths)}")
    if len(test_paths) != 3:
        raise ValueError(f"expected 3 test partitions; got {len(test_paths)}")
    if not source_commit.strip():
        raise ValueError("source_commit must be nonempty")
    payload = {
        "variant": "g0_gap0_parent",
        "source_commit": source_commit.strip(),
        "train_partitions": [_record(path) for path in train_paths],
        "test_partitions": [_record(path) for path in test_paths],
        "sample_submission": _record(sample_submission),
        "source_files": [_record(path) for path in source_files],
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--source-file", type=Path, action="append", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = build_input_manifest(
        args.train_dir,
        args.test_dir,
        args.sample_submission,
        args.source_file,
        args.source_commit,
        args.out,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
