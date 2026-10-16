from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


SHARED_ARTIFACTS = (
    "weight_model.txt",
    "level_model.txt",
    "rolling_v1.txt",
    "rolling_history.npz",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reuse_shared_artifacts(source_dir: Path, destination_dir: Path) -> dict[str, Any]:
    source = Path(source_dir).resolve()
    destination = Path(destination_dir).resolve()
    if source == destination:
        raise ValueError("source and destination model directories must differ")
    missing = [name for name in SHARED_ARTIFACTS if not (source / name).is_file()]
    if missing:
        raise ValueError(f"source shared model is missing {missing}")
    destination.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    for name in SHARED_ARTIFACTS:
        source_path = source / name
        destination_path = destination / name
        temporary_path = destination / f".{name}.tmp"
        shutil.copy2(source_path, temporary_path)
        os.replace(temporary_path, destination_path)
        source_hash = _sha256(source_path)
        destination_hash = _sha256(destination_path)
        if source_hash != destination_hash:
            raise IOError(f"hash mismatch while reusing {name}")
        artifacts[name] = {
            "source": str(source_path),
            "destination": str(destination_path),
            "bytes": destination_path.stat().st_size,
            "source_sha256": source_hash,
            "destination_sha256": destination_hash,
        }
    return {
        "source_dir": str(source),
        "destination_dir": str(destination),
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = reuse_shared_artifacts(args.source, args.destination)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
