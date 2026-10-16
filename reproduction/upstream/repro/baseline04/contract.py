from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from repro.baseline04.config import (
    BASE_OOF_FILENAMES,
    CHILD_MEMBERS,
    EXPECTED_BASE_OOF_MD5,
    EXPECTED_SHA256,
    FOLDS,
    PARENT_MEMBERS,
    RIDGE_ALPHA,
    TARGET_STD,
)


@dataclass(frozen=True)
class InputPaths:
    base_oof_dir: Path
    cs_rank_dir: Path
    long_oof_dir: Path
    parent_recipe: Path
    parent_report: Path
    parent_submission: Path
    public_members: Path
    public_model_c_long: Path

    def files_by_contract_name(self) -> dict[str, Path]:
        files = {
            "parent_recipe": Path(self.parent_recipe),
            "parent_report": Path(self.parent_report),
            "parent_submission": Path(self.parent_submission),
            "public_members": Path(self.public_members),
            "public_model_c_long": Path(self.public_model_c_long),
        }
        for fold in FOLDS:
            files[f"base_fold_{fold}"] = (
                Path(self.base_oof_dir) / BASE_OOF_FILENAMES[fold]
            )
            files[f"cs_rank_fold_{fold}"] = Path(self.cs_rank_dir) / f"fold_{fold}.npz"
            files[f"long_fold_{fold}"] = Path(self.long_oof_dir) / f"fold_{fold}.npz"
        return files


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    return file_digest(path, "sha256")


def write_json(path: Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(destination)


def audit_inputs(
    paths: InputPaths,
    output: Path,
    expected_sha256: Mapping[str, str] | None = None,
    expected_base_md5: Mapping[int, str] | None = None,
) -> dict[str, object]:
    expected_sha = EXPECTED_SHA256 if expected_sha256 is None else expected_sha256
    expected_md5 = (
        EXPECTED_BASE_OOF_MD5 if expected_base_md5 is None else expected_base_md5
    )
    records: dict[str, object] = {}
    for name, path in paths.files_by_contract_name().items():
        if not path.is_file():
            raise ValueError(f"{name} input is missing: {path}")
        sha256 = sha256_file(path)
        expected = expected_sha.get(name)
        if expected is not None and sha256 != expected:
            raise ValueError(f"{name} SHA256 differs: {sha256} != {expected}")
        record: dict[str, object] = {
            "path": str(path.resolve()),
            "bytes": int(path.stat().st_size),
            "sha256": sha256,
        }
        if name.startswith("base_fold_"):
            fold = int(name.rsplit("_", 1)[1])
            md5 = file_digest(path, "md5")
            expected_fold_md5 = expected_md5.get(fold)
            if expected_fold_md5 is not None and md5 != expected_fold_md5:
                raise ValueError(
                    f"{name} MD5 differs: {md5} != {expected_fold_md5}"
                )
            record["md5"] = md5
        records[name] = record
    payload = {
        "parent_members": list(PARENT_MEMBERS),
        "child_members": list(CHILD_MEMBERS),
        "ridge_alpha": RIDGE_ALPHA,
        "target_std_pop": TARGET_STD,
        "files": records,
    }
    write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-oof-dir", type=Path, required=True)
    parser.add_argument("--cs-rank-dir", type=Path, required=True)
    parser.add_argument("--long-oof-dir", type=Path, required=True)
    parser.add_argument("--parent-recipe", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--parent-submission", type=Path, required=True)
    parser.add_argument("--public-members", type=Path, required=True)
    parser.add_argument("--public-model-c-long", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit_inputs(
        InputPaths(
            base_oof_dir=args.base_oof_dir,
            cs_rank_dir=args.cs_rank_dir,
            long_oof_dir=args.long_oof_dir,
            parent_recipe=args.parent_recipe,
            parent_report=args.parent_report,
            parent_submission=args.parent_submission,
            public_members=args.public_members,
            public_model_c_long=args.public_model_c_long,
        ),
        args.out,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
