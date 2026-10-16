from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from repro.baseline00.data_contract import audit_prediction_csv
from repro.baseline00.make_submission import (
    EXPECTED_PUBLIC_TEST_ROWS,
    test_index,
    validate_submission,
)
from repro.baseline01.config import BASELINE
from repro.config import TEST_DIR, ensure_data2_path


EXPECTED_PARENT_AMPLITUDE = 1.0884372894699135
EXPECTED_CANDIDATE_AMPLITUDE = 1.0
ALLOWED_CANDIDATE_ADDITIONS = {
    "variant",
    "parent_variant",
    "rolling_operations",
    "source_report",
    "fold8_status",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_contract(
    parent: dict[str, Any],
    candidate: dict[str, Any],
    promotion: dict[str, Any],
    e1_recipe: Path,
) -> tuple[float, float]:
    if promotion.get("latest_accepted") != "e1_scale1":
        raise ValueError("promotion log has not accepted e1_scale1")
    entry = promotion.get("variants", {}).get("e1_scale1", {})
    if entry.get("passed") is not True:
        raise ValueError("promotion log does not mark e1_scale1 as passed")
    recorded_recipe = entry.get("recipe")
    if recorded_recipe is not None and Path(recorded_recipe).resolve() != e1_recipe.resolve():
        raise ValueError("promotion log E1 recipe path differs from the supplied recipe")
    if candidate.get("variant") != "e1_scale1":
        raise ValueError("candidate recipe is not e1_scale1")
    if candidate.get("parent_variant") != "e0_baseline00":
        raise ValueError("candidate recipe does not inherit e0_baseline00")
    for field, value in parent.items():
        if field != "amplitude_multiplier" and candidate.get(field) != value:
            raise ValueError(f"E1 unexpectedly changes recipe field {field}")
    unexpected = set(candidate) - set(parent) - ALLOWED_CANDIDATE_ADDITIONS
    if unexpected:
        raise ValueError(f"E1 recipe has undeclared fields: {sorted(unexpected)}")
    if tuple(candidate.get("rolling_operations", ())) != BASELINE.rolling_operations:
        raise ValueError("E1 rolling operations differ from Baseline00")
    parent_amplitude = float(parent["amplitude_multiplier"])
    candidate_amplitude = float(candidate["amplitude_multiplier"])
    if parent_amplitude != EXPECTED_PARENT_AMPLITUDE:
        raise ValueError("Baseline00 amplitude multiplier differs from the frozen value")
    if candidate_amplitude != EXPECTED_CANDIDATE_AMPLITUDE:
        raise ValueError("E1 amplitude multiplier must equal 1.0")
    return parent_amplitude, candidate_amplitude


def _validate_source_manifest(
    source: Path,
    manifest: dict[str, Any],
    expected_rows: int,
) -> dict[str, Any]:
    source = source.resolve()
    matching = [
        artifact
        for artifact in manifest.get("artifacts", [])
        if Path(artifact["path"]).resolve() == source
    ]
    if len(matching) != 1:
        raise ValueError("reference manifest must contain the source CSV exactly once")
    frozen = matching[0]
    actual_hash = _sha256(source)
    if frozen.get("sha256") != actual_hash:
        raise ValueError("source CSV SHA256 differs from the frozen reference manifest")
    if int(frozen.get("rows", -1)) != expected_rows:
        raise ValueError("source CSV row count differs from the frozen reference manifest")
    return {
        "path": str(source),
        "rows": expected_rows,
        "bytes": source.stat().st_size,
        "sha256": actual_hash,
    }


def _write_scaled_csv(
    source: Path,
    output: Path,
    ratio: float,
    chunk_rows: int,
) -> tuple[Path, int]:
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.unlink(missing_ok=True)
    rows = 0
    first = True
    try:
        for chunk in pd.read_csv(source, chunksize=chunk_rows):
            if list(chunk.columns) != ["row_id", "target"]:
                raise ValueError("source submission columns must be row_id,target")
            prediction = chunk["target"].to_numpy(dtype=np.float64, copy=False)
            if not np.isfinite(prediction).all():
                raise ValueError("source submission contains non-finite predictions")
            chunk["target"] = prediction * ratio
            chunk.to_csv(
                temporary,
                mode="w" if first else "a",
                header=first,
                index=False,
                float_format="%.17g",
            )
            rows += len(chunk)
            first = False
        if first:
            raise ValueError("source submission is empty")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary, rows


def build_e1_submission(
    *,
    source: Path,
    output: Path,
    parent_recipe: Path,
    e1_recipe: Path,
    promotion_log: Path,
    reference_manifest: Path,
    test_dir: Path,
    report_output: Path,
    expected_rows: int = EXPECTED_PUBLIC_TEST_ROWS,
    chunk_rows: int = 250_000,
) -> dict[str, Any]:
    source = Path(source).resolve()
    output = ensure_data2_path(Path(output)).resolve()
    report_output = ensure_data2_path(Path(report_output)).resolve()
    parent_recipe = Path(parent_recipe).resolve()
    e1_recipe = Path(e1_recipe).resolve()
    reference_manifest = Path(reference_manifest).resolve()
    promotion_log = Path(promotion_log).resolve()

    parent = _read_json(parent_recipe)
    candidate = _read_json(e1_recipe)
    promotion = _read_json(promotion_log)
    parent_amplitude, candidate_amplitude = _validate_contract(
        parent, candidate, promotion, e1_recipe
    )
    source_metadata = _validate_source_manifest(
        source, _read_json(reference_manifest), expected_rows
    )
    expected_row_id, _ = test_index(Path(test_dir))
    if expected_row_id.size != expected_rows:
        raise ValueError(
            f"official test index has {expected_row_id.size} rows, expected {expected_rows}"
        )
    source_validation, source_prediction = validate_submission(expected_row_id, source)
    ratio = candidate_amplitude / parent_amplitude
    temporary: Path | None = None
    try:
        temporary, written_rows = _write_scaled_csv(
            source, output, ratio, chunk_rows
        )
        if written_rows != expected_rows:
            raise ValueError(f"wrote {written_rows} rows, expected {expected_rows}")
        output_validation, output_prediction = validate_submission(
            expected_row_id, temporary
        )
        expected_prediction = source_prediction * ratio
        absolute_error = np.abs(output_prediction - expected_prediction)
        max_absolute_error = float(absolute_error.max(initial=0.0))
        if max_absolute_error > 1e-15:
            raise ValueError(
                "serialized E1 predictions differ from exact rescaling by "
                f"{max_absolute_error}"
            )
        output_audit = audit_prediction_csv(temporary, chunk_rows=chunk_rows)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    output_validation["path"] = str(output)
    output_audit["path"] = str(output)
    report = {
        "variant": "e1_scale1",
        "method": "exact_final_amplitude_rescaling",
        "source": source_metadata,
        "output": output_audit,
        "source_validation": source_validation,
        "output_validation": output_validation,
        "parent_amplitude_multiplier": parent_amplitude,
        "candidate_amplitude_multiplier": candidate_amplitude,
        "rescaling_ratio": ratio,
        "api_replayed": False,
        "equivalence": {
            "formula": "e1 = baseline00 * candidate_amplitude / parent_amplitude",
            "rows_compared": expected_rows,
            "max_absolute_error": max_absolute_error,
            "exact_after_csv_roundtrip": bool(max_absolute_error == 0.0),
        },
        "provenance": {
            "parent_recipe": str(parent_recipe),
            "parent_recipe_sha256": _sha256(parent_recipe),
            "e1_recipe": str(e1_recipe),
            "e1_recipe_sha256": _sha256(e1_recipe),
            "promotion_log": str(promotion_log),
            "promotion_log_sha256": _sha256(promotion_log),
            "reference_manifest": str(reference_manifest),
            "reference_manifest_sha256": _sha256(reference_manifest),
        },
        "submitted_to_platform": False,
    }
    _atomic_write_json(report_output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent-recipe", type=Path, required=True)
    parser.add_argument("--e1-recipe", type=Path, required=True)
    parser.add_argument("--promotion-log", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_PUBLIC_TEST_ROWS)
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    args = parser.parse_args()
    report = build_e1_submission(
        source=args.source,
        output=args.out,
        parent_recipe=args.parent_recipe,
        e1_recipe=args.e1_recipe,
        promotion_log=args.promotion_log,
        reference_manifest=args.reference_manifest,
        test_dir=args.test_dir,
        report_output=args.report_out,
        expected_rows=args.expected_rows,
        chunk_rows=args.chunk_rows,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
