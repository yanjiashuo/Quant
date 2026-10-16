from __future__ import annotations

from dataclasses import dataclass

from repro.baseline00 import MEMBER_NAMES
from repro.baseline00.rolling import V1_OPERATIONS


@dataclass(frozen=True)
class Baseline0002Variant:
    name: str = "baseline00_02_strict_recent_gap0"
    oof_folds: tuple[int, ...] = (6, 7, 8)
    validation_gap_time_ids: int = 0
    production_gap_time_ids: int = 0
    base_seed: int = 0
    shared_seed: int = 0
    member_names: tuple[str, ...] = MEMBER_NAMES
    rolling_operations: tuple[str, ...] = V1_OPERATIONS
    ridge_alpha: float = 0.5
    scale_source_fold: int = 8


VARIANT = Baseline0002Variant()

EXPECTED_ITERATIONS = {
    "model_a": 139,
    "model_b": 56,
    "model_c": 201,
    "weight_model": 400,
    "level_model": 60,
    "rolling_v1": 200,
}
