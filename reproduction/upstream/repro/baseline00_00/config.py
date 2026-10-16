from __future__ import annotations

from dataclasses import dataclass

from repro.baseline00 import MEMBER_NAMES
from repro.baseline00.rolling import V1_OPERATIONS


@dataclass(frozen=True)
class Baseline0000Variant:
    name: str
    validation_gap_time_ids: int
    production_gap_time_ids: int
    base_seed: int
    shared_seed: int
    rolling_operations: tuple[str, ...]
    member_names: tuple[str, ...]
    target_prediction_std: float


VARIANT = Baseline0000Variant(
    name="baseline00_00_github_gap0",
    validation_gap_time_ids=0,
    production_gap_time_ids=0,
    base_seed=42,
    shared_seed=0,
    rolling_operations=V1_OPERATIONS,
    member_names=MEMBER_NAMES,
    target_prediction_std=0.046,
)
