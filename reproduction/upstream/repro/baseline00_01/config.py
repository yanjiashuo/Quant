from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantRecipe:
    name: str
    validation_gap_time_ids: int
    production_gap_time_ids: int
    base_seed: int
    shared_seed: int


VARIANT_A = VariantRecipe("baseline00_01a_gap0", 500, 0, 0, 0)
VARIANT_B = VariantRecipe("baseline00_01b_github_seed42", 500, 0, 42, 0)

BASE_COMPONENTS = frozenset({"model_a", "model_b", "model_c"})
SHARED_COMPONENTS = frozenset({"weight_model", "level_model", "rolling_v1"})


def resolve_production_cutoff(
    test_min_time_id: int, production_gap_time_ids: int
) -> int:
    if production_gap_time_ids < 0:
        raise ValueError("production gap must be nonnegative")
    return int(test_min_time_id) - int(production_gap_time_ids)


def seed_for(component: str, variant: VariantRecipe) -> int:
    if component in BASE_COMPONENTS:
        return variant.base_seed
    if component in SHARED_COMPONENTS:
        return variant.shared_seed
    raise ValueError(f"unknown training component: {component}")


def identify_variant(production_gap_time_ids: int, random_state: int) -> str:
    if (
        production_gap_time_ids == VARIANT_A.production_gap_time_ids
        and random_state == VARIANT_A.base_seed == VARIANT_A.shared_seed
    ):
        return VARIANT_A.name
    if production_gap_time_ids == 500 and random_state == 0:
        return "baseline00"
    return "custom"
