from __future__ import annotations

from dataclasses import dataclass, replace


BASELINE_MEMBERS = ("final", "L88", "model_c", "level", "rolling_v1")
SIX_OPERATIONS = (
    "lag1",
    "lag2",
    "lag5",
    "roll_mean5",
    "roll_mean10",
    "delta1",
)
FOUR_OPERATIONS = ("lag1", "roll_mean5", "roll_mean10", "delta1")

REPORT_FOLDS = (6, 7, 8)
FIT_FOLDS_BY_REPORT = {6: (5,), 7: (5, 6), 8: (5, 6, 7)}

E0_REFERENCE_TOLERANCE = 1e-8
E0_REFERENCE_SCORES = {
    6: {
        "raw_score": 0.002240041566661133,
        "calibrated_score": 0.0022243279066535004,
    },
    7: {
        "raw_score": 0.002049878831763219,
        "calibrated_score": 0.002049622359017844,
    },
    8: {
        "raw_score": 0.0018318370413572538,
        "calibrated_score": 0.001741494513040176,
    },
}


@dataclass(frozen=True)
class ResolvedVariant:
    name: str
    member_names: tuple[str, ...] = BASELINE_MEMBERS
    rolling_operations: tuple[str, ...] = SIX_OPERATIONS
    amplitude_multiplier: float = 1.0884372895


@dataclass(frozen=True)
class VariantPatch:
    name: str
    changed_factor: str
    member_names: tuple[str, ...] | None = None
    rolling_operations: tuple[str, ...] | None = None
    amplitude_multiplier: float | None = None


BASELINE = ResolvedVariant("e0_baseline00")

VARIANT_PATCHES = {
    "e1_scale1": VariantPatch(
        "e1_scale1",
        "amplitude_multiplier",
        amplitude_multiplier=1.0,
    ),
    "e2_rolling4": VariantPatch(
        "e2_rolling4",
        "rolling_operations",
        rolling_operations=FOUR_OPERATIONS,
    ),
    "e3_drop_model_c": VariantPatch(
        "e3_drop_model_c",
        "member_names",
        member_names=("final", "L88", "level", "rolling_v1"),
    ),
}


def resolve_variant(parent: ResolvedVariant, patch: VariantPatch) -> ResolvedVariant:
    values = {
        "member_names": patch.member_names,
        "rolling_operations": patch.rolling_operations,
        "amplitude_multiplier": patch.amplitude_multiplier,
    }
    changed = {name: value for name, value in values.items() if value is not None}
    if tuple(changed) != (patch.changed_factor,):
        raise ValueError(f"{patch.name} must override only {patch.changed_factor}")
    return replace(parent, name=patch.name, **changed)


GATE = {
    "minimum_mean_delta": 0.00002,
    "minimum_nonnegative_folds": 2,
    "minimum_single_fold_delta": -0.00003,
}

E1_CORRECTNESS_GATE = {
    "mode": "never_harm",
    "minimum_single_fold_delta": 0.0,
}

GATE_BY_VARIANT = {
    "e1_scale1": "correctness",
    "e2_rolling4": "numeric",
    "e3_drop_model_c": "numeric",
}

# E1 intentionally changes amplitude. The isolated E2/E3 controls must instead
# share the parent's chronological legacy calibration on every report fold.
AMPLITUDE_POLICY_BY_VARIANT = {
    "e1_scale1": "candidate_fixed",
    "e2_rolling4": "shared_parent_legacy",
    "e3_drop_model_c": "shared_parent_legacy",
}
