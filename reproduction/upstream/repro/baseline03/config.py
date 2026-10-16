from __future__ import annotations

from dataclasses import dataclass


PARENT_MEMBERS = ("final", "L88", "model_c", "level", "rolling_v1")
CANDIDATE_MEMBERS = (*PARENT_MEMBERS, "model_c_long")
OOF_FOLDS = (6, 7, 8)
FIT_FOLDS_BY_REPORT = {7: (6,), 8: (6, 7)}
PRODUCTION_RIDGE_FOLDS = OOF_FOLDS
ROLLING_OPERATIONS = (
    "lag1",
    "lag2",
    "lag5",
    "roll_mean5",
    "roll_mean10",
    "delta1",
)
AMPLITUDE_MULTIPLIER = 1.0884372894699135
RIDGE_ALPHA = 0.5


@dataclass(frozen=True)
class VariantConfig:
    name: str
    member_names: tuple[str, ...]
    validation_gap_time_ids: int = 0
    production_gap_time_ids: int = 0
    ridge_fit_folds: tuple[int, ...] = PRODUCTION_RIDGE_FOLDS
    ridge_alpha: float = RIDGE_ALPHA
    amplitude_multiplier: float = AMPLITUDE_MULTIPLIER
    random_state: int = 0
    rolling_operations: tuple[str, ...] = ROLLING_OPERATIONS


G0 = VariantConfig(name="g0_gap0_parent", member_names=PARENT_MEMBERS)
E1 = VariantConfig(name="e1_model_c_long", member_names=CANDIDATE_MEMBERS)


MODEL_C_LONG_RECIPE = {
    "window": "all",
    "n_estimators": 201,
    "num_leaves": 63,
    "learning_rate": 0.03,
    "min_data_in_leaf": 2_000,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "prediction_scale": 1.1921623492997822,
    "prediction_clip": 3.0,
    "label_mode": "target_demean",
    "random_state": 0,
    "deterministic": True,
    "force_col_wise": True,
}
