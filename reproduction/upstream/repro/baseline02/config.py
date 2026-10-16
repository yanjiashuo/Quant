from __future__ import annotations

from pathlib import Path


PARENT_MEMBERS = ("final", "L88", "model_c", "level_pred", "rolling_v1")
OOF_FILENAMES = {
    6: "fold6_five_member_oof.npz",
    7: "fold7_five_member_oof.npz",
    8: "fold8_five_member_oof.npz",
}
OOF_MD5 = {
    6: "ec0b39bcb9e463e8d93384d300f432c5",
    7: "112578112fdfc09430a37658013bd565",
    8: "bdfdd54eeb64035c932662a9df8dba09",
}
PARENT_RIDGE_ALPHA = 0.01
DEVELOPMENT_DELTA = 0.000020
CONFIRMATION_MEAN_DELTA = 0.000020
DEVELOPMENT_POSITIVE_BLOCKS = 12
CONFIRMATION_POSITIVE_BLOCKS = 10
N_TIME_BLOCKS = 20

CS_RANK_WINDOW = 4
CS_RANK_LGB = {
    "n_estimators": 201,
    "num_leaves": 63,
    "learning_rate": 0.03,
    "min_data_in_leaf": 2_000,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 0,
    "deterministic": True,
    "force_col_wise": True,
}

ROLLING_V2_LGB = {
    "n_estimators": 200,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 0,
    "deterministic": True,
    "force_col_wise": True,
}


def oof_paths(root: Path) -> dict[int, Path]:
    root = Path(root)
    return {fold: root / filename for fold, filename in OOF_FILENAMES.items()}
