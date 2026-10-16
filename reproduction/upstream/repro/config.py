"""Central config for reproducing rolling_stack_std46 (public 0.0034426160)."""
from __future__ import annotations

import os
from pathlib import Path

# --- roots ---
QROOT_RAW = os.environ.get("QROOT")
QROOT = Path(QROOT_RAW).resolve() if QROOT_RAW else None

REPO = (QROOT / "repo") if QROOT else Path(__file__).resolve().parents[1]
DATA = (
    QROOT / "data"
    if QROOT
    else Path(os.environ.get("QUANT_DATA_ROOT", r"E:\QuantData\public_release_20260630\data"))
)
TRAIN_DIR = DATA / "train"
TEST_DIR = DATA / "test"
API_DIR = (
    QROOT / "timeseries_api"
    if QROOT
    else Path(
        os.environ.get(
            "QUANT_API_ROOT", r"E:\QuantData\public_release_20260630\timeseries_api"
        )
    )
)
DEPLOY_DIR = REPO / "artifacts" / "models" / "final_deploy_package"

# Imports never create directories. Entrypoints create only the exact output directory they need.
WORK = (
    QROOT / "work" / "baseline00"
    if QROOT
    else Path(os.environ.get("QUANT_WORK_ROOT", r"E:\QuantData\repro_work\baseline00"))
)

# --- data facts (from manifest.json) ---
N_FEATURES = 323
FEATS = [f"feature_{i:03d}" for i in range(N_FEATURES)]
N_ASSETS = 15
N_TRAIN_PARTITIONS = 9  # 000..008
INDEX_COLS = ["row_id", "time_id", "asset_id"]

# --- base model configs (recovered from lgb_model.json) ---
# model_a: main long-window target model, minleaf 5000
MODEL_A = dict(
    num_leaves=63, learning_rate=0.03, min_data_in_leaf=5000,
    feature_fraction=0.85, bagging_fraction=0.85, lambda_l1=0.0, lambda_l2=1.0,
    n_estimators=3000, early_stopping=50, prediction_clip=3.0,
)
# model_b: recent-4-partition raw target model
MODEL_B = dict(
    num_leaves=63, learning_rate=0.03, min_data_in_leaf=2000,
    feature_fraction=0.85, bagging_fraction=0.85, lambda_l1=0.0, lambda_l2=1.0,
    n_estimators=3000, early_stopping=50, prediction_clip=3.0,
)
# model_c: target - weighted-time-mean residual model
MODEL_C = dict(
    num_leaves=63, learning_rate=0.03, min_data_in_leaf=2000,
    feature_fraction=0.85, bagging_fraction=0.85, lambda_l2=1.0,
    n_estimators=3000, early_stopping=50, prediction_clip=3.0,
    label_transform="target_minus_weighted_time_mean",
)

# --- staged post-processing pipeline params (fold7-learned, from main.py/wf_eval.py) ---
WEIGHT_A = 0.65
WEIGHT_B = 0.35
WEIGHT_C = 0.32
POST_BIAS = 0.005889961279703308
POST_SCALE = 1.0480819684000628
ASSET_SCALE_SHRINK = 0.55
ASSET_SCALE = [
    1.4116374757334096, 0.3341465123009946, 0.753727004704546, 0.3302530931871289,
    0.5279549320107922, 1.289829425727532, 0.8473264392809863, 1.0330450035467813,
    1.1519543491899, 0.6242309779680862, 0.6905705835838911, 0.8887866135632178,
    1.7548785863413472, 0.9059584745300867, 1.0818777240999817,
]
CS_DEV_ALPHA = 0.80
CS_MEAN_BETA = 1.45
OUTPUT_SCALE = 0.93
W_LEVEL = 0.15
SCALE_LEVEL = 0.640266

# --- L88 level/deviation calibration (for the L88 stack member) ---
L88_LEVEL = 0.88
L88_DEV = 1.30

# --- stack ---
RIDGE_ALPHA = 0.5
TARGET_STD = 0.046  # std046 amplitude calibration

# --- rolling_v1 member (from s14_build_cache.py) ---
ROLLING_TOP50 = [
    "feature_305", "feature_308", "feature_135", "feature_034", "feature_074", "feature_290",
    "feature_292", "feature_288", "feature_293", "feature_014", "feature_039", "feature_250",
    "feature_013", "feature_245", "feature_125", "feature_181", "feature_273", "feature_275",
    "feature_276", "feature_182", "feature_210", "feature_092", "feature_212", "feature_077",
    "feature_084", "feature_085", "feature_086", "feature_087", "feature_299", "feature_300",
    "feature_291", "feature_294", "feature_296", "feature_303", "feature_307", "feature_309",
    "feature_310", "feature_311", "feature_312", "feature_313", "feature_314", "feature_315",
    "feature_316", "feature_317", "feature_318", "feature_319", "feature_320", "feature_321",
    "feature_322", "feature_289",
]
ROLLING_LGB = dict(
    n_estimators=200, num_leaves=31, learning_rate=0.05,
    min_data_in_leaf=500, early_stopping=30,
)


def train_partition_path(p: int) -> Path:
    return TRAIN_DIR / f"train_partition_{p:03d}.parquet"


def ensure_data2_path(path: Path) -> Path:
    """Reject server outputs outside this user's project directory on data2."""
    resolved = Path(path).resolve()
    if QROOT is not None:
        data2_root = Path("/data2").resolve()
        try:
            resolved.relative_to(QROOT)
        except ValueError as exc:
            raise ValueError(f"server output must stay beneath QROOT={QROOT}: {resolved}") from exc
        if QROOT.exists() and QROOT.stat().st_dev != data2_root.stat().st_dev:
            raise ValueError(f"QROOT is not mounted on /data2: {QROOT}")
    return resolved


def weighted_zero_mean_r2(y_true, y_pred, weight) -> float:
    import numpy as np
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    denom = float(np.sum(weight * y_true * y_true))
    if denom <= 0:
        return 0.0
    num = float(np.sum(weight * (y_true - y_pred) ** 2))
    return 1.0 - num / denom
