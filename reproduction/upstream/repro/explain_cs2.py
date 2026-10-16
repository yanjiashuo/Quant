"""Walk one real time_id through BOTH cross-sectional calibrations with actual numbers.

Prints: raw data shape -> per-asset pred at the CS2 input stage -> CS2 output ->
final -> L88 input/output. So the user can see exactly what 'level' and 'deviation'
mean on concrete data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

# --- paths ---
DEPLOY_LF = Path(r"E:\QuantData\repro_work\deploy_lf")
TEST_PARQ = Path(r"E:\QuantData\public_release_20260630\data\test\test_partition_000.parquet")

# params copied from main.py / config.py
W_A, W_B = 0.65, 0.35
W_C = 0.32
POST_BIAS = 0.005889961279703308
POST_SCALE = 1.0480819684000628
ASSET_SCALE_SHRINK = 0.55
ASSET_SCALE = np.array(
    [1.4116374757334096, 0.3341465123009946, 0.753727004704546, 0.3302530931871289,
     0.5279549320107922, 1.289829425727532, 0.8473264392809863, 1.0330450035467813,
     1.1519543491899, 0.6242309779680862, 0.6905705835838911, 0.8887866135632178,
     1.7548785867513472, 0.905958474530867, 1.0818777240999817],
    dtype=np.float64,
)
ASSET_MULT = 1.0 + ASSET_SCALE_SHRINK * (ASSET_SCALE - 1.0)
CS_DEV_ALPHA = 0.80
CS_MEAN_BETA = 1.45
OUTPUT_SCALE = 0.93
W_LEVEL = 0.15
SCALE_LEVEL = 0.640266
L88_LEVEL = 0.88
L88_DEV = 1.30


def load_booster(model_dir: Path):
    import json
    payload = json.loads((model_dir / "lgb_model.json").read_text(encoding="utf-8"))
    feat = list(payload["feature_columns"])
    scale = float(payload.get("prediction_scale", 1.0))
    clip = float(payload.get("prediction_clip", 3.0))
    booster = lgb.Booster(model_file=str(model_dir / payload["model_file"]))
    return feat, scale, clip, booster


def raw_booster(model_dir: Path):
    import json
    payload = json.loads((model_dir / "lgb_model.json").read_text(encoding="utf-8"))
    return list(payload["feature_columns"]), lgb.Booster(model_file=str(model_dir / payload["model_file"]))


def main():
    # pick ONE time_id from test
    print("=" * 72)
    print("STEP 0: what your raw test data looks like")
    print("=" * 72)
    df = pd.read_parquet(TEST_PARQ, columns=["row_id", "time_id", "asset_id"])
    tid = int(df["time_id"].iloc[0])
    print(f"test_partition_000: {len(df)} rows total, time_id range "
          f"[{df['time_id'].min()} .. {df['time_id'].max()}]")
    # how many assets per time_id (a few examples)
    counts = df.groupby("time_id").size()
    print(f"rows per time_id (first 5): {counts.head().tolist()}")
    print(f"asset_id per time_id (first time_id): "
          f"{sorted(df[df.time_id==tid]['asset_id'].tolist())}")
    print(f"\n=> We will follow ONE time_id = {tid} through the pipeline.")

    # load full slice for that time_id
    cols = ["row_id", "time_id", "asset_id"] + [f"feature_{i:03d}" for i in range(323)]
    sub = pd.read_parquet(TEST_PARQ, columns=cols)
    test = sub[sub.time_id == tid].sort_values("asset_id").reset_index(drop=True)
    print(f"\nOur slice: {len(test)} rows (one per asset), asset_ids="
          f"{test['asset_id'].tolist()}")
    print(f"feature_000 sample values: {test['feature_000'].round(4).tolist()}")

    # load models
    fa, sa, ca, ba = load_booster(DEPLOY_LF / "model_a_000008")
    fb, sb, cb, bb = load_booster(DEPLOY_LF / "model_b")
    fc, sc, cc, bc = load_booster(DEPLOY_LF / "model_c")

    def predict_scaled(feat, scale, clip, booster, t):
        x = t.loc[:, feat].to_numpy(dtype=np.float32, copy=True)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        p = booster.predict(x) * scale
        return np.clip(p, -clip, clip)

    pa = predict_scaled(fa, sa, ca, ba, test)
    pb = predict_scaled(fb, sb, cb, bb, test)
    pc = predict_scaled(fc, sc, cc, bc, test)

    assets = test["asset_id"].to_numpy(dtype=np.int16, copy=False)

    # ===== STEP 1: blend -> gcalib -> asset  (this is 'pred' AT the CS2 input) =====
    pred = W_A * pa + W_B * pb
    pred = (pred - POST_BIAS) * POST_SCALE
    pred = pred * ASSET_MULT[assets]

    print("\n" + "=" * 72)
    print("STEP 1: 'pred' AT THE MOMENT CS2 IS APPLIED  (= asset-stage output)")
    print("=" * 72)
    print("This is one number per asset (15 numbers) for time_id =", tid)
    for a, v in zip(assets, pred):
        print(f"  asset {a:2d}: pred = {v:+.5f}")
    mean_pred = float(np.mean(pred))
    print(f"\n  >>> level   (mean of the 15 preds)        = {mean_pred:+.5f}")
    print(f"  >>> deviation for asset0 (pred - mean)    = {pred[0]-mean_pred:+.5f}")

    # ===== STEP 2: CS2 =====
    print("\n" + "=" * 72)
    print("STEP 2: CS2 = 0.80*deviation + 1.45*level   (per time_id)")
    print("=" * 72)
    deviation = pred - mean_pred
    cs2 = CS_DEV_ALPHA * deviation + CS_MEAN_BETA * mean_pred
    print("Formula applied to EACH of the 15 assets:")
    print("  cs2[asset] = 0.80 * (pred - mean) + 1.45 * mean")
    print()
    print(f"  {'asset':>5} | {'pred':>10} | {'deviation':>10} | {'0.80*dev':>10} | {'1.45*level':>10} | {'CS2 out':>10}")
    print("  " + "-" * 72)
    for a, p, d in zip(assets, pred, deviation):
        print(f"  {a:5d} | {p:+10.5f} | {d:+10.5f} | "
              f"{0.80*d:+10.5f} | {1.45*mean_pred:+10.5f} | {0.80*d+1.45*mean_pred:+10.5f}")
    print(f"\n  note: 1.45*level is the SAME for all 15 assets ({1.45*mean_pred:+.5f});")
    print(f"        only the 0.80*deviation column differs per asset.")
    pred_after_cs2 = cs2  # this becomes 'pred' again in main.py

    # ===== STEP 3: build 'final' (td + level) =====
    print("\n" + "=" * 72)
    print("STEP 3: finish the pipeline -> 'final'  (td = cs2 + 0.32*model_c, then level)")
    print("=" * 72)
    td = pred_after_cs2 + W_C * pc
    # level model (per time_id, single value)
    fw, wboost = raw_booster(DEPLOY_LF / "weight_model")
    fl, lboost = raw_booster(DEPLOY_LF / "level_model")
    xw = test.loc[:, fw].to_numpy(dtype=np.float32, copy=True)
    xw = np.nan_to_num(xw, nan=0.0, posinf=0.0, neginf=0.0)
    pw = np.maximum(wboost.predict(xw), 0.0)
    psum = float(pw.sum())
    wfm = (pw[:, None] * xw).sum(axis=0) / psum
    level_val = float(lboost.predict(wfm.reshape(1, -1))[0])
    final = OUTPUT_SCALE * (td + W_LEVEL * SCALE_LEVEL * level_val)
    print(f"  level_model output (single number for this time_id) = {level_val:+.5f}")
    print(f"  td = cs2 + 0.32*model_c  ;  final = 0.93*(td + 0.15*0.640266*level)")
    print()
    print(f"  {'asset':>5} | {'final':>10}")
    print("  " + "-" * 22)
    for a, v in zip(assets, final):
        print(f"  {a:5d} | {v:+10.5f}")
    final_mean = float(np.mean(final))

    # ===== STEP 4: L88 (second calibration, on final) =====
    print("\n" + "=" * 72)
    print("STEP 4: L88 = 1.30*deviation + 0.88*level   (applied to 'final')")
    print("=" * 72)
    print("Same operation as CS2, but input is 'final' and coefficients swap.")
    dev_f = final - final_mean
    l88 = L88_DEV * dev_f + L88_LEVEL * final_mean
    print("Formula: l88[asset] = 1.30 * (final - mean) + 0.88 * mean")
    print()
    print(f"  {'asset':>5} | {'final':>10} | {'deviation':>10} | {'1.30*dev':>10} | {'0.88*level':>10} | {'L88 out':>10}")
    print("  " + "-" * 72)
    for a, f, d in zip(assets, final, dev_f):
        print(f"  {a:5d} | {f:+10.5f} | {d:+10.5f} | "
              f"{1.30*d:+10.5f} | {0.88*final_mean:+10.5f} | {1.30*d+0.88*final_mean:+10.5f}")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  CS2 input  'pred'   : 15 numbers, level={mean_pred:+.5f}  (spread {pred.max()-pred.min():.5f})")
    print(f"  CS2 output           : 15 numbers, level={np.mean(cs2):+.5f}  (spread {cs2.max()-cs2.min():.5f})  <- mean pushed up, spread shrunk")
    print(f"  L88 input  'final'   : 15 numbers, level={final_mean:+.5f}  (spread {final.max()-final.min():.5f})")
    print(f"  L88 output           : 15 numbers, level={np.mean(l88):+.5f}  (spread {l88.max()-l88.min():.5f})  <- mean pulled down, spread stretched")


if __name__ == "__main__":
    main()
