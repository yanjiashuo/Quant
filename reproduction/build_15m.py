"""Build the BEST submission (15m, public score 0.0039380).
15m = 12 structural members + 3 MLP members, pure L1.0/D1.0 Ridge, std=0.046.

Members (15 total):
  BASE (12):
    1. factor     = (final + L88) / 2          [derived from teammate members]
    2. contrast    = final - L88               [derived from teammate members]
    3. level_pred                             [teammate member]
    4. rolling_v1                             [teammate member]
    5. cs_rank                                [teammate: within-time rank of 323 feats]
    6. model_c_long                           [teammate: full-history LGB, target_demean]
    7. csrd_resid  = cs_rank_delta - proj(cs_rank)   [residualized cross-section delta]
    8. csrd5_resid = cs_rank_delta5 - proj(cs_rank_delta)  [residualized, long5 window]
    9. csrv_resid  = cs_rank_vol - reg(cs_rank, delta, delta5)  [residualized volatility]
    10. asset_demean                          [LGB: 323 raw -> target - asset_hist_mean]
    11. double_demean                         [LGB: 323 raw -> target - asset_mean - time_mean]
    12. double_demean_assetaware               [LGB: 323 raw + asset_id -> double_demean target]
  MLP (3):
    13. mlp_ad                                 [PyTorch MLP on asset_demean target]
    14. mlp_member                            [PyTorch MLP on double_demean target]
    15. mlp_deep                              [PyTorch MLP(1024,512,256) on double_demean]

Ridge: alpha=0.01, positive=True, fit_intercept=False, per-column pop-std (ddof=0)
Calibration: pure L1.0/D1.0 (NO amplification, NO dev-shrink), std=0.046 mean-preserving

Usage:
  # Set paths via environment variables (or edit defaults below)
  export QUANT_DATA_ROOT=/path/to/public_release_20260630/data
  export TEAMMATE_WT=/path/to/writable/upstream-runtime
  export SOTA_WORK=/path/to/work/teammate_sota_integration
  python build_15m.py
Output:
  submission.csv (3,217,458 rows, std=0.046000, public score=0.0039380)

Prerequisites:
  - Upstream runtime (TEAMMATE_WT): a writable copy of bundled upstream/ plus generated artifacts/oof/ and artifacts/member_predictions/
  - Competition data (QUANT_DATA_ROOT): train/test parquet partitions
  - SOTA work dir (SOTA_WORK): contains all member fold_6/7/8.npz and production/*.parquet
  - Python packages: numpy, pandas, scikit-learn, lightgbm, torch (GPU)
"""
import sys,os,json,hashlib
from pathlib import Path
import numpy as np, pandas as pd
# === Paths: set via env vars or edit these defaults ===
SCRIPT_DIR=Path(__file__).parent.resolve()
WT=Path(os.environ.get("TEAMMATE_WT", SCRIPT_DIR/"../../worktrees/teammate_baseline05_0ad2c97")).resolve()
WORK=Path(os.environ.get("SOTA_WORK", SCRIPT_DIR/"..")).resolve()
DATA=Path(os.environ.get("QUANT_DATA_ROOT", SCRIPT_DIR/"../../public_release_20260630/data")).resolve()
OOF=WT/"artifacts/oof"; PUB=WT/"artifacts/member_predictions"; ALPHA=0.01; STD=0.046
EXPECTED_ROWS=3_217_458
EXPECTED_MD5="92ef61c93ac86f14f606f0ea0bc66dd0"
EXPECTED_SHA256="ae254141a97dafd20ba0b885c2351c5273e51128eede8c808804f996fd44df2f"
EXPECTED_UPSTREAM="0ad2c976d02ff3fc2d1965b5cb0a484800503470"
print(f"  TEAMMATE_WT = {WT}",flush=True)
print(f"  SOTA_WORK   = {WORK}",flush=True)
print(f"  DATA_ROOT   = {DATA}",flush=True)
sys.path.insert(0,str(WT))
os.environ.setdefault("QUANT_DATA_ROOT",str(DATA))
from repro.baseline02.artifacts import load_oof, OOFFold
from repro.baseline02.stack import augment_fold, evaluate_walk_forward
from repro.baseline02.make_submission import load_public_members, calibrate_mean_preserving_std
from repro.baseline04.assemble import adapt_model_c_long_fold
from repro.baseline00.metrics import weighted_zero_mean_r2
from sklearn.linear_model import Ridge
# === Member definitions ===
E3=["final","L88","model_c","level_pred","rolling_v1","cs_rank","cs_rank_long","model_c_long"]
BASE=["factor","contrast","level_pred","rolling_v1","cs_rank","model_c_long",
       "csrd_resid","csrd5_resid","csrv_resid","asset_demean","double_demean","double_demean_assetaware"]
MLP=[("mlp_ad","mlp_member"),("mlp_member","mlp_member"),("mlp_deep","mlp_member")]
MEMS=BASE+[m[0] for m in MLP]
def fit_ridge(fl):
    X=np.vstack([f.matrix for f in fl]);y=np.concatenate([f.target for f in fl]);w=np.concatenate([f.weight for f in fl])
    sc=X.std(0);sc[sc<1e-12]=1
    return Ridge(alpha=ALPHA,fit_intercept=False,positive=True).fit(X/sc,y,sample_weight=w).coef_/sc
def proj_coef(a,b,w):
    sw=w.sum();ma=(a*w).sum()/sw;mb=(b*w).sum()/sw
    return ((a-ma)*w*(b-mb)).sum()/sw/(((a-ma)*w*(a-ma)).sum()/sw+1e-12),ma,mb
def main():
    print(f"### Building 15m submission (target: 0.0039380) ###",flush=True)
    print(f"  Members ({len(MEMS)}): {MEMS}",flush=True)
    # Step 1: Build E3 folds (teammate's 8-member base)
    print("  [1/5] Building E3 folds...",flush=True)
    folds={};ad=WORK/"model_c_long_adapted"
    for f in (6,7,8):
        base=load_oof(OOF/f"fold{f}_five_member_oof.npz")
        six=augment_fold(base,WORK/f"cs_rank/fold_{f}.npz","cs_rank")
        adapt_model_c_long_fold(OOF/f"fold{f}_five_member_oof.npz",WORK/f"cs_rank/fold_{f}.npz",
                                WORK/f"model_c_long/fold_{f}.npz",ad/f"fold_{f}.npz")
        folds[f]=augment_fold(augment_fold(six,WORK/f"cs_rank_long/fold_{f}.npz","cs_rank_long"),
                              ad/f"fold_{f}.npz","model_c_long")
    e3=evaluate_walk_forward(folds,alpha=ALPHA)
    e3_7,e3_8=e3["fold7"]["raw_score"],e3["fold8"]["raw_score"]
    print(f"  E3: f7={e3_7:.8f} f8={e3_8:.8f}",flush=True)
    # Step 2: Compute projection coefficients (from fold6)
    print("  [2/5] Computing projections...",flush=True)
    cs6=folds[6].matrix[:,E3.index("cs_rank")];w6=folds[6].weight;sw=w6.sum()
    d1_6=np.load(WORK/"cs_rank_delta/fold_6.npz")["cs_rank_delta"]
    d5_6=np.load(WORK/"cs_rank_delta5/fold_6.npz")["cs_rank_delta5"]
    vol6=np.load(WORK/"cs_rank_vol/fold_6.npz")["cs_rank_vol"]
    al,ma,mb=proj_coef(cs6,d1_6,w6);beta=mb-al*ma
    ma1=(d1_6*w6).sum()/sw;ma5=(d5_6*w6).sum()/sw
    al5=((d1_6-ma1)*w6*(d5_6-ma5)).sum()/sw/(((d1_6-ma1)*w6*(d1_6-ma1)).sum()/sw+1e-12);beta5=ma5-al5*ma1
    X6r=np.column_stack([cs6,d1_6,d5_6,np.ones(len(cs6))])
    coef_vol=np.linalg.solve(X6r.T@(w6[:,None]*X6r),X6r.T@(w6*vol6))
    # Step 3: Build 15-member fold matrices
    print("  [3/5] Building fold matrices...",flush=True)
    A={}
    for f in (6,7,8):
        fd=folds[f];F=fd.matrix[:,0];L=fd.matrix[:,1];cs=fd.matrix[:,E3.index("cs_rank")]
        d1=np.load(WORK/f"cs_rank_delta/fold_{f}.npz")["cs_rank_delta"]
        d5=np.load(WORK/f"cs_rank_delta5/fold_{f}.npz")["cs_rank_delta5"]
        vol=np.load(WORK/f"cs_rank_vol/fold_{f}.npz")["cs_rank_vol"]
        Xf=np.column_stack([cs,d1,d5,np.ones(len(cs))]);rv=vol-Xf@coef_vol
        cols=[(F+L)/2,  # factor
              F-L,      # contrast
              fd.matrix[:,E3.index("level_pred")],
              fd.matrix[:,E3.index("rolling_v1")],
              cs,
              fd.matrix[:,E3.index("model_c_long")],
              d1-(al*cs+beta),       # csrd_resid
              d5-(al5*d1+beta5),     # csrd5_resid
              rv,                     # csrv_resid
              np.load(WORK/f"asset_demean/fold_{f}.npz")["asset_demean"],
              np.load(WORK/f"double_demean/fold_{f}.npz")["double_demean"],
              np.load(WORK/f"double_demean_assetaware/fold_{f}.npz")["double_demean_assetaware"]]
        for nm,fk in MLP:
            cols.append(np.load(WORK/f"{nm}/fold_{f}.npz")[fk])
        A[f]=OOFFold(np.column_stack(cols),fd.target,fd.weight,fd.row_id,fd.time_id,fd.asset_id,tuple(MEMS),fd.source)
    # Step 4: Fit Ridge and compute scores
    print("  [4/5] Fitting Ridge...",flush=True)
    c6=fit_ridge([A[6]])
    c67=fit_ridge([A[6],A[7]])
    c678=fit_ridge([A[6],A[7],A[8]])
    s7=weighted_zero_mean_r2(A[7].target,A[7].matrix@c6,A[7].weight)
    s8=weighted_zero_mean_r2(A[8].target,A[8].matrix@c67,A[8].weight)
    pub_est=0.0035934+(s8-e3_8)+0.000029
    print(f"  15m offline: f7={s7-e3_7:+.6f} f8={s8-e3_8:+.6f} (vs E3)",flush=True)
    print(f"  Model-predicted public ~ {pub_est:.6f} (actual: 0.0039380)",flush=True)
    print(f"  Coefficients (c67): "+" ".join(f"{n}={c:.4f}" for n,c in zip(MEMS,c67)),flush=True)
    # Step 5: Build submission
    print("  [5/5] Building submission...",flush=True)
    pub=load_public_members(PUB/"test_metadata.csv",{
        "final":PUB/"public_final_fallback.csv",
        "model_c":PUB/"public_model_c.csv",
        "level_pred":PUB/"public_level_pred.csv",
        "rolling_v1":PUB/"public_rolling_v1.csv"})
    F=pub["final"].to_numpy(np.float64);Lv=pub["L88"].to_numpy(np.float64)
    rid=pub["row_id"].to_numpy(np.int64)
    pub["factor"]=(F+Lv)/2;pub["contrast"]=F-Lv
    for nm,path in [("cs_rank",WORK/"cs_rank/production/public_cs_rank.parquet"),
                    ("model_c_long",WORK/"public_model_c_long.parquet")]:
        pub[nm]=pd.read_parquet(path).set_index("row_id").reindex(rid)[nm].to_numpy(np.float64)
    cs_pub=pub["cs_rank"]
    csd_pub=pd.read_parquet(WORK/"cs_rank_delta/production/public_cs_rank_delta.parquet").set_index("row_id").reindex(rid)["cs_rank_delta"].to_numpy(np.float64)
    pub["csrd_resid"]=csd_pub-(al*cs_pub+beta)
    csd5_pub=pd.read_parquet(WORK/"cs_rank_delta5/production/public_cs_rank_delta5.parquet").set_index("row_id").reindex(rid)["cs_rank_delta5"].to_numpy(np.float64)
    pub["csrd5_resid"]=csd5_pub-(al5*csd_pub+beta5)
    csv_pub=pd.read_parquet(WORK/"cs_rank_vol/production/public_cs_rank_vol.parquet").set_index("row_id").reindex(rid)["cs_rank_vol"].to_numpy(np.float64)
    Xpub=np.column_stack([cs_pub,csd_pub,csd5_pub,np.ones(len(cs_pub))]);pub["csrv_resid"]=csv_pub-Xpub@coef_vol
    pub["asset_demean"]=pd.read_parquet(WORK/"asset_demean/production/public_asset_demean.parquet").set_index("row_id").reindex(rid)["asset_demean"].to_numpy(np.float64)
    pub["double_demean"]=pd.read_parquet(WORK/"double_demean/production/public_double_demean.parquet").set_index("row_id").reindex(rid)["double_demean"].to_numpy(np.float64)
    pub["double_demean_assetaware"]=pd.read_parquet(WORK/"double_demean_assetaware/production/public_double_demean_assetaware.parquet").set_index("row_id").reindex(rid)["double_demean_assetaware"].to_numpy(np.float64)
    for nm,fk in MLP:
        pub[nm]=pd.read_parquet(WORK/f"{nm}/production/public_mlp_member.parquet").set_index("row_id").reindex(rid)[fk].to_numpy(np.float64)
    raw=pub[MEMS].to_numpy(np.float64)@c678
    cal=calibrate_mean_preserving_std(raw,STD)
    out=Path(__file__).parent
    sub=pd.DataFrame({"row_id":rid.astype(np.int64),"target":cal}).sort_values("row_id").reset_index(drop=True)
    if len(sub)!=EXPECTED_ROWS or sub["row_id"].duplicated().any():
        raise RuntimeError(f"invalid submission row contract: rows={len(sub)}")
    if not np.isfinite(sub["target"].to_numpy(np.float64)).all():
        raise RuntimeError("submission contains non-finite target values")
    submission_path=out/"submission.csv"
    sub.to_csv(submission_path,index=False)
    t=sub.target.to_numpy(np.float64)
    payload=submission_path.read_bytes()
    md5=hashlib.md5(payload).hexdigest()
    sha256=hashlib.sha256(payload).hexdigest()
    if md5!=EXPECTED_MD5 or sha256!=EXPECTED_SHA256:
        raise RuntimeError(
            "rebuild is not byte-identical: "
            f"md5={md5} sha256={sha256}; check pinned environment and inputs"
        )
    print(f"  Done! rows={len(sub)} std={t.std(ddof=0):.6f} md5={md5[:12]}",flush=True)
    print(f"  Saved to: {submission_path}",flush=True)
    json.dump(dict(members=MEMS,ridge_alpha=ALPHA,calibration="pure_L1.0_D1.0",std=STD,
                   fold7=float(s7),fold8=float(s8),d7=float(s7-e3_7),d8=float(s8-e3_8),
                   e3_fold7=float(e3_7),e3_fold8=float(e3_8),model_pred_public=float(pub_est),
                   actual_public=0.0039380,md5=md5,sha256=sha256,
                   upstream_commit=EXPECTED_UPSTREAM,
                   coefficients_c67={n:float(c) for n,c in zip(MEMS,c67)}),
              open(out/"manifest.json","w"),indent=2)
if __name__=="__main__":main()
