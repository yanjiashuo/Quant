"""within-asset-demean member: LGB 323 raw -> (target - per-asset HISTORICAL weighted mean).
NEW signal source: within-asset-demean target has feature_095 corr -0.031 (>0.03 ceiling!),
50% orthogonal to within-time-demean (the existing cs_rank/model_c_long target).
Per-asset historical mean computed from TRAINING partitions only (past-only, legal).
Full-history window. Same LGB hyperparams as model_c_long (n_est=201, leaves=63, lr=0.03, minleaf=2000).
"""
import time,gc
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from common import TRAIN as DATA, TEST, cs_rank_fold, member_work
from repro.config import FEATS
WORK=member_work("asset_demean")
LGB=dict(n_estimators=201,num_leaves=63,learning_rate=0.03,min_data_in_leaf=2000,feature_fraction=0.85,bagging_fraction=0.85,bagging_freq=1,reg_alpha=0.0,reg_lambda=1.0,random_state=0,deterministic=True,force_col_wise=True)
def asset_means(parts):
    """per-asset weighted mean of target from given partitions."""
    am={}
    for p in parts:
        fr=pd.read_parquet(DATA/f"train_partition_{p:03d}.parquet",columns=["asset_id","weight","target"])
        for a in range(15):
            m=fr["asset_id"].to_numpy()==a
            if m.sum()>0:
                w=fr["weight"].to_numpy(np.float64);y=fr["target"].to_numpy(np.float64)
                am[a]=am.get(a,0.0)+ (w[m]*y[m]).sum()
                am[f"_w{a}"]=am.get(f"_w{a}",0.0)+w[m].sum()
    return {a: am.get(a,0)/max(am.get(f"_w{a}",1e-12),1e-12) for a in range(15)}
def build_cache(parts,cd,am):
    cd.mkdir(parents=True,exist_ok=True)
    rows=sum(pd.read_parquet(DATA/f"train_partition_{p:03d}.parquet",columns=["row_id"]).shape[0] for p in parts)
    x=np.lib.format.open_memmap(cd/"features.npy",mode="w+",dtype=np.float32,shape=(rows,len(FEATS)))
    y=np.lib.format.open_memmap(cd/"target.npy",mode="w+",dtype=np.float64,shape=(rows,))
    w=np.lib.format.open_memmap(cd/"weight.npy",mode="w+",dtype=np.float64,shape=(rows,))
    cur=0
    for p in parts:
        fr=pd.read_parquet(DATA/f"train_partition_{p:03d}.parquet",columns=["row_id","time_id","asset_id","weight","target",*FEATS])
        wt=np.maximum(fr["weight"].to_numpy(np.float64),0); tg=fr["target"].to_numpy(np.float64); aid=fr["asset_id"].to_numpy()
        tg_ad=tg-np.array([am.get(a,0.0) for a in aid])  # target - per-asset historical mean
        n=len(fr); x[cur:cur+n]=fr[FEATS].to_numpy(np.float32); y[cur:cur+n]=tg_ad; w[cur:cur+n]=wt; cur+=n; del fr; gc.collect()
    x.flush();y.flush();w.flush(); return x,y,w
def fit(x,y,w): m=lgb.LGBMRegressor(**LGB,n_jobs=64,verbosity=-1); m.fit(x,y,sample_weight=w); return m
def main():
    t0=time.time()
    for f in (6,7,8):
        parts=list(range(f)); am=asset_means(parts); cd=WORK/f"fold_{f}/cache"
        x,y,w=build_cache(parts,cd,am); m=fit(x,y,w)
        # validation: predict fold f, target = target - asset_mean(from training)
        vf=pd.read_parquet(DATA/f"train_partition_{f:03d}.parquet",columns=["row_id","time_id","asset_id",*FEATS])
        Xv=vf[FEATS].to_numpy(np.float32);Xv=np.where(np.isfinite(Xv),Xv,0)
        pred=m.predict(Xv).astype(np.float64)
        # align to parent (via cs_rank fold archive, re-key)
        csr=np.load(cs_rank_fold(f))
        np.savez(WORK/f"fold_{f}.npz",row_id=csr["row_id"],source_row_id=csr["source_row_id"],time_id=csr["time_id"],asset_id=csr["asset_id"],asset_demean=pred)
        m.booster_.save_model(str(WORK/f"fold_{f}/asset_demean.txt")); print(f"fold{f} done {len(pred)}"); del x,y,w,m; gc.collect()
    # production
    parts=list(range(9)); am=asset_means(parts); cd=WORK/"production/cache"
    x,y,w=build_cache(parts,cd,am); m=fit(x,y,w); m.booster_.save_model(str(WORK/"production/asset_demean.txt"))
    rps=[]
    for path in sorted(TEST.glob("test_partition_*.parquet")):
        vf=pd.read_parquet(path,columns=["row_id","time_id","asset_id",*FEATS])
        Xv=vf[FEATS].to_numpy(np.float32);Xv=np.where(np.isfinite(Xv),Xv,0)
        rps.append(pd.DataFrame({"row_id":vf["row_id"].to_numpy(np.int64),"asset_demean":m.predict(Xv).astype(np.float64)}))
    pd.concat(rps,ignore_index=True).to_parquet(WORK/"production/public_asset_demean.parquet",index=False)
    print("done",time.time()-t0)
if __name__=="__main__":main()
