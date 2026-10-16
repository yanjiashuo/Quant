"""cs_rank_delta: within-time rank of feature DELTAS (feature_t - feature_{t-1} per asset).
Genuinely different from cs_rank (level rank) -> a NEW transform/cluster, not multiplicity.
Mirrors the cs_rank trainer (recent4, within-time rank -> weighted-time-mean-demean target, LGB
n_est201 leaves63 lr0.03 minleaf2000) EXCEPT features = within-time rank of per-asset feature deltas.
Trains fold6/7/8 + production. Legal (per-asset 1-lag = past; within-time rank = current cross-section).
"""
import os,time,gc
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from common import TRAIN as DATA, TEST, cs_rank_fold, member_work
_LAG=int(os.environ.get("CSRD_LAG","1"))
if _LAG not in (1,5):
    raise ValueError("CSRD_LAG must be 1 or 5 for the frozen 15-member recipe")
_SUF="" if _LAG==1 else str(_LAG)
WORK=member_work(f"cs_rank_delta{_SUF}")
_MNAME=f"cs_rank_delta{_SUF}"
from repro.baseline00.training_io import weighted_time_mean_rows
from repro.config import FEATS
LGB=dict(n_estimators=201,num_leaves=63,learning_rate=0.03,min_data_in_leaf=2000,feature_fraction=0.85,bagging_fraction=0.85,bagging_freq=1,reg_alpha=0.0,reg_lambda=1.0,random_state=0,deterministic=True,force_col_wise=True)
def gb(tid):
    s=np.r_[0,np.flatnonzero(tid[1:]!=tid[:-1])+1]; e=np.r_[s[1:],len(tid)]; return s,e
def within_time_rank_delta(frame):
    """per-asset feature delta (t - t-LAG), then within-time rank mapped [-1,1]."""
    LAG=int(os.environ.get("CSRD_LAG","1"))
    f=frame.sort_values(["asset_id","time_id"],kind="stable").reset_index(drop=True)
    X=f[FEATS].to_numpy(np.float32)
    delta=np.zeros_like(X)
    # per-asset shift by LAG via pandas groupby
    sh=f.groupby("asset_id")[FEATS].shift(LAG)
    delta=X-sh.to_numpy(np.float32); delta[np.isnan(delta)]=0
    f=f[["time_id","asset_id"]].reset_index(drop=True).join(pd.DataFrame(delta,columns=FEATS))
    f=f.sort_values(["time_id","asset_id"],kind="stable").reset_index(drop=True)
    tid=f["time_id"].to_numpy(np.int64); order=np.argsort(tid,kind="stable"); inv=np.empty(len(tid),np.int64); inv[order]=np.arange(len(tid))
    Xd=f[FEATS].to_numpy(np.float32)[order]; ranked=np.zeros_like(Xd)
    s,e=gb(tid[order]); sizes=e-s
    for sz in np.unique(sizes):
        gs=s[sizes==sz]; off=np.arange(int(sz))
        for b in range(0,len(gs),4096):
            cur=gs[b:b+4096]; idx=cur[:,None]+off[None,:]; v=Xd[idx]
            fc=np.isfinite(v).sum(1,keepdims=True); r=rankdata(v,axis=1,method="average",nan_policy="omit")
            den=fc-1; m=np.divide(2*(r-1),den,out=np.zeros_like(r),where=den>0)-np.where(den>0,1,0); m[~np.isfinite(m)]=0; ranked[idx]=m
    return ranked[inv]
def build_cache(parts,cache_dir):
    cache_dir.mkdir(parents=True,exist_ok=True)
    rows=sum(pd.read_parquet(DATA/f"train_partition_{p:03d}.parquet",columns=["row_id"]).shape[0] for p in parts)
    x=np.lib.format.open_memmap(cache_dir/"features.npy",mode="w+",dtype=np.float32,shape=(rows,len(FEATS)))
    y=np.lib.format.open_memmap(cache_dir/"target.npy",mode="w+",dtype=np.float64,shape=(rows,))
    w=np.lib.format.open_memmap(cache_dir/"weight.npy",mode="w+",dtype=np.float64,shape=(rows,))
    cur=0
    for p in parts:
        fr=pd.read_parquet(DATA/f"train_partition_{p:03d}.parquet",columns=["row_id","time_id","asset_id","weight","target",*FEATS])
        wt=np.maximum(fr["weight"].to_numpy(np.float64),0); tg=fr["target"].to_numpy(np.float64)
        lvl=weighted_time_mean_rows(fr["time_id"].to_numpy(np.int64),tg,wt)
        Xd=within_time_rank_delta(fr)
        n=len(fr); x[cur:cur+n]=Xd; y[cur:cur+n]=tg-lvl; w[cur:cur+n]=wt; cur+=n
        del fr; gc.collect()
    x.flush();y.flush();w.flush(); return x,y,w
def fit(x,y,w): m=lgb.LGBMRegressor(**LGB,n_jobs=64,verbosity=-1); m.fit(x,y,sample_weight=w); return m
def recent4(stop): return list(range(max(0,stop-4),stop))
def main():
    t0=time.time()
    for f in (6,7,8):
        parts=recent4(f); cd=WORK/f"fold_{f}/cache"; x,y,w=build_cache(parts,cd); m=fit(x,y,w)
        vf=pd.read_parquet(DATA/f"train_partition_{f:03d}.parquet",columns=["row_id","time_id","asset_id",*FEATS])
        pred=m.predict(within_time_rank_delta(vf)).astype(np.float64)
        # align to parent local row_id (re-key via cs_rank fold archive)
        csr=np.load(cs_rank_fold(f))
        np.savez(WORK/f"fold_{f}.npz",row_id=csr["row_id"],source_row_id=csr["source_row_id"],time_id=csr["time_id"],asset_id=csr["asset_id"],**{_MNAME:pred})
        m.booster_.save_model(str(WORK/f"fold_{f}"/f"{_MNAME}.txt")); print(f"fold{f} done {len(pred)} rows"); del x,y,w,m,vf; gc.collect()
    # production
    parts=recent4(9); cd=WORK/"production/cache"; x,y,w=build_cache(parts,cd); m=fit(x,y,w); m.booster_.save_model(str(WORK/"production"/f"{_MNAME}.txt"))
    rps=[]
    for path in sorted(TEST.glob("test_partition_*.parquet")):
        vf=pd.read_parquet(path,columns=["row_id","time_id","asset_id",*FEATS]); rps.append(pd.DataFrame({"row_id":vf["row_id"].to_numpy(np.int64),_MNAME:m.predict(within_time_rank_delta(vf)).astype(np.float64)}))
    pd.concat(rps,ignore_index=True).to_parquet(WORK/"production"/f"public_{_MNAME}.parquet",index=False)
    print("production done",time.time()-t0)
if __name__=="__main__":main()
