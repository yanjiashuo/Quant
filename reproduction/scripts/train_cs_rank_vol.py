"""cs_rank_vol: within-time rank of per-asset rolling-VOLATILITY (rolling std of features).
Genuinely different view from cs_rank (level) and cs_rank_delta (change) -> a volatility-regime view.
Mirrors cs_rank trainer (recent4, within-time rank -> weighted-time-mean-demean target, LGB).
"""
import time,gc
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from common import TRAIN as DATA, TEST, cs_rank_fold, member_work
from repro.config import FEATS
WORK=member_work("cs_rank_vol")
from repro.baseline00.training_io import weighted_time_mean_rows
LGB=dict(n_estimators=201,num_leaves=63,learning_rate=0.03,min_data_in_leaf=2000,feature_fraction=0.85,bagging_fraction=0.85,bagging_freq=1,reg_alpha=0.0,reg_lambda=1.0,random_state=0,deterministic=True,force_col_wise=True)
def gb(tid):
    s=np.r_[0,np.flatnonzero(tid[1:]!=tid[:-1])+1]; e=np.r_[s[1:],len(tid)]; return s,e
def within_time_rank_vol(frame,win=10):
    f=frame.sort_values(["asset_id","time_id"],kind="stable").reset_index(drop=True)
    X=f[FEATS].to_numpy(np.float32); aid=f["asset_id"].to_numpy()
    # per-asset rolling std (window win, min 1)
    df=f[["asset_id","time_id"]].copy()
    for j,fn in enumerate(FEATS):
        s=pd.Series(X[:,j]); df[f"_v{j}"]=s.groupby(aid).rolling(win,min_periods=1).std().reset_index(level=0,drop=True).to_numpy()
    vol=df[[f"_v{j}" for j in range(len(FEATS))]].to_numpy(np.float32); vol=np.where(np.isfinite(vol),vol,0)
    f2=pd.concat([f[["time_id","asset_id"]].reset_index(drop=True),pd.DataFrame(vol,columns=FEATS)],axis=1)
    f2=f2.sort_values(["time_id","asset_id"],kind="stable").reset_index(drop=True)
    tid=f2["time_id"].to_numpy(np.int64); order=np.argsort(tid,kind="stable"); inv=np.empty(len(tid),np.int64); inv[order]=np.arange(len(tid))
    Xd=f2[FEATS].to_numpy(np.float32)[order]; ranked=np.zeros_like(Xd)
    s,e=gb(tid[order]); sizes=e-s
    for sz in np.unique(sizes):
        gs=s[sizes==sz]; off=np.arange(int(sz))
        for b in range(0,len(gs),4096):
            cur=gs[b:b+4096]; idx=cur[:,None]+off[None,:]; v=Xd[idx]
            fc=np.isfinite(v).sum(1,keepdims=True); r=rankdata(v,axis=1,method="average",nan_policy="omit")
            den=fc-1; m=np.divide(2*(r-1),den,out=np.zeros_like(r),where=den>0)-np.where(den>0,1,0); m[~np.isfinite(m)]=0; ranked[idx]=m
    return ranked[inv]
def build_cache(parts,cd):
    cd.mkdir(parents=True,exist_ok=True)
    rows=sum(pd.read_parquet(DATA/f"train_partition_{p:03d}.parquet",columns=["row_id"]).shape[0] for p in parts)
    x=np.lib.format.open_memmap(cd/"features.npy",mode="w+",dtype=np.float32,shape=(rows,len(FEATS)))
    y=np.lib.format.open_memmap(cd/"target.npy",mode="w+",dtype=np.float64,shape=(rows,))
    w=np.lib.format.open_memmap(cd/"weight.npy",mode="w+",dtype=np.float64,shape=(rows,))
    cur=0
    for p in parts:
        fr=pd.read_parquet(DATA/f"train_partition_{p:03d}.parquet",columns=["row_id","time_id","asset_id","weight","target",*FEATS])
        wt=np.maximum(fr["weight"].to_numpy(np.float64),0); tg=fr["target"].to_numpy(np.float64)
        lvl=weighted_time_mean_rows(fr["time_id"].to_numpy(np.int64),tg,wt)
        Xd=within_time_rank_vol(fr); n=len(fr); x[cur:cur+n]=Xd; y[cur:cur+n]=tg-lvl; w[cur:cur+n]=wt; cur+=n; del fr; gc.collect()
    x.flush();y.flush();w.flush(); return x,y,w
def fit(x,y,w): m=lgb.LGBMRegressor(**LGB,n_jobs=64,verbosity=-1); m.fit(x,y,sample_weight=w); return m
def recent4(stop): return list(range(max(0,stop-4),stop))
def main():
    t0=time.time()
    for f in (6,7,8):
        parts=recent4(f); cd=WORK/f"fold_{f}/cache"; x,y,w=build_cache(parts,cd); m=fit(x,y,w)
        vf=pd.read_parquet(DATA/f"train_partition_{f:03d}.parquet",columns=["row_id","time_id","asset_id",*FEATS])
        pred=m.predict(within_time_rank_vol(vf)).astype(np.float64)
        csr=np.load(cs_rank_fold(f))
        np.savez(WORK/f"fold_{f}.npz",row_id=csr["row_id"],source_row_id=csr["source_row_id"],time_id=csr["time_id"],asset_id=csr["asset_id"],cs_rank_vol=pred)
        m.booster_.save_model(str(WORK/f"fold_{f}/cs_rank_vol.txt")); print(f"fold{f} done"); del x,y,w,m,vf; gc.collect()
    parts=recent4(9); cd=WORK/"production/cache"; x,y,w=build_cache(parts,cd); m=fit(x,y,w); m.booster_.save_model(str(WORK/"production/cs_rank_vol.txt"))
    rps=[]
    for path in sorted(TEST.glob("test_partition_*.parquet")):
        vf=pd.read_parquet(path,columns=["row_id","time_id","asset_id",*FEATS]); rps.append(pd.DataFrame({"row_id":vf["row_id"].to_numpy(np.int64),"cs_rank_vol":m.predict(within_time_rank_vol(vf)).astype(np.float64)}))
    pd.concat(rps,ignore_index=True).to_parquet(WORK/"production/public_cs_rank_vol.parquet",index=False)
    print("done",time.time()-t0)
if __name__=="__main__":main()
