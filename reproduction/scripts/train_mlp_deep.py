"""Train the frozen mlp_deep member.

Architecture: 1024-512-256, BatchNorm, ReLU and dropout 0.3. The target is
double-demeaned and seed 0 is fixed. Outputs fold6/7/8 OOF plus production.
"""
import time,gc
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from common import TRAIN as DATA, TEST, cs_rank_fold, member_work
from repro.config import FEATS
from repro.baseline00.training_io import weighted_time_mean_rows
DEVICE=torch.device("cuda")
class MLP(nn.Module):
    def __init__(self,in_dim,n_hidden=(512,256,128),dropout=0.3):
        super().__init__(); layers=[]; prev=in_dim
        for h in n_hidden:
            layers+=[nn.Linear(prev,h),nn.BatchNorm1d(h),nn.ReLU(),nn.Dropout(dropout)]; prev=h
        layers+=[nn.Linear(prev,1)]; self.net=nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
    def forward(self,x): return self.net(x).squeeze(-1)
def asset_means(parts):
    am={}
    for p in parts:
        fr=pd.read_parquet(DATA/f"train_partition_{p:03d}.parquet",columns=["asset_id","weight","target"])
        for a in range(15):
            m=fr["asset_id"].to_numpy()==a
            if m.sum()>0:
                w=fr["weight"].to_numpy(np.float64);y=fr["target"].to_numpy(np.float64)
                am[a]=am.get(a,0.0)+(w[m]*y[m]).sum();am[f"_w{a}"]=am.get(f"_w{a}",0.0)+w[m].sum()
    return {a:am.get(a,0)/max(am.get(f"_w{a}",1e-12),1e-12) for a in range(15)}
def load_X(fr): X=fr[FEATS].to_numpy(np.float32);return np.nan_to_num(X,nan=0.0,posinf=0.0,neginf=0.0)
def build_target(fr,am,target_type):
    wt=np.maximum(fr["weight"].to_numpy(np.float64),0);tg=fr["target"].to_numpy(np.float64)
    aid=fr["asset_id"].to_numpy();tid=fr["time_id"].to_numpy(np.int64)
    if target_type=="double_demean":
        am_arr=np.array([am.get(a,0.0) for a in aid]);tm=weighted_time_mean_rows(tid,tg,wt);return tg-am_arr-tm,wt
    elif target_type=="time_demean":
        tm=weighted_time_mean_rows(tid,tg,wt);return tg-tm,wt
    elif target_type=="asset_demean":
        am_arr=np.array([am.get(a,0.0) for a in aid]);return tg-am_arr,wt
def train_mlp(Xt,yt,wt,Xv,yv,wv,arch=(512,256,128),epochs=50,bs=4096,lr=0.001,patience=10,seed=0):
    torch.manual_seed(seed);N=len(Xt);model=MLP(Xt.shape[1],arch).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=lr);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    Xt_c=torch.tensor(Xt,dtype=torch.float32);yt_c=torch.tensor(yt,dtype=torch.float32);wt_c=torch.tensor(wt,dtype=torch.float32)
    Xv_c=torch.tensor(Xv,dtype=torch.float32);yv_c=torch.tensor(yv,dtype=torch.float32);wv_c=torch.tensor(wv,dtype=torch.float32)
    best_loss=float('inf');best_state=None;no_improve=0
    for ep in range(epochs):
        model.train();perm=torch.randperm(N)
        for i in range(0,N,bs):
            idx=perm[i:i+bs];opt.zero_grad()
            xb=Xt_c[idx].to(DEVICE);yb=yt_c[idx].to(DEVICE);wb=wt_c[idx].to(DEVICE)
            pred=model(xb);loss=((wb*(pred-yb)**2)).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        sched.step();model.eval()
        with torch.no_grad():
            vp=torch.zeros(len(Xv_c))
            for i in range(0,len(Xv_c),bs): vp[i:i+bs]=model(Xv_c[i:i+bs].to(DEVICE)).cpu()
            vloss=((wv_c*(vp-yv_c)**2)).mean().item()
        if vloss<best_loss:best_loss=vloss;best_state={k:v.clone() for k,v in model.state_dict().items()};no_improve=0
        else:no_improve+=1
        if no_improve>=patience:break
    if best_state:model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred=torch.zeros(len(Xv_c))
        for i in range(0,len(Xv_c),bs): pred[i:i+bs]=model(Xv_c[i:i+bs].to(DEVICE)).cpu()
    del Xt_c,yt_c,wt_c,Xv_c,yv_c,wv_c;torch.cuda.empty_cache();gc.collect()
    return pred.numpy(),model
def predict_mlp(model,Xv,bs=4096):
    Xv_c=torch.tensor(Xv,dtype=torch.float32);model.eval()
    with torch.no_grad():
        pred=torch.zeros(len(Xv_c))
        for i in range(0,len(Xv_c),bs): pred[i:i+bs]=model(Xv_c[i:i+bs].to(DEVICE)).cpu()
    return pred.numpy()
def run_variant(name,target_type,arch=(512,256,128),seed=0):
    WORK=member_work(f"mlp_{name}")
    t0=time.time();print(f"\n### MLP variant: {name} (target={target_type}, arch={arch}, seed={seed}) ###")
    for f in (6,7,8):
        parts=list(range(f));am=asset_means(parts)
        Xt_parts=[];yt_parts=[];wt_parts=[]
        for p in parts:
            fr=pd.read_parquet(DATA/f"train_partition_{p:03d}.parquet",columns=["row_id","time_id","asset_id","weight","target",*FEATS])
            t,w=build_target(fr,am,target_type);Xt_parts.append(load_X(fr));yt_parts.append(t);wt_parts.append(w)
        Xt=np.vstack(Xt_parts);yt=np.concatenate(yt_parts);wt=np.concatenate(wt_parts)
        vf=pd.read_parquet(DATA/f"train_partition_{f:03d}.parquet",columns=["row_id","time_id","asset_id","weight","target",*FEATS])
        Xv=load_X(vf);yv,wv=build_target(vf,am,target_type)
        mu=Xt.mean(0);sd=Xt.std(0)+1e-8;Xt=(Xt-mu)/sd;Xv=(Xv-mu)/sd
        pred,model=train_mlp(Xt,yt,wt,Xv,yv,wv,arch=arch,seed=seed)
        csr=np.load(cs_rank_fold(f))
        (WORK/f"fold_{f}").mkdir(parents=True,exist_ok=True)
        np.savez(WORK/f"fold_{f}.npz",row_id=csr["row_id"],source_row_id=csr["source_row_id"],time_id=csr["time_id"],asset_id=csr["asset_id"],mlp_member=pred)
        print(f"  fold{f} done ({time.time()-t0:.0f}s)")
        del Xt,yt,wt,Xv,yv,wv;gc.collect();torch.cuda.empty_cache()
    parts=list(range(9));am=asset_means(parts)
    Xt_parts=[];yt_parts=[];wt_parts=[]
    for p in parts:
        fr=pd.read_parquet(DATA/f"train_partition_{p:03d}.parquet",columns=["row_id","time_id","asset_id","weight","target",*FEATS])
        t,w=build_target(fr,am,target_type);Xt_parts.append(load_X(fr));yt_parts.append(t);wt_parts.append(w)
    Xt=np.vstack(Xt_parts);yt=np.concatenate(yt_parts);wt=np.concatenate(wt_parts)
    mu=Xt.mean(0);sd=Xt.std(0)+1e-8;Xt=(Xt-mu)/sd
    _,model=train_mlp(Xt,yt,wt,Xt[:10],yt[:10],wt[:10],arch=arch,seed=seed)
    (WORK/"production").mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"mu":mu,"sd":sd},WORK/"production/mlp_prod.pt")
    rps=[]
    for path in sorted(TEST.glob("test_partition_*.parquet")):
        vf=pd.read_parquet(path,columns=["row_id","time_id","asset_id",*FEATS])
        Xv=load_X(vf);Xv=(Xv-mu)/sd;p=predict_mlp(model,Xv)
        rps.append(pd.DataFrame({"row_id":vf["row_id"].to_numpy(np.int64),"mlp_member":p}))
    pd.concat(rps,ignore_index=True).to_parquet(WORK/"production/public_mlp_member.parquet",index=False)
    print(f"  production done, total {time.time()-t0:.0f}s")
def main():
    run_variant("deep","double_demean",arch=(1024,512,256),seed=0)
    print("\nmlp_deep done")
if __name__=="__main__":main()
