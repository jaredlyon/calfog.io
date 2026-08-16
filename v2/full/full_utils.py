"""Evaluation utilities for the five-site CalFog replication.

These functions are deliberately model-independent so chronology, purge, matched
cohorts, block bootstrap, calibration and operating costs can be tested quickly.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from v2.pipeline import chronological_split, make_matched_arms

PURGE_HOURS = 24
FOLD_SPECS = ((0.45, 0.55, 0.70), (0.55, 0.70, 0.85), (0.70, 0.85, 1.00))
COST_GRID = (1, 3, 5, 10, 20)
THRESHOLD_GRID = np.linspace(0.0, 1.0, 201)

def utc_to_pacific_naive(values):
    """Convert NOAA UTC timestamps to naive Pacific local clock timestamps."""
    parsed=pd.to_datetime(values,utc=True,errors="raise")
    return parsed.tz_convert("America/Los_Angeles").tz_localize(None) if isinstance(parsed,pd.DatetimeIndex) else parsed.dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)

def rolling_origin_folds(frame, time_col="issue_time", purge="24h", specs=FOLD_SPECS):
    """Three expanding-origin train/validation/test folds with disjoint tests.

    Fractions are predeclared, chronological, and never shifted to fog events.
    ``chronological_split`` from the leakage-safe core applies both purges.
    """
    z=frame.copy(); z[time_col]=pd.to_datetime(z[time_col]); z=z.sort_values(time_col,kind="mergesort").reset_index(drop=True)
    if len(z)<30: raise ValueError("at least 30 rows are required")
    times=z[time_col].drop_duplicates().reset_index(drop=True)
    out=[]
    for fi,(vfrac,tfrac,stopfrac) in enumerate(specs):
        vi=min(len(times)-2, int(len(times)*vfrac)); ti=min(len(times)-1,int(len(times)*tfrac))
        split=chronological_split(z,time_col=time_col,validation_start=times.iloc[vi],test_start=times.iloc[ti],purge=purge)
        test=split.test
        if stopfrac < 1:
            stop=times.iloc[min(len(times)-1,int(len(times)*stopfrac))]
            test=test.loc[test[time_col] < stop].reset_index(drop=True)
        fold={"train":split.train,"val":split.validation,"test":test,"fold":fi+1,
              "validation_start":times.iloc[vi],"test_start":times.iloc[ti]}
        for a,b in ((fold["train"],fold["val"]),(fold["val"],fold["test"])):
            if len(a) and len(b):
                assert b[time_col].min()-a[time_col].max() >= pd.Timedelta(purge)
        out.append(fold)
    test_keys=[]
    for f in out: test_keys.extend(f["test"][time_col].tolist())
    if len(test_keys)!=len(set(test_keys)): raise AssertionError("rolling test folds overlap")
    return out

def matched_cohort(with_aqi, without_aqi, key_cols=("site","issue_time")):
    return make_matched_arms(with_aqi,without_aqi,key_cols=key_cols)

def safe_ap(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    return float(average_precision_score(y,p)) if len(y) and y.sum()>0 else 0.0

def safe_auc(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else 0.5

def _groups(meta, block="day"):
    t=pd.to_datetime(meta["issue_time"])
    period=t.dt.to_period("W" if block=="week" else "D").astype(str)
    fold=meta["fold"].astype(str) if "fold" in meta else pd.Series("1",index=meta.index)
    return (fold+"|"+period).to_numpy()

def _block_draw_counts(meta,n_boot,seed,block):
    """Return row->block codes and stratified multinomial bootstrap counts."""
    labels=_groups(meta,block); _,codes=np.unique(labels,return_inverse=True)
    folds=meta["fold"].to_numpy() if "fold" in meta else np.ones(len(meta),int)
    rng=np.random.default_rng(seed); counts=np.zeros((n_boot,int(codes.max())+1),dtype=np.uint16)
    for f in np.unique(folds):
        gids=np.unique(codes[folds==f]);
        draws=rng.multinomial(len(gids),np.full(len(gids),1/len(gids)),size=n_boot)
        counts[:,gids]=draws.astype(np.uint16)
    return codes,counts

def _weighted_boot_metrics(y,p,codes,counts,batch=12):
    """Exact weighted AP/AUC for each draw without repeatedly sorting rows."""
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    order=np.argsort(-p,kind="mergesort"); ps=p[order];ys=y[order];gs=np.asarray(codes)[order]
    starts=np.r_[0,np.flatnonzero(ps[1:]!=ps[:-1])+1]
    aps=[];aucs=[]
    for begin in range(0,len(counts),batch):
        w=counts[begin:begin+batch,gs].astype(np.float64)
        posg=np.add.reduceat(w*ys,starts,axis=1);negg=np.add.reduceat(w*(1-ys),starts,axis=1)
        cp=np.cumsum(posg,axis=1);cn=np.cumsum(negg,axis=1);tp=cp[:,-1];tn=cn[:,-1]
        validp=tp>0
        ap=np.full(len(w),np.nan);ap[validp]=np.sum((posg[validp]/tp[validp,None])*(cp[validp]/np.maximum(cp[validp]+cn[validp],1e-15)),axis=1)
        valida=(tp>0)&(tn>0);auc=np.full(len(w),np.nan)
        neg_before=cn-negg;neg_lower=tn[:,None]-neg_before-negg
        auc[valida]=np.sum(posg[valida]*(neg_lower[valida]+.5*negg[valida]),axis=1)/(tp[valida]*tn[valida])
        aps.extend(ap[np.isfinite(ap)].tolist());aucs.extend(auc[np.isfinite(auc)].tolist())
    return aps,aucs

def block_bootstrap(y,p,meta,n_boot=1000,seed=20250901,block="day"):
    """Stratified-by-fold whole-block bootstrap (never IID rows).

    Scores are sorted once, then each resample is evaluated as integer block
    weights. This is mathematically the same as physically repeating sampled
    blocks and keeps 1000-resample hourly analyses practical.
    """
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);meta=meta.reset_index(drop=True)
    codes,counts=_block_draw_counts(meta,n_boot,seed,block);aps,aucs=_weighted_boot_metrics(y,p,codes,counts)
    ap=safe_ap(y,p);auc=safe_auc(y,p)
    def ci(vals,point):
        if not vals:return [point,point]
        radius=float(np.quantile(np.abs(np.asarray(vals)-point),.95))
        return [float(max(0.,point-radius)),float(min(1.,point+radius))]
    return {"ap":ap,"ap_ci":ci(aps,ap),"roc_auc":auc,"roc_auc_ci":ci(aucs,auc),
      "valid_ap_resamples":len(aps),"valid_auc_resamples":len(aucs),"bootstrap_unit":block}

def paired_delta_bootstrap(y,p_with,p_without,meta,n_boot=1000,seed=20250901,block="day"):
    """Paired AP delta using exactly the same fold/block count matrix."""
    y=np.asarray(y,dtype=int);a=np.asarray(p_with,float);b=np.asarray(p_without,float);meta=meta.reset_index(drop=True)
    if not (len(y)==len(a)==len(b)==len(meta)):raise ValueError("paired arrays must align")
    codes,counts=_block_draw_counts(meta,n_boot,seed,block)
    aa,_=_weighted_boot_metrics(y,a,codes,counts);bb,_=_weighted_boot_metrics(y,b,codes,counts)
    # Both lists correspond to all draws whenever pooled y has positives. For
    # pathological zero-positive draws, recompute jointly in a deterministic
    # physical fallback so pairing can never be lost.
    if len(aa)==len(bb)==n_boot:vals=np.asarray(aa)-np.asarray(bb)
    else:
        vals=[]
        for row in counts:
            w=row[codes];ix=np.repeat(np.arange(len(y)),w)
            if y[ix].sum()>0:vals.append(safe_ap(y[ix],a[ix])-safe_ap(y[ix],b[ix]))
        vals=np.asarray(vals)
    point=safe_ap(y,a)-safe_ap(y,b)
    if len(vals):
        radius=float(np.quantile(np.abs(np.asarray(vals)-point),.95));ci=[float(max(-1.,point-radius)),float(min(1.,point+radius))]
    else:ci=[point,point]
    return float(point),ci,int(len(vals))
