from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from .data import N_BOOT
from .visibility import CATEGORY_NOMINAL_M, FOG_MAX_CATEGORY, N_CATEGORIES


def pmf_to_fog(pmf) -> np.ndarray:
    return np.asarray(pmf)[:, :FOG_MAX_CATEGORY+1].sum(axis=1)


def _rps_rows(c, pmf):
    cumulative = np.asarray(pmf, dtype=float).cumsum(axis=1)[:, :-1]
    observed = np.asarray(c, dtype=int)[:, None] <= np.arange(N_CATEGORIES-1)
    return np.mean((cumulative-observed)**2, axis=1)


def rps(c, pmf) -> float:
    return float(np.mean(_rps_rows(c, pmf)))


def nll(c, pmf, eps=1e-9) -> float:
    c = np.asarray(c, dtype=int)
    return float(-np.log(np.clip(np.asarray(pmf)[np.arange(len(c)), c], eps, 1.)).mean())


def median_category(pmf) -> np.ndarray:
    return (np.asarray(pmf).cumsum(axis=1) >= .5).argmax(axis=1)


def ordinal_summary(c, pmf) -> dict:
    c = np.asarray(c, dtype=int)
    median = median_category(pmf)
    error = np.abs(c-median)
    # Both endpoints must be uncensored; category 0 has no nominal meters.
    known = (c > 0) & (c < N_CATEGORIES-1) & (median > 0) & (median < N_CATEGORIES-1)
    meters = np.asarray(CATEGORY_NOMINAL_M)
    count = int(known.sum())
    return {'rps': rps(c, pmf), 'nll': nll(c, pmf),
            'median_category_mae': float(error.mean()), 'within_one': float((error <= 1).mean()),
            'median_meters_mae_log10': float(np.abs(np.log10(meters[c[known]])-np.log10(meters[median[known]])).mean()) if count else None,
            'median_meters_n': count}


def _loss_bootstrap(loss, meta, n_boot, seed, block):
    codes, counts = _block_draw_counts(meta.reset_index(drop=True), n_boot, seed, block)
    totals = np.bincount(codes, weights=loss)
    sizes = np.bincount(codes)
    return counts @ totals / (counts @ sizes)


def ordinal_bootstrap(c, pmf, meta, n_boot=1000, seed=20250901, block='day'):
    result = ordinal_summary(c, pmf)
    draws = _loss_bootstrap(_rps_rows(c, pmf), meta, n_boot, seed, block)
    point = result['rps']
    radius = float(np.quantile(np.abs(draws-point), .95)) if len(draws) else 0.
    result.update(rps_ci=[max(0., point-radius), min(1., point+radius)],
                  valid_rps_resamples=len(draws), bootstrap_unit=block)
    return result


def paired_rps_delta_bootstrap(c, pmf_with, pmf_without, meta, n_boot=1000, seed=20250901, block='day'):
    loss = _rps_rows(c, pmf_with)-_rps_rows(c, pmf_without)
    point = float(loss.mean())
    draws = _loss_bootstrap(loss, meta, n_boot, seed, block)
    radius = float(np.quantile(np.abs(draws-point), .95)) if len(draws) else 0.
    return point, [max(-1., point-radius), min(1., point+radius)], len(draws)

COST_GRID = (1, 3, 5, 10, 20)
THRESHOLD_GRID = np.linspace(0.0, 1.0, 201)

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

def fit_calibrator(y,p):
    y=np.asarray(y); p=np.asarray(p,float)
    if len(np.unique(y))<2 or len(np.unique(p))<2: return None
    return IsotonicRegression(out_of_bounds='clip').fit(p,y)

def calibrate(cal,p):
    return np.clip(np.asarray(p,float) if cal is None else cal.predict(np.asarray(p,float)),0,1)

def ece(y,p,bins=10):
    y=np.asarray(y);p=np.asarray(p); edges=np.linspace(0,1,bins+1); total=0.
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        ix=(p>=lo)&((p<hi) if i<bins-1 else (p<=hi))
        if ix.any(): total+=ix.mean()*abs(float(y[ix].mean()-p[ix].mean()))
    return float(total)

def reliability(y,p,bins=10):
    y=np.asarray(y);p=np.asarray(p); rows=[]
    for i,(lo,hi) in enumerate(zip(np.linspace(0,1,bins+1)[:-1],np.linspace(0,1,bins+1)[1:])):
        ix=(p>=lo)&((p<hi) if i<bins-1 else (p<=hi))
        if ix.any(): rows.append({'lo':float(lo),'hi':float(hi),'n':int(ix.sum()),'mean_probability':float(p[ix].mean()),'observed_rate':float(y[ix].mean())})
    return rows

def cost_threshold(y,p,ratio):
    y=np.asarray(y,dtype=int);p=np.asarray(p,float); costs=[]
    for th in THRESHOLD_GRID:
        q=p>=th; costs.append((ratio*np.sum((~q)&(y==1))+np.sum(q&(y==0)))/max(1,len(y)))
    j=int(np.argmin(costs));return float(THRESHOLD_GRID[j]),np.asarray(costs,float)

def finalize_model(y,p,pc,meta,val_records,model,seed, *, n_boot: int = N_BOOT):
    boot=block_bootstrap(y,p,meta,n_boot,seed,'day')
    d=dict(boot);d.update({'brier':float(brier_score_loss(y,p)),'ece':ece(y,p),'n_test':int(len(y)),'pos_test':int(np.sum(y)),
      'calibration':{'method':'fold-local validation-fit isotonic','brier_before':float(brier_score_loss(y,p)),
       'brier_after':float(brier_score_loss(y,pc)),'ece_before':ece(y,p),'ece_after':ece(y,pc),
       'helps_brier':bool(brier_score_loss(y,pc)<brier_score_loss(y,p)),'reliability_before':reliability(y,p),'reliability_after':reliability(y,pc)}})
    ops={}; curves={}
    for ratio in COST_GRID:
        totals={'tp':0,'fp':0,'fn':0,'tn':0};ths=[]
        curve_sum=np.zeros(len(THRESHOLD_GRID)); nv=0
        for vr in val_records:
            th,curve=cost_threshold(vr['y'],vr[model],ratio);ths.append(th);curve_sum+=curve*len(vr['y']);nv+=len(vr['y'])
            ix=meta.fold.to_numpy()==vr['fold']; yy=y[ix];qq=pc[ix]>=th
            totals['tp']+=int(np.sum(qq&(yy==1)));totals['fp']+=int(np.sum(qq&(yy==0)));totals['fn']+=int(np.sum((~qq)&(yy==1)));totals['tn']+=int(np.sum((~qq)&(yy==0)))
        tp,fp,fn,tn=(totals[k] for k in ('tp','fp','fn','tn')); prec=tp/max(1,tp+fp);rec=tp/max(1,tp+fn);f1=2*prec*rec/max(1e-12,prec+rec)
        ops[str(ratio)]={'fn_fp':f'{ratio}:1','fold_thresholds':ths,'mean_threshold':float(np.mean(ths)),'precision':prec,'recall':rec,'f1':f1,
          'alert_rate':(tp+fp)/max(1,len(y)),'expected_cost_per_issue':(ratio*fn+fp)/max(1,len(y)),**totals}
        curves[str(ratio)]={'thresholds':THRESHOLD_GRID.tolist(),'validation_expected_cost':(curve_sum/max(1,nv)).tolist()}
    # Compatibility scalar is explicitly the first predeclared grid point, not a claimed preferred ratio.
    o=ops['1'];d.update({'cost_threshold':o['mean_threshold'],'precision':o['precision'],'recall':o['recall'],'f1':o['f1'],
      'alert_rate':o['alert_rate'],'cost_threshold_reference':'1:1 compatibility field; see full predeclared grid','cost_operating_points':ops,'cost_threshold_curves':curves})
    if d['pos_test']<20: d['ci_warning']='Few positives: interval is likely uninformative.'
    return d
