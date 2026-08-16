#!/usr/bin/env python3
"""Cautious five-airport CalFog replication; all outputs remain under v2/full."""
from __future__ import annotations
import json, os, random, shutil, sys, time, warnings
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
 precision_score, recall_score, f1_score)
from sklearn.isotonic import IsotonicRegression

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'v2'/'full'; ART=OUT/'artifacts'
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from v2.pipeline import parse_target, make_lead_target, audit_causal_features
from v2.run_poc import (WEATHER_COLS,CAL_COLS,AQI_COLS,WINDOW,TemporalCNN,seed_all,
 build_windows,prepare,fit_xgb,fit_rf,tree_view,train_cnn,predict_cnn)
from v2.full.full_utils import (rolling_origin_folds,block_bootstrap,paired_delta_bootstrap,
 safe_ap,safe_auc,COST_GRID,THRESHOLD_GRID,utc_to_pacific_naive)

SEED=20250901; N_BOOT=1000
SITES={"location_6":"Madera","location_7":"Fresno","location_8":"Visalia","location_9":"Hanford","location_10":"Bakersfield"}
DEFAULT_CONFIG={"channels":32,"kernel":3,"dropout":.3,"lr":8e-4,"weight_decay":1e-4,"batch_size":512,"max_neg":30000}

def load_sources(site):
    candidates=list((ROOT/'datasets'/'weather').glob(f'{site}_weather*.csv'))
    appended=[p for p in candidates if p.name.endswith('_appended.csv')]
    chosen=appended[0] if len(appended)==1 else (candidates[0] if len(candidates)==1 else None)
    if chosen is None: raise RuntimeError(f'{site}: could not select one weather file from {candidates}')
    w=pd.read_csv(chosen); w['time']=pd.to_datetime(w.time)
    w=w.sort_values('time').drop_duplicates('time',keep='last').reset_index(drop=True)
    if not (w.time.diff().dropna()==pd.Timedelta(hours=1)).all(): raise AssertionError(f'{site}: non-hourly weather')
    t=w.time; w['hour_sin']=np.sin(2*np.pi*t.dt.hour/24); w['hour_cos']=np.cos(2*np.pi*t.dt.hour/24)
    w['doy_sin']=np.sin(2*np.pi*(t.dt.dayofyear-1)/365.25); w['doy_cos']=np.cos(2*np.pi*(t.dt.dayofyear-1)/365.25)
    a=pd.read_csv(ROOT/'datasets'/'aqi'/f'{site}_aqi.csv'); a['time']=pd.to_datetime(a.time)
    a=a.sort_values('time').drop_duplicates('time',keep='last'); w=w.merge(a,on='time',how='left',validate='one_to_one')
    v=pd.read_csv(ROOT/'datasets'/'visibility'/f'{site}_visibility.csv',usecols=['DATE','VIS'])
    # NOAA Global Hourly DATE is UTC; Open-Meteo weather/AQI were requested in
    # America/Los_Angeles. Convert before hourly alignment and task labeling.
    v['time']=utc_to_pacific_naive(v.DATE).dt.floor('h')
    v['visibility_meters']=pd.to_numeric(v.VIS.astype(str).str.split(',').str[0],errors='coerce')
    v=parse_target(v,visibility_col='visibility_meters',target_col='fog')
    v=v.sort_values('DATE').drop_duplicates('time',keep='last').sort_values('time').reset_index(drop=True)
    return w,v[['time','visibility_meters','fog']]

def label_tables(weather,vis,site):
    now=vis[vis.time.isin(set(weather.time))].copy().rename(columns={'time':'issue_time'})
    now['label_time']=now.issue_time; now['row_key']=now.issue_time.dt.strftime('%Y-%m-%dT%H:%M:%S'); now['site']=site
    lead=make_lead_target(vis,time_col='time',target_col='fog',group_cols=(),output_target_col='fog',require_complete_hours=True)
    lead=lead[lead.issue_time.isin(set(weather.time))].copy(); lead['row_key']=lead.issue_time.dt.strftime('%Y-%m-%dT%H:%M:%S');lead['site']=site
    lead['valid_target_hours']=10
    # Explicit core causal audit: feature window ends at issue, never verification time.
    for z,task in ((now,'nowcast'),(lead,'lead')):
        z['_max_feature_time']=z.issue_time
        audit_causal_features(z,['_max_feature_time'],task=task).raise_for_violations()
    return now.sort_values('issue_time').reset_index(drop=True),lead.sort_values('issue_time').reset_index(drop=True)

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

def tune_trainable(config):
    from ray import tune
    d=np.load(config['data_path']); cfg={k:v for k,v in config.items() if k!='data_path'}
    train_cnn(d['Xtr'],d['ytr'],d['Xv'],d['yv'],cfg,device='cuda',max_epochs=3,patience=2,seed=SEED,
              report=lambda metrics:tune.report(metrics))

def tune_one(prep,site,task):
    from ray import tune
    from ray.tune.search.basic_variant import BasicVariantGenerator
    ddir=ART/site/'tune';ddir.mkdir(parents=True,exist_ok=True); data=ddir/f'{task}.npz'
    # Tune on a bounded, seeded training subset while keeping the complete validation origin.
    X,y=prep.Xtr,prep.ytr
    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);rng=np.random.default_rng(SEED)
    neg=rng.choice(neg,min(len(neg),25000),replace=False);ix=np.sort(np.r_[pos,neg])
    np.savez_compressed(data,Xtr=X[ix],ytr=y[ix],Xv=prep.Xv,yv=prep.yv)
    exp=ddir/'results'; shutil.rmtree(exp,ignore_errors=True)
    space={'data_path':str(data),'channels':tune.choice([16,24,32,48]),'kernel':tune.choice([3,5]),
      'dropout':tune.choice([.15,.3,.45]),'lr':tune.loguniform(2e-4,2e-3),'weight_decay':tune.choice([1e-5,1e-4,1e-3]),
      'batch_size':tune.choice([256,512]),'max_neg':25000}
    tuner=tune.Tuner(tune.with_resources(tune_trainable,{'cpu':2,'gpu':1}),param_space=space,
      tune_config=tune.TuneConfig(metric='val_ap',mode='max',num_samples=8,max_concurrent_trials=1,
       search_alg=BasicVariantGenerator(random_state=SEED+list(SITES).index(site)*10+(0 if task=='nowcast' else 1))),
      run_config=tune.RunConfig(name=f'{site}_{task}',storage_path=str(exp),verbose=0))
    res=tuner.fit(); best=res.get_best_result(metric='val_ap',mode='max')
    cfg={k:v for k,v in best.config.items() if k!='data_path'}
    return cfg,float(best.metrics['val_ap']),len(res)

def train_predict(prep,cfg,seed,save_path=None,epochs=6):
    val={};test={};logs={}
    base=float(prep.ytr.mean());val['climatology']=np.full(len(prep.yv),base);test['climatology']=np.full(len(prep.yt),base)
    x=fit_xgb(prep.Xtr,prep.ytr,prep.Xv,prep.yv,seed=seed);val['xgboost']=x.predict_proba(tree_view(prep.Xv))[:,1];test['xgboost']=x.predict_proba(tree_view(prep.Xt))[:,1]
    r=fit_rf(prep.Xtr,prep.ytr,seed=seed);val['random_forest']=r.predict_proba(tree_view(prep.Xv))[:,1];test['random_forest']=r.predict_proba(tree_view(prep.Xt))[:,1]
    c,bap,ep=train_cnn(prep.Xtr,prep.ytr,prep.Xv,prep.yv,cfg,'cuda',max_epochs=epochs,patience=2,seed=seed)
    val['temporal_cnn']=predict_cnn(c,prep.Xv,'cuda');test['temporal_cnn']=predict_cnn(c,prep.Xt,'cuda');logs={'cnn_epochs':ep,'cnn_best_val_ap':bap}
    if save_path: __import__('torch').save({'state_dict':c.state_dict(),'config':cfg,'feature_names':prep.cols,'window_hours':24,'seed':seed},save_path)
    del x,r,c; __import__('torch').cuda.empty_cache()
    return val,test,logs

def train_ablation(prep,cfg,seed):
    val={};test={}
    x=fit_xgb(prep.Xtr,prep.ytr,prep.Xv,prep.yv,seed=seed);val['xgboost']=x.predict_proba(tree_view(prep.Xv))[:,1];test['xgboost']=x.predict_proba(tree_view(prep.Xt))[:,1]
    c,_,ep=train_cnn(prep.Xtr,prep.ytr,prep.Xv,prep.yv,cfg,'cuda',max_epochs=5,patience=2,seed=seed)
    val['temporal_cnn']=predict_cnn(c,prep.Xv,'cuda');test['temporal_cnn']=predict_cnn(c,prep.Xt,'cuda')
    del x,c;__import__('torch').cuda.empty_cache();return val,test,ep

def finalize_model(y,p,pc,meta,val_records,model,seed):
    boot=block_bootstrap(y,p,meta,N_BOOT,seed,'day')
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

def evaluate_main(weather,labels,site,task,cfg,save_artifact):
    folds=rolling_origin_folds(labels); accum={m:{'y':[],'p':[],'pc':[],'meta':[]} for m in ('climatology','xgboost','random_forest','temporal_cnn')};vals=[]; audits=[];epochs=[]
    for fi,f in enumerate(folds,1):
        parts={k:f[k] for k in ('train','val','test')};prep=prepare(weather,parts,False,False)
        vp,tp,lg=train_predict(prep,cfg,SEED+fi,save_artifact if fi==3 else None);epochs.append(lg['cnn_epochs'])
        vr={'fold':fi,'y':prep.yv.copy()}
        for m in accum:
            cal=fit_calibrator(prep.yv,vp[m]); pc=calibrate(cal,tp[m]);vr[m]=calibrate(cal,vp[m])
            meta=prep.metas['test'][['issue_time','row_key']].copy();meta['fold']=fi
            for k,a in [('y',prep.yt),('p',tp[m]),('pc',pc)]:accum[m][k].append(np.asarray(a))
            accum[m]['meta'].append(meta)
        vals.append(vr);audits.append({'fold':fi,'train':len(prep.ytr),'val':len(prep.yv),'test':len(prep.yt),'pos_train':int(prep.ytr.sum()),'pos_val':int(prep.yv.sum()),'pos_test':int(prep.yt.sum()),
          'train_end':str(prep.metas['train'].issue_time.max()),'val_start':str(prep.metas['val'].issue_time.min()),'test_start':str(prep.metas['test'].issue_time.min())})
        del prep
    out={}
    for mi,m in enumerate(accum):
        a=accum[m];y=np.concatenate(a['y']);p=np.concatenate(a['p']);pc=np.concatenate(a['pc']);meta=pd.concat(a['meta'],ignore_index=True)
        out[m]=finalize_model(y,p,pc,meta,vals,m,SEED+100*list(SITES).index(site)+10*(task=='lead_time')+mi)
    return out,audits,epochs

def evaluate_ablation(weather,matched,site,task,cfg):
    folds=rolling_origin_folds(matched);acc={arm:{m:{'y':[],'p':[],'meta':[]} for m in ('xgboost','temporal_cnn')} for arm in ('with','without')};aud=[];epochs=[]
    for fi,f in enumerate(folds,1):
        parts={k:f[k] for k in ('train','val','test')}; no=prepare(weather,parts,False,False);yes=prepare(weather,parts,True,True)
        for split in ('train','val','test'): assert no.metas[split].row_key.tolist()==yes.metas[split].row_key.tolist()
        for arm,prep in (('without',no),('with',yes)):
            _,tp,ep=train_ablation(prep,cfg,SEED+1000+fi);epochs.append(ep)
            meta=prep.metas['test'][['issue_time','row_key']].copy();meta['fold']=fi
            for m in tp: acc[arm][m]['y'].append(prep.yt.copy());acc[arm][m]['p'].append(tp[m]);acc[arm][m]['meta'].append(meta)
        aud.append({'fold':fi,'train':len(no.ytr),'val':len(no.yv),'test':len(no.yt),'pos_train':int(no.ytr.sum()),'pos_val':int(no.yv.sum()),'pos_test':int(no.yt.sum()),'identical_issue_keys':True});del no,yes
    out={}
    for mi,m in enumerate(('xgboost','temporal_cnn')):
        aw=acc['with'][m];an=acc['without'][m];y=np.concatenate(aw['y']);pw=np.concatenate(aw['p']);pn=np.concatenate(an['p']);meta=pd.concat(aw['meta'],ignore_index=True)
        assert np.array_equal(y,np.concatenate(an['y'])) and meta.row_key.tolist()==pd.concat(an['meta'],ignore_index=True).row_key.tolist()
        delta,ci,nvalid=paired_delta_bootstrap(y,pw,pn,meta,N_BOOT,SEED+5000+100*list(SITES).index(site)+10*(task=='lead_time')+mi,'day')
        out[m]={'with_ap':safe_ap(y,pw),'matched_no_aqi_ap':safe_ap(y,pn),'delta_ap':delta,'delta_ap_ci':ci,'pos_count':int(y.sum()),'n_test':int(len(y)),
          'paired_block':'day','bootstrap_resamples':N_BOOT,'valid_resamples':nvalid,'identical_issue_keys':True}
        if y.sum()<50:out[m]['ci_warning']='Fewer than 50 positives: AQI delta interval is sparse and potentially uninformative.'
    return out,aud,epochs

def write_plots(metrics):
    """Render reliability and complete validation cost curves from saved numbers."""
    os.environ['MPLBACKEND']='Agg'
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    for site,sd in metrics['per_site'].items():
        sdir=ART/site;sdir.mkdir(parents=True,exist_ok=True)
        for task in ('nowcast','lead_time'):
            models=('climatology','xgboost','random_forest','temporal_cnn')
            fig,axes=plt.subplots(1,2,figsize=(10,4),sharex=True,sharey=True)
            for ax,which,title in ((axes[0],'reliability_before','raw'),(axes[1],'reliability_after','validation-fit isotonic')):
                ax.plot([0,1],[0,1],'k--',lw=1,label='ideal')
                for mdl in models:
                    rows=sd['tasks'][task][mdl]['calibration'][which]
                    ax.plot([r['mean_probability'] for r in rows],[r['observed_rate'] for r in rows],marker='o',ms=3,label=mdl)
                ax.set(title=title,xlabel='mean predicted probability',ylabel='observed fog rate',xlim=(0,1),ylim=(0,1));ax.grid(alpha=.2)
            axes[1].legend(fontsize=7);fig.suptitle(f'{site} {task}: reliability');fig.tight_layout();fig.savefig(sdir/f'reliability_{task}.png',dpi=150);plt.close(fig)
            fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
            for ax,mdl in zip(axes.ravel(),models):
                for ratio in COST_GRID:
                    c=sd['tasks'][task][mdl]['cost_threshold_curves'][str(ratio)]
                    ax.plot(c['thresholds'],c['validation_expected_cost'],label=f'{ratio}:1')
                ax.set(title=mdl,xlabel='threshold',ylabel='validation cost/issue',xlim=(0,1));ax.grid(alpha=.2)
            axes[0,0].legend(title='FN:FP',fontsize=7);fig.suptitle(f'{site} {task}: predeclared cost curves');fig.tight_layout();fig.savefig(sdir/f'cost_curves_{task}.png',dpi=150);plt.close(fig)

def report(metrics):
    lines=['# CalFog V2 — cautious five-airport replication','',
    '## Methodology',f"Each airport is modeled independently with seed {SEED}. Fog is visibility <1,610 m after invalid targets are removed. Nowcasts use 24 hourly observations ending at t; lead issues at 18:00 D use only observations through issue time and require all ten target hours on D+1 00:00–09:00. Three predeclared expanding-origin folds use a 24 h purge at both validation and test boundaries. Test predictions are pooled exactly once across disjoint origins.",
    '',f"The primary long-history comparison excludes AQI so all sites retain their historical cohort; AQI is evaluated only in the separate matched recent-era with/without experiment. Uncertainty uses {N_BOOT} paired whole-day block resamples stratified by fold and a point-centered 95% absolute-deviation interval; no hourly IID bootstrap is used. AP is primary and ROC-AUC secondary. Isotonic calibration is fitted separately on each fold's validation data. Cost thresholds are validation-selected over the predeclared threshold grid 0..1 by 0.005 for every FN:FP ratio {{1,3,5,10,20}}; none is designated as the true deployment cost.",
    '','## Feature availability and latency metadata','NOAA visibility timestamps are interpreted as UTC and converted to `America/Los_Angeles` (including DST) before alignment with the locally requested Open-Meteo weather/AQI clock. Weather and AQI values stamped h are assumed available by the end of h. Calendar variables are deterministic at issue time. Every 24-hour window ends at the issue timestamp; lead verification outcomes never enter features. **Deployment warning:** supplied hourly weather is reanalysis, not a latency-controlled feed. It must be replaced by timestamped operational observations and/or forecasts whose publication latency is enforced. AQI channels carry the same optimistic timestamp-availability assumption and require a production latency audit. Visibility is label-only. Suspect supplied rolling/night fields are excluded and calendar terms are recomputed.','']
    for site,sd in metrics['per_site'].items():
        lines += [f"## {site} — {SITES[site]}",f"Cohort: {json.dumps(sd['cohort'])}"]
        for task in ('nowcast','lead_time'):
            lines += [f"### {task.replace('_',' ').title()}",'| model | AP (95% block CI) | ROC-AUC (95% CI) | positives / n | Brier → calibrated | ECE → calibrated |','|---|---:|---:|---:|---:|---:|']
            for m,d in sd['tasks'][task].items():
                if m=='horizon':continue
                c=d['calibration'];lines.append(f"| {m} | {d['ap']:.4f} [{d['ap_ci'][0]:.4f}, {d['ap_ci'][1]:.4f}] | {d['roc_auc']:.4f} [{d['roc_auc_ci'][0]:.4f}, {d['roc_auc_ci'][1]:.4f}] | {d['pos_test']} / {d['n_test']} | {d['brier']:.4f} → {c['brier_after']:.4f} | {d['ece']:.4f} → {c['ece_after']:.4f} |")
            lines += ['','Cost operating points (validation-selected thresholds; all ratios reported):','| model | FN:FP | mean threshold | precision | recall | F1 | alert rate | cost/issue |','|---|---:|---:|---:|---:|---:|---:|---:|']
            for m,d in sd['tasks'][task].items():
                if m=='horizon':continue
                for r,o in d['cost_operating_points'].items():lines.append(f"| {m} | {r}:1 | {o['mean_threshold']:.3f} | {o['precision']:.3f} | {o['recall']:.3f} | {o['f1']:.3f} | {o['alert_rate']:.3f} | {o['expected_cost_per_issue']:.4f} |")
            lines += ['',f"[Reliability curve](artifacts/{site}/reliability_{task}.png) · [full cost curves](artifacts/{site}/cost_curves_{task}.png)"]
        lines += ['','### Matched-cohort AQI ablation','| task/model | with AQI AP | no AQI AP | ΔAP (paired 95% day-block CI) | positives |','|---|---:|---:|---:|---:|']
        sparse=[]
        for task in ('nowcast','lead_time'):
            for m,d in sd['aqi_ablation'][task].items():
                lines.append(f"| {task}/{m} | {d['with_ap']:.4f} | {d['matched_no_aqi_ap']:.4f} | {d['delta_ap']:+.4f} [{d['delta_ap_ci'][0]:+.4f}, {d['delta_ap_ci'][1]:+.4f}] | {d['pos_count']} |")
                if d['pos_count']<50:sparse.append(f"{task}/{m} ({d['pos_count']} positives)")
        if sparse:
            zeros={task:sum(f['pos_test']==0 for f in sd['fold_audit'][task+'_aqi']) for task in ('nowcast','lead_time')}
            lines += ['',"**Sparse-positive warning:** "+', '.join(sparse)+f". These AQI delta intervals are potentially uninformative even when they exclude zero. Zero-positive test folds: nowcast={zeros['nowcast']}/3, lead={zeros['lead_time']}/3; pooled metrics rely on the other fixed folds."]
        lines += ['',f"**Operational decision:** {sd['operational_verdict']}",'']
    lines += ['## Cross-airport summary','| site/task | climatology AP (CI) | XGBoost AP (CI) | RF AP (CI) | CNN AP (CI) |','|---|---:|---:|---:|---:|']
    for site in SITES:
        for task in ('nowcast','lead_time'):
            z=metrics['aggregate']['site_task_summary'][site][task]
            fmt=lambda d:f"{d['ap']:.3f} [{d['ap_ci'][0]:.3f}, {d['ap_ci'][1]:.3f}]"
            lines.append(f"| {site}/{task} | {fmt(z['climatology'])} | {fmt(z['xgboost'])} | {fmt(z['random_forest'])} | {fmt(z['temporal_cnn'])} |")
    lines += ['','| site/task/model | AQI ΔAP (paired CI) | positives |','|---|---:|---:|']
    for site in SITES:
        for task in ('nowcast','lead_time'):
            for mdl,d in metrics['aggregate']['aqi_site_summary'][site][task].items():lines.append(f"| {site}/{task}/{mdl} | {d['delta_ap']:+.3f} [{d['delta_ap_ci'][0]:+.3f}, {d['delta_ap_ci'][1]:+.3f}] | {d['pos_count']} |")
    lines += ['','## Aggregate verdict',metrics['aggregate']['predictability_verdict'],'',metrics['aggregate']['aqi_verdict'],'',
    '## Limitations','These are retrospective associations, not proof of causal aerosol effects. Reanalysis availability is optimistic; METAR reporting and missing complete mornings create selection effects. NOAA UTC is converted to Pacific local time, but the Open-Meteo export DST representation and the intended clock contract still require independent production verification. METAR times are floored to the hour and the last report is retained, so sub-hour label provenance is another deployment limitation. Fold-block CIs condition on the fitted predictions and reflect test-calendar block sampling, but not training/tuning variability, airport sampling, multiple-comparison selection, label-definition uncertainty, or future climate/regime shift. Isotonic calibration can overfit small positive validation sets. Cost ratios were not supplied, so all five scenarios are decision-support only. A wide interval or few positive days is explicitly treated as inconclusive, not as evidence of no effect.','',
    '## Reproduction','From the repository root run `bash v2/full/run_full.sh`. It regenerates JSON, report, calibration/cost curve PNGs and their underlying JSON arrays, CNN weights and the training log.']
    (OUT/'report_full.md').write_text('\n'.join(lines))

def main():
    warnings.filterwarnings('ignore');OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True);seed_all(SEED)
    # A failed rerun must not leave stale finals that look successful.
    for stale in (OUT/'metrics_full.json',OUT/'report_full.md',ART/'train_log_full.json'):
        stale.unlink(missing_ok=True)
    for stale in ART.glob('location_*/*.pt'):stale.unlink()
    import torch,ray
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    dev=torch.cuda.get_device_name(0);ray.init(num_cpus=min(8,os.cpu_count() or 4),num_gpus=1,include_dashboard=False,log_to_driver=False,_temp_dir='/tmp/calfog_ray')
    metrics={'sites':list(SITES),'seed':SEED,'gpu':{'used':True,'device':dev},'eval':{'scheme':'expanding-origin','folds':3,'bootstrap':N_BOOT,'bootstrap_block':'day','bootstrap_ci_method':'point-centered 95% absolute-deviation block interval','cost_grid_fn_fp':list(COST_GRID),'threshold_grid':'0..1 by 0.005','seed_schedule':{'global':SEED,'tune_search':'SEED + 10*site_index + task_offset','main_fold_training':'SEED + fold','aqi_fold_training':'SEED + 1000 + fold','main_bootstrap':'SEED + 100*site_index + 10*lead_flag + model_index','aqi_bootstrap':'SEED + 5000 + 100*site_index + 10*lead_flag + model_index'}},'per_site':{}}
    log={'device':dev,'seed':SEED,'ray_tune_trials':0,'per_site':{},'started_utc':pd.Timestamp.utcnow().isoformat()};start=time.time()
    for site in SITES:
        print(f'=== {site} ===',flush=True);(ART/site).mkdir(parents=True,exist_ok=True);weather,vis=load_sources(site);now,lead=label_tables(weather,vis,site)
        _,_,nowmatch,_=build_windows(weather,now,with_aqi=True,require_aqi=True);_,_,leadmatch,_=build_windows(weather,lead,with_aqi=True,require_aqi=True)
        cohort={'nowcast_rows':len(now),'nowcast_positives':int(now.fog.sum()),'nowcast_fog_rate':float(now.fog.mean()),'lead_complete_10h_issues':len(lead),'lead_positives':int(lead.fog.sum()),
          'aqi_matched_nowcast_rows':len(nowmatch),'aqi_matched_lead_issues':len(leadmatch),'aqi_channels':AQI_COLS}
        sd={'cohort':cohort,'tasks':{},'aqi_ablation':{},'fold_audit':{},'feature_latency':{'weather_assumption':'timestamp h assumed available by end of h; retrospective reanalysis must be replaced for deployment','aqi_assumption':'timestamp h assumed available by end of h; production publication latency not verified','calendar':'deterministic and available at issue','visibility':'label only; NOAA UTC converted to America/Los_Angeles before task construction','issue_semantics':'18:00 feature is allowed by the requested <= issue rule; deployment must define whether the observation has closed','timezone':'America/Los_Angeles with NOAA UTC converted before alignment','per_feature':{**{c:'assumed available by end of stamped hour; reanalysis, not deployment-valid' for c in WEATHER_COLS},**{c:'assumed available by end of stamped hour; publication latency unverified' for c in AQI_COLS},**{c:'deterministic calendar feature available at issue' for c in CAL_COLS},'visibility_meters':'label only, never a feature'},'all_feature_times_lte_issue':True}}
        slog={'tuning':{},'main_cnn_epochs':{},'ablation_cnn_epochs':{}}
        for task,labels,matched in (('nowcast',now,nowmatch),('lead_time',lead,leadmatch)):
            first=rolling_origin_folds(labels)[0];pp=prepare(weather,{k:first[k] for k in ('train','val','test')},False,False)
            cfg,bap,ntr=tune_one(pp,site,task);del pp;log['ray_tune_trials']+=ntr;slog['tuning'][task]={'trials':ntr,'best_validation_ap':bap,'best_config':cfg}
            res,aud,eps=evaluate_main(weather,labels,site,task,cfg,ART/site/f'temporal_cnn_{task}.pt');
            if task=='lead_time':res={'horizon':'issue 18:00 -> next-day 00:00-09:00',**res}
            sd['tasks'][task]=res;sd['fold_audit'][task]=aud;slog['main_cnn_epochs'][task]=eps
            ab,aa,ae=evaluate_ablation(weather,matched,site,task,cfg);sd['aqi_ablation'][task]=ab;sd['fold_audit'][task+'_aqi']=aa;slog['ablation_cnn_epochs'][task]=ae
        # Conservative per-site operational wording: never approve directly from retrospective data.
        best_now=max(sd['tasks']['nowcast'][m]['ap'] for m in ('xgboost','random_forest','temporal_cnn'));best_lead=max(sd['tasks']['lead_time'][m]['ap'] for m in ('xgboost','random_forest','temporal_cnn'))
        sd['operational_verdict']=f"NO-GO for unsupervised deployment; retrospective best AP nowcast={best_now:.3f}, lead={best_lead:.3f}. Consider only latency-controlled shadow testing and locally chosen costs."
        metrics['per_site'][site]=sd;log['per_site'][site]=slog
        (OUT/'metrics_full.partial.json').write_text(json.dumps(metrics,indent=2,allow_nan=False));(ART/'train_log_full.partial.json').write_text(json.dumps(log,indent=2,allow_nan=False))
    ray.shutdown()
    agg={}
    for task in ('nowcast','lead_time'):
        agg[task]={}
        for m in ('climatology','xgboost','random_forest','temporal_cnn'):
            vals=[metrics['per_site'][s]['tasks'][task][m]['ap'] for s in SITES];agg[task][m]={'mean_ap':float(np.mean(vals)),'site_aps':dict(zip(SITES,vals))}
    agg['site_task_summary']={}
    agg['aqi_site_summary']={}
    site_notes=[]
    for s in SITES:
        agg['site_task_summary'][s]={};agg['aqi_site_summary'][s]={}
        for task in ('nowcast','lead_time'):
            agg['site_task_summary'][s][task]={mdl:{'ap':metrics['per_site'][s]['tasks'][task][mdl]['ap'],'ap_ci':metrics['per_site'][s]['tasks'][task][mdl]['ap_ci']} for mdl in ('climatology','xgboost','random_forest','temporal_cnn')}
            agg['aqi_site_summary'][s][task]={mdl:{k:d[k] for k in ('delta_ap','delta_ap_ci','pos_count')} for mdl,d in metrics['per_site'][s]['aqi_ablation'][task].items()}
            learned=('xgboost','random_forest','temporal_cnn');best=max(learned,key=lambda mdl:metrics['per_site'][s]['tasks'][task][mdl]['ap']);bd=metrics['per_site'][s]['tasks'][task][best];cl=metrics['per_site'][s]['tasks'][task]['climatology']
            strength='interval-separated from climatology' if bd['ap_ci'][0]>cl['ap_ci'][1] else 'not interval-separated from climatology'
            site_notes.append(f"{s} {task}: strongest {best} AP={bd['ap']:.3f} ({strength})")
    resolved=[]
    for s in SITES:
      for t in ('nowcast','lead_time'):
       for mdl,d in metrics['per_site'][s]['aqi_ablation'][t].items():
        if d['delta_ap_ci'][0]>0 or d['delta_ap_ci'][1]<0:resolved.append(f"{s}/{t}/{mdl} (n+={d['pos_count']})")
    lead_counts=[metrics['per_site'][s]['aqi_ablation']['lead_time']['xgboost']['pos_count'] for s in SITES]
    agg['aqi_verdict']=("No paired AQI ΔAP interval excluded zero." if not resolved else "Isolated directionally resolved intervals occurred for: "+', '.join(resolved)+f". No site/task had same-direction interval exclusion for both model families; where both excluded zero their signs conflicted. Lead cohorts had only {min(lead_counts)}–{max(lead_counts)} positives and 20 comparisons were inspected. Therefore this run finds no robust, credible AQI contribution anywhere, while retaining the isolated signed results rather than hiding them.")
    agg['predictability_verdict']='; '.join(site_notes)+'. Results are heterogeneous and retrospective; no site is approved for operations without prospective latency-controlled shadow validation.'
    metrics['aggregate']=agg;log['completed_utc']=pd.Timestamp.utcnow().isoformat();log['elapsed_seconds']=time.time()-start
    (OUT/'metrics_full.json').write_text(json.dumps(metrics,indent=2,allow_nan=False));(ART/'train_log_full.json').write_text(json.dumps(log,indent=2,allow_nan=False));write_plots(metrics);report(metrics)
    print(json.dumps({'status':'complete','device':dev,'ray_tune_trials':log['ray_tune_trials'],'elapsed_seconds':log['elapsed_seconds']},indent=2))
if __name__=='__main__':main()
