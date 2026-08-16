#!/usr/bin/env python3
"""Build private serving bundles. Run only with /home/pa/work/venv/bin/python."""
from __future__ import annotations
import gc, json, random, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from v2.pipeline import parse_target, make_lead_target
from v2.full.full_utils import utc_to_pacific_naive, COST_GRID
from v2.run_poc import build_windows, TemporalCNN, predict_cnn, balanced_subset
from v2.shadow.shadow_common import FEATURES,SITES,SHADOW,WINDOW_HOURS,tree_view
SEED=20250917

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def sources(site):
    w=pd.read_csv(ROOT/'datasets'/'weather'/f'{site}_weather_appended.csv')
    w['time']=pd.to_datetime(w.time);w=w.sort_values('time').drop_duplicates('time',keep='last').reset_index(drop=True)
    t=w.time;w['hour_sin']=np.sin(2*np.pi*t.dt.hour/24);w['hour_cos']=np.cos(2*np.pi*t.dt.hour/24)
    w['doy_sin']=np.sin(2*np.pi*(t.dt.dayofyear-1)/365.25);w['doy_cos']=np.cos(2*np.pi*(t.dt.dayofyear-1)/365.25)
    v=pd.read_csv(ROOT/'datasets'/'visibility'/f'{site}_visibility.csv',usecols=['DATE','VIS'])
    v['time']=utc_to_pacific_naive(v.DATE).dt.floor('h')
    v['visibility_meters']=pd.to_numeric(v.VIS.astype(str).str.split(',').str[0],errors='coerce')
    v=parse_target(v,visibility_col='visibility_meters',target_col='fog')
    v=v.sort_values('DATE').drop_duplicates('time',keep='last').sort_values('time').reset_index(drop=True)
    good=set(w.time);now=v[v.time.isin(good)].copy().rename(columns={'time':'issue_time'});now['label_time']=now.issue_time;now['row_key']=now.issue_time.astype(str)
    lead=make_lead_target(v,time_col='time',target_col='fog',group_cols=(),output_target_col='fog',require_complete_hours=True)
    lead=lead[lead.issue_time.isin(good)].copy();lead['row_key']=lead.issue_time.astype(str)
    return w,now.reset_index(drop=True),lead.reset_index(drop=True)

def fit_transformer(X):
    n=X.shape[-1];im=SimpleImputer(strategy='median').fit(X.reshape(-1,n)); sc=StandardScaler().fit(im.transform(X.reshape(-1,n)))
    return im,sc

def transform(X,im,sc):
    n=X.shape[-1];return sc.transform(im.transform(X.reshape(-1,n))).reshape(X.shape).astype(np.float32)

def fit_xgb(X,y,seed):
    ratio=max(1.,float((y==0).sum()/max(1,(y==1).sum())))
    m=XGBClassifier(n_estimators=120,max_depth=4,learning_rate=.055,min_child_weight=3,subsample=.8,
      colsample_bytree=.8,reg_lambda=2.,reg_alpha=.2,scale_pos_weight=ratio,random_state=seed,n_jobs=8,
      tree_method='hist',eval_metric='aucpr')
    m.fit(tree_view(X),y,verbose=False);return m

def fit_cnn_fixed(X,y,config,seed,epochs=2):
    seed_all(seed);Xs,ys=balanced_subset(X,y,max_neg=30000,seed=seed)
    m=TemporalCNN(X.shape[-1],int(config['channels']),int(config['kernel']),float(config['dropout'])).to('cuda')
    pos=max(1,int((ys==1).sum()));neg=max(1,int((ys==0).sum()))
    lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos],device='cuda'))
    opt=torch.optim.AdamW(m.parameters(),lr=float(config['lr']),weight_decay=float(config['weight_decay']))
    ds=torch.utils.data.TensorDataset(torch.from_numpy(Xs),torch.from_numpy(ys.astype(np.float32)))
    dl=torch.utils.data.DataLoader(ds,batch_size=int(config['batch_size']),shuffle=True,generator=torch.Generator().manual_seed(seed))
    for _ in range(epochs):
        m.train()
        for xb,yb in dl:
            xb=xb.cuda();yb=yb.cuda();opt.zero_grad(set_to_none=True);lossfn(m(xb),yb).backward();opt.step()
    return m

def fit_cal(y,p):
    if len(np.unique(y))<2 or len(np.unique(p))<2:return None
    return IsotonicRegression(out_of_bounds='clip').fit(np.asarray(p,float),np.asarray(y,int))

def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required to build temporal CNN serving artifacts')
    serving=SHADOW/'serving';serving.mkdir(parents=True,exist_ok=True)
    retro=json.loads((ROOT/'v2'/'full'/'metrics_full.json').read_text())
    summary={"created_utc":pd.Timestamp.now(tz='UTC').isoformat(),"seed":SEED,"fit_contract":"auxiliary chronological 15% calibrator holdout, then base model refit on all labeled history with fixed settings","sites":{}}
    for si,site in enumerate(SITES):
      print('BUILD',site,flush=True);site_dir=serving/site;site_dir.mkdir(parents=True,exist_ok=True);w,now,lead=sources(site); summary['sites'][site]={}
      feature_spec={"feature_order":FEATURES,"window_hours":24,"oldest_to_newest":True,"window_end_inclusive":True,
        "issue_time_contract":{"nowcast":"trailing 24 hourly values ending at t predict fog at t","lead_time":"trailing 24 hourly values ending 18:00 America/Los_Angeles D predict any fog D+1 00:00-09:00"},
        "trained_engineered_feature_finding":"The approved v2/full checkpoint feature_names contain none of dewpoint_depression/cooling_rate_6h/cooling_rate_12h/previous_night_low. They are reconstructed as audit diagnostics but MUST NOT be inserted into these weights.",
        "diagnostics_not_model_inputs":["dewpoint_depression","cooling_rate_6h","cooling_rate_12h","previous_night_low"],
        "preprocessing":"per-feature median imputer then StandardScaler, fitted on all labeled history for final refit","feature_end_rule":"feature_end_time_utc <= issue_time_utc","timezone":"America/Los_Angeles issue clock; logs UTC"}
      (site_dir/'feature_spec.json').write_text(json.dumps(feature_spec,indent=2))
      for ti,(task,labels) in enumerate((('nowcast',now),('lead_time',lead))):
        td=site_dir/task;td.mkdir(parents=True,exist_ok=True); X,y,meta,cols=build_windows(w,labels,False,False)
        if cols!=FEATURES:raise AssertionError((cols,FEATURES))
        cut=int(len(y)*.85);cal_start=meta.issue_time.iloc[cut];base_mask=(meta.issue_time < cal_start-pd.Timedelta(hours=24)).to_numpy();cal_mask=(meta.issue_time>=cal_start).to_numpy()
        # Auxiliary chronological models generate genuinely held-out calibration scores.
        bim,bsc=fit_transformer(X[base_mask]);Xbase=transform(X[base_mask],bim,bsc);Xcal=transform(X[cal_mask],bim,bsc)
        auxx=fit_xgb(Xbase,y[base_mask],SEED+100*si+10*ti);px=auxx.predict_proba(tree_view(Xcal))[:,1]
        old=torch.load(ROOT/'v2'/'full'/'artifacts'/site/f'temporal_cnn_{task}.pt',map_location='cpu',weights_only=False);cfg=old['config']
        auxc=fit_cnn_fixed(Xbase,y[base_mask],cfg,SEED+100*si+10*ti+1);pc=predict_cnn(auxc,Xcal,'cuda')
        calx,calc=fit_cal(y[cal_mask],px),fit_cal(y[cal_mask],pc)
        del Xbase,Xcal,auxx,auxc,bim,bsc;torch.cuda.empty_cache();gc.collect()
        # Final base models and preprocessing are refit on every labeled window.
        im,sc=fit_transformer(X);Xa=transform(X,im,sc)
        finalx=fit_xgb(Xa,y,SEED+1000+100*si+10*ti);finalc=fit_cnn_fixed(Xa,y,cfg,SEED+1001+100*si+10*ti)
        finalx.save_model(td/'xgboost.json');joblib.dump({'imputer':im,'scaler':sc},td/'preprocessing.joblib');joblib.dump(calx,td/'calibrator_xgboost.joblib');joblib.dump(calc,td/'calibrator_temporal_cnn.joblib')
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in finalc.state_dict().items()},'config':cfg,'feature_names':FEATURES,'window_hours':24,'seed':SEED+1001+100*si+10*ti,'fit_rows':int(len(y))},td/'temporal_cnn.pt')
        thresholds={model:{r:float(retro['per_site'][site]['tasks'][task][model]['cost_operating_points'][str(r)]['mean_threshold']) for r in COST_GRID} for model in ('xgboost','temporal_cnn')}
        md={'site':site,'task':task,'fit_rows':int(len(y)),'fit_positives':int(y.sum()),'fit_start_local':str(meta.issue_time.min()),'fit_end_local':str(meta.issue_time.max()),'all_history_refit':True,'calibration_method':'isotonic on last chronological 15% predictions from auxiliary earlier-history model; final base refit on all history','calibration_rows':int(cal_mask.sum()),'calibration_positives':int(y[cal_mask].sum()),'cost_thresholds':thresholds,'feature_spec':'../feature_spec.json'}
        (td/'metadata.json').write_text(json.dumps(md,indent=2));summary['sites'][site][task]=md
        del X,Xa,y,meta,im,sc,finalx,finalc;torch.cuda.empty_cache();gc.collect()
    (serving/'build_manifest.json').write_text(json.dumps(summary,indent=2));print('DONE')
if __name__=='__main__':main()
