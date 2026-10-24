#!/usr/bin/env python3
"""Build ordinal serving bundles with a chronological calibration holdout."""
from __future__ import annotations
import gc, json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from models._shared.data import load_sources
from models._shared.labels import label_tables
from models._shared.evaluation import COST_GRID, fit_calibrator, pmf_to_fog
from models._shared.visibility import CATEGORY_SM, CATEGORY_LABELS, FOG_MAX_CATEGORY
from models._shared.windows import build_windows
from models._shared.families import xgboost, temporal_cnn
from ops.shadow.shadow_common import FEATURES,SITES,SHADOW,tree_view
SEED=20250917


def sources(site):
    weather,vis=load_sources(site)
    now,lead=label_tables(weather,vis,site)
    return weather,now,lead


def fit_transformer(X):
    n=X.shape[-1];im=SimpleImputer(strategy='median').fit(X.reshape(-1,n));sc=StandardScaler().fit(im.transform(X.reshape(-1,n)))
    return im,sc


def transform(X,im,sc):
    n=X.shape[-1];return sc.transform(im.transform(X.reshape(-1,n))).reshape(X.shape).astype(np.float32)


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required to build temporal CNN serving artifacts')
    serving=SHADOW/'serving';serving.mkdir(parents=True,exist_ok=True)
    retro={model:json.loads((ROOT/'models'/model/f'{model}_metrics_v2.json').read_text()) for model in ('xgboost','temporal_cnn')}
    target={'target':'ordinal-18','category_sm':list(CATEGORY_SM),'category_labels':list(CATEGORY_LABELS)}
    summary={**target,'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),'seed':SEED,
             'fit_contract':'chronological 15% validation/calibration holdout with 24-hour purge; no all-history refit','sites':{}}
    for si,site in enumerate(SITES):
        print('BUILD',site,flush=True);site_dir=serving/site;site_dir.mkdir(parents=True,exist_ok=True)
        weather,now,lead=sources(site);summary['sites'][site]={}
        feature_spec={**target,'feature_order':FEATURES,'window_hours':24,'oldest_to_newest':True,'window_end_inclusive':True,
          'issue_time_contract':{'nowcast':'category at issue hour','lead_time':'minimum category over next fixed-PST 00:00-09:00; all ten hours required'},
          'diagnostics_not_model_inputs':['dewpoint_depression','cooling_rate_6h','cooling_rate_12h','previous_night_low'],
          'preprocessing':'per-feature median imputer then StandardScaler, fitted on chronological training history',
          'feature_end_rule':'feature_end_time_utc <= issue_time_utc','timezone':'fixed PST (Etc/GMT+8) issue clock; logs UTC'}
        (site_dir/'feature_spec.json').write_text(json.dumps(feature_spec,indent=2))
        for ti,(task,labels) in enumerate((('nowcast',now),('lead_time',lead))):
            td=site_dir/task;td.mkdir(parents=True,exist_ok=True)
            X,c,meta,cols=build_windows(weather,labels,False,False)
            if cols!=FEATURES:raise AssertionError((cols,FEATURES))
            cut=int(len(c)*.85);cal_start=meta.issue_time.iloc[cut]
            base=(meta.issue_time<cal_start-pd.Timedelta(hours=24)).to_numpy();cal=(meta.issue_time>=cal_start).to_numpy()
            im,sc=fit_transformer(X[base]);Xbase=transform(X[base],im,sc);Xcal=transform(X[cal],im,sc)
            seed=SEED+100*si+10*ti
            model=xgboost.fit_model(tree_view(Xbase),c[base],tree_view(Xcal),c[cal],seed=seed)
            cfg=retro['temporal_cnn']['per_site'][site]['cnn_cfg'][task]
            cnn,info=temporal_cnn.train_cnn(Xbase,c[base],Xcal,c[cal],cfg,device='cuda',seed=seed+1)
            fog=(c[cal]<=FOG_MAX_CATEGORY).astype(int)
            calx=fit_calibrator(fog,pmf_to_fog(model.predict_proba(tree_view(Xcal))))
            calc=fit_calibrator(fog,pmf_to_fog(temporal_cnn.predict_cnn(cnn,Xcal,'cuda')))
            joblib.dump(model,td/'xgboost.joblib');joblib.dump({'imputer':im,'scaler':sc},td/'preprocessing.joblib')
            joblib.dump(calx,td/'calibrator_xgboost.joblib');joblib.dump(calc,td/'calibrator_temporal_cnn.joblib')
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in cnn.state_dict().items()},'config':cfg,
                        'feature_names':FEATURES,'window_hours':24,'seed':seed+1,'fit_rows':int(base.sum()),**target},td/'temporal_cnn.pt')
            thresholds={m:{r:float(retro[m]['per_site'][site]['tasks'][task][m]['cost_operating_points'][str(r)]['mean_threshold']) for r in COST_GRID} for m in ('xgboost','temporal_cnn')}
            md={**target,**info,'site':site,'task':task,'fit_rows':int(base.sum()),'fit_positives':int((c[base]<=FOG_MAX_CATEGORY).sum()),
                'fit_start_local':str(meta.issue_time[base].min()),'fit_end_local':str(meta.issue_time[base].max()),'all_history_refit':False,
                'calibration_method':'isotonic on P(fog) derived from ordinal PMF in last chronological 15%; 24-hour purge',
                'calibration_rows':int(cal.sum()),'calibration_positives':int(fog.sum()),'cost_thresholds':thresholds,'feature_spec':'../feature_spec.json'}
            (td/'metadata.json').write_text(json.dumps(md,indent=2));summary['sites'][site][task]=md
            del X,Xbase,Xcal,c,meta,im,sc,model,cnn;torch.cuda.empty_cache();gc.collect()
    (serving/'build_manifest.json').write_text(json.dumps(summary,indent=2));print('DONE')
if __name__=='__main__':main()
