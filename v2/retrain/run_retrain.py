#!/usr/bin/env python3
"""CalFog schema-compatible (not faithful) five-airport retrain.

All network inputs are public, all outputs stay below v2/retrain, and the
leakage-safe window/evaluation machinery is reused from v2.pipeline,
v2.run_poc and v2.full. This is intentionally not a deployment endorsement.
"""
from __future__ import annotations
import io, json, hashlib, os, pickle, random, shutil, sys, time, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import average_precision_score
from sklearn.isotonic import IsotonicRegression

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'v2'/'retrain'; DATA=OUT/'data'; ART=OUT/'artifacts'; SERV=OUT/'serving'
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import v2.run_poc as rp
import v2.full.run_full as ff
from v2.full.full_utils import rolling_origin_folds, COST_GRID
from v2.pipeline import audit_causal_features

SEED=20260228
N_BOOT=1000
SITES={
 'location_6':{'name':'Madera','icao':'KMAE','iem':'MAE','lat':36.989159,'lon':-120.111030},
 'location_7':{'name':'Fresno','icao':'KFAT','iem':'FAT','lat':36.777045,'lon':-119.716862},
 'location_8':{'name':'Visalia','icao':'KVIS','iem':'VIS','lat':36.322528,'lon':-119.394712},
 'location_9':{'name':'Hanford','icao':'KHJO','iem':'HJO','lat':36.313201,'lon':-119.626015},
 'location_10':{'name':'Bakersfield','icao':'KBFL','iem':'BFL','lat':35.328506,'lon':-118.997813},
}
CANDIDATES=['temperature_2m','relative_humidity_2m','dew_point_2m','precipitation',
 'surface_pressure','vapour_pressure_deficit','wind_speed_10m','wind_speed_100m',
 'wind_gusts_10m','weather_code','cloud_cover_low','soil_temperature_0cm',
 'soil_temperature_6cm','soil_temperature_0_to_7cm']
# Jointly usable set: non-null in live Forecast AND actually reproducible from ERA5 Archive.
# Forecast serves the two point-depth soil fields, but Archive returns them entirely NULL;
# they are therefore explicitly dropped rather than silently replaced by a depth layer.
APPROVED=['temperature_2m','relative_humidity_2m','dew_point_2m','precipitation',
 'surface_pressure','vapour_pressure_deficit','wind_speed_10m','wind_speed_100m',
 'wind_gusts_10m','weather_code','cloud_cover_low']
CALENDAR=['hour_sin','hour_cos','doy_sin','doy_cos']
FORECAST_URL='https://api.open-meteo.com/v1/forecast'
ARCHIVE_URL='https://archive-api.open-meteo.com/v1/archive'
HF_URL='https://historical-forecast-api.open-meteo.com/v1/forecast'
IEM_URL='https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py'

# Repoint reused full harness and its underlying window builder.
rp.WEATHER_COLS=list(APPROVED); ff.WEATHER_COLS=list(APPROVED)
ff.SEED=SEED; ff.N_BOOT=N_BOOT; ff.OUT=OUT; ff.ART=ART; ff.SITES={k:v['name'] for k,v in SITES.items()}

def seed_all():
    rp.seed_all(SEED)

def get(url,params,timeout=240,retries=4):
    err=None
    for i in range(retries):
        try:
            r=requests.get(url,params=params,timeout=timeout)
            if r.status_code==200:return r
            err=RuntimeError(f'HTTP {r.status_code}: {r.text[:300]}')
        except Exception as e: err=e
        time.sleep(2**i)
    raise err

def add_calendar(w):
    w=w.copy();w['time']=pd.to_datetime(w.time);w=w.sort_values('time').drop_duplicates('time',keep='last').reset_index(drop=True)
    if not (w.time.diff().dropna()==pd.Timedelta(hours=1)).all(): raise AssertionError('weather not contiguous hourly')
    t=w.time
    w['hour_sin']=np.sin(2*np.pi*t.dt.hour/24);w['hour_cos']=np.cos(2*np.pi*t.dt.hour/24)
    w['doy_sin']=np.sin(2*np.pi*(t.dt.dayofyear-1)/365.25);w['doy_cos']=np.cos(2*np.pi*(t.dt.dayofyear-1)/365.25)
    return w

def probe(round_name,manifest):
    evidence={}
    for site,sd in SITES.items():
        params={'latitude':sd['lat'],'longitude':sd['lon'],'hourly':','.join(CANDIDATES),
                'timezone':'America/Los_Angeles','past_days':7,'forecast_days':1}
        r=get(FORECAST_URL,params,120); d=r.json().get('hourly',{}); times=d.get('time',[])
        counts={v:{'rows':len(d.get(v,[])),'non_null':sum(x is not None for x in d.get(v,[])),
                   'non_null_fraction':sum(x is not None for x in d.get(v,[]))/max(1,len(d.get(v,[])))} for v in CANDIDATES}
        evidence[site]={'round':round_name,'rows':len(times),'time_start':times[0] if times else None,
                        'time_end':times[-1] if times else None,'variables':counts}
        manifest.append({'kind':'production_forecast_probe','round':round_name,'site':site,'url':FORECAST_URL,
          'params':params,'resolved_url':r.url,'http_status':r.status_code,'rows':len(times),
          'sha256':hashlib.sha256(r.content).hexdigest()})
    return evidence

def probe_archive_soil(manifest):
    evidence_path=OUT/'archive_soil_probe_observed.json'
    if evidence_path.exists():
        evidence=json.loads(evidence_path.read_text())
        manifest.append({'kind':'archive_replacement_soil_feasibility_observation','sites':list(SITES),
          'source_file':str(evidence_path.relative_to(OUT)),'http_status':None,
          'note':'Successful requests occurred in an earlier interrupted attempt; retained evidence has no raw-response hash.'})
        return evidence
    evidence={};vars_=['soil_temperature_0cm','soil_temperature_6cm']
    for site,sd in SITES.items():
        params={'latitude':sd['lat'],'longitude':sd['lon'],'start_date':'2025-01-01','end_date':'2025-01-07','hourly':','.join(vars_),'timezone':'America/Los_Angeles'}
        r=get(ARCHIVE_URL,params,120);h=r.json().get('hourly',{});rows=len(h.get('time',[]))
        counts={v:{'rows':len(h.get(v,[])),'non_null':sum(x is not None for x in h.get(v,[]))} for v in vars_};evidence[site]={'rows':rows,'variables':counts}
        manifest.append({'kind':'archive_replacement_soil_feasibility_probe','site':site,'url':ARCHIVE_URL,'params':params,'resolved_url':r.url,'http_status':r.status_code,'rows':rows,'sha256':hashlib.sha256(r.content).hexdigest()})
    evidence_path.write_text(json.dumps(evidence,indent=2));return evidence

def fetch_archive(site,sd,manifest):
    DATA.mkdir(parents=True,exist_ok=True);path=DATA/f'{site}_era5_archive.csv'
    params={'latitude':sd['lat'],'longitude':sd['lon'],'start_date':'1980-01-01','end_date':'2025-08-27','hourly':','.join(APPROVED),'timezone':'America/Los_Angeles'}
    if path.exists():
        w=pd.read_csv(path);side=OUT/'archive_refetch_manifest.json';ref=json.loads(side.read_text()).get(site) if side.exists() else None
        if ref:
            manifest.append({'kind':'era5_archive_training_refetch','site':site,'url':ARCHIVE_URL,'params':ref['params'],'resolved_url':ref['resolved_url'],'http_status':ref['status'],'rows':ref['rows'],'sha256':ref['sha256'],'sha256_scope':'raw HTTP response','saved_file':str(path.relative_to(OUT))})
        else:
            manifest.append({'kind':'era5_archive_training_cache_from_interrupted_run','site':site,'url':ARCHIVE_URL,'params':params,'http_status':None,'rows':len(w),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'sha256_scope':'saved normalized CSV','saved_file':str(path.relative_to(OUT))})
        return add_calendar(w)
    try:
        r=get(ARCHIVE_URL,params,600);h=r.json().get('hourly',{});w=pd.DataFrame({'time':h['time']})
        for v in APPROVED:
            if v not in h:raise AssertionError(f'{site} archive missing {v}')
            w[v]=h[v]
        status=r.status_code;resolved=r.url;sha=hashlib.sha256(r.content).hexdigest();scope='raw HTTP response'
    except RuntimeError as e:
        if '429' not in str(e):raise
        # Public hourly quota fallback: these repository weather files are earlier
        # Open-Meteo Archive pulls. Select exactly the approved columns, never rename.
        src=ROOT/'datasets'/'weather'/f'{site}_weather_appended.csv';w=pd.read_csv(src,usecols=['time']+APPROVED)
        w=w[pd.to_datetime(w.time)<=pd.Timestamp('2025-08-27 23:00:00')].copy()
        status=429;resolved=None;sha=hashlib.sha256(src.read_bytes()).hexdigest();scope='pre-existing Open-Meteo Archive source file'
        print(f'{site}: public Archive quota 429; using exact-column prior Archive pull {src}',flush=True)
    if w[APPROVED].isna().any().any():
        bad=w[APPROVED].isna().sum();raise AssertionError(f'{site} archive nulls: {bad[bad>0].to_dict()}')
    w.to_csv(path,index=False)
    manifest.append({'kind':'era5_archive_training','site':site,'url':ARCHIVE_URL,'params':params,'resolved_url':resolved,'http_status':status,'rows':len(w),'sha256':sha,'sha256_scope':scope,'saved_file':str(path.relative_to(OUT))})
    return add_calendar(w)

def load_visibility(site):
    v=pd.read_csv(ROOT/'datasets'/'visibility'/f'{site}_visibility.csv',usecols=['DATE','VIS'])
    v['time']=ff.utc_to_pacific_naive(v.DATE).dt.floor('h')
    v['visibility_meters']=pd.to_numeric(v.VIS.astype(str).str.split(',').str[0],errors='coerce')
    v=ff.parse_target(v,visibility_col='visibility_meters',target_col='fog')
    return v.sort_values('DATE').drop_duplicates('time',keep='last').sort_values('time').reset_index(drop=True)[['time','visibility_meters','fog']]

def raw_prepared(weather,parts):
    built={k:rp.build_windows(weather,v,False,False) for k,v in parts.items()}
    Xtr,ytr,mtr,cols=built['train'];Xv,yv,mv,_=built['val'];Xt,yt,mt,_=built['test']
    arr,im,sc=rp.fit_preprocess(Xtr,Xv,Xt);Xtr,Xv,Xt=arr
    return rp.Prepared(Xtr,ytr,Xv,yv,Xt,yt,{'train':mtr,'val':mv,'test':mt},cols),im,sc

def save_serving(site,task,weather,labels,cfg):
    """Refit a deterministic serving bundle on fold-3 training, calibrate on fold-3 validation."""
    f=rolling_origin_folds(labels)[-1]; prep,im,sc=raw_prepared(weather,{k:f[k] for k in ('train','val','test')})
    sdir=SERV/site;sdir.mkdir(parents=True,exist_ok=True)
    x=rp.fit_xgb(prep.Xtr,prep.ytr,prep.Xv,prep.yv,seed=SEED+303)
    r=rp.fit_rf(prep.Xtr,prep.ytr,seed=SEED+303)
    c,bap,epochs=rp.train_cnn(prep.Xtr,prep.ytr,prep.Xv,prep.yv,cfg,'cuda',max_epochs=6,patience=2,seed=SEED+303)
    raw={'climatology':np.full(len(prep.yv),float(prep.ytr.mean())),
         'xgboost':x.predict_proba(rp.tree_view(prep.Xv))[:,1],
         'random_forest':r.predict_proba(rp.tree_view(prep.Xv))[:,1],
         'temporal_cnn':rp.predict_cnn(c,prep.Xv,'cuda')}
    calibrators={m:ff.fit_calibrator(prep.yv,p) for m,p in raw.items()}
    joblib.dump({'imputer':im,'scaler':sc},sdir/f'{task}_preprocessor.joblib')
    joblib.dump(x,sdir/f'{task}_xgboost.joblib');joblib.dump(r,sdir/f'{task}_random_forest.joblib')
    joblib.dump(calibrators,sdir/f'{task}_calibrators.joblib')
    import torch
    torch.save({'state_dict':c.state_dict(),'config':cfg,'feature_names':prep.cols,'window_hours':24,'seed':SEED+303},sdir/f'{task}_temporal_cnn.pt')
    taskmeta={'task':task,'training_rows':int(len(prep.ytr)),'validation_rows':int(len(prep.yv)),
      'climatology':float(prep.ytr.mean()),'cnn_best_validation_ap':float(bap),'cnn_epochs':int(epochs),
      'calibration':'isotonic fit only on chronological fold-3 validation','models':['climatology','xgboost','random_forest','temporal_cnn']}
    (sdir/f'{task}_manifest.json').write_text(json.dumps(taskmeta,indent=2))
    bundle={'prep':{'imputer':im,'scaler':sc},'xgboost':x,'random_forest':r,'cnn':c,
            'calibrators':calibrators,'climatology':float(prep.ytr.mean()),'cols':prep.cols}
    return bundle,{'epochs':epochs,'best_validation_ap':bap,'training_rows':len(prep.ytr),'validation_rows':len(prep.yv)}

def load_serving_bundle(site,task):
    sdir=SERV/site;prep=joblib.load(sdir/f'{task}_preprocessor.joblib')
    x=joblib.load(sdir/f'{task}_xgboost.joblib');r=joblib.load(sdir/f'{task}_random_forest.joblib')
    calibrators=joblib.load(sdir/f'{task}_calibrators.joblib')
    import torch
    ck=torch.load(sdir/f'{task}_temporal_cnn.pt',map_location='cuda',weights_only=False);cfg=ck['config']
    c=rp.TemporalCNN(len(ck['feature_names']),cfg['channels'],cfg['kernel'],cfg['dropout']).cuda();c.load_state_dict(ck['state_dict']);c.eval()
    tm=json.loads((sdir/f'{task}_manifest.json').read_text())
    return {'prep':prep,'xgboost':x,'random_forest':r,'cnn':c,'calibrators':calibrators,
      'climatology':float(tm['climatology']),'cols':ck['feature_names']}

def write_site_spec(site):
    sdir=SERV/site;sdir.mkdir(parents=True,exist_ok=True)
    spec={'schema_version':'calfog-v2-schema-compatible','window_hours':24,
      'features':APPROVED,'weather_variables':APPROVED,'derived_calendar_features':CALENDAR,
      'model_input_order':APPROVED+CALENDAR,'production_endpoint':FORECAST_URL,
      'timezone':'America/Los_Angeles','feature_time_rule':'24 hourly rows ending at and including issue_time'}
    meta={'site':site,'airport':SITES[site]['icao'],'faithful':False,'train_source':'era5_archive',
      'schema_compatible':True,'production_ready':False,
      'causal_features':{'window_hours':24,'all_feature_times_lte_issue':True,'lead_label_after_issue':True},
      'warning':'Reanalysis-to-forecast distribution shift is unresolved; forward-feed fog-season evaluation is required.'}
    (sdir/'feature_spec.json').write_text(json.dumps(spec,indent=2));(sdir/'metadata.json').write_text(json.dumps(meta,indent=2))

def fetch_proxy_and_labels(site,sd,manifest):
    params={'latitude':sd['lat'],'longitude':sd['lon'],'start_date':'2025-11-29','end_date':'2026-02-28','hourly':','.join(APPROVED),'timezone':'America/Los_Angeles'}
    p=DATA/f'{site}_historical_forecast_proxy.csv'
    if p.exists():
        w=pd.read_csv(p);manifest.append({'kind':'historical_forecast_proxy_cache_from_interrupted_run','site':site,'url':HF_URL,'params':params,'http_status':None,'rows':len(w),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'sha256_scope':'saved normalized CSV','saved_file':str(p.relative_to(OUT))})
    else:
        r=get(HF_URL,params,300);h=r.json().get('hourly',{});w=pd.DataFrame({'time':h.get('time',[])})
        for v in APPROVED:w[v]=h[v] if v in h else np.nan
        w.to_csv(p,index=False);manifest.append({'kind':'historical_forecast_best_match_proxy','site':site,'url':HF_URL,'params':params,'resolved_url':r.url,'http_status':r.status_code,'rows':len(w),'sha256':hashlib.sha256(r.content).hexdigest(),'sha256_scope':'raw HTTP response','saved_file':str(p.relative_to(OUT))})
    missing=[v for v in APPROVED if v not in w or w[v].isna().any()]
    lp=DATA/f'{site}_fog_season_iem_metar.csv'
    ip={'station':sd['iem'],'data':'vsby','year1':2025,'month1':12,'day1':1,'year2':2026,'month2':3,'day2':1,'tz':'Etc/UTC','format':'comma','latlon':'no','elev':'no','missing':'empty','trace':'empty','direct':'no'}
    if lp.exists():
        labels=pd.read_csv(lp);labels['time']=pd.to_datetime(labels.time);manifest.append({'kind':'archival_metar_labels_cache','site':site,'url':IEM_URL,'params':ip,'http_status':None,'rows':len(labels),'sha256':hashlib.sha256(lp.read_bytes()).hexdigest(),'sha256_scope':'saved normalized CSV','saved_file':str(lp.relative_to(OUT))})
    else:
        ir=get(IEM_URL,ip,300);lines=[x for x in ir.text.splitlines() if not x.startswith('#')];iv=pd.read_csv(io.StringIO('\n'.join(lines)))
        iv['valid']=pd.to_datetime(iv.valid,utc=True).dt.tz_convert('America/Los_Angeles').dt.tz_localize(None);iv['time']=iv.valid.dt.floor('h');iv['vsby']=pd.to_numeric(iv.vsby,errors='coerce');iv=iv.dropna(subset=['vsby']).sort_values('valid').drop_duplicates('time',keep='last');iv['visibility_meters']=iv.vsby*1609.344
        iv=ff.parse_target(iv,visibility_col='visibility_meters',target_col='fog');labels=iv[['time','visibility_meters','fog']].sort_values('time').reset_index(drop=True);labels.to_csv(lp,index=False)
        manifest.append({'kind':'archival_metar_labels','site':site,'url':IEM_URL,'params':ip,'resolved_url':ir.url,'http_status':ir.status_code,'rows':len(labels),'sha256':hashlib.sha256(ir.content).hexdigest(),'sha256_scope':'raw HTTP response','saved_file':str(lp.relative_to(OUT))})
    return add_calendar(w),labels,sorted(set(missing))

def predict_proxy(weather,part,bundle):
    X,y,meta,cols=rp.build_windows(weather,part,False,False)
    im=bundle['prep']['imputer'];sc=bundle['prep']['scaler'];shape=X.shape
    X=sc.transform(im.transform(X.reshape(-1,shape[-1]))).reshape(shape).astype(np.float32)
    raw={'climatology':np.full(len(y),bundle['climatology']),
      'xgboost':bundle['xgboost'].predict_proba(rp.tree_view(X))[:,1],
      'random_forest':bundle['random_forest'].predict_proba(rp.tree_view(X))[:,1],
      'temporal_cnn':rp.predict_cnn(bundle['cnn'],X,'cuda')}
    return y,raw,meta

def proxy_crosscheck(site,weather,vis,bundles,archive_tasks,missing):
    now,lead=ff.label_tables(weather,vis,site)
    out={'label':'APPROXIMATE Historical-Forecast best-match proxy; non-vintage; not proof of faithfulness',
      'season':'2025-12-01 through 2026-02-28','vintage_preserved':False,'missing_features':missing,
      'status':'not_evaluated' if missing else 'evaluated','tasks':{}}
    if missing:return out
    for task,part in [('nowcast',now),('lead_time',lead)]:
        y,p,_=predict_proxy(weather,part,bundles[task]);models={}
        for m,pp in p.items():
            hap=float(average_precision_score(y,pp)) if len(y) and y.sum() else 0.0
            aap=float(archive_tasks[task][m]['ap'])
            models[m]={'hf_proxy_ap':hap,'archive_oof_ap':aap,'delta_ap_proxy_minus_archive':hap-aap,
                       'n_proxy':int(len(y)),'pos_proxy':int(y.sum())}
        out['tasks'][task]=models
    return out

def report(metrics,contract,old):
    L=['# CalFog V2 — schema-compatible archive retrain','',
    '> **STATUS: SCHEMA-COMPATIBLE ONLY; NOT FAITHFUL; NOT PRODUCTION-READY.** The models were trained on ERA5 Archive reanalysis while the operational endpoint supplies forecast/analysis values. Matching names and non-null availability removes the null-channel blocker but does not resolve distribution shift. A forward-feed evaluation from the separately deployed shadow logger over a fog season is required before any production claim.','',
    '## Methodology',f'All five weather-only airport models use the existing `v2/pipeline.py` and `v2/full` harness with seed {SEED}. We fetched the approved variables from the Open-Meteo ERA5 Archive from 1980-01-01 through 2025-08-27, rebuilt 24-hour causal windows, and evaluated nowcast and 18:00-to-next-morning lead tasks. Three expanding-origin folds have a 24-hour purge. Predictions are pooled out-of-fold. AP is primary and ROC-AUC secondary. Uncertainty is {N_BOOT} seeded whole-day block bootstrap resamples, never IID hourly sampling. Calibration is fold-local validation-fit isotonic regression. The complete validation-selected cost grid is FN:FP {1,3,5,10,20}:1. CNN tuning uses 8 Ray Tune trials per site/task on the RTX GPU.','',
    '## Production availability contract','The live Forecast API was queried for seven recent past days plus one forecast day at every site, then queried again after artifact construction. Approval requires every returned value to be non-null at all five airports in both rounds. Calendar sine/cosine terms are deterministic derived inputs, not remote API fields.','',
    '**Approved, exact API order:** `'+ '`, `'.join(contract['approved'])+'`','',
    '**Dropped:** '+', '.join(f"`{x['name']}` ({x['reason']})" for x in contract['dropped']),'',
    'Temperature, RH, dew point, precipitation, surface pressure, VPD, 10 m/100 m wind, gusts, weather code, and low cloud passed the all-site non-null and Archive-reproducibility rule. Forecast served the 0 cm and 6 cm soil fields, but ERA5 Archive returned them entirely NULL even in a dedicated feasibility probe. Because `train_source=era5_archive` is mandatory, both replacement soil fields were explicitly dropped rather than silently synthesized from a different depth layer. This is a narrower no-soil schema and an unavoidable deviation from the requested soil replacement; it preserves source honesty.','',
    '## Archive evaluation and skill cost versus v2/full','The table reports AP with paired-day-block confidence intervals from this retrain and the corresponding old-feature-set AP. Deltas are descriptive across two completed runs; they are not a paired refit uncertainty interval.','',
    '| site/task/model | retrain AP [95% CI] | v2/full AP | Δ AP |','|---|---:|---:|---:|']
    for s in SITES:
      for t in ('nowcast','lead_time'):
       for m in ('climatology','xgboost','random_forest','temporal_cnn'):
        d=metrics['per_site'][s]['tasks'][t][m]; od=old['per_site'][s]['tasks'][t][m]
        L.append(f"| {s}/{t}/{m} | {d['ap']:.4f} [{d['ap_ci'][0]:.4f}, {d['ap_ci'][1]:.4f}] | {od['ap']:.4f} | {d['ap']-od['ap']:+.4f} |")
    L += ['','## Approximate distribution-shift cross-check',
    '**This is an approximate, non-vintage estimate—not evidence of faithfulness.** Open-Meteo Historical Forecast supplies a best-match historical forecast/analysis series, not the exact vintage that would have been available at each issue time. Each 24-hour feature window ends at issue time, but model-run vintages are not preserved. Archival METAR labels are fetched from the Iowa Environmental Mesonet. No missing proxy variable is silently substituted; a missing channel makes that site cross-check not evaluated.','',
    '| site/task/model | archive OOF AP | HF-proxy AP | proxy − archive | proxy positives/n |','|---|---:|---:|---:|---:|']
    for s,z in metrics['shift_crosscheck']['per_site'].items():
      for t,models in z.get('tasks',{}).items():
       for m,d in models.items():L.append(f"| {s}/{t}/{m} | {d['archive_oof_ap']:.4f} | {d['hf_proxy_ap']:.4f} | {d['delta_ap_proxy_minus_archive']:+.4f} | {d['pos_proxy']}/{d['n_proxy']} |")
    L += ['','The proxy-minus-archive delta mixes distribution shift, a different calendar/season, label availability, and sampling uncertainty. It is only a present-day warning signal. It cannot prove equivalence or quantify the vintage forecast penalty cleanly.','',
    '## Serving artifacts and limitations','Each `serving/location_*` directory contains serialized XGBoost, RandomForest, temporal-CNN, preprocessing, and isotonic-calibrator artifacts for both tasks, plus a strictly ordered feature spec and metadata. Every metadata file sets `faithful=false`, `train_source="era5_archive"`, and `production_ready=false`. The re-probe proves only that required schema fields are currently populated.','',
    'Other limitations include METAR time flooring, airport/grid mismatch, rare positive events, fitted-prediction bootstrap intervals that omit training variability, possible calibration instability, and changing upstream forecast models. Thresholds under all cost ratios are decision-support scenarios, not chosen operational utilities.','',
    '## Required next step','Continue the separately deployed forward-feed shadow logger through a representative fog season, preserve issue-time vintages and publication latency, join prospective METAR outcomes, and compare drift, AP, calibration, and operating costs. Only that forward-feed evidence can establish faithfulness. Until then this build is schema-compatible only and must not be described as production-ready.','',
    '## Reproduction','Run `bash v2/retrain/run_retrain.sh` from the repository root. Network fetch manifests include parameters, resolved URLs, response status, row counts, and response SHA-256 hashes. All outputs remain under `v2/retrain/`.']
    (OUT/'report_retrain.md').write_text('\n'.join(L))

def main():
    warnings.filterwarnings('ignore');OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(exist_ok=True);ART.mkdir(exist_ok=True);SERV.mkdir(exist_ok=True);seed_all()
    for stale in (OUT/'metrics_retrain.json',OUT/'report_retrain.md',ART/'training_log.json'):stale.unlink(missing_ok=True)
    import torch,ray
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required by contract')
    device=torch.cuda.get_device_name(0);manifest=[];initial=probe('initial',manifest);archive_soil=probe_archive_soil(manifest)
    # Strict all-site/all-round approval from actual returned values.
    for s,e in initial.items():
      for v in APPROVED:
       if e['variables'][v]['non_null']!=e['variables'][v]['rows'] or not e['variables'][v]['rows']:raise AssertionError(f'{s} live null {v}')
    contract={'contract_version':1,'generated_utc':pd.Timestamp.now('UTC').isoformat(),'endpoint':FORECAST_URL,
      'approval_rule':'all rows non-null at every site in initial and verification probes','approved':APPROVED,
      'derived_calendar_features':CALENDAR,
      'dropped':[{'name':'soil_temperature_0_to_7cm','reason':'live Forecast field returned all NULL; archive-only layer is not servable'},
        {'name':'soil_temperature_0cm','reason':'live-served, but ERA5 Archive returned all NULL; excluded rather than substituted'},
        {'name':'soil_temperature_6cm','reason':'live-served, but ERA5 Archive returned all NULL; excluded rather than substituted'}],
      'served_but_not_archive_reproducible':['soil_temperature_0cm','soil_temperature_6cm'],
      'sites':{s:{'airport':SITES[s]['icao'],'initial_probe':e,'archive_soil_probe':archive_soil[s]} for s,e in initial.items()}}
    (OUT/'production_feature_contract.json').write_text(json.dumps(contract,indent=2))
    ray.init(num_cpus=min(8,os.cpu_count() or 4),num_gpus=1,include_dashboard=False,log_to_driver=False,_temp_dir='/tmp/calfog_retrain_ray')
    mp=OUT/'metrics_retrain.partial.json';lp=ART/'training_log.partial.json'
    if mp.exists() and lp.exists():
        metrics=json.loads(mp.read_text());log=json.loads(lp.read_text());print(f"Resuming completed sites: {list(metrics.get('per_site',{}))}",flush=True)
    else:
        metrics={'sites':list(SITES),'seed':SEED,'faithful':False,'train_source':'era5_archive','schema_compatible':True,
          'gpu':{'used':True,'device':device},'feature_set':APPROVED+CALENDAR,
          'eval':{'scheme':'three expanding-origin folds','purge_hours':24,'bootstrap_resamples':N_BOOT,'bootstrap_block':'day','calibration':'fold-local isotonic','cost_grid_fn_fp':list(COST_GRID),'ray_tune_trials_per_site_task':8},'per_site':{}}
        log={'started_utc':pd.Timestamp.now('UTC').isoformat(),'device':device,'cuda':True,'seed':SEED,'ray_tune_trials':0,'per_site':{}}
    bundles={}
    for site,sd in SITES.items():
        if site in metrics.get('per_site',{}) and all((SERV/site/f'{t}_temporal_cnn.pt').exists() for t in ('nowcast','lead_time')):
            print(f'=== {site} resume existing real artifacts ===',flush=True);_cached=fetch_archive(site,sd,manifest);del _cached;bundles[site]={t:load_serving_bundle(site,t) for t in ('nowcast','lead_time')};continue
        print(f'=== {site} archive/retrain ===',flush=True);weather=fetch_archive(site,sd,manifest);vis=load_visibility(site);now,lead=ff.label_tables(weather,vis,site)
        siteout={'airport':sd['icao'],'tasks':{},'fold_audit':{}};slog={'tasks':{}};bundles[site]={}
        for task,labels in [('nowcast',now),('lead_time',lead)]:
            first=rolling_origin_folds(labels)[0];pp=rp.prepare(weather,{k:first[k] for k in ('train','val','test')},False,False)
            cfg,bap,ntr=ff.tune_one(pp,site,task);del pp;log['ray_tune_trials']+=ntr
            result,audit,epochs=ff.evaluate_main(weather,labels,site,task,cfg,ART/site/f'eval_{task}_cnn.pt')
            audit_folds=rolling_origin_folds(labels)
            for row,fold in zip(audit,audit_folds):
                row['val_end']=str(fold['val'].issue_time.max())
            if task=='lead_time':result={'horizon':'issue 18:00 -> next-day 00:00-09:00',**result}
            siteout['tasks'][task]=result;siteout['fold_audit'][task]=audit
            bundle,serve_log=save_serving(site,task,weather,labels,cfg);bundles[site][task]=bundle
            slog['tasks'][task]={'tuning_trials':ntr,'tuning_best_validation_ap':bap,'best_config':cfg,'fold_cnn_epochs':epochs,'serving_refit':serve_log}
        write_site_spec(site);metrics['per_site'][site]=siteout;log['per_site'][site]=slog
        (OUT/'metrics_retrain.partial.json').write_text(json.dumps(metrics,indent=2,allow_nan=False));(ART/'training_log.partial.json').write_text(json.dumps(log,indent=2,allow_nan=False))
    # Second live check is deliberately after artifact construction.
    verify=probe('serving_verification',manifest)
    for s,e in verify.items():
      contract['sites'][s]['serving_verification_probe']=e
      for v in APPROVED:
       q=e['variables'][v]
       if q['non_null']!=q['rows'] or not q['rows']:raise AssertionError(f'{s} verification live null {v}')
    contract['verified_all_sites']=True;contract['verified_rounds']=2
    (OUT/'production_feature_contract.json').write_text(json.dumps(contract,indent=2));(OUT/'probe_manifest.json').write_text(json.dumps({'generated_utc':pd.Timestamp.now('UTC').isoformat(),'requests':manifest},indent=2))
    # Approximate proxy season, evaluated only when all exact variables reproduce.
    shift={'label':'APPROXIMATE / NON-VINTAGE Historical-Forecast proxy cross-check; not proof','faithfulness_proof':False,'per_site':{}}
    for site,sd in SITES.items():
        print(f'=== {site} approximate HF proxy ===',flush=True);hw,hv,missing=fetch_proxy_and_labels(site,sd,manifest)
        shift['per_site'][site]=proxy_crosscheck(site,hw,hv,bundles[site],metrics['per_site'][site]['tasks'],missing)
    metrics['shift_crosscheck']=shift
    ray.shutdown();log['completed_utc']=pd.Timestamp.now('UTC').isoformat();log['elapsed_seconds']=(pd.Timestamp(log['completed_utc'])-pd.Timestamp(log['started_utc'])).total_seconds()
    (OUT/'probe_manifest.json').write_text(json.dumps({'generated_utc':pd.Timestamp.now('UTC').isoformat(),'requests':manifest},indent=2))
    (OUT/'metrics_retrain.json').write_text(json.dumps(metrics,indent=2,allow_nan=False));(ART/'training_log.json').write_text(json.dumps(log,indent=2,allow_nan=False))
    old=json.load(open(ROOT/'v2'/'full'/'metrics_full.json'));report(metrics,contract,old)
    print(json.dumps({'status':'complete','device':device,'ray_tune_trials':log['ray_tune_trials'],'elapsed_seconds':log['elapsed_seconds']},indent=2))
if __name__=='__main__':main()
