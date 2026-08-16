#!/usr/bin/env python3
"""Private shadow predictor. It only appends local logs; it exposes no server/endpoint."""
from __future__ import annotations
import argparse, json, sqlite3, sys, uuid
from functools import lru_cache
from pathlib import Path
import joblib, numpy as np, pandas as pd, requests, torch
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from v2.run_poc import TemporalCNN
from v2.shadow.shadow_common import (SHADOW,SITES,FEATURES,build_log_record,apply_calibrator,
 raw_hourly_to_frame,window_ending,tree_view,sha256_bytes,utc_iso)
FORECAST_URL='https://api.open-meteo.com/v1/forecast'
HISTORICAL_URL='https://historical-forecast-api.open-meteo.com/v1/forecast'
HOURLY=','.join(FEATURES[:12])
FEED_HIST='Open-Meteo Historical-Forecast best-match (non-vintage approximate proxy)'
FEED_FORWARD='Open-Meteo Forecast API most-recent analysis hours (retrieval snapshot)'

def append_manifest(entry):
    path=SHADOW/'fetch_manifest.json'
    try:d=json.loads(path.read_text()) if path.exists() else {'fetches':[]}
    except json.JSONDecodeError:d={'fetches':[]}
    if isinstance(d,list):d={'fetches':d}
    d.setdefault('fetches',[]).append(entry);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(path)

def fetch_payload(site,issue_local,feed='forward',session=requests):
    spec=SITES[site];issue=pd.Timestamp(issue_local)
    if issue.tzinfo is not None:issue=issue.tz_convert('America/Los_Angeles')
    issue=issue.tz_localize(None)
    if feed=='historical':
        url=HISTORICAL_URL;params={'latitude':spec['latitude'],'longitude':spec['longitude'],'start_date':str((issue-pd.Timedelta(days=2)).date()),'end_date':str((issue+pd.Timedelta(days=1)).date()),'hourly':HOURLY,'timezone':'UTC','models':'best_match','temperature_unit':'celsius','wind_speed_unit':'kmh','precipitation_unit':'mm'}
    else:
        now=pd.Timestamp.now(tz='America/Los_Angeles')
        aware=pd.Timestamp(issue_local)
        if aware.tzinfo is None:aware=aware.tz_localize('America/Los_Angeles')
        else:aware=aware.tz_convert('America/Los_Angeles')
        if abs((now-aware).total_seconds())>3*3600:raise ValueError('forward feed may only be used within 3 hours of retrieval; use historical for replay')
        url=FORECAST_URL;params={'latitude':spec['latitude'],'longitude':spec['longitude'],'hourly':HOURLY,'timezone':'UTC','past_days':2,'forecast_days':1,'models':'best_match','temperature_unit':'celsius','wind_speed_unit':'kmh','precipitation_unit':'mm'}
    retrieved=pd.Timestamp.now(tz='UTC');r=session.get(url,params=params,timeout=90);body=r.content
    rows=0
    try:payload=r.json();rows=len(payload.get('hourly',{}).get('time',[]))
    except Exception:payload={}
    snapdir=SHADOW/'raw_snapshots'/('forward' if feed=='forward' else 'historical_single');snapdir.mkdir(parents=True,exist_ok=True)
    snap=snapdir/f'{site}_{retrieved.strftime("%Y%m%dT%H%M%S%fZ")}_{uuid.uuid4().hex[:8]}.json';snap.write_bytes(body)
    append_manifest({'kind':'forecast_input','site':site,'feed':feed,'source_url':url,'params':params,'resolved_url':getattr(r,'url',url),'retrieved_utc':utc_iso(retrieved),'http_status':int(r.status_code),'rows':int(rows),'sha256':sha256_bytes(body),'raw_path':str(snap.relative_to(SHADOW))})
    r.raise_for_status();return payload,retrieved

@lru_cache(maxsize=10)
def load_bundle(site,task):
    td=SHADOW/'serving'/site/task;pre=joblib.load(td/'preprocessing.joblib');md=json.loads((td/'metadata.json').read_text())
    x=XGBClassifier();x.load_model(td/'xgboost.json')
    ck=torch.load(td/'temporal_cnn.pt',map_location='cpu',weights_only=False);cfg=ck['config'];c=TemporalCNN(len(FEATURES),int(cfg['channels']),int(cfg['kernel']),float(cfg['dropout']));c.load_state_dict(ck['state_dict']);c.eval()
    return pre,md,x,c,joblib.load(td/'calibrator_xgboost.joblib'),joblib.load(td/'calibrator_temporal_cnn.joblib')

def transform_window(X,pre):
    n=X.shape[-1];im,sc=pre['imputer'],pre['scaler'];return sc.transform(im.transform(X.reshape(-1,n))).reshape(X.shape).astype(np.float32)

def predict_from_frame(site,issue_local,frame,feed_source,created_utc=None,append=True):
    local=pd.Timestamp(issue_local)
    if local.tzinfo is None:local=local.tz_localize('America/Los_Angeles')
    else:local=local.tz_convert('America/Los_Angeles')
    X,end_local=window_ending(frame,local.tz_localize(None));issue_utc=local.tz_convert('UTC');end_utc=end_local.tz_localize('America/Los_Angeles').tz_convert('UTC')
    if end_utc>issue_utc:raise AssertionError('feature latency violation')
    missing=[c for j,c in enumerate(FEATURES) if np.isnan(X[:,:,j]).any()]
    faithful=not missing
    records=[]
    for task in ('nowcast','lead_time'):
        if task=='lead_time' and local.hour!=18:continue
        if task=='nowcast':target={'start':utc_iso(issue_utc),'end_exclusive':utc_iso(issue_utc+pd.Timedelta(hours=1))}
        else:
            start=(local.normalize()+pd.Timedelta(days=1)).tz_convert('UTC');target={'start':utc_iso(start),'end_exclusive':utc_iso(start+pd.Timedelta(hours=10))}
        pre,md,xgb,cnn,calx,calc=load_bundle(site,task);Xt=transform_window(X,pre)
        rawx=float(xgb.predict_proba(tree_view(Xt))[0,1])
        with torch.no_grad():rawc=float(torch.sigmoid(cnn(torch.from_numpy(Xt))).numpy()[0])
        for model,raw,cal in (('xgboost',rawx,calx),('temporal_cnn',rawc,calc)):
            prob=float(apply_calibrator(cal,[raw])[0]);ths=md['cost_thresholds'][model];decisions={str(k):bool(prob>=float(v)) for k,v in ths.items()}
            rec=build_log_record(site=site,task=task,issue_time_utc=issue_utc,target_window_utc=target,model=model,prob=prob,feed_source=feed_source,feature_end_time_utc=end_utc,costgrid_decisions=decisions,created_utc=created_utc,faithful=faithful,nonfaithful_features=missing)
            records.append(rec)
    if append:
        for rec in records:append_prediction(rec)
    return records

def append_predictions(records, jsonl_path=None):
    records=list(records); path=Path(jsonl_path) if jsonl_path is not None else SHADOW/'predictions.jsonl'
    with path.open('a') as f:
        for rec in records:f.write(json.dumps(rec,separators=(',',':'),allow_nan=False)+'\n')
    # A caller-specified test path deliberately tests JSONL only, without touching the production-private SQLite log.
    if jsonl_path is not None:return
    db=sqlite3.connect(SHADOW/'predictions.sqlite');db.execute("CREATE TABLE IF NOT EXISTS predictions (site TEXT,task TEXT,issue_time_utc TEXT,target_window_utc TEXT,model TEXT,prob REAL,feed_source TEXT,feature_end_time_utc TEXT,costgrid_decisions TEXT,created_utc TEXT,faithful INTEGER,nonfaithful_features TEXT)")
    rows=[(r['site'],r['task'],r['issue_time_utc'],json.dumps(r['target_window_utc']),r['model'],r['prob'],r['feed_source'],r['feature_end_time_utc'],json.dumps(r['costgrid_decisions']),r['created_utc'],int(r.get('faithful',False)),json.dumps(r.get('nonfaithful_features',[]))) for r in records]
    db.executemany('INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',rows);db.commit();db.close()

def append_prediction(path_or_record, record=None):
    """Append a record; `(record)` is normal and `(path, record)` supports isolated tests."""
    if record is None:append_predictions([path_or_record])
    else:append_predictions([record],jsonl_path=path_or_record)

def main(argv=None):
    p=argparse.ArgumentParser(description='private log-only CalFog shadow predictor (no output endpoint)');p.add_argument('site',choices=SITES);p.add_argument('issue_time',help='ISO time; naive means America/Los_Angeles');p.add_argument('--feed',choices=['forward','historical'],default='forward');a=p.parse_args(argv)
    issue=pd.Timestamp(a.issue_time);payload,retrieved=fetch_payload(a.site,issue,a.feed);frame=raw_hourly_to_frame(payload);source=FEED_FORWARD if a.feed=='forward' else FEED_HIST
    recs=predict_from_frame(a.site,issue,frame,source,created_utc=retrieved,append=True)
    print(json.dumps({'logged_private_records':len(recs),'site':a.site,'issue_time':a.issue_time,'shadow_only':True}))
if __name__=='__main__':main()
