#!/usr/bin/env python3
"""Fetch and replay the completed 2025-12-01..2026-02-28 issue-time season."""
from __future__ import annotations
import argparse, io, json, sys, time, uuid
from pathlib import Path
import numpy as np,pandas as pd,requests,torch
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from v2.shadow.shadow_common import SHADOW,SITES,FEATURES,raw_hourly_to_frame,window_ending,tree_view,apply_calibrator,build_log_record,sha256_bytes,utc_iso
from v2.shadow.shadow_predict import HISTORICAL_URL,HOURLY,FEED_HIST,append_manifest,append_predictions,load_bundle,transform_window
IEM_URL='https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py'
MONTHS=[('2025-12-01','2025-12-31'),('2026-01-01','2026-01-31'),('2026-02-01','2026-02-28')]

def save_raw(kind,site,label,body):
    d=SHADOW/'raw_snapshots'/kind;d.mkdir(parents=True,exist_ok=True);p=d/f'{site}_{label}_{sha256_bytes(body)[:12]}.json' if kind=='historical_forecast' else d/f'{site}_{label}_{sha256_bytes(body)[:12]}.csv';p.write_bytes(body);return p

def get(session,url,params,kind,site,label):
    last=None
    for attempt in range(1,6):
        retrieved=pd.Timestamp.now(tz='UTC');r=session.get(url,params=params,timeout=(15,150));last=r;body=r.content;p=save_raw(kind,site,f'{label}_attempt{attempt}',body)
        if kind=='historical_forecast':
            try:rows=len(r.json().get('hourly',{}).get('time',[]))
            except Exception:rows=0
        else:rows=max(0,len(body.decode(errors='replace').splitlines())-1)
        append_manifest({'kind':kind,'site':site,'source_url':url,'params':params,'resolved_url':r.url,'retrieved_utc':utc_iso(retrieved),'http_status':int(r.status_code),'rows':int(rows),'sha256':sha256_bytes(body),'raw_path':str(p.relative_to(SHADOW)),'attempt':attempt})
        if r.status_code==200:return r,retrieved
        if r.status_code not in (429,500,502,503,504):r.raise_for_status()
        time.sleep(4*attempt)
    last.raise_for_status()

def fetch_weather(session,site):
    frames=[];retrievals=[]
    # UTC padding avoids the historical API's observed fixed-current-offset local response defect.
    pads=[('2025-11-29','2026-01-02'),('2025-12-30','2026-02-02'),('2026-01-30','2026-03-02')]
    for mi,(start,end) in enumerate(pads):
        spec=SITES[site];params={'latitude':spec['latitude'],'longitude':spec['longitude'],'start_date':start,'end_date':end,'hourly':HOURLY,'timezone':'UTC','models':'best_match','temperature_unit':'celsius','wind_speed_unit':'kmh','precipitation_unit':'mm'}
        r,ret=get(session,HISTORICAL_URL,params,'historical_forecast',site,f'{mi+1}_{start}_{end}');frames.append(raw_hourly_to_frame(r.json()));retrievals.append(ret);time.sleep(.15)
    z=pd.concat(frames,ignore_index=True).sort_values('time').drop_duplicates('time',keep='last').reset_index(drop=True)
    return z,max(retrievals)

def fetch_iem(session,site):
    chunks=[]
    # Ten-day requests avoid overloading the free IEM service while padding through Mar-1 labels.
    starts=pd.date_range('2025-11-30','2026-03-01',freq='10D')
    for mi,start in enumerate(starts):
        end=min(start+pd.Timedelta(days=10),pd.Timestamp('2026-03-02'))
        params=[('station',SITES[site]['iem_station']),('data','vsby'),('year1',start.year),('month1',start.month),('day1',start.day),('year2',end.year),('month2',end.month),('day2',end.day),('tz','UTC'),('format','onlycomma'),('latlon','no'),('elev','no'),('missing','empty'),('trace','empty'),('direct','no'),('report_type','3'),('report_type','4')]
        r,_=get(session,IEM_URL,params,'iem_metar',site,f'{mi+1}_{start:%Y%m%d}_{end:%Y%m%d}');d=pd.read_csv(io.BytesIO(r.content));chunks.append(d);time.sleep(.3)
    z=pd.concat(chunks,ignore_index=True);z['valid']=pd.to_datetime(z.valid,utc=True,errors='coerce');z['vsby']=pd.to_numeric(z.vsby,errors='coerce');z=z.dropna(subset=['valid']).drop_duplicates(['valid','vsby']).sort_values('valid')
    numeric=z.dropna(subset=['vsby']).copy();numeric['hour_utc']=numeric.valid.dt.floor('h');hour=numeric.groupby('hour_utc',as_index=False).agg(min_vsby_miles=('vsby','min'),metar_reports=('vsby','size'))
    hour['visibility_meters']=hour.min_vsby_miles*1609.344;hour['label']=(hour.visibility_meters<1610.0).astype(int);return z,hour

def outcomes_for_site(site,hour):
    h=hour.copy();h['local']=h.hour_utc.dt.tz_convert('America/Los_Angeles');start=pd.Timestamp('2025-12-01',tz='America/Los_Angeles');end=pd.Timestamp('2026-03-01',tz='America/Los_Angeles')
    hn=h[(h.local>=start)&(h.local<end)].copy();out=[]
    for r in hn.itertuples():
        issue=r.hour_utc;out.append({'site':site,'task':'nowcast','issue_time_utc':utc_iso(issue),'label':int(r.label),'visibility_meters':float(r.visibility_meters),'label_source':'IEM ASOS archival METAR min numeric visibility within UTC hour','metar_reports':int(r.metar_reports)})
    by={x.hour_utc:x for x in hour.itertuples()}
    for day in pd.date_range('2025-12-01','2026-02-28',freq='D'):
        issue=day+pd.Timedelta(hours=18);aware=issue.tz_localize('America/Los_Angeles');target_start=(day+pd.Timedelta(days=1)).tz_localize('America/Los_Angeles').tz_convert('UTC');hours=pd.date_range(target_start,periods=10,freq='h');rows=[by.get(x) for x in hours]
        if all(x is not None for x in rows):out.append({'site':site,'task':'lead_time','issue_time_utc':utc_iso(aware.tz_convert('UTC')),'label':int(any(x.label for x in rows)),'visibility_meters':float(min(x.visibility_meters for x in rows)),'label_source':'IEM ASOS archival METAR; all 10 local 00:00-09:00 hours complete; any hourly min <1610m','metar_reports':int(sum(x.metar_reports for x in rows))})
    return out

def predict_batches(site,weather,created):
    records=[]
    task_issues={'nowcast':pd.date_range('2025-12-01','2026-02-28 23:00',freq='h'),'lead_time':pd.date_range('2025-12-01 18:00','2026-02-28 18:00',freq='D')}
    for task,issues in task_issues.items():
        arrays=[];kept=[]
        for issue in issues:
            try:X,end=window_ending(weather,issue)
            except Exception:continue
            arrays.append(X[0]);kept.append((issue,end))
        X=np.stack(arrays).astype(np.float32);missing_by=[[c for j,c in enumerate(FEATURES) if np.isnan(row[:,j]).any()] for row in X]
        pre,md,xgb,cnn,calx,calc=load_bundle(site,task);Xt=transform_window(X,pre);rawx=xgb.predict_proba(tree_view(Xt))[:,1]
        with torch.no_grad():rawc=torch.sigmoid(cnn(torch.from_numpy(Xt))).numpy()
        for model,raw,cal in (('xgboost',rawx,calx),('temporal_cnn',rawc,calc)):
            probs=apply_calibrator(cal,raw);ths=md['cost_thresholds'][model]
            for i,((issue,end),prob) in enumerate(zip(kept,probs)):
                local=issue.tz_localize('America/Los_Angeles');iu=local.tz_convert('UTC');eu=end.tz_localize('America/Los_Angeles').tz_convert('UTC')
                if task=='nowcast':target={'start':utc_iso(iu),'end_exclusive':utc_iso(iu+pd.Timedelta(hours=1))}
                else:
                    st=(local.normalize()+pd.Timedelta(days=1)).tz_convert('UTC');target={'start':utc_iso(st),'end_exclusive':utc_iso(st+pd.Timedelta(hours=10))}
                missing=missing_by[i];records.append(build_log_record(site=site,task=task,issue_time_utc=iu,target_window_utc=target,model=model,prob=float(prob),feed_source=FEED_HIST,feature_end_time_utc=eu,costgrid_decisions={str(k):bool(prob>=float(v)) for k,v in ths.items()},created_utc=created,faithful=not missing,nonfaithful_features=missing))
    return records

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--reset',action='store_true');a=p.parse_args(argv)
    if a.reset:
        for x in ('predictions.jsonl','predictions.sqlite','outcomes.jsonl','fetch_manifest.json'): (SHADOW/x).unlink(missing_ok=True)
    session=requests.Session();session.headers['User-Agent']='CalFog-v2-private-shadow-research/1.0 (public-data-only)';allout=[];n=0
    for site in SITES:
        print('FETCH/REPLAY',site,flush=True);weather,created=fetch_weather(session,site);_,hour=fetch_iem(session,site);outs=outcomes_for_site(site,hour);recs=predict_batches(site,weather,created);append_predictions(recs);allout.extend(outs);n+=len(recs);print(site,len(recs),len(outs),flush=True)
    with (SHADOW/'outcomes.jsonl').open('w') as f:
        for r in allout:f.write(json.dumps(r,separators=(',',':'))+'\n')
    print(json.dumps({'prediction_records':n,'outcomes':len(allout),'successful_fetches':sum(int(x.get('http_status',0))==200 for x in json.loads((SHADOW/'fetch_manifest.json').read_text())['fetches'])}))
if __name__=='__main__':main()
