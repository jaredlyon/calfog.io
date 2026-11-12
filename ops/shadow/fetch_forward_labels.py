#!/usr/bin/env python3
"""One-shot public AviationWeather METAR label update for recent forward predictions."""
from __future__ import annotations
import argparse,json,sys,uuid
from pathlib import Path
import pandas as pd,requests
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from ops.shadow.shadow_common import SHADOW,SITES,sha256_bytes,utc_iso
from ops.shadow.shadow_predict import append_manifest
from models._shared.visibility import decode_statute_miles, hourly_labels, FOG_MAX_CATEGORY
URL='https://aviationweather.gov/api/data/metar'


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--hours',type=int,default=48);a=p.parse_args(argv)
    if not 1<=a.hours<=360:raise ValueError('AviationWeather recent archive is bounded to 1..360 hours')
    out=[]
    for site,spec in SITES.items():
        params={'ids':spec['airport'],'format':'json','hours':a.hours};ret=pd.Timestamp.now(tz='UTC');r=requests.get(URL,params=params,timeout=60);body=r.content
        d=SHADOW/'raw_snapshots'/'aviationweather_metar';d.mkdir(parents=True,exist_ok=True);path=d/f'{site}_{ret.strftime("%Y%m%dT%H%M%SZ")}_{uuid.uuid4().hex[:8]}.json';path.write_bytes(body)
        try:payload=r.json();rows=len(payload) if isinstance(payload,list) else 0
        except Exception:payload=[];rows=0
        append_manifest({'kind':'forward_metar_label','site':site,'source_url':URL,'params':params,'resolved_url':r.url,'retrieved_utc':utc_iso(ret),'http_status':int(r.status_code),'rows':rows,'sha256':sha256_bytes(body),'raw_path':str(path.relative_to(SHADOW))});r.raise_for_status()
        obs=[]
        for x in payload:
            when=x.get('obsTime') or x.get('reportTime') or x.get('receiptTime')
            t=pd.to_datetime(when,utc=True,unit='s' if isinstance(when,(int,float)) else None,errors='coerce')
            if pd.notna(t):obs.append((t,x.get('visib'),x.get('rawOb')))
        if not obs:continue
        z=pd.DataFrame(obs,columns=['time_utc','vsby','raw'])
        decoded=decode_statute_miles(z.vsby,raw=z.raw);decoded['time_utc']=z.time_utc
        h=hourly_labels(decoded)
        h['hour']=h.time.dt.tz_localize('Etc/GMT+8').dt.tz_convert('UTC')
        # Declared settling lag: one full hour after the verification hour closes.
        h=h[h.hour+pd.Timedelta(hours=2)<=ret]
        if h.empty:continue
        for row in h.dropna(subset=['fog']).itertuples():
            out.append({'site':site,'task':'nowcast','issue_time_utc':utc_iso(row.hour),
                        'category':None if pd.isna(row.category) else int(row.category),'fog':bool(row.fog),
                        'label_source':'AviationWeather ordinal hourly minimum','metar_reports':int(row.n_reports),
                        'finalized_utc':utc_iso(ret),'verification_end_utc':utc_iso(row.hour+pd.Timedelta(hours=1))})
        by={row.hour:row for row in h.itertuples()}
        for issue in pd.date_range(h.hour.min().tz_convert('Etc/GMT+8').normalize(),h.hour.max().tz_convert('Etc/GMT+8').normalize(),freq='D')+pd.Timedelta(hours=18):
            st=(issue.normalize()+pd.Timedelta(days=1)).tz_convert('UTC')
            rows=[by.get(x) for x in pd.date_range(st,periods=10,freq='h')]
            if not all(x is not None for x in rows):continue
            lo=min(x.cat_lo for x in rows);hi=min(x.cat_hi for x in rows)
            if not (hi<=FOG_MAX_CATEGORY or lo>FOG_MAX_CATEGORY):continue
            out.append({'site':site,'task':'lead_time','issue_time_utc':utc_iso(issue),
                        'category':int(lo) if lo==hi else None,'fog':bool(hi<=FOG_MAX_CATEGORY),
                        'label_source':'AviationWeather ordinal minimum; complete ten-hour lead target',
                        'finalized_utc':utc_iso(ret),'verification_end_utc':utc_iso(st+pd.Timedelta(hours=10))})
    path=SHADOW/'outcomes_forward.jsonl';existing=pd.read_json(path,lines=True,convert_dates=False).to_dict('records') if path.exists() and path.stat().st_size else []
    merged={(x['site'],x['task'],x['issue_time_utc']):x for x in [*existing,*out]}
    with path.open('w') as f:
        for x in merged.values():f.write(json.dumps(x,separators=(',',':'))+'\n')
    print(json.dumps({'recent_forward_outcomes':len(out),'shadow_only':True}))
if __name__=='__main__':main()
