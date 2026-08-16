#!/usr/bin/env python3
"""One-shot public AviationWeather METAR label update for recent forward predictions."""
from __future__ import annotations
import argparse,json,re,sys,uuid
from pathlib import Path
import pandas as pd,requests
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from v2.shadow.shadow_common import SHADOW,SITES,sha256_bytes,utc_iso
from v2.shadow.shadow_predict import append_manifest
URL='https://aviationweather.gov/api/data/metar'

def visibility_miles(value):
    if value is None:return None
    m=re.search(r'\d+(?:\.\d+)?',str(value));return float(m.group()) if m else None

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
            when=x.get('obsTime') or x.get('reportTime') or x.get('receiptTime');t=pd.to_datetime(when,utc=True,unit='s' if isinstance(when,(int,float)) else None,errors='coerce');v=visibility_miles(x.get('visib'))
            if pd.notna(t) and v is not None:obs.append((t.floor('h'),v))
        if not obs:continue
        z=pd.DataFrame(obs,columns=['hour','vsby']);h=z.groupby('hour').vsby.agg(['min','size']).reset_index();labels={r.hour:int(r['min']*1609.344<1610) for _,r in h.iterrows()}
        for hour,label in labels.items():out.append({'site':site,'task':'nowcast','issue_time_utc':utc_iso(hour),'label':label,'label_source':'AviationWeather Data API METAR hourly minimum visibility','metar_reports':int(h.loc[h.hour==hour,'size'].iloc[0])})
        for issue in pd.date_range(h.hour.min().tz_convert('America/Los_Angeles').normalize(),h.hour.max().tz_convert('America/Los_Angeles').normalize(),freq='D')+pd.Timedelta(hours=18):
            st=(issue.normalize()+pd.Timedelta(days=1)).tz_convert('UTC');hs=pd.date_range(st,periods=10,freq='h')
            if all(x in labels for x in hs):out.append({'site':site,'task':'lead_time','issue_time_utc':utc_iso(issue.tz_convert('UTC')),'label':int(any(labels[x] for x in hs)),'label_source':'AviationWeather Data API METAR; complete 10-hour lead target'})
    path=SHADOW/'outcomes_forward.jsonl';existing=pd.read_json(path,lines=True).to_dict('records') if path.exists() and path.stat().st_size else []
    merged={(x['site'],x['task'],x['issue_time_utc']):x for x in [*existing,*out]}
    with path.open('w') as f:
        for x in merged.values():f.write(json.dumps(x,separators=(',',':'))+'\n')
    print(json.dumps({'recent_forward_outcomes':len(out),'shadow_only':True}))
if __name__=='__main__':main()
