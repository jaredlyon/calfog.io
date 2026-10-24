#!/usr/bin/env python3
"""Score one predeclared feed/time cohort against its own METAR outcomes."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,precision_score,recall_score,f1_score
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from models._shared.evaluation import block_bootstrap,COST_GRID,rps
from ops.shadow.shadow_common import SHADOW,SITES

FEEDS={'historical':'Open-Meteo Historical-Forecast best-match (non-vintage approximate proxy)',
       'forward':'Open-Meteo Forecast API most-recent analysis hours (retrieval snapshot)'}

def ece(y,p,bins=10):
    y=np.asarray(y,int);p=np.asarray(p,float);total=0.;edges=np.linspace(0,1,bins+1)
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        ix=(p>=lo)&((p<hi) if i<bins-1 else (p<=hi))
        if ix.any():total+=float(ix.mean()*abs(y[ix].mean()-p[ix].mean()))
    return total

def join_predictions_outcomes(predictions,outcomes):
    keys=['site','task','issue_time_utc']
    predictions=predictions.copy();outcomes=outcomes.copy()
    for frame in (predictions,outcomes):
        frame['issue_time_utc']=pd.to_datetime(frame.issue_time_utc,utc=True)
    if 'model' in outcomes.columns and 'model' in predictions.columns:keys.append('model')
    identity=['site','task','issue_time_utc','model']
    if 'feed_source' in predictions.columns:identity.append('feed_source')
    # Append order is the capture order; first eligible issuance is frozen, not revised after labels.
    p=predictions.drop_duplicates(identity,keep='first')
    o=outcomes.dropna(subset=['fog']).drop_duplicates(keys,keep='last')
    fields=keys+['fog']+(['category'] if 'category' in o else [])
    return p.merge(o[fields],on=keys,how='inner',validate='many_to_one' if 'model' not in keys else 'one_to_one')

def join_and_score(predictions:pd.DataFrame,outcomes:pd.DataFrame)->dict:
    z=join_predictions_outcomes(predictions,outcomes);y=z.fog.to_numpy(dtype=int);p=z.p_fog.to_numpy(dtype=float)
    if not len(z):return {'status':'no_evaluable_issues','n_issues':0,'n_pos':0,'live_ap':None,'roc_auc':None,'brier':None,'ece':None,'rps':None,'n_ordinal':0}
    ordinal=z.dropna(subset=['category']) if 'category' in z else z.iloc[:0]
    ranked=rps(ordinal.category.to_numpy(int),np.stack([json.loads(x) if isinstance(x,str) else x for x in ordinal.pmf])) if len(ordinal) else None
    return {'status':'scored','n_issues':int(len(z)),'n_pos':int(y.sum()),'live_ap':float(average_precision_score(y,p)) if y.sum() else 0.0,'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else .5,'brier':float(brier_score_loss(y,p)),'ece':ece(y,p),'rps':ranked,'n_ordinal':int(len(ordinal))}

def operating_points(z):
    y=z.fog.to_numpy(int);out={}
    for ratio in COST_GRID:
        dec=np.array([bool((x if isinstance(x,dict) else json.loads(x)).get(str(ratio),False)) for x in z.costgrid_decisions])
        tp=int(np.sum(dec&(y==1)));fp=int(np.sum(dec&(y==0)));fn=int(np.sum((~dec)&(y==1)));tn=int(np.sum((~dec)&(y==0)))
        out[str(ratio)]={'fn_fp':f'{ratio}:1','precision':float(precision_score(y,dec,zero_division=0)),'recall':float(recall_score(y,dec,zero_division=0)),'f1':float(f1_score(y,dec,zero_division=0)),'alert_rate':float(dec.mean()),'expected_cost_per_issue':float((ratio*fn+fp)/max(1,len(y))),'tp':tp,'fp':fp,'fn':fn,'tn':tn}
    return out

def read_log(path,columns):
    if not path.exists() or not path.stat().st_size:return pd.DataFrame(columns=columns)
    frame=pd.read_json(path,lines=True,convert_dates=False)
    return frame.reindex(columns=list(dict.fromkeys([*frame.columns,*columns])))

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--feed',choices=FEEDS,required=True)
    parser.add_argument('--start',required=True,help='inclusive UTC issue cutoff');parser.add_argument('--end',required=True,help='exclusive UTC issue cutoff');a=parser.parse_args(argv)
    start,end=pd.Timestamp(a.start),pd.Timestamp(a.end)
    if start.tzinfo is None or end.tzinfo is None or start>=end:raise ValueError('require aware start < end')
    cols=['site','task','issue_time_utc','model','feed_source','p_fog','pmf','faithful','nonfaithful_features','costgrid_decisions']
    preds=read_log(SHADOW/'predictions.jsonl',cols)
    times=pd.to_datetime(preds.issue_time_utc,utc=True)
    mask=(preds.feed_source==FEEDS[a.feed])&(times>=start)&(times<end)&preds.p_fog.notna()&preds.pmf.notna()
    if a.feed=='forward':
        # Old rows without actual completion evidence are not prospective evidence.
        for field in ('created_utc','issued_at_utc'):
            actual=pd.to_datetime(preds[field],utc=True,errors='coerce') if field in preds else pd.Series(pd.NaT,index=preds.index,dtype='datetime64[ns, UTC]')
            mask &= actual.notna() & (actual<=times) & (actual>=times-pd.Timedelta(minutes=1))
    preds=preds[mask].copy()
    outcome_name='outcomes.jsonl' if a.feed=='historical' else 'outcomes_forward.jsonl'
    outs=read_log(SHADOW/outcome_name,['site','task','issue_time_utc','fog','category'])
    if a.feed=='forward':
        if {'finalized_utc','verification_end_utc'}<=set(outs.columns):
            finalized=pd.to_datetime(outs.finalized_utc,utc=True,errors='coerce')
            ended=pd.to_datetime(outs.verification_end_utc,utc=True,errors='coerce')
            outs=outs[finalized.notna()&ended.notna()&(ended+pd.Timedelta(hours=1)<=finalized)].copy()
        else:outs=outs.iloc[:0]
    retro={model:json.loads((ROOT/'models'/model/f'{model}_metrics_v2.json').read_text()) for model in ('xgboost','temporal_cnn')}
    result={'generated_utc':pd.Timestamp.now(tz='UTC').isoformat(),'cohort':{'feed':a.feed,'feed_source':FEEDS[a.feed],'issue_start_inclusive_utc':start.isoformat(),'issue_end_exclusive_utc':end.isoformat(),'timezone':'Etc/GMT+8','duplicate_policy':'first eligible append per site/task/issue/model/feed'},'label_contract':'decoded ordinal hourly minimum; fog category <=4; lead all ten fixed-PST hours 00-09; forward finalization lag 1h','bootstrap':{'resamples':1000,'unit':'Etc/GMT+8 issue day','method':'shared whole-day block bootstrap; point-centered 95% absolute-deviation interval'},'per_site':{}}
    for si,site in enumerate(SITES):
      sd={'tasks':{}};result['per_site'][site]=sd
      for ti,task in enumerate(('nowcast','lead_time')):
        td={'feed_source':FEEDS[a.feed]};sd['tasks'][task]=td
        for mi,model in enumerate(('xgboost','temporal_cnn')):
            p=preds[(preds.site==site)&(preds.task==task)&(preds.model==model)].copy();z=join_predictions_outcomes(p,outs);base=join_and_score(p,outs)
            rap=float(retro[model]['per_site'][site]['tasks'][task][model]['ap'])
            if z.empty:
                td[model]={**base,'live_ap_ci':None,'live_roc_auc_ci':None,'bootstrap_resamples':0,'valid_ap_resamples':0,'retrospective_ap':rap,'delta_ap':None,'feed_source':FEEDS[a.feed],'faithful':None,'nonfaithful_features':[],'vintage_faithful':False,'cost_operating_points':{}}
                continue
            local=pd.to_datetime(z.issue_time_utc,utc=True).dt.tz_convert('Etc/GMT+8').dt.tz_localize(None);meta=pd.DataFrame({'issue_time':local,'fold':np.ones(len(z),int)})
            boot=block_bootstrap(z.fog.to_numpy(int),z.p_fog.to_numpy(float),meta,n_boot=1000,seed=20250917+100*si+10*ti+mi,block='day')
            nonfaith=sorted(set(sum((x if isinstance(x,list) else json.loads(x) for x in z.nonfaithful_features),[])))
            td[model]={**base,'live_ap_ci':boot['ap_ci'],'live_roc_auc_ci':boot['roc_auc_ci'],'bootstrap_resamples':1000,'valid_ap_resamples':boot['valid_ap_resamples'],'retrospective_ap':rap,'delta_ap':float(base['live_ap']-rap),'feed_source':FEEDS[a.feed],'faithful':bool(z.faithful.all()),'nonfaithful_features':nonfaith,'vintage_faithful':False,'cost_operating_points':operating_points(z)}
    (SHADOW/f'shadow_metrics_{a.feed}.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'sites':len(result['per_site']),'prediction_rows':len(preds),'outcome_rows':len(outs)}))
if __name__=='__main__':main()
