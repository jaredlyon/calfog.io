#!/usr/bin/env python3
"""Score private logged predictions against separately fetched METAR outcomes."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,precision_score,recall_score,f1_score
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from v2.full.full_utils import block_bootstrap,COST_GRID
from v2.shadow.shadow_common import SHADOW,SITES

def ece(y,p,bins=10):
    y=np.asarray(y,int);p=np.asarray(p,float);total=0.;edges=np.linspace(0,1,bins+1)
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        ix=(p>=lo)&((p<hi) if i<bins-1 else (p<=hi))
        if ix.any():total+=float(ix.mean()*abs(y[ix].mean()-p[ix].mean()))
    return total

def join_predictions_outcomes(predictions,outcomes):
    keys=['site','task','issue_time_utc'];
    if 'model' in outcomes.columns and 'model' in predictions.columns:keys.append('model')
    o=outcomes.drop_duplicates(keys,keep='last');p=predictions.drop_duplicates(keys,keep='last') if 'model' in keys else predictions
    return p.merge(o[keys+['label']],on=keys,how='inner',validate='many_to_one' if 'model' not in keys else 'one_to_one')

def join_and_score(predictions:pd.DataFrame,outcomes:pd.DataFrame)->dict:
    z=join_predictions_outcomes(predictions,outcomes);y=z.label.to_numpy(dtype=int);p=z.prob.to_numpy(dtype=float)
    if not len(z):return {'n_issues':0,'n_pos':0,'live_ap':0.0,'roc_auc':0.5,'brier':0.0,'ece':0.0}
    return {'n_issues':int(len(z)),'n_pos':int(y.sum()),'live_ap':float(average_precision_score(y,p)) if y.sum() else 0.0,'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else .5,'brier':float(brier_score_loss(y,p)),'ece':ece(y,p)}

def operating_points(z):
    y=z.label.to_numpy(int);out={}
    for ratio in COST_GRID:
        dec=np.array([bool((x if isinstance(x,dict) else json.loads(x)).get(str(ratio),False)) for x in z.costgrid_decisions])
        tp=int(np.sum(dec&(y==1)));fp=int(np.sum(dec&(y==0)));fn=int(np.sum((~dec)&(y==1)));tn=int(np.sum((~dec)&(y==0)))
        out[str(ratio)]={'fn_fp':f'{ratio}:1','precision':float(precision_score(y,dec,zero_division=0)),'recall':float(recall_score(y,dec,zero_division=0)),'f1':float(f1_score(y,dec,zero_division=0)),'alert_rate':float(dec.mean()),'expected_cost_per_issue':float((ratio*fn+fp)/max(1,len(y))),'tp':tp,'fp':fp,'fn':fn,'tn':tn}
    return out

def main():
    preds=pd.read_json(SHADOW/'predictions.jsonl',lines=True);outs=pd.read_json(SHADOW/'outcomes.jsonl',lines=True)
    if (SHADOW/'outcomes_forward.jsonl').exists():outs=pd.concat([outs,pd.read_json(SHADOW/'outcomes_forward.jsonl',lines=True)],ignore_index=True).drop_duplicates(['site','task','issue_time_utc'],keep='last')
    retro=json.loads((ROOT/'v2'/'full'/'metrics_full.json').read_text())
    result={'generated_utc':pd.Timestamp.now(tz='UTC').isoformat(),'season_issue_times_local':'2025-12-01 00:00 through 2026-02-28 23:00 America/Los_Angeles (lead issues daily 18:00)','label_contract':'IEM ASOS METAR min numeric visibility within hour; fog strictly <1610m; lead requires all 10 local hours 00-09 and is any fog','bootstrap':{'resamples':1000,'unit':'America/Los_Angeles issue day','method':'same whole-day block implementation as v2/full; point-centered 95% absolute-deviation interval'},'per_site':{}}
    for si,site in enumerate(SITES):
      sd={'tasks':{}};result['per_site'][site]=sd
      for ti,task in enumerate(('nowcast','lead_time')):
        td={'feed_source':'Open-Meteo Historical-Forecast best-match (non-vintage approximate proxy)','faithful':False,'nonfaithful_features':['soil_temperature_0_to_7cm']};sd['tasks'][task]=td
        for mi,model in enumerate(('xgboost','temporal_cnn')):
            p=preds[(preds.site==site)&(preds.task==task)&(preds.model==model)].copy();z=join_predictions_outcomes(p,outs);base=join_and_score(p,outs)
            local=pd.to_datetime(z.issue_time_utc,utc=True).dt.tz_convert('America/Los_Angeles').dt.tz_localize(None);meta=pd.DataFrame({'issue_time':local,'fold':np.ones(len(z),int)})
            boot=block_bootstrap(z.label.to_numpy(int),z.prob.to_numpy(float),meta,n_boot=1000,seed=20250917+100*si+10*ti+mi,block='day')
            rap=float(retro['per_site'][site]['tasks'][task][model]['ap']);nonfaith=sorted(set(sum((x if isinstance(x,list) else json.loads(x) for x in p.nonfaithful_features),[])))
            d={**base,'live_ap_ci':boot['ap_ci'],'live_roc_auc_ci':boot['roc_auc_ci'],'bootstrap_resamples':1000,'valid_ap_resamples':boot['valid_ap_resamples'],'retrospective_ap':rap,'delta_ap':float(base['live_ap']-rap),'feed_source':'Open-Meteo Historical-Forecast best-match (non-vintage approximate proxy)','faithful':bool(p.faithful.all()),'nonfaithful_features':nonfaith,'vintage_faithful':False,'cost_operating_points':operating_points(z)}
            td[model]=d
    (SHADOW/'shadow_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'sites':len(result['per_site']),'prediction_rows':len(preds),'outcome_rows':len(outs)}))
if __name__=='__main__':main()
