"""CPU-only synthetic checks for ordinal serving and calibrated fog logs."""
import numpy as np
import pandas as pd
import pytest
from ops.shadow import shadow_predict as predictor, shadow_score as scorer
from models._shared.visibility import N_CATEGORIES


def test_prediction_logs_pmf_and_calibrates_only_derived_fog(monkeypatch):
    pmf=np.zeros((1,N_CATEGORIES));pmf[0,4]=.3;pmf[0,17]=.7
    class Model:
        def predict_proba(self,X):return np.tile(pmf,(len(X),1))
    class Calibrator:
        def predict(self,p):
            np.testing.assert_allclose(p,[.3])
            return np.array([.8])
    metadata={'cost_thresholds':{m:{'1':.5} for m in ('xgboost','temporal_cnn')}}
    monkeypatch.setattr(predictor,'load_bundle',lambda *a:(None,metadata,Model(),None,Calibrator(),Calibrator()))
    monkeypatch.setattr(predictor,'transform_window',lambda X,pre:X)
    monkeypatch.setattr(predictor,'predict_cnn',lambda model,X,device:np.tile(pmf,(len(X),1)))
    times=pd.date_range('2026-01-01',periods=24,freq='h')
    frame=pd.DataFrame({'time':times,**{c:np.zeros(24) for c in predictor.FEATURES}})
    rows=predictor.predict_from_frame('location_6',times[-1],frame,predictor.FEED_HIST,append=False)
    assert len(rows)==2
    for row in rows:
        np.testing.assert_allclose(row['pmf'],pmf[0])
        assert row['p_fog']==.8
        assert row['costgrid_decisions']=={'1':True}
        assert 'prob' not in row


def test_rps_uses_raw_pmf_but_fog_score_uses_calibrated_probability():
    rows=pd.DataFrame({'site':['location_6']*2,'task':['nowcast']*2,'model':['xgboost']*2,
                       'issue_time_utc':['2026-01-01T08:00Z','2026-01-01T09:00Z'],
                       'p_fog':[.8,.2],'pmf':[np.eye(18)[4],np.eye(18)[17]]})
    outcomes=rows[['site','task','issue_time_utc']].copy()
    outcomes['fog']=[True,False];outcomes['category']=[4,17]
    result=scorer.join_and_score(rows,outcomes)
    assert result['rps']==0
    assert result['brier']==pytest.approx(.04)
    outcomes.loc[1,'category']=np.nan
    result=scorer.join_and_score(rows,outcomes)
    assert result['n_issues']==2
    assert result['n_ordinal']==1
    assert result['rps']==0

def test_replay_iem_uses_decoder_and_complete_ordinal_lead(monkeypatch):
    from ops.shadow import replay_hindcast as replay
    times=pd.date_range('2025-12-02T08:00Z',periods=10,freq='h')
    raw=pd.DataFrame({'valid':[t.isoformat() for t in times],'vsby':[1.]+[1.25]*9})
    class Response:
        content=raw.to_csv(index=False).encode()
    def get(session,url,params,*args):
        assert ('data','vsby') in params and ('data','metar') in params
        return Response(),times[0]
    monkeypatch.setattr(replay,'get',get)
    monkeypatch.setattr(replay.time,'sleep',lambda seconds:None)
    _,hourly=replay.fetch_iem(None,'location_6')
    assert hourly.category.tolist()==[4]+[5]*9
    outcomes=replay.outcomes_for_site('location_6',hourly)
    lead=[row for row in outcomes if row['task']=='lead_time']
    assert len(lead)==1
    assert lead[0]['issue_time_utc']=='2025-12-02T02:00:00Z'
    assert lead[0]['category']==4 and lead[0]['fog'] is True
    assert not any(row['task']=='lead_time' for row in replay.outcomes_for_site('location_6',hourly.iloc[:-1]))
