"""Small, offline retrain PMF contract checks; no training or GPU access."""
import json
import numpy as np
import pandas as pd
from ops.retrain import run_retrain as rr


class OrdinalModel:
    def predict_proba(self,X):
        return np.tile(np.eye(18)[4],(len(X),1))
    def state_dict(self):return {}


def test_serving_fit_uses_categories_and_calibrates_pmf_fog(tmp_path,monkeypatch):
    categories=np.array([0,4,17]);X=np.zeros((3,24,1),dtype=np.float32)
    prepared=rr.windows.Prepared(X,categories,X,categories,X,categories,{},['temperature_2m'])
    monkeypatch.setattr(rr,'SERV',tmp_path)
    monkeypatch.setattr(rr,'rolling_origin_folds',lambda labels:[{k:None for k in ('train','val','test')}])
    monkeypatch.setattr(rr,'raw_prepared',lambda *a:(prepared,None,None))
    seen=[]
    def fit(Xtr,ctr,Xv,cv,*,seed):
        np.testing.assert_array_equal(ctr,categories);np.testing.assert_array_equal(cv,categories)
        seen.append('fit');return OrdinalModel()
    monkeypatch.setattr(rr.xgboost,'fit_model',fit)
    monkeypatch.setattr(rr.random_forest,'fit_model',fit)
    def train(Xtr,ctr,Xv,cv,*args,**kwargs):
        np.testing.assert_array_equal(ctr,categories);np.testing.assert_array_equal(cv,categories)
        return OrdinalModel(),{'cnn_epochs':1,'cnn_best_val_nll':1.,'cnn_best_val_ap':1.}
    monkeypatch.setattr(rr.temporal_cnn,'train_cnn',train)
    monkeypatch.setattr(rr.temporal_cnn,'predict_cnn',lambda m,X,device:m.predict_proba(X))
    calibrated=[]
    def calibrate(y,p):
        np.testing.assert_array_equal(y,[1,1,0]);calibrated.append(p);return None
    monkeypatch.setattr(rr.evaluation,'fit_calibrator',calibrate)
    bundle,info=rr.save_serving('location_6','nowcast',None,None,{})
    assert len(seen)==2 and len(calibrated)==4
    for p in calibrated[1:]:np.testing.assert_array_equal(p,[1,1,1])
    np.testing.assert_allclose(calibrated[0],[2/3]*3)
    assert len(bundle['climatology'])==18
    manifest=json.loads((tmp_path/'location_6'/'nowcast_manifest.json').read_text())
    assert manifest['target']=='ordinal-18'
    assert manifest['category_sm']==list(rr.CATEGORY_SM)
    assert manifest['category_labels']==list(rr.CATEGORY_LABELS)
    assert info['cnn_epochs']==1


def test_proxy_crosscheck_reports_fog_ap_and_rps(monkeypatch):
    pmf=np.stack([np.eye(18)[4],np.eye(18)[17]])
    now=pd.DataFrame({'issue_time':pd.to_datetime(['2025-12-01','2026-02-28'])})
    lead=pd.DataFrame({'verification_start':pd.to_datetime(['2025-12-01','2026-02-28']),
                       'verification_end':pd.to_datetime(['2025-12-01 10:00','2026-02-28 10:00'])})
    monkeypatch.setattr(rr,'label_tables',lambda *a:(now,lead))
    monkeypatch.setattr(rr,'predict_proxy',lambda *a:(np.array([4,17]),{'xgboost':pmf},None))
    archive={task:{'xgboost':{'ap':1.}} for task in ('nowcast','lead_time')}
    result=rr.proxy_crosscheck('location_6',None,None,{'nowcast':{},'lead_time':{}},archive,[])
    for task in ('nowcast','lead_time'):
        metrics=result['tasks'][task]['xgboost']
        assert metrics['hf_proxy_ap']==1
        assert metrics['hf_proxy_rps']==0
        assert metrics['pos_proxy']==1


def test_report_labels_between_run_contrast(tmp_path,monkeypatch):
    monkeypatch.setattr(rr,'OUT',tmp_path)
    monkeypatch.setattr(rr,'SITES',{'location_6':{}})
    tasks={task:{family:{'ap':.5,'ap_ci':[.4,.6]} for family in
        ('climatology','xgboost','random_forest','temporal_cnn')} for task in ('nowcast','lead_time')}
    metrics={'per_site':{'location_6':{'tasks':tasks}},'shift_crosscheck':{'per_site':{}}}
    old={'per_site':{'location_6':{'tasks':{task:{family:{'ap':.4} for family in
        ('climatology','xgboost','random_forest','temporal_cnn')} for task in tasks}}}}
    rr.report(metrics,{'approved':rr.APPROVED,'dropped':[]},old)
    text=(tmp_path/'report_retrain.md').read_text()
    assert 'comparison-run AP' in text
    assert 'different evaluation cohorts and chronological fold boundaries' in text
    assert 'not intervals for a paired schema effect' in text
    assert 'does not establish the precise upstream product/version' in text
    assert '| 0.4000 | +0.1000 |' in text
    assert 'metrics_retrain.partial.json' in text
    assert '## Archive evaluation and skill cost' not in text
