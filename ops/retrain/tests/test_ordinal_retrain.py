"""Small, offline retrain PMF contract checks; no training or GPU access."""
import json
import numpy as np
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
    monkeypatch.setattr(rr,'label_tables',lambda *a:(None,None))
    monkeypatch.setattr(rr,'predict_proxy',lambda *a:(np.array([4,17]),{'xgboost':pmf},None))
    archive={task:{'xgboost':{'ap':1.}} for task in ('nowcast','lead_time')}
    result=rr.proxy_crosscheck('location_6',None,None,{'nowcast':{},'lead_time':{}},archive,[])
    for task in ('nowcast','lead_time'):
        metrics=result['tasks'][task]['xgboost']
        assert metrics['hf_proxy_ap']==1
        assert metrics['hf_proxy_rps']==0
        assert metrics['pos_proxy']==1
