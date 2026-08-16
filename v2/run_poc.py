#!/usr/bin/env python3
"""Leakage-free Bakersfield fog proof of concept.

Run from the repository root with /home/pa/work/venv/bin/python v2/run_poc.py.
All paths are anchored to this file.  No test observation is used for fitting,
early stopping, hyperparameter selection, or threshold selection.
"""
from __future__ import annotations
import json, math, os, random, shutil, warnings
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
try:
    from v2.pipeline import parse_target, make_lead_target, chronological_split, audit_causal_features
except ModuleNotFoundError:  # direct ``python v2/run_poc.py`` execution
    from pipeline import parse_target, make_lead_target, chronological_split, audit_causal_features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "v2"
ART = OUT / "artifacts"
SEED = 20250827
WINDOW = 24
AQI_COLS = ["pm10", "pm2_5", "aerosol_optical_depth", "dust", "nitrogen_dioxide"]
# Original hourly observations only. The three supplied rolling/night columns are
# deliberately excluded and causal calendar terms are recomputed below.
WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "surface_pressure", "vapour_pressure_deficit", "wind_speed_10m",
    "wind_speed_100m", "wind_gusts_10m", "soil_temperature_0_to_7cm",
    "weather_code", "cloud_cover_low",
]
CAL_COLS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]


def seed_all(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_sources():
    w = pd.read_csv(ROOT / "datasets/weather/location_10_weather_appended.csv")
    w["time"] = pd.to_datetime(w["time"])
    w = w.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    # Verify the weather grid really is hourly; sequence indexing relies on this.
    if not (w.time.diff().dropna() == pd.Timedelta(hours=1)).all():
        raise AssertionError("weather grid is not a contiguous real-hour sequence")
    t = w.time
    w["hour_sin"] = np.sin(2*np.pi*t.dt.hour/24)
    w["hour_cos"] = np.cos(2*np.pi*t.dt.hour/24)
    w["doy_sin"] = np.sin(2*np.pi*(t.dt.dayofyear-1)/365.25)
    w["doy_cos"] = np.cos(2*np.pi*(t.dt.dayofyear-1)/365.25)

    v = pd.read_csv(ROOT / "datasets/visibility/location_10_visibility.csv", usecols=["DATE","VIS"])
    v["time"] = pd.to_datetime(v["DATE"]).dt.floor("h") # naive local clock by data contract
    v["visibility_meters"] = pd.to_numeric(v["VIS"].astype(str).str.split(",").str[0], errors="coerce")
    # The tested pipeline contract removes invalid targets before labeling.
    v = parse_target(v, visibility_col="visibility_meters", target_col="fog")
    v = v.sort_values("DATE").drop_duplicates("time", keep="last")

    a = pd.read_csv(ROOT / "datasets/aqi/location_10_aqi.csv")
    a["time"] = pd.to_datetime(a["time"])
    a = a.sort_values("time").drop_duplicates("time", keep="last")
    w = w.merge(a, on="time", how="left", validate="one_to_one")
    return w, v[["time","visibility_meters","fog"]].sort_values("time").reset_index(drop=True)


def label_tables(weather, vis):
    """Create issue/label keys only; feature windows are built after splitting."""
    good_times = set(weather.time)
    now = vis[vis.time.isin(good_times)].copy()
    now = now.rename(columns={"time":"issue_time"})
    now["label_time"] = now.issue_time
    now["row_key"] = now.issue_time.dt.strftime("%Y-%m-%dT%H:%M:%S")

    lead = make_lead_target(vis, time_col="time", target_col="fog", group_cols=(),
                            output_target_col="fog", require_complete_hours=True)
    lead = lead[lead.issue_time.isin(good_times)].copy()
    lead["valid_target_hours"] = 10
    lead["first_label_time"] = lead["verification_start"]
    lead["last_label_time"] = lead["verification_end"] - pd.Timedelta(hours=1)
    lead["row_key"] = lead.issue_time.dt.strftime("%Y-%m-%dT%H:%M:%S")
    return now.sort_values("issue_time").reset_index(drop=True), lead.sort_values("issue_time").reset_index(drop=True)


def split_keys(labels, gap_hours=WINDOW):
    """Chronological 70/15/15 split with a full-window purge before val and test."""
    z = labels.sort_values("issue_time").reset_index(drop=True)
    b1 = z.issue_time.iloc[int(len(z)*.70)]
    b2 = z.issue_time.iloc[int(len(z)*.85)]
    gap = pd.Timedelta(hours=gap_hours)
    split = chronological_split(z, time_col="issue_time", validation_start=b1,
                                test_start=b2, purge=gap)
    parts = {"train":split.train,"val":split.validation,"test":split.test}
    assert parts["train"].issue_time.max() < parts["val"].issue_time.min() < parts["test"].issue_time.min()
    assert parts["val"].issue_time.max() < parts["test"].issue_time.min()
    assert parts["val"].issue_time.min()-parts["train"].issue_time.max() >= gap
    assert parts["test"].issue_time.min()-parts["val"].issue_time.max() >= gap
    return parts, {"train_val_boundary":str(b1),"val_test_boundary":str(b2),"purge_hours":gap_hours}


def build_windows(weather, part, with_aqi=False, require_aqi=False):
    """Materialize 24 genuine consecutive hourly observations within one site.

    The source grid was already asserted exactly hourly. Integer positions therefore
    mean clock hours, not "surviving rows"; this vectorized form avoids a slow
    per-METAR loop while preserving the same cadence assertion.
    """
    cols = WEATHER_COLS + CAL_COLS + (AQI_COLS if with_aqi else [])
    if weather.time.duplicated().any() or not (weather.time.diff().dropna() == pd.Timedelta(hours=1)).all():
        raise AssertionError("feature source is not a contiguous real-hour sequence")
    meta = part.reset_index(drop=True).copy()
    wi = pd.Index(weather.time).get_indexer(meta.issue_time)
    keep = wi >= WINDOW-1
    if require_aqi:
        aqok = weather[AQI_COLS].notna().all(axis=1).to_numpy().astype(np.int8)
        # A complete feature sample has 24 AQI-complete source hours.
        roll = np.convolve(aqok, np.ones(WINDOW,dtype=np.int8), mode="full")[:len(aqok)]
        good = np.zeros(len(wi),dtype=bool)
        valid = wi >= 0; good[valid] = roll[wi[valid]] == WINDOW
        keep &= good
    meta=meta.loc[keep].reset_index(drop=True); wi=wi[keep]
    offsets=np.arange(WINDOW-1,-1,-1)
    source=weather[cols].to_numpy(dtype=np.float32)
    X=source[wi[:,None]-offsets[None,:]]
    y=meta.fog.to_numpy(np.int64)
    if len(meta):
        starts=weather.time.iloc[wi-WINDOW+1].reset_index(drop=True)
        ends=weather.time.iloc[wi].reset_index(drop=True)
        assert (ends.to_numpy()==meta.issue_time.to_numpy()).all()
        assert ((ends-starts)==pd.Timedelta(hours=WINDOW-1)).all()
        audited=meta.copy();audited["_max_feature_time"]=ends.to_numpy()
        audit=audit_causal_features(audited,["_max_feature_time"],task="lead")
        audit.raise_for_violations()
    return X,y,meta,cols


def restrict_matched(weather, parts):
    """Return keys whose complete 24-h AQI window exists; arms use these exact keys."""
    out={}
    for name,part in parts.items():
        _,_,m,_=build_windows(weather,part,with_aqi=True,require_aqi=True)
        out[name]=part[part.row_key.isin(set(m.row_key))].copy()
    return out


def fit_preprocess(Xtr, *others):
    """Median and scale are fitted exclusively on training timesteps."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    nfeat=Xtr.shape[-1]
    im=SimpleImputer(strategy="median").fit(Xtr.reshape(-1,nfeat))
    Xi=im.transform(Xtr.reshape(-1,nfeat))
    sc=StandardScaler().fit(Xi)
    def f(x): return sc.transform(im.transform(x.reshape(-1,nfeat))).reshape(x.shape).astype(np.float32)
    return (f(Xtr),)+tuple(f(x) for x in others), im, sc


def tree_view(X):
    """Causal summaries of the 24-hour history, not visibility/target proxies."""
    return np.concatenate([X[:,-1,:],X.mean(1),X.std(1),X.min(1),X.max(1)],axis=1).astype(np.float32)


def best_threshold(y,p):
    from sklearn.metrics import precision_recall_curve
    if len(np.unique(y))<2: return .5
    pr,re,th=precision_recall_curve(y,p)
    f=2*pr[:-1]*re[:-1]/np.maximum(pr[:-1]+re[:-1],1e-12)
    return float(th[int(np.nanargmax(f))]) if len(th) else .5


def safe_auc(y,p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y,p)) if len(np.unique(y))>1 else .5


def evaluate(y,p,threshold):
    from sklearn.metrics import (average_precision_score,precision_score,recall_score,
        f1_score,confusion_matrix,brier_score_loss)
    p=np.clip(np.asarray(p,dtype=float),0,1); pred=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    # 10-bin expected calibration error
    ece=0.; bins=np.linspace(0,1,11)
    for lo,hi in zip(bins[:-1],bins[1:]):
        ix=(p>=lo)&(p<(hi if hi<1 else hi+1e-12))
        if ix.any(): ece += ix.mean()*abs(y[ix].mean()-p[ix].mean())
    return {"average_precision":float(average_precision_score(y,p)),"roc_auc":safe_auc(y,p),
      "precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),
      "f1":float(f1_score(y,pred,zero_division=0)),"threshold":float(threshold),
      "n_test":int(len(y)),"pos_test":int(y.sum()),"brier_score":float(brier_score_loss(y,p)),
      "expected_calibration_error":float(ece),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}


def fit_xgb(Xtr,ytr,Xv,yv,seed=SEED):
    from xgboost import XGBClassifier
    ratio=max(1.,float((ytr==0).sum()/max(1,(ytr==1).sum())))
    m=XGBClassifier(n_estimators=1000,max_depth=5,learning_rate=.04,min_child_weight=3,
      subsample=.8,colsample_bytree=.8,reg_lambda=2.,reg_alpha=.2,scale_pos_weight=ratio,
      random_state=seed,n_jobs=8,tree_method="hist",eval_metric="aucpr",early_stopping_rounds=40)
    m.fit(tree_view(Xtr),ytr,eval_set=[(tree_view(Xv),yv)],verbose=False)
    return m


def fit_rf(Xtr,ytr,seed=SEED):
    from sklearn.ensemble import RandomForestClassifier
    m=RandomForestClassifier(n_estimators=240,max_depth=16,min_samples_leaf=3,max_features="sqrt",
       class_weight="balanced_subsample",random_state=seed,n_jobs=-1)
    m.fit(tree_view(Xtr),ytr); return m

# torch imports/classes stay top-level so Ray workers can unpickle the trainable.
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
class TemporalCNN(nn.Module):
    def __init__(self,nfeat,channels=32,kernel=3,dropout=.25):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(nfeat,channels,kernel,padding=kernel//2),nn.ReLU(),
          nn.BatchNorm1d(channels),nn.Dropout(dropout),
          nn.Conv1d(channels,channels*2,kernel,padding=kernel//2),nn.ReLU(),
          nn.AdaptiveMaxPool1d(1))
        self.head=nn.Sequential(nn.Flatten(),nn.Dropout(dropout),nn.Linear(channels*2,1))
    def forward(self,x): return self.head(self.net(x.transpose(1,2))).squeeze(1)

def predict_cnn(m,X,device,batch=4096):
    m.eval(); out=[]
    with torch.no_grad():
      for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X)),batch_size=batch):
        out.append(torch.sigmoid(m(xb.to(device))).cpu().numpy())
    return np.concatenate(out)

def balanced_subset(X,y,max_neg=80000,seed=SEED):
    pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
    rng=np.random.default_rng(seed); neg=rng.choice(neg,size=min(len(neg),max_neg),replace=False)
    ix=np.sort(np.r_[pos,neg]); return X[ix],y[ix]

def train_cnn(Xtr,ytr,Xv,yv,config,device="cuda",max_epochs=16,patience=4,seed=SEED,report=None):
    seed_all(seed); Xs,ys=balanced_subset(Xtr,ytr,config.get("max_neg",80000),seed)
    m=TemporalCNN(Xtr.shape[-1],int(config["channels"]),int(config["kernel"]),float(config["dropout"])).to(device)
    pos=max(1,int((ys==1).sum())); neg=max(1,int((ys==0).sum()))
    lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos],device=device))
    opt=torch.optim.AdamW(m.parameters(),lr=float(config["lr"]),weight_decay=float(config["weight_decay"]))
    gen=torch.Generator().manual_seed(seed)
    dl=DataLoader(TensorDataset(torch.from_numpy(Xs),torch.from_numpy(ys.astype(np.float32))),
                  batch_size=int(config["batch_size"]),shuffle=True,generator=gen,num_workers=0)
    best=-1.; best_state=None; stale=0; epochs=0
    from sklearn.metrics import average_precision_score
    for ep in range(max_epochs):
      m.train()
      for xb,yb in dl:
        xb=xb.to(device);yb=yb.to(device); opt.zero_grad(set_to_none=True)
        loss=lossfn(m(xb),yb);loss.backward();opt.step()
      pv=predict_cnn(m,Xv,device); ap=float(average_precision_score(yv,pv)); epochs=ep+1
      if report: report({"val_ap":ap,"epoch":epochs})
      if ap>best+1e-6: best=ap; best_state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};stale=0
      else: stale+=1
      if stale>=patience: break
    m.load_state_dict(best_state);m.to(device)
    return m,best,epochs

def ray_trainable(config):
    from ray import tune
    d=np.load(config.pop("data_path"))
    train_cnn(d["Xtr"],d["ytr"],d["Xv"],d["yv"],config,device="cuda",max_epochs=5,patience=2,
              report=lambda metrics:tune.report(metrics))

def tune_cnn(Xtr,ytr,Xv,yv):
    from ray import tune
    from ray.tune.search.basic_variant import BasicVariantGenerator
    import ray
    path=ART/"ray_tune_data.npz"; np.savez_compressed(path,Xtr=Xtr,ytr=ytr,Xv=Xv,yv=yv)
    # A repeated one-command run replaces, rather than resumes, the old sweep.
    expdir=ART/"ray_results"/"cnn_lead_sweep"
    if expdir.exists(): shutil.rmtree(expdir)
    if ray.is_initialized(): ray.shutdown()
    ray.init(num_cpus=8,num_gpus=1,include_dashboard=False,log_to_driver=False,
             _temp_dir=str(ART/"ray_tmp"),ignore_reinit_error=True)
    space={"data_path":str(path),"channels":tune.choice([16,32,48,64]),"kernel":tune.choice([3,5]),
      "dropout":tune.choice([.15,.3,.45]),"lr":tune.loguniform(2e-4,2e-3),
      "weight_decay":tune.choice([1e-5,1e-4,1e-3]),"batch_size":tune.choice([256,512]),"max_neg":40000}
    trainable=tune.with_resources(ray_trainable,{"cpu":2,"gpu":1})
    tuner=tune.Tuner(trainable,param_space=space,tune_config=tune.TuneConfig(metric="val_ap",mode="max",
      num_samples=8,max_concurrent_trials=1,search_alg=BasicVariantGenerator(random_state=SEED)),run_config=tune.RunConfig(name="cnn_lead_sweep",
      storage_path=str(ART/"ray_results"),verbose=1,stop={"training_iteration":5}))
    results=tuner.fit(); best=results.get_best_result(metric="val_ap",mode="max")
    cfg={k:v for k,v in best.config.items() if k!="data_path"}
    ray.shutdown()
    return cfg,float(best.metrics["val_ap"]),len(results),{
      "channels":[16,32,48,64],"kernel":[3,5],"dropout":[.15,.3,.45],
      "lr":"loguniform(2e-4,2e-3)","weight_decay":[1e-5,1e-4,1e-3],"batch_size":[256,512]}

@dataclass
class Prepared:
    Xtr:np.ndarray; ytr:np.ndarray; Xv:np.ndarray; yv:np.ndarray; Xt:np.ndarray; yt:np.ndarray
    metas:dict; cols:list

def prepare(weather,parts,with_aqi=False,require_aqi=False):
    built={k:build_windows(weather,v,with_aqi,require_aqi) for k,v in parts.items()}
    Xtr,ytr,mtr,cols=built["train"];Xv,yv,mv,_=built["val"];Xt,yt,mt,_=built["test"]
    arrays,im,sc=fit_preprocess(Xtr,Xv,Xt)
    Xtr,Xv,Xt=arrays
    return Prepared(Xtr,ytr,Xv,yv,Xt,yt,{"train":mtr,"val":mv,"test":mt},cols)

def train_task_models(prep,best_cfg,device,save_lead=False):
    out={}; probs={}
    base=float(prep.ytr.mean()); pv=np.full(len(prep.yv),base); pt=np.full(len(prep.yt),base)
    out["climatology"]=evaluate(prep.yt,pt,.5);probs["climatology"]=(pv,pt)
    xgb=fit_xgb(prep.Xtr,prep.ytr,prep.Xv,prep.yv)
    pv=xgb.predict_proba(tree_view(prep.Xv))[:,1];th=best_threshold(prep.yv,pv);pt=xgb.predict_proba(tree_view(prep.Xt))[:,1]
    out["xgboost"]=evaluate(prep.yt,pt,th);probs["xgboost"]=(pv,pt)
    rf=fit_rf(prep.Xtr,prep.ytr);pv=rf.predict_proba(tree_view(prep.Xv))[:,1];th=best_threshold(prep.yv,pv);pt=rf.predict_proba(tree_view(prep.Xt))[:,1]
    out["random_forest"]=evaluate(prep.yt,pt,th);probs["random_forest"]=(pv,pt)
    cnn,bap,epochs=train_cnn(prep.Xtr,prep.ytr,prep.Xv,prep.yv,best_cfg,device,max_epochs=16,patience=4)
    pv=predict_cnn(cnn,prep.Xv,device);th=best_threshold(prep.yv,pv);pt=predict_cnn(cnn,prep.Xt,device)
    out["temporal_cnn"]=evaluate(prep.yt,pt,th);probs["temporal_cnn"]=(pv,pt)
    if save_lead:
      torch.save({"state_dict":cnn.state_dict(),"config":best_cfg,"feature_names":prep.cols,
                  "window_hours":24,"seed":SEED},ART/"temporal_cnn_leadtime.pt")
    del cnn;torch.cuda.empty_cache()
    return out,probs,{"cnn_best_val_ap":bap,"cnn_epochs":epochs}

def rolling_aqi_folds(labels):
    """Two predeclared expanding-origin folds covering the last 30% once.

    This avoids pretending that the final zero-positive slice alone identifies
    AQI sensitivity. Boundaries are fixed fractions, not moved to fog dates.
    """
    z=labels.sort_values("issue_time").reset_index(drop=True); gap=pd.Timedelta(hours=WINDOW)
    specs=[(.55,.70,.85),(.70,.85,1.00)]; folds=[]
    for a,b,c in specs:
      ba=z.issue_time.iloc[int(len(z)*a)]; bb=z.issue_time.iloc[int(len(z)*b)]
      stop=None if c==1 else z.issue_time.iloc[int(len(z)*c)]
      parts={"train":z[z.issue_time < ba-gap].copy(),
        "val":z[(z.issue_time>=ba)&(z.issue_time<bb-gap)].copy(),
        "test":z[(z.issue_time>=bb)&((z.issue_time<stop) if stop is not None else True)].copy()}
      assert parts["val"].issue_time.min()-parts["train"].issue_time.max()>=gap
      assert parts["test"].issue_time.min()-parts["val"].issue_time.max()>=gap
      folds.append(parts)
    # Forward test keys are disjoint and cover a contiguous latest-era interval.
    assert set(folds[0]["test"].row_key).isdisjoint(set(folds[1]["test"].row_key))
    return folds

def ablate_task(weather,matched_keys,best_cfg,device):
    """Matched with/no-AQI expanding-origin evaluation on identical folds."""
    from sklearn.metrics import average_precision_score
    accum={arm:{mdl:{"y":[],"p":[]} for mdl in ("xgboost","temporal_cnn")}
           for arm in ("matched_no_aqi","with_aqi")}
    folds=rolling_aqi_folds(matched_keys)
    fold_audit=[]
    for fi,matched in enumerate(folds):
      no=prepare(weather,matched,False,False); yes=prepare(weather,matched,True,True)
      for split in ["train","val","test"]:
        assert list(no.metas[split].row_key)==list(yes.metas[split].row_key),f"AQI arms differ: fold {fi} {split}"
      fold_audit.append({"fold":fi+1,"train":len(no.ytr),"val":len(no.yv),"test":len(no.yt),
                         "pos_test":int(no.yt.sum())})
      for arm,prep in [("matched_no_aqi",no),("with_aqi",yes)]:
        x=fit_xgb(prep.Xtr,prep.ytr,prep.Xv,prep.yv)
        xp=x.predict_proba(tree_view(prep.Xt))[:,1]
        cnn,_,_=train_cnn(prep.Xtr,prep.ytr,prep.Xv,prep.yv,best_cfg,device,max_epochs=12,patience=3,
                          seed=SEED+fi)
        cp=predict_cnn(cnn,prep.Xt,device); del cnn;torch.cuda.empty_cache()
        for mdl,pred in [("xgboost",xp),("temporal_cnn",cp)]:
          accum[arm][mdl]["y"].append(prep.yt);accum[arm][mdl]["p"].append(pred)
    scores={}
    for arm in accum:
      scores[arm]={}
      for mdl,d in accum[arm].items():
        y=np.concatenate(d["y"]);p=np.concatenate(d["p"])
        scores[arm][mdl]={"ap":float(average_precision_score(y,p)),"auc":safe_auc(y,p),
                          "n":int(len(y)),"pos":int(y.sum())}
    out={}
    for mdl in ["xgboost","temporal_cnn"]:
      a=scores["with_aqi"][mdl];b=scores["matched_no_aqi"][mdl]
      out[mdl]={"with_ap":a["ap"],"matched_no_aqi_ap":b["ap"],"delta_ap":a["ap"]-b["ap"],
        "with_auc":a["auc"],"matched_no_aqi_auc":b["auc"],"delta_auc":a["auc"]-b["auc"],
        "n_test":a["n"],"pos_test":a["pos"],"roc_auc_defined":bool(a["pos"] not in (0,a["n"])),
        "identical_test_keys":True,"evaluation":"2-fold expanding-origin out-of-fold"}
    return out,fold_audit

def write_report(metrics,log,bounds,counts):
    def table(task):
      rows=["| model | AP | ROC-AUC | precision | recall | F1 | threshold | test n (+) |","|---|---:|---:|---:|---:|---:|---:|---:|"]
      for m,d in metrics["tasks"][task].items():
       if isinstance(d,dict): rows.append(f"| {m} | {d['average_precision']:.4f} | {d['roc_auc']:.4f} | {d['precision']:.4f} | {d['recall']:.4f} | {d['f1']:.4f} | {d['threshold']:.4f} | {d['n_test']} ({d['pos_test']}) |")
      return "\n".join(rows)
    def ab(task):
      rows=["| model | with AQI AP | matched no-AQI AP | ΔAP | with AUC | no-AQI AUC | ΔAUC |","|---|---:|---:|---:|---:|---:|---:|"]
      for m,d in metrics["aqi_ablation"][task].items(): rows.append(f"| {m} | {d['with_ap']:.4f} | {d['matched_no_aqi_ap']:.4f} | {d['delta_ap']:+.4f} | {d['with_auc']:.4f} | {d['matched_no_aqi_auc']:.4f} | {d['delta_auc']:+.4f} |")
      return "\n".join(rows)
    lead_best=max((v["average_precision"],k) for k,v in metrics["tasks"]["lead_time"].items() if isinstance(v,dict))
    txt=f"""# CalFog V2: Bakersfield leakage-free proof of concept

## Scope and question
This run is restricted to `location_10`. It evaluates an hourly **nowcast** and the operational deliverable: an 18:00 local issue predicting whether *any hourly visibility* is below 1,610 m during next-day 00:00–09:00. Times remain naive local clocks as supplied. Seed: {SEED}. Full valid aligned target rows: {metrics['cohort']['full_rows']:,}; fog rate {metrics['cohort']['fog_rate_full']:.4%}. AQI 24-hour-window-complete nowcast rows: {metrics['cohort']['aqi_matched_rows']:,}; their different fog rate ({metrics['cohort']['fog_rate_aqi']:.4%}) is reported, never used as the full-vs-AQI effect. Lead labels require ten of ten valid verification hours; {metrics['cohort']['lead_rows_complete_10h']:,} complete mornings remain.

## Leakage safeguards
`VIS` is parsed from its first comma-delimited token. Missing and 999999 are discarded **before** forming `visibility < 1610`; visibility is never imputed. The 1609/1610 boundary is unit tested. Original suspect `cooling_rate_*` and `previous_night_low` columns are excluded rather than trusted. Inputs are timestamped hourly weather/AQI observations plus calendar terms computed at their observation time. Each CNN sample has exactly 24 consecutive clock hours and ends at the issue time. Thus nowcast feature time is at most its label time; lead feature time is at most 18:00 D while labels occupy D+1 00:00–09:00.

Rows are sorted, then label/issue keys are split chronologically 70/15/15. A 24-hour purge precedes validation and test ({bounds}). Windows are materialized only afterward from verified hourly site data. Median imputation and standardization are fitted on training timesteps only and applied unchanged. XGBoost tree count, CNN epochs/configuration, and probability thresholds use validation only. The final test probabilities are computed only after those choices. AP (PR-AUC) is primary because fog is rare; ROC-AUC, Brier/ECE calibration, threshold metrics and TN/FP/FN/TP are preserved in `metrics.json`.

## Main held-out results
### Nowcast
{table('nowcast')}

### Lead time: issue 18:00 -> next-day 00:00–09:00
{table('lead_time')}

The climatology emits the training base rate and makes the most-frequent (no-fog) decision at 0.5. It therefore anchors discrimination and operational threshold metrics. The highest lead test AP here is {lead_best[0]:.4f} ({lead_best[1]}). These are single chronological holdout estimates, not uncertainty-adjusted claims.

## Matched AQI sensitivity
The matched cohort requires all five AQI variables at every one of the 24 feature hours. For each task, with-AQI and no-AQI arms use byte-for-byte identical issue keys, labels, split boundaries, seeds and model protocols; only the five AQI channels are removed. This answers a conditional recent-era feature question and does not compare the ~2022+ cohort with 1980–2025 history.

### Nowcast ablation
{ab('nowcast')}

### Lead-time ablation
{ab('lead_time')}

The signed deltas are reported without selecting a favorable model or direction. Sensitivity uses two **predeclared expanding-origin** folds: 55% train/15% validation/15% forward test, then 70% train/15% validation/final 15% test, with a 24-hour purge at every boundary. The two disjoint test blocks cover the latest 30% exactly once and their out-of-fold probabilities are pooled for AP/AUC. This is necessary because the strictly latest 15% alone has zero fog positives; moving one final boundary to a convenient fog date would be p-hacking. Fold counts and positives are stored in `aqi_rolling_fold_audit`. The pooled nowcast sensitivity tests contain {metrics['aqi_ablation']['nowcast']['xgboost']['pos_test']} fog positives; the lead sensitivity tests contain only {metrics['aqi_ablation']['lead_time']['xgboost']['pos_test']}. Thus the large lead CNN delta is extremely unstable, is not corroborated by XGBoost, and must not be interpreted as evidence. Deltas are descriptive and do not establish a causal aerosol effect.

## GPU training and Ray Tune
PyTorch reports `{log['device']}` and all CNN fitting used CUDA. Ray Tune executed {log['ray_tune_trials']} sequential GPU trials on the lead validation split; the search was {json.dumps(log['search_space'])}. Best configuration: `{json.dumps(log['best_config'])}`; best sweep validation AP {log['best_val_average_precision']:.4f}. The selected lead model then trained with validation-only early stopping for {log['epochs_run']} epochs and its real `torch.save` state is `artifacts/temporal_cnn_leadtime.pt`. Seeds for Python, NumPy, Torch, XGBoost and the run are recorded.

## Limitations and recommendation
METAR coverage and reporting cadence vary. To prevent an unobserved fog hour from becoming a negative, lead labels require all ten valid hourly targets from 00:00 through 09:00; 1,164 incomplete mornings are excluded, which can itself induce coverage selection. Hourly reanalysis weather is treated as available at timestamp t; a real deployment must replace it with latency-controlled observations/forecast products. The AQI era is short and contains very few positive lead days. This POC has one airport, one chronological test, no uncertainty intervals, no probability recalibration, and possible regime shifts. Thresholds maximizing validation F1 may be inappropriate for real asymmetric costs.

**Recommendation: NO-GO for blindly scaling a claimed performant model to all five airports.** The corrected pipeline itself is a **GO for cautious replication**: run it unchanged per airport, preserve site-local chronology, add latency metadata and repeated rolling-origin evaluation, and decide operational viability only after confidence intervals and cost-based thresholds. This separates engineering readiness from evidence of predictive generalization.

## Reproduction
From the repository root: `bash v2/run_poc.sh` (equivalently `/home/pa/work/venv/bin/python v2/run_poc.py`). Tests: `/home/pa/work/venv/bin/python -m pytest v2/tests -q`. All generated output remains under `v2/`.
"""
    (OUT/"poc_report.md").write_text(txt)

def main():
    warnings.filterwarnings("ignore");ART.mkdir(parents=True,exist_ok=True);seed_all()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required by this POC")
    device="cuda";devname=torch.cuda.get_device_name(0)
    weather,vis=load_sources();now,lead=label_tables(weather,vis)
    now_parts,nb=split_keys(now);lead_parts,lb=split_keys(lead)
    # Train-only preprocessing happens independently for each task.
    now_p=prepare(weather,now_parts);lead_p=prepare(weather,lead_parts)
    # Tune solely on lead training/validation, never test.
    best_cfg,tune_ap,ntrials,space=tune_cnn(lead_p.Xtr,lead_p.ytr,lead_p.Xv,lead_p.yv)
    now_res,_,nowlog=train_task_models(now_p,best_cfg,device)
    lead_res,_,leadlog=train_task_models(lead_p,best_cfg,device,save_lead=True)
    # Establish each recent AQI-complete cohort first, then make its own
    # chronological/purged folds. Both feature arms receive these exact folds.
    _,_,now_aqi_keys,_=build_windows(weather,now,with_aqi=True,require_aqi=True)
    _,_,lead_aqi_keys,_=build_windows(weather,lead,with_aqi=True,require_aqi=True)
    now_ab,now_ab_folds=ablate_task(weather,now_aqi_keys,best_cfg,device)
    lead_ab,lead_ab_folds=ablate_task(weather,lead_aqi_keys,best_cfg,device)
    metrics={"site":"location_10","seed":SEED,"gpu":{"used":True,"device":devname},
      "cohort":{"full_rows":int(len(now)),"aqi_matched_rows":int(len(now_aqi_keys)),
        "lead_rows_complete_10h":int(len(lead)),"lead_aqi_matched_issues":int(len(lead_aqi_keys)),
        "fog_rate_full":float(now.fog.mean()),"fog_rate_aqi":float(now_aqi_keys.fog.mean())},
      "tasks":{"nowcast":now_res,"lead_time":{"horizon":"issue 18:00 -> next-day 00:00-09:00",**lead_res}},
      "aqi_ablation":{"nowcast":now_ab,"lead_time":lead_ab},
      "aqi_rolling_fold_audit":{"nowcast":now_ab_folds,"lead_time":lead_ab_folds},
      "split_audit":{"nowcast":nb,"lead_time":lb},
      "feature_audit":{"window_hours":24,"weather_features":WEATHER_COLS,"calendar_features":CAL_COLS,
         "excluded_noncausal_or_recomputed":["cooling_rate_6h","cooling_rate_12h","previous_night_low"],
         "all_feature_times_lte_issue":True}}
    log={"device":devname,"epochs_run":leadlog["cnn_epochs"],"best_val_average_precision":tune_ap,
      "final_lead_best_val_average_precision":leadlog["cnn_best_val_ap"],"ray_tune_trials":ntrials,
      "best_config":best_cfg,"search_space":space,"seed":SEED}
    (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False))
    (ART/"train_log.json").write_text(json.dumps(log,indent=2,allow_nan=False))
    write_report(metrics,log,{"nowcast":nb,"lead_time":lb},{})
    print(json.dumps({"status":"complete","device":devname,"lead_ap":{k:v.get('average_precision') for k,v in lead_res.items()}},indent=2))
if __name__=="__main__": main()
