from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from .data import WEATHER_COLS, CAL_COLS, AQI_COLS, WINDOW
from .labels import audit_causal_features
from .visibility import FOG_MAX_CATEGORY

def build_windows(weather, part, with_aqi=False, require_aqi=False, *, weather_cols: list[str] = WEATHER_COLS):
    """Materialize 24 genuine consecutive hourly observations within one site.

    The source grid was already asserted exactly hourly. Integer positions therefore
    mean clock hours, not "surviving rows"; this vectorized form avoids a slow
    per-METAR loop while preserving the same cadence assertion.
    """
    cols = list(weather_cols) + CAL_COLS + (AQI_COLS if with_aqi else [])
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
    c=meta.category.to_numpy(np.int64)
    if len(meta):
        starts=weather.time.iloc[wi-WINDOW+1].reset_index(drop=True)
        ends=weather.time.iloc[wi].reset_index(drop=True)
        assert (ends.to_numpy()==meta.issue_time.to_numpy()).all()
        assert ((ends-starts)==pd.Timedelta(hours=WINDOW-1)).all()
        audited=meta.copy();audited["_max_feature_time"]=ends.to_numpy()
        audit=audit_causal_features(audited,["_max_feature_time"],task="lead")
        audit.raise_for_violations()
    return X,c,meta,cols

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

@dataclass
class Prepared:
    Xtr:np.ndarray; ctr:np.ndarray; Xv:np.ndarray; cv:np.ndarray; Xt:np.ndarray; ct:np.ndarray
    metas:dict; cols:list

    @property
    def ytr(self):
        return (self.ctr <= FOG_MAX_CATEGORY).astype(int)

    @property
    def yv(self):
        return (self.cv <= FOG_MAX_CATEGORY).astype(int)

    @property
    def yt(self):
        return (self.ct <= FOG_MAX_CATEGORY).astype(int)

def prepare(weather,parts,with_aqi=False,require_aqi=False, *, weather_cols: list[str] = WEATHER_COLS):
    built={k:build_windows(weather,v,with_aqi,require_aqi,weather_cols=weather_cols) for k,v in parts.items()}
    Xtr,ctr,mtr,cols=built["train"];Xv,cv,mv,_=built["val"];Xt,ct,mt,_=built["test"]
    arrays,im,sc=fit_preprocess(Xtr,Xv,Xt)
    Xtr,Xv,Xt=arrays
    return Prepared(Xtr,ctr,Xv,cv,Xt,ct,{"train":mtr,"val":mv,"test":mt},cols)

def tree_feature_names(cols: list[str]) -> list[str]:
    return [f"{stat}_{col}" for stat in ("latest", "mean", "std", "min", "max") for col in cols]
