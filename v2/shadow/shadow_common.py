"""Shared, side-effect-free contracts for the private shadow harness."""
from __future__ import annotations
import hashlib, json, math
from datetime import timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SHADOW = Path(__file__).resolve().parent
WINDOW_HOURS = 24
FEATURES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "surface_pressure", "vapour_pressure_deficit", "wind_speed_10m",
    "wind_speed_100m", "wind_gusts_10m", "soil_temperature_0_to_7cm",
    "weather_code", "cloud_cover_low", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
ENGINEERED_DIAGNOSTICS = ["dewpoint_depression", "cooling_rate_6h", "cooling_rate_12h", "previous_night_low"]
SITES = {
    "location_6": {"airport":"KMAE", "iem_station":"MAE", "name":"Madera", "latitude":36.989159, "longitude":-120.111030},
    "location_7": {"airport":"KFAT", "iem_station":"FAT", "name":"Fresno", "latitude":36.777045, "longitude":-119.716862},
    "location_8": {"airport":"KVIS", "iem_station":"VIS", "name":"Visalia", "latitude":36.322528, "longitude":-119.394712},
    "location_9": {"airport":"KHJO", "iem_station":"HJO", "name":"Hanford", "latitude":36.313201, "longitude":-119.626015},
    "location_10":{"airport":"KBFL", "iem_station":"BFL", "name":"Bakersfield", "latitude":35.328506, "longitude":-118.997813},
}
LOG_FIELDS = ["site","task","issue_time_utc","target_window_utc","model","prob","feed_source","feature_end_time_utc","costgrid_decisions","created_utc"]

def utc_iso(value: Any) -> str:
    t=pd.Timestamp(value)
    if t.tzinfo is None: t=t.tz_localize("UTC")
    else: t=t.tz_convert("UTC")
    return t.isoformat().replace("+00:00","Z")

def build_log_record(*,site:str,task:str,issue_time_utc:Any,target_window_utc:dict,model:str,prob:float,
                     feed_source:str,feature_end_time_utc:Any,costgrid_decisions:dict,created_utc:Any|None=None,
                     faithful:bool|None=None,nonfaithful_features:list[str]|None=None)->dict:
    if site not in SITES: raise ValueError(f"unknown site: {site}")
    if task not in {"nowcast","lead_time"}: raise ValueError("task must be nowcast or lead_time")
    p=float(prob)
    if not math.isfinite(p) or not 0 <= p <= 1: raise ValueError("prob must be finite and in [0,1]")
    issue=pd.Timestamp(issue_time_utc); end=pd.Timestamp(feature_end_time_utc)
    if issue.tzinfo is None: issue=issue.tz_localize("UTC")
    else: issue=issue.tz_convert("UTC")
    if end.tzinfo is None: end=end.tz_localize("UTC")
    else: end=end.tz_convert("UTC")
    if end > issue: raise AssertionError("feature_end_time_utc must be <= issue_time_utc")
    rec={"site":site,"task":task,"issue_time_utc":utc_iso(issue),"target_window_utc":target_window_utc,
         "model":model,"prob":p,"feed_source":str(feed_source),"feature_end_time_utc":utc_iso(end),
         "costgrid_decisions":costgrid_decisions,"created_utc":utc_iso(created_utc or pd.Timestamp.now(tz="UTC"))}
    if faithful is not None: rec["faithful"]=bool(faithful)
    if nonfaithful_features is not None: rec["nonfaithful_features"]=list(nonfaithful_features)
    return rec

def apply_calibrator(calibrator:Any, probabilities:Any)->np.ndarray:
    p=np.asarray(probabilities,dtype=float).reshape(-1)
    if calibrator is None: out=p
    elif hasattr(calibrator,"predict_proba"): out=np.asarray(calibrator.predict_proba(p.reshape(-1,1)))[:,1]
    else: out=np.asarray(calibrator.predict(p),dtype=float)
    return np.clip(out,0.0,1.0)

def add_calendar_and_diagnostics(frame:pd.DataFrame)->pd.DataFrame:
    z=frame.copy(); z["time"]=pd.to_datetime(z["time"]); z=z.sort_values("time").drop_duplicates("time",keep="last").reset_index(drop=True)
    t=z["time"]
    z["hour_sin"]=np.sin(2*np.pi*t.dt.hour/24); z["hour_cos"]=np.cos(2*np.pi*t.dt.hour/24)
    z["doy_sin"]=np.sin(2*np.pi*(t.dt.dayofyear-1)/365.25); z["doy_cos"]=np.cos(2*np.pi*(t.dt.dayofyear-1)/365.25)
    z["dewpoint_depression"]=(z["temperature_2m"]-z["dew_point_2m"]).round(2)
    z["cooling_rate_6h"]=((z["temperature_2m"]-z["temperature_2m"].shift(6))/6).round(2)
    z["cooling_rate_12h"]=((z["temperature_2m"]-z["temperature_2m"].shift(12))/12).round(2)
    # Reproduce the legacy diagnostic: 18:00 previous date through 05:00 current date.
    dates=t.dt.normalize(); lows={}
    for day in dates.drop_duplicates():
        mask=(t >= day-pd.Timedelta(hours=6)) & (t < day+pd.Timedelta(hours=6))
        vals=pd.to_numeric(z.loc[mask,"temperature_2m"],errors="coerce")
        lows[day]=float(vals.min()) if vals.notna().any() else np.nan
    z["previous_night_low"]=[lows.get(x,np.nan) for x in dates]
    return z

def tree_view(X:np.ndarray)->np.ndarray:
    return np.concatenate([X[:,-1,:],X.mean(1),X.std(1),X.min(1),X.max(1)],axis=1).astype(np.float32)

def raw_hourly_to_frame(payload:dict)->pd.DataFrame:
    if "hourly" not in payload or "time" not in payload["hourly"]: raise ValueError("Open-Meteo payload lacks hourly.time")
    z=pd.DataFrame(payload["hourly"])
    # Request UTC and convert with the IANA zone per timestamp. Open-Meteo's historical local-time
    # response can expose one present-day fixed offset across winter history, so it is not trusted.
    if str(payload.get("timezone","")).upper() in {"GMT","UTC","ETC/UTC"}:
        z["time"]=pd.to_datetime(z["time"],utc=True,errors="raise").dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
    else:z["time"]=pd.to_datetime(z["time"],errors="raise")
    for c in FEATURES[:12]:
        if c not in z: z[c]=np.nan
        z[c]=pd.to_numeric(z[c],errors="coerce")
    return add_calendar_and_diagnostics(z)

def window_ending(frame:pd.DataFrame, issue_local:Any)->tuple[np.ndarray,pd.Timestamp]:
    issue=pd.Timestamp(issue_local).tz_localize(None); z=frame.copy(); z["time"]=pd.to_datetime(z.time).dt.tz_localize(None)
    expected=pd.date_range(issue-pd.Timedelta(hours=WINDOW_HOURS-1),issue,freq="h")
    w=z.set_index("time").reindex(expected)
    if len(w)!=WINDOW_HOURS or not w.index.equals(expected): raise AssertionError("window must contain exactly 24 hourly slots")
    return w[FEATURES].to_numpy(dtype=np.float32)[None,:,:], expected[-1]

def sha256_bytes(data:bytes)->str: return hashlib.sha256(data).hexdigest()
