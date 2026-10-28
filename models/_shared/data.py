from __future__ import annotations

from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
from .visibility import FOG_MAX_CATEGORY, N_CATEGORIES

ROOT = Path(__file__).resolve().parents[2]
SEED = 20250901
N_BOOT = 1000
SITES = {"location_6":"Madera", "location_7":"Fresno", "location_8":"Visalia", "location_9":"Hanford", "location_10":"Bakersfield"}
WINDOW = 24
AQI_COLS = ["pm10", "pm2_5", "aerosol_optical_depth", "dust", "nitrogen_dioxide"]
MEASURED_AQ_COLS = ["nox_ppb", "pm25_ugm3"]
AQ_SETS: dict[str, list[str]] = {"nox": ["nox_ppb"], "pm25": ["pm25_ugm3"], "cams": AQI_COLS}
AQ_MIN_HOURS = 20
PRIMARY_FAMILIES = ("logistic_regression", "xgboost", "ebm")
# Original hourly observations only. The three supplied rolling/night columns are
# deliberately excluded and causal calendar terms are recomputed below.
WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "surface_pressure", "vapour_pressure_deficit", "wind_speed_10m",
    "wind_speed_100m", "wind_gusts_10m", "soil_temperature_0_to_7cm",
    "weather_code", "cloud_cover_low",
]
CAL_COLS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]


def utc_to_pst_naive(values):
    """Convert UTC observations to the legacy fixed UTC-8 feature clock."""
    parsed = pd.to_datetime(values, utc=True, errors="raise")
    if isinstance(parsed, pd.DatetimeIndex):
        return parsed.tz_convert("Etc/GMT+8").tz_localize(None)
    return parsed.dt.tz_convert("Etc/GMT+8").dt.tz_localize(None)

def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"missing required column(s): {missing}")

def _as_timestamps(values: pd.Series, column: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"column {column!r} is not valid datetime data") from exc
    if parsed.isna().any():
        raise ValueError(f"column {column!r} contains missing timestamps")
    return parsed


def load_sources(site, *, weather_cols=WEATHER_COLS):
    candidates=list((ROOT/'datasets'/'weather').glob(f'{site}_weather*.csv'))
    appended=[p for p in candidates if p.name.endswith('_appended.csv')]
    chosen=appended[0] if len(appended)==1 else (candidates[0] if len(candidates)==1 else None)
    if chosen is None: raise RuntimeError(f'{site}: could not select one weather file from {candidates}')
    w=pd.read_csv(chosen); w['time']=pd.to_datetime(w.time)
    w=w.sort_values('time').drop_duplicates('time',keep='last').reset_index(drop=True)
    if not (w.time.diff().dropna()==pd.Timedelta(hours=1)).all(): raise AssertionError(f'{site}: non-hourly weather')
    t=w.time; w['hour_sin']=np.sin(2*np.pi*t.dt.hour/24); w['hour_cos']=np.cos(2*np.pi*t.dt.hour/24)
    w['doy_sin']=np.sin(2*np.pi*(t.dt.dayofyear-1)/365.25); w['doy_cos']=np.cos(2*np.pi*(t.dt.dayofyear-1)/365.25)
    a=pd.read_csv(ROOT/'datasets'/'aqi'/f'{site}_aqi.csv'); a['time']=pd.to_datetime(a.time)
    a=a.sort_values('time').drop_duplicates('time',keep='last'); w=w.merge(a,on='time',how='left',validate='one_to_one')
    measured = ROOT/'datasets'/'aq_measured'/f'{site}_aq_measured.csv'
    if measured.exists():
        aq = pd.read_csv(measured, parse_dates=['time'],
                         dtype={'nox_monitor': 'string', 'pm25_monitor': 'string'})
        w = w.merge(aq, on='time', how='left', validate='one_to_one')
    else:
        for col in MEASURED_AQ_COLS:
            w[col] = np.nan
        for col in ('nox_monitor', 'pm25_monitor'):
            w[col] = pd.Series(pd.NA, index=w.index, dtype='string')
    for monitor in sorted(w.nox_monitor.dropna().unique()):
        w[f'nox_mon_{monitor}'] = w.nox_monitor.eq(monitor).fillna(False).astype(float)
    v=pd.read_csv(ROOT/'datasets'/'visibility'/f'{site}_visibility_hourly.csv',
                  usecols=['time','cat_lo','cat_hi','category','fog'],
                  dtype={'cat_lo':'Int64','cat_hi':'Int64','category':'Int64','fog':'boolean'},
                  parse_dates=['time'])
    v=v.dropna(subset=['category']).copy()
    if not v.category.between(0,N_CATEGORIES-1).all():
        raise ValueError(f'{site}: category outside the ordinal visibility scale')
    v['category']=v.category.astype(int)
    v['fog']=v.category.le(FOG_MAX_CATEGORY)
    return w,v.sort_values('time').reset_index(drop=True)
