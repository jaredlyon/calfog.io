"""Acquire pinned ERA5 winters separately from the retained study inputs (D-106)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models._shared.data import WEATHER_COLS

try:
    from . import dataset_gen as acquisition
except ImportError:
    import dataset_gen as acquisition


def winter_hours(winter):
    return pd.date_range(f"{winter - 1}-11-01", f"{winter}-03-31 23:00:00", freq="h", name="time")


def checked_frame(frame, winter):
    columns = ["time"] + WEATHER_COLS
    if frame.columns.tolist() != columns:
        raise ValueError("ERA5 response lacks the required ordered channels")
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    if not pd.DatetimeIndex(frame.time).equals(winter_hours(winter)):
        raise ValueError(f"ERA5 winter {winter}: incomplete or unordered fixed-PST hours")
    frame[WEATHER_COLS] = frame[WEATHER_COLS].apply(pd.to_numeric, errors="raise").astype(float)
    if not np.isfinite(frame[WEATHER_COLS].to_numpy()).all():
        raise ValueError(f"ERA5 winter {winter}: null/nonfinite required channel")
    return frame


def csv_bytes(frame):
    return frame.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S", lineterminator="\n").encode("utf-8")


def acquire(winters, out):
    out = Path(out)
    roster = pd.read_csv(winters, dtype={"site": str})
    if roster.columns.tolist() != ["site", "winter"] or roster.empty:
        raise ValueError("winters CSV must contain nonempty site,winter rows")
    coordinates = {coord["name"]: coord for coord in acquisition.COORDINATES}
    years = pd.to_numeric(roster.winter, errors="raise")
    if years.isna().any() or not years.eq(years.astype(int)).all() or not roster.site.isin(coordinates).all():
        raise ValueError("invalid site or winter in winters CSV")
    roster["winter"] = years.astype(int)
    if roster.duplicated(["site", "winter"]).any():
        raise ValueError("duplicate site/winter in winters CSV")
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"requests": {}}
    frames = {}
    for site in roster.site.unique():
        path = out / f"{site}_weather_era5_trend.csv"
        frames[site] = acquisition.read_hourly_csv(path) if path.exists() else pd.DataFrame(columns=["time"] + WEATHER_COLS)
    last_request = -float("inf")
    for row in roster.itertuples(index=False):
        site, winter = row.site, row.winter
        key = f"{site}/{winter}"
        hours = winter_hours(winter)
        coord = coordinates[site]
        query = {
            "latitude": coord["lat"], "longitude": coord["lon"],
            "start_date": hours[0].strftime("%Y-%m-%d"), "end_date": hours[-1].strftime("%Y-%m-%d"),
            "hourly": ",".join(WEATHER_COLS), "timezone": acquisition.TIMEZONE, "models": "era5",
            "temperature_unit": "celsius", "wind_speed_unit": "kmh", "precipitation_unit": "mm",
            "domains": None,
        }
        frame = frames[site]
        if key in manifest["requests"]:
            record = manifest["requests"][key]
            if record["query"] != query:
                raise ValueError(f"{key}: completed query differs")
            saved = checked_frame(frame.loc[frame.time.between(hours[0], hours[-1])].reset_index(drop=True), winter)
            if len(saved) != record["rows"] or hashlib.sha256(csv_bytes(saved)).hexdigest() != record["csv_chunk_sha256"]:
                raise ValueError(f"{key}: completed ERA5 chunk hash mismatch")
            continue
        time.sleep(max(0, 2 - (time.monotonic() - last_request)))
        last_request = time.monotonic()
        try:
            fetched = acquisition.fetch_weather_data(coord["lat"], coord["lon"], query["start_date"],
                                                       query["end_date"], model="era5", variables=WEATHER_COLS)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                raise RuntimeError("ERA5 quota exhausted (HTTP 429); stop and resume after quota resets") from exc
            raise
        if fetched is None:
            raise RuntimeError(f"{key}: ERA5 acquisition failed; resume to retry this request")
        new, record = fetched
        new = checked_frame(new, winter)
        if record["query"] != query or record["timezone"] != acquisition.TIMEZONE or record["utc_offset_seconds"] != -28800:
            raise ValueError(f"{key}: ERA5 response query/clock mismatch")
        combined = pd.concat([frame.loc[~frame.time.isin(new.time)], new], ignore_index=True).sort_values("time").reset_index(drop=True)
        if combined.time.duplicated().any():
            raise ValueError(f"{site}: duplicate ERA5 hours")
        path = out / f"{site}_weather_era5_trend.csv"
        temporary = path.with_suffix(".csv.part")
        temporary.write_bytes(csv_bytes(combined))
        temporary.replace(path)
        manifest["requests"][key] = {
            "query": record["query"], "retrieval_utc": record["retrieval_utc"],
            "response_sha256": record["response_sha256"], "units": record["units"],
            "latitude": record["latitude"], "longitude": record["longitude"], "rows": len(new),
            "csv_chunk_sha256": hashlib.sha256(csv_bytes(new)).hexdigest(),
            "csv_chunk_hash_scope": "canonical CSV bytes for this request's winter",
        }
        acquisition.save_json(manifest_path, manifest)
        frames[site] = combined


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winters", type=Path, default=ROOT / "datasets/weather_era5_trend/winters.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "datasets/weather_era5_trend")
    args = parser.parse_args(argv)
    acquire(args.winters, args.out)


if __name__ == "__main__":
    main()
