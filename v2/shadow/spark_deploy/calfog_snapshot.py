#!/usr/bin/env python3
"""CalFog V2 shadow SNAPSHOT logger (model-independent).

Banks, per run, the RAW production-feed payloads needed to later replay ANY future CalFog model on exactly what
was available at each issue time:
  - Open-Meteo FORECAST API hourly response (a broad SUPERSET of candidate variables) per airport, and
  - AviationWeather METAR (visibility outcomes) per station.
Stores raw JSON + retrieval timestamp + source URL + sha256 to snapshots.sqlite AND to dated files, tagged with
an issue label (env CALFOG_ISSUE, default "hourly"; the fixed 18:00 Pacific run tags "lead_1800").
It does NOT load any model, predict, or expose anything. Predictions/scoring are replayed later against these
snapshots. Never contacts the mesh/Ollama/Minecraft.
"""
import os, sys, json, time, hashlib, sqlite3, datetime, urllib.parse
import requests

BASE = os.path.expanduser("~/calfog-shadow")
DB = os.path.join(BASE, "snapshots.sqlite")
RAW = os.path.join(BASE, "snapshots")
LOG = os.path.join(BASE, "snapshot.log")
ISSUE = os.environ.get("CALFOG_ISSUE", "hourly")

SITES = [
    {"id": "location_6",  "name": "Madera",      "lat": 36.989, "lon": -120.111, "station": "KMAE"},
    {"id": "location_7",  "name": "Fresno",      "lat": 36.777, "lon": -119.717, "station": "KFAT"},
    {"id": "location_8",  "name": "Visalia",     "lat": 36.323, "lon": -119.395, "station": "KVIS"},
    {"id": "location_9",  "name": "Hanford",     "lat": 36.313, "lon": -119.626, "station": "KHJO"},
    {"id": "location_10", "name": "Bakersfield", "lat": 35.329, "lon": -118.998, "station": "KBFL"},
]

FORECAST_HOURLY = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "precipitation", "rain", "surface_pressure", "pressure_msl",
    "vapour_pressure_deficit", "et0_fao_evapotranspiration",
    "wind_speed_10m", "wind_speed_100m", "wind_gusts_10m", "wind_direction_10m",
    "soil_temperature_0cm", "soil_temperature_6cm", "soil_moisture_0_to_1cm",
    "weather_code", "cloud_cover", "cloud_cover_low", "visibility",
]

def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()

def init_db():
    os.makedirs(RAW, exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS snapshots(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        site TEXT, kind TEXT, retrieved_utc TEXT, source_url TEXT,
        http_status INTEGER, sha256 TEXT, n_bytes INTEGER, raw_path TEXT, ok INTEGER)""")
    try:
        c.execute("ALTER TABLE snapshots ADD COLUMN issue_label TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists
    c.commit()
    return c

def log(msg):
    line = f"{now_utc()} [{ISSUE}] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

def fetch(url):
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "calfog-shadow-snapshot/1.1"})
        return r.status_code, r.content
    except Exception as e:  # network hiccup: record, never crash the timer
        return 0, (f'{{"_fetch_error": {json.dumps(str(e))}}}').encode()

def store(c, site, kind, url, status, body):
    ts = now_utc().replace(":", "").replace("-", "")
    sha = hashlib.sha256(body).hexdigest()
    daydir = os.path.join(RAW, now_utc()[:10])
    os.makedirs(daydir, exist_ok=True)
    raw_path = os.path.join(daydir, f"{site}_{kind}_{ISSUE}_{ts}.json")
    with open(raw_path, "wb") as f:
        f.write(body)
    ok = 1 if (status == 200 and b"_fetch_error" not in body) else 0
    c.execute("INSERT INTO snapshots(site,kind,retrieved_utc,source_url,http_status,sha256,n_bytes,raw_path,ok,issue_label)"
              " VALUES(?,?,?,?,?,?,?,?,?,?)",
              (site, kind, now_utc(), url, status, sha, len(body), raw_path, ok, ISSUE))
    c.commit()
    return ok

def main():
    c = init_db()
    ok_n = tot = 0
    for s in SITES:
        fq = urllib.parse.urlencode({
            "latitude": s["lat"], "longitude": s["lon"],
            "hourly": ",".join(FORECAST_HOURLY),
            "past_days": 3, "forecast_days": 2,
            "timezone": "America/Los_Angeles", "cell_selection": "nearest",
        })
        furl = "https://api.open-meteo.com/v1/forecast?" + fq
        st, body = fetch(furl); tot += 1; ok_n += store(c, s["id"], "forecast", furl, st, body)

        mq = urllib.parse.urlencode({"ids": s["station"], "format": "json", "hours": 6})
        murl = "https://aviationweather.gov/api/data/metar?" + mq
        st, body = fetch(murl); tot += 1; ok_n += store(c, s["id"], "metar", murl, st, body)
        time.sleep(1)
    total_rows = c.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
    log(f"run complete: {ok_n}/{tot} fetches ok this run; {total_rows} total snapshot rows")
    c.close()
    return 0 if ok_n == tot else 2

if __name__ == "__main__":
    sys.exit(main())
