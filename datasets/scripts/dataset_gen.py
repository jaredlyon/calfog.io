import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import hashlib
import json
from pathlib import Path
from datetime import timezone

try:
    from .append_inferred_variables import process_weather_dataset
except ImportError:
    from append_inferred_variables import process_weather_dataset

RAW_DIRECTORY = Path("datasets/raw/open_meteo")
ACQUISITION_MANIFEST = Path("datasets/manifests/weather_cams_acquisition.json")

# api: open-meteo.com
# TODO: get real api key? limited to 10k requests per day <---- monitor this!!!

# from N -> S
# 37.056698, -120.970441 : location_1 / los banos
# 36.639813, -120.623363 : location_2 / panoche
# 36.211872, -120.217350 : location_3 / 198-turk
# 35.732028, -119.739409 : location_4 / twisselman
# 35.208969, -119.163142 : location_5 / bear mountain-bakersfield
# 36.989159, -120.111030 : location_6 / madera airport
# 36.777045, -119.716862 : location_7 / fresno airport
# 36.322528, -119.394712 : location_8 / visalia airport
# 36.313201, -119.626015 : location_9 / hanford airport
# 35.4342, -119.0553 : location_10 / bakersfield meadows field (KBFL; was 35.328506, -118.997813 = Bakersfield Municipal L45)

COORDINATES = [
    # {"lat": 37.056698, "lon": -120.970441, "name": "location_1"},
    # {"lat": 36.639813, "lon": -120.623363, "name": "location_2"},
    # {"lat": 36.211872, "lon": -120.217350, "name": "location_3"},
    # {"lat": 35.732028, "lon": -119.739409, "name": "location_4"},
    # {"lat": 35.208969, "lon": -119.163142, "name": "location_5"},
    {"lat": 36.989159, "lon": -120.111030, "name": "location_6"},
    {"lat": 36.777045, "lon": -119.716862, "name": "location_7"},
    {"lat": 36.322528, "lon": -119.394712, "name": "location_8"},
    {"lat": 36.313201, "lon": -119.626015, "name": "location_9"},
    {"lat": 35.4342, "lon": -119.0553, "name": "location_10"}
]

# data point intervals
# Requests start in 1980 for weather and 2013 for CAMS; observed complete CAMS
# coverage in the retained California arrays starts 2022-08-03 16:00 fixed PST.
# at ALL hours: 0:00 1:00 2:00 3:00 4:00 5:00 6:00 7:00 8:00 9:00 10:00 11:00 12:00 13:00 14:00 15:00 16:00 17:00 18:00 19:00 20:00 21:00 22:00 23:00
# fog usually peaks 3-10am

# Incremental: each location resumes from the last day already in datasets/{weather,aqi}/ (re-fetching that
# day to fill any missing hours), so a re-run only pulls new data. A full 1980->now pull of 5 sites costs
# ~13k weighted calls (> 10k/day free cap); an incremental year is ~250.
START_DATE = "1980-01-01"
END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

# Fixed UTC-8 (PST) all year. Open-Meteo applies the offset in effect at REQUEST time to the whole
# response (no historical DST), and the existing CSVs were pulled in winter, so they are fixed UTC-8.
# "America/Los_Angeles" would shift rows pulled during daylight time by 1 h.
TIMEZONE = "Etc/GMT+8"
HOURS = list(range(24))  # All 24 hours

# feature variables from open-meteo.com - INCLUDED VIA API
# Temperature (2 m)
# Relative Humidity (2 m)
# Dewpoint (2 m)
# Precipitation (rain + snow)
# Rain
# Surface Pressure
# Reference Evapotranspiration (ET₀)
# Vapour Pressure Deficit
# Wind Speed (10 m)
# Wind Speed (100 m)
# Wind Gusts (10 m)
# Soil Temperature (0-7 cm)
# Soil Temperature (7-28 cm)
# Soil Moisture (0-7 cm)
# Soil Moisture (7-28 cm)

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "surface_pressure",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_gusts_10m",
    "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "weather_code", # response
    "cloud_cover_low" # response
]

# variables to infer and append using code - SEE EXTRA SCRIPT IN dataset/scripts/
# dewpoint depression (temperature - dewpoint)
# previous night's low temperature
# cooling rate over last 6 hours
# cooling rate over last 12 hours

# feature variables from open-meteo.com AQI API that help with fog - INCLUDED VIA API
# Particulate Matter PM10
# Particulate Matter PM2.5
# Aerosol Optical Depth
# Dust
# Nitrogen Dioxide NO2

AIR_QUALITY_VARS = [
    "pm10",
    "pm2_5",
    "aerosol_optical_depth",
    "dust",
    "nitrogen_dioxide"
]

# feature variables from open-meteo.com AQI API that may help with fog but are not as well studied - NOT INCLUDED IN DATASET FOR NOW
# Carbon Monoxide CO
# Carbon Dioxide CO2
# Nitrogen Dioxide NO2
# Sulphur Dioxide SO2
# Ozone O3
# UV Index
# UV Index Clear Sky
# Ammonia NH3
# Methane CH4

# response variables - INCLUDED VIA API
# Weather code
#   to interpret, see WMO code: https://www.nodc.noaa.gov/archive/arc0021/0002199/1.1/data/0-data/HTML/WMO-CODE/WMO4677.HTM
#   fog: codes 40-49
#   codes 45 and and 48 are medium fog
#   code 49 for severe fog
#   all other codes are not fog, so this would be negative binary classification
# Cloud Cover Low

def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def get_json(url, params, retries=5):
    for attempt in range(retries):
        response = requests.get(url, params=params, timeout=120)
        if response.status_code != 429 or attempt == retries - 1:
            break
        wait = 60 * (attempt + 1)
        print(f"  Rate limited (429), sleeping {wait}s...")
        time.sleep(wait)
    response.raise_for_status()
    retrieval = datetime.now(timezone.utc).isoformat()
    content = response.content
    data = json.loads(content)
    digest = hashlib.sha256(content).hexdigest()
    raw_path = RAW_DIRECTORY / f"{digest}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_suffix(".json.part")
    temporary.write_bytes(content)
    temporary.replace(raw_path)
    record = {
        "endpoint": url, "resolved_url": response.url,
        "query": {**params, "models": params.get("models"), "domains": params.get("domains")},
        "retrieval_utc": retrieval, "status": response.status_code,
        "latitude": data.get("latitude"), "longitude": data.get("longitude"),
        "elevation": data.get("elevation"), "timezone": data.get("timezone"),
        "utc_offset_seconds": data.get("utc_offset_seconds"), "units": data.get("hourly_units"),
        "raw_path": raw_path.as_posix(), "response_sha256": digest,
        "response_hash_scope": "HTTP response-content bytes", "upstream_version": None,
    }
    return data, record


def fetch_weather_data(lat, lon, start_date, end_date, *, model=None, variables=WEATHER_VARS):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date,
        "hourly": ",".join(variables), "timezone": TIMEZONE,
    }
    if model is not None:
        params.update(models=model, temperature_unit="celsius", wind_speed_unit="kmh",
                      precipitation_unit="mm")
    print(f"  Fetching weather data for {start_date} to {end_date}...")
    try:
        # A pinned winter request stops on quota exhaustion; resume it the next day.
        data, record = get_json(url, params, retries=1) if model == "era5" else get_json(url, params)
        hourly = data.get("hourly", {})
        df = pd.DataFrame({"time": hourly.get("time", [])})
        for var in variables:
            df[var] = hourly[var] if var in hourly else pd.NA
        df["time"] = pd.to_datetime(df["time"])
        print(f"Weather data: {len(df)} records")
        return df, record
    except requests.exceptions.RequestException as e:
        if model == "era5":
            raise
        print(f"Error fetching weather data: {e}")
        return None


def fetch_air_quality_data(lat, lon, start_date, end_date):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date,
        "hourly": ",".join(AIR_QUALITY_VARS), "timezone": TIMEZONE,
    }
    print(f"  Fetching air quality data for {start_date} to {end_date}...")
    try:
        data, record = get_json(url, params)
        hourly = data.get("hourly", {})
        df = pd.DataFrame({"time": hourly.get("time", [])})
        for var in AIR_QUALITY_VARS:
            df[var] = hourly[var] if var in hourly else pd.NA
        df["time"] = pd.to_datetime(df["time"])
        print(f"Air quality data: {len(df)} records")
        return df, record
    except requests.exceptions.RequestException as e:
        print(f"Error fetching air quality data: {e}")
        return None


def read_hourly_csv(path):
    frame = pd.read_csv(path, float_precision="round_trip")
    frame["time"] = pd.to_datetime(frame["time"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    return frame


def pull(fetch, path, coord, first_date=START_DATE):
    # Preserve old bytes and strict >last retention even when archive overlap changes.
    path = Path(path)
    old = read_hourly_csv(path) if path.exists() else None
    last = old.time.max() if old is not None and not old.empty else None
    start = last.normalize() if last is not None else pd.Timestamp(first_date)
    end = pd.Timestamp(END_DATE)
    manifest = json.loads(ACQUISITION_MANIFEST.read_text(encoding="utf-8")) if ACQUISITION_MANIFEST.exists() else {"files": {}}
    entry = manifest["files"].setdefault(path.as_posix(), {
        "imported": None if old is None else {
            "endpoint": None, "resolved_url": None, "query": None, "retrieval_utc": None,
            "status": "existing_csv", "response_sha256": None, "raw_path": None,
            "latitude": None, "longitude": None, "elevation": None, "timezone": None,
            "utc_offset_seconds": None, "units": None, "upstream_version": None,
            "csv_sha256": file_sha256(path), "csv_hash_scope": "complete retained CSV bytes",
        },
        "requests": [],
    })
    save_json(ACQUISITION_MANIFEST, manifest)
    frames, records = [], []
    for year in range(start.year, end.year + 1):
        chunk_start = max(start, pd.Timestamp(year, 1, 1)).strftime("%Y-%m-%d")
        chunk_end = min(end, pd.Timestamp(year, 12, 31)).strftime("%Y-%m-%d")
        if chunk_start > chunk_end:
            continue
        fetched = fetch(coord["lat"], coord["lon"], chunk_start, chunk_end)
        if fetched is None:
            print(f"  Failed {chunk_start}..{chunk_end}; nothing written for {path}")
            return False
        df, record = fetched
        accepted = df if last is None else df.loc[df.time.gt(last)]
        record = {**record, "accepted_start": None, "accepted_end": None,
                  "accepted_rows": len(accepted), "csv_sha256": None,
                  "csv_hash_scope": "complete resulting CSV bytes"}
        if not accepted.empty:
            record.update(accepted_start=accepted.time.min().isoformat(),
                          accepted_end=accepted.time.max().isoformat())
            frames.append(accepted)
        entry["requests"].append(record)
        records.append(record)
        save_json(ACQUISITION_MANIFEST, manifest)
        time.sleep(5)
    if frames:
        new = pd.concat(frames, ignore_index=True).sort_values("time").drop_duplicates("time", keep="first")
        if old is not None:
            new = new[old.columns]
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        if old is not None and not old.empty:
            content = path.read_bytes()
            with temporary.open("wb") as stream:
                stream.write(content)
                if not content.endswith(b"\n"):
                    stream.write(b"\n")
                stream.write(new.to_csv(index=False, header=False, date_format="%Y-%m-%d %H:%M:%S").encode("utf-8"))
        else:
            new.to_csv(temporary, index=False, date_format="%Y-%m-%d %H:%M:%S")
        temporary.replace(path)
        print(f"  Wrote {len(new)} rows to {path}")
    digest = file_sha256(path) if path.exists() else None
    for record in records:
        record["csv_sha256"] = digest
    entry["csv_sha256"] = digest
    entry["csv_hash_scope"] = "complete resulting CSV bytes"
    save_json(ACQUISITION_MANIFEST, manifest)
    return True


def generate_dataset_for_location(coord):
    print(f"Generating datasets for {coord['name']}")
    weather_path = Path("datasets/weather") / f"{coord['name']}_weather.csv"
    appended_path = weather_path.with_name(f"{coord['name']}_weather_appended.csv")
    columns = ["time"] + WEATHER_VARS
    if appended_path.exists():
        appended = read_hourly_csv(appended_path)[columns]
        if not weather_path.exists():
            temporary = weather_path.with_suffix(".csv.part")
            appended.to_csv(temporary, index=False, date_format="%Y-%m-%d %H:%M:%S")
            temporary.replace(weather_path)
        else:
            raw = read_hourly_csv(weather_path)[columns]
            if raw.time.duplicated().any() or appended.time.duplicated().any() or not appended.time.isin(raw.time).all():
                raise ValueError("raw weather does not cover retained appended weather")
            overlap = raw.set_index("time").loc[appended.time].reset_index()
            try:
                pd.testing.assert_frame_equal(overlap, appended.reset_index(drop=True),
                                              check_dtype=False, check_exact=True)
            except AssertionError as exc:
                raise ValueError("raw/appended weather overlap differs") from exc
    weather_ok = pull(fetch_weather_data, weather_path, coord)
    if weather_ok:
        weather_ok = process_weather_dataset(weather_path)
    # 2013 is the request boundary, not the start of observed California coverage.
    aqi_ok = pull(fetch_air_quality_data, Path("datasets/aqi") / f"{coord['name']}_aqi.csv",
                  coord, first_date="2013-01-01")
    return weather_ok and aqi_ok

def main():
    print(f"Time range: {START_DATE} to {END_DATE}")
    print(f"Locations: {len(COORDINATES)}")
    
    success_count = 0
    
    for coord in COORDINATES:
        try:
            if generate_dataset_for_location(coord):
                success_count += 1
            # Wait between locations to respect API rate limits
            time.sleep(2)
        except Exception as e:
            print(f"Error processing {coord['name']}: {e}")
            continue
    
    print(f"Successfully generated {success_count}/{len(COORDINATES)} datasets")

if __name__ == "__main__":
    main()