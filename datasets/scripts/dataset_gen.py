import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

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
# every day, 1980 through yesterday, inclusive ----> AQI DATASET ONLY STARTS AT 2022-08-03 18:00:00
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

def get_json(url, params, retries=5):
    # back off on HTTP 429; a daily-quota 429 still fails after retries, so re-run tomorrow to resume
    for attempt in range(retries):
        response = requests.get(url, params=params, timeout=120)
        if response.status_code != 429:
            break
        wait = 60 * (attempt + 1)
        print(f"  Rate limited (429), sleeping {wait}s...")
        time.sleep(wait)
    response.raise_for_status()
    return response.json()

def fetch_weather_data(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(WEATHER_VARS),
        "timezone": TIMEZONE
    }
    
    print(f"  Fetching weather data for {start_date} to {end_date}...")
    
    try:
        data = get_json(url, params)
        
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        df = pd.DataFrame({"time": times})
        
        for var in WEATHER_VARS:
            if var in hourly:
                df[var] = hourly[var]
        
        df["time"] = pd.to_datetime(df["time"])
        print(f"Weather data: {len(df)} records")
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def fetch_air_quality_data(lat, lon, start_date, end_date):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(AIR_QUALITY_VARS),
        "timezone": TIMEZONE
    }
    
    print(f"  Fetching air quality data for {start_date} to {end_date}...")
    
    try:
        data = get_json(url, params)
        
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        
        if not times:
            print(f"Air quality data empty (no data available)")
            return pd.DataFrame(columns=["time"] + AIR_QUALITY_VARS)
        
        df = pd.DataFrame({"time": times})
        
        for var in AIR_QUALITY_VARS:
            if var in hourly:
                df[var] = hourly[var]
            else:
                df[var] = pd.NA
        
        df["time"] = pd.to_datetime(df["time"])
        print(f"Air quality data: {len(df)} records")
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching air quality data: {e}")
        return None



def pull(fetch, path, coord, first_date=START_DATE):
    # Append rows after the last timestamp on disk, through END_DATE. Existing rows are never rewritten, and
    # nothing is written if any request fails, so a re-run resumes cleanly.
    last = pd.read_csv(path, usecols=["time"], parse_dates=["time"])["time"].max() if os.path.exists(path) else None
    start = last.normalize() if last is not None else pd.Timestamp(first_date)  # re-fetch last day to finish it
    end = pd.Timestamp(END_DATE)
    if start > end:
        print(f"  {path} already up to date")
        return True

    frames = []
    for year in range(start.year, end.year + 1):
        chunk_start = max(start, pd.Timestamp(year, 1, 1)).strftime("%Y-%m-%d")
        chunk_end = min(end, pd.Timestamp(year, 12, 31)).strftime("%Y-%m-%d")
        df = fetch(coord["lat"], coord["lon"], chunk_start, chunk_end)
        if df is None:
            print(f"  Failed {chunk_start}..{chunk_end}; nothing written for {path}")
            return False
        if not df.empty:
            frames.append(df)
        time.sleep(5)  # stay under the per-minute limit

    new = pd.concat(frames, ignore_index=True).sort_values("time") if frames else pd.DataFrame()
    if last is not None and not new.empty:
        new = new[new["time"] > last][pd.read_csv(path, nrows=0).columns]  # keep on-disk column order
    if new.empty:
        print(f"  No new data for {path}")
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new.to_csv(path, mode="a" if last is not None else "w", header=last is None, index=False)
    print(f"  Wrote {len(new)} rows to {path}: {new['time'].min()} to {new['time'].max()}")
    return True


def generate_dataset_for_location(coord):
    # run from the repository root, like the other datasets/ and eda/ scripts
    print(f"Generating datasets for {coord['name']}")
    print(f"Coordinates: Lat {coord['lat']}, Lon {coord['lon']}")
    weather_ok = pull(fetch_weather_data, os.path.join("datasets", "weather", f"{coord['name']}_weather.csv"), coord)
    # the air-quality API rejects dates before 2013 (CAMS coverage for California starts 2022-08)
    aqi_ok = pull(fetch_air_quality_data, os.path.join("datasets", "aqi", f"{coord['name']}_aqi.csv"), coord,
                  first_date="2013-01-01")
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