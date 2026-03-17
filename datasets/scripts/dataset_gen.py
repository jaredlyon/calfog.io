import requests
import pandas as pd
from datetime import datetime
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
# 35.328506, -118.997813 : location_10 / bakersfield airport

COORDINATES = [
    # {"lat": 37.056698, "lon": -120.970441, "name": "location_1"},
    # {"lat": 36.639813, "lon": -120.623363, "name": "location_2"},
    # {"lat": 36.211872, "lon": -120.217350, "name": "location_3"},
    # {"lat": 35.732028, "lon": -119.739409, "name": "location_4"},
    # {"lat": 35.208969, "lon": -119.163142, "name": "location_5"},
    # {"lat": 36.989159, "lon": -120.111030, "name": "location_6"},
    # {"lat": 36.777045, "lon": -119.716862, "name": "location_7"},
    # {"lat": 36.322528, "lon": -119.394712, "name": "location_8"},
    # {"lat": 36.313201, "lon": -119.626015, "name": "location_9"},
    # {"lat": 35.328506, "lon": -118.997813, "name": "location_10"}
]

# data point intervals
# every day, 1980 through 2025, inclusive ----> AQI DATASET ONLY STARTS AT 2022-08-03 18:00:00
# at ALL hours: 0:00 1:00 2:00 3:00 4:00 5:00 6:00 7:00 8:00 9:00 10:00 11:00 12:00 13:00 14:00 15:00 16:00 17:00 18:00 19:00 20:00 21:00 22:00 23:00
# fog usually peaks 3-10am

START_DATE = "1980-01-01"
END_DATE = "2025-12-31"
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

def fetch_weather_data(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "America/Los_Angeles"
    }
    
    print(f"  Fetching weather data for {start_date} to {end_date}...")
    
    try:
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()
        
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
        "timezone": "America/Los_Angeles"
    }
    
    print(f"  Fetching air quality data for {start_date} to {end_date}...")
    
    try:
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()
        
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
        print(f"Air quality data not available: {e}")
        # Return empty DataFrame with proper columns
        return pd.DataFrame(columns=["time"] + AIR_QUALITY_VARS)



def generate_dataset_for_location(coord, output_dir="dataset"):
    print(f"Generating datasets for {coord['name']}")
    print(f"Coordinates: Lat {coord['lat']}, Lon {coord['lon']}")
    
    weather_datasets = []
    aqi_datasets = []
    start_year = 1980
    end_year = 2025
    
    for year in range(start_year, end_year + 1):
        print(f"\nProcessing year {year}...")
        
        year_start = f"{year}-01-01"
        year_end = f"{year}-12-31"
        
        # Fetch weather data for this year
        weather_df = fetch_weather_data(coord["lat"], coord["lon"], year_start, year_end)
        if weather_df is not None and not weather_df.empty:
            weather_datasets.append(weather_df)
        else:
            print(f"Failed to fetch weather data for {year}")
        
        # Wait a bit to avoid hitting rate limits
        time.sleep(4)
        
        # Fetch air quality data for this year (may be empty for years before 2013)
        aqi_df = fetch_air_quality_data(coord["lat"], coord["lon"], year_start, year_end)
        if aqi_df is not None and not aqi_df.empty:
            aqi_datasets.append(aqi_df)
        
        time.sleep(5)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if weather_datasets:
        print(f"\nCombining {len(weather_datasets)} weather datasets...")
        weather_combined = pd.concat(weather_datasets, ignore_index=True)
        weather_combined = weather_combined.sort_values("time").reset_index(drop=True)
        
        weather_file = os.path.join(output_dir, f"{coord['name']}_weather.csv")
        weather_combined.to_csv(weather_file, index=False)
        
        print(f"Weather dataset saved to: {weather_file}")
        print(f"  Total records: {len(weather_combined)}")
        print(f"  Date range: {weather_combined['time'].min()} to {weather_combined['time'].max()}")
        print(f"  Columns: {list(weather_combined.columns)}")
    else:
        print(f"\no weather data collected for {coord['name']}")
        return False
    
    if aqi_datasets:
        print(f"\nCombining {len(aqi_datasets)} AQI datasets...")
        aqi_combined = pd.concat(aqi_datasets, ignore_index=True)
        aqi_combined = aqi_combined.sort_values("time").reset_index(drop=True)
        
        aqi_file = os.path.join(output_dir, f"{coord['name']}_aqi.csv")
        aqi_combined.to_csv(aqi_file, index=False)
        
        print(f"AQI dataset saved to: {aqi_file}")
        print(f"  Total records: {len(aqi_combined)}")
        print(f"  Date range: {aqi_combined['time'].min()} to {aqi_combined['time'].max()}")
        print(f"  Columns: {list(aqi_combined.columns)}")
    else:
        print(f"\nNo AQI data collected for {coord['name']} (expected for early years)")
    
    return True

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