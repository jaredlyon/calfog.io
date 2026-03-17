import pandas as pd
from pathlib import Path
import os

def combine_location_datasets(location_num, aqi_dir, weather_dir, output_dir):
    aqi_file = aqi_dir / f"location_{location_num}_aqi.csv"
    weather_file = weather_dir / f"location_{location_num}_weather_appended.csv"
    
    print(f"Processing Location {location_num}")
    
    if not aqi_file.exists():
        print(f"AQI file not found: {aqi_file}")
        return False
    
    if not weather_file.exists():
        print(f"Weather file not found: {weather_file}")
        return False
    
    aqi_df = pd.read_csv(aqi_file)
    weather_df = pd.read_csv(weather_file)
    
    aqi_cols = ['time', 'pm10', 'pm2_5', 'aerosol_optical_depth', 'dust', 'nitrogen_dioxide']
    aqi_selected = aqi_df[aqi_cols].copy()
    
    weather_cols = [
        'time', 'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 
        'precipitation', 'rain', 'surface_pressure', 'et0_fao_evapotranspiration', 
        'vapour_pressure_deficit', 'wind_speed_10m', 'wind_speed_100m', 'wind_gusts_10m', 
        'soil_temperature_0_to_7cm', 'soil_temperature_7_to_28cm', 'soil_moisture_0_to_7cm', 
        'soil_moisture_7_to_28cm', 'dewpoint_depression', 'cooling_rate_6h', 
        'cooling_rate_12h', 'previous_night_low'
    ]
    weather_selected = weather_df[weather_cols].copy()
    
    aqi_selected['time'] = pd.to_datetime(aqi_selected['time'])
    weather_selected['time'] = pd.to_datetime(weather_selected['time'])
    combined_df = pd.merge(weather_selected, aqi_selected, on='time', how='left') # preserve all weather rows
    
    output_file = output_dir / f"location_{location_num}_combined.csv"
    combined_df.to_csv(output_file, index=False)
    
    print(f"Saved combined dataset to {output_file}")
    print(f"Date range: {combined_df['time'].min()} to {combined_df['time'].max()}")
    
    return True

def combine_with_visibility(location_num, aqi_dir, weather_dir, visibility_dir, output_dir):
    aqi_file = aqi_dir / f"location_{location_num}_aqi.csv"
    weather_file = weather_dir / f"location_{location_num}_weather_appended.csv"
    visibility_file = visibility_dir / f"location_{location_num}_visibility.csv"
    
    if not aqi_file.exists():
        print(f"AQI file not found: {aqi_file}")
        return False
    if not weather_file.exists():
        print(f"Weather file not found: {weather_file}")
        return False
    if not visibility_file.exists():
        print(f"Visibility file not found: {visibility_file}")
        return False
    
    aqi_df = pd.read_csv(aqi_file)
    weather_df = pd.read_csv(weather_file)
    visibility_df = pd.read_csv(visibility_file)
    
    aqi_cols = ['time', 'pm10', 'pm2_5', 'aerosol_optical_depth', 'dust', 'nitrogen_dioxide']
    aqi_selected = aqi_df[aqi_cols].copy()
    
    weather_cols = [
        'time', 'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 
        'precipitation', 'rain', 'surface_pressure', 'et0_fao_evapotranspiration', 
        'vapour_pressure_deficit', 'wind_speed_10m', 'wind_speed_100m', 'wind_gusts_10m', 
        'soil_temperature_0_to_7cm', 'soil_temperature_7_to_28cm', 'soil_moisture_0_to_7cm', 
        'soil_moisture_7_to_28cm', 'dewpoint_depression', 'cooling_rate_6h', 
        'cooling_rate_12h', 'previous_night_low'
    ]
    weather_selected = weather_df[weather_cols].copy()
    
    # Process visibility data
    # Extract visibility value (first number from VIS column like "016093,5,N,5")
    def parse_visibility(vis_str):
        if pd.isna(vis_str):
            return None
        try:
            return int(str(vis_str).split(',')[0])
        except:
            return None
    
    visibility_df['visibility_meters'] = visibility_df['VIS'].apply(parse_visibility)
    
    visibility_selected = visibility_df[['DATE', 'visibility_meters']].copy()
    visibility_selected = visibility_selected.dropna(subset=['visibility_meters'])
    
    visibility_selected['time'] = pd.to_datetime(visibility_selected['DATE'])
    visibility_selected = visibility_selected[['time', 'visibility_meters']]
    visibility_selected = visibility_selected.sort_values('time')
    
    aqi_selected['time'] = pd.to_datetime(aqi_selected['time'])
    weather_selected['time'] = pd.to_datetime(weather_selected['time'])
    
    print(f"Merging weather and AQI data.")
    combined_df = pd.merge(weather_selected, aqi_selected, on='time', how='left')
    combined_df = combined_df.sort_values('time')
    
    print(f"Combined dataset (before visibility): {len(combined_df)} records")
    
    print(f"Merging with visibility data using closest time match...")
    final_df = pd.merge_asof(
        combined_df,
        visibility_selected,
        on='time',
        direction='nearest',
        tolerance=pd.Timedelta('30min')
    )
    
    final_df = final_df.dropna(subset=['visibility_meters'])
    
    print(f"Final dataset (with visibility): {len(final_df)} records")
    
    output_file = output_dir / f"location_{location_num}_combined.csv"
    final_df.to_csv(output_file, index=False)
    
    print(f"Saved combined dataset to {output_file}")
    print(f"Date range: {final_df['time'].min()} to {final_df['time'].max()}")
    print(f"Columns: {len(final_df.columns)}")
    
    return True

def main():
    aqi_dir = Path("datasets/aqi")
    weather_dir = Path("datasets/weather")
    visibility_dir = Path("datasets/visibility")
    output_dir = Path("eda/datasets")

    print(f"AQI directory: {aqi_dir}")
    print(f"Weather directory: {weather_dir}")
    print(f"Visibility directory: {visibility_dir}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    aqi_files = sorted(aqi_dir.glob("location_*_aqi.csv"))
    
    if not aqi_files:
        print("\nNo AQI files found!")
        return
    
    location_nums = []
    for aqi_file in aqi_files:
        parts = aqi_file.stem.split('_')
        if len(parts) >= 2:
            try:
                location_num = int(parts[1])
                location_nums.append(location_num)
            except ValueError:
                continue
    
    location_nums = sorted(location_nums)
    print(f"\nLocations to process: {location_nums}")
    
    success_count = 0
    for location_num in location_nums:
        try:
            if combine_with_visibility(location_num, aqi_dir, weather_dir, visibility_dir, output_dir):
                success_count += 1
        except Exception as e:
            print(f"Error processing location {location_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"Successfully combined: {success_count}/{len(location_nums)} datasets")


if __name__ == "__main__":
    main()
