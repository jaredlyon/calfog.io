import pandas as pd
import numpy as np
import os
from pathlib import Path

def calculate_inferred_variables(df):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    
    # Dewpoint Depression (temperature - dewpoint)
    df['dewpoint_depression'] = df['temperature_2m'] - df['dew_point_2m']

    # Cooling Rate over Last 6 Hours
    # Temperature change = (current temp - temp 6 hours ago) / 6
    df['temp_6h_ago'] = df['temperature_2m'].shift(6)
    df['cooling_rate_6h'] = (df['temperature_2m'] - df['temp_6h_ago']) / 6.0
    df.drop('temp_6h_ago', axis=1, inplace=True)
    
    # Cooling Rate over Last 12 Hours
    # Temperature change = (current temp - temp 12 hours ago) / 12
    df['temp_12h_ago'] = df['temperature_2m'].shift(12)
    df['cooling_rate_12h'] = (df['temperature_2m'] - df['temp_12h_ago']) / 12.0
    df.drop('temp_12h_ago', axis=1, inplace=True)
    
    # Previous Night's Low Temperature
    # For each row, go back one day and find minimum temperature between 6pm and 6am
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour

    previous_night_lows = np.empty(len(df))
    previous_night_lows[:] = np.nan  # default NaN

    dates = df['date'].to_numpy()
    unique_dates, first_indices = np.unique(dates, return_index=True)

    # deny first day
    for i in range(1, len(unique_dates)):
        prev_start = first_indices[i - 1]
        curr_start = first_indices[i]
        
        # Previous night: 18:00–23:59 of previous day
        night_prev_day_idx = np.arange(prev_start, prev_start + 24)[18:24]
        
        # Current night portion: 00:00–05:59 of current day
        night_curr_day_idx = np.arange(curr_start, curr_start + 6)
        
        # Combine indices and take min temperature
        night_temps = df.loc[np.concatenate([night_prev_day_idx, night_curr_day_idx]), 'temperature_2m']
        
        night_min = night_temps.min() if not night_temps.empty else np.nan
        
        # Assign the previous night low to all rows of the current day
        if i < len(unique_dates) - 1:
            next_start = first_indices[i + 1]
        else:
            next_start = len(df)
            
        previous_night_lows[curr_start:next_start] = night_min

    df['previous_night_low'] = previous_night_lows
    
    # Drop temporary columns
    df.drop(['date', 'hour'], axis=1, inplace=True)
    
    # Round calculated values to 2 decimal places
    df['dewpoint_depression'] = df['dewpoint_depression'].round(2)
    df['cooling_rate_6h'] = df['cooling_rate_6h'].round(2)
    df['cooling_rate_12h'] = df['cooling_rate_12h'].round(2)
    df['previous_night_low'] = df['previous_night_low'].round(2)
    
    return df

def process_weather_dataset(file_path):
    print(f"\nProcessing: {file_path.name}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records")
        print(f"Original columns: {len(df.columns)}")
        
        df_updated = calculate_inferred_variables(df)
        
        output_path = file_path.parent / file_path.name.replace("_weather.csv", "_weather_appended.csv")
        df_updated.to_csv(output_path, index=False)
        print(f"Saved updated dataset to {output_path.name} with {len(df_updated.columns)} columns")
        print(f"New columns added: dewpoint_depression, previous_night_low, cooling_rate_6h, cooling_rate_12h")
        
        return True
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return False

def main():
    dataset_dir = Path("datasets/weather")
    print(f"Dataset directory: {dataset_dir}")
    
    weather_files = sorted(dataset_dir.glob("*_weather.csv"))
    if not weather_files:
        print("\nNo weather dataset files found!")
        return
    
    print(f"\nFound {len(weather_files)} weather dataset(s):")
    for f in weather_files:
        print(f"  - {f.name}")
    
    success_count = 0
    for weather_file in weather_files:
        if process_weather_dataset(weather_file):
            success_count += 1
    
    print(f"Successfully processed: {success_count}/{len(weather_files)} datasets")

if __name__ == "__main__":
    main()
