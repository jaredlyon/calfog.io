import pandas as pd
import numpy as np
import os
from datetime import datetime

LOCATIONS = [
    "location_1",
    "location_2",
    "location_3",
    "location_4",
    "location_5",
    "location_6",
    "location_7",
    "location_8",
    "location_9",
    "location_10"
]

AQI_VARS = ["pm10", "pm2_5", "aerosol_optical_depth", "dust"]
AQI_START_DATE = "2022-08-03 18:00:00"
SPECIAL_VARS = ["time", "weather_code", "year"]

def load_dataset(location_name, dataset_dir="dataset"):
    """
    Load a location dataset CSV
    """
    file_path = os.path.join(dataset_dir, f"{location_name}_dataset.csv")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    
    return df

def get_clean_value(value, var_name):
    if pd.isna(value):
        return None
    
    if var_name in AQI_VARS and value == -9999:
        return None
    
    return value

def calculate_statistics(df, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        numeric_vars = [col for col in df.columns if col not in SPECIAL_VARS]
        f.write(f"Total timestamps: {len(df)}\n")
        f.write(f"Date range: {df['time'].min()} to {df['time'].max()}\n\n")
        
        for var in numeric_vars:
            clean_values = []
            for val in df[var]:
                clean_val = get_clean_value(val, var)
                if clean_val is not None:
                    clean_values.append(clean_val)
            
            if not clean_values:
                f.write(f"{var}:\n")
                f.write(f"  No valid data available\n\n")
                continue
            
            clean_series = pd.Series(clean_values)
            
            var_min = clean_series.min()
            var_max = clean_series.max()
            var_median = clean_series.median()
            var_range = var_max - var_min
            var_mean = clean_series.mean()
            var_std = clean_series.std()
            
            f.write(f"{var}:\n")
            f.write(f"  Min:              {var_min:12.4f}\n")
            f.write(f"  Max:              {var_max:12.4f}\n")
            f.write(f"  Median:           {var_median:12.4f}\n")
            f.write(f"  Range:            {var_range:12.4f}\n")
            f.write(f"  Average:          {var_mean:12.4f}\n")
            f.write(f"  Std Deviation:    {var_std:12.4f}\n")
            
            if var in AQI_VARS:
                f.write(f"  Valid samples:    {len(clean_values)} (excludes -9999 markers)\n")
            
            f.write("\n")

def process_location(location_name):
    print(f"\nProcessing {location_name}...")
    
    df = load_dataset(location_name)
    
    if df is None:
        return False
    
    output_dir = os.path.join("eda", location_name)
    os.makedirs(output_dir, exist_ok=True)
    
    overall_file = os.path.join(output_dir, f"{location_name}_overall.txt")
    calculate_statistics(df, overall_file)
    print(f"  Generated: {overall_file}")

    years = sorted(df["year"].unique())
    for year in years:
        year_df = df[df["year"] == year].copy()
        year_file = os.path.join(output_dir, f"{location_name}_{year}.txt")
        calculate_statistics(year_df, year_file)
        print(f"  Generated: {year_file}")
    
    return True

def main():
    
    success_count = 0
    
    for location in LOCATIONS:
        try:
            if process_location(location):
                success_count += 1
        except Exception as e:
            print(f"\nError processing {location}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"Successfully processed {success_count}/{len(LOCATIONS)} locations")

if __name__ == "__main__":
    main()
