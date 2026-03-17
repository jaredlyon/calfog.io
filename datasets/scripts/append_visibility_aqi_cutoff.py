import pandas as pd
import os
from datetime import datetime, timedelta

LOCATION_AIRPORT_MAPPING = {
    "location_5": "bear mountain-bakersfield",
    "location_6": "madera airport",
    "location_7": "fresno airport",
    "location_8": "visalia airport",
    "location_9": "hanford airport",
    "location_10": "bakersfield airport"
}

CUTOFF_DATE = "2022-08-03 18:00:00"

def parse_visibility(vis_string):
    try:
        if pd.isna(vis_string):
            return None
        parts = str(vis_string).split(',')
        visibility_meters = int(parts[0])
        return visibility_meters
    except (ValueError, IndexError):
        return None

def visibility_to_fog_binary(visibility_meters):

    if visibility_meters is None:
        return -9999
    return 1 if visibility_meters <= 1000 else 0

def find_closest_timestamp(target_time, airport_times, max_diff_hours=2):
    if len(airport_times) == 0:
        return None, None
    
    time_diffs = (airport_times - target_time).abs()
    min_diff_idx = time_diffs.idxmin()
    min_diff = time_diffs[min_diff_idx]
    
    if min_diff > timedelta(hours=max_diff_hours):
        return None, None
    
    closest_time = airport_times[min_diff_idx]
    diff_minutes = min_diff.total_seconds() / 60
    
    return closest_time, diff_minutes

def process_location(location_name, airport_name):
    print(f"\nProcessing {location_name} with {airport_name}...")
    location_file = f"dataset/{location_name}_dataset.csv"
    if not os.path.exists(location_file):
        print(f"  Error: Location file not found - {location_file}")
        return False
    
    location_df = pd.read_csv(location_file)
    location_df["time"] = pd.to_datetime(location_df["time"])
    airport_file = f"dataset/visibility datasets/{airport_name}.csv"
    if not os.path.exists(airport_file):
        print(f"  Error: Airport file not found - {airport_file}")
        return False
    
    airport_df = pd.read_csv(airport_file)
    airport_df["DATE"] = pd.to_datetime(airport_df["DATE"])
    
    print(f"  Location records: {len(location_df)}")
    print(f"  Airport records: {len(airport_df)}")
    cutoff_dt = pd.to_datetime(CUTOFF_DATE)
    location_filtered = location_df[location_df["time"] >= cutoff_dt].copy()
    print(f"  Location records after {CUTOFF_DATE}: {len(location_filtered)}")
    location_df["fog"] = -9999
    
    exact_matches = 0
    approx_matches = 0
    no_matches = 0
    total_diff_minutes = 0
    
    for idx, row in location_filtered.iterrows():
        target_time = row["time"]
        
        closest_time, diff_minutes = find_closest_timestamp(target_time, airport_df["DATE"])
        
        if closest_time is None:
            print(f"  WARNING: No airport match found for {target_time} (outside 2-hour window)")
            no_matches += 1
            continue
        
        airport_row = airport_df[airport_df["DATE"] == closest_time].iloc[0]
        vis_string = airport_row["VIS"]
        visibility_meters = parse_visibility(vis_string)
        fog_binary = visibility_to_fog_binary(visibility_meters)
        
        location_df.at[idx, "fog"] = fog_binary
        

        if diff_minutes == 0:
            exact_matches += 1
        else:
            approx_matches += 1
            total_diff_minutes += diff_minutes
            print(f"  Non-exact match: {target_time} → {closest_time} (off by {diff_minutes:.1f} minutes, vis={visibility_meters}m, fog={fog_binary})")
    
    print(f"\n  Match Statistics:")
    print(f"    Exact matches:        {exact_matches}")
    print(f"    Approximate matches:  {approx_matches}")
    if approx_matches > 0:
        avg_diff = total_diff_minutes / approx_matches
        print(f"    Average time diff:    {avg_diff:.1f} minutes")
    print(f"    No matches found:     {no_matches}")
    
    fog_records = (location_df["fog"] == 1).sum()
    no_fog_records = (location_df["fog"] == 0).sum()
    missing_records = (location_df["fog"] == -9999).sum()
    
    print(f"\n  Fog Statistics:")
    print(f"    Fog (<=1000m):        {fog_records}")
    print(f"    No fog (>1000m):      {no_fog_records}")
    print(f"    Missing data:         {missing_records}")
    
    output_file = f"dataset/{location_name}_with_visibility.csv"
    location_df.to_csv(output_file, index=False)
    print(f"\n  Saved to: {output_file}")
    
    return True

def main():
    success_count = 0
    for location_name, airport_name in LOCATION_AIRPORT_MAPPING.items():
        try:
            if process_location(location_name, airport_name):
                success_count += 1
        except Exception as e:
            print(f"\nError processing {location_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"Successfully processed {success_count}/{len(LOCATION_AIRPORT_MAPPING)} locations")

if __name__ == "__main__":
    main()
