import pandas as pd
import os

LOCATIONS = ["location_7_weather", "location_10_weather"]

def main():
    for location in LOCATIONS:
        file_path = f"dataset/{location}.csv"
        df = pd.read_csv(file_path)
        
        codes = df[df["weather_code"] != ""]["weather_code"].dropna().astype(int)
        counts = codes.value_counts().sort_index()
        
        print(f"\n{location}:")
        for code, count in counts.items():
            print(f"  {code}: {count}")

if __name__ == "__main__":
    main()
