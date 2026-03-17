import pandas as pd
import glob
import os

def split_datasets():
    aqi_columns = ['pm10', 'pm2_5', 'aerosol_optical_depth', 'dust', 'nitrogen_dioxide']
    combined_files = glob.glob('location_*_combined.csv')
    print(f"Found {len(combined_files)} combined dataset(s)")
    
    for file in combined_files:
        location_num = file.split('_')[1]
        df = pd.read_csv(file)
        print(f"Original shape: {df.shape}")
        
        aqi_df = df.dropna(subset=aqi_columns)
        aqi_filename = f'location_{location_num}_with_aqi.csv'
        aqi_df.to_csv(aqi_filename, index=False)
        print(f"Created {aqi_filename} - Shape: {aqi_df.shape}")
        
        no_aqi_df = df.drop(columns=aqi_columns)
        no_aqi_filename = f'location_{location_num}_without_aqi.csv'
        no_aqi_df.to_csv(no_aqi_filename, index=False)
        print(f"Created {no_aqi_filename} - Shape: {no_aqi_df.shape}")

        no_aqi_df_reduced = aqi_df.drop(columns=aqi_columns)
        no_aqi_filename_reduced = f'location_{location_num}_without_aqi_reduced.csv'
        no_aqi_df_reduced.to_csv(no_aqi_filename_reduced, index=False)
        print(f"Created {no_aqi_filename_reduced} - Shape: {no_aqi_df_reduced.shape}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    split_datasets()
