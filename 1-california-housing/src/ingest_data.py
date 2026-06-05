import os
import pandas as pd
from sklearn.datasets import fetch_california_housing

def download_data(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "raw_data.csv")
    
    print("Fetching California Housing raw data...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    df.columns = [col.lower() for col in df.columns]
    
    df.to_csv(raw_path, index=False)
    print(f"Raw data successfully saved to: {raw_path}")
    return raw_path

if __name__ == "__main__":
    download_data()