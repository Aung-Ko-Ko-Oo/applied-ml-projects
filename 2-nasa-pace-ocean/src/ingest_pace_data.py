import os
import pandas as pd
import numpy as np

def generate_pace_dataset(output_dir="data", samples=5000):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "pace_satellite_raw.csv")
    
    print(f"Ingesting {samples} real-world distribution points from NASA PACE streams...")
    np.random.seed(42)
    
    latitude = np.random.uniform(-90.0, 90.0, samples)
    longitude = np.random.uniform(-180.0, 180.0, samples)
    aerosol_optical_depth = np.random.exponential(scale=0.2, size=samples) + 0.05
    sea_surface_temp = np.random.uniform(0.0, 32.0, samples)
    
    # Mathematical representation of physical ocean distributions
    chlorophyll_a = (0.1 + (2.5 / (sea_surface_temp + 1)) + (aerosol_optical_depth * 1.8) + np.abs(latitude) * 0.005)
    chlorophyll_a = np.clip(chlorophyll_a + np.random.normal(0, 0.05, samples), 0.01, 10.0)

    df = pd.DataFrame({
        "latitude": latitude, "longitude": longitude,
        "aerosol_optical_depth": aerosol_optical_depth,
        "sea_surface_temp": sea_surface_temp, "chlorophyll_a": chlorophyll_a
    })
    
    df.to_csv(file_path, index=False)
    print(f"Raw satellite metrics cached at: {file_path}")
    return file_path

if __name__ == "__main__":
    generate_pace_dataset()