import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def prepare_datasets(input_path="data/raw_data.csv", output_dir="data", model_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    df = pd.read_csv(input_path)
    
    # Cap severe statistical outliers
    df = df[(df["averooms"] < 15) & (df["avebedrms"] < 5) & (df["population"] < 5000)]
    
    X = df.drop(columns=["medhouseval"])
    y = df["medhouseval"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    
    np.save(os.path.join(output_dir, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test_scaled)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train.values)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test.values)
    
    print("Data preprocessing complete. Scaler and matrices cached.")

if __name__ == "__main__":
    prepare_datasets()