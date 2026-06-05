import os
import pandas as pd
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

class PaceInferenceEngine:
    def __init__(self, model_dir=os.path.join(PROJECT_ROOT, "models")):
        scaler_path = os.path.join(model_dir, "pace_scaler.joblib")
        model_path = os.path.join(model_dir, "pace_gbm_model.joblib")
        
        if not os.path.exists(scaler_path) or not os.path.exists(model_path):
            raise FileNotFoundError("Model artifacts not found! Run preprocessing and training scripts first.")
            
        self.scaler = joblib.load(scaler_path)
        self.model = joblib.load(model_path)
        self.features = ["latitude", "longitude", "aerosol_optical_depth", "sea_surface_temp"]

    def predict_density(self, features_dict):
        df = pd.DataFrame([features_dict])[self.features]
        scaled = self.scaler.transform(df)
        return max(0.0, self.model.predict(scaled)[0])
        
if __name__ == "__main__":
    sample_coordinates = {"latitude": -14.23, "longitude": -35.12, "aerosol_optical_depth": 0.12, "sea_surface_temp": 24.5}
    engine = PaceInferenceEngine()
    print(f"Predicted Chlorophyll-a: {engine.predict_density(sample_coordinates):.4f} mg/m³")