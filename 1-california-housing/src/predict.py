import os
import pandas as pd
import joblib

class ProductionInferencePipeline:
    def __init__(self, model_dir="models"):
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        self.model = joblib.load(os.path.join(model_dir, "random_forest_model.joblib"))
        self.feature_order = ["medinc", "houseage", "averooms", "avebedrms", "population", "aveoccup", "latitude", "longitude"]

    def predict(self, raw_input_dict):
        df = pd.DataFrame([raw_input_dict])[self.feature_order]
        scaled_features = self.scaler.transform(df)
        return self.model.predict(scaled_features)[0]

if __name__ == "__main__":
    sample_payload = {
        "medinc": 4.5, "houseage": 25.0, "averooms": 5.2, "avebedrms": 1.0,
        "population": 1200.0, "aveoccup": 3.0, "latitude": 34.05, "longitude": -118.24
    }
    pipeline = ProductionInferencePipeline()
    result = pipeline.predict(sample_payload)
    print(f"Predicted House Value: ${result * 100000:,.2f}")