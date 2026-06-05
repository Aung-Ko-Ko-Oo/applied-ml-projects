import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def run_training(data_dir="data", model_dir="models"):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    print("Training Random Forest Architecture...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print("\n--- Evaluation Metrics Summary ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Coefficient of Determination (R²): {r2:.4f}\n")
    
    joblib.dump(model, os.path.join(model_dir, "random_forest_model.joblib"))
    print("Trained model serialized to disk.")

if __name__ == "__main__":
    run_training()