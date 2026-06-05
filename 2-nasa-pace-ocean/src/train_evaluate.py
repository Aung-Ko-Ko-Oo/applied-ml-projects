import os
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

def train_model(data_dir="data", model_dir="models"):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    print("Fitting Gradient Boosting pipeline topology...")
    model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    print("\n================== PACE MODEL EVALUATION ==================")
    print(f"Mean Absolute Error (MAE):     {mean_absolute_error(y_test, preds):.4f} mg/m³")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, preds)):.4f} mg/m³")
    print(f"Variance Explained (R² Score): {r2_score(y_test, preds):.4f}")
    print("===========================================================\n")
    
    joblib.dump(model, os.path.join(model_dir, "pace_gbm_model.joblib"))
    print("Production weights exported successfully.")

if __name__ == "__main__":
    train_model()