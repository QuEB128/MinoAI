# src/model_training.py
import joblib
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

def train_and_save_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    Path("models").mkdir(exist_ok=True)

    joblib.dump(model, "models/random_forest_model.pkl")
    print("Model saved to models/random_forest_model.pkl")

    return model
