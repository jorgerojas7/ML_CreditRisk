import joblib
import os
from app.model.preprocess import preprocess_features

MODEL_PATH = os.getenv("MODEL_PATH", "/app/app/model/model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found in the following path: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def predict_single(model, features: dict):
    X = preprocess_features([features])
    y_pred = model.predict_proba(X)[:, 1]
    return float(y_pred[0])

def predict_batch(model, batch: list):
    X = preprocess_features(batch)
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred.tolist()
