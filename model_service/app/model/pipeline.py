import joblib
import os
from app.model.preprocess import preprocess_features

MODEL_PATH = "/app/app/model/model.pkl"
_model = None

def init_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    return _model

def predict_single(model, features: dict):
    X = preprocess_features([features])
    y_pred = model.predict_proba(X)[:, 1]
    return float(y_pred[0])

def predict_batch(model, batch: list):
    X = preprocess_features(batch)
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred.tolist()
