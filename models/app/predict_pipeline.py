import joblib
import pandas as pd
import os

def predict(data: dict, model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found.")
    
    bundle = joblib.load(model_path)
    model = bundle["model"]
    encoders = bundle["encoders"]

    df = pd.DataFrame([data])
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(lambda x: x if x in le.classes_ else "UNKNOWN")
            df[col] = le.transform(df[col])

    prob = model.predict_proba(df)[:, 1]
    pred = (prob > 0.5).astype(int)
    return {"prediction": int(pred[0]), "probability": float(prob[0])}
