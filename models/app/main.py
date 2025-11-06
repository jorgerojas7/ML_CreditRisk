from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.s3_loader import download_data_from_s3
from app.data_loader import load_credit_dataset
from app.train_pipeline import train_model
from app.predict_pipeline import predict
from models.app.schema import PredictionRequest

app = FastAPI(title="ML Model Service", version="1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://api:8000"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

TARGET_COL = "TARGET_LABEL_BAD=1"
MODEL_PATH = "app/models/model.pkl"

@app.lifespan("startup")
def startup_event():
    _ = download_data_from_s3()
    df = load_credit_dataset()
    metrics = train_model(df, TARGET_COL, MODEL_PATH)
    return {"message": "Modelo entrenado correctamente", "metrics": metrics}


@app.post("/predict")
def predict_endpoint(request: PredictionRequest):
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        prediction = predict(model, request.features)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
