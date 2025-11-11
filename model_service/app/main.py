from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.model.pipeline import load_model, predict_single, predict_batch
from app.utils.schema import PredictionRequest, BatchPredictionRequest
import os

app = FastAPI(title="ML Model Service", version="1.0")

ALLOWED_ORIGINS = [os.getenv("API_SERVICE_URL", "http://api:8000")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

model = None

@app.on_event("startup")
def startup_event():
    global model
    model = load_model()
    print("Model loaded successfully.")


@app.post("/predict")
def predict_endpoint(request: PredictionRequest):
    try:
        prediction = predict_single(model, request.dict())
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-batch")
def predict_batch_endpoint(request: BatchPredictionRequest):
    try:
        data = [item.dict() for item in request.data]
        predictions = predict_batch(model, data)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
