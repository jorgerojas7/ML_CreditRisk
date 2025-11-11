from fastapi import APIRouter, Depends, HTTPException
from app.auth.dependencies import require_client
from app.users.models import User
import httpx

router = APIRouter(prefix="/predictions", tags=["Predictions"])

MODEL_SERVICE_URL = "http://ml_model:8002"

@router.post("/predict-one")
async def predict_one(
    payload: dict,
    current_user: User = Depends(require_client),
):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{MODEL_SERVICE_URL}/predict", json=payload)
            response.raise_for_status()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Error calling model service: {e}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=response.status_code, detail=f"Error calling model service: {response.text}")

    return response.json()

@router.post("/predict-batch")
async def predict_batch(
    payload: dict,
    current_user: User = Depends(require_client),
):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{MODEL_SERVICE_URL}/predict-batch", json=payload)
            response.raise_for_status()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Error calling model service: {e}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=response.status_code, detail=f"Error calling model service: {response.text}")

    return response.json()
