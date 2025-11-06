from pydantic import BaseModel, Field
from typing import Dict, Any

class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Input features for prediction")
