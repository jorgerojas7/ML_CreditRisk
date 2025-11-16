from pydantic import BaseModel
from typing import List, Dict, Any

class PredictionRequest(BaseModel):
    features: Dict[str, Any]

class BatchPredictionRequest(BaseModel):
    features: List[Dict[str, Any]]
