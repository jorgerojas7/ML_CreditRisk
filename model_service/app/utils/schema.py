from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    age: float = Field(..., example=30)
    income: float = Field(..., example=25000)
    loan_amount: float = Field(..., example=5000)

class BatchPredictionRequest(BaseModel):
    data: List[PredictionRequest]
