from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    # Diabetes dataset = 10 features
    features: List[float] = Field(..., min_length=10, max_length=10)

class PredictResponse(BaseModel):
    prediction: float