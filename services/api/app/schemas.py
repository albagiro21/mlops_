from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_length=1)

class PredictResponse(BaseModel):
    prediction: float