from pydantic import BaseModel
from typing import Dict, Any

class PredictRequest(BaseModel):
    data: Dict[str, Any]

class PredictResponse(BaseModel):
    proba_churn: float
    churn_pred: int
    threshold: float
    model_name: str