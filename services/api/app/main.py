import os
from fastapi import FastAPI
from app.schemas import PredictRequest, PredictResponse

app = FastAPI(title="MLOps Starter API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    # Placeholder: en el próximo paso cargamos el modelo desde MLflow Registry
    pred = sum(payload.features) / len(payload.features)
    return {"prediction": float(pred)}