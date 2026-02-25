import os, json
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from app.schemas import PredictRequest, PredictResponse

app = FastAPI()

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
model_name = os.getenv("MODEL_NAME", "churn_logistic_champion")
model_stage = os.getenv("MODEL_STAGE", "latest")  # keep "latest" for now
artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "/app/artifacts/logistic"))

mlflow.set_tracking_uri(tracking_uri)

# Load schema + threshold
schema = json.loads((artifacts_dir / "schema.json").read_text())
threshold = float(json.loads((artifacts_dir / "threshold.json").read_text())["threshold"])
expected_features = schema["features"]

# Load model from MLflow Model Registry
model_uri = f"models:/{model_name}/{model_stage}"
model = mlflow.sklearn.load_model(model_uri)

@app.get("/health")
def health():
    return {"status": "ok", "model_uri": model_uri, "threshold": threshold}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    missing = [c for c in expected_features if c not in req.data]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    X = pd.DataFrame([[req.data[c] for c in expected_features]], columns=expected_features)

    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= threshold)

    return PredictResponse(
        proba_churn=proba,
        churn_pred=pred,
        threshold=threshold,
        model_name=model_name
    )