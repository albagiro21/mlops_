import os
import logging
import time
from fastapi import FastAPI, HTTPException
from app.schemas import PredictRequest, PredictResponse

import mlflow

# --------------------------------------------------
# Basic logging configuration
# In production this would be more structured (JSON logs, etc.)
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlops-api")

# --------------------------------------------------
# Create FastAPI application instance
# This is the entry point used by Uvicorn:
# uvicorn app.main:app
# --------------------------------------------------
app = FastAPI(title="MLOps Starter API")


# --------------------------------------------------
# Helper function to build the MLflow model URI
# Format: models:/<model_name>/<stage>
# Example: models:/demo-model/Production
# --------------------------------------------------
def _model_uri() -> str:
    model_name = os.getenv("MODEL_NAME", "demo-model")
    model_stage = os.getenv("MODEL_STAGE", "Production")
    return f"models:/{model_name}/{model_stage}"


# --------------------------------------------------
# Load model from MLflow Model Registry
# This connects to the MLflow tracking server,
# fetches the model currently in the specified stage,
# and returns a loaded model ready for inference.
# --------------------------------------------------
def load_model():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    uri = _model_uri()
    logger.info(f"Loading model from MLflow: {uri} (tracking_uri={tracking_uri})")

    # Retry loop: MLflow may not be ready when API starts
    last_err = None
    for attempt in range(1, 11):  # 10 attempts
        try:
            model = mlflow.pyfunc.load_model(uri)
            return model
        except Exception as e:
            last_err = e
            logger.warning(f"Model load attempt {attempt}/10 failed. Retrying in 3s... Error: {e}")
            time.sleep(3)

    # If it still fails after retries, raise the last error
    raise last_err


# --------------------------------------------------
# Startup event
# This runs once when the API container starts.
# We load the model here so we avoid loading it
# on every prediction request.
# --------------------------------------------------
@app.on_event("startup")
def startup_event():
    try:
        app.state.model = load_model()
        app.state.loaded_model_uri = _model_uri()
        logger.info(f"Model loaded successfully: {app.state.loaded_model_uri}")
    except Exception:
        # If model loading fails, we keep the API alive
        # but mark model as not available
        app.state.model = None
        app.state.loaded_model_uri = None
        logger.exception("Failed to load model during startup")


# --------------------------------------------------
# Health check endpoint
# Used by monitoring systems and container orchestration.
# It also tells us if a model is currently loaded.
# --------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": app.state.model is not None,
        "model_uri": app.state.loaded_model_uri,
    }


# --------------------------------------------------
# Model information endpoint
# Useful for debugging and transparency.
# Shows what model and stage the API is configured to use.
# --------------------------------------------------
@app.get("/model-info")
def model_info():
    return {
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        "model_name": os.getenv("MODEL_NAME", "demo-model"),
        "model_stage": os.getenv("MODEL_STAGE", "Production"),
        "model_uri": app.state.loaded_model_uri,
        "model_loaded": app.state.model is not None,
    }


# --------------------------------------------------
# Prediction endpoint
# Receives validated input (PredictRequest),
# runs inference using the loaded model,
# and returns a structured response (PredictResponse).
# --------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):

    # If the model failed to load at startup,
    # we return a 503 (Service Unavailable)
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check MLflow server and model stage.",
        )

    try:
        # ML models expect 2D input: [[f1, f2, ..., fn]]
        prediction = app.state.model.predict([payload.features])

        # Convert numpy scalar to standard Python float
        try:
            pred_value = float(prediction[0].item())
        except Exception:
            pred_value = float(prediction[0])

        return {"prediction": pred_value}

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )