import os
import json
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn

def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    artifacts_dir = Path("/app/artifacts/logistic")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("churn-logistic")

    pipeline = joblib.load(artifacts_dir / "logistic_pipeline.joblib")
    schema = json.loads((artifacts_dir / "schema.json").read_text())
    threshold = json.loads((artifacts_dir / "threshold.json").read_text())["threshold"]

    with mlflow.start_run(run_name="logistic_champion_from_research"):

        mlflow.log_dict(schema, "schema.json")
        mlflow.log_dict({"threshold": threshold}, "threshold.json")

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="churn_logistic_champion"
        )

        print("✅ Model logged correctly")

if __name__ == "__main__":
    main()