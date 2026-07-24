"""FastAPI app for serving credit risk predictions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi import HTTPException

from src.api.pydantic_models import PredictionRequest
from src.api.pydantic_models import PredictionResponse

app = FastAPI(title="Credit Risk API", version="0.1.0")


def _load_best_available_model() -> Any:
    """Load best model from local artifact path or MLflow model URI if provided."""
    local_model_path = Path("models/best_model.joblib")
    mlflow_uri = os.getenv("MLFLOW_MODEL_URI", "")

    if local_model_path.exists():
        return joblib.load(local_model_path)

    if mlflow_uri:
        try:
            import mlflow.pyfunc

            return mlflow.pyfunc.load_model(mlflow_uri)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model from MLFLOW_MODEL_URI: {exc}") from exc

    raise RuntimeError(
        "No model artifact found. Train and save models/best_model.joblib or set MLFLOW_MODEL_URI."
    )


@app.on_event("startup")
def startup_event() -> None:
    """Load model once at API startup for low-latency prediction."""
    app.state.model = _load_best_available_model()


@app.get("/health")
def health() -> dict[str, str]:
    """Health endpoint for container/platform checks."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Predict high-risk probability for one customer feature payload."""
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    input_df = pd.DataFrame([payload.features])

    try:
        if hasattr(model, "predict_proba"):
            risk_probability = float(model.predict_proba(input_df)[0][1])
        else:
            pred = model.predict(input_df)[0]
            risk_probability = float(pred)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    risk_label = "high_risk" if risk_probability >= 0.5 else "low_risk"
    return PredictionResponse(risk_probability=risk_probability, risk_label=risk_label)
