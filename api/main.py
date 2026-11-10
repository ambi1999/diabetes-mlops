"""Prediction API: the model's "front door".

What this file is:
  A small web service (FastAPI) that other software calls over HTTP. The web
  app, the smoke test and the demos all talk to the model through it.

Endpoints:
  GET  /health   is the service alive? Which app and model versions are running?
  POST /predict  send one person's answers, get a risk probability back
  GET  /docs     interactive documentation, generated automatically

How to run it locally:
  uv run uvicorn api.main:app --reload     # then open http://localhost:8000/docs

The API only imports lightweight modules (config, schema, prediction_log), so
the Docker image doesn't need MLflow, Prefect or matplotlib."""

import json
import sqlite3

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import ConfigDict, Field, create_model

from src.config import APP_VERSION, METADATA_PATH, MODEL_PATH, PREDICTION_DB, risk_level
from src.prediction_log import log_prediction
from src.schema import EXAMPLE_INPUT, FEATURE_RANGES, FEATURES

DISCLAIMER = (
    "This is an educational estimate based on survey data, not a medical diagnosis. Talk to a doctor about your health."
)

# CONCEPT - input validation at the boundary: the request schema is generated
# from the shared data contract (src/schema.py). Anything the model was never
# trained on (BMI = 500, Age = 40 instead of an age-group code) is rejected
# with a clear 422 error instead of producing a meaningless prediction.
PatientInput = create_model(
    "PatientInput",
    # The shared example from src/schema.py pre-fills "Try it out" in /docs
    __config__=ConfigDict(json_schema_extra={"examples": [EXAMPLE_INPUT]}),
    **{
        name: (float if name == "BMI" else int, Field(..., ge=low, le=high))
        for name, (low, high) in FEATURE_RANGES.items()
    },
)

# The version shown in /docs comes from pyproject.toml via src/config.py
app = FastAPI(title="Diabetes Risk API", version=APP_VERSION)


class ModelStore:
    """Holds the production model in memory.

    It checks the model file's modification time on each request and reloads
    when the pipeline promotes a new model, so no restart is needed."""

    def __init__(self):
        self.model = None
        self.version = None
        self._loaded_mtime = None

    def get(self):
        if not MODEL_PATH.exists():
            raise HTTPException(503, "No model available yet. Run the training pipeline.")
        mtime = MODEL_PATH.stat().st_mtime
        if mtime != self._loaded_mtime:
            self.model = joblib.load(MODEL_PATH)
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
            self.version = metadata.get("model_version", "unknown")
            self._loaded_mtime = mtime
        return self.model, self.version


store = ModelStore()


@app.get("/health")
def health():
    """Is the service alive, and which versions are running?

    Used by Render's health check (render.yaml), the smoke test, Docker
    Compose's healthcheck, and humans verifying a release."""
    _, version = store.get()
    return {"status": "ok", "app_version": APP_VERSION, "model_version": version}


@app.post("/predict")
def predict(patient: PatientInput):
    model, version = store.get()
    features = patient.model_dump()
    # Column order must match training, so build the row from the shared FEATURES list
    row = pd.DataFrame([features], columns=FEATURES)
    probability = float(model.predict_proba(row)[0, 1])

    # Monitoring depends on these logs, but a logging failure must never
    # stop a user from getting their result.
    try:
        log_prediction(PREDICTION_DB, features, probability, version)
    except sqlite3.Error as exc:
        print(f"Warning: could not log prediction: {exc}")

    return {
        "probability": round(probability, 4),
        "risk_level": risk_level(probability),
        "model_version": version,
        "disclaimer": DISCLAIMER,
    }
