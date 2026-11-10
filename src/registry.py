"""Model registry and production model: everything about saving, versioning
and promoting models lives here, so the pipeline only has to orchestrate.

Two places hold models, on purpose:
  * MLflow registry: the full history of every model version (for people).
  * models/ folder:  the ONE production model the API serves (for software).
Promotion updates both, so they never disagree."""

import json
from datetime import UTC, datetime

import joblib
import mlflow
import pandas as pd
import sklearn
from mlflow.tracking import MlflowClient

from src.config import (
    APP_VERSION,
    EXPERIMENT_NAME,
    METADATA_PATH,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    MODELS_DIR,
    PRODUCTION_ALIAS,
    RANDOM_SEED,
    REFERENCE_PATH,
    REFERENCE_SAMPLE_SIZE,
    REGISTERED_MODEL_NAME,
    SKOPS_TRUSTED_TYPES,
)
from src.evaluation import evaluate_model
from src.schema import FEATURES


def make_reference_sample(train: pd.DataFrame) -> pd.DataFrame:
    """A fixed-size random sample of the training inputs (no labels).

    Saved next to the production model: drift monitoring compares live users
    against it, and the quality gate checks the model's outputs on it.

    Args:
        train: the data the model was trained on.

    Returns:
        Up to REFERENCE_SAMPLE_SIZE rows of feature columns (same rows every
        time for the same data, thanks to the fixed random seed)."""
    sample_size = min(REFERENCE_SAMPLE_SIZE, len(train))
    return train[FEATURES].sample(sample_size, random_state=RANDOM_SEED)


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_and_register_model(model, train: pd.DataFrame) -> str:
    """Save the model to MLflow and register it as a new version.

    Must be called inside an active MLflow run. Registering does NOT put the
    model into production; only promote_model() does that."""
    info = mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name=REGISTERED_MODEL_NAME,
        input_example=train[FEATURES].head(3),  # documents the expected input format
        # MLflow test-loads the saved model; these are the types we vouch for (see config)
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
    )
    return str(info.registered_model_version)


def score_production_model(evaluation: pd.DataFrame) -> dict | None:
    """Score the current production model (the "champion") on the given data.

    CONCEPT - champion vs. challenger: the new model and the production model
    are scored on the SAME evaluation data, so the comparison is fair."""
    if not MODEL_PATH.exists():
        return None
    return evaluate_model(joblib.load(MODEL_PATH), evaluation)


def is_better(challenger: dict, champion: dict | None, metric: str) -> bool:
    """Should the challenger replace the champion? (higher metric = better)"""
    return champion is None or challenger[metric] > champion[metric]


def promote_model(
    model, version: str, run_id: str, model_name: str, metrics: dict, train: pd.DataFrame, reason: str
) -> None:
    """Make this model version the production model."""
    # 1. Point the "production" alias in the MLflow registry at this version
    MlflowClient(MLFLOW_TRACKING_URI).set_registered_model_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, version)

    # 2. Export the model for the API
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # 3. Save a sample of the training data: monitoring compares live inputs
    #    against this to detect drift. It must match the data THIS model learned from.
    make_reference_sample(train).to_csv(REFERENCE_PATH, index=False)

    # 4. Metadata: lets anyone trace the production model back to its MLflow run
    METADATA_PATH.write_text(
        json.dumps(
            {
                "model_version": version,
                "app_version": APP_VERSION,  # which release of the code trained it
                "model_type": model_name,
                "mlflow_run_id": run_id,
                "trained_at": datetime.now(UTC).isoformat(),
                "training_rows": len(train),
                "reason": reason,
                # The API must load the model with this exact scikit-learn version
                "sklearn_version": sklearn.__version__,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
