"""Shared test helpers (pytest loads this file automatically).

CONCEPT - fast, self-contained tests: tests use small SYNTHETIC datasets and
temporary folders, so they run in seconds, need no internet or downloaded
data, and never touch your real project files.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn

from src.schema import FEATURE_RANGES, FEATURES, TARGET


def make_dataset(n: int = 2000, bmi_shift: float = 0, seed: int = 0, signal: float = 1.0) -> pd.DataFrame:
    """Random but valid data where higher BMI and blood pressure raise the risk.

    Args:
        n: number of rows.
        bmi_shift: added to every BMI (simulates drift).
        seed: random seed, so the same arguments always give the same data.
        signal: how strongly the features predict the label (higher = easier
            to learn, so models score higher).
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.integers(low, high + 1, n) for col, (low, high) in FEATURE_RANGES.items()})
    df["BMI"] = np.clip(rng.normal(28 + bmi_shift, 6, n).round(), 12, 98)
    logit = -2.5 + signal * (0.1 * (df["BMI"] - 28) + 1.0 * df["HighBP"] + 0.3 * (df["Age"] - 7))
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


@pytest.fixture
def dataset() -> pd.DataFrame:
    return make_dataset()


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """The REAL API (api/main.py), serving a small model trained on synthetic data.

    The API's file locations are pointed at a temporary folder, so the tests
    work without a trained production model and never touch real logs.
    """
    from fastapi.testclient import TestClient

    from api import main
    from src.models import build_model

    data = make_dataset(3000, signal=2.0)
    model = build_model("logistic_regression").fit(data[FEATURES], data[TARGET])
    joblib.dump(model, tmp_path / "model.joblib")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"model_version": "test-1", "sklearn_version": sklearn.__version__}), encoding="utf-8"
    )
    monkeypatch.setattr(main, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(main, "METADATA_PATH", tmp_path / "metadata.json")
    monkeypatch.setattr(main, "PREDICTION_DB", tmp_path / "predictions.db")
    monkeypatch.setattr(main, "store", main.ModelStore())  # forget any previously loaded model
    return TestClient(main.app)
