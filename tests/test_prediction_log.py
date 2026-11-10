"""Prediction log tests: what the API writes, monitoring can read back.

Uses pytest's tmp_path (a temporary folder per test), so tests never touch the
real logs/ folder."""

from src.prediction_log import load_recent_inputs, log_prediction
from src.schema import FEATURES


def test_logged_predictions_can_be_read_back(tmp_path, dataset):
    db = tmp_path / "predictions.db"
    rows = dataset[FEATURES].head(3).to_dict(orient="records")
    for row in rows:
        log_prediction(db, {k: float(v) for k, v in row.items()}, 0.2, "1")

    recent = load_recent_inputs(db, limit=2)
    assert len(recent) == 2
    assert list(recent.columns) == FEATURES


def test_missing_db_returns_empty(tmp_path):
    assert load_recent_inputs(tmp_path / "nope.db", limit=10).empty
