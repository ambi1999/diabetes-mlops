"""Prediction log: the API writes here, monitoring reads from here.

Keeping both sides in one module means the table format is defined once.
SQLite is a single local file: perfect for v1. In v2 this module can switch
to a hosted database (e.g. Postgres) without changing the API or monitoring."""

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.schema import FEATURES

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY,
    ts TEXT,               -- when the prediction was made (UTC)
    model_version TEXT,    -- which model made it
    features TEXT,         -- the inputs, as JSON (no names or contact details)
    probability REAL       -- the model's output
)
"""


def log_prediction(db_path: Path, features: dict, probability: float, model_version: str) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # closing(...) closes the database file afterwards; the inner "conn" block
    # commits the changes. (A plain `with sqlite3.connect(...)` only commits and
    # leaves the file open, which on Windows blocks deleting it, e.g. in the demo reset.)
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute(CREATE_TABLE)
        conn.execute(
            "INSERT INTO predictions (ts, model_version, features, probability) VALUES (?, ?, ?, ?)",
            (datetime.now(UTC).isoformat(), model_version, json.dumps(features), probability),
        )


def load_recent_inputs(db_path: Path, limit: int) -> pd.DataFrame:
    """The inputs of the most recent `limit` predictions, one column per feature."""
    if not db_path.exists():
        return pd.DataFrame(columns=FEATURES)
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute("SELECT features FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return pd.DataFrame([json.loads(r[0]) for r in rows], columns=FEATURES)
