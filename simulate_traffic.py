"""Pretend to be real users: replay rows from data/stream.csv through the API.

    uv run simulate_traffic.py --n 2000           # normal traffic
    uv run simulate_traffic.py --n 2000 --drift   # a population that has changed
    uv run --env-file quick.env simulate_traffic.py --drift   # quick profile (500 users by default)

CONCEPT - delayed labels: in real life you get a person's inputs immediately,
but whether they actually have diabetes is only known later (if ever). Here,
after the predictions are made, the true labels "arrive" and are appended to
data/new_labeled.csv, which the training pipeline can then learn from."""

import argparse

import pandas as pd
import requests

from src.config import (
    DRIFT_AGE_SHIFT,
    DRIFT_BMI_SHIFT,
    NEW_LABELS,
    SIMULATED_BATCH_SIZE,
    STREAM_DATA,
    STREAM_OFFSET,
)
from src.schema import FEATURE_RANGES, FEATURES


def apply_drift(batch: pd.DataFrame) -> pd.DataFrame:
    """Simulate a heavier, older population (amounts set in config)."""
    batch = batch.copy()
    batch["BMI"] = (batch["BMI"] + DRIFT_BMI_SHIFT).clip(*FEATURE_RANGES["BMI"])
    batch["Age"] = (batch["Age"] + DRIFT_AGE_SHIFT).clip(*FEATURE_RANGES["Age"])
    return batch


def next_batch(size: int) -> tuple[pd.DataFrame, int]:
    """Take the next unused rows from the stream, so users are never replayed twice."""
    offset = int(STREAM_OFFSET.read_text(encoding="utf-8")) if STREAM_OFFSET.exists() else 0
    return pd.read_csv(STREAM_DATA).iloc[offset : offset + size], offset


def send_requests(batch: pd.DataFrame, api_url: str) -> None:
    with requests.Session() as session:  # reuses one connection: much faster
        for i, row in enumerate(batch[FEATURES].to_dict(orient="records"), start=1):
            payload = {k: (float(v) if k == "BMI" else int(v)) for k, v in row.items()}
            session.post(f"{api_url}/predict", json=payload, timeout=10).raise_for_status()
            if i % 500 == 0:
                print(f"  sent {i}/{len(batch)}")


def simulate(n: int, drift: bool = False, api_url: str = "http://localhost:8000") -> dict:
    """Replay the next n users through the API, then let their labels "arrive".

    Reusable from other code (the demo notebook calls it), not just the command line.
    Returns a small summary of what was sent."""
    batch, offset = next_batch(n)
    if batch.empty:
        raise RuntimeError(
            "No stream data left. Run the demo reset (or delete "
            "data/stream_offset.txt and data/new_labeled.csv) to start over."
        )
    if drift:
        batch = apply_drift(batch)

    send_requests(batch, api_url)

    # Labels "arrive" after the predictions were made
    batch.to_csv(NEW_LABELS, mode="a", header=not NEW_LABELS.exists(), index=False)
    STREAM_OFFSET.write_text(str(offset + len(batch)), encoding="utf-8")
    return {
        "users_sent": len(batch),
        "drift": drift,
        "average_bmi": round(float(batch["BMI"].mean()), 1),
        "average_age_group": round(float(batch["Age"].mean()), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--n",
        type=int,
        default=SIMULATED_BATCH_SIZE,
        help=f"number of users to simulate (default {SIMULATED_BATCH_SIZE}, set per profile in src/config.py)",
    )
    parser.add_argument("--drift", action="store_true", help="shift the population")
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()

    summary = simulate(args.n, args.drift, args.api_url)
    print(f"Sent {summary['users_sent']} requests (drift={args.drift}). Labels saved to {NEW_LABELS.name}.")


if __name__ == "__main__":
    main()
