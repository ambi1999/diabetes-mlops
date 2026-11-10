"""Monitoring flow: is the model still seeing the kind of people it was trained on?

    uv run monitor.py               # check once (and retrain if needed)
    uv run monitor.py --no-retrain  # check only
    uv run monitor.py --serve       # check every config.MONITOR_INTERVAL_SECONDS
    uv run --env-file quick.env monitor.py   # quick profile: smaller window, its own logs and model

CONCEPT - closing the loop: drift detected AND enough new labeled data
-> trigger the training pipeline. Drift alone can't trigger retraining, because
you can't train on data without labels. And the pipeline still promotes the
new model only if it beats the current one, because drift is a warning, not
proof that the model got worse."""

import argparse
import json
from datetime import UTC, datetime

import pandas as pd
from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from pipeline import training_flow
from src.config import (
    MIN_NEW_LABELS_FOR_RETRAIN,
    MIN_PREDICTIONS_FOR_DRIFT_CHECK,
    MONITOR_INTERVAL_SECONDS,
    MONITOR_WINDOW,
    NEW_LABELS,
    PREDICTION_DB,
    REFERENCE_PATH,
    REPORTS_DIR,
)
from src.drift import detect_drift
from src.prediction_log import load_recent_inputs


@task
def load_current_inputs(window: int) -> pd.DataFrame:
    return load_recent_inputs(PREDICTION_DB, window)


@task(cache_policy=NO_CACHE)
def save_reports(reference: pd.DataFrame, current: pd.DataFrame, result: dict) -> None:
    """JSON report (for software and history) + optional HTML report (for people)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    (REPORTS_DIR / f"drift_{stamp}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Our own PSI check makes the decision; Evidently just draws a visual report.
    # If Evidently's API changes in a future version, monitoring still works.
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report([DataDriftPreset()])
        report.run(current_data=current, reference_data=reference).save_html(str(REPORTS_DIR / f"drift_{stamp}.html"))
    # A deliberately broad catch (hence "noqa: BLE001" to tell ruff it's intended):
    # the HTML report is an optional extra, and NO failure in it (missing package,
    # changed API, a bug inside Evidently) may stop the drift check itself.
    except Exception as exc:  # noqa: BLE001
        print(f"(Skipped Evidently HTML report: {exc})")


def count_new_labels() -> int:
    return len(pd.read_csv(NEW_LABELS)) if NEW_LABELS.exists() else 0


@flow(name="diabetes-monitoring")
def monitoring_flow(
    window: int = MONITOR_WINDOW,
    allow_retrain: bool = True,
    cv_sample_size: int | None = None,
    cv_folds: int | None = None,
) -> dict:
    """Check recent predictions for drift; retrain if needed. Returns a summary.

    window: how many of the most recent predictions to check.
    cv_sample_size / cv_folds: optional speed overrides, passed on to the
    training flow if it runs (None = use src/config.py, as in normal runs)."""
    current = load_current_inputs(window)
    if len(current) < MIN_PREDICTIONS_FOR_DRIFT_CHECK:
        # Too few rows make PSI noisy and cause false alarms
        print(f"Only {len(current)} predictions logged; need {MIN_PREDICTIONS_FOR_DRIFT_CHECK} to check drift.")
        return {"checked": False, "predictions": len(current)}

    # Reference = a sample of the data the CURRENT production model was trained on
    reference = pd.read_csv(REFERENCE_PATH)
    result = detect_drift(reference, current)
    result["checked"] = True
    result["retraining"] = None  # filled in below if retraining runs
    save_reports(reference, current, result)

    print("PSI per feature (top 5):")
    for name, score in sorted(result["psi"].items(), key=lambda item: -item[1])[:5]:
        print(f"  {name:22s} {score:.3f}")

    if not result["drift_detected"]:
        print("No significant drift. Nothing to do.")
        return result

    print(f"DRIFT DETECTED in: {list(result['drifted_features'])}")
    new_labels = count_new_labels()
    result["new_labels"] = new_labels
    if new_labels < MIN_NEW_LABELS_FOR_RETRAIN:
        print(f"Only {new_labels} new labels; waiting for {MIN_NEW_LABELS_FOR_RETRAIN} before retraining.")
    elif allow_retrain:
        print("Triggering retraining…")
        result["retraining"] = training_flow(
            include_new_labels=True,
            reason="drift",
            cv_sample_size=cv_sample_size,
            cv_folds=cv_folds,
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serve", action="store_true", help="run on a schedule instead of once")
    parser.add_argument("--no-retrain", action="store_true", help="only report drift")
    args = parser.parse_args()

    if args.serve:
        monitoring_flow.serve(name="scheduled-drift-check", interval=MONITOR_INTERVAL_SECONDS)
    else:
        monitoring_flow(allow_retrain=not args.no_retrain)
