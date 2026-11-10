"""MLOps demo steps for demo/mlops_demo.ipynb, plus the demo's snapshot and reset.

What this file is:
  Each demo step is a small function that calls the REAL project code
  (src/, pipeline.py, monitor.py, simulate_traffic.py - nothing is faked) and
  returns a JSON-friendly dict, so the result can be saved as a fallback.
  Shared machinery (modes, fallback, display) lives in demo/common.py.

Command line (handy before and after a demo):
  uv run python -m demo.mlops_helpers snapshot   # save the current model as the baseline
  uv run python -m demo.mlops_helpers reset      # return to a clean starting point

Functions marked "pragma: no cover" need live services (the API, the Prefect
server) or the real downloaded data. They are exercised by rehearsing the demo,
not by unit tests, so they're excluded from the coverage measurement.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from demo.common import DEMO_DIR, FALLBACK_ROOT, missing_fallbacks, print_checks, reachable, rel
from src.config import (
    DEMO_BATCH_SIZE,
    DEMO_CV_FOLDS,
    DEMO_CV_SAMPLE_SIZE,
    METADATA_PATH,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    MODELS_DIR,
    NEW_LABELS,
    PREDICTION_DB,
    PRODUCTION_ALIAS,
    PSI_THRESHOLD,
    RAW_DATA,
    REGISTERED_MODEL_NAME,
    REPORTS_DIR,
    STREAM_OFFSET,
    TRAIN_DATA,
)
from src.schema import EXAMPLE_INPUT, TARGET

BASELINE_FILES = ("model.joblib", "metadata.json", "reference_sample.csv")
FALLBACK_DIR = FALLBACK_ROOT / "mlops"
# The steps that have a saved rehearsal result (one JSON file each)
FALLBACK_STEPS = ("data_overview", "validation", "compare_models", "prediction", "normal_traffic", "drifted_traffic")


# =============================================================================
# Paths the reset is allowed to touch
# =============================================================================
@dataclass(frozen=True)
class DemoPaths:
    """Every file or folder the reset may change, in one place.

    CONCEPT - least privilege: the reset can ONLY touch what is listed here.
    Downloaded data, MLflow history and code are deliberately absent, which
    is what makes the reset safe to run again and again. Tests build this for
    a temporary folder, so they never touch your real project."""

    models_dir: Path
    baseline_dir: Path
    fallback_dir: Path
    reports_dir: Path
    stream_offset: Path
    new_labels: Path
    prediction_db: Path

    @classmethod
    def default(cls) -> DemoPaths:
        return cls(
            models_dir=MODELS_DIR,
            baseline_dir=DEMO_DIR / "baseline",
            fallback_dir=FALLBACK_DIR,
            reports_dir=REPORTS_DIR,
            stream_offset=STREAM_OFFSET,
            new_labels=NEW_LABELS,
            prediction_db=PREDICTION_DB,
        )

    @classmethod
    def for_root(cls, root: Path) -> DemoPaths:
        """The same layout under another folder (used by tests)."""
        return cls(
            models_dir=root / "models",
            baseline_dir=root / "demo" / "baseline",
            fallback_dir=root / "demo" / "fallback" / "mlops",
            reports_dir=root / "reports",
            stream_offset=root / "data" / "stream_offset.txt",
            new_labels=root / "data" / "new_labeled.csv",
            prediction_db=root / "logs" / "predictions.db",
        )


# =============================================================================
# Pre-flight checks
# =============================================================================
def preflight(api_url: str, prefect_api_url: str, paths: DemoPaths | None = None) -> bool:  # pragma: no cover
    """Check everything the MLOps demo needs; print a ✅/❌ list with fixes.

    Args:
        api_url: the local API, e.g. http://localhost:8000.
        prefect_api_url: the local Prefect server's API, e.g. http://127.0.0.1:4200/api.
        paths: the demo's folders (defaults to the real project).
    """
    paths = paths or DemoPaths.default()
    missing = missing_fallbacks(paths.fallback_dir, FALLBACK_STEPS)
    return print_checks(
        [
            ("Training data downloaded", TRAIN_DATA.exists(), "run: uv run pipeline.py"),
            ("Production model present", MODEL_PATH.exists(), "run: uv run pipeline.py"),
            (
                "Baseline model saved",
                (paths.baseline_dir / "model.joblib").exists(),
                "run: uv run python -m demo.mlops_helpers snapshot",
            ),
            (
                "Fallback results saved",
                not missing,
                f"rehearse once with MODE = 'record' (missing: {', '.join(missing)})",
            ),
            ("API running", reachable(f"{api_url}/health"), "run: docker compose up"),
            ("Prefect server running", reachable(f"{prefect_api_url}/health"), "run: uv run prefect server start"),
        ]
    )


# =============================================================================
# Demo steps (each returns a JSON-friendly dict, so it can be saved as a fallback)
# =============================================================================
def data_overview() -> dict:  # pragma: no cover
    raw = pd.read_csv(RAW_DATA)
    return {"people": len(raw), "features": raw.shape[1] - 1, "diabetes_rate": round(float(raw[TARGET].mean()), 3)}


def validation_example() -> dict:  # pragma: no cover
    """Validate a few clean rows, then the same rows with one impossible BMI."""
    from src.data import validate_data

    sample = pd.read_csv(TRAIN_DATA, nrows=200)
    broken = sample.copy()
    broken.loc[0, "BMI"] = 500
    return {"clean_errors": validate_data(sample), "broken_errors": validate_data(broken)}


def compare_models() -> dict:  # pragma: no cover
    """Race all candidate models with the fast demo setting; draw the winner's learning curve.
    Uses the same functions as the training pipeline."""
    from src.evaluation import cross_validate_candidates, plot_learning_curve, select_best_model

    train = pd.read_csv(TRAIN_DATA)
    results = cross_validate_candidates(train, sample_size=DEMO_CV_SAMPLE_SIZE, folds=DEMO_CV_FOLDS)
    best = select_best_model(results)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    image = plot_learning_curve(
        best, train, REPORTS_DIR / "demo_learning_curve.png", sample_size=DEMO_CV_SAMPLE_SIZE, folds=DEMO_CV_FOLDS
    )
    table = results[["model", "train_score", "val_score", "gap", "diagnosis"]].round(3)
    return {"table": table.to_dict(orient="records"), "best": best, "image_path": rel(image)}


def production_model() -> dict:  # pragma: no cover
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {
        "version": meta.get("model_version"),
        "model_type": meta.get("model_type"),
        "trained_at": meta.get("trained_at", "")[:16].replace("T", " "),
        "reason": meta.get("reason"),
        "roc_auc": round(meta.get("metrics", {}).get("roc_auc", float("nan")), 3),
    }


def predict(api_url: str, person: dict | None = None) -> dict:  # pragma: no cover
    """Ask the running API for a prediction, exactly as the web app does."""
    good = requests.post(f"{api_url}/predict", json=person or EXAMPLE_INPUT, timeout=30)
    good.raise_for_status()
    # Now send impossible values: the API must refuse them
    bad = requests.post(f"{api_url}/predict", json={**EXAMPLE_INPUT, "BMI": 500, "Age": 40}, timeout=30)
    problems = [f"{d['loc'][-1]}: {d['msg']}" for d in bad.json().get("detail", [])]
    return {"prediction": good.json(), "invalid_status": bad.status_code, "invalid_problems": problems}


def traffic_and_monitoring(api_url: str, drift: bool) -> dict:  # pragma: no cover
    """Send a batch of simulated users through the API, then run the REAL Prefect
    monitoring flow. With drift, monitoring triggers the REAL training flow."""
    from monitor import monitoring_flow
    from simulate_traffic import simulate

    sent = simulate(DEMO_BATCH_SIZE, drift=drift, api_url=api_url)
    result = monitoring_flow(
        window=DEMO_BATCH_SIZE, allow_retrain=drift, cv_sample_size=DEMO_CV_SAMPLE_SIZE, cv_folds=DEMO_CV_FOLDS
    )
    top = sorted(result.get("psi", {}).items(), key=lambda item: -item[1])[:5]
    retraining = result.get("retraining")
    summary = {
        "sent": sent,
        "top_psi": [{"feature": f, "psi": s} for f, s in top],
        "drift_detected": result.get("drift_detected", False),
        "drifted_features": list(result.get("drifted_features", {})),
        "retraining": None,
    }
    if retraining:
        champion = retraining.get("champion_metrics") or {}
        challenger = retraining["challenger_metrics"]
        summary["retraining"] = {
            "promoted": retraining["promoted"],
            "selected_model": retraining["selected_model"],
            "new_version": retraining["model_version"],
            "challenger_roc_auc": round(challenger["roc_auc"], 4),
            "champion_roc_auc": round(champion.get("roc_auc", float("nan")), 4),
            "challenger_brier": round(challenger["brier"], 4),
            "champion_brier": round(champion.get("brier", float("nan")), 4),
        }
    return summary


# =============================================================================
# Display helpers (plain-language output for beginners)
# =============================================================================
def print_drift(summary: dict) -> None:
    sent = summary["sent"]
    print(
        f"Sent {sent['users_sent']} simulated users "
        f"(average BMI {sent['average_bmi']}, average age group {sent['average_age_group']}).\n"
    )
    print("Biggest changes compared with the training data (PSI):")
    for row in summary["top_psi"]:
        # 0.1 is the usual "some change" line; PSI_THRESHOLD (0.25) means real drift
        if row["psi"] > PSI_THRESHOLD:
            flag = "🔴 drift"
        else:
            flag = "🟡 some change" if row["psi"] > 0.1 else "🟢 stable"
        print(f"  {row['feature']:22s} {row['psi']:.3f}  {flag}")
    if summary["drift_detected"]:
        print(f"\n🚨 DRIFT DETECTED in: {', '.join(summary['drifted_features'])}")
    else:
        print("\n✅ No significant drift - the model is seeing the kind of people it learned from.")


def explain_retraining(summary: dict) -> None:
    r = summary.get("retraining")
    if not r:
        print("Retraining did not run (it needs drift AND enough newly labeled data).")
        return
    print(
        f"Challenger (new model, v{r['new_version']}, {r['selected_model']}): "
        f"ROC-AUC {r['challenger_roc_auc']}, Brier {r['challenger_brier']}"
    )
    print(
        f"Champion   (current production model):          ROC-AUC {r['champion_roc_auc']}, Brier {r['champion_brier']}"
    )
    if r["promoted"]:
        print("\n🏆 The challenger won and is now in production. The API picked it up automatically.")
    else:
        print("\n🛡️ The champion stays. Drift was a WARNING, not proof the model got worse:")
        print("   the new model didn't rank people better, so replacing the current one isn't justified.")


# =============================================================================
# Baseline snapshot and reset
# =============================================================================
def snapshot_baseline(paths: DemoPaths | None = None, force: bool = False) -> str:
    """Save the current production model as the demo's starting point.

    Refuses to overwrite an existing baseline unless force=True, so a model
    promoted DURING a demo can't accidentally become the new baseline."""
    paths = paths or DemoPaths.default()
    if (paths.baseline_dir / "model.joblib").exists() and not force:
        raise FileExistsError("A baseline already exists. Use force=True (or --force) to replace it.")
    paths.baseline_dir.mkdir(parents=True, exist_ok=True)
    for name in BASELINE_FILES:
        shutil.copy(paths.models_dir / name, paths.baseline_dir / name)
    version = json.loads((paths.baseline_dir / "metadata.json").read_text(encoding="utf-8")).get("model_version")
    print(f"Saved model version {version} as the demo baseline in {rel(paths.baseline_dir)}/")
    return str(version)


def reset_demo(paths: DemoPaths | None = None, restore_alias: bool = True) -> list[str]:
    """Return the demo to a clean starting point. Safe to run any number of times.

    Clears: simulated-user progress, newly arrived labels, prediction logs and
    drift reports. Restores: the baseline production model and (optionally)
    the MLflow "production" alias. Never touches: downloaded data, MLflow run
    history, or code."""
    paths = paths or DemoPaths.default()
    actions = []

    # 1. Simulated users, their labels and the API's prediction logs
    #    (the glob also catches SQLite's temporary "-journal" file, if any)
    logs = paths.prediction_db.parent.glob(paths.prediction_db.name + "*")
    leftovers = [paths.stream_offset, paths.new_labels, *logs]
    for f in leftovers:
        if f.exists():
            f.unlink()
            actions.append(f"removed {rel(f)}")

    # 2. Drift reports and demo charts (only files the demo/monitoring created)
    if paths.reports_dir.exists():
        for f in [*paths.reports_dir.glob("drift_*"), *paths.reports_dir.glob("demo_*")]:
            f.unlink()
            actions.append(f"removed {rel(f)}")

    # 3. Restore the baseline model. shutil.copy gives the files a new timestamp,
    #    which is how the running API notices the change and reloads the model.
    if not (paths.baseline_dir / "model.joblib").exists():
        raise FileNotFoundError("No baseline yet. Run: uv run python -m demo.demo_helpers snapshot")
    for name in BASELINE_FILES:
        shutil.copy(paths.baseline_dir / name, paths.models_dir / name)
    version = json.loads((paths.baseline_dir / "metadata.json").read_text(encoding="utf-8")).get("model_version")
    actions.append(f"restored baseline model v{version}")

    # 4. Point the registry's "production" alias back at the baseline version
    if restore_alias:
        try:
            from mlflow.tracking import MlflowClient

            MlflowClient(MLFLOW_TRACKING_URI).set_registered_model_alias(
                REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, str(version)
            )
            actions.append(f"MLflow '{PRODUCTION_ALIAS}' alias -> v{version}")
        # Deliberately broad ("noqa: BLE001" tells ruff it's intended): MLflow may be
        # missing, unreachable or raise its own errors, and none of that may stop the
        # reset - the demo works without the alias, so we report it and move on.
        except Exception as exc:  # noqa: BLE001
            actions.append(f"could not reset MLflow alias ({type(exc).__name__}); the demo still works")

    for action in actions:
        print("•", action)
    print("✅ Demo reset complete.")
    return actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps demo utilities: snapshot the baseline or reset the demo.")
    parser.add_argument(
        "command",
        choices=["snapshot", "reset"],
        help="snapshot = save the current model as the demo baseline; reset = clean start",
    )
    parser.add_argument("--force", action="store_true", help="snapshot: replace an existing baseline")
    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot_baseline(force=args.force)
    else:
        reset_demo()
