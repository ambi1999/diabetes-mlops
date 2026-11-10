"""Training pipeline: orchestrated with Prefect, tracked with MLflow.

    uv run pipeline.py                          # run once (full profile)
    uv run --env-file quick.env pipeline.py     # quick run: small sample, isolated outputs, seconds
    uv run pipeline.py --serve   # run on the schedule in config.TRAINING_SCHEDULE_CRON

Flow:
  1. Data:       prepare -> load -> validate
  2. Selection:  cross-validate candidate models -> diagnose bias/variance -> pick best
  3. Training:   retrain the best candidate on all training data
  4. Evaluation: score it on the evaluation set (never used for selection)
  5. Release:    register in MLflow -> quality gate -> promote ONLY if the model
                 passes the gate AND beats the production model

Design: Prefect TASKS do the computation (each one calls a function from src/).
MLflow logging happens in the FLOW body, so all logging for one run happens in
one place, in order. The real logic lives in src/, so this file only orchestrates."""

import argparse
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import sklearn
from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from src.config import (
    CANDIDATE_MODELS,
    DATA_SAMPLE_ROWS,
    MODELS_DIR,
    QUICK,
    RUN_PROFILE,
    SELECTION_METRIC,
    TRAIN_DATA,
    TRAINING_SCHEDULE_CRON,
    config_snapshot,
)
from src.data import (
    download_raw_data,
    load_training_and_evaluation_data,
    save_splits,
    split_data,
    stratified_sample,
    validate_data,
)
from src.evaluation import (
    cross_validate_candidates,
    evaluate_model,
    format_results_table,
    plot_learning_curve,
    select_best_model,
)
from src.models import build_model
from src.quality_gate import run_quality_gate
from src.registry import (
    is_better,
    log_and_register_model,
    make_reference_sample,
    promote_model,
    score_production_model,
    setup_mlflow,
)
from src.schema import FEATURES, TARGET
from src.timing import StepTimer

# Tasks that receive DataFrames or models use NO_CACHE: Prefect would otherwise
# try to hash those large inputs to build a cache key, which is slow and unneeded.


# =============================================================================
# Tasks
# =============================================================================
@task(retries=2, retry_delay_seconds=10)  # downloads can fail transiently: retry
def prepare_data() -> None:
    """Download and split the data, only on the very first run of each profile.

    A quick run (RUN_PROFILE=quick) first takes a stratified sample of
    DATA_SAMPLE_ROWS rows; a full run uses every row."""
    if TRAIN_DATA.exists():
        return
    raw = download_raw_data()  # shared by both profiles: downloaded only once
    errors = validate_data(raw)
    if errors:
        raise ValueError(f"Raw data failed validation: {errors}")
    if DATA_SAMPLE_ROWS:
        raw = stratified_sample(raw, DATA_SAMPLE_ROWS)
    save_splits(*split_data(raw))


@task
def load_data(include_new_labels: bool) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    return load_training_and_evaluation_data(include_new_labels)


@task(cache_policy=NO_CACHE)
def validate(df: pd.DataFrame, name: str) -> None:
    """Stop the pipeline before training if the data breaks the contract."""
    errors = validate_data(df)
    if errors:
        raise ValueError(f"{name} data failed validation: {errors}")


@task(cache_policy=NO_CACHE)
def compare_candidates(train: pd.DataFrame, sample_size: int | None, folds: int | None) -> pd.DataFrame:
    return cross_validate_candidates(train, sample_size=sample_size, folds=folds)


@task(cache_policy=NO_CACHE)
def make_learning_curve(
    model_name: str, train: pd.DataFrame, output_dir: Path, sample_size: int | None, folds: int | None
) -> Path:
    return plot_learning_curve(
        model_name, train, output_dir / "learning_curve.png", sample_size=sample_size, folds=folds
    )


@task(cache_policy=NO_CACHE)
def train_final_model(model_name: str, train: pd.DataFrame):
    """Cross-validation only CHOSE the model type. Now train that type on all
    training rows, because more data generally means a better model."""
    model = build_model(model_name)
    model.fit(train[FEATURES], train[TARGET])
    return model


@task(cache_policy=NO_CACHE)
def evaluate(model, evaluation: pd.DataFrame) -> dict:
    return evaluate_model(model, evaluation)


@task(cache_policy=NO_CACHE)
def score_champion(evaluation: pd.DataFrame) -> dict | None:
    return score_production_model(evaluation)


# =============================================================================
# MLflow logging helpers
# =============================================================================
def log_candidate_results(results: pd.DataFrame) -> None:
    """One nested MLflow run per candidate, so they can be compared side by side
    in the MLflow UI (select the runs -> Compare)."""
    for row in results.itertuples():
        with mlflow.start_run(run_name=f"cv-{row.model}", nested=True):
            mlflow.log_param("model", row.model)
            mlflow.log_params(CANDIDATE_MODELS[row.model])  # this candidate's hyperparameters
            mlflow.log_metrics(
                {
                    "cv_train_score": row.train_score,
                    "cv_val_score": row.val_score,
                    "cv_val_std": row.val_std,
                    "cv_gap": row.gap,
                    "cv_fit_seconds": row.fit_seconds,  # average training time per fold: spot slow models
                }
            )
            mlflow.set_tag("diagnosis", row.diagnosis)
    mlflow.log_table(results, "candidate_comparison.json")


# =============================================================================
# Flow
# =============================================================================
@flow(name="diabetes-training")
def training_flow(
    include_new_labels: bool = True,
    reason: str = "manual",
    cv_sample_size: int | None = None,
    cv_folds: int | None = None,
) -> dict:
    """Train, evaluate and (maybe) promote a model. Returns a summary of what happened.

    cv_sample_size / cv_folds: optional overrides for model comparison speed.
    Left as None, the values in src/config.py are used (normal and scheduled
    runs). The classroom demo passes smaller values so it runs in seconds."""
    timer = StepTimer()  # times each step; printed at the end and logged to MLflow
    print(
        f"Run profile: {RUN_PROFILE.upper()}"
        + (
            f" (a {DATA_SAMPLE_ROWS:,}-row sample; outputs go to {MODELS_DIR.name}/, never to production)"
            if QUICK
            else ""
        )
    )

    # --- 1. Data --------------------------------------------------------------
    with timer.step("prepare data"):
        prepare_data()
    with timer.step("load and validate"):
        train, evaluation, eval_name = load_data(include_new_labels)
        validate(train, "training")
        validate(evaluation, "evaluation")

    setup_mlflow()
    with mlflow.start_run(run_name=f"train-{reason}") as run, tempfile.TemporaryDirectory() as tmp:
        # Reproducibility: record every setting used for this run (including RUN_PROFILE)
        mlflow.log_dict(config_snapshot(), "config.json")
        mlflow.log_params(
            {
                "reason": reason,
                "run_profile": RUN_PROFILE,
                "training_rows": len(train),
                "eval_set": eval_name,
                # Record the speed overrides too, so a demo run is never mistaken for a full one
                "cv_sample_size_override": cv_sample_size,
                "cv_folds_override": cv_folds,
            }
        )

        # --- 2. Model selection (cross-validation on training data only) ------
        with timer.step("compare models"):
            results = compare_candidates(train, cv_sample_size, cv_folds)
            log_candidate_results(results)
        print("\nCandidate comparison (cross-validated):\n" + format_results_table(results) + "\n")

        best_name = select_best_model(results)
        mlflow.log_param("selected_model", best_name)
        with timer.step("learning curve"):
            mlflow.log_artifact(str(make_learning_curve(best_name, train, Path(tmp), cv_sample_size, cv_folds)))
        print(f"Selected: {best_name}")

        # --- 3 & 4. Train the winner, evaluate on unseen data -----------------
        with timer.step("train final model"):
            model = train_final_model(best_name, train)
        with timer.step("evaluate"):
            metrics = evaluate(model, evaluation)
        mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items()})
        with timer.step("log and register (MLflow)"):
            version = log_and_register_model(model, train)
        print(f"Challenger (v{version}) on {eval_name}: {metrics}")

        # --- 5. Champion vs. challenger ---------------------------------------
        with timer.step("score champion"):
            champion = score_champion(evaluation)
        if champion:
            mlflow.log_metrics({f"champion_{k}": v for k, v in champion.items()})
            print(f"Champion on {eval_name}:          {champion}")

        # --- 6. Quality gate: an ABSOLUTE minimum standard (src/quality_gate.py).
        # The same checks run in CI before deployment, so "good enough to
        # promote" and "good enough to deploy" always mean the same thing.
        with timer.step("quality gate"):
            gate = run_quality_gate(
                model, {"sklearn_version": sklearn.__version__, "metrics": metrics}, make_reference_sample(train)
            )
        mlflow.log_dict(gate.as_dict(), "quality_gate.json")
        mlflow.set_tag("quality_gate_passed", str(gate.passed))
        print("\n" + gate.as_text())

        # Promote only if the challenger is good enough AND better than the champion
        beats_champion = is_better(metrics, champion, SELECTION_METRIC)
        promoted = gate.passed and beats_champion
        mlflow.set_tag("promoted", str(promoted))
        target = "the quick model folder" if QUICK else "production"
        if promoted:
            with timer.step("promote"):
                promote_model(model, version, run.info.run_id, best_name, metrics, train, reason)
            print(f"Promoted version {version} ({best_name}) to {target} ({MODELS_DIR.name}/).")
        elif not gate.passed:
            print("Challenger failed the quality gate. Keeping the current model.")
        else:
            print("Challenger is not better. Keeping the current model.")

        # CONCEPT - measure before optimizing: knowing where the time goes shows
        # which step is worth speeding up (often not the one you'd guess).
        mlflow.log_metrics(timer.as_metrics())
        print("\n" + timer.summary())

    # A plain summary (JSON-friendly) so callers - like the demo notebook - can
    # show what happened without digging through MLflow.
    return {
        "promoted": promoted,
        "quality_gate_passed": gate.passed,
        "beats_champion": beats_champion,
        "selected_model": best_name,
        "model_version": version,
        "eval_set": eval_name,
        "challenger_metrics": metrics,
        "champion_metrics": champion,
        "candidates": results.to_dict(orient="records"),
        "step_seconds": timer.seconds,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serve", action="store_true", help="run on a schedule instead of once")
    args = parser.parse_args()

    if args.serve:
        training_flow.serve(name="scheduled-retrain", cron=TRAINING_SCHEDULE_CRON, parameters={"reason": "scheduled"})
    else:
        training_flow()
