"""Central configuration: every tunable PROJECT setting lives here.

What this file is:
  The single home for values you might change between runs: data split shares
  (e.g. the 40% "stream" share), model settings, thresholds for drift, the
  quality gate and the smoke test, and the demo's speed settings.

Why one config file?
  * Change a setting in one place instead of hunting through scripts.
  * Reproducibility: the pipeline logs every setting below to MLflow with each
    run, so you can always see exactly which settings produced which model.

"One home per kind of setting" (the project-wide rule):
  * Project settings (this file)            -> src/config.py
  * Fixed facts / the data contract          -> src/schema.py
  * Tool settings (ruff, pytest, coverage)   -> pyproject.toml
  * CI settings                              -> named variables at the top of each workflow file
  * Values only the deployment knows (URLs)  -> GitHub repository variables
  * Secrets (passwords, keys)                -> never in Git; GitHub secrets / environment variables
  * Math details nobody tunes (PSI epsilon)  -> next to the code that uses them

How to use it:
  from src.config import TRAIN_SHARE   # then use the name, never the raw number
"""

import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read_app_version() -> str:
    """The app's version, defined ONCE in pyproject.toml ([project] version).

    Reading it here (instead of typing it again) means the API, the model
    metadata and the release tag can never disagree about the version."""
    try:
        with open(ROOT / "pyproject.toml", "rb") as f:
            return tomllib.load(f)["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return "unknown"


# Semantic version "MAJOR.MINOR.PATCH": bump MAJOR for breaking changes,
# MINOR for new features, PATCH for fixes. Shown by the API's /health endpoint.
APP_VERSION = _read_app_version()

# =============================================================================
# Run profile: "full" (the real thing) or "quick" (a fast end-to-end check)
# =============================================================================
# CONCEPT - quick run: run the WHOLE pipeline on a small sample in seconds, to
# check everything works before a full run. Like a rehearsal before the show:
# it answers "does it work?", while the full run answers "how good is it?"
#
# How to choose: the RUN_PROFILE environment variable. quick.env sets it, so
#   uv run --env-file quick.env pipeline.py     -> quick
#   uv run pipeline.py                          -> full (the default)
# Quick runs are fully ISOLATED: their own data, model, logs and MLflow
# experiment, so a quick model can never overwrite, ship or deploy as the
# production model. They share only the downloaded raw dataset.
RUN_PROFILE = os.getenv("RUN_PROFILE", "full").strip().lower()
if RUN_PROFILE not in ("full", "quick"):
    raise ValueError(f"RUN_PROFILE must be 'full' or 'quick', not {RUN_PROFILE!r}")
QUICK = RUN_PROFILE == "quick"


def _by_profile(full, quick):
    """Pick a setting's value for the active profile.

    Both values sit side by side wherever a setting differs between profiles,
    so it's easy to see (and change) what a quick run does differently."""
    return quick if QUICK else full


# Rows of the raw dataset to use. None = all ~253,700 rows. A quick run takes a
# STRATIFIED random sample (the diabetes rate stays ~14%), using RANDOM_SEED, so
# every quick run uses exactly the same people. 20,000 is small enough to run in
# seconds, yet large enough for every model and step to behave realistically.
DATA_SAMPLE_ROWS = _by_profile(full=None, quick=20_000)

# =============================================================================
# Folders and files
# =============================================================================
DATA_DIR = ROOT / "data"  # raw + split data (NOT committed to Git)
LOGS_DIR = ROOT / "logs"  # prediction logs written by the API
# The raw download is shared by both profiles, so it's only downloaded once
RAW_DATA = DATA_DIR / "raw.csv"

# Everything below is separate per profile (quick runs live in their own folders)
PROFILE_DATA_DIR = _by_profile(full=DATA_DIR, quick=DATA_DIR / "quick")
# models/ holds the production model (committed to Git); models_quick/ is git-ignored
MODELS_DIR = _by_profile(full=ROOT / "models", quick=ROOT / "models_quick")
REPORTS_DIR = _by_profile(full=ROOT / "reports", quick=ROOT / "reports" / "quick")

TRAIN_DATA = PROFILE_DATA_DIR / "train.csv"  # what the first model learns from
TEST_DATA = PROFILE_DATA_DIR / "test.csv"  # fixed, untouched evaluation set
STREAM_DATA = PROFILE_DATA_DIR / "stream.csv"  # pretend "future users", replayed later
NEW_LABELS = PROFILE_DATA_DIR / "new_labeled.csv"  # labels that "arrive" after predictions
STREAM_OFFSET = PROFILE_DATA_DIR / "stream_offset.txt"

MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
REFERENCE_PATH = MODELS_DIR / "reference_sample.csv"  # training data snapshot for drift checks

# =============================================================================
# Environment-specific settings (override with environment variables)
# =============================================================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
_DEFAULT_PREDICTION_DB = LOGS_DIR / _by_profile(full="predictions.db", quick="predictions_quick.db")
PREDICTION_DB = Path(os.getenv("PREDICTION_DB", str(_DEFAULT_PREDICTION_DB)))

# =============================================================================
# Reproducibility
# =============================================================================
# One seed for every random operation (splits, CV folds, models, sampling),
# so the same code + same data always gives the same result.
RANDOM_SEED = 42

# =============================================================================
# Data split
# =============================================================================
# Applied ONCE on the first pipeline run. To re-split after changing these,
# delete data/train.csv, test.csv, stream.csv, stream_offset.txt, new_labeled.csv.
TRAIN_SHARE = 0.50  # used for model selection (cross-validation) and training
TEST_SHARE = 0.10  # final evaluation only; never used to choose a model
STREAM_SHARE = 0.40  # simulated future users (see simulate_traffic.py)

# =============================================================================
# Model selection: candidates, cross-validation, bias vs. variance
# =============================================================================
# Each candidate is a different point on the bias-variance spectrum, so the
# comparison teaches you what underfitting and overfitting look like.
# Model classes are mapped to these names in src/models.py.
# To skip a slow candidate, comment out its entry.
CANDIDATE_MODELS = {
    # Simple, linear: tends toward HIGH BIAS (can't capture curved relationships).
    # Knob: C (smaller = stronger regularization = more bias, less variance).
    "logistic_regression": {"C": 1.0, "max_iter": 1000},
    # Unlimited depth: memorizes the training data -> HIGH VARIANCE (overfitting).
    "decision_tree_deep": {"max_depth": None, "min_samples_leaf": 1},
    # Same algorithm, constrained ("pruned"): trades a little bias for much less variance.
    # Knobs: max_depth, min_samples_leaf.
    "decision_tree_pruned": {"max_depth": 8, "min_samples_leaf": 50},
    # Many small trees built one after another, each fixing the previous ones' errors.
    # Same family as XGBoost/LightGBM. Usually the best balance on tabular data.
    # Knobs: max_iter (number of trees), learning_rate, max_leaf_nodes.
    "gradient_boosting": {"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 31},
    # Predicts from the k most similar people in the training data.
    # Knob: n_neighbors. Small k -> high variance (follows noise); large k -> high bias (over-smooths).
    # Slowest candidate: every prediction compares against all training rows,
    # and the saved model contains the entire training set.
    "knn": {"n_neighbors": 50, "weights": "uniform", "n_jobs": -1},
    # Small neural network (multi-layer perceptron), from scikit-learn: no extra framework.
    # Knobs: hidden_layer_sizes (bigger = more variance), alpha (L2 regularization),
    # early_stopping (stops when an internal 10% validation split stops improving).
    "neural_network": {"hidden_layer_sizes": (32, 16), "alpha": 1e-3, "early_stopping": True, "max_iter": 200},
}

# How candidates are ranked AND how challenger vs. champion is decided (higher = better).
# Must be both a scikit-learn scoring name and a key returned by evaluation.evaluate_model().
SELECTION_METRIC = "roc_auc"
CV_FOLDS = _by_profile(full=5, quick=2)  # k in k-fold cross-validation
# Cross-validation and the learning curve use a random sample of the training rows
# to keep runs fast (k-NN is slow to score on large data). All candidates get the
# SAME sample, so the comparison stays fair, and the winner is still retrained on
# ALL training rows. Set to None to use every row (k-NN then takes several minutes).
CV_SAMPLE_SIZE = _by_profile(full=40_000, quick=5_000)

# Bias/variance diagnosis rules of thumb (dataset-dependent, adjust as you learn):
# - train score much higher than validation score -> overfitting (high variance)
VARIANCE_GAP_THRESHOLD = 0.03
# - validation score well below the best candidate's, with a small gap -> underfitting (high bias)
BIAS_SHORTFALL_THRESHOLD = 0.01

LEARNING_CURVE_SIZES = [0.1, 0.25, 0.5, 0.75, 1.0]  # shares of training data to try
LEARNING_CURVE_FOLDS = _by_profile(full=3, quick=2)

# =============================================================================
# Model registry (MLflow)
# =============================================================================
# Quick runs get their own experiment and registry entry, so they never mix
# with (or take the "production" label from) real models in MLflow.
EXPERIMENT_NAME = _by_profile(full="diabetes-risk", quick="diabetes-risk-quick")
REGISTERED_MODEL_NAME = _by_profile(full="diabetes-risk", quick="diabetes-risk-quick")
PRODUCTION_ALIAS = "production"
REFERENCE_SAMPLE_SIZE = 10_000  # rows saved as the "normal" baseline for drift checks

# =============================================================================
# Retraining
# =============================================================================
# When new labeled data exists, it is split by arrival order: the older part is
# added to training, the newest part becomes the evaluation set, because the most
# recent data best represents what the model will face next.
NEW_LABELS_TRAIN_SHARE = 0.8
MIN_RECENT_HOLDOUT_ROWS = 200  # below this, fall back to the fixed test set
# Retrain only once this many newly labeled rows have arrived
MIN_NEW_LABELS_FOR_RETRAIN = _by_profile(full=2000, quick=1000)
TRAINING_SCHEDULE_CRON = "0 3 1 * *"  # 03:00 on the 1st of every month

# =============================================================================
# Monitoring
# =============================================================================
PSI_THRESHOLD = 0.25  # PSI above this = significant drift for a feature
PSI_BINS = 10  # bins for continuous features like BMI
MONITOR_WINDOW = _by_profile(full=2000, quick=500)  # how many recent predictions to check
MIN_PREDICTIONS_FOR_DRIFT_CHECK = _by_profile(full=500, quick=200)  # fewer = noisy PSI, false alarms
MONITOR_INTERVAL_SECONDS = 3600  # how often the scheduled check runs

# =============================================================================
# Simulated users (simulate_traffic.py)
# =============================================================================
# Users sent per run of simulate_traffic.py (unless --n says otherwise).
# Two batches (normal, then drifted) must reach MIN_NEW_LABELS_FOR_RETRAIN.
SIMULATED_BATCH_SIZE = _by_profile(full=2000, quick=500)
# Simulated drift (simulate_traffic.py --drift):
DRIFT_BMI_SHIFT = 6  # add to every user's BMI
DRIFT_AGE_SHIFT = 2  # add to every user's age group (each group = 5 years)

# =============================================================================
# Model quality gate (src/quality_gate.py)
# =============================================================================
# CONCEPT - quality gate: a minimum standard a model must meet before it may
# serve users, checked by the pipeline before promotion AND by CI before
# deployment. Like a safety inspection: passing it doesn't make a model great,
# failing it means the model must not ship.
#
# Minimum ROC-AUC (how well the model ranks people by risk; 0.5 = coin flip,
# 1.0 = perfect). Good models on this dataset score about 0.82. Raising this
# makes the gate stricter; setting it near 0.5 would let useless models through.
MIN_PRODUCTION_ROC_AUC = 0.75
# The Brier score (probability error, lower is better) must also beat the
# "no-skill" baseline: always predicting the average diabetes rate. A model
# that can't beat a constant guess is useless, however good it looks otherwise.

# =============================================================================
# Smoke test (scripts/smoke_test.py)
# =============================================================================
# CONCEPT - smoke test: a quick "does it switch on and answer?" check of a
# running service, named after powering on new hardware to see if it smokes.
# It doesn't test everything; it catches broken deployments fast.
#
# How long to keep retrying while a service starts up, in seconds. A local
# container starts in seconds; a free Render service can take a minute or more
# to wake from sleep, so deployed checks wait longer.
SMOKE_MAX_WAIT_SECONDS = 60
SMOKE_MAX_WAIT_SECONDS_DEPLOYED = 180
SMOKE_RETRY_INTERVAL_SECONDS = 3  # pause between attempts while waiting
SMOKE_REQUEST_TIMEOUT_SECONDS = 30  # give up on a single request after this long

# =============================================================================
# Classroom demos (demo/mlops_demo.ipynb, demo/devops_demo.ipynb)
# =============================================================================
# Faster settings so each live demo step takes seconds, not minutes. They are
# passed to the pipeline as parameters; the normal settings above are untouched.
DEMO_CV_SAMPLE_SIZE = 10_000  # rows used to compare models in the demo
DEMO_CV_FOLDS = 3  # fewer folds = faster, slightly less precise
# Users per simulated batch. The demo replays two batches (normal, then
# drifted), so 2 x DEMO_BATCH_SIZE must reach MIN_NEW_LABELS_FOR_RETRAIN,
# otherwise the drift alert won't trigger retraining.
DEMO_BATCH_SIZE = 1000

# =============================================================================
# Risk bands shown to users
# =============================================================================
# About 14% of people in this dataset have prediabetes or diabetes.
MODERATE_RISK_FROM = 0.15
HIGH_RISK_FROM = 0.35


def risk_level(probability: float) -> str:
    """Turn a probability into the label users see."""
    if probability >= HIGH_RISK_FROM:
        return "high"
    if probability >= MODERATE_RISK_FROM:
        return "moderate"
    return "low"


def config_snapshot() -> dict:
    """All UPPERCASE settings as plain values, for logging to MLflow."""
    return {
        name: str(value) if isinstance(value, Path) else value for name, value in globals().items() if name.isupper()
    }


# Fail fast on an impossible configuration instead of producing odd results later.
assert 2 * SIMULATED_BATCH_SIZE >= MIN_NEW_LABELS_FOR_RETRAIN, "Two simulated batches must allow retraining"
assert 2 * DEMO_BATCH_SIZE >= MIN_NEW_LABELS_FOR_RETRAIN, "Demo batches too small to trigger retraining"
assert abs(TRAIN_SHARE + TEST_SHARE + STREAM_SHARE - 1.0) < 1e-9, "Data split shares must add up to 1"
