"""Evaluation: metrics, model comparison, and bias vs. variance diagnosis.

Big picture of how a model is chosen (each step uses DIFFERENT data):
  1. Cross-validation on the TRAINING data compares candidates fairly.
  2. The winner is retrained on all training data.
  3. The EVALUATION set (never seen before) gives an honest final score,
     used to compare the new model with the one in production.
Using the evaluation set to pick between candidates would make its score
optimistic, because we'd be choosing whatever happened to fit it best."""

from pathlib import Path

import pandas as pd

# Figure (instead of pyplot) draws straight to a file: no screen needed, so it
# works on servers and in CI.
from matplotlib.figure import Figure
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve

from src.config import (
    BIAS_SHORTFALL_THRESHOLD,
    CANDIDATE_MODELS,
    CV_FOLDS,
    CV_SAMPLE_SIZE,
    HIGH_RISK_FROM,
    LEARNING_CURVE_FOLDS,
    LEARNING_CURVE_SIZES,
    RANDOM_SEED,
    SELECTION_METRIC,
    VARIANCE_GAP_THRESHOLD,
)
from src.models import build_model
from src.schema import FEATURES, TARGET


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def evaluate_model(model, df: pd.DataFrame) -> dict:
    """Score a trained model on labeled data.

    - roc_auc: how well the model RANKS people by risk (0.5 = random, 1 = perfect).
    - brier: how accurate the probabilities themselves are (lower is better).
      A model can rank well but give probabilities that are too high or low;
      that matters here because users see the probability.
    - no_skill_brier: the Brier score of always predicting the average rate.
    - recall/precision at the "high risk" threshold: of people with diabetes,
      how many we flag; of people we flag, how many have diabetes."""
    y_true = df[TARGET]
    proba = model.predict_proba(df[FEATURES])[:, 1]
    flagged = (proba >= HIGH_RISK_FROM).astype(int)
    # The "no-skill" baseline: the Brier score you'd get by giving EVERYONE the
    # average rate p. Its error works out to p * (1 - p). A useful model must
    # beat it; the quality gate (src/quality_gate.py) checks exactly that.
    positive_rate = float(y_true.mean())
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "no_skill_brier": positive_rate * (1 - positive_rate),
        "recall_high_risk": float(recall_score(y_true, flagged, zero_division=0)),
        "precision_high_risk": float(precision_score(y_true, flagged, zero_division=0)),
    }


# -----------------------------------------------------------------------------
# Comparing candidates with cross-validation
# -----------------------------------------------------------------------------
def _cv_splitter(folds: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)


def _sample_for_cv(train: pd.DataFrame, sample_size: int | None) -> pd.DataFrame:
    """Same random sample every time (fixed seed), so all candidates are compared on identical rows."""
    if sample_size and len(train) > sample_size:
        return train.sample(sample_size, random_state=RANDOM_SEED)
    return train


def cross_validate_candidates(
    train: pd.DataFrame, sample_size: int | None = None, folds: int | None = None
) -> pd.DataFrame:
    """Score every candidate with k-fold cross-validation.

    CONCEPT - cross-validation: the training data is split into k folds; each
    model trains on k-1 folds and is scored on the held-out fold, k times.
    Averaging gives a more reliable estimate than a single split, and the
    spread (std) shows how stable the model is.

    We record the TRAINING score too, because comparing it with the
    validation score is how we diagnose bias vs. variance.

    sample_size and folds default to config (CV_SAMPLE_SIZE, CV_FOLDS). The demo
    passes smaller values to run faster; normal runs never pass them."""
    sample_size = CV_SAMPLE_SIZE if sample_size is None else sample_size
    folds = CV_FOLDS if folds is None else folds
    train = _sample_for_cv(train, sample_size)
    rows = []
    for name in CANDIDATE_MODELS:
        scores = cross_validate(
            build_model(name),
            train[FEATURES],
            train[TARGET],
            cv=_cv_splitter(folds),
            scoring=SELECTION_METRIC,
            return_train_score=True,
        )
        rows.append(
            {
                "model": name,
                "train_score": scores["train_score"].mean(),
                "val_score": scores["test_score"].mean(),  # sklearn calls the held-out fold "test"
                "val_std": scores["test_score"].std(),
                "fit_seconds": scores["fit_time"].mean(),
            }
        )

    results = pd.DataFrame(rows)
    results["gap"] = results["train_score"] - results["val_score"]
    results["diagnosis"] = diagnose_bias_variance(results)
    return results.sort_values("val_score", ascending=False).reset_index(drop=True)


def diagnose_bias_variance(results: pd.DataFrame) -> list[str]:
    """Label each model as overfitting, underfitting, or balanced.

    CONCEPT - bias vs. variance:
    - HIGH VARIANCE (overfitting): the model memorizes training data. Training
      score is much higher than validation score (a large gap).
      Fixes: simpler model, more regularization, more data.
    - HIGH BIAS (underfitting): the model is too simple to capture the pattern.
      Training and validation scores are close, but both are low compared with
      what other models achieve.
      Fixes: more flexible model, better features.
    - Good balance: small gap AND a validation score near the best available.

    "Low" is judged relative to the best candidate, because what counts as a
    good score depends on the dataset."""
    best_val = results["val_score"].max()
    labels = []
    for _, row in results.iterrows():
        if row["gap"] > VARIANCE_GAP_THRESHOLD:
            labels.append("high variance (overfitting)")
        elif best_val - row["val_score"] > BIAS_SHORTFALL_THRESHOLD:
            labels.append("high bias (underfitting)")
        else:
            labels.append("good balance")
    return labels


def select_best_model(results: pd.DataFrame) -> str:
    """Pick the candidate with the best validation score.

    We rank by VALIDATION score, never training score: the training score
    rewards memorization, which is exactly what we want to avoid."""
    return str(results.loc[results["val_score"].idxmax(), "model"])


# -----------------------------------------------------------------------------
# Learning curve
# -----------------------------------------------------------------------------
def plot_learning_curve(
    model_name: str,
    train: pd.DataFrame,
    output_path: Path,
    sample_size: int | None = None,
    folds: int | None = None,
) -> Path:
    """Plot training vs. validation score as the amount of training data grows.

    How to read it:
    - Curves far apart, validation still rising -> high variance; more data helps.
    - Curves close together and flat -> more data won't help much; if the score
      is too low, that's high bias and you need a more flexible model or better features.

    sample_size and folds default to config (CV_SAMPLE_SIZE, LEARNING_CURVE_FOLDS)."""
    sample_size = CV_SAMPLE_SIZE if sample_size is None else sample_size
    folds = LEARNING_CURVE_FOLDS if folds is None else folds
    train = _sample_for_cv(train, sample_size)
    sizes, train_scores, val_scores = learning_curve(
        build_model(model_name),
        train[FEATURES],
        train[TARGET],
        train_sizes=LEARNING_CURVE_SIZES,
        cv=_cv_splitter(folds),
        scoring=SELECTION_METRIC,
    )
    fig = Figure(figsize=(7, 4.5))
    ax = fig.subplots()
    for scores, label in [(train_scores, "Training score"), (val_scores, "Validation score")]:
        mean, std = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(sizes, mean, marker="o", label=label)
        ax.fill_between(sizes, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("Number of training rows")
    ax.set_ylabel(SELECTION_METRIC)
    ax.set_title(f"Learning curve: {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    return output_path


def format_results_table(results: pd.DataFrame) -> str:
    """Readable table for the console."""
    shown = results[["model", "train_score", "val_score", "gap", "diagnosis"]].copy()
    for col in ["train_score", "val_score", "gap"]:
        shown[col] = shown[col].map(lambda x: f"{x:.4f}")
    return shown.to_string(index=False)
