"""Evaluation tests: model comparison, metrics and the learning curve.

Uses tiny settings (small sample, 2 folds) so these run in seconds; the real
pipeline uses the values in src/config.py.
"""

from src.config import CANDIDATE_MODELS
from src.evaluation import cross_validate_candidates, evaluate_model, format_results_table, plot_learning_curve
from src.models import build_model
from src.schema import FEATURES, TARGET
from tests.conftest import make_dataset


def test_comparison_covers_every_candidate():
    results = cross_validate_candidates(make_dataset(1500, signal=2.0), sample_size=1000, folds=2)
    assert set(results["model"]) == set(CANDIDATE_MODELS)
    assert {"train_score", "val_score", "gap", "diagnosis"} <= set(results.columns)
    assert results["val_score"].is_monotonic_decreasing  # best model first
    assert "decision_tree_deep" in format_results_table(results)


def test_no_skill_baseline_matches_its_formula(dataset):
    model = build_model("logistic_regression").fit(dataset[FEATURES], dataset[TARGET])
    p = dataset[TARGET].mean()
    assert abs(evaluate_model(model, dataset)["no_skill_brier"] - p * (1 - p)) < 1e-12


def test_learning_curve_is_saved(tmp_path):
    path = plot_learning_curve(
        "logistic_regression", make_dataset(1200), tmp_path / "curve.png", sample_size=1000, folds=2
    )
    assert path.exists() and path.stat().st_size > 0
