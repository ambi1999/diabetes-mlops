"""Model tests: every candidate works, and the bias/variance logic is right.

CONCEPT - test behavior, not exact numbers: model scores change with data and
library versions, so tests check properties that must always hold (probabilities
between 0 and 1, better than random, a deep tree overfits more than a pruned one)
instead of exact scores, which would make tests fragile."""

import pandas as pd
import pytest

from src.config import CANDIDATE_MODELS
from src.evaluation import diagnose_bias_variance, evaluate_model, select_best_model
from src.models import build_model
from src.schema import FEATURES, TARGET


@pytest.mark.parametrize("name", list(CANDIDATE_MODELS))
def test_every_candidate_trains_and_predicts(name, dataset):
    model = build_model(name).fit(dataset[FEATURES], dataset[TARGET])
    proba = model.predict_proba(dataset[FEATURES])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_unknown_model_name_raises():
    with pytest.raises(ValueError):
        build_model("not_a_model")


def test_metrics_are_in_valid_ranges(dataset):
    model = build_model("logistic_regression").fit(dataset[FEATURES], dataset[TARGET])
    metrics = evaluate_model(model, dataset)
    assert 0.5 < metrics["roc_auc"] <= 1  # better than random on learnable data
    assert 0 <= metrics["brier"] <= 1


def test_bias_variance_diagnosis():
    results = pd.DataFrame(
        {
            "model": ["overfit", "underfit", "balanced"],
            "train_score": [1.00, 0.70, 0.83],
            "val_score": [0.75, 0.69, 0.82],
        }
    )
    results["gap"] = results["train_score"] - results["val_score"]
    assert diagnose_bias_variance(results) == [
        "high variance (overfitting)",
        "high bias (underfitting)",
        "good balance",
    ]
    assert select_best_model(results) == "balanced"


def test_deep_tree_overfits_more_than_pruned_tree(dataset):
    """The deep tree should memorize training data: bigger train/validation gap."""
    train, valid = dataset.iloc[:1500], dataset.iloc[1500:]
    gaps = {}
    for name in ["decision_tree_deep", "decision_tree_pruned"]:
        model = build_model(name).fit(train[FEATURES], train[TARGET])
        gaps[name] = evaluate_model(model, train)["roc_auc"] - evaluate_model(model, valid)["roc_auc"]
    assert gaps["decision_tree_deep"] > gaps["decision_tree_pruned"]
