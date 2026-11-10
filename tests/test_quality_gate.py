"""Quality gate tests: good models pass, bad models are stopped.

CONCEPT - test the gatekeeper both ways: a gate that never opens is useless,
and so is one that never closes. We check it passes a genuinely good model and
fails each kind of bad one: no better than guessing, built with a different
library version, or missing its files.
"""

import json

import joblib
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.evaluation import evaluate_model
from src.models import build_model
from src.quality_gate import check_saved_model, run_quality_gate
from src.schema import FEATURES, TARGET
from tests.conftest import make_dataset

TRAIN, EVALUATION = make_dataset(4000, seed=1, signal=2.0), make_dataset(2000, seed=2, signal=2.0)


def good_model_and_metadata():
    model = build_model("logistic_regression").fit(TRAIN[FEATURES], TRAIN[TARGET])
    return model, {"sklearn_version": sklearn.__version__, "metrics": evaluate_model(model, EVALUATION)}


def test_good_model_passes():
    model, metadata = good_model_and_metadata()
    report = run_quality_gate(model, metadata, EVALUATION[FEATURES])
    assert report.passed, report.as_text()


def test_no_skill_model_fails():
    """A model that gives everyone the average rate must not pass."""
    useless = Pipeline([("classifier", DummyClassifier(strategy="prior"))]).fit(TRAIN[FEATURES], TRAIN[TARGET])
    metadata = {"sklearn_version": sklearn.__version__, "metrics": evaluate_model(useless, EVALUATION)}
    report = run_quality_gate(useless, metadata, EVALUATION[FEATURES])
    assert not report.passed
    failed = {c.name for c in report.checks if not c.passed}
    assert {"Minimum ROC-AUC", "Beats the no-skill baseline"} <= failed


def test_library_version_mismatch_fails():
    model, metadata = good_model_and_metadata()
    report = run_quality_gate(model, {**metadata, "sklearn_version": "0.0.1"}, EVALUATION[FEATURES])
    assert not report.passed
    assert "FAILED" in report.as_text()


def test_invalid_reference_sample_fails():
    model, metadata = good_model_and_metadata()
    broken = EVALUATION[FEATURES].copy()
    broken.loc[0, "BMI"] = 500
    report = run_quality_gate(model, metadata, broken)
    assert not report.passed


def test_saved_model_files_pass(tmp_path):
    model, metadata = good_model_and_metadata()
    joblib.dump(model, tmp_path / "model.joblib")
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    EVALUATION[FEATURES].to_csv(tmp_path / "reference.csv", index=False)
    report = check_saved_model(tmp_path / "model.joblib", tmp_path / "metadata.json", tmp_path / "reference.csv")
    assert report.passed, report.as_text()
    assert report.as_dict()["passed"] is True


def test_missing_model_files_fail(tmp_path):
    report = check_saved_model(tmp_path / "nope.joblib", tmp_path / "nope.json", tmp_path / "nope.csv")
    assert not report.passed
    assert report.checks[0].name == "Model files load"
