"""Quick run mode tests: quick runs must be fast AND harmless.

CONCEPT - test the safety guarantee, not just the feature: the most important
promise of quick mode is that it can never touch production (the model in
models/, its MLflow registry entry, its data and logs). These tests switch the
profile the same way quick.env does, then check every path and name differs.
"""

import importlib

import pytest

from src import config
from src.data import stratified_sample
from src.schema import TARGET
from src.timing import StepTimer
from tests.conftest import make_dataset


@pytest.fixture
def load_profile(monkeypatch):
    """Reload src/config.py under a chosen RUN_PROFILE; restore the default afterwards."""
    monkeypatch.delenv("PREDICTION_DB", raising=False)  # use the profile's default log location

    def load(profile: str | None):
        if profile is None:
            monkeypatch.delenv("RUN_PROFILE", raising=False)
        else:
            monkeypatch.setenv("RUN_PROFILE", profile)
        return importlib.reload(config)

    yield load
    monkeypatch.delenv("RUN_PROFILE", raising=False)
    importlib.reload(config)  # leave the default (full) profile for other tests


def test_full_is_the_default(load_profile):
    cfg = load_profile(None)
    assert cfg.RUN_PROFILE == "full" and cfg.DATA_SAMPLE_ROWS is None
    assert cfg.MODELS_DIR.name == "models"


def test_quick_never_touches_production(load_profile):
    full = load_profile("full")
    full_paths = {
        name: getattr(full, name)
        for name in (
            "MODELS_DIR",
            "MODEL_PATH",
            "TRAIN_DATA",
            "TEST_DATA",
            "STREAM_DATA",
            "NEW_LABELS",
            "STREAM_OFFSET",
            "PREDICTION_DB",
            "REPORTS_DIR",
            "EXPERIMENT_NAME",
            "REGISTERED_MODEL_NAME",
        )
    }
    quick = load_profile("quick")
    for name, full_value in full_paths.items():
        assert getattr(quick, name) != full_value, f"{name} is shared with production"
    # ...except the raw download, which both profiles share on purpose
    assert quick.RAW_DATA == full.RAW_DATA


def test_quick_settings_are_smaller_but_still_trigger_retraining(load_profile):
    quick = load_profile("quick")
    assert quick.DATA_SAMPLE_ROWS == 20_000
    assert quick.CV_SAMPLE_SIZE < 40_000 and quick.CV_FOLDS < 5
    # Two simulated batches must be enough for the drift check and for retraining
    assert quick.SIMULATED_BATCH_SIZE >= quick.MIN_PREDICTIONS_FOR_DRIFT_CHECK
    assert 2 * quick.SIMULATED_BATCH_SIZE >= quick.MIN_NEW_LABELS_FOR_RETRAIN
    # ...and the sample must hold enough future users for several batches
    assert quick.DATA_SAMPLE_ROWS * quick.STREAM_SHARE >= 4 * quick.SIMULATED_BATCH_SIZE


def test_unknown_profile_is_rejected(load_profile):
    with pytest.raises(ValueError):
        load_profile("fast")


def test_stratified_sample_keeps_the_diabetes_rate():
    data = make_dataset(10_000)
    sample = stratified_sample(data, 2_000)
    assert len(sample) == 2_000
    assert abs(sample[TARGET].mean() - data[TARGET].mean()) < 0.005
    # Same seed, same rows: every quick run uses exactly the same people
    assert sample.index.equals(stratified_sample(data, 2_000).index)
    assert len(stratified_sample(data, 50_000)) == 10_000  # asking for more than exists: unchanged


def test_step_timer_records_each_step():
    timer = StepTimer()
    with timer.step("compare models"):
        pass
    with pytest.raises(RuntimeError), timer.step("log and register (MLflow)"):
        raise RuntimeError("a failing step is still timed")
    assert list(timer.seconds) == ["compare models", "log and register (MLflow)"]
    assert set(timer.as_metrics()) == {"step_seconds_compare_models", "step_seconds_log_and_register_mlflow"}
    assert "TOTAL" in timer.summary()
