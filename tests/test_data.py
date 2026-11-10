"""Data tests: validation and splitting.

CONCEPT - test the guards: data validation protects the whole pipeline, so we
test that it catches each kind of problem (out-of-range values, missing
columns) and that the split follows the shares in config."""

from src.config import STREAM_SHARE, TEST_SHARE, TRAIN_SHARE
from src.data import split_data, validate_data
from src.schema import TARGET


def test_valid_data_passes(dataset):
    assert validate_data(dataset) == []


def test_out_of_range_bmi_fails(dataset):
    dataset.loc[0, "BMI"] = 500
    assert any("BMI" in error for error in validate_data(dataset))


def test_missing_column_fails(dataset):
    assert validate_data(dataset.drop(columns=["Age"]))


def test_split_sizes_follow_config(dataset):
    train, test, stream = split_data(dataset)
    n = len(dataset)
    assert len(train) + len(test) + len(stream) == n
    assert abs(len(train) / n - TRAIN_SHARE) < 0.01
    assert abs(len(test) / n - TEST_SHARE) < 0.01
    assert abs(len(stream) / n - STREAM_SHARE) < 0.01


def test_split_keeps_diabetes_rate(dataset):
    """Stratification: every split should have about the same positive rate."""
    rate = dataset[TARGET].mean()
    for part in split_data(dataset):
        assert abs(part[TARGET].mean() - rate) < 0.02


def test_validate_features_needs_no_label(dataset):
    from src.data import validate_features
    from src.schema import FEATURES

    assert validate_features(dataset[FEATURES]) == []


def test_missing_label_column_fails(dataset):
    assert validate_data(dataset.drop(columns=[TARGET]))


def test_training_data_adds_older_new_labels_and_evaluates_on_newest(tmp_path, monkeypatch, dataset):
    """Cumulative retraining: old training data + the older 80% of new labels;
    the newest 20% becomes the evaluation set (see src/data.py)."""
    from src import data

    train, test, _ = split_data(dataset)
    new = dataset.head(1000)
    for name, frame in [("TRAIN_DATA", train), ("TEST_DATA", test), ("NEW_LABELS", new)]:
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        monkeypatch.setattr(data, name, path)

    training, evaluation, eval_name = data.load_training_and_evaluation_data(include_new_labels=True)
    assert eval_name == "recent_labels"
    assert len(training) == len(train) + 800
    assert len(evaluation) == 200

    _, evaluation, eval_name = data.load_training_and_evaluation_data(include_new_labels=False)
    assert eval_name == "test_set" and len(evaluation) == len(test)
