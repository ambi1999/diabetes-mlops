"""Drift tests: PSI stays quiet on similar data and fires on shifted data.

CONCEPT - testing a detector both ways: an alarm must stay silent when nothing
changed (no false alarms) AND go off when something did (no missed alarms)."""

from src.drift import detect_drift, psi
from tests.conftest import make_dataset


def test_same_distribution_has_low_psi():
    a, b = make_dataset(5000, seed=1), make_dataset(5000, seed=2)
    assert psi(a["BMI"], b["BMI"], categorical=False) < 0.1


def test_shifted_bmi_is_detected():
    result = detect_drift(make_dataset(5000, seed=1), make_dataset(5000, bmi_shift=6, seed=2))
    assert result["drift_detected"]
    assert "BMI" in result["drifted_features"]


def test_no_drift_on_identical_data():
    df = make_dataset(5000)
    assert not detect_drift(df, df)["drift_detected"]
