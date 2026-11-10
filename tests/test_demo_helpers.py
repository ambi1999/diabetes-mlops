"""Demo helper tests: the reset must be safe, and the fallback must be honest.

CONCEPT - test the safety net: the reset deletes files, so the most important
tests prove what it must NEVER delete (downloaded data, MLflow history,
unrelated files). Everything runs in a temporary copy of the project layout
(pytest's tmp_path), so the real project is never touched."""

import json

import pytest

from demo.common import DemoRunner, rel, show_table
from demo.mlops_helpers import DemoPaths, print_drift, reset_demo, snapshot_baseline

MODEL_FILES = ("model.joblib", "metadata.json", "reference_sample.csv")


def write_model(folder, version):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.joblib").write_text(f"model v{version}", encoding="utf-8")
    (folder / "metadata.json").write_text(json.dumps({"model_version": version}), encoding="utf-8")
    (folder / "reference_sample.csv").write_text("BMI\n25\n", encoding="utf-8")


@pytest.fixture
def project(tmp_path):
    """A fake project mid-demo: baseline v1 saved, v3 promoted during the demo."""
    paths = DemoPaths.for_root(tmp_path)
    write_model(paths.baseline_dir, "1")
    write_model(paths.models_dir, "3")
    # Files the demo creates (should be cleared)
    paths.stream_offset.parent.mkdir(parents=True, exist_ok=True)
    paths.stream_offset.write_text("2000", encoding="utf-8")
    paths.new_labels.write_text("BMI,Diabetes_binary\n30,1\n", encoding="utf-8")
    paths.prediction_db.parent.mkdir(parents=True, exist_ok=True)
    paths.prediction_db.write_text("fake sqlite", encoding="utf-8")
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    for name in ("drift_1.json", "drift_1.html", "demo_learning_curve.png"):
        (paths.reports_dir / name).write_text("x", encoding="utf-8")
    # Files the reset must NEVER touch
    protected = [
        tmp_path / "data" / "raw.csv",
        tmp_path / "data" / "train.csv",
        tmp_path / "data" / "stream.csv",
        tmp_path / "mlflow.db",
        paths.reports_dir / "my_notes.txt",
    ]
    for f in protected:
        f.write_text("keep me", encoding="utf-8")
    return paths, protected


def test_reset_clears_demo_state(project):
    paths, _ = project
    reset_demo(paths, restore_alias=False)
    assert not paths.stream_offset.exists()
    assert not paths.new_labels.exists()
    assert not paths.prediction_db.exists()
    assert not list(paths.reports_dir.glob("drift_*"))
    assert not list(paths.reports_dir.glob("demo_*"))


def test_reset_never_touches_protected_files(project):
    paths, protected = project
    reset_demo(paths, restore_alias=False)
    for f in protected:
        assert f.exists(), f"reset deleted {f.name}"
        assert f.read_text(encoding="utf-8") == "keep me"


def test_reset_restores_baseline_model(project):
    paths, _ = project
    reset_demo(paths, restore_alias=False)
    assert json.loads((paths.models_dir / "metadata.json").read_text(encoding="utf-8"))["model_version"] == "1"


def test_reset_is_safe_to_repeat(project):
    paths, protected = project
    reset_demo(paths, restore_alias=False)
    reset_demo(paths, restore_alias=False)  # nothing left to clear: must not fail
    assert all(f.exists() for f in protected)


def test_reset_without_baseline_fails_clearly(tmp_path):
    paths = DemoPaths.for_root(tmp_path)
    write_model(paths.models_dir, "1")
    with pytest.raises(FileNotFoundError):
        reset_demo(paths, restore_alias=False)


def test_snapshot_refuses_to_overwrite_baseline(project):
    paths, _ = project
    with pytest.raises(FileExistsError):
        snapshot_baseline(paths)
    assert snapshot_baseline(paths, force=True) == "3"


def test_record_then_saved_mode_returns_the_same_result(tmp_path):
    paths = DemoPaths.for_root(tmp_path)
    DemoRunner("record", paths.fallback_dir).run("step", lambda: {"answer": 42})
    assert DemoRunner("saved", paths.fallback_dir).run("step", lambda: {"answer": 0}) == {"answer": 42}


def test_live_failure_falls_back_and_is_reported(tmp_path):
    paths = DemoPaths.for_root(tmp_path)
    DemoRunner("record", paths.fallback_dir).run("step", lambda: {"answer": 42})

    def broken():
        raise ConnectionError("API is down")

    runner = DemoRunner("live", paths.fallback_dir)
    assert runner.run("step", broken) == {"answer": 42}
    assert runner.fallbacks_used == ["step"]  # the handout check will flag this


def test_live_failure_without_saved_result_raises(tmp_path):
    def broken():
        raise ConnectionError("API is down")

    with pytest.raises(RuntimeError):
        DemoRunner("live", DemoPaths.for_root(tmp_path).fallback_dir).run("step", broken)


def test_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        DemoRunner("fast", tmp_path)


def test_handout_check_reports_fallbacks(tmp_path, capsys):
    runner = DemoRunner("live", tmp_path)
    runner.handout_check()
    assert "Every step ran live" in capsys.readouterr().out
    runner.fallbacks_used.append("prediction")
    runner.handout_check()
    assert "prediction" in capsys.readouterr().out


def test_paths_in_outputs_are_never_personal(tmp_path):
    from src.config import ROOT

    assert rel(ROOT / "demo" / "x.png") == "demo/x.png"
    assert rel(tmp_path / "secret" / "file.txt") == "file.txt"  # outside the project: name only


def test_display_helpers(capsys):
    table = show_table([{"model": "knn", "val_score": 0.8}], {"model": "Model", "val_score": "Score"})
    assert list(table.columns) == ["Model", "Score"]
    print_drift(
        {
            "sent": {"users_sent": 10, "average_bmi": 30.0, "average_age_group": 8.0},
            "top_psi": [{"feature": "BMI", "psi": 0.9}, {"feature": "Age", "psi": 0.05}],
            "drift_detected": True,
            "drifted_features": ["BMI"],
        }
    )
    out = capsys.readouterr().out
    assert "DRIFT DETECTED" in out and "🟢 stable" in out


def test_retraining_explanations(capsys):
    from demo.mlops_helpers import explain_retraining

    explain_retraining({"retraining": None})
    assert "did not run" in capsys.readouterr().out
    base = {
        "new_version": "2",
        "selected_model": "knn",
        "challenger_roc_auc": 0.8,
        "champion_roc_auc": 0.81,
        "challenger_brier": 0.1,
        "champion_brier": 0.1,
    }
    explain_retraining({"retraining": {**base, "promoted": False}})
    assert "champion stays" in capsys.readouterr().out
    explain_retraining({"retraining": {**base, "promoted": True}})
    assert "challenger won" in capsys.readouterr().out


def test_preflight_building_blocks(tmp_path, capsys):
    from demo.common import missing_fallbacks, print_checks, reachable

    assert print_checks([("thing", True, "fix")]) is True
    assert print_checks([("thing", False, "do this")]) is False
    assert "do this" in capsys.readouterr().out
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    assert missing_fallbacks(tmp_path, ("a", "b")) == ["b"]
    assert reachable("http://127.0.0.1:9/nothing-listens-here") is False


def test_configure_quiets_logs(monkeypatch):
    import os

    from demo.common import configure, deployed_status, url_is_set

    monkeypatch.delenv("PREFECT_API_URL", raising=False)
    configure("http://127.0.0.1:4200/api")
    assert os.environ["PREFECT_API_URL"] == "http://127.0.0.1:4200/api"
    assert os.environ["PREFECT_LOGGING_LEVEL"] == "WARNING"
    assert url_is_set("https://x.onrender.com") and not url_is_set("https://<your-service>.onrender.com")
    assert isinstance(deployed_status(), str)


def test_default_paths_point_into_the_project():
    from src.config import ROOT

    assert DemoPaths.default().baseline_dir == ROOT / "demo" / "baseline"
