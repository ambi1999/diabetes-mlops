"""DevOps demo helper tests: the demo shows real, current facts.

settings_homes() and ci_jobs() read the project's REAL files, so these tests
also act as a tripwire: if someone renames a setting or a CI job, the demo
would break, and this test says so first.
"""

import pytest

from demo.devops_helpers import ci_jobs, lint_demo, settings_homes, useless_model_report
from tests.conftest import make_dataset


def test_settings_homes_reads_real_files():
    rows = settings_homes()
    homes = {row["lives_in"] for row in rows}
    assert {"src/config.py", "pyproject.toml", ".github/workflows/ci.yml", "render.yaml"} <= homes
    assert all(row["value"] not in (None, "") for row in rows)


def test_ci_jobs_lists_the_five_jobs():
    rows = ci_jobs()
    assert len(rows) == 5
    assert all(row["steps"] for row in rows)


def test_useless_model_is_rejected_by_the_gate():
    report = useless_model_report(make_dataset(3000, seed=1), make_dataset(1000, seed=2))
    assert report["passed"] is False


def test_lint_demo_finds_the_planted_problems():
    pytest.importorskip("ruff")  # ruff is in the dev group; skip if run without it
    result = lint_demo()
    assert "F401" in result["lint_findings"]  # unused import
    assert result["format_diff"]  # the formatter wants changes


def test_gate_report_printing(capsys):
    from demo.devops_helpers import print_checks_report

    print_checks_report(
        {"passed": False, "checks": [{"name": "ROC-AUC", "passed": False, "detail": "0.5"}]}, "Quality gate"
    )
    out = capsys.readouterr().out
    assert "❌ ROC-AUC" in out and "FAILED" in out


def test_check_model_script_exit_codes(monkeypatch):
    """CI relies on the exit code: 0 = deploy may continue, 1 = stop."""
    import scripts.check_model as script
    from src.quality_gate import Check, GateReport

    monkeypatch.setattr(script, "check_saved_model", lambda: GateReport([Check("x", True, "ok", "why")]))
    assert script.main() == 0
    monkeypatch.setattr(script, "check_saved_model", lambda: GateReport([Check("x", False, "bad", "why")]))
    assert script.main() == 1
