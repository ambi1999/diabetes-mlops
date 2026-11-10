"""DevOps demo steps for demo/devops_demo.ipynb.

What this file is:
  Each step runs a REAL part of the project's DevOps tooling - the linter and
  formatter, the CI workflow definition, test coverage, the model quality
  gate and the smoke test - and returns a JSON-friendly dict, so the result
  can be saved as a fallback. Shared machinery lives in demo/common.py.

Functions marked "pragma: no cover" need live services (running containers,
the deployed app) or the real model and data. They're exercised by
rehearsing the demo, not by unit tests, so they're excluded from coverage.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import pandas as pd
import requests
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from demo.common import FALLBACK_ROOT, missing_fallbacks, print_checks, reachable, url_is_set
from src import config
from src.config import MODEL_PATH, ROOT, TEST_DATA, TRAIN_DATA
from src.evaluation import evaluate_model
from src.quality_gate import check_saved_model, run_quality_gate
from src.schema import FEATURES, TARGET

FALLBACK_DIR = FALLBACK_ROOT / "devops"
FALLBACK_STEPS = (
    "lint",
    "coverage",
    "gate_production",
    "gate_useless",
    "smoke_local",
    "smoke_deployed",
    "model_history",
)

# A deliberately messy file for the "catching problems before commit" step.
# It has unused imports, an unused variable and untidy spacing.
MESSY_CODE = """import os, sys
def risk_label(bmi,age) :
    unused = 1
    if bmi>30 and age > 50 :
        return  "high"
    return "low"
"""


# =============================================================================
# Pre-flight checks
# =============================================================================
def preflight(api_url: str, ui_url: str, deployed_api_url: str, deployed_ui_url: str) -> bool:  # pragma: no cover
    """Check everything the DevOps demo needs; print a ✅/❌ list with fixes.

    Args:
        api_url / ui_url: the local containers, e.g. http://localhost:8000 and :8501.
        deployed_api_url / deployed_ui_url: the public deployment's addresses.
    """
    missing = missing_fallbacks(FALLBACK_DIR, FALLBACK_STEPS)
    return print_checks(
        [
            ("Production model present", MODEL_PATH.exists(), "run: uv run pipeline.py"),
            ("API container running", reachable(f"{api_url}/health"), "run: docker compose up"),
            ("Web app container running", reachable(f"{ui_url}/_stcore/health"), "run: docker compose up"),
            (
                "Deployed URLs filled in",
                url_is_set(deployed_api_url) and url_is_set(deployed_ui_url),
                "set DEPLOYED_API_URL and DEPLOYED_UI_URL in the settings cell",
            ),
            (
                "Fallback results saved",
                not missing,
                f"rehearse once with MODE = 'record' (missing: {', '.join(missing)})",
            ),
        ]
    )


# =============================================================================
# Demo steps
# =============================================================================
def settings_homes() -> list[dict]:
    """Show where each KIND of setting lives, read live from the real files.

    Returns:
        One row per example setting: what it is, its current value, and the
        file that is its single home.
    """
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    render = yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))
    return [
        {
            "setting": "Share of data held back as future users",
            "value": f"{config.STREAM_SHARE:.0%}",
            "lives_in": "src/config.py",
            "kind": "project setting",
        },
        {
            "setting": "Minimum ROC-AUC to deploy a model",
            "value": config.MIN_PRODUCTION_ROC_AUC,
            "lives_in": "src/config.py",
            "kind": "project setting",
        },
        {"setting": "Allowed BMI range", "value": "12 to 98", "lives_in": "src/schema.py", "kind": "data contract"},
        {
            "setting": "App version",
            "value": pyproject["project"]["version"],
            "lives_in": "pyproject.toml",
            "kind": "project metadata",
        },
        {
            "setting": "Minimum test coverage",
            "value": f"{pyproject['tool']['coverage']['report']['fail_under']}%",
            "lives_in": "pyproject.toml",
            "kind": "tool setting",
        },
        {
            "setting": "Python version",
            "value": (ROOT / ".python-version").read_text(encoding="utf-8").strip(),
            "lives_in": ".python-version",
            "kind": "environment",
        },
        {
            "setting": "uv version used by CI",
            "value": ci["env"]["UV_VERSION"],
            "lives_in": ".github/workflows/ci.yml",
            "kind": "CI setting",
        },
        {
            "setting": "Cloud region for the API",
            "value": render["services"][0]["region"],
            "lives_in": "render.yaml",
            "kind": "infrastructure as code",
        },
    ]


def lint_demo() -> dict:
    """Run the real formatter and linter on a deliberately messy file.

    The file lives in a temporary folder, so the project is never touched.
    This is exactly what the pre-commit hooks do before every commit.

    Returns:
        The messy code, the formatter's proposed changes (a "diff"), and the
        linter's findings.
    """
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "messy.py").write_text(MESSY_CODE, encoding="utf-8")

        def ruff(*args: str) -> str:
            # --isolated: use ruff's defaults, not this project's settings;
            # cwd=tmp keeps file paths in the output short ("messy.py").
            # check=False: ruff exits with code 1 when it FINDS problems, which is
            # exactly what we want to show, so a non-zero exit isn't an error here.
            done = subprocess.run(
                [sys.executable, "-m", "ruff", *args, "--isolated", "--no-cache", "messy.py"],
                cwd=tmp,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if "No module named" in done.stderr:  # ruff missing: fail loudly, never record an error as a result
                raise RuntimeError("ruff isn't installed in this environment; run `uv sync`")
            return (done.stdout + done.stderr).strip()

        return {
            "before": MESSY_CODE,
            "format_diff": ruff("format", "--diff"),
            "lint_findings": ruff("check", "--output-format", "concise"),
        }


# Routine steps that every job has (setup) or that only report results.
# Hidden from the demo's table so the steps that actually CHECK something stand out.
ROUTINE_STEP_PREFIXES = ("Check out", "Install", "Cache", "Upload", "Add ", "Show container logs", "Warn if")


def ci_jobs() -> list[dict]:
    """Describe the CI pipeline, read live from .github/workflows/ci.yml.

    Generated from the real file, so the explanation can never drift out of
    date with the actual pipeline.

    Returns:
        One row per job: its name and its main steps (routine setup and
        reporting steps left out for readability).
    """
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    rows = []
    for job_id, job in ci["jobs"].items():
        names = [step["name"] for step in job.get("steps", []) if "name" in step]
        steps = [name for name in names if not name.startswith(ROUTINE_STEP_PREFIXES)]
        rows.append({"job": job.get("name", job_id), "steps": " → ".join(steps)})
    return rows


def coverage_summary() -> dict:  # pragma: no cover  (runs the whole test suite)
    """Run the test suite with coverage, exactly as CI does, and summarize it.

    Returns:
        The total percentage, the minimum required, and the least-covered files.
    """
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "coverage.json"
        # check=False: pytest exits non-zero if a test fails or coverage is below the
        # minimum; we still want the report, and check below that it was written.
        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--cov",
                f"--cov-report=json:{report_path}",
                "-p",
                "no:cacheprovider",
            ],
            cwd=ROOT,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if not report_path.exists():
            raise RuntimeError(f"the test run produced no coverage report: {done.stderr.strip()[-200:]}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
    files = sorted(
        (
            {"file": name, "covered": round(info["summary"]["percent_covered"])}
            for name, info in report["files"].items()
        ),
        key=lambda row: row["covered"],
    )
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    minimum = pyproject["tool"]["coverage"]["report"]["fail_under"]
    return {"total": round(report["totals"]["percent_covered"], 1), "minimum": minimum, "lowest": files[:5]}


def gate_on_production_model() -> dict:  # pragma: no cover  (needs the real model)
    """Run the real quality gate on the committed production model."""
    return check_saved_model().as_dict()


def useless_model_report(train: pd.DataFrame, evaluation: pd.DataFrame) -> dict:
    """Build a deliberately useless model and run the real quality gate on it.

    The "model" ignores every answer and gives everyone the same probability:
    the average diabetes rate in the training data. It's what any real model
    must beat, so the gate should reject it.

    Args:
        train: labeled data to learn the average rate from.
        evaluation: labeled data the gate's metrics are measured on.

    Returns:
        The gate's report as a dict (passed=False expected).
    """
    import sklearn

    useless = Pipeline([("classifier", DummyClassifier(strategy="prior"))])
    useless.fit(train[FEATURES], train[TARGET])
    metadata = {"sklearn_version": sklearn.__version__, "metrics": evaluate_model(useless, evaluation)}
    return run_quality_gate(useless, metadata, evaluation[FEATURES]).as_dict()


def gate_on_useless_model() -> dict:  # pragma: no cover  (needs the real data)
    return useless_model_report(pd.read_csv(TRAIN_DATA, nrows=20_000), pd.read_csv(TEST_DATA))


def smoke(api_url: str, ui_url: str | None, deployed: bool) -> dict:  # pragma: no cover  (needs live services)
    """Run the real smoke test (scripts/smoke_test.py) against running services.

    Args:
        api_url / ui_url: the addresses to check.
        deployed: True for the cloud deployment (waits longer for it to wake up).
    """
    from scripts.smoke_test import run

    results = run(api_url, ui_url, deployed)
    health = requests.get(f"{api_url.rstrip('/')}/health", timeout=30).json()
    return {
        "checks": [r.__dict__ for r in results],
        "passed": all(r.passed for r in results),
        "app_version": health.get("app_version"),
        "model_version": health.get("model_version"),
    }


def model_history(limit: int = 5) -> list[str]:  # pragma: no cover  (needs the Git repository)
    """The latest commits that changed models/: the model release history.

    Because the production model is committed to Git, every model release is a
    commit, and rolling back a model is just `git revert <commit>`.

    Args:
        limit: how many recent commits to show.

    Returns:
        One line per commit: short commit ID, date and message.
    """
    done = subprocess.run(
        ["git", "log", f"-{limit}", "--date=short", "--format=%h  %ad  %s", "--", "models/"],
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return done.stdout.strip().splitlines() or ["(no commits touching models/ yet)"]


# =============================================================================
# Display helpers (plain-language output for beginners)
# =============================================================================
def print_checks_report(report: dict, title: str) -> None:
    """Print a quality-gate or smoke-test report as ✅/❌ lines.

    Args:
        report: a dict with "checks" (each having name, passed, detail) and "passed".
        title: a heading, e.g. "Quality gate: production model".
    """
    print(title)
    for check in report["checks"]:
        print(f"  {'✅' if check['passed'] else '❌'} {check['name']}: {check['detail']}")
    print(f"\n  → {'PASSED' if report['passed'] else 'FAILED'}")
