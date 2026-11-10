"""Smoke test: is a running deployment alive and answering correctly?

What this is:
  A quick end-to-end check of a RUNNING service, from the outside, exactly the
  way a user's browser or the web app would talk to it. It checks:
    1. the API answers /health and reports its app and model versions
    2. a valid request gets a sensible prediction
    3. an impossible request (BMI = 500) is refused with HTTP 422
    4. (optional) the web app answers its health endpoint

Where it's used:
  * CI ("containers" job): against the freshly built Docker images
  * .github/workflows/production-check.yml: against the deployed app, daily
  * The DevOps demo notebook, live

Usage:
  uv run python -m scripts.smoke_test --api-url http://localhost:8000 --ui-url http://localhost:8501
  uv run python -m scripts.smoke_test --api-url https://<your-service>.onrender.com --deployed

Exit code: 0 if every check passes, 1 otherwise (CI then marks the step failed).
Timing settings (how long to wait, retry interval) live in src/config.py.
"""

import argparse
import sys
import time
from dataclasses import dataclass

import requests

from src.config import (
    SMOKE_MAX_WAIT_SECONDS,
    SMOKE_MAX_WAIT_SECONDS_DEPLOYED,
    SMOKE_REQUEST_TIMEOUT_SECONDS,
    SMOKE_RETRY_INTERVAL_SECONDS,
)
from src.schema import EXAMPLE_INPUT

VALID_RISK_LEVELS = {"low", "moderate", "high"}
# Streamlit's built-in health endpoint: answers "ok" when the web app is up
UI_HEALTH_PATH = "/_stcore/health"


@dataclass
class Result:
    """One smoke-test check: its name, whether it passed, and what was seen."""

    name: str
    passed: bool
    detail: str


def wait_until_up(url: str, max_wait: float, http=requests) -> Result:
    """Keep asking `url` until it answers HTTP 200, or until max_wait runs out.

    CONCEPT - retry with a deadline: services take time to start (containers
    boot, free cloud services wake from sleep). Retrying for a while avoids
    false alarms; the deadline stops a dead service from hanging the check forever.

    Args:
        url: the full address to poll, e.g. http://localhost:8000/health.
        max_wait: give up after this many seconds.
        http: anything with a requests-style .get() (swappable in tests).
    """
    deadline = time.monotonic() + max_wait
    last_problem = "no answer"
    while True:
        try:
            response = http.get(url, timeout=SMOKE_REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 200:
                return Result(f"Reachable: {url}", True, "HTTP 200")
            last_problem = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_problem = type(exc).__name__
        if time.monotonic() >= deadline:
            return Result(f"Reachable: {url}", False, f"gave up after {max_wait:.0f}s ({last_problem})")
        time.sleep(SMOKE_RETRY_INTERVAL_SECONDS)


def check_api(api_url: str, http=requests) -> list[Result]:
    """Check a running API's health, a valid prediction, and input validation.

    Args:
        api_url: the API's base address, e.g. http://localhost:8000.
        http: anything with requests-style .get()/.post() (the tests pass
            FastAPI's TestClient here, so no real server is needed).

    Returns:
        One Result per check, in order.
    """
    api_url = api_url.rstrip("/")
    results = []

    health = http.get(f"{api_url}/health", timeout=SMOKE_REQUEST_TIMEOUT_SECONDS)
    body = health.json() if health.status_code == 200 else {}
    has_versions = bool(body.get("app_version")) and bool(body.get("model_version"))
    results.append(
        Result(
            "Health check reports versions",
            health.status_code == 200 and has_versions,
            f"app {body.get('app_version')}, model {body.get('model_version')}"
            if has_versions
            else f"HTTP {health.status_code}, body {body}",
        )
    )

    good = http.post(f"{api_url}/predict", json=EXAMPLE_INPUT, timeout=SMOKE_REQUEST_TIMEOUT_SECONDS)
    answer = good.json() if good.status_code == 200 else {}
    probability = answer.get("probability", -1)
    sensible = 0 <= probability <= 1 and answer.get("risk_level") in VALID_RISK_LEVELS
    results.append(
        Result(
            "Valid request gets a sensible prediction",
            good.status_code == 200 and sensible,
            f"probability {probability}, risk {answer.get('risk_level')}" if sensible else f"HTTP {good.status_code}",
        )
    )

    bad = http.post(f"{api_url}/predict", json={**EXAMPLE_INPUT, "BMI": 500}, timeout=SMOKE_REQUEST_TIMEOUT_SECONDS)
    results.append(
        Result(
            "Impossible input is refused",
            bad.status_code == 422,
            f"HTTP {bad.status_code} (expected 422 = rejected as invalid)",
        )
    )
    return results


def run(api_url: str, ui_url: str | None = None, deployed: bool = False) -> list[Result]:  # pragma: no cover
    """Wait for the services, then run every check. (Needs live services, so
    it's exercised by CI's containers job and the demo, not by unit tests.)

    Args:
        api_url: the API's base address.
        ui_url: the web app's base address, or None to skip the UI check.
        deployed: True for a cloud deployment: waits longer, because free
            services sleep when idle and need time to wake up.
    """
    max_wait = SMOKE_MAX_WAIT_SECONDS_DEPLOYED if deployed else SMOKE_MAX_WAIT_SECONDS
    results = [wait_until_up(f"{api_url.rstrip('/')}/health", max_wait)]
    if results[0].passed:
        results += check_api(api_url)
    if ui_url:
        results.append(wait_until_up(f"{ui_url.rstrip('/')}{UI_HEALTH_PATH}", max_wait))
    return results


def format_results(results: list[Result]) -> str:
    lines = [f"{'✅' if r.passed else '❌'} {r.name}: {r.detail}" for r in results]
    verdict = "PASSED" if results and all(r.passed for r in results) else "FAILED"
    return "\n".join(lines + ["", f"Smoke test {verdict}"])


def main() -> int:  # pragma: no cover  (command-line wrapper around run())
    parser = argparse.ArgumentParser(description="Smoke-test a running deployment.")
    parser.add_argument("--api-url", required=True, help="base address of the API, e.g. http://localhost:8000")
    parser.add_argument("--ui-url", help="base address of the web app (optional), e.g. http://localhost:8501")
    parser.add_argument(
        "--deployed",
        action="store_true",
        help="checking a cloud deployment: wait longer for sleeping free-tier services",
    )
    args = parser.parse_args()

    results = run(args.api_url, args.ui_url, args.deployed)
    print(format_results(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
