"""Smoke test tests: the checks themselves must be right.

The smoke test guards every deployment, so it needs tests too. Here it runs
against the REAL API through FastAPI's TestClient (no server needed), plus a
fake service for the waiting logic.
"""

import requests

from scripts.smoke_test import Result, check_api, format_results, wait_until_up


def test_all_checks_pass_against_a_healthy_api(api_client):
    results = check_api("http://testserver", http=api_client)
    assert len(results) == 3
    assert all(r.passed for r in results), format_results(results)


class FakeService:
    """Stands in for a service: answers with the given status codes, in turn."""

    def __init__(self, statuses):
        self.statuses = list(statuses)

    def get(self, url, timeout):
        status = self.statuses.pop(0) if self.statuses else 503

        class Response:
            status_code = status

        return Response()


def test_wait_succeeds_when_service_answers():
    assert wait_until_up("http://svc/health", max_wait=5, http=FakeService([200])).passed


def test_wait_gives_up_on_a_dead_service():
    class DeadService:
        def get(self, url, timeout):
            raise requests.ConnectionError("refused")

    result = wait_until_up("http://svc/health", max_wait=0, http=DeadService())
    assert not result.passed
    assert "gave up" in result.detail


def test_report_says_failed_if_any_check_fails():
    text = format_results([Result("a", True, "ok"), Result("b", False, "broken")])
    assert "❌ b" in text and text.endswith("FAILED")
