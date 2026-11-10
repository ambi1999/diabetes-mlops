"""Step timing: how long did each part of a run take?

What this file is:
  A tiny stopwatch for pipeline steps. The training pipeline prints a timing
  table at the end of every run and logs each step's duration to MLflow, so
  slow steps are visible immediately instead of being a mystery.

How to use it:
    timer = StepTimer()
    with timer.step("compare models"):
        ...  # the work
    print(timer.summary())
"""

import re
import time
from contextlib import contextmanager


class StepTimer:
    """Records how many seconds each named step takes, in the order they ran."""

    def __init__(self):
        self.seconds: dict[str, float] = {}

    @contextmanager
    def step(self, name: str):
        """Time everything inside a `with timer.step("name"):` block.

        Args:
            name: a short, readable step name, e.g. "compare models".
        """
        start = time.perf_counter()  # a high-resolution clock meant for measuring durations
        try:
            yield
        finally:  # record the time even if the step fails, so you see where it stopped
            self.seconds[name] = time.perf_counter() - start

    def summary(self) -> str:
        """A readable table: each step's seconds and share of the total."""
        total = sum(self.seconds.values()) or 1.0
        lines = [f"  {name:28s} {secs:7.1f} s  {secs / total:4.0%}" for name, secs in self.seconds.items()]
        return "\n".join(["Step timings:", *lines, f"  {'TOTAL':28s} {total:7.1f} s"])

    def as_metrics(self) -> dict[str, float]:
        """Durations as MLflow metric names, e.g. {"step_seconds_compare_models": 12.3}.

        MLflow only allows letters, digits and a few symbols in metric names,
        so anything else in a step's name (spaces, brackets...) becomes "_"."""

        def metric_name(step: str) -> str:
            return "step_seconds_" + re.sub(r"[^a-z0-9]+", "_", step.lower()).strip("_")

        return {metric_name(name): round(secs, 2) for name, secs in self.seconds.items()}
