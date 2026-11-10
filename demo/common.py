"""Shared building blocks for both demo notebooks (MLOps and DevOps).

What this file is:
  Everything the two notebooks have in common, written once:
    * DemoRunner      - the live / record / saved modes and the fallback
    * configure()     - quiet, predictable output for a live audience
    * waking the deployed API (free cloud services sleep when idle)
    * pre-flight check printing, and display helpers for tables and images

The notebooks stay short; the logic lives here and in mlops_helpers.py /
devops_helpers.py, where it can be tested (tests/test_demo_helpers.py).

The three modes (set at the top of each notebook):
  "live"    run each step for real; if a step fails, show the saved rehearsal result
  "record"  run each step for real AND save the result as the fallback (rehearsal)
  "saved"   run nothing live; show saved results only (the safest option)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import warnings
from pathlib import Path

import pandas as pd
import requests

from src.config import ROOT

DEMO_DIR = ROOT / "demo"
FALLBACK_ROOT = DEMO_DIR / "fallback"


def rel(path: Path) -> str:
    """A path relative to the project folder.

    The notebooks are committed WITH their outputs as handouts, so outputs must
    never reveal personal folders such as C:\\Users\\<name>\\...

    Args:
        path: any path.

    Returns:
        e.g. "demo/fallback/mlops/compare_models.png"; just the file name if
        the path is outside the project.
    """
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).name


# =============================================================================
# Environment setup
# =============================================================================
def configure(prefect_api_url: str | None = None) -> None:
    """Quiet, predictable output for a live audience. Call before any demo step.

    Args:
        prefect_api_url: address of the running Prefect server (MLOps demo),
            e.g. http://127.0.0.1:4200/api, so flow runs appear in its
            dashboard and there's no start-up delay. None if not needed.

    Effects (this notebook session only, never the project's settings):
        Prefect's and MLflow's log chatter is reduced to warnings: the flows
        still print their key results, and full logs stay in the dashboard.
    """
    if prefect_api_url:
        os.environ.setdefault("PREFECT_API_URL", prefect_api_url)
    os.environ["PREFECT_LOGGING_LEVEL"] = "WARNING"
    warnings.filterwarnings("ignore")
    for name in ("mlflow", "alembic", "prefect", "httpx"):
        logging.getLogger(name).setLevel(logging.ERROR)


_deployed_status: dict = {"checked": False}


def url_is_set(url: str | None) -> bool:
    """True if a URL setting was filled in (not empty, not a <placeholder>)."""
    return bool(url) and "<" not in url


def wake_deployed_api(url: str | None) -> None:  # pragma: no cover  (needs the internet)
    """Ping the deployed API in the background, without waiting for an answer.

    Render's free tier puts the service to sleep when idle, and waking it takes
    up to a minute. Pinging at the start of the demo means it's awake by the end.

    Args:
        url: the deployed API's base address, e.g. https://<name>.onrender.com.
    """
    if not url_is_set(url):
        _deployed_status.update(checked=True, ok=False, note="no deployed URL set")
        return

    def ping():
        try:
            response = requests.get(f"{url.rstrip('/')}/health", timeout=90)
            _deployed_status.update(checked=True, ok=response.ok, note=f"HTTP {response.status_code}")
        except requests.RequestException as exc:
            _deployed_status.update(checked=True, ok=False, note=type(exc).__name__)

    threading.Thread(target=ping, daemon=True).start()


def deployed_status() -> str:
    """One line describing the background wake-up ping's result so far."""
    if not _deployed_status["checked"]:
        return "⏳ still waking up (free tier) - give it a few more seconds"
    return ("✅ awake" if _deployed_status["ok"] else "❌ not reachable") + f" ({_deployed_status['note']})"


# =============================================================================
# Pre-flight checks
# =============================================================================
def reachable(url: str) -> bool:
    """True if `url` answers with a success status within 5 seconds."""
    try:
        return requests.get(url, timeout=5).ok
    except requests.RequestException:
        return False


def print_checks(checks: list[tuple[str, bool, str]]) -> bool:
    """Print a ✅/❌ list, with the fix next to each ❌.

    Args:
        checks: (label, passed, how to fix it) for each item.

    Returns:
        True if everything passed.
    """
    for label, ok, fix in checks:
        print(f"{'✅' if ok else '❌'} {label}" + ("" if ok else f"   → {fix}"))
    all_ok = all(ok for _, ok, _ in checks)
    print("\nReady to present! 🎉" if all_ok else "\nFix the ❌ items, then run this cell again.")
    return all_ok


def missing_fallbacks(fallback_dir: Path, steps: tuple[str, ...]) -> list[str]:
    """Which steps have no saved rehearsal result yet."""
    return [s for s in steps if not (fallback_dir / f"{s}.json").exists()]


# =============================================================================
# Live-or-fallback runner
# =============================================================================
class DemoRunner:
    """Runs each demo step live, recording or falling back as the mode says.

    CONCEPT - graceful degradation: when something fails, show a clearly
    labelled saved result instead of an error, so the demo (like a good
    production system) keeps working while being honest about it.

    Args:
        mode: "live", "record" or "saved" (see the top of this file).
        fallback_dir: where saved rehearsal results are stored, e.g.
            demo/fallback/mlops.
    """

    MODES = ("live", "record", "saved")

    def __init__(self, mode: str, fallback_dir: Path):
        if mode not in self.MODES:
            raise ValueError(f"MODE must be one of {self.MODES}, not {mode!r}")
        self.mode = mode
        self.fallback_dir = Path(fallback_dir)
        self.fallbacks_used: list[str] = []

    def run(self, step: str, live_fn) -> dict:
        """Run one step.

        Args:
            step: a short name for the step; also the saved file's name.
            live_fn: a function with no arguments that does the step for real
                and returns a JSON-friendly dict.

        Returns:
            The live result, or the saved one if running live wasn't possible.
        """
        if self.mode == "saved":
            return self._load(step, note="📼 Saved mode: showing results from the rehearsal run.")
        try:
            result = live_fn()
        except Exception as exc:  # any failure -> fall back, but say so clearly
            self.fallbacks_used.append(step)
            if not (self.fallback_dir / f"{step}.json").exists():
                raise RuntimeError(
                    f"Live step '{step}' failed and there is no saved result yet. "
                    "Fix the error, then rehearse once with MODE = 'record'."
                ) from exc
            reason = str(exc).split("(")[0][:100]  # keep long technical errors short on screen
            return self._load(
                step,
                note=f"⚠️ Live step failed ({type(exc).__name__}: {reason}).\n"
                "📼 Showing results from the rehearsal run instead.",
            )
        if self.mode == "record":
            self._save(step, result)
        return result

    def _save(self, step: str, result: dict) -> None:
        self.fallback_dir.mkdir(parents=True, exist_ok=True)
        result = dict(result)
        if result.get("image_path"):  # keep a copy of any chart next to the JSON
            saved_image = self.fallback_dir / f"{step}.png"
            shutil.copy(ROOT / result["image_path"], saved_image)
            result["image_path"] = rel(saved_image)
        saved = self.fallback_dir / f"{step}.json"
        saved.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")

    def _load(self, step: str, note: str) -> dict:
        saved = self.fallback_dir / f"{step}.json"
        if not saved.exists():
            raise RuntimeError(f"No saved result for '{step}'. Rehearse once with MODE = 'record'.")
        print(note)
        return json.loads(saved.read_text(encoding="utf-8"))

    def handout_check(self) -> None:
        """Before committing a notebook as a handout, every step should have run live."""
        if self.mode == "saved":
            print("ℹ️ Saved mode: outputs come from the rehearsal run.")
        elif self.fallbacks_used:
            print(f"⚠️ These steps used saved results: {self.fallbacks_used}. Fix them before committing.")
        else:
            print("✅ Every step ran live. This notebook is ready to commit as a handout.")


# =============================================================================
# Display helpers
# =============================================================================
def show_table(records: list[dict], columns: dict[str, str]) -> pd.DataFrame:
    """Turn a list of dicts into a table with friendly column names.

    Args:
        records: the rows, e.g. [{"model": "knn", "val_score": 0.81}, ...].
        columns: which keys to show, mapped to their display names, in order.

    Returns:
        A DataFrame; as the last line of a notebook cell, it displays as a table.
    """
    return pd.DataFrame(records)[list(columns)].rename(columns=columns)


def show_image(image_path: str):  # pragma: no cover  (needs a notebook to display)
    """Display a saved chart; the image is embedded, so it also shows on GitHub."""
    from IPython.display import Image

    return Image(filename=str(ROOT / image_path))
