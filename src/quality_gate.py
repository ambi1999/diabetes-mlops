"""Model quality gate: the minimum standard a model must meet to serve users.

What this file is:
  One shared set of checks, used in two places:
    * pipeline.py  - before PROMOTING a newly trained model (MLOps)
    * CI           - before DEPLOYING the committed model (DevOps),
                     via `uv run python -m scripts.check_model`
  Using the same code in both places means "good enough to promote" and
  "good enough to deploy" can never mean different things.

CONCEPT - quality gate vs. champion/challenger:
  * Champion/challenger is RELATIVE: is the new model better than the current one?
  * The quality gate is ABSOLUTE: is this model good enough at all?
  A model must pass both to be promoted. Without the absolute gate, a slightly
  "better" model could still be useless if the current one were broken.

Thresholds live in src/config.py (MIN_PRODUCTION_ROC_AUC).
"""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.config import METADATA_PATH, MIN_PRODUCTION_ROC_AUC, MODEL_PATH, REFERENCE_PATH
from src.data import validate_features
from src.schema import FEATURES


@dataclass
class Check:
    """The result of one check.

    Attributes:
        name: short name shown in reports, e.g. "Minimum ROC-AUC".
        passed: True if the model met this requirement.
        detail: what was actually measured, e.g. "0.82 (minimum 0.75)".
        why: one plain-language sentence on why this check matters.
    """

    name: str
    passed: bool
    detail: str
    why: str


@dataclass
class GateReport:
    """All checks for one model. The gate passes only if EVERY check passes."""

    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def as_text(self) -> str:
        """A readable report for terminals, CI logs and notebooks."""
        lines = [f"{'✅' if c.passed else '❌'} {c.name}: {c.detail}\n     why: {c.why}" for c in self.checks]
        verdict = "PASSED: the model may serve users." if self.passed else "FAILED: the model must NOT serve users."
        return "\n".join(lines + ["", f"Quality gate {verdict}"])

    def as_dict(self) -> dict:
        """JSON-friendly version, for MLflow and the demo's saved results."""
        return {"passed": self.passed, "checks": [c.__dict__ for c in self.checks]}


def run_quality_gate(model, metadata: dict, reference: pd.DataFrame) -> GateReport:
    """Run every check on a model that is already loaded in memory.

    Args:
        model: a trained scikit-learn model (with predict_proba).
        metadata: facts about the model; must include "sklearn_version" and
            "metrics" (with "roc_auc", "brier" and "no_skill_brier"), measured
            on evaluation data the model never trained on.
        reference: a sample of the model's input data (features only), used to
            check the model produces valid probabilities.

    Returns:
        A GateReport; report.passed tells you whether the model may serve users.
    """
    report = GateReport()
    metrics = metadata.get("metrics", {})

    # 1. Same library version: a model saved by one scikit-learn version may
    #    load with errors, or silently behave differently, under another.
    trained_with = metadata.get("sklearn_version", "unknown")
    report.checks.append(
        Check(
            "scikit-learn version matches",
            trained_with == sklearn.__version__,
            f"trained with {trained_with}, installed {sklearn.__version__}",
            "a model must run with the exact library version that created it",
        )
    )

    # 2. Good enough at ranking people by risk
    roc_auc = metrics.get("roc_auc")
    report.checks.append(
        Check(
            "Minimum ROC-AUC",
            roc_auc is not None and roc_auc >= MIN_PRODUCTION_ROC_AUC,
            f"{roc_auc:.3f} (minimum {MIN_PRODUCTION_ROC_AUC})" if roc_auc is not None else "not recorded",
            "0.5 is a coin flip; the model must rank people by risk clearly better than that",
        )
    )

    # 3. Better probabilities than a constant guess
    brier, no_skill = metrics.get("brier"), metrics.get("no_skill_brier")
    beats_baseline = brier is not None and no_skill is not None and brier < no_skill
    report.checks.append(
        Check(
            "Beats the no-skill baseline",
            beats_baseline,
            f"Brier {brier:.4f} vs. {no_skill:.4f} for always guessing the average"
            if beats_baseline or (brier is not None and no_skill is not None)
            else "not recorded (retrain with the current pipeline to record it)",
            "if always predicting the average rate is as accurate, the model adds nothing",
        )
    )

    # 4. The reference sample (used for drift checks) obeys the data contract
    problems = validate_features(reference)
    report.checks.append(
        Check(
            "Reference sample is valid",
            not problems,
            "matches the data contract" if not problems else "; ".join(problems),
            "drift monitoring compares live users against this sample, so it must be clean",
        )
    )

    # 5. Sensible outputs: every probability between 0 and 1, none missing
    if not problems:
        proba = model.predict_proba(reference[FEATURES])[:, 1]
        valid = bool(np.isfinite(proba).all() and (proba >= 0).all() and (proba <= 1).all())
        detail = f"{len(proba):,} predictions checked" + ("" if valid else ", some invalid")
    else:
        valid, detail = False, "skipped: the reference sample is invalid"
    report.checks.append(
        Check(
            "Valid probabilities",
            valid,
            detail,
            "users see these numbers; a probability outside 0-1 or missing means something is broken",
        )
    )
    return report


def check_saved_model(
    model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH, reference_path: Path = REFERENCE_PATH
) -> GateReport:
    """Run the gate on the model files in models/ (what CI does before deploying).

    Args:
        model_path: the saved model file (model.joblib).
        metadata_path: the model's metadata.json (version, metrics, library version).
        reference_path: the reference sample saved with the model.

    Returns:
        A GateReport. If a file is missing or can't be loaded, the report
        contains a single failed check explaining why.
    """
    try:
        model = joblib.load(model_path)
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        reference = pd.read_csv(reference_path)
    # CONCEPT - catch specific errors: list exactly the failures we expect, so a
    # real bug elsewhere still surfaces loudly instead of being hidden.
    #   OSError           a file is missing or unreadable
    #   ValueError        corrupt JSON/CSV (their parse errors are ValueErrors)
    #   EOFError, pickle.UnpicklingError   a truncated or corrupt model file
    #   AttributeError, ImportError        a model saved by an incompatible library version
    except (OSError, ValueError, EOFError, pickle.UnpicklingError, AttributeError, ImportError) as exc:
        return GateReport(
            [
                Check(
                    "Model files load",
                    False,
                    f"{type(exc).__name__}: {exc}",
                    "the API can't serve a model it can't load",
                )
            ]
        )
    report = run_quality_gate(model, metadata, reference)
    report.checks.insert(
        0,
        Check(
            "Model files load",
            True,
            "model, metadata and reference sample",
            "the API can't serve a model it can't load",
        ),
    )
    return report
