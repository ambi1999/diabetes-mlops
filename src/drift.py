"""Drift detection with the Population Stability Index (PSI).

CONCEPT - data drift: the people using the model start to look different from
the people it was trained on (e.g. average BMI rises). The model may still be
fine, but it is now working in less familiar territory, so we want to know.

Why PSI instead of a statistical test (like KS)? With thousands of rows, tests
flag even tiny, harmless shifts as "significant". PSI measures HOW MUCH a
distribution moved. Common rule of thumb:
    < 0.10 stable · 0.10-0.25 moderate change · > 0.25 significant drift"""

import numpy as np
import pandas as pd

from src.config import PSI_BINS, PSI_THRESHOLD
from src.schema import CONTINUOUS_FEATURES, FEATURES

EPSILON = 1e-4  # stands in for an empty bin's share, avoiding log(0) and division by zero


def psi(reference: pd.Series, current: pd.Series, categorical: bool, bins: int = PSI_BINS) -> float:
    """PSI = sum over bins of (current% - reference%) * ln(current% / reference%)."""
    if categorical:
        # Each category (e.g. each age group) is its own bin
        categories = sorted(set(reference.unique()) | set(current.unique()))
        ref_share = reference.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
        cur_share = current.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    else:
        # Bin edges come from the REFERENCE data's quantiles, so each bin holds
        # ~10% of reference rows. Outer edges are open so new extreme values
        # (e.g. a BMI higher than any in training) still land in a bin.
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_share = np.histogram(reference, edges)[0] / len(reference)
        cur_share = np.histogram(current, edges)[0] / len(current)

    ref_share = np.clip(ref_share, EPSILON, None)
    cur_share = np.clip(cur_share, EPSILON, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """PSI for every feature, plus which features crossed the threshold."""
    scores = {
        col: round(psi(reference[col], current[col], categorical=col not in CONTINUOUS_FEATURES), 4) for col in FEATURES
    }
    drifted = {col: score for col, score in scores.items() if score > PSI_THRESHOLD}
    return {"psi": scores, "drifted_features": drifted, "drift_detected": bool(drifted)}
