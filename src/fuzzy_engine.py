"""Fuzzy logic primitives and the Trust Engine.

Combines Image Quality Score and model top-1 softmax confidence into a single
trust value using a small Mamdani fuzzy inference system (min/max operators,
centroid defuzzification). Uses scikit-fuzzy membership functions when
available and falls back to equivalent NumPy implementations otherwise.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

try:
    import skfuzzy as fuzz

    HAS_SKFUZZY = True
except Exception:  # pragma: no cover - fallback path
    fuzz = None
    HAS_SKFUZZY = False

UNCERTAIN_MESSAGE = "Not sure, please consult an agricultural expert."
TRUST_THRESHOLD = float(os.getenv("TRUST_THRESHOLD", 0.65))

# Output universe used for defuzzification of every FIS in this project.
UNIVERSE = np.linspace(0.0, 1.0, 201)


# --------------------------------------------------------------------------- #
# Membership functions
# --------------------------------------------------------------------------- #
def trapmf(x, abcd):
    """Trapezoidal membership. Works on scalars and arrays."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if HAS_SKFUZZY:
        return fuzz.trapmf(x, list(abcd))
    a, b, c, d = abcd
    y = np.zeros_like(x)
    if b > a:
        rise = (x >= a) & (x < b)
        y[rise] = (x[rise] - a) / (b - a)
    y[(x >= b) & (x <= c)] = 1.0
    if d > c:
        fall = (x > c) & (x <= d)
        y[fall] = (d - x[fall]) / (d - c)
    return y


def trimf(x, abc):
    a, b, c = abc
    return trapmf(x, (a, b, b, c))


def mu(x, abcd) -> float:
    """Membership degree of a scalar input."""
    return float(trapmf(float(x), abcd)[0])


def defuzzify(activations: dict[str, float], output_sets: dict[str, tuple]) -> float:
    """Mamdani aggregation (clip + max) followed by centroid defuzzification."""
    agg = np.zeros_like(UNIVERSE)
    for label, strength in activations.items():
        if strength <= 0:
            continue
        agg = np.maximum(agg, np.minimum(strength, trapmf(UNIVERSE, output_sets[label])))
    if agg.sum() == 0:
        return 0.0
    if HAS_SKFUZZY:
        return float(fuzz.defuzz(UNIVERSE, agg, "centroid"))
    return float((UNIVERSE * agg).sum() / agg.sum())


# --------------------------------------------------------------------------- #
# Trust Engine
# --------------------------------------------------------------------------- #
QUALITY_SETS = {
    "low": (0.0, 0.0, 0.35, 0.6),
    "medium": (0.35, 0.55, 0.65, 0.8),
    "high": (0.6, 0.8, 1.0, 1.0),
}
CONF_SETS = {
    "low": (0.0, 0.0, 0.45, 0.65),
    "medium": (0.45, 0.6, 0.7, 0.85),
    "high": (0.7, 0.85, 1.0, 1.0),
}
TRUST_SETS = {
    "uncertain": (0.0, 0.0, 0.3, 0.55),
    "moderate": (0.4, 0.55, 0.65, 0.8),
    "high": (0.65, 0.85, 1.0, 1.0),
}


@dataclass
class TrustResult:
    trust: float
    status: str  # "TRUSTED" | "UNCERTAIN"
    message: str
    memberships: dict


def evaluate_trust(quality: float, confidence: float, threshold: float = TRUST_THRESHOLD) -> TrustResult:
    q = {k: mu(quality, v) for k, v in QUALITY_SETS.items()}
    c = {k: mu(confidence, v) for k, v in CONF_SETS.items()}

    # Rule base (OR = max, AND = min)
    uncertain = max(q["low"], c["low"])                     # R1: Q LOW OR C LOW -> UNCERTAIN
    high = min(q["high"], c["high"])                        # R2: Q HIGH AND C HIGH -> HIGH
    moderate = max(
        min(q["medium"], c["medium"]),                      # R3: Q MED AND C MED -> MODERATE
        min(q["high"], c["medium"]),                        # R4: Q HIGH AND C MED -> MODERATE
        min(q["medium"], c["high"]),                        # R5: Q MED AND C HIGH -> MODERATE
    )

    trust = defuzzify({"uncertain": uncertain, "moderate": moderate, "high": high}, TRUST_SETS)

    if trust < threshold:
        status, message = "UNCERTAIN", UNCERTAIN_MESSAGE
    else:
        status, message = "TRUSTED", "Prediction is reliable."

    return TrustResult(
        trust=trust,
        status=status,
        message=message,
        memberships={"quality": q, "confidence": c,
                     "rules": {"uncertain": uncertain, "moderate": moderate, "high": high}},
    )
