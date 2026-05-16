"""Abstention gate. Threshold from 2026-05-16-G20-320-abstention-
benchmark: encoded top-rate mean ~796, control max ~584 -> gate 650
cleanly separates know/don't-know (AUC 0.990). The no-confabulation
moat: below gate => "I don't know" instead of the noisy top associate."""
from __future__ import annotations

DEFAULT_THRESHOLD = 650.0

def abstain(top_confidence: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    return float(top_confidence) <= threshold

def gate(ranked, threshold: float = DEFAULT_THRESHOLD):
    """ranked: list of (concept, rate, tag) desc. Return top tuple if
    its rate clears the gate, else None (=> abstain)."""
    if not ranked: return None
    top = ranked[0]
    return None if abstain(top[1], threshold) else top
