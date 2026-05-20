"""Pre-registered fixed-bar compositional-regime trustworthy-abstention
gate. Sits ALONGSIDE the existing direct-retrieval-regime gate
(`research.runners.abstention_gate`, `DEFAULT_THRESHOLD = 650.0`,
byte-unchanged at 7/7) per the Miyamoto-2017 doubly-dissociable
parallel-metamemory-streams design.

The `COMPOSITIONAL_THRESHOLD` constant below is FROZEN at the
calibrated value 5.688725490196079 (median of the per-seed
calibrated thresholds [4.676470588235294, 5.688725490196079,
6.316176470588236] from the full-scale held-out calibration run on
2026-05-20, seeds 42/43/44, CuPy/RTX3090; method = median_midpoint
of held-out groundable vs ungroundable compositional-readout raw
firing-rate confidences, calibration set = Cartesian-product of
the validated v16 concept vocabulary MINUS the evaluation set
(zero pair overlap verified by the dedicated adversarial review).
Provenance: research/findings/raw/per_regime_CALIBRATION_fullscale.json.
Once frozen and committed, retroactive recalibration is forbidden
(it would itself be goalpost-moving).

Stdlib + typing only; ASCII; mirrors the existing moat's discipline.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

COMPOSITIONAL_THRESHOLD = 5.688725490196079


def abstain(
    top_confidence: float,
    threshold: float = COMPOSITIONAL_THRESHOLD,
) -> bool:
    return float(top_confidence) <= threshold


def gate(
    ranked: Optional[List[Tuple[Any, float, Any]]],
    threshold: float = COMPOSITIONAL_THRESHOLD,
) -> Optional[Tuple[Any, float, Any]]:
    """ranked: list of (concept, rate, tag) desc. Return top tuple if
    its rate clears the gate, else None (=> abstain). Defensive on
    None / non-list inputs: returns None."""
    if ranked is None:
        return None
    if not isinstance(ranked, list):
        return None
    if not ranked:
        return None
    top = ranked[0]
    return None if abstain(top[1], threshold) else top
