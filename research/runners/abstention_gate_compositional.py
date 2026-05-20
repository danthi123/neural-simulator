"""Pre-registered fixed-bar compositional-regime trustworthy-abstention
gate. Sits ALONGSIDE the existing direct-retrieval-regime gate
(`research.runners.abstention_gate`, `DEFAULT_THRESHOLD = 650.0`,
byte-unchanged at 7/7) per the Miyamoto-2017 doubly-dissociable
parallel-metamemory-streams design.

The `COMPOSITIONAL_THRESHOLD` constant below is a PLACEHOLDER (0.0).
The runner's pre-registered calibration step in Task 3 of the
per-regime metacognitive-monitor implementation plan
(`docs/plans/2026-05-20-per-regime-metacognitive-monitor-implementation.md`)
replaces it with the calibrated value and commits the source-file
change as a separate frozen step. Once calibrated and committed, the
value is frozen and retroactive recalibration is forbidden.

Stdlib + typing only; ASCII; mirrors the existing moat's discipline.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

COMPOSITIONAL_THRESHOLD = 0.0


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
