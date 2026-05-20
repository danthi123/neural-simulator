"""Pre-registered fixed-bar substrate-specific direct-retrieval
trustworthy-abstention gate. Sits ALONGSIDE the existing
``research.runners.abstention_gate`` (``DEFAULT_THRESHOLD = 650.0``,
byte-unchanged, calibrated on G.20 SharedPool ``recall_rates`` --
scale ~500-800) AND
``research.runners.abstention_gate_compositional``
(``COMPOSITIONAL_THRESHOLD = 5.688725490196079``, calibrated on the
per-regime stage's hippocampal one-shot substrate) per the
Miyamoto-2017 doubly-dissociable parallel-metamemory-streams design,
extended to the substrate-specific dimension (the FIFTH consecutive
adversarial review's defect #2 closure).

The ``DIRECT_UNIFIED_THRESHOLD`` constant below is a PLACEHOLDER (0.0)
until the unified per-regime runner's calibration step on the
``build_biological_brain_regions`` substrate (the SAME substrate
Stage-1 / SPEAR / Pirazzini / Per-regime / Unified all use) produces
the calibrated value, which the controller commits as a separate
frozen step (mirrors the pattern of the compositional gate's
pre-committed calibration ``abe65f6``). Once frozen and committed,
retroactive recalibration is forbidden (it would itself be
goalpost-moving).

Biology-translatable insight (the adversarial-review-block on defect
#2): trustworthy abstention thresholds are SUBSTRATE-specific, not
regime-specific. The existing 650 moat was calibrated on G.20
SharedPool ``recall_rates`` (scale ~500-800), but the unified runner's
direct readout uses ``measure_pool_firing`` which returns a per-neuron
mean firing rate (scale ~0.5-2 documented in CLAUDE.md). 650 is
structurally unreachable by the direct readout regardless of how well
Phase-1 trains. The disciplined fix is a NEW substrate-specific direct
gate alongside the existing 650 (mirroring exactly the pattern that
added the compositional gate alongside the G.20 direct gate). The
existing 650 moat stays byte-unchanged as historical calibration for
G.20 SharedPool.

Stdlib + typing only; ASCII; mirrors the existing moats' discipline.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

DIRECT_UNIFIED_THRESHOLD = 0.0


def abstain(
    top_confidence: float,
    threshold: float = DIRECT_UNIFIED_THRESHOLD,
) -> bool:
    return float(top_confidence) <= threshold


def gate(
    ranked: Optional[List[Tuple[Any, float, Any]]],
    threshold: float = DIRECT_UNIFIED_THRESHOLD,
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
