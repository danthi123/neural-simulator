"""Pre-registered fixed-bar substrate-specific compositional-regime
trustworthy-abstention gate. Sits ALONGSIDE the existing
``research.runners.abstention_gate`` (``DEFAULT_THRESHOLD = 650.0``,
byte-unchanged, calibrated on G.20 SharedPool ``recall_rates`` --
scale ~500-800),
``research.runners.abstention_gate_compositional``
(``COMPOSITIONAL_THRESHOLD = 5.688725490196079``, byte-unchanged,
calibrated on the per-regime stage's hippocampal one-shot substrate),
AND ``research.runners.abstention_gate_direct_unified``
(``DIRECT_UNIFIED_THRESHOLD = 0.2841666666666667``, just-committed
``0711e1d``, calibrated on the unified
``build_biological_brain_regions`` substrate via v2 protocol)
per the Miyamoto-2017 doubly-dissociable parallel-metamemory-streams
design, extended to the substrate-specific dimension (the SECOND
of the two unified-substrate-specific calibrated moats; mirrors the
direct-unified gate pattern shipped at ``0711e1d``).

The ``COMPOSITIONAL_UNIFIED_THRESHOLD`` constant below is CALIBRATED
(frozen). The calibration was run on the
``build_biological_brain_regions`` substrate (the SAME substrate
Stage-1 / SPEAR / Pirazzini / Per-regime / Unified all use), via the
existing v1 compositional calibration protocol (the unified runner's
``_calibrate_compositional_one_seed``, which encodes held-out
(noun, adj) pairs at sub-seed offset +30000 and measures raw firing-
rate confidence at lang_output for GROUNDABLE vs UNGROUNDABLE
queries). Three seeds (42, 43, 44) calibrated independently at full
biological scale:
    seed 42: groundable_median=0.250, ungroundable_median=0.186
             (margin 0.064); threshold=0.218
    seed 43: groundable_median=0.265, ungroundable_median=0.147
             (margin 0.118); threshold=0.206
    seed 44: groundable_median=0.201, ungroundable_median=0.137
             (margin 0.064); threshold=0.169
    aggregate                                       = 0.197712418300...

All 3 seeds positive direction (groundable_median > ungroundable_median
at every seed), so the calibration's INSUFFICIENT-SEPARATION fail-
closed criterion does NOT fire; the controller commits the aggregate
value as the frozen compositional-unified moat. Calibration durable
JSON:
``research/findings/raw/unified_CALIBRATION_fullscale.json``
(key ``compositional_gate.aggregate_calibrated_threshold``).

This commit pattern mirrors the just-shipped direct-unified gate
calibration commit (``0711e1d``, which committed 0.2841666666666667 for
the unified substrate's direct regime). Once frozen and committed,
retroactive recalibration is forbidden (it would itself be goalpost-
moving). Future substrate changes (different region set, different
connectivity, different readout) require a separate substrate-specific
gate alongside this one, NOT a re-calibration of this constant.

Biology-translatable insight (now empirically validated 4 times across
the calibrated moats: 650 / 5.6887 / 0.284167 / 0.197712): trustworthy
abstention thresholds are SUBSTRATE-AND-PROTOCOL-specific. Even on
the same regime (compositional), the per-regime stage's hippocampal
one-shot substrate calibrates to 5.6887 (scale ~5) while the unified
substrate's compositional readout calibrates to 0.197712 (scale ~0.2)
-- a ~29x difference in scale. The unified substrate's compositional
gate NEEDS its own threshold; the per-regime stage's 5.6887 is
structurally unreachable here. The substrate-specific gate sits
ALONGSIDE the per-regime gate, NOT replacing it.

Stdlib + typing only; ASCII; mirrors the existing moats' discipline.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

# Calibrated 2026-05-20 via v1 compositional protocol on the unified
# ``build_biological_brain_regions`` substrate; 3 seeds (42/43/44);
# full biological scale; aggregate of per-seed median midpoints.
# Durable evidence:
# ``research/findings/raw/unified_CALIBRATION_fullscale.json``
# (key ``compositional_gate.aggregate_calibrated_threshold``).
COMPOSITIONAL_UNIFIED_THRESHOLD = 0.1977124183006536


def abstain(
    top_confidence: float,
    threshold: float = COMPOSITIONAL_UNIFIED_THRESHOLD,
) -> bool:
    return float(top_confidence) <= threshold


def gate(
    ranked: Optional[List[Tuple[Any, float, Any]]],
    threshold: float = COMPOSITIONAL_UNIFIED_THRESHOLD,
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
