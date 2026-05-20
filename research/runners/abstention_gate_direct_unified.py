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

The ``DIRECT_UNIFIED_THRESHOLD`` constant below is CALIBRATED (frozen).
The calibration was run on the ``build_biological_brain_regions``
substrate (the SAME substrate Stage-1 / SPEAR / Pirazzini / Per-regime
/ Unified all use), via the v2 protocol (per-word target-vs-best-off-
target gap aggregated over the full trained 16-word vocab; no per-seed
half-split). Three seeds (42, 43, 44) calibrated independently at
biological scale:
    seed 42: groundable_median=0.265, ungroundable_median=0.235,
             threshold=0.250
    seed 43: groundable_median=0.365, ungroundable_median=0.255,
             threshold=0.310
    seed 44: groundable_median=0.353, ungroundable_median=0.232,
             threshold=0.293
    aggregate                                      = 0.284166666...

All 3 seeds positive direction (groundable_median > ungroundable_median
by margins 0.030/0.110/0.121), so the v2 protocol's
INSUFFICIENT-SEPARATION fail-closed criterion does NOT fire; the
controller commits the aggregate value as the frozen direct-unified
moat. Calibration durable JSON:
``research/findings/raw/unified_CALIBRATION_v2_fullscale.json``;
calibration durable log:
``research/findings/raw/unified_CALIBRATION_v2_fullscale.log``.

This commit pattern mirrors the per-regime stage's compositional-gate
calibration commit (``abe65f6``, which committed 5.688725490196079 for
the per-regime substrate). Once frozen and committed, retroactive
recalibration is forbidden (it would itself be goalpost-moving). Future
substrate changes (different region set, different connectivity,
different readout) require a separate substrate-specific gate alongside
this one, NOT a re-calibration of this constant.

Biology-translatable insight: trustworthy abstention thresholds are
SUBSTRATE-AND-PROTOCOL-specific, not regime-specific or
substrate-only. (a) The existing 650 moat is on a different scale
than ``measure_pool_firing`` (defect #2 closure); the unified
substrate's direct readout NEEDS its own gate. (b) The v1 calibration
protocol (per-seed random half-split of the trained vocab) was
methodologically fragile and produced INSUFFICIENT-SEPARATION at 2/3
seeds even though the substrate genuinely retains per-word direct
binding -- because the protocol measured (strong-half-median) vs
(other-strong-half-median) on a TRAINED-ONLY population, not
trained-vs-untrained. The v2 protocol (per-word within-word
target-vs-best-off-target gap aggregated over the full vocab; no
half-split) is the principled fix and shows clean positive separation
across all 3 seeds. This sharpens the original substrate-specific
insight: thresholds must be calibrated on the right SIGNAL of the
right SUBSTRATE.

Stdlib + typing only; ASCII; mirrors the existing moats' discipline.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

# Calibrated 2026-05-20 via v2 protocol on the unified
# ``build_biological_brain_regions`` substrate; 3 seeds (42/43/44);
# full biological scale; aggregate of per-seed median midpoints.
# Durable evidence:
# ``research/findings/raw/unified_CALIBRATION_v2_fullscale.json``.
DIRECT_UNIFIED_THRESHOLD = 0.2841666666666667


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
