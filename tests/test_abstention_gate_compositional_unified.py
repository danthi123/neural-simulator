"""Tests for the substrate-specific COMPOSITIONAL-regime trustworthy-
abstention gate.

This module mirrors the discipline of `tests/test_abstention_gate.py`
(the G.20 SharedPool direct moat, byte-unchanged at 7/7,
``DEFAULT_THRESHOLD = 650.0``),
`tests/test_abstention_gate_compositional.py` (the per-regime stage's
hippocampal one-shot compositional moat, byte-unchanged at 7/7,
``COMPOSITIONAL_THRESHOLD = 5.688725490196079``), AND
`tests/test_abstention_gate_direct_unified.py` (the unified-substrate
direct moat, ``DIRECT_UNIFIED_THRESHOLD = 0.2841666666666667``). The
new unified-substrate compositional gate sits ALONGSIDE all three
existing moats with its own pre-registered
``COMPOSITIONAL_UNIFIED_THRESHOLD``.

The threshold value (0.1977124183006536) is the FROZEN aggregate of the
per-seed median midpoints produced by the unified runner's v1
compositional calibration step on the
``build_biological_brain_regions`` substrate (the SAME substrate
Stage-1 / SPEAR / Pirazzini / Per-regime / Unified all use). The
calibration was run at full biological scale across 3 seeds (42/43/44)
with all 3 seeds positive direction (groundable_median >
ungroundable_median); no INSUFFICIENT-SEPARATION fires. Calibration
durable JSON: ``research/findings/raw/unified_CALIBRATION_fullscale.json``
(key ``compositional_gate.aggregate_calibrated_threshold``). Once
frozen and committed, retroactive recalibration is forbidden.

Biology-translatable insight (now empirically validated 4 times across
the calibrated moats: 650 / 5.6887 / 0.284167 / 0.197712): trustworthy
abstention thresholds are SUBSTRATE-AND-PROTOCOL-specific. Even on the
same regime (compositional), the per-regime stage's hippocampal one-
shot substrate calibrates to 5.6887 (scale ~5) while the unified
substrate's compositional readout calibrates to 0.197712 (scale ~0.2),
a ~29x difference in scale. The substrate-specific gate sits alongside
the per-regime gate, NOT replacing it.
"""
from research.runners.abstention_gate_compositional_unified import (
    abstain,
    gate,
    COMPOSITIONAL_UNIFIED_THRESHOLD,
)


def test_compositional_unified_threshold_constant_is_pinned():
    # The COMPOSITIONAL_UNIFIED_THRESHOLD is FROZEN at the calibrated
    # value 0.1977124183006536 (aggregate of per-seed median midpoints
    # from the unified runner's v1 compositional calibration on the
    # build_biological_brain_regions substrate; 3 seeds 42/43/44; all 3
    # positive direction; no INSUFFICIENT-SEPARATION). Once frozen and
    # committed, retroactive recalibration is forbidden.
    assert COMPOSITIONAL_UNIFIED_THRESHOLD == 0.1977124183006536


def test_abstain_returns_true_iff_top_confidence_at_or_below_threshold():
    assert abstain(0.0, threshold=100.0) is True
    assert abstain(100.0, threshold=100.0) is True
    assert abstain(100.1, threshold=100.0) is False
    # Values sandwiching the calibrated threshold near the
    # _ranked_from_pattern compositional readout scale (~0.1-0.4):
    assert abstain(0.1, threshold=0.2) is True   # below
    assert abstain(0.2, threshold=0.2) is True   # at
    assert abstain(0.3, threshold=0.2) is False  # above
    # At the calibrated COMPOSITIONAL_UNIFIED_THRESHOLD ~ 0.198:
    assert abstain(0.0, threshold=COMPOSITIONAL_UNIFIED_THRESHOLD) is True
    assert abstain(0.1, threshold=COMPOSITIONAL_UNIFIED_THRESHOLD) is True
    assert abstain(0.3, threshold=COMPOSITIONAL_UNIFIED_THRESHOLD) is False


def test_gate_returns_top_tuple_when_rate_exceeds_threshold():
    ranked = [("big", 0.30, "compositional"), ("hot", 0.10, "compositional")]
    out = gate(ranked, threshold=COMPOSITIONAL_UNIFIED_THRESHOLD)
    assert out == ("big", 0.30, "compositional")


def test_gate_returns_none_when_top_rate_does_not_exceed_threshold():
    ranked = [("big", 0.10, "compositional"), ("hot", 0.05, "compositional")]
    out = gate(ranked, threshold=COMPOSITIONAL_UNIFIED_THRESHOLD)
    assert out is None


def test_gate_handles_empty_or_none_input_gracefully():
    assert gate([], threshold=COMPOSITIONAL_UNIFIED_THRESHOLD) is None
    assert gate(None, threshold=COMPOSITIONAL_UNIFIED_THRESHOLD) is None
    # Non-list inputs must be defensively rejected too.
    assert gate("not a list", threshold=COMPOSITIONAL_UNIFIED_THRESHOLD) is None  # type: ignore[arg-type]


def test_gate_uses_default_threshold_when_none_passed():
    # Mirror the existing moats: gate must use
    # COMPOSITIONAL_UNIFIED_THRESHOLD as the default when no threshold
    # is provided. With the calibrated 0.1977..., any top rate above
    # clears the gate, and a top rate at or below abstains.
    ranked_above = [("big", 0.30, "compositional")]
    out = gate(ranked_above)  # 0.30 > COMPOSITIONAL_UNIFIED_THRESHOLD
    assert out == ("big", 0.30, "compositional")
    ranked_at = [("big", COMPOSITIONAL_UNIFIED_THRESHOLD, "compositional")]
    assert gate(ranked_at) is None  # equal does NOT exceed
    ranked_below = [("big", 0.10, "compositional")]
    assert gate(ranked_below) is None  # 0.10 <= COMPOSITIONAL_UNIFIED_THRESHOLD


def test_module_is_stdlib_only_and_does_not_touch_existing_moats():
    import research.runners.abstention_gate_compositional_unified as comp_unified
    import research.runners.abstention_gate_direct_unified as direct_unified
    import research.runners.abstention_gate_compositional as comp
    import research.runners.abstention_gate as existing
    # All four modules are distinct (different files, different
    # constant names); no shared mutable state.
    assert comp_unified is not existing
    assert comp_unified is not comp
    assert comp_unified is not direct_unified
    assert direct_unified is not existing
    assert direct_unified is not comp
    assert comp is not existing
    # Existing G.20 SharedPool moat byte-unchanged DEFAULT_THRESHOLD
    # remains 650.0:
    assert existing.DEFAULT_THRESHOLD == 650.0
    # Existing per-regime compositional moat byte-unchanged
    # COMPOSITIONAL_THRESHOLD remains the frozen calibrated value:
    assert comp.COMPOSITIONAL_THRESHOLD == 5.688725490196079
    # Existing unified-substrate direct moat retains its calibrated
    # value:
    assert direct_unified.DIRECT_UNIFIED_THRESHOLD == 0.2841666666666667
    # New module's threshold is the calibrated unified-substrate
    # compositional value:
    assert comp_unified.COMPOSITIONAL_UNIFIED_THRESHOLD == 0.1977124183006536
