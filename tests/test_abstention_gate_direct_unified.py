"""Tests for the substrate-specific direct-regime trustworthy-abstention
gate.

This module mirrors the discipline of `tests/test_abstention_gate.py`
(the G.20 SharedPool direct moat, byte-unchanged at 7/7,
``DEFAULT_THRESHOLD = 650.0``) and
`tests/test_abstention_gate_compositional.py` (the per-regime stage's
compositional moat, byte-unchanged at 7/7,
``COMPOSITIONAL_THRESHOLD = 5.688725490196079``). The new direct gate
sits ALONGSIDE both existing moats with its own pre-registered
``DIRECT_UNIFIED_THRESHOLD``.

The threshold value is now CALIBRATED (0.2841666666666667) per the
v2 protocol calibration on the unified ``build_biological_brain_
regions`` substrate (the SAME substrate Stage-1 / SPEAR / Pirazzini /
Per-regime / Unified all use); 3 seeds (42/43/44) at full biological
scale; aggregate of per-seed median midpoints. The controller commit
``0711e1d`` lands the calibrated value (mirrors the pattern of the
compositional gate's pre-committed calibration ``abe65f6``). Once
frozen and committed, retroactive recalibration is forbidden.

Biology-translatable insight (the adversarial-review-block on defect
#2): trustworthy abstention thresholds are SUBSTRATE-specific, not
regime-specific. The existing 650 moat was calibrated on G.20
SharedPool ``recall_rates`` (scale ~500-800), but the unified runner's
direct readout uses ``measure_pool_firing`` which returns a per-neuron
mean rate (scale ~0.5-2). 650 is structurally unreachable by the
direct readout regardless of how well Phase-1 trains. The disciplined
fix is a NEW substrate-specific direct gate alongside the existing 650
(mirroring exactly the pattern that added the compositional gate
alongside the G.20 direct gate). The existing 650 moat stays
byte-unchanged as historical calibration for G.20 SharedPool.
"""
from research.runners.abstention_gate_direct_unified import (
    abstain,
    gate,
    DIRECT_UNIFIED_THRESHOLD,
)


def test_direct_unified_threshold_constant_is_pinned():
    # The DIRECT_UNIFIED_THRESHOLD is CALIBRATED (0.2841666666666667)
    # per the v2 protocol calibration on the unified
    # ``build_biological_brain_regions`` substrate (3 seeds 42/43/44
    # at full biological scale; aggregate of per-seed median
    # midpoints; durable JSON
    # ``research/findings/raw/unified_CALIBRATION_v2_fullscale.json``;
    # controller commit ``0711e1d``). Once frozen and committed,
    # retroactive recalibration is forbidden.
    assert DIRECT_UNIFIED_THRESHOLD == 0.2841666666666667


def test_abstain_returns_true_iff_top_confidence_at_or_below_threshold():
    assert abstain(0.0, threshold=100.0) is True
    assert abstain(100.0, threshold=100.0) is True
    assert abstain(100.1, threshold=100.0) is False
    # Values sandwiching a hypothetical calibrated threshold near the
    # measure_pool_firing scale (~0.5-2):
    assert abstain(0.0, threshold=1.0) is True   # below
    assert abstain(1.0, threshold=1.0) is True   # at
    assert abstain(2.0, threshold=1.0) is False  # above
    # At the calibrated DIRECT_UNIFIED_THRESHOLD (0.2841666...): any
    # value at-or-below the calibrated threshold abstains. The v2
    # protocol calibrates this so that trained-word target-pool firing
    # rates typically exceed it (per-seed median margins 0.030/0.110/
    # 0.121) while ungroundable / out-of-distribution top-pool rates
    # typically fall below it.
    assert abstain(0.0, threshold=DIRECT_UNIFIED_THRESHOLD) is True
    assert abstain(0.1, threshold=DIRECT_UNIFIED_THRESHOLD) is True   # 0.1 <= 0.284 -> abstain
    assert abstain(DIRECT_UNIFIED_THRESHOLD,
                     threshold=DIRECT_UNIFIED_THRESHOLD) is True       # at threshold -> abstain
    assert abstain(0.5, threshold=DIRECT_UNIFIED_THRESHOLD) is False  # 0.5 > 0.284 -> emit


def test_gate_returns_top_tuple_when_rate_exceeds_threshold():
    ranked = [("apple", 700.0, "direct"), ("river", 500.0, "direct")]
    out = gate(ranked, threshold=650.0)
    assert out == ("apple", 700.0, "direct")


def test_gate_returns_none_when_top_rate_does_not_exceed_threshold():
    ranked = [("apple", 600.0, "direct"), ("river", 500.0, "direct")]
    out = gate(ranked, threshold=650.0)
    assert out is None


def test_gate_handles_empty_or_none_input_gracefully():
    assert gate([], threshold=650.0) is None
    assert gate(None, threshold=650.0) is None
    # Non-list inputs must be defensively rejected too.
    assert gate("not a list", threshold=650.0) is None  # type: ignore[arg-type]


def test_gate_uses_default_threshold_when_none_passed():
    # Mirror the existing moats: gate must use DIRECT_UNIFIED_THRESHOLD
    # as the default when no threshold is provided. With the calibrated
    # 0.2841666... a top rate above it clears the gate, an at-or-below
    # rate abstains.
    ranked_above = [("apple", 1.0, "direct")]
    out = gate(ranked_above)  # 1.0 > DIRECT_UNIFIED_THRESHOLD (0.284)
    assert out == ("apple", 1.0, "direct")
    ranked_at = [("apple", DIRECT_UNIFIED_THRESHOLD, "direct")]
    assert gate(ranked_at) is None  # equal does NOT exceed
    ranked_negative = [("apple", -0.1, "direct")]
    assert gate(ranked_negative) is None  # -0.1 <= DIRECT_UNIFIED_THRESHOLD
    ranked_below = [("apple", 0.2, "direct")]
    assert gate(ranked_below) is None  # 0.2 <= 0.284 -> abstain


def test_module_is_stdlib_only_and_does_not_touch_existing_moats():
    import research.runners.abstention_gate_direct_unified as direct_unified
    import research.runners.abstention_gate_compositional as comp
    import research.runners.abstention_gate as existing
    # All three modules are distinct (different file, different
    # constant name); no shared mutable state.
    assert direct_unified is not existing
    assert direct_unified is not comp
    assert comp is not existing
    # Existing G.20 SharedPool moat byte-unchanged DEFAULT_THRESHOLD
    # remains 650.0:
    assert existing.DEFAULT_THRESHOLD == 650.0
    # Existing per-regime compositional moat byte-unchanged
    # COMPOSITIONAL_THRESHOLD remains the frozen calibrated value:
    assert comp.COMPOSITIONAL_THRESHOLD == 5.688725490196079
    # New module's threshold is the calibrated value committed in
    # ``0711e1d`` (mirrors the ``abe65f6`` pattern for the
    # compositional gate):
    assert direct_unified.DIRECT_UNIFIED_THRESHOLD == 0.2841666666666667
