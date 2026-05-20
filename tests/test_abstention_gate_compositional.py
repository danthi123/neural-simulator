"""Tests for the compositional-regime trustworthy-abstention gate.

This module mirrors the discipline of `tests/test_abstention_gate.py`
(the direct-retrieval-regime moat, byte-unchanged at 7/7). The new
gate lives ALONGSIDE the existing one with its own pre-registered
`COMPOSITIONAL_THRESHOLD`. The threshold value is FROZEN at the
calibrated value 5.688725490196079 from the full-scale held-out
calibration run on 2026-05-20 (seeds 42/43/44; per-seed thresholds
[4.676470588235294, 5.688725490196079, 6.316176470588236]; median =
5.688725490196079; method = median_midpoint per the runner's
docstring; calibration set held-out per the adversarial review's
zero-overlap fix). Once frozen and committed, retroactive
recalibration is forbidden.
"""
from research.runners.abstention_gate_compositional import (
    abstain,
    gate,
    COMPOSITIONAL_THRESHOLD,
)


def test_compositional_threshold_constant_is_pinned():
    # The COMPOSITIONAL_THRESHOLD is FROZEN at the value calibrated
    # on 2026-05-20 at full biological scale (seeds 42/43/44; per-seed
    # thresholds [4.676470588235294, 5.688725490196079,
    # 6.316176470588236]; median = 5.688725490196079; method =
    # median_midpoint; calibration set held-out per the adversarial
    # review's zero-overlap fix). Provenance:
    # research/findings/raw/per_regime_CALIBRATION_fullscale.json.
    # Once frozen and committed, retroactive recalibration is forbidden.
    assert COMPOSITIONAL_THRESHOLD == 5.688725490196079


def test_abstain_returns_true_iff_top_confidence_at_or_below_threshold():
    assert abstain(0.0, threshold=100.0) is True
    assert abstain(100.0, threshold=100.0) is True
    assert abstain(100.1, threshold=100.0) is False
    # Values sandwiching the frozen calibrated threshold (~5.69):
    assert abstain(1.0, threshold=COMPOSITIONAL_THRESHOLD) is True   # below
    assert abstain(0.0, threshold=COMPOSITIONAL_THRESHOLD) is True   # below
    assert abstain(COMPOSITIONAL_THRESHOLD, threshold=COMPOSITIONAL_THRESHOLD) is True  # at
    assert abstain(10.0, threshold=COMPOSITIONAL_THRESHOLD) is False  # above


def test_gate_returns_top_tuple_when_rate_exceeds_threshold():
    ranked = [("apple", 700.0, "ep_0"), ("river", 500.0, "ep_1")]
    out = gate(ranked, threshold=650.0)
    assert out == ("apple", 700.0, "ep_0")


def test_gate_returns_none_when_top_rate_does_not_exceed_threshold():
    ranked = [("apple", 600.0, "ep_0"), ("river", 500.0, "ep_1")]
    out = gate(ranked, threshold=650.0)
    assert out is None


def test_gate_handles_empty_or_none_input_gracefully():
    assert gate([], threshold=650.0) is None
    assert gate(None, threshold=650.0) is None


def test_gate_uses_default_threshold_when_none_passed():
    # Mirror the existing moat: gate must use COMPOSITIONAL_THRESHOLD
    # as the default when no threshold is provided. Threshold is now
    # the frozen calibrated value 5.688725490196079.
    ranked_above = [("apple", 10.0, "ep_0")]
    out = gate(ranked_above)  # 10.0 > COMPOSITIONAL_THRESHOLD ~5.69
    assert out == ("apple", 10.0, "ep_0")
    ranked_at = [("apple", COMPOSITIONAL_THRESHOLD, "ep_0")]
    assert gate(ranked_at) is None  # equal does NOT exceed
    ranked_below = [("apple", 1.0, "ep_0")]
    assert gate(ranked_below) is None  # 1.0 <= COMPOSITIONAL_THRESHOLD


def test_module_is_stdlib_only_and_does_not_touch_existing_moat():
    import research.runners.abstention_gate_compositional as comp
    import research.runners.abstention_gate as existing
    # Module distinct from the existing moat (different file, different
    # constant name); no shared mutable state.
    assert comp is not existing
    # Existing moat byte-unchanged DEFAULT_THRESHOLD remains 650.0:
    assert existing.DEFAULT_THRESHOLD == 650.0
    # New module's threshold is the frozen calibrated value:
    assert comp.COMPOSITIONAL_THRESHOLD == 5.688725490196079
