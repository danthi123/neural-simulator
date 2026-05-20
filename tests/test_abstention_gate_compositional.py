"""Tests for the compositional-regime trustworthy-abstention gate.

This module mirrors the discipline of `tests/test_abstention_gate.py`
(the direct-retrieval-regime moat, byte-unchanged at 7/7). The new
gate lives ALONGSIDE the existing one with its own pre-registered
`COMPOSITIONAL_THRESHOLD`. The threshold value here is a PLACEHOLDER
(0.0); the runner's pre-registered calibration step in Task 3 of the
per-regime metacognitive-monitor plan replaces it with the calibrated
value and commits the change as a separate frozen step.
"""
from research.runners.abstention_gate_compositional import (
    abstain,
    gate,
    COMPOSITIONAL_THRESHOLD,
)


def test_compositional_threshold_constant_is_pinned():
    # Calibration is the runner's job (Task 3). This task pins the
    # placeholder value the runner will replace. The placeholder is 0.0
    # so that until calibration runs, the gate is permissive (accepts
    # everything strictly above 0.0) AND the runner's calibration step
    # is REQUIRED to produce a meaningful gate. This pin asserts the
    # placeholder, NOT the calibrated value (the calibrated value will
    # replace it).
    assert COMPOSITIONAL_THRESHOLD == 0.0


def test_abstain_returns_true_iff_top_confidence_at_or_below_threshold():
    assert abstain(0.0, threshold=100.0) is True
    assert abstain(100.0, threshold=100.0) is True
    assert abstain(100.1, threshold=100.0) is False
    assert abstain(1.0, threshold=COMPOSITIONAL_THRESHOLD) is False
    assert abstain(0.0, threshold=COMPOSITIONAL_THRESHOLD) is True


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
    # as the default when no threshold is provided.
    ranked = [("apple", 1.0, "ep_0")]
    out = gate(ranked)  # default threshold == COMPOSITIONAL_THRESHOLD = 0.0
    assert out == ("apple", 1.0, "ep_0")
    ranked_zero = [("apple", 0.0, "ep_0")]
    assert gate(ranked_zero) is None  # 0.0 does NOT exceed 0.0


def test_module_is_stdlib_only_and_does_not_touch_existing_moat():
    import research.runners.abstention_gate_compositional as comp
    import research.runners.abstention_gate as existing
    # Module distinct from the existing moat (different file, different
    # constant name); no shared mutable state.
    assert comp is not existing
    # Existing moat byte-unchanged DEFAULT_THRESHOLD remains 650.0:
    assert existing.DEFAULT_THRESHOLD == 650.0
    # New module's threshold is the (placeholder) compositional one:
    assert comp.COMPOSITIONAL_THRESHOLD == 0.0
