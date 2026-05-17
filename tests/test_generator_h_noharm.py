"""LOAD-BEARING no-harm: Generator-H is PURELY ADDITIVE; the validated
no-confab moat (abstention_gate) + the frozen anti-cheat cores stay
byte-untouched and green; NO new global bar; no song_g1_core /
subword_lm_gate_core / generator_g_core pull. The validated moat is
the project's distinctive contribution and MUST NOT be regressed."""
import sys


def test_validated_moat_byte_contract_intact():
    from research.runners.abstention_gate import (
        gate, abstain, DEFAULT_THRESHOLD)
    assert DEFAULT_THRESHOLD == 650.0
    assert abstain(650.0, 650.0) is True
    assert abstain(650.01, 650.0) is False
    assert gate([("x", 700.0, "t")], 650.0) == ("x", 700.0, "t")
    assert gate([("x", 600.0, "t")], 650.0) is None
    assert gate([], 650.0) is None


def test_generator_h_does_not_pull_protected_cores():
    before = (
        "research.runners.song_g1_core" in sys.modules,
        "research.runners.subword_lm_gate_core" in sys.modules,
        "research.runners.generator_g_core" in sys.modules)
    import sim.constrained_realize  # noqa: F401
    import research.runners.generator_h_core  # noqa: F401
    import research.runners.generator_h_gate  # noqa: F401
    after = (
        "research.runners.song_g1_core" in sys.modules,
        "research.runners.subword_lm_gate_core" in sys.modules,
        "research.runners.generator_g_core" in sys.modules)
    assert before == after


def test_gh_core_owns_its_frozen_bars():
    import research.runners.generator_h_core as c
    assert (c._GH_UNGROUNDED_ENTITY_MAX, c._GH_MIN_COVERAGE,
            c._GH_MAX_REPEAT, c._GH_MIN_GROUNDED_ANSWER_RATE,
            c._GH_MIN_SEEDS) == (0.20, 1.0, 0.50, 0.5, 3)
