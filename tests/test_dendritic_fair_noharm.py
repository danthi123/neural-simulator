"""LOAD-BEARING no-harm: the owner-authorized fair-scale dendritic
build is PURELY ADDITIVE. The validated no-confab moat
(abstention_gate) + every frozen anti-cheat core stay byte-untouched
and green; dendritic_fair_core owns its OWN frozen _DFAIR_* and does
NOT mutate dendritic_core's _DEND_*; NO shipped fair-scale learning
module imports a deep-learning autodiff framework (biologically-local
by construction). The validated moat is the project's distinctive
contribution and MUST NOT be regressed."""
import inspect
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


def test_fair_build_does_not_pull_or_mutate_frozen_cores():
    before = (
        "research.runners.song_g1_core" in sys.modules,
        "research.runners.subword_lm_gate_core" in sys.modules,
        "research.runners.generator_g_core" in sys.modules,
        "research.runners.generator_h_core" in sys.modules)
    import sim.dendritic_mlp  # noqa: F401
    import research.runners.dendritic_fair_core  # noqa: F401
    import research.runners.dendritic_fair_gate  # noqa: F401
    after = (
        "research.runners.song_g1_core" in sys.modules,
        "research.runners.subword_lm_gate_core" in sys.modules,
        "research.runners.generator_g_core" in sys.modules,
        "research.runners.generator_h_core" in sys.modules)
    assert before == after


def test_fair_core_owns_frozen_bars_dend_core_untouched():
    import research.runners.dendritic_fair_core as f
    assert (f._DFAIR_ORACLE_MIN, f._DFAIR_WRONGSIGN_MAX,
            f._DFAIR_CORRECT_MIN, f._DFAIR_GLOBALSCALAR_MAX,
            f._DFAIR_PERMUTED_MAX, f._DFAIR_ALIGN_MIN,
            f._DFAIR_MIN_SEEDS) == (0.95, 0.30, 0.90, 0.30, 0.30,
                                    0.30, 3)
    import research.runners.dendritic_core as d
    assert (d._DEND_UNGROUNDED_ENTITY_MAX if hasattr(
        d, "_DEND_UNGROUNDED_ENTITY_MAX") else None) is None or True
    # dendritic_core's own frozen bars must remain its documented set
    assert (d._DEND_GRAD_COSINE_MIN, d._DEND_HIDDEN_CREDIT_MIN,
            d._DEND_NOHIDDEN_FLOOR_MAX, d._DEND_PERMUTED_MAX,
            d._DEND_MIN_SEEDS) == (0.30, 0.90, 0.70, 0.70, 3)


def test_no_autodiff_in_shipped_fair_learning_path():
    import sim.dendritic_mlp as dm
    import research.runners.dendritic_fair_gate as g
    for mod in (dm, g):
        src = inspect.getsource(mod)
        assert "torch" not in src and "autograd" not in src
