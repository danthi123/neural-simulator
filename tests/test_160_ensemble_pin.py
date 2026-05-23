"""Grounding pin for the 160-concept ensemble vocab-scaling arc.

Pins the design doc contract that the frozen 0.80 bar is unchanged,
the K=16 PASS recipe is imported byte-unchanged, and the 5-bridge
ensemble has exactly 160 unique concepts. The runner-module-exists
check goes green only after Task 2 lands -- intentional: surfaces any
later drift in the runner's public surface the moment Task 2 is wired.
"""
from research.findings.raw.vocabulary_scaling_run import (
    BAR, LOADS, SEEDS, N_DIM, K_RECOG, K_VOCAB, N_TRIALS, N_CONCEPTS,
)
from research.runners.g20_vocab_spec import ALL_BRIDGES, TOTAL_VOCAB


def test_compositional_bar_frozen():
    """The frozen 0.80 compositional bar must not drift."""
    assert BAR == 0.80


def test_k16_pass_recipe_imported_unchanged():
    """The 64-concept K=16 PASS recipe (loads, seeds, n_trials,
    phasor dim, recognition windows) is the same recipe each bridge
    of the 160-ensemble runs under. Pinned so neither can silently
    drift. (K_VOCAB the constant is the project default 8; the 160-
    ensemble runner explicitly passes K_VOCAB=16 to run_pipeline,
    matching the K=16 PASS recipe.)"""
    assert LOADS == [2, 3, 5]
    assert SEEDS == [42, 43, 44]
    assert K_RECOG == 8 and K_VOCAB == 8 and N_TRIALS == 200
    assert N_DIM == 512
    # 64-concept test grid still pinned; the 160-ensemble runner
    # uses its own per-bridge concept count = 32, not the global
    # N_CONCEPTS = 64.
    assert N_CONCEPTS == 64


def test_five_bridges_at_32_concepts_each_160_total():
    """The validated 5-bridge sparse-distributed concept ensemble has
    exactly 5 bridges × 32 concepts each = 160 unique concepts, with
    global uniqueness across all bridges."""
    assert TOTAL_VOCAB == 160
    assert sorted(ALL_BRIDGES.keys()) == sorted([
        "bridgeA_nouns", "bridgeB_verbs", "bridgeC_adj",
        "bridgeD_spatial", "bridgeE_functional",
    ])
    for name, vocab in ALL_BRIDGES.items():
        assert len(vocab) == 32, f"{name} has {len(vocab)} words"
    all_words = [w for v in ALL_BRIDGES.values() for w in v]
    assert len(all_words) == 160
    assert len(set(all_words)) == 160


def test_runner_module_exists():
    """The 160-ensemble runner must exist and expose its three
    load-bearing public symbols. Red until Task 2 lands -- intentional:
    surfaces any drift in Task 2's public surface."""
    from research.findings.raw import (
        vocabulary_scaling_run_160ensemble as m,
    )
    assert hasattr(m, "bridge_vocab_and_patterns")
    assert hasattr(m, "run_one_bridge_seed")
    assert hasattr(m, "main")
