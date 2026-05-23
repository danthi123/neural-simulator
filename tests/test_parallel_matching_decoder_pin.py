"""Grounding pin for the parallel-population-matching decoder arc.

Pins the frozen 0.80 bar; the K=16 PASS recipe constants imported
unchanged; the gamma-slot count (7), per-bridge concept count (32);
the test bridge (bridgeA_nouns matching the pre-registered mode-
unification runner so the comparison is head-to-head); the new
runner's public surface (red until Task 2 lands).
"""
from research.findings.raw.vocabulary_scaling_run import (
    BAR, LOADS, SEEDS, N_DIM, K_RECOG, K_VOCAB, N_TRIALS,
)


def test_compositional_bar_frozen():
    assert BAR == 0.80


def test_k16_pass_recipe_imported_unchanged():
    assert LOADS == [2, 3, 5]
    assert SEEDS == [42, 43, 44]
    assert K_RECOG == 8 and K_VOCAB == 8 and N_TRIALS == 200
    assert N_DIM == 512


def test_runner_constants_match_mode_unification_arc():
    """The parallel-matching runner uses the SAME substrate constants
    as the pre-registered mode-unification runner (so the comparison
    of identification mechanisms is head-to-head, not at a moved
    test grid)."""
    from research.findings.raw.biologized_spiking_mode_unification_parallel_matching_runner import (
        N_GAMMA_SLOTS, N_CONCEPTS_PER_BRIDGE, K_VOCAB_TARGET,
        TEST_BRIDGE, DERIV_SEED,
    )
    assert N_GAMMA_SLOTS == 7
    assert N_CONCEPTS_PER_BRIDGE == 32
    assert K_VOCAB_TARGET == 16
    assert TEST_BRIDGE == "bridgeA_nouns"
    assert DERIV_SEED == 90909  # same deriver as FHRR-biologization arc


def test_runner_module_exposes_public_surface():
    """Red until Task 2 lands."""
    from research.findings.raw import (
        biologized_spiking_mode_unification_parallel_matching_runner as m,
    )
    assert hasattr(m, "run_one_seed")
    assert hasattr(m, "main")
    # Must NOT use TPAM -- this is the parallel-matching alternative.
    import inspect
    src = inspect.getsource(m)
    assert "ResonateFireTPAM" not in src, (
        "parallel-matching runner must not use the TPAM attractor; "
        "the whole point is to replace it with parallel population "
        "matching")
    assert "settle_annealed" not in src, (
        "parallel-matching runner must not use the TPAM settle; the "
        "order-bearing decoder is feedforward argmax-of-similarities")
