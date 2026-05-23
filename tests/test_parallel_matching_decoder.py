"""Soundness tests for the parallel-population-matching decoder
runner.

Structural properties pinned here:
- The runner does NOT import or use the TPAM attractor (the WHOLE
  point of this build is to test the alternative biology-grounded
  identification mechanism).
- The runner's substrate/encoding constants match the mode-
  unification runner exactly (so the comparison is head-to-head).
- The order-bearing decoder uses the full vocabulary (not
  restricted to true items).

Runtime-trace properties (both readouts share the SAME encoded C;
true items never index the decoder) are best verified by the
dedicated adversarial reviewer.
"""
from research.findings.raw.biologized_spiking_mode_unification_parallel_matching_runner import (
    DERIV_SEED, N_GAMMA_SLOTS, N_CONCEPTS_PER_BRIDGE, K_VOCAB_TARGET,
    TEST_BRIDGE, M_OBS_FULL,
)
from research.findings.raw.vocabulary_scaling_run import BAR
from research.findings.raw.vocabulary_scaling_run_trained import (
    N_TRAIN_EVENTS,
)


def test_no_TPAM_import_or_usage_in_runner():
    """The runner must not IMPORT or INSTANTIATE the TPAM attractor
    -- the whole point is to test the alternative biology-grounded
    identification mechanism (parallel population matching), not the
    TPAM. (Docstring/comment mentions of "ResonateFireTPAM" explaining
    the design choice are OK; the check is on actual usage.)"""
    import inspect
    from research.findings.raw import (
        biologized_spiking_mode_unification_parallel_matching_runner as m,
    )
    src = inspect.getsource(m)
    assert "from research.runners.resonate_fire_fhrr import ResonateFireTPAM" not in src
    assert "ResonateFireTPAM(" not in src
    assert ".settle_annealed(" not in src
    assert "ANNEAL_THETA_LOW" not in src and "ANNEAL_THETA_HIGH" not in src


def test_substrate_constants_match_mode_unification_runner():
    """Head-to-head comparison: substrate sizing, K=16 recipe,
    deriver seed, test bridge must all match the pre-registered
    mode-unification runner exactly."""
    from research.findings.raw.biologized_spiking_mode_unification_runner import (
        DERIV_SEED as MU_DERIV_SEED,
        N_GAMMA_SLOTS as MU_GAMMA_SLOTS,
        N_CONCEPTS_PER_BRIDGE as MU_N_CONCEPTS,
        K_VOCAB_TARGET as MU_K_VOCAB,
        TEST_BRIDGE as MU_BRIDGE,
        M_OBS_FULL as MU_M_OBS,
    )
    assert DERIV_SEED == MU_DERIV_SEED == 90909
    assert N_GAMMA_SLOTS == MU_GAMMA_SLOTS == 7
    assert N_CONCEPTS_PER_BRIDGE == MU_N_CONCEPTS == 32
    assert K_VOCAB_TARGET == MU_K_VOCAB == 16
    assert TEST_BRIDGE == MU_BRIDGE == "bridgeA_nouns"
    assert M_OBS_FULL == MU_M_OBS == 16


def test_K16_PASS_recipe_pinned_in_runner():
    assert K_VOCAB_TARGET == 16
    assert N_TRAIN_EVENTS == 400
    assert BAR == 0.80


def test_runner_uses_phase_similarity_for_decoder():
    """The order-bearing decoder must use phase_similarity (the FHRR
    primitive) over the full vocabulary -- this is the feedforward
    parallel-matching mechanism, not an oracle table lookup."""
    import inspect
    from research.findings.raw import (
        biologized_spiking_mode_unification_parallel_matching_runner as m,
    )
    src = inspect.getsource(m)
    # The decoder must score against grounded[w] for w in words (full
    # vocab); not against any restricted subset.
    assert "phase_similarity(unbinds[k], grounded[w])" in src
    assert "for w in words" in src
