"""Soundness tests for the biologized spiking mode-unification runner.

Structural properties pinned here:
  (a) Gamma-slot positions are FIXED per seed (calling the helper
      twice yields byte-identical positions; the runner never
      regenerates them per-trial).
  (b) The deriver seed is pinned (DERIV_SEED == 90909 matching the
      FHRR-biologization arc).
  (c) The frozen 0.80 bar is imported unchanged, never redefined.
  (d) The runner's load-bearing constants match the design contract:
      N_GAMMA_SLOTS=7, N_CONCEPTS_PER_BRIDGE=32, K_VOCAB_TARGET=16,
      TEST_BRIDGE='bridgeA_nouns'.

Runtime-trace properties (both readouts share the SAME encoded C; the
recognition pathway / true labels never index the decoding) are best
verified by the dedicated adversarial reviewer at Task 4, by reading
the runner end-to-end. The pytest layer pins the structural
invariants only.
"""
import numpy as np

from research.findings.raw.biologized_spiking_mode_unification_runner import (
    DERIV_SEED, N_GAMMA_SLOTS, N_CONCEPTS_PER_BRIDGE, K_VOCAB_TARGET,
    TEST_BRIDGE, gamma_slot_positions,
)
from research.findings.raw.vocabulary_scaling_run import BAR


def test_gamma_slot_positions_fixed_per_seed():
    """The runner uses gamma_slot_positions(seed, 7, 512) once per
    seed; calling it again with the same args must return identical
    positions (so the runner's encoding is deterministic per seed)."""
    p1 = gamma_slot_positions(42, 7, 512)
    p2 = gamma_slot_positions(42, 7, 512)
    for a, b in zip(p1, p2):
        assert np.array_equal(a, b)


def test_deriver_seed_pinned():
    """The grounded symbols are derived via make_deriver(N_DIM, d_act,
    DERIV_SEED). DERIV_SEED must match the FHRR-biologization arc's
    seed (90909) so the symbol derivation is byte-identical to the
    validated grounded-composition pipeline."""
    assert DERIV_SEED == 90909


def test_frozen_bar_imported_unchanged():
    """BAR imported from vocabulary_scaling_run is the project's
    frozen 0.80 compositional bar; the mode-unification runner uses
    it unchanged for its verdict."""
    assert BAR == 0.80


def test_runner_constants_match_design_contract():
    """The runner's load-bearing constants must match the design
    doc's contract: 7 gamma slots (Lisman-Idiart), 32 concepts per
    bridge (matching the 160-ensemble per-bridge size), K=16 (the
    K=16 PASS recipe), bridgeA_nouns as the test bridge."""
    assert N_GAMMA_SLOTS == 7
    assert N_CONCEPTS_PER_BRIDGE == 32
    assert K_VOCAB_TARGET == 16
    assert TEST_BRIDGE == "bridgeA_nouns"
