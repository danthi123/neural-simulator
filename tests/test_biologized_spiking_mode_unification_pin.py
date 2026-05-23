"""Grounding pin for the biologized spiking mode-unification arc.

Pins the frozen 0.80 bar, the K=16 PASS recipe constants imported
unchanged, the gamma-slot count (7, the Lisman-Idiart biologically
grounded value), per-bridge concept count (32, matching the
160-ensemble per-bridge size), multi-seed grid, loads, FHRR phasor
dim (512), and the runner's public surface (red until Task 2 lands).
"""
from research.findings.raw.vocabulary_scaling_run import (
    BAR, LOADS, SEEDS, N_DIM, K_RECOG, K_VOCAB, N_TRIALS,
)


def test_compositional_bar_frozen():
    """The frozen 0.80 compositional bar must not drift."""
    assert BAR == 0.80


def test_k16_pass_recipe_imported_unchanged():
    """The K=16 PASS recipe constants (loads, seeds, n_trials, phasor
    dim, recognition window) imported unchanged from the trained-
    substrate runner. K_VOCAB module constant is the project default
    8; the mode-unification runner explicitly passes K_VOCAB=16
    (the K=16 PASS recipe)."""
    assert LOADS == [2, 3, 5]
    assert SEEDS == [42, 43, 44]
    assert K_RECOG == 8 and K_VOCAB == 8 and N_TRIALS == 200
    assert N_DIM == 512


def test_gamma_slot_count_lisman_idiart():
    """N_GAMMA_SLOTS = 7 is the catalog-documented Lisman-Idiart 1995
    biologically grounded value (5-7 gamma slots per theta cycle).
    The runner uses 7."""
    from research.findings.raw.biologized_spiking_mode_unification_runner import (
        N_GAMMA_SLOTS,
    )
    assert N_GAMMA_SLOTS == 7


def test_per_bridge_concept_count():
    """The runner tests on 1 bridge at 32 concepts (matching the
    160-ensemble per-bridge size)."""
    from research.findings.raw.biologized_spiking_mode_unification_runner import (
        N_CONCEPTS_PER_BRIDGE,
    )
    assert N_CONCEPTS_PER_BRIDGE == 32


def test_runner_module_exists():
    """The mode-unification runner must exist and expose its three
    load-bearing public symbols. Red until Task 2 lands."""
    from research.findings.raw import (
        biologized_spiking_mode_unification_runner as m,
    )
    assert hasattr(m, "gamma_slot_positions")
    assert hasattr(m, "run_one_seed")
    assert hasattr(m, "main")
