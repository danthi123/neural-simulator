"""Grounding pin for the pattern-grounded symbol arc.

Pins the design doc's contract that the frozen 0.80 compositional bar
is unchanged and the test grid (multi-seed, loads {2,3,5}, 64 concepts,
validated deriver dimension 512) is identical to the trained-substrate
decisive run. The module-exists check goes green only after Task 2 is
in place -- intentional: a Task 0 failure surfaces any drift in the
load-bearing constants the moment a later task is wired.
"""
from research.findings.raw.vocabulary_scaling_run import (
    BAR, LOADS, SEEDS, N_DIM, N_CONCEPTS, K_RECOG, K_VOCAB, N_TRIALS,
)


def test_compositional_bar_frozen():
    """The frozen 0.80 compositional bar must not drift."""
    assert BAR == 0.80


def test_test_grid_unchanged():
    """The multi-seed grid, loads, vocabulary size, FHRR phasor
    dimension, and recognition/registration settings must match the
    trained-substrate decisive run so the pattern-grounded result is
    strictly comparable."""
    assert N_CONCEPTS == 64
    assert LOADS == [2, 3, 5]
    assert SEEDS == [42, 43, 44]
    assert N_DIM == 512
    assert K_RECOG == 8 and K_VOCAB == 8 and N_TRIALS == 200


def test_pattern_grounded_runner_module_exists():
    """The pattern-grounded runner must exist and expose its three
    load-bearing public symbols. Red until Task 2 lands -- intentional:
    this pin surfaces if Task 2's public surface drifts."""
    from research.findings.raw import (
        vocabulary_scaling_run_pattern_grounded as m,
    )
    assert hasattr(m, "pattern_vector")
    assert hasattr(m, "_ground_symbols_pattern")
    assert hasattr(m, "run_one_seed_pattern")
