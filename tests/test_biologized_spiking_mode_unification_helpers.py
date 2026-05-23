"""Unit tests for the biologized spiking mode-unification helpers.

`gamma_slot_positions(seed, n_slots, n_dim)` is a load-bearing pure
helper: any drift in determinism, shape, or pairwise structure would
silently change every gamma-slot encoding in the runner. These tests
pin shape, dtype, determinism, per-seed independence, and pairwise
near-orthogonality (the FHRR algebra's load-bearing assumption).
"""
import numpy as np

from research.findings.raw.biologized_spiking_mode_unification_helpers import (
    gamma_slot_positions,
)
from research.runners.spiking_phasor_fhrr import phase_similarity


def test_returns_n_slots_positions():
    pos = gamma_slot_positions(seed=42, n_slots=7, n_dim=512)
    assert len(pos) == 7


def test_each_position_is_ndarray_of_dim():
    pos = gamma_slot_positions(seed=42, n_slots=7, n_dim=512)
    for p in pos:
        assert isinstance(p, np.ndarray)
        assert p.shape == (512,)


def test_deterministic_in_seed():
    p1 = gamma_slot_positions(seed=42, n_slots=7, n_dim=512)
    p2 = gamma_slot_positions(seed=42, n_slots=7, n_dim=512)
    for a, b in zip(p1, p2):
        assert np.array_equal(a, b)


def test_per_seed_independence():
    """Different seeds must produce different position sets so the
    biologized arc can sweep seeds independently."""
    p42 = gamma_slot_positions(seed=42, n_slots=7, n_dim=512)
    p43 = gamma_slot_positions(seed=43, n_slots=7, n_dim=512)
    # At least one position differs.
    assert not all(np.array_equal(a, b) for a, b in zip(p42, p43))


def test_pairwise_near_orthogonal():
    """The 21 pairs of the 7 positions must be near-orthogonal in the
    FHRR sense: mean absolute phase-similarity below 0.15 (the FHRR
    capacity-curve regime at N_dim=512 puts random phasor pairs at
    a few percent overlap)."""
    pos = gamma_slot_positions(seed=42, n_slots=7, n_dim=512)
    sims = []
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            sims.append(abs(phase_similarity(pos[i], pos[j])))
    mean_sim = float(np.mean(sims))
    assert mean_sim < 0.15, (
        f"gamma-slot positions not near-orthogonal: mean abs phase-"
        f"similarity = {mean_sim:.4f}, expected < 0.15")
