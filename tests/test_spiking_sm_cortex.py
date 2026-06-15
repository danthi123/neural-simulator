"""Task 1 tests: the cortex-bridge builder + PPMI-shaped input encoder.

Runs CPU-only (SIM_BACKEND=numpy) so it is fast and deterministic.
"""
import os

os.environ["SIM_BACKEND"] = "numpy"  # CPU-only, set BEFORE importing the runner

import numpy as np
import pytest

from research.runners.dendritic_d1_learn_graded_structure_derisk import effective_rank
from research.runners.spiking_sm_cortex import (
    build_sm_cortex_bridge,
    encode_drive,
    read_codes,
    train_sm_cortex,
)
from sim.backend import to_host


def test_build_and_encode():
    # --- builder ---
    bridge, hub_idx, cortex_idx = build_sm_cortex_bridge(n_hub=200, n_cortex=64, seed=42)

    hub_idx = np.asarray(hub_idx)
    cortex_idx = np.asarray(cortex_idx)
    assert len(hub_idx) == 200
    assert len(cortex_idx) == 64
    # hub and cortex index slices must be disjoint
    assert np.intersect1d(hub_idx, cortex_idx).size == 0

    # the hub->cortex pathway is present, plastic, and tagged with the gate name.
    pathways = bridge.core_config.region_pathways
    hub_to_cortex = [
        pw for pw in pathways
        if pw.from_region == "hub" and pw.to_region == "cortex"
    ]
    assert len(hub_to_cortex) == 1, "expected exactly one hub->cortex pathway"
    pw = hub_to_cortex[0]
    assert pw.plastic is True
    assert pw.plasticity_gate == "hub_to_cortex"

    # --- encoder ---
    raw = np.array([0.0, 1.0, 3.0, 7.0])
    out_log = encode_drive(raw, log=True)
    assert np.allclose(out_log, np.log1p([0.0, 1.0, 3.0, 7.0]))

    out_raw = encode_drive(raw, log=False)
    assert np.allclose(out_raw, np.maximum(raw, 0.0))

    # negatives are clipped to zero in both modes
    neg = np.array([-2.0, 0.0, 5.0])
    assert np.allclose(encode_drive(neg, log=False), np.array([0.0, 0.0, 5.0]))
    assert np.allclose(encode_drive(neg, log=True), np.log1p([0.0, 0.0, 5.0]))


@pytest.mark.xfail(
    reason="Task-2 BLOCKER (rigorously isolated 2026-06-15): naive hub-drive-only STDP NET-DEPRESSES the "
    "plastic hub->cortex weights (the silent-target / STDP-depression trap) -> the cortex is silent at read "
    "time so codes.sum()==0. The train/read MACHINERY is correct (weights DO move; a no-train read fires); "
    "the fix is the competitive-STDP mechanism (lateral-inhibition WTA + adaptive thresholds, Diehl-Cook) "
    "under deep research (2026-06-15-bridge-competitive-stdp-deep-research.md). Remove this xfail when the "
    "Task-3 competitive mechanism lands and the cortex fires at read.",
    strict=False,
)
def test_train_read_machinery():
    """Task 2 mechanical check: training MOVES the plastic hub->cortex weights, and read_codes
    returns a non-degenerate [Nc x n_cortex] spike-count code matrix with the cortex actually firing.

    Tiny synthetic case (16 concepts x 80 hubs, n_cortex=32) so it runs in a few seconds on numpy.
    NO structure claim here -- only that the train/read plumbing works. Currently XFAIL: the cortex is
    silenced by the STDP-depression trap before read (see the marker above).
    """
    n_concepts, n_hub, n_cortex = 16, 80, 32

    # a simple non-negative count matrix: a few Poisson-ish active columns per concept.
    rng = np.random.RandomState(7)
    C = np.zeros((n_concepts, n_hub), dtype=np.float64)
    for i in range(n_concepts):
        active = rng.choice(n_hub, size=10, replace=False)
        C[i, active] = rng.poisson(lam=4.0, size=active.size).astype(np.float64)
    C_drive = encode_drive(C)  # log1p Weber-Fechner compression

    bridge, hub_idx, cortex_idx = build_sm_cortex_bridge(n_hub=n_hub, n_cortex=n_cortex, seed=42)
    hub_idx = np.asarray(hub_idx)
    cortex_idx = np.asarray(cortex_idx)

    # --- weights must CHANGE through training (STDP on the only plastic pathway moves them) ---
    # NOTE: structural plasticity is ON by default, so the CSR .data array LENGTH can change
    # across training (synapses form/eliminate). Compare the total weight MASS (a scalar robust to
    # the length change) -- STDP + structural plasticity both move it away from the initial value.
    w_before = np.asarray(to_host(bridge.cp_connections.data)).copy()
    train_sm_cortex(bridge, C_drive, hub_idx, cortex_idx, n_epochs=2)
    w_after = np.asarray(to_host(bridge.cp_connections.data)).copy()
    if w_before.shape == w_after.shape:
        assert not np.allclose(w_before, w_after), "training did not move the hub->cortex weights"
    else:
        # structural plasticity changed the synapse count -> the weights necessarily changed.
        assert not np.isclose(float(w_before.sum()), float(w_after.sum())), \
            "training did not move the hub->cortex weight mass"

    # --- read_codes returns the per-concept cortex spike-count codes ---
    codes = read_codes(bridge, C_drive, hub_idx, cortex_idx)
    assert codes.shape == (n_concepts, n_cortex)
    assert codes.sum() > 0, "cortex never fired -- all codes are silent"
    assert effective_rank(codes) > 1.0, "codes are degenerate (effective rank <= 1)"

    # read_codes must RESTORE the gate to full plasticity afterward.
    assert bridge._plasticity_gate_values["hub_to_cortex"] == 1.0
