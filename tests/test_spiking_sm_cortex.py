"""Task 1 tests: the cortex-bridge builder + PPMI-shaped input encoder.

Runs CPU-only (SIM_BACKEND=numpy) so it is fast and deterministic.
"""
import os

os.environ["SIM_BACKEND"] = "numpy"  # CPU-only, set BEFORE importing the runner

import numpy as np

from research.runners.spiking_sm_cortex import build_sm_cortex_bridge, encode_drive


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
