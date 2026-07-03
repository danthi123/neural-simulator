"""CPU tests for EMERGE-87 -- co-residence of the form->role reservoir region with a conversational region on ONE bridge.

Structural tests (a 2-region disjoint bridge, no cross-region pathways) + a slow single-seed smoke asserting the reservoir
GOes co-resident and its form->role RESULT is functionally isolated from a concurrently-driven conversational region.
CPU/numpy, offline.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import pytest

import research.runners._emerge87_reservoir_coresident_derisk as m87


def test_builds_two_disjoint_regions_no_cross_pathways():
    from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder
    enc = Encoder({"the", "to", "on"})
    lsm = m87.CoResidentReservoirLSM(enc.dim, seed=42, n=80)
    assert len(lsm.res_idx) == 80 and len(lsm.conv_idx) == m87._N_CONV
    assert set(lsm.res_idx).isdisjoint(set(lsm.conv_idx))        # the two slices are disjoint
    # no cross-region pathways in the built bridge config
    assert lsm.bridge.core_config.region_pathways == []


def test_conv_slice_silent_when_only_reservoir_driven():
    from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder
    enc = Encoder({"the", "to", "on"})
    lsm = m87.CoResidentReservoirLSM(enc.dim, seed=42, n=100)
    U = enc.encode(["the", "dog", "zeps", "to", "the", "pond"])
    lsm.final_state(U, conv_drive=False)                          # drive only the reservoir
    assert lsm._last_conv_spikes <= 0.5                          # the conv slice stays silent (isolated)
    assert lsm._last_mean_spikes > 0.3                           # the reservoir is active


@pytest.mark.slow
def test_seed42_reservoir_goes_coresident_and_is_functionally_isolated():
    d = m87._derisk_one(42)
    assert d["mean_spikes_per_neuron"] > 0.5
    assert d["train_acc"] >= 0.90 and d["relhead"] >= 0.85       # form->role GOes co-resident
    assert d["relhead_gov"] <= 0.65 and d["relhead_symwin"] <= 0.65
    # the co-resident conversation does NOT change the reservoir's form->role RESULT (functional isolation)
    assert d["isolation_class_flip_rate"] <= 0.02
    assert d["conv_silent_when_res_only"] <= 0.5
