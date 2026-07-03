"""CPU tests for EMERGE-80 -- the spiking-LSM port (recurrent Izhikevich liquid-state machine) of the EMERGE-78 reservoir.

Verifies the SpikingLSM (genuinely active pool, deterministic, silence-collapse) and (slow) the seed-42 de-risk: the
spiking pool learns the form->role map + resolves the non-local rel-clause head where both fixed-window baselines are at
chance, with the read genuinely from pool spikes (pool-silence lesion collapses it). CPU/numpy, offline.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import pytest

import research.runners._emerge80_spiking_lsm_port_derisk as m80
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder


def test_spiking_pool_is_genuinely_active():
    enc = Encoder({"the", "to", "on", "can"})
    lsm = m80.SpikingLSM(enc.dim, seed=42, n=200)
    U = enc.encode(["the", "dog", "zeps", "to", "the", "pond"])
    state = lsm.final_state(U)
    assert state.shape == (lsm.n,)
    assert lsm._last_mean_spikes > 0.3, lsm._last_mean_spikes        # the Izhikevich pool actually spikes


def test_pool_silence_collapses_activity():
    enc = Encoder({"the", "to", "on"})
    lsm = m80.SpikingLSM(enc.dim, seed=42, n=200)
    U = enc.encode(["the", "dog", "zeps", "the", "cat"])
    lsm.final_state(U)
    active = lsm._last_mean_spikes
    s = lsm.final_state(U, silence=True)                            # zero all drive -> no spikes
    assert lsm._last_mean_spikes == 0.0 and active > 0.0
    assert np.all(s == 0.0)


def test_spiking_lsm_deterministic_per_seed():
    r1 = m80.SpikingLSM(16, seed=42, n=100)
    r2 = m80.SpikingLSM(16, seed=42, n=100)
    assert np.allclose(r1.W_rec, r2.W_rec) and np.allclose(r1.W_in, r2.W_in)
    assert not np.allclose(r1.W_rec, m80.SpikingLSM(16, seed=43, n=100).W_rec)


def test_uses_project_izhikevich_operating_point():
    # the pool uses the project's Izhikevich 2007 RS cortical params (not an ad-hoc unit)
    assert m80._IZH["C"] == 100.0 and m80._IZH["k"] == 0.7 and m80._IZH["vr"] == -60.0 and m80._IZH["v_peak"] == 35.0


@pytest.mark.slow
def test_seed42_spiking_lsm_learns_and_resolves_nonlocal():
    d = m80._derisk_one(42)
    assert d["mean_spikes_per_neuron"] > 0.5                        # genuinely spiking
    assert d["train_acc"] >= 0.90                                   # learns the form->role map on spikes
    assert d["relhead_reservoir"] >= 0.85                           # resolves the non-local rel-head
    assert d["relhead_gov_baseline"] <= 0.65 and d["relhead_symwin_baseline"] <= 0.65   # both baselines at chance
    assert (d["relhead_reservoir"] - d["silence_lesion_acc"]) >= 0.20   # read genuinely from pool spikes
    assert d["relhead_scramble"] <= d["chance_binary"] + 0.18       # reads structure
