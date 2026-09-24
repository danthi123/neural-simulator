"""Pins the gap#4 on-bridge feedforward-transmission defect found 2026-09-24 (plan step S19).

With the engine's default Tsodyks-Markram short-term depression ON, the explicit feedforward synapses of the on-bridge
BDSP net (OnBridgeBDSPNet -> Gap4InEngineNet -> Gap4ReadoutNet) carry almost nothing at the operating rates: zeroing
the input->H1 weights leaves H1's read unchanged, so no arm's learning can reach the output read. `--no-ff-stp` (plus
the stronger feedforward gain it is calibrated with) restores transmission. This test fails if either direction
changes: the legacy net starts transmitting (the diagnosis would be stale) or the fix stops transmitting.

Dev seed 7, tiny net, numpy; a few seconds.
"""
import os

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")


def _h1_read_change(**kw):
    from research.runners._gap4_transport_ceiling_readout_derisk import Gap4ReadoutNet
    from research.runners._semantic_inheritance_deep_credit_derisk import make_task_semantic_inheritance
    from sim.backend import to_host
    (Xtr, ytr, _), _te, meta, _idx = make_task_semantic_inheritance(7, n_super=8, n_members=4, held_per_super=1,
                                                                    n_prop=2, member_id_dim=3, n_obs=4)
    k = int(meta["k_classes"])
    net = Gap4ReadoutNet(Xtr.shape[1], 8, k, seed=7, feedback="reservoir", n_hidden_layers=2, pool_k=2,
                         settle_steps=40, credit_steps=5, graded_credit=True, read_quantity="spikes", read_window=30,
                         eval_frozen=True, tonic_h_pA=0.0, tonic_o_pA=0.0, **kw)
    coo = net.br._get_cached_coo()
    row = np.asarray(to_host(coo.row)); col = np.asarray(to_host(coo.col))
    pre, post = net._ff_edges[0]
    m = (row >= pre[0]) & (row <= pre[-1]) & (col >= post[0]) & (col <= post[-1])
    w0 = np.asarray(to_host(net.br.cp_connections.data)).copy()
    X = Xtr[:12]

    def h1():
        with net.frozen_reads():
            return float(np.mean(net._forward_batch(X)[1]))
    intact = h1()
    w = w0.copy(); w[m] = 0.0
    net.br.cp_connections.data[...] = net._xp.asarray(w)
    zeroed = h1()
    return intact, zeroed


def test_legacy_feedforward_barely_transmits_and_the_bypass_restores_it():
    legacy_intact, legacy_zeroed = _h1_read_change()
    fixed_intact, fixed_zeroed = _h1_read_change(no_ff_stp=True, no_structural=True, ff_w_init=40.0,
                                                  propagation_strength=0.5)
    gain_intact, gain_zeroed = _h1_read_change(no_structural=True, ff_w_init=40.0, propagation_strength=0.5)
    print("legacy", legacy_intact, legacy_zeroed, "gain-only(STP on)", gain_intact, gain_zeroed,
          "bypass", fixed_intact, fixed_zeroed)
    # legacy: with no tonic drive, H1 barely fires whether or not its input weights exist
    assert abs(legacy_intact - legacy_zeroed) < 0.01, (legacy_intact, legacy_zeroed)
    # the same 10x feedforward gain with short-term depression ON still does not transmit (STP owns the block)
    assert abs(gain_intact - gain_zeroed) < 0.01, (gain_intact, gain_zeroed)
    # bypass: the input now drives H1, and removing the input weights silences it
    assert fixed_intact - fixed_zeroed > 0.01, (fixed_intact, fixed_zeroed)
