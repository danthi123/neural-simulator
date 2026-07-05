"""Tests for RUNG B-1c -- the SPIKING reservoir put on the SAME UnifiedBrainBridge as the parser/composer/WTA, and
the read-out `Ws` realized as REAL SYNAPSES (reservoir -> WTA ensembles), removing the last host shortcuts in role
SELECTION.

FAST structural tests (no full seed run): the additive `reservoir_n` bridge support is byte-identical by default; the
reservoir wiring lays the slice + inhibitory subset; the Ws_shifted read-out is argmax-preserving (incl. the per-role
bias row -- the B-1c correction); the synaptic-readout source-check. Plus SLOW seed-42 gates for c1 (GO) and c2 (the
characterized on-substrate close-out).

Run fast only:   SIM_BACKEND=numpy python -m pytest tests/test_rungB1c_spiking_reservoir_synaptic_readout.py -m "not slow" -q
Run everything:  SIM_BACKEND=numpy python -m pytest tests/test_rungB1c_spiking_reservoir_synaptic_readout.py -q
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as B1C  # noqa: E402
from research.runners.unified_brain_bridge import (  # noqa: E402
    build_unified_bridge, UnifiedBrainBridge, PARSER_SLICE_SIZE, SYNAPTIC_ROUTE_ROLES,
)


# ── fast structural tests ────────────────────────────────────────────────────────────────────────────────────
def test_reservoir_n_default_byte_identical():
    """The additive `reservoir_n` support keeps the DEFAULT build (reservoir_n=0) byte-identical: same num_neurons
    (parser + 8*proj_dim composer) and num_traits=1 (no inhibitory trait allocated)."""
    b0 = build_unified_bridge(seed=42, proj_dim=32, enable_synaptic_route=False)
    assert b0.core_config.num_neurons == PARSER_SLICE_SIZE + 8 * 32
    assert b0.core_config.num_traits == 1                       # no WTA/reservoir -> single (excitatory) trait


def test_reservoir_n_allocates_slice_and_trait():
    """With reservoir_n set (+ role_wta_n), the bridge allocates the extra neurons past the WTA slice and switches to
    num_traits=2 (an inhibitory trait for the reservoir's inhibitory subset)."""
    proj = 32
    b = build_unified_bridge(seed=42, proj_dim=proj, enable_synaptic_route=True,
                             role_wta_n=B1C.ROLE_WTA_N, reservoir_n=B1C.RES_N)
    expect = PARSER_SLICE_SIZE + 8 * proj + len(SYNAPTIC_ROUTE_ROLES) * proj + B1C.ROLE_WTA_N + B1C.RES_N
    assert b.core_config.num_neurons == expect
    assert b.core_config.num_traits == 2
    ub = UnifiedBrainBridge(seed=42, proj_dim=proj, concepts={w: np.eye(proj)[i] for i, w in enumerate("abcdef")},
                            enable_synaptic_route=True, role_wta_n=B1C.ROLE_WTA_N, reservoir_n=B1C.RES_N)
    assert ub.reservoir_base is not None
    # the reservoir slice sits PAST the WTA slice
    assert ub.reservoir_base == ub.role_wta_base + B1C.ROLE_WTA_N


def test_wire_reservoir_builds_recurrence_and_inhibitory_subset():
    """wire_reservoir lays the reservoir slice, flips a ~20% inhibitory subset to trait 1, and installs a fixed-random
    recurrence (nonzero synapses) -- the LSM's fixed recurrent connectivity, wired RUNNER-SIDE."""
    from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder
    import research.runners._emerge62_discover_function_words_derisk as m62
    stream = m62.build_stream(42, n_sentences=2000)
    words, freq, cover, _c = m62.compute_stats(stream)
    discovered, *_ = m62.discover_closed_class(words, freq, cover)
    enc = Encoder(discovered)
    concepts = {w: np.eye(64)[i] for i, w in enumerate("abcdef")}
    ub = UnifiedBrainBridge(seed=42, proj_dim=64, concepts=concepts,
                            enable_synaptic_route=True, role_wta_n=B1C.ROLE_WTA_N, reservoir_n=B1C.RES_N)
    nnz0 = int(ub.bridge.cp_connections.nnz)
    res_idx, W_in = B1C.wire_reservoir(ub, enc.dim, seed=42)
    assert len(res_idx) == B1C.RES_N
    assert W_in.shape == (B1C.RES_N, enc.dim)
    # a ~20% inhibitory subset flipped to trait 1
    n_inh = int(np.asarray(ub.bridge.cp_traits[res_idx]).sum())
    assert abs(n_inh - int(round(0.2 * B1C.RES_N))) <= 1
    # the recurrence added synapses (fixed-random Erdos-Renyi at internal_density)
    assert int(ub.bridge.cp_connections.nnz) > nnz0


def test_ws_shifted_argmax_preserving_with_bias_row():
    """The Ws_shifted read-out preserves the host argmax EXACTLY when the +1 BIAS ROW is carried (the B-1c
    correction): shifted-positive argmax over (reservoir rows + the per-role bias row) == host argmax; DROPPING the
    bias row can flip it (so the bias row is genuinely per-role, not a role-independent constant)."""
    rng = np.random.default_rng(0)
    n_res = 40
    # a synthetic Ws with negative entries + a per-role bias row that FLIPS the argmax if dropped.
    W = rng.standard_normal((n_res + 1, 5))
    W[n_res, :] = np.array([5.0, -5.0, -5.0, 0.0, 0.0])   # a strong per-role intercept on role 0
    f = np.concatenate([np.abs(rng.standard_normal(n_res)) * 0.05, [1.0]])
    Wsh = W - W.min()
    host = int(np.argmax((f @ W)[[0, 1, 2]]))
    full = int(np.argmax(f[:n_res] @ Wsh[:n_res, :3] + f[n_res] * Wsh[n_res, :3]))   # incl. per-role bias tonic
    assert full == host                                     # argmax-preserving WITH the bias row
    # _ws_weights returns the per-role bias = scale * Ws_shifted[bias_row, :3]
    ens = [np.arange(0, 3), np.arange(3, 6), np.arange(6, 9)]
    _w, role_bias = B1C._ws_weights(np.arange(n_res), ens, Wsh, scale=2.0)
    assert np.allclose(role_bias, Wsh[n_res, :3] * 2.0)


def test_synaptic_readout_source_clean():
    """(anti-cheat 10, source half) the c2 SELECTION path has NO host `f @ Ws` / argmax over the read-out deciding
    the role: `_bind_c2` and `_op_wta_synaptic` never reference `Ws` in executable code; the winner is a NEURAL read
    of the ensembles' firing, and the composer field comes from the latched role."""
    assert B1C._source_synaptic_readout_clean() is True


def test_learned_readout_source_clean():
    """(mode c3, source) the read-out is BIOLOGICALLY LEARNED, not host-fit: `_learn_Ws_spiking` has NO host ridge/
    least-squares solve (no `linalg.solve`/`lstsq`/`pinv`) and never calls the c2 host ridge fitter `_fit_Ws_spiking`;
    it DRIVES the spiking reservoir (`run_with_ens`) and updates the res2ens synapses by a per-role LOCAL delta
    (`np.outer(target - actual_ens_firing, reservoir_firing)`). This removes the last host shortcut in the read-out
    itself (c2 still used a host ridge to DEFINE Ws; c3 learns Ws on the substrate)."""
    assert B1C._source_learned_readout_clean() is True


def test_learned_readout_returns_c2_format():
    """`_learn_Ws_spiking` returns Ws in the SAME {slot: (n_res+1) x n_roles} format `_fit_Ws_spiking` returns (bias
    row + GOAL/LOCATION cols = 0), so the whole c2 SlotReadout/_bind_c2/_op_wta_synaptic/anti-cheat machinery is reused
    unchanged. Structural check with a tiny (epochs=1, 1-sentence) learn so it stays fast."""
    from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _gen, _TRAIN_KINDS
    corpus = B1C.setup_corpus(seed=42)
    enc = Encoder(corpus["discovered"])
    rng = np.random.default_rng(42 * 101 + 5)
    train = _gen(_TRAIN_KINDS, 1, rng, corpus["subj"], corpus["verb"], corpus["obj"])
    ub, ens, _inh = B1C._build_wired_bridge(42, corpus, mode="c3")
    res_idx, W_in = B1C.wire_reservoir(ub, enc.dim, 42)
    res = B1C.UBReservoir(ub, res_idx, W_in); res.snapshot_after_wiring()
    Ws = B1C._learn_Ws_spiking(ub, res, ens, enc, train, 42, epochs=1)
    n_res = len(res_idx); n_roles = len(B1C._ROLES)
    assert set(Ws.keys()) == {0, 1, 2}
    for k in (0, 1, 2):
        assert Ws[k].shape == (n_res + 1, n_roles)
        assert np.allclose(Ws[k][n_res, :], 0.0)          # bias row zero (the learned rows carry the argmax)
        assert np.allclose(Ws[k][:, 3:], 0.0)             # GOAL/LOCATION cols zero (3-way canonical read)
        assert (Ws[k] >= 0.0).all()                       # Dale-legal (clipped >= 0)


# ── slow end-to-end gates ────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.slow
def test_seed42_c1_GO():
    """B-1c.1 (incremental): the SPIKING reservoir co-resident on the unified bridge + host f@Ws -> WTA. Full seed-42
    GO -- all 9 B-1b anti-cheats hold on the co-resident spiking substrate."""
    corpus = B1C.setup_corpus(seed=42)
    d = B1C.run_seed(42, corpus, mode="c1")
    assert d["route_recall"] >= 0.80, d
    assert d["route_not_worse_than_dict"], d
    assert d["moat_clean"], d
    assert d["provenance_clean"], d
    assert d["route_lesion_collapses"], d
    assert d["res_lesion_collapses"], d
    assert d["neural_select_latched_eq_firing"], d
    assert d["wta_lesion_collapses"], d
    assert d["ws_scramble_collapses"], d
    assert d["seed_GO"], d


@pytest.mark.slow
def test_seed42_c2_synaptic_readout_GO():
    """B-1c.2 (the FULL on-substrate close-out -- the B-1c.2 BOUNDARY SURPASSED). The SPIKING reservoir drives the WTA
    ensembles SYNAPTICALLY through the Ws_shifted read-out (NO host f@Ws), and the whole comprehend->select->bind turn
    runs on ONE bridge with nothing host-computed. Two coupled fixes surpass the prior boundary:
      (1) READ-OUT RESOLUTION (the CRUX): P=80 role ensembles (WTA_P_C2) + a T=30 read window (READ_T_STEP_C2) resolve
          the sub-1% post-Dale-offset margin, so route now recovers the fact EXACTLY as the host-dict on the same
          substrate (route 12/12 == dict, route-not-worse-than-dict reinstated).
      (2) RESERVOIR LOAD-BEARING (step-3 option a): the per-role BIAS intercept tonic is DROPPED (WS_BIAS_SCALE_C2=0) --
          it was a lesion-immune prior carrying the canonical roles even under lesion; at P=80/T=30 the reservoir rows
          alone resolve the argmax, so dropping it makes the reservoir GENUINELY LOAD-BEARING: the reservoir-lesion
          (SILENCE its W_in input map) COLLAPSES recall.
    Asserts the FULL seed-42 GO (all c2 anti-cheats)."""
    corpus = B1C.setup_corpus(seed=42)
    d = B1C.run_seed(42, corpus, mode="c2")
    assert d["route_correct"] == d["n_queries"], d         # route 12/12 == host-dict (the CRUX P=80/T=30 resolution)
    assert d["route_not_worse_than_dict"], d               # route not worse than the host-argmax dict on the substrate
    assert d["moat_clean"], d                              # no confabulation on unstored (agent, action)
    assert d["synaptic_source_clean"], d                   # no host f@Ws decides the role
    assert d["synaptic_readout_collapses"], d              # the synaptic read-out is load-bearing
    assert d["route_lesion_collapses"], d                  # the role route is load-bearing (gate-conditioned)
    assert d["res_lesion_collapses"], d                    # the RESERVOIR is load-bearing (silence-lesion collapses)
    assert d["ws_scramble_collapses"], d                   # the read-out routes the reservoir firing
    assert d["neural_select_latched_eq_firing"], d         # the winner is the neural read of the ensembles
    assert d["seed_GO"], d


@pytest.mark.slow
def test_seed42_c3_learned_readout_GO():
    """B-1c.3 (the FULL host-shortcut removal): the read-out matrix Ws is no longer a HOST RIDGE fit -- it is
    BIOLOGICALLY LEARNED on the frozen spiking reservoir by a per-role LOCAL delta rule (`W_k[r] += eta*(T_r - a_r)*rho`,
    a = ACTUAL spiking-ensemble firing incl. the f-I nonlinearity + WTA ignition-order INSIDE the error). This fixes the
    ridge's train/deploy objective MISMATCH (ridge = linear rate-reconstruction; deploy = spiking WTA whose winner is
    ignition-order) that made c2 seed-fragile. Seed 42 GO (== step6 18/18): route recovers the fact EXACTLY as the
    host-dict, BOTH source-checks hold (SELECT path + the LEARNED read-out), and the learned synapses are load-bearing."""
    corpus = B1C.setup_corpus(seed=42)
    d = B1C.run_seed(42, corpus, mode="c3")
    assert d["route_correct"] == d["n_queries"], d         # route recovers every fact as the host-dict (seed-42 clean)
    assert d["route_not_worse_than_dict"], d
    assert d["moat_clean"], d                              # no confabulation on unstored (agent, action)
    assert d["synaptic_source_clean"], d                   # the SELECT path is neural (no host f@Ws)
    assert B1C._source_learned_readout_clean(), d          # the read-out itself is LEARNED (no host ridge solve)
    assert d["synaptic_readout_collapses"], d              # the learned read-out synapses are load-bearing
    assert d["res_lesion_collapses"], d                    # the RESERVOIR is load-bearing (silence-lesion collapses)
    assert d["ws_scramble_collapses"], d
    assert d["neural_select_latched_eq_firing"], d
    assert d["seed_GO"], d
