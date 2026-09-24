"""Pins for research/runners/ca3_superposed_fact_attractor.py (a default-off research runner).

Each test names the review class it guards:
  * the covariance write is a LOCAL online rule (Welford identity), not a batch host computation;
  * storage is SHARED (a later fact changes synapses an earlier fact's recall reads);
  * the read never sees the answer (no host routing by the patient / exact key);
  * the seed controls every random draw (the cfg.seed trap);
  * the lesion instruments do what they claim (zero -> all-zero drive, shuffle preserves each row's values);
  * every gate can FAIL (a localist, flat, or censored curve is rejected; UNDEFINED is never a pass).
"""
import numpy as np
import pytest

from research.runners import ca3_superposed_fact_attractor as M

TINY = dict(n_ec_role=200, k_ec=8, n_ent=100, n_rel=8, n_ca3=1000, c_rec=200, c_pp=60, c_out=200, n_dg=3000,
            a_dg=0.01, c_ecdg=40, T=5, p_grid=(20, 40), n_query=20, n_recent=10, n_novel=10, n_xtalk=10,
            encode_batch=20)


def _tiny(arm="sparse_dg", seed=42, **kw):
    over = dict(TINY)
    if arm == "sparse_dg_c2":
        over.update(c_rec=400, c_pp=120, c_out=400)
    if arm == "dense_nodg":
        over.update(a_ca3=0.05)
    over.update(kw)
    return M.make_cfg(arm, seed, **over)


def test_welford_identity_is_the_online_local_rule():
    rng = np.random.default_rng(0)
    proj = M.Projection(30, 40, 12, np.random.default_rng(1), "covariance")
    n_post, n_pre = 30, 40
    welford = np.zeros((n_post, 12))
    cnt_post = np.zeros(n_post)
    cnt_pre = np.zeros(n_pre)
    for t in range(1, 26):
        post = rng.random(n_post) < 0.2
        pre = rng.random(n_pre) < 0.3
        m_post_old = cnt_post / max(t - 1, 1) if t > 1 else np.zeros(n_post)
        cnt_post += post
        cnt_pre += pre
        m_pre_new = cnt_pre / t
        welford += (post[:, None] - m_post_old[:, None]) * (pre[proj.idx] - m_pre_new[proj.idx])
        proj.write(post, pre)
    np.testing.assert_allclose(proj.values(), welford, rtol=1e-4, atol=1e-4)


def test_storage_is_shared_between_facts():
    net = M.Network(_tiny())
    F, _ = M.draw_facts(net, 2, 1)
    X = net.ec_in(F, with_patient=True)
    Y = net.ec_out_patient(F)
    ca3, _ = net.encode_ca3(X)
    net.write_fact(ca3[:, 0], X[:, 0], Y[:, 0])
    w1 = net.rec.values().copy()
    net.write_fact(ca3[:, 1], X[:, 1], Y[:, 1])
    w2 = net.rec.values()
    rows_of_fact0 = np.flatnonzero(ca3[:, 0])
    # the second write changes synapses in rows fact 0's recall reads (the covariance threshold moves every row)
    assert not np.array_equal(w1[rows_of_fact0], w2[rows_of_fact0])
    assert net.synapse_bytes() == M.Network(_tiny()).synapse_bytes()   # no per-fact allocation


def test_read_never_sees_the_answer():
    cfg = _tiny()
    net = M.Network(cfg)
    F, _ = M.draw_facts(net, 20, 1)
    X = net.ec_in(F, with_patient=True)
    ca3, _ = net.encode_ca3(X)
    Y = net.ec_out_patient(F)
    for q in range(len(F)):
        net.write_fact(ca3[:, q], X[:, q], Y[:, q])
    G = F.copy()
    G[:, 2] = np.roll(G[:, 2], 7)
    _, s1, y1, _ = net.recall(F)
    _, s2, y2, _ = net.recall(G)
    assert np.array_equal(y1, y2) and np.array_equal(s1, s2)


def test_seed_controls_every_draw():
    a, b, c = M.Network(_tiny(seed=42)), M.Network(_tiny(seed=42)), M.Network(_tiny(seed=43))
    assert np.array_equal(a.rec.idx, b.rec.idx) and np.array_equal(a.mossy.W, b.mossy.W)
    assert all(np.array_equal(x, y) for x, y in zip(a.ent_codes, b.ent_codes))
    assert not np.array_equal(a.rec.idx, c.rec.idx)
    fa, _ = M.draw_facts(a, 20, 5)
    fb, _ = M.draw_facts(b, 20, 5)
    assert np.array_equal(fa, fb)


def test_lesion_instruments():
    net = M.Network(_tiny())
    F, _ = M.draw_facts(net, 20, 1)
    X = net.ec_in(F, with_patient=True)
    ca3, _ = net.encode_ca3(X)
    Y = net.ec_out_patient(F)
    for q in range(len(F)):
        net.write_fact(ca3[:, q], X[:, q], Y[:, q])
    assert net.rec.csr("zero").count_nonzero() == 0
    a = net.rec.csr("intact").toarray()
    b = net.rec.csr("shuffle", np.random.default_rng(3)).toarray()
    # each row keeps its multiset of values, attached to other presynaptic cells
    for i in range(0, 1000, 97):
        np.testing.assert_allclose(np.sort(a[i][net.rec.idx[i]]), np.sort(b[i][net.rec.idx[i]]), rtol=1e-6)
    assert not np.allclose(a, b)


def test_bounded_synapses_stay_in_bounds():
    cfg = _tiny(arm="sparse_dg_bounded")
    net = M.Network(cfg)
    F, _ = M.draw_facts(net, 40, 1)
    X = net.ec_in(F, with_patient=True)
    ca3, _ = net.encode_ca3(X)
    Y = net.ec_out_patient(F)
    for q in range(len(F)):
        net.write_fact(ca3[:, q], X[:, q], Y[:, q])
    for p in (net.rec, net.pp, net.out):
        assert p.W.min() >= -cfg.bound and p.W.max() <= cfg.bound
        assert (p.W != 0).any()


def test_tiny_end_to_end_learns_and_freeze_is_chance():
    rec = M.run(_tiny(), None, log=lambda m: None)
    s = rec["summary"]
    assert s["recall"][0] >= 0.8                    # learns below capacity
    assert rec["checkpoints"][0]["freeze_all_recall"] <= 0.1
    assert len(set(s["synapse_bytes"])) == 1


def _summary(recall, recent=None, dprime=None, rz=None, rs=None, t=None, P=M.P_GRID):
    n = len(P)
    return dict(P=list(P), recall=list(recall), recall_recent=list(recent or recall),
                recall_rec_zero=list(rz or [0.0] * n), recall_rec_shuffle=list(rs or [0.0] * n),
                dprime=list(dprime or [10.0 / np.sqrt(p) for p in P]),
                per_query_ms_single=list(t or [5.0] * n), synapse_bytes=[1] * n,
                P50=M._log_interp_cross(list(P), list(recall), 0.5),
                P90=M._log_interp_cross(list(P), list(recall), 0.9))


def test_gates_can_fail():
    cliff = [1, 1, 1, 1, 0.95, 0.6, 0.1, 0.0, 0.0]
    early = [1, 1, 0.9, 0.3, 0.05, 0.0, 0.0, 0.0, 0.0]
    localist = [1.0] * 9
    S_ok = dict(sparse_dg=_summary(cliff, rz=[0.5] * 9, rs=[0.5] * 9),
                sparse_dg_c2=_summary(cliff[:4] + [1, 0.95, 0.4, 0.05, 0.0]),
                dense_nodg=_summary(early), sparse_nodg=_summary(cliff),
                sparse_dg_bounded=_summary(cliff, recent=[1.0] * 9))
    g = M.gates_for_seed(S_ok)
    assert g["G2_cliff"][0] is True and g["G3_companion_capacity"][0] is True
    # a localist store (no crosstalk, no cliff) FAILS the cliff and the shared-crosstalk gates
    S_loc = dict(S_ok, sparse_dg=_summary(localist, dprime=[10.0] * 9))
    g = M.gates_for_seed(S_loc)
    assert g["G2_cliff"][0] is False and g["G8_shared_crosstalk"][0] is False
    assert g["G3_companion_capacity"][0] is None          # censored P50 -> UNDEFINED, never a pass
    # the recurrent lesion gate fails when the perforant path alone already recalls
    S_pp = dict(S_ok, sparse_dg=_summary(cliff, rz=cliff, rs=cliff))
    assert M.gates_for_seed(S_pp)["G5_recurrent_loadbearing"][0] is False
    # query time growing with P fails the cost integrity check
    S_scan = dict(S_ok, sparse_dg=_summary(cliff, t=[float(p) for p in M.P_GRID]))
    assert M.gates_for_seed(S_scan)["G6_cost_flat_INTEGRITY"][0] is False
