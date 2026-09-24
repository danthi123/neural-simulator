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
    if arm == "sparse_dg_recx2":
        over.update(c_rec=400)                     # c_rec ONLY doubled; c_pp, c_out stay at TINY's values
    elif "c2" in arm:
        over.update(c_rec=400, c_pp=120, c_out=400)
    if arm.startswith("dense"):
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
    assert net.rec.matrix("zero").count_nonzero() == 0
    a = net.rec.matrix("intact").toarray()
    b = net.rec.matrix("shuffle", np.random.default_rng(3)).toarray()
    # each row keeps its multiset of values, attached to other presynaptic cells
    for i in range(0, 1000, 97):
        np.testing.assert_allclose(np.sort(a[i][net.rec.idx[i]]), np.sort(b[i][net.rec.idx[i]]), rtol=1e-6)
    assert not np.allclose(a, b)


def test_bounded_synapses_stay_in_bounds():
    cfg = _tiny(arm="sparse_dg_bounded_hub")
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
    n = len(M.P_GRID)
    cliff = [1, 1, 1, 1, 0.95, 0.6, 0.1, 0.0, 0.0, 0.0]
    early = [1, 1, 0.9, 0.3, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
    doubled = [1, 1, 1, 1, 1, 0.95, 0.4, 0.05, 0.0, 0.0]
    localist = [1.0] * n
    assert len(cliff) == n
    S_ok = dict(sparse_dg=_summary(cliff, rz=early, rs=early), sparse_dg_c2=_summary(doubled),
                sparse_dg_recx2=_summary(cliff), dense_nodg=_summary(early), sparse_dg_hub=_summary(cliff),
                dense_nodg_hub=_summary(early), sparse_nodg_hub=_summary(early),
                sparse_dg_c2_hub=_summary(cliff), sparse_dg_bounded_hub=_summary(early, recent=[1.0] * n))
    assert set(S_ok) == set(M.ARMS)
    g = M.gates_for_seed(S_ok)
    assert all(v[0] is True for v in g.values()), {k: v for k, v in g.items() if v[0] is not True}
    # a localist store (no crosstalk, no cliff) FAILS the cliff and the shared-crosstalk gates
    S_loc = dict(S_ok, sparse_dg=_summary(localist, dprime=[10.0] * n))
    g = M.gates_for_seed(S_loc)
    assert g["G2_cliff"][0] is False and g["G8_shared_crosstalk"][0] is False
    assert g["G4_capacity_law"][0] is None                # censored P50 -> UNDEFINED, never a pass
    # a missing arm makes the cliff gate UNDEFINED, not a pass
    S_miss = {k: v for k, v in S_ok.items() if k != "dense_nodg_hub"}
    g = M.gates_for_seed(S_miss)
    assert g["G2_cliff"][0] is None and g["G3_companion_capacity"][0] is None
    # the companion gate fails when sparse DG coding does not raise capacity
    assert M.gates_for_seed(dict(S_ok, dense_nodg_hub=_summary(cliff)))["G3_companion_capacity"][0] is False
    # the capacity-law gate fails when doubling fan-in does not move capacity
    assert M.gates_for_seed(dict(S_ok, sparse_dg_c2=_summary(cliff)))["G4_capacity_law"][0] is False
    # the recurrent lesion gate fails when the perforant path alone holds the same capacity
    S_pp = dict(S_ok, sparse_dg=_summary(cliff, rz=cliff, rs=early))
    assert M.gates_for_seed(S_pp)["G5_recurrent_loadbearing"][0] is False
    # query time growing with P fails the cost integrity check
    S_scan = dict(S_ok, sparse_dg=_summary(cliff, rz=early, rs=early, t=[float(p) for p in M.P_GRID]))
    assert M.gates_for_seed(S_scan)["G6_cost_flat_INTEGRITY"][0] is False
    # a palimpsest that loses even its recent facts fails G7
    assert M.gates_for_seed(dict(S_ok, sparse_dg_bounded_hub=_summary(early)))["G7_palimpsest"][0] is False
    # G9 fails if extra synapses DO cure the hub-regime limit
    assert M.gates_for_seed(dict(S_ok, sparse_dg_c2_hub=_summary(doubled)))["G9_hub_limit_not_synaptic"][0] is False


def test_g5_attributable_fraction_is_linear_not_log():
    """G5's reported fraction must be LINEAR in P50, not a ratio built from log(P50) (log(P50) has an arbitrary
    zero -- one fact -- and is not a meaningful fraction; 2026-09-24 review). It must equal exactly
    (P50_intact - P50_rec_zero) / P50_intact, and the old log-based 'attributable' key must be gone."""
    cliff = [1, 1, 1, 1, 0.95, 0.6, 0.1, 0.0, 0.0, 0.0]
    rz = [1, 1, 1, 0.95, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0]     # a smaller store: crosses 0.5 one grid point earlier
    S = dict(sparse_dg=_summary(cliff, rz=rz, rs=rz))
    detail = M.gates_for_seed(S)["G5_recurrent_loadbearing"][1]
    assert "attributable_linear_frac" in detail and "attributable" not in detail
    p_i, p_z = detail["P50_intact"], detail["P50_rec_zero"]
    assert p_i is not None and p_z is not None and 0.0 < p_z < p_i
    assert detail["attributable_linear_frac"] == pytest.approx((p_i - p_z) / p_i, rel=1e-9)
    # sanity: the linear fraction is in (0, 1) for a real partial lesion, not a log-space quantity
    assert 0.0 < detail["attributable_linear_frac"] < 1.0


def test_capacity_law_fit_is_the_marginal_not_the_per_arm_ratio(tmp_path):
    """k_fit (and the GPU extrapolation) must come from the per-seed MARGINAL contrast between sparse_dg and
    sparse_dg_recx2 -- (P50_recx2 - P50_sparse_dg) * a ln(1/a) / (c_rec_recx2 - c_rec_sparse_dg) -- NEVER from
    fitting k per arm (P50 * a ln(1/a) / c_rec) and taking the median over the two arms. 2026-09-24 SECOND review
    (fix-required): that per-arm-median fit is the SAME all-fan-in-confounded quantity the FIRST review rejected
    off sparse_dg_c2 (the sparse_dg half is identical to it), so it cannot be attributed to the recurrent edge
    even though sparse_dg_recx2 varies c_rec alone -- the recx2 arm must be used as a DIFFERENCE, not averaged
    in. MUTATION GUARD: on these P50s the two fits provably disagree (~0.055 vs ~0.154, matching the review's own
    worked numbers), so this test FAILS if aggregate() reverts to the per-arm-median fit."""
    import json as _json
    cliff = [1, 1, 1, 1, 0.9, 0.5, 0.1, 0.0, 0.0, 0.0]
    p50s = dict(sparse_dg=8119.0, sparse_dg_recx2=10500.0, sparse_dg_c2=16650.0)
    for arm in M.ARMS:
        s = _summary(cliff)
        s["P50"] = p50s.get(arm, 8119.0)
        (tmp_path / ("%s_s42.json" % arm)).write_text(_json.dumps(dict(summary=s)))
    res = M.aggregate(str(tmp_path), seeds=(42,))
    assert res["k_fit_arm_pair"] == ("sparse_dg", "sparse_dg_recx2")

    c_sd = M.make_cfg("sparse_dg", 42)
    c_rx = M.make_cfg("sparse_dg_recx2", 42)
    assert c_sd.a_ca3 == c_rx.a_ca3
    a_term = c_sd.a_ca3 * np.log(1 / c_sd.a_ca3)

    # the CORRECT fit: the marginal contrast between the two arms, divided by the DIFFERENCE in c_rec
    k_marginal = (p50s["sparse_dg_recx2"] - p50s["sparse_dg"]) * a_term / (c_rx.c_rec - c_sd.c_rec)
    assert res["k_fit"] == pytest.approx(k_marginal, rel=1e-6)
    assert res["k_marginal_per_seed"]["42"] == pytest.approx(k_marginal, rel=1e-6)

    # the WRONG fit the pre-fix code computed: k per arm, then median over the two arms
    k_sd_allfanin = p50s["sparse_dg"] * a_term / c_sd.c_rec
    k_rx_allfanin = p50s["sparse_dg_recx2"] * a_term / c_rx.c_rec
    k_per_arm_median = float(np.median([k_sd_allfanin, k_rx_allfanin]))
    assert k_per_arm_median != pytest.approx(k_marginal, rel=1e-3)      # the two numbers are genuinely different
    assert res["k_fit"] != pytest.approx(k_per_arm_median, rel=1e-3)    # MUTATION GUARD

    # the per-arm all-fan-in numbers are still reported -- but ONLY descriptively, never feeding k_fit
    descriptive = {(d["seed"], d["arm"]): d["k_allfanin"] for d in res["k_allfanin_per_arm_DESCRIPTIVE_ONLY"]}
    assert descriptive[(42, "sparse_dg")] == pytest.approx(k_sd_allfanin, rel=1e-6)
    assert descriptive[(42, "sparse_dg_recx2")] == pytest.approx(k_rx_allfanin, rel=1e-6)

    # sparse_dg_c2 (all fan-ins doubled at once) must still NOT feed the fit
    c_c2 = M.make_cfg("sparse_dg_c2", 42)
    k_c2_if_used = p50s["sparse_dg_c2"] * c_c2.a_ca3 * np.log(1 / c_c2.a_ca3) / c_c2.c_rec
    assert res["k_fit"] != pytest.approx(float(np.median([k_marginal, k_c2_if_used])), rel=1e-6)


def test_capacity_law_marginal_not_positive_is_reported_not_clipped(tmp_path):
    """A seed where doubling c_rec alone does not raise P50 (a realistic failing outcome the prereg's capacity-
    law prediction must name) reports a non-positive marginal AS MEASURED: never clipped to zero, never dropped
    from k_marginal_per_seed, and it still sets k_fit (median of one value here) rather than being silently
    excluded."""
    import json as _json
    cliff = [1, 1, 1, 1, 0.9, 0.5, 0.1, 0.0, 0.0, 0.0]
    p50s = dict(sparse_dg=8119.0, sparse_dg_recx2=7000.0)   # recx2 P50 LOWER than sparse_dg's despite c_rec x2
    for arm in M.ARMS:
        s = _summary(cliff)
        s["P50"] = p50s.get(arm, 8119.0)
        (tmp_path / ("%s_s42.json" % arm)).write_text(_json.dumps(dict(summary=s)))
    res = M.aggregate(str(tmp_path), seeds=(42,))
    assert res["k_marginal_per_seed"]["42"] < 0
    assert res["k_fit"] < 0                                             # NOT clipped to 0.0
    assert res["gpu_point_extrapolation"]["note"].startswith("k_fit <= 0")
