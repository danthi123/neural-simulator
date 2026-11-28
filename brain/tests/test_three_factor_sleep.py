# Capability: Sleep consolidation
# Record: brain/research/2026-11-28-local-rule-research-round.md
# Expected-fail: none (plumbing on a tiny random cortex and synthetic records; no model weights)
"""Three-factor sleep: the perturbable forward is PlasticCortex exactly, the re-parameterisation keeps the cortex
pretrained, the expected update has positive cosine with the true gradient (negative for the wrong sign, near zero for
the shuffled scalar), the night is forward-only and spends exactly its budget, and the registered pick / verdict give
the exact label with each clause, kill rule and engagement gate failing alone. Engine specs render and delegate."""
import copy
import json
import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from brain.experiments import three_factor_sleep as T
from brain.plastic_cortex import PlasticCortex
from brain.tests.test_source_tag import _Cortex

TEXT = "my cat is Pixel"
EPS = [("User: my cat is Pixel\n\n", 1.0), ("User: my dog is Bolt\n\n", 1.0)]
GEN = ["User: hi\n\nAssistant: yo\n\n"]


def _pc(noise=0.0):
    torch.manual_seed(1)
    p = PlasticCortex(_Cortex(), rank=2)
    if noise:
        with torch.no_grad():
            for d in (p.B_k, p.B_v):
                for v in d.values():
                    v.normal_(0, noise)
    return p


def _st(pc):
    return [s.float() for s in pc.zero_state()]


# ------------------------------------------------------------------ mechanism
def test_perturbable_forward_is_plastic_cortex_exactly():
    pc = _pc(0.1)
    st, ids = _st(pc), pc.cortex.pipe.encode(TEXT)
    with torch.no_grad():
        ref = pc(ids, st)[0]
        tf = T.adopt_tf(pc)
        assert torch.equal(tf(ids, st)[0], ref)                          # pert and rec None: the stock path
        tf.pert = {i: (torch.zeros(len(ids), 2), torch.zeros(len(ids), 2)) for i in range(pc.L)}
        tf.rec = []
        assert torch.equal(tf(ids, st)[0], ref) and len(tf.rec) == pc.L   # zero noise, recording on: identical
        tf.pert = {i: (torch.ones(len(ids), 2), torch.zeros(len(ids), 2)) for i in range(pc.L)}
        assert not torch.allclose(tf(ids, st)[0], ref)                    # noise on the rank units reaches the output
        tf.pert, tf.rec = None, None


def test_reparam_keeps_the_cortex_pretrained_and_freezes_b():
    pc = _pc(0.1)
    st, ids = _st(pc), pc.cortex.pipe.encode(TEXT)
    T.reparam_rank(pc, 3)
    pc.enabled = False
    with torch.no_grad():
        off = pc(ids, st)[0]
    pc.enabled = True
    with torch.no_grad():
        assert torch.equal(pc(ids, st)[0], off) and pc.adapter_norm() == 0.0
    assert all(not pc.B_k[s].requires_grad and not pc.B_v[s].requires_grad and pc.A_k[s].requires_grad for s in pc.A_k)
    assert float(pc.B_k["0"].abs().sum()) > 0
    q = _pc(0.1)
    T.reparam_rank(q, 3)
    assert torch.equal(q.B_v["1"], pc.B_v["1"])                           # seeded


def test_instrument_backprops_one_text_at_a_time_with_the_one_graph_value():
    """Pre-data review item 1: full_objective(grad=True) accumulates per-text gradients; value and G must equal the
    single-graph objective (mean replay CE + kl_weight x mean keep KL)."""
    pc = _pc()
    T.reparam_rank(pc, 0)
    with torch.no_grad():
        for s_ in pc.A_k:
            pc.A_k[s_].normal_(0, 0.05)
            pc.A_v[s_].normal_(0, 0.05)
    st = _st(pc)
    tells = [pc.cortex.pipe.encode(t) for t, _ in EPS]
    gens = [pc.cortex.pipe.encode(g) for g in GEN + ["hi there"]]
    pc.enabled = False
    with torch.no_grad():
        refs = [F.log_softmax(pc(g, st)[0], -1) for g in gens]
    pc.enabled = True
    L, rep, kp, G = T.full_objective(pc, st, tells, gens, refs, 0.7, grad=True)
    params = T.a_params(pc)
    with torch.enable_grad():
        r1 = sum(F.cross_entropy(pc(i, st)[0][:-1], torch.tensor(i[1:])) for i in tells) / len(tells)
        k1 = sum(F.kl_div(F.log_softmax(pc(g, st)[0], -1), r, log_target=True, reduction="batchmean")
                 for g, r in zip(gens, refs)) / len(gens)
        L1 = r1 + 0.7 * k1
        G1 = torch.autograd.grad(L1, [p for _, p in params])
    assert L == pytest.approx(float(L1), rel=1e-5) and rep == pytest.approx(float(r1), rel=1e-5)
    assert kp == pytest.approx(float(k1), rel=1e-5, abs=1e-8)
    for (k, _), g in zip(params, G1):
        assert torch.allclose(G[k], g, rtol=1e-4, atol=1e-7), k
    assert all(p.grad is None for _, p in params)
    L0, *_ , G0 = T.full_objective(pc, st, tells, gens, refs, 0.7, grad=False)
    assert G0 is None and L0 == pytest.approx(L, rel=1e-6)


def _expected_update(pc, estimator, M=200, sigma=1e-2):
    """{mode: cosine of the mean of M single-sample updates (the scalar as `mode` applies it, the same samples for
    every mode) with the true gradient of the same pair objective}."""
    T.reparam_rank(pc, 0)
    with torch.no_grad():
        for s in pc.A_k:
            pc.A_k[s].normal_(0, 0.05)
            pc.A_v[s].normal_(0, 0.05)
    st, tf = _st(pc), T.adopt_tf(pc)
    ids, gids = tf.cortex.pipe.encode(TEXT), tf.cortex.pipe.encode("hi there")
    tf.enabled = False
    with torch.no_grad():
        ref = F.log_softmax(tf(gids, st)[0], -1)
    tf.enabled = True
    texts = [(ids, "replay", None, 1.0), (gids, "keep", ref, 1.0)]
    params = T.a_params(tf)
    with torch.enable_grad():
        L = F.cross_entropy(tf(ids, st)[0][:-1], torch.tensor(ids[1:])) + \
            F.kl_div(F.log_softmax(tf(gids, st)[0], -1), ref, log_target=True, reduction="batchmean")
        G = dict(zip([k for k, _ in params], torch.autograd.grad(L, [p for _, p in params])))
    modes = ("true", "wrong_sign", "shuffled")
    gen, c = torch.Generator().manual_seed(0), {"n": 0}
    third = {m: T._Scalar(m, 0) for m in modes}
    acc = {m: {k: torch.zeros_like(v) for k, v in G.items()} for m in modes}
    for _ in range(M):
        nz = [T.draw_noise(tf, len(i), gen) for i, *_ in texts]
        s, N, _, _ = T.zo_sample(tf, texts, st, sigma, nz, estimator, c)
        for m in modes:
            su = third[m](s)
            for k in N:
                acc[m][k] += su * N[k] / M
    assert c["n"] == 4 * M and all(p.grad is None for _, p in params)
    return {m: T._flat_cos(acc[m], G) for m in modes}


@pytest.mark.parametrize("estimator", ["antithetic", "forward"])
def test_expected_update_has_positive_cosine_with_the_true_gradient(estimator):
    cos = _expected_update(_pc(), estimator)
    assert cos["true"] > 0.3, cos                       # the rule's expected update points down the true gradient
    assert cos["wrong_sign"] == pytest.approx(-cos["true"]) and abs(cos["shuffled"]) < 0.2, cos


def test_scalar_modes():
    t, w, s = T._Scalar("true", 0), T._Scalar("wrong_sign", 0), T._Scalar("shuffled", 0)
    assert t(2.0) == 2.0 and w(2.0) == -2.0
    assert s(5.0) == 0.0 and s(7.0) == 5.0                               # no own scalar ever: earlier ones only
    with pytest.raises(AssertionError):
        T._Scalar("x", 0)


def test_budget_is_the_full_read_number():
    # 11-28 full read: 64 x the gradient's 480 forwards = 30720 = 7680 antithetic steps = 32 x its 240 steps (MeZO OPT)
    assert T.BUDGET_MULT == 64 and T.plan_steps(T.BUDGET_MULT * 480, "antithetic", 1, 0, 1, 15) == (7680, 30720)


def test_budget_plan_counts_every_forward():
    assert T.plan_steps(14400, "antithetic", 1, 0, 1, 15) == (3600, 14400)
    assert T.plan_steps(14400, "forward", 1, 0, 1, 15) == (3600, 14400)
    assert T.plan_steps(14400, "antithetic", 2, 0, 1, 15) == (1800, 14400)
    st, used = T.plan_steps(14400, "antithetic", 1, 50, 1, 15)          # svrg: 8 per step + 30 per refresh
    assert used <= 14400 and used == st * 8 + 30 * math.ceil(st / 50)


@pytest.mark.parametrize("svrg", [0, 5])
def test_night_is_forward_only_spends_its_budget_and_records_engagement(svrg):
    pc = _pc()
    info = {}
    T.tf_sleep(pc, _st(pc), EPS, GEN, grad_steps=2, lr=1e-3, sigma=0.03, svrg=svrg, info=info)
    b = T.BUDGET_MULT * 2 * 2                                            # 64 x 2 steps x 2 forwards = 256
    assert info["budget"] == b and info["forwards_used"] == info["forwards_planned"] <= b
    assert info["forwards_used"] >= T.BUDGET_USE_MIN * b
    assert all(p.grad is None for p in pc.parameters())
    assert pc.adapter_norm() > 0 and float(sum(pc.B_k[s].abs().sum() for s in pc.B_k)) > 0
    assert info["scalar_corr"] > 0.99 and len(info["cos_windows"]) == T.N_WINDOWS
    for k in ("objective_start", "objective_end", "replay_start", "replay_end", "cos_window", "cos_night"):
        assert math.isfinite(info[k]), k


def test_true_night_lowers_the_objective_and_wrong_sign_raises_it():
    out = {}
    for sc in ("true", "wrong_sign", "shuffled"):
        pc = _pc()
        info = {}
        T.tf_sleep(pc, _st(pc), [("my cat Pixel", 1.0), ("my dog Bolt", 1.0)], ["hi yo"], grad_steps=20, lr=3e-4, sigma=0.03, scalar=sc, estimator="forward", info=info)
        out[sc] = info
    assert out["true"]["objective_end"] < out["true"]["objective_start"]
    assert out["wrong_sign"]["objective_end"] > out["wrong_sign"]["objective_start"]
    assert out["true"]["cos_window"] > 0 > out["wrong_sign"]["cos_window"]
    assert out["wrong_sign"]["scalar_corr"] < -0.99 and abs(out["shuffled"]["scalar_corr"]) < 0.3


def test_graded_readout_on_stub():
    pc = _pc()
    g = T.graded(pc, _st(pc), [("My cat is called Pixel.", "What is my cat called?", "pixel")])
    assert math.isfinite(g["answer_logprob"]) and g["answer_logprob"] < 0
    assert 1 <= g["answer_rank_median"] <= 128 and g["graded_items"][0]["key"] == "pixel"


def test_guard():
    with pytest.raises(SystemExit):
        T.guard({})
    T.guard({"POOL_NODE": "1"})
    T.guard({"BRAIN_ALLOW_W3090": "1"})


# ------------------------------------------------------------------ registered rules
def _eng_tf(scalar="true", start=3.0, end=2.0, replay=0.5, cos=0.3, used=30720, corr=None):
    corr = {"true": 1.0, "wrong_sign": -1.0, "shuffled": 0.01}[scalar] if corr is None else corr
    return {"kind": "three_factor", "scalar": scalar, "objective_start": start, "objective_end": end,
            "replay_end": replay, "cos_window": cos, "budget": 30720, "budget_mult": 64, "grad_forwards": 480,
            "forwards_used": used, "scalar_corr": corr}


def _arm(recall, g=0.03, h=0.04, false=1, norm=1.0, eng=None, lp=-5.0):
    return {"recall": recall, "false_recall": false, "generic_kl": g, "heldout_kl": h, "adapter_norm": norm,
            "answer_logprob": lp, "answer_log10_rank": 1.0, "answer_top10": 3, "engagement": eng or {}}


def _go():
    S = T.SEEDS
    return {"no_sleep": {s: _arm(1, g=0.0, h=0.0, norm=0.0, eng={"kind": "none"}) for s in S},
            "gradient": {s: _arm(6, eng={"loss_start": 3.0, "loss_end": 0.2}) for s in S},
            "three_factor": {s: _arm(4, eng=_eng_tf()) for s in S},
            "tf_shuffled": {s: _arm(1, eng=_eng_tf("shuffled", end=3.1)) for s in S},
            "tf_wrong_sign": {s: _arm(1, eng=_eng_tf("wrong_sign", end=3.5)) for s in S},
            "gradient_rank": {s: _arm(5, eng={"loss_start": 3.0, "loss_end": 0.3}) for s in S}}


def _set(recs, arm, seeds=T.SEEDS, **kw):
    for s in seeds:
        for k, v in kw.items():
            if k == "eng":
                recs[arm][s]["engagement"].update(v)
            else:
                recs[arm][s][k] = v
    return recs


def test_verdict_go_baseline():
    v = T.verdict(_go())
    assert v["verdict"] == "GO-TF" and all(v["clauses"].values()) and v["bar"] == 3.0
    assert v["means"]["three_factor"]["answer_logprob"] == -5.0 and v["engagement"] == {"core": [], "controls": [], "report": []}


CLAUSES = [
    ("a recall under bar", lambda r: _set(r, "three_factor", recall=2), "NEGATIVE"),
    ("c generic", lambda r: _set(r, "three_factor", generic_kl=0.05), "NEGATIVE"),
    ("c heldout", lambda r: _set(r, "three_factor", heldout_kl=0.06), "NEGATIVE"),
    ("c night", lambda r: (_set(r, "three_factor", [42], generic_kl=1.5), _set(r, "gradient", generic_kl=1.0))[0], "NEGATIVE"),
    ("e false recall", lambda r: _set(r, "three_factor", false_recall=3), "NEGATIVE"),
    ("b shuffled wins", lambda r: _set(r, "tf_shuffled", [42, 43], recall=4), "NEGATIVE"),
    ("b shuffled mean", lambda r: _set(r, "tf_shuffled", recall=2.5), "NEGATIVE"),
    ("b wrong-sign wins", lambda r: _set(r, "tf_wrong_sign", [42, 43], recall=4), "NEGATIVE"),
    ("b wrong-sign mean", lambda r: _set(r, "tf_wrong_sign", recall=2.5), "NEGATIVE"),
    ("kill replay", lambda r: _set(r, "three_factor", eng={"replay_end": 1.0}), "KILL"),
    ("kill cos", lambda r: _set(r, "three_factor", eng={"cos_window": 0.09}), "KILL"),
    ("gradient under 2", lambda r: _set(r, "gradient", recall=1.5), "UNDEFINED"),
]


@pytest.mark.parametrize("name,edit,want", CLAUSES, ids=[c[0] for c in CLAUSES])
def test_each_clause_failing_alone_gives_the_exact_verdict(name, edit, want):
    r = edit(copy.deepcopy(_go()))
    assert T.verdict(r)["verdict"] == want, name


GATES = [
    ("no_sleep moved", lambda r: _set(r, "no_sleep", [44], adapter_norm=0.1)),
    ("gradient flat", lambda r: _set(r, "gradient", [44], eng={"loss_end": 3.0})),
    ("tf objective flat", lambda r: _set(r, "three_factor", [44], eng={"objective_end": 3.0})),
    ("tf over budget", lambda r: _set(r, "three_factor", [44], eng={"forwards_used": 30721})),
    ("tf under budget", lambda r: _set(r, "three_factor", [44], eng={"forwards_used": 25000})),
    ("tf wrong budget mult", lambda r: _set(r, "three_factor", [44], eng={"budget_mult": 10})),
    ("tf not own scalar", lambda r: _set(r, "three_factor", [44], eng={"scalar_corr": 0.5})),
    ("tf zero norm", lambda r: _set(r, "three_factor", [44], adapter_norm=0.0)),
    ("shuffled small", lambda r: _set(r, "tf_shuffled", [44], adapter_norm=0.4)),
    ("shuffled correlated", lambda r: _set(r, "tf_shuffled", [44], eng={"scalar_corr": 0.5})),
    ("shuffled under budget", lambda r: _set(r, "tf_shuffled", [44], eng={"forwards_used": 100})),
    ("wrong-sign small", lambda r: _set(r, "tf_wrong_sign", [44], adapter_norm=0.4)),
    ("wrong-sign not ascending", lambda r: _set(r, "tf_wrong_sign", [44], eng={"objective_end": 2.9})),
    ("wrong-sign sign", lambda r: _set(r, "tf_wrong_sign", [44], eng={"scalar_corr": 1.0})),
]


@pytest.mark.parametrize("name,edit", GATES, ids=[g[0] for g in GATES])
def test_each_engagement_gate_alone_is_undefined_not_go(name, edit):
    v = T.verdict(edit(copy.deepcopy(_go())))
    assert v["verdict"] == "UNDEFINED", (name, v["verdict"])


def test_a_core_failure_with_a_dead_control_stays_negative():
    r = _set(_set(copy.deepcopy(_go()), "three_factor", recall=2), "tf_shuffled", [44], adapter_norm=0.0)
    assert T.verdict(r)["verdict"] == "NEGATIVE"


def test_gradient_rank_is_report_only_and_reads_a_negative():
    r = _set(copy.deepcopy(_go()), "gradient_rank", recall=0, eng={"loss_end": 3.0})
    assert T.verdict(r)["verdict"] == "GO-TF"                               # never gates a GO
    neg = lambda r: T.verdict(_set(r, "three_factor", recall=2))["reading"]
    assert neg(copy.deepcopy(_go())).startswith("rule-limit")
    assert neg(_set(copy.deepcopy(_go()), "gradient_rank", recall=1)).startswith("parameterisation-limit")
    assert neg(_set(copy.deepcopy(_go()), "gradient_rank", [42], eng={"loss_end": 3.0})).startswith("undetermined")


@pytest.mark.parametrize("arm,key", [("three_factor", "recall"), ("tf_shuffled", "heldout_kl"), ("gradient", "generic_kl")])
def test_any_nonfinite_gated_value_is_undefined(arm, key):
    assert T.verdict(_set(copy.deepcopy(_go()), arm, [101], **{key: float("nan")}))["verdict"] == "UNDEFINED"


def test_verdict_needs_exactly_the_six_seeds_in_every_arm():
    r = copy.deepcopy(_go())
    del r["gradient_rank"][42]
    with pytest.raises(ValueError):
        T.verdict(r)
    r = copy.deepcopy(_go())
    r["three_factor"][7] = r["three_factor"][42]
    with pytest.raises(ValueError):
        T.verdict(r)


def test_combine_two_scales():
    assert T.combine("GO-TF", "GO-TF") == "GO-TF" and T.combine("GO-TF", "NEGATIVE") == "GO-TF-PARTIAL"
    assert T.combine("KILL", "NEGATIVE-DEV") == "NEGATIVE" and T.combine("GO-TF", "UNDEFINED") == "UNDEFINED"
    assert T.combine("GO-TF", None) == "UNDEFINED" and T.combine("HOLD-DEV", "GO-TF") == "UNDEFINED"


REF = {"no_sleep": {"recall": 1, "generic_kl": 0.0, "heldout_kl_dev": 0.0},
       "gradient": {"recall": 6, "generic_kl": 0.04, "heldout_kl_dev": 0.05}}


def _row(lr=1e-4, sigma=0.03, recall=4, g=0.03, h=0.04, norm=1.0, lp=-5.0, **eng):
    return {"lr": lr, "sigma": sigma, "recall": recall, "generic_kl": g, "heldout_kl_dev": h, "adapter_norm": norm,
            "answer_logprob": lp, "engagement": {**_eng_tf(), **eng}}


def test_dev_pick_rule_ties_and_each_disqualifier_alone():
    p = T.dev_pick(REF, [_row(), _row(lr=3e-4, recall=5), _row(lr=1e-3, recall=5, lp=-4.0)])
    assert p["verdict"] == "PICKED" and p["pick"] == {"lr": 1e-3, "sigma": 0.03}      # tie on recall: higher logprob
    assert T.dev_pick(REF, [_row(), _row(lr=3e-4)])["pick"]["lr"] == 1e-4              # full tie: lower lr
    for bad in (dict(recall=2), dict(g=0.06), dict(h=0.07), dict(norm=0.0), dict(objective_end=3.5),
                dict(forwards_used=10000), dict(scalar_corr=0.2), dict(lp=float("nan"))):
        assert T.dev_pick(REF, [_row(**bad)])["verdict"] == "NEGATIVE-DEV", bad
    big = {"no_sleep": REF["no_sleep"], "gradient": {"recall": 6, "generic_kl": 2.0, "heldout_kl_dev": 2.0}}
    assert T.dev_pick(big, [_row(g=1.1)])["verdict"] == "NEGATIVE-DEV"               # KL above 1 even under the cap
    assert T.dev_pick(REF, [_row(replay_end=1.2), _row(cos_window=0.05)])["verdict"] == "KILL-DEV"
    assert T.dev_pick(REF, [_row(replay_end=1.2), _row(recall=2)])["verdict"] == "NEGATIVE-DEV"   # one alive cell
    assert T.dev_pick({**REF, "gradient": {**REF["gradient"], "recall": 1}}, [_row()])["verdict"] == "UNDEFINED-DEV"
    assert T.dev_pick({**REF, "gradient": {**REF["gradient"], "generic_kl": None}}, [_row()])["verdict"] == "UNDEFINED-DEV"


def test_dev_bar_binds_on_half_gradient():
    assert T.dev_bar(1, 8) == 4.0 and T.dev_bar(1, 4) == 3.0


# ------------------------------------------------------------------ CLI on synthetic units
def _unit(tag, seed, arm, res, eng=None, lr=1e-4, sigma=0.03, model="m.pth", commit="abc"):
    tf = arm in T.TF_ARMS
    cfg = dict(lr=lr, sigma=sigma, estimator="antithetic", n_pert=1, svrg=0, svrg_full=1, budget_mult=64.0, max_step=0.0)
    return {"tag": tag, "seed": seed, "arm": arm, "model_id": model, "commit": commit, "prompts_sha256": "p",
            **{k: (cfg[k] if tf else None) for k in T.TF_KEYS}, "engagement": eng or {}, "result": res}


def _write_six(d, recs, **kw):
    os.makedirs(d, exist_ok=True)
    for arm, by in recs.items():
        for s, r in by.items():
            r = dict(r)
            eng = r.pop("engagement")
            T.atomic_write(T.unit_path(d, "six", s, arm), _unit("six", s, arm, r, eng, **kw))


def _pickfile(tmp_path, lr=1e-4, sigma=0.03, model="m.pth", commit="abc"):
    p = str(tmp_path / "pick.json")
    T.atomic_write(p, {"verdict": "PICKED", "pick": {"lr": lr, "sigma": sigma}, "model_id": model, "commit": commit,
                       "tf_config": {"estimator": "antithetic", "n_pert": 1, "svrg": 0, "svrg_full": 1,
                                     "budget_mult": 64.0, "max_step": 0.0}})
    return p


def test_score_cli_go_missing_and_mixed(tmp_path):
    d = str(tmp_path / "six")
    _write_six(d, _go())
    out = str(tmp_path / "v.json")
    T.main(["score", "--dir", d, "--tag", "six", "--out", out, "--pick", _pickfile(tmp_path)])
    assert json.load(open(out))["verdict"] == "GO-TF"
    with pytest.raises(SystemExit, match="pick"):
        T.main(["score", "--dir", d, "--tag", "six", "--pick", _pickfile(tmp_path, sigma=0.1)])
    os.remove(T.unit_path(d, "six", 42, "gradient_rank"))
    with pytest.raises(SystemExit, match="missing units"):
        T.main(["score", "--dir", d, "--tag", "six"])
    d2 = str(tmp_path / "mixed")
    _write_six(d2, _go())
    u = json.load(open(T.unit_path(d2, "six", 43, "three_factor")))
    u["commit"] = "zzz"
    T.atomic_write(T.unit_path(d2, "six", 43, "three_factor"), u)
    with pytest.raises(SystemExit, match="commit"):
        T.main(["score", "--dir", d2, "--tag", "six"])
    d3 = str(tmp_path / "mixcfg")
    _write_six(d3, _go())
    u = json.load(open(T.unit_path(d3, "six", 43, "tf_shuffled")))
    u["sigma"] = 0.1
    T.atomic_write(T.unit_path(d3, "six", 43, "tf_shuffled"), u)
    with pytest.raises(SystemExit, match="sigma"):
        T.main(["score", "--dir", d3, "--tag", "six"])


def _write_dev(d, skip=None, killed=False):
    os.makedirs(d, exist_ok=True)
    for arm in T.REF_ARMS:
        T.atomic_write(T.unit_path(d, "ref", 7, arm), _unit("ref", 7, arm, {**REF.get(arm, REF["gradient"]), "answer_logprob": -6.0}))
    for lr in T.DEV_LRS:
        for sg in T.DEV_SIGMAS:
            if (lr, sg) == skip:
                continue
            tag = f"lr{lr:g}_sig{sg:g}"
            r = _row(lr, sg, recall=5 if (lr, sg) == (T.DEV_LRS[1], T.DEV_SIGMAS[2]) else 1,
                     **({"replay_end": 2.0} if killed else {}))
            eng = r.pop("engagement")
            T.atomic_write(T.unit_path(d, tag, 7, "three_factor"), _unit(tag, 7, "three_factor", r, eng, lr=lr, sigma=sg))


def test_pick_cli(tmp_path):
    d, out = str(tmp_path / "dev"), str(tmp_path / "pick.json")
    _write_dev(d)
    T.main(["pick", "--dir", d, "--out", out])
    p = json.load(open(out))
    assert p["verdict"] == "PICKED" and p["pick"] == {"lr": T.DEV_LRS[1], "sigma": T.DEV_SIGMAS[2]}
    assert p["tf_config"]["estimator"] == "antithetic"
    d2 = str(tmp_path / "dev2")
    _write_dev(d2, killed=True)
    T.main(["pick", "--dir", d2, "--out", out])
    assert json.load(open(out))["verdict"] == "KILL-DEV"
    d3 = str(tmp_path / "dev3")
    _write_dev(d3, skip=(T.DEV_LRS[0], T.DEV_SIGMAS[0]))
    with pytest.raises(SystemExit, match="9 registered"):
        T.main(["pick", "--dir", d3])
    d4 = str(tmp_path / "dev4")
    _write_dev(d4)
    os.remove(T.unit_path(d4, "ref", 7, "gradient_rank"))
    with pytest.raises(SystemExit, match="ref units"):
        T.main(["pick", "--dir", d4])


# ------------------------------------------------------------------ run_units on a stub cortex
def _run_args(tmp_path, **kw):
    a = dict(seed=42, arm_list=list(T.ARMS), out_dir=str(tmp_path / "six"), tag="six", model="/x/m.pth", steps=2,
             lr=1e-3, sigma=0.03, estimator="antithetic", n_pert=1, svrg=0, svrg_full=1, budget_mult=64.0,
             max_step=0.0, kl_weight=1.0, grad_lr=5e-4, grad_rank_lr=5e-5, device="cpu", pick=None, prompts_sha=T.PROMPTS_SHA256)
    a.update(kw)
    return SimpleNamespace(**a)


def _kw():
    from brain.experiments.mix_night_screen import PROMPTS
    from brain.experiments.sleep_probe import evaluate
    from brain.sleep import sleep
    return dict(PROMPTS=PROMPTS, GENERIC=GEN, build_arm_episodes=lambda s: {"sleep_true": EPS}, evaluate=evaluate,
                sleep=sleep, keys_all=["pixel", "bolt"],
                facts=[("My cat is Pixel.", "What is my cat?", "pixel"), ("My dog is Bolt.", "What is my dog?", "bolt")],
                PlasticCortex=lambda cx: PlasticCortex(cx, rank=2))


def test_run_units_on_stub_cortex_resume_provenance_and_pick(tmp_path, monkeypatch):
    monkeypatch.setenv("COMMIT", "abc")
    c = _Cortex()
    base = _st(_pc())
    a = _run_args(tmp_path, pick=_pickfile(tmp_path, lr=3e-4, sigma=0.1, model="m.pth", commit="abc"))
    done = T.run_units(a, c, base, **_kw())
    assert done == list(T.ARMS)
    units = {arm: json.load(open(T.unit_path(a.out_dir, "six", 42, arm))) for arm in T.ARMS}
    for arm, u in units.items():
        assert {"host", "commit", "device", "precision", "model_id", "prompts_sha256", "engagement"} <= set(u), arm
        assert all(k in u["result"] for k in T.GATED + ("heldout_kl_dev", "adapter_norm", "answer_logprob",
                                                        "answer_rank_median")), arm
        assert u["engagement"].get("kind"), arm
    assert units["three_factor"]["lr"] == 3e-4 and units["three_factor"]["sigma"] == 0.1 and units["gradient"]["lr"] is None
    assert units["no_sleep"]["result"]["adapter_norm"] == 0.0
    assert units["tf_wrong_sign"]["engagement"]["scalar"] == "wrong_sign" and units["tf_shuffled"]["scalar"] == "shuffled"
    for arm in T.TF_ARMS:
        e = units[arm]["engagement"]
        assert e["budget"] == 256 and e["forwards_used"] <= 256 and math.isfinite(e["cos_window"]), arm
    for arm in ("gradient", "gradient_rank"):
        assert all(math.isfinite(units[arm]["engagement"][k]) for k in ("loss_start", "loss_end")), arm
    T.check_units(list(units.values()), T.read_pick(a.pick))
    assert T.run_units(a, c, base, **_kw()) == []                                     # resume
    with pytest.raises(SystemExit, match="model"):
        T.run_units(_run_args(tmp_path, model="/x/m04.pth", out_dir=str(tmp_path / "o2"), pick=a.pick), c, base, **_kw())
    with pytest.raises(SystemExit, match="svrg"):
        T.run_units(_run_args(tmp_path, svrg=5, out_dir=str(tmp_path / "o4"), pick=a.pick), c, base, **_kw())
    monkeypatch.setenv("COMMIT", "zzz")
    with pytest.raises(SystemExit, match="commit"):
        T.run_units(_run_args(tmp_path, out_dir=str(tmp_path / "o3"), pick=a.pick), c, base, **_kw())


def test_dev_seed_never_computes_the_verdict_half(tmp_path, monkeypatch):
    monkeypatch.setenv("COMMIT", "abc")
    a = _run_args(tmp_path, seed=7, arm_list=["no_sleep", "three_factor"], out_dir=str(tmp_path / "dev"), tag="cell")
    T.run_units(a, _Cortex(), _st(_pc()), **_kw())
    for arm in ("no_sleep", "three_factor"):
        r = json.load(open(T.unit_path(a.out_dir, "cell", 7, arm)))["result"]
        assert r["heldout_kl"] is None and math.isfinite(r["heldout_kl_dev"])


def test_run_cli_refuses_without_the_host_guard(tmp_path, monkeypatch):
    monkeypatch.delenv("POOL_NODE", raising=False)
    monkeypatch.delenv("BRAIN_ALLOW_W3090", raising=False)
    with pytest.raises(SystemExit, match="refuses"):
        T.main(["run", "--model", "m.pth", "--seed", "7", "--tag", "x", "--out-dir", str(tmp_path)])


# ------------------------------------------------------------------ engine specs
def test_engine_specs_render_chain_and_are_registered():
    from tools.engine import core
    a, b = core.Spec.find("three_factor_sleep_01b"), core.Spec.find("three_factor_sleep_04b")
    assert (a.status, b.status) == ("REGISTERED", "REGISTERED") and a.question == b.question
    assert a.model == "rwkv7-g1d-0.1b-20260129-ctx8192.pth" and b.model == "rwkv7-g1d-0.4b-20260210-ctx8192.pth"
    assert "{model}" in a.cmd and "{model}" in b.cmd and a.smoke["params"].startswith("--arms ")
    assert len(a.dev_grid) == 10 and a.dev_grid == b.dev_grid
    assert a.dev_grid["ref"] == "--arms no_sleep,gradient,gradient_rank --tag ref"
    cells = {(float(p.split("--lr ")[1].split()[0]), float(p.split("--sigma ")[1].split()[0]))
             for k, p in a.dev_grid.items() if k != "ref"}
    assert cells == {(lr, s) for lr in T.DEV_LRS for s in T.DEV_SIGMAS}
    c = a.render(7, a.dev_grid["ref"], "/o.json", commit="c" * 12, job="ref")
    assert "three_factor_sleep_01b/cccccccccccc/ref" in c and "-m brain.experiments.three_factor_sleep run" in c
    assert a.next_for("GO-TF") == "three_factor_sleep_04b"
    for v in ("NEGATIVE", "NEGATIVE-DEV", "KILL", "UNDEFINED", "ERROR", "HOLD-DEV"):
        assert a.next_for(v) is None, v
    assert b.next_for("GO-TF") is None


def test_engine_pick_and_verdict_delegate_to_the_harness(tmp_path):
    from tools.engine.specs import three_factor_sleep as m
    d = tmp_path / "src"
    _write_dev(str(d))
    dev = {"all": {"units": {f: json.load(open(d / f)) for f in os.listdir(d)}}}
    p = m.pick(dev, {"res": str(tmp_path / "res")})
    assert p["params"] == (f"--arms {','.join(T.ARMS)} --lr {T.DEV_LRS[1]} --sigma {T.DEV_SIGMAS[2]} --tag six"
                           ) and p["harness_pick"]["verdict"] == "PICKED"
    d2 = tmp_path / "src2"
    _write_dev(str(d2), killed=True)
    assert m.pick({"all": {"units": {f: json.load(open(d2 / f)) for f in os.listdir(d2)}}}, {"res": str(tmp_path / "r2")}) is None
    assert json.load(open(tmp_path / "r2" / "harness_pick_dev.json"))["verdict"] == "KILL-DEV"   # survives the engine's pick.json
    s = tmp_path / "six"
    _write_six(str(s), _go(), lr=T.DEV_LRS[1], sigma=T.DEV_SIGMAS[2])
    six = {seed: {"units": {f: json.load(open(s / f)) for f in os.listdir(s) if f"_s{seed}_" in f}} for seed in T.SEEDS}
    v = m.verdict(six, {"harness_pick": dict(p["harness_pick"])}, {"res": str(tmp_path / "res")})
    assert v["verdict"] == "GO-TF" and "failed clauses []" in v["summary"]
    del six[42]
    assert m.verdict(six, {"harness_pick": dict(p["harness_pick"])}, {"res": str(tmp_path / "res")})["verdict"] == "ERROR"
