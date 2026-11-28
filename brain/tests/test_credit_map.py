# Capability: Sleep consolidation
# Record: brain/research/2026-11-28-local-rule-research-round.md
# Expected-fail: none (plumbing on a tiny random cortex and synthetic units; no model weights)
"""T0 credit map: the masked gradient night with every row is brain.sleep's night, answer rows predict the key, the
cosine is the flattened effective-update cosine, the init direction is minus the B gradient, the graded read-out is
consistent, the run writes every unit (with provenance, engagement, graded read-out) and resumes, and the verdict gives
the exact word with each control failing alone. The engine spec renders and delegates."""
import copy
import json
import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from brain.experiments import credit_map as C
from brain.plastic_cortex import PlasticCortex
from brain.sleep import sleep
from brain.tests.test_source_tag import _Cortex

FACTS_S = [("my cat is Pixel", "What is my cat?", "pixel"), ("my dog is Bolt", "What is my dog?", "bolt")]
TOLD = [(f"User: {s}\n\nAssistant: ok\n\n", 1.0) for s, _, _ in FACTS_S]
SHUF = [(TOLD[0][0].replace("Pixel", "Bolt"), 1.0), (TOLD[1][0].replace("Bolt", "Pixel"), 1.0)]
GEN = ["User: hi\n\nAssistant: yo\n\n"]


def _pc(layers=None):
    torch.manual_seed(1)
    return PlasticCortex(_Cortex(), rank=2, layers=layers)


def _st(pc):
    return [s.float() for s in pc.zero_state()]


# ------------------------------------------------------------------ mechanism
def test_masked_sleep_with_every_row_is_brain_sleep():
    a, b = _pc(), _pc()
    la = sleep(a, _st(a), TOLD, GEN, steps=6, lr=1e-2, seed=3)
    lb = C.masked_sleep(b, _st(b), TOLD, [None, None], GEN, steps=6, lr=1e-2, seed=3)
    assert torch.allclose(torch.tensor(la), torch.tensor(lb), atol=1e-6)
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.allclose(p, q, atol=1e-6)


def test_answer_rows_predict_the_key_and_masked_night_learns_them():
    pipe = _Cortex().pipe
    t = TOLD[0][0]
    rows = C.answer_rows(pipe, t, "pixel")
    ids = pipe.encode(t)
    assert pipe.decode([ids[r + 1] for r in rows]) == "Pixel"
    pc = _pc()
    rr = [C.answer_rows(pipe, x, k) for (x, _), (_, _, k) in zip(TOLD, FACTS_S)]
    l0 = C.masked_replay_ce(pc, _st(pc), TOLD, rr)
    C.masked_sleep(pc, _st(pc), TOLD, rr, GEN, steps=30, lr=3e-2, seed=0)
    assert C.masked_replay_ce(pc, _st(pc), TOLD, rr) < l0


def test_layer_restricted_night_moves_only_its_layer():
    pc = _pc(layers=[1])
    sleep(pc, _st(pc), TOLD, GEN, steps=5, lr=1e-2, seed=0)
    assert list(pc.A_k) == ["1"] and pc.adapter_norm() > 0 and set(C.layer_norms(pc)) == {"1"}


def test_layer_cosines_equal_the_flattened_cosine():
    torch.manual_seed(0)
    f = lambda: {"A_k": {"0": torch.randn(8, 2)}, "B_k": {"0": torch.randn(2, 16)},
                 "A_v": {"0": torch.randn(16, 2)}, "B_v": {"0": torch.randn(2, 8)}}
    f1, f2 = f(), f()
    c = C.layer_cosines(f1, f2)["0"]
    dk = [(x["A_k"]["0"].double() @ x["B_k"]["0"].double()).flatten() for x in (f1, f2)]
    dv = [(x["A_v"]["0"].double() @ x["B_v"]["0"].double()).flatten() for x in (f1, f2)]
    assert math.isclose(c["k"], float(F.cosine_similarity(dk[0], dk[1], 0)), abs_tol=1e-9)
    assert math.isclose(c["v"], float(F.cosine_similarity(dv[0], dv[1], 0)), abs_tol=1e-9)
    kv = [torch.cat((a, b)) for a, b in zip(dk, dv)]
    assert math.isclose(c["kv"], float(F.cosine_similarity(kv[0], kv[1], 0)), abs_tol=1e-9)
    assert math.isclose(C.layer_cosines(f1, f1)["0"]["kv"], 1.0, abs_tol=1e-9)
    neg = copy.deepcopy(f1)
    neg["B_k"]["0"], neg["B_v"]["0"] = -neg["B_k"]["0"], -neg["B_v"]["0"]
    assert math.isclose(C.layer_cosines(f1, neg)["0"]["kv"], -1.0, abs_tol=1e-9)
    z = copy.deepcopy(f1)
    z["B_k"]["0"].zero_()
    z["B_v"]["0"].zero_()
    assert C.layer_cosines(f1, z)["0"]["kv"] is None


def test_init_direction_is_minus_the_b_gradient_and_descends():
    pc = _pc()
    st = _st(pc)
    f, gn = C.init_direction(pc, st, TOLD, [None, None])
    assert gn > 0 and pc.adapter_norm() == 0.0 and all(p.grad is None for p in pc.parameters())
    l0 = C.masked_replay_ce(pc, st, TOLD, [None, None])
    with torch.no_grad():
        for s in pc.A_k:
            assert torch.equal(pc.A_k[s], f["A_k"][s])
            pc.B_k[s].add_(1e-2 * f["B_k"][s])
            pc.B_v[s].add_(1e-2 * f["B_v"][s])
    assert C.masked_replay_ce(pc, st, TOLD, [None, None]) < l0


def test_graded_readout_is_consistent():
    pc = _pc()
    st = _st(pc)
    g = C.graded(pc, st, FACTS_S)
    assert len(g["items"]) == 2 and all(1 <= it["rank_first"] <= 128 for it in g["items"])
    it = g["items"][0]
    pipe = pc.cortex.pipe
    p, a = pipe.encode("User: What is my cat?\n\nAssistant:"), pipe.encode(" Pixel")
    with torch.no_grad():
        lp = F.log_softmax(pc(p + a, st)[0], -1)
    assert math.isclose(it["logp_first"], float(lp[len(p) - 1, a[0]]), rel_tol=1e-6) and it["n_tok"] == len(a)
    assert math.isclose(g["mean_logp"], sum(x["logp"] for x in g["items"]) / 2, rel_tol=1e-9)
    assert C.surface("My cat is Pixel.", "pixel") == "Pixel"
    assert C.surface("I am allergic to peanuts.", "peanut") == "peanuts"      # the told word, not the stem
    assert C.surface("She lives in New York, sadly.", "new york") == "New York"


def test_guard():
    with pytest.raises(SystemExit, match="POOL_NODE"):
        C.guard({})
    C.guard({"POOL_NODE": "1"})
    C.guard({"BRAIN_ALLOW_W3090": "1"})


# ------------------------------------------------------------------ run on a stub cortex
def _args(tmp_path, **kw):
    a = dict(seed=7, arm_list=["all"], out_dir=str(tmp_path / "t0"), tag="t0", model="/x/m.pth", steps=3,
             grad_lr=1e-2, kl_weight=1.0, device="cpu")
    a.update(kw)
    return SimpleNamespace(**a)


def _kw():
    from brain.experiments.sleep_probe import GENERIC, evaluate
    return dict(GENERIC=GENERIC, build_arm_episodes=lambda s: {"sleep_true": TOLD, "sleep_shuffled": SHUF},
                evaluate=evaluate, sleep=sleep, keys_all=["pixel", "bolt"], facts=FACTS_S,
                PlasticCortex=lambda cx, layers=None: PlasticCortex(cx, rank=2, layers=layers))


def test_run_units_writes_every_unit_resumes_and_scores(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("COMMIT", "abc")
    c = _Cortex()
    a = _args(tmp_path)
    done = C.run_units(a, c, _st(_pc()), **_kw())
    L = c.model.n_layer
    assert done == C.all_arms(L) and done[-1] == "cosine"
    for arm in done:
        u = json.load(open(C.unit_path(a.out_dir, "t0", 7, arm)))
        assert {"host", "commit", "device", "precision", "model_id", "engagement", "L", "steps"} <= set(u), arm
        assert u["commit"] == "abc" and u["engagement"].get("kind"), arm
        if arm != "cosine":
            assert {"recall", "graded", "adapter_norm", "layer_norms", "generic_kl"} <= set(u["result"]), arm
            assert math.isfinite(u["result"]["graded"]["mean_logp"]), arm
    for arm in C.DELTA_ARMS:
        assert os.path.exists(C.delta_path(a.out_dir, "t0", 7, arm)), arm
    u = {arm: json.load(open(C.unit_path(a.out_dir, "t0", 7, arm))) for arm in done}
    assert u["grad_layer1"]["engagement"]["layers_on"] == [1] and list(u["grad_layer1"]["result"]["layer_norms"]) == ["1"]
    assert u["grad_reseed"]["engagement"]["replay_seed"] == 8 and u["v1"]["engagement"]["feedback"] == "transport"
    assert u["v3"]["engagement"]["teacher"] == "double" and u["grad_init"]["result"]["adapter_norm"] == 0.0
    assert u["grad_init"]["engagement"]["grad_norm"] > 0
    pairs = u["cosine"]["result"]["pairs"]
    assert set(pairs) == {f"{x}|{y}" for x, y in C.PAIRS} and set(pairs["v1|gradient"]) == {"0", "1"}
    assert math.isclose(pairs["grad_reseed|gradient"]["0"]["kv"], pairs["grad_reseed|gradient"]["0"]["kv"])
    assert C.run_units(a, c, _st(_pc()), **_kw()) == []                       # resume
    C.main(["score", "--dir", a.out_dir, "--tag", "t0", "--out", str(tmp_path / "v.json")])
    v = json.load(open(tmp_path / "v.json"))
    assert v["verdict"] in ("CONFIRMED", "FALSIFIED", "UNDEFINED") and v["instrument"]["commit"] == "abc"


def test_cosine_refuses_without_deltas_and_unknown_arms(tmp_path, monkeypatch):
    monkeypatch.setenv("COMMIT", "abc")
    with pytest.raises(SystemExit, match="delta files missing"):
        C.run_units(_args(tmp_path, arm_list=["no_sleep", "cosine"]), _Cortex(), _st(_pc()), **_kw())
    with pytest.raises(SystemExit, match="unknown arms"):
        C.run_units(_args(tmp_path, arm_list=["grad_layer9"]), _Cortex(), _st(_pc()), **_kw())


def test_score_refuses_mixed_commits(tmp_path):
    d = tmp_path / "d"
    for arm, cm in (("no_sleep", "a"), ("gradient", "b")):
        C.atomic_write(C.unit_path(str(d), "t0", 7, arm), {"tag": "t0", "seed": 7, "arm": arm, "commit": cm,
                                                           "model_id": "m", "L": 2, "steps": 3})
    with pytest.raises(SystemExit, match="commit"):
        C.main(["score", "--dir", str(d), "--tag", "t0"])


# ------------------------------------------------------------------ verdict on synthetic units
L4 = 4


def _night(norm=1.0, layers=None, start=2.0, end=1.0, recall=0, logp=-10.0, **eng):
    return {"result": {"recall": recall, "adapter_norm": norm, "graded": {"mean_logp": logp}},
            "engagement": {"steps_run": 240, "loss_start": start, "loss_end": end,
                           "layers_on": list(range(L4)) if layers is None else layers, **eng}}


def _confirmed():
    u = {"no_sleep": _night(norm=0.0, logp=-10.0), "gradient": _night(recall=4, logp=-6.0),
         "grad_reseed": _night(recall=4, logp=-6.1), "grad_shuffled": _night(recall=0, logp=-10.2),
         "v1": _night(norm=0.3, feedback="transport"),
         "v3": _night(norm=2.0, err_start=459.0, err_end=150.0, teacher="double"),
         "grad_distill": _night(recall=1), "grad_ans": _night(recall=3, logp=-7.0),
         "grad_init": _night(norm=0.0, grad_norm=1.5), "grad_init_ans": _night(norm=0.0, grad_norm=0.7)}
    for i in range(L4):
        u[f"grad_layer{i}"] = _night(layers=[i], recall=2 if i == 0 else 0, logp={0: -9.0, 1: -9.8}.get(i, -10.0))
        u[f"grad_ans_layer{i}"] = _night(layers=[i], logp=-9.5 if i == 0 else -10.0)
    pairs = {}
    for x, y in C.PAIRS:
        pairs[f"{x}|{y}"] = {str(i): {"k": 0.0, "v": 0.0, "kv": 0.02} for i in range(L4)}
    pairs["v3|gradient"]["0"]["kv"] = 0.05
    pairs["grad_reseed|gradient"]["0"]["kv"] = 0.8
    u["cosine"] = {"result": {"pairs": pairs}, "engagement": {"kind": "instrument"}}
    return u


def _set_cos(u, pair, layer, val):
    u["cosine"]["result"]["pairs"][pair][str(layer)]["kv"] = val


def test_confirmed_baseline_and_its_carrier_map():
    v = C.verdict(_confirmed(), L4)
    assert v["verdict"] == "CONFIRMED" and v["carriers"] == [0] and v["minority"] is True
    assert math.isclose(v["best_gain"], 1.0) and v["distill_reading"].startswith("target-limit")
    assert v["position_reading"].startswith("answer-positions-only keeps 0.75")


def _m(path, val):
    def f(u):
        d = u
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = val
    return f


CONTROLS = [
    ("missing arm", lambda u: u.pop("grad_layer3"), "missing arms"),
    ("no_sleep norm", _m(["no_sleep", "result", "adapter_norm"], 0.5), "no_sleep adapter_norm"),
    ("gradient not engaged", _m(["gradient", "engagement", "loss_end"], 3.0), "gradient not engaged"),
    ("gradient weak", _m(["gradient", "result", "recall"], 1), "instrument is too weak"),
    ("reseed not engaged", _m(["grad_reseed", "result", "adapter_norm"], 0.0), "grad_reseed not engaged"),
    ("shuffled not engaged", _m(["grad_shuffled", "engagement", "steps_run"], 0), "grad_shuffled not engaged"),
    ("shuffled recalls", _m(["grad_shuffled", "result", "recall"], 2), "test broken"),
    ("layer arm not engaged", _m(["grad_layer2", "engagement", "layers_on"], [0, 1, 2, 3]), "grad_layer2 not engaged"),
    ("v1 not engaged", _m(["v1", "result", "adapter_norm"], 0.0), "v1 not engaged"),
    ("v1 wrong feedback", _m(["v1", "engagement", "feedback"], "random"), "v1 not engaged"),
    ("v3 not engaged", _m(["v3", "engagement", "err_end"], 500.0), "v3 not engaged"),
    ("v3 not all layers", _m(["v3", "engagement", "layers_on"], [3]), "v3 not engaged"),
    ("recall non-finite", _m(["v1", "result", "recall"], None), "v1 recall non-finite"),
    ("graded non-finite", _m(["grad_layer1", "result", "graded", "mean_logp"], float("nan")), "graded mean_logp non-finite"),
    ("no carrier gain", lambda u: [u[f"grad_layer{i}"]["result"]["graded"].update(mean_logp=-9.95) for i in range(L4)],
     "no carrier map"),
    ("cosine non-finite", lambda u: _set_cos(u, "v1|gradient", 0, None), "cos(v1, gradient) layer 0 non-finite"),
    ("positive control", lambda u: _set_cos(u, "grad_reseed|gradient", 0, 0.4), "positive control"),
]


@pytest.mark.parametrize("name,mut,why", CONTROLS, ids=[c[0] for c in CONTROLS])
def test_each_control_failing_alone_is_undefined(name, mut, why):
    u = _confirmed()
    mut(u)
    v = C.verdict(u, L4)
    assert v["verdict"] == "UNDEFINED" and any(why in w for w in v["why"]), (name, v.get("why"))


@pytest.mark.parametrize("arm,mut", [
    ("grad_distill", _m(["grad_distill", "engagement", "loss_end"], 9.0)),
    ("grad_ans", _m(["grad_ans", "result", "adapter_norm"], 0.0)),
    ("grad_ans_layer2", _m(["grad_ans_layer2", "engagement", "steps_run"], 0)),
    ("grad_init", _m(["grad_init", "engagement", "grad_norm"], 0.0)),
    ("grad_init_ans", _m(["grad_init_ans", "engagement", "grad_norm"], None)),
])
def test_report_only_arms_failing_alone_leave_the_word(arm, mut):
    u = _confirmed()
    mut(u)
    v = C.verdict(u, L4)
    assert v["verdict"] == "CONFIRMED" and any(x.startswith(arm) for x in v["engagement"]["report"])
    if arm == "grad_distill":
        assert v["distill_reading"].startswith("undetermined")
    if arm == "grad_ans":
        assert v["position_reading"].startswith("undetermined")


def test_falsified_by_each_rule_with_its_reading():
    for rule in ("v1", "v3"):
        u = _confirmed()
        _set_cos(u, f"{rule}|gradient", 0, 0.35)
        v = C.verdict(u, L4)
        assert v["verdict"] == "FALSIFIED" and v["falsified_by"] == [rule] and "step size" in v["reading"][rule]
    u = _confirmed()
    _set_cos(u, "v1|gradient", 0, 0.35)
    u["v1"]["result"]["recall"] = 3
    assert C.verdict(u, L4)["reading"]["v1"].startswith("aligned and recalling")
    u = _confirmed()                                   # exactly at the bar is not above it
    _set_cos(u, "v1|gradient", 0, 0.3)
    assert C.verdict(u, L4)["verdict"] == "UNDEFINED"


def test_between_the_bars_and_majority_are_undefined():
    u = _confirmed()
    _set_cos(u, "v3|gradient", 0, 0.1)                 # not under 0.1
    v = C.verdict(u, L4)
    assert v["verdict"] == "UNDEFINED" and "between the bars" in v["why"][0]
    u = _confirmed()
    for i in (1, 2):
        u[f"grad_layer{i}"]["result"]["graded"]["mean_logp"] = -9.2   # gains 0.8 >= 0.5 x 1.0: R = [0, 1, 2]
        _set_cos(u, "grad_reseed|gradient", i, 0.8)
    v = C.verdict(u, L4)
    assert v["carriers"] == [0, 1, 2] and v["verdict"] == "UNDEFINED" and "not concentrated" in v["why"][0]


def test_positive_control_gates_the_falsifier_too():
    u = _confirmed()
    _set_cos(u, "v1|gradient", 0, 0.6)
    _set_cos(u, "grad_reseed|gradient", 0, 0.45)
    assert C.verdict(u, L4)["verdict"] == "UNDEFINED"


def test_distill_reading_rule_limit():
    u = _confirmed()
    u["grad_distill"]["result"]["recall"] = 2          # bar = max(2, 0 + 2) = 2
    assert C.verdict(u, L4)["distill_reading"].startswith("rule-limit")


# ------------------------------------------------------------------ engine spec
def test_engine_spec_renders_and_is_registered():
    from tools.engine import core
    s = core.Spec.find("credit_map")
    assert s.status == "REGISTERED" and s.model == "rwkv7-g1d-0.1b-20260129-ctx8192.pth" and "{model}" in s.cmd
    assert s.seeds == [7] and not s.dev_grid and s.pick_fn is None and s.smoke["params"].startswith("--arms ")
    c = s.render(7, s.six_params, "/o.json", commit="c" * 12, job="s7")
    assert "credit_map/cccccccccccc/s7" in c and "--model $BRAIN_MODEL_DIR/rwkv7-g1d-0.1b" in c
    assert "-m brain.experiments.credit_map run" in c and "--arms all --tag t0" in c and "--glob '*.json'" in c
    assert s.prereg in s.requires_local and os.path.exists(s.prereg)
    assert "STATUS: REGISTERED" in open(s.prereg).read() and "DRAFT" not in os.path.basename(s.prereg)


def test_engine_verdict_delegates_to_the_harness(tmp_path):
    from tools.engine.specs import credit_map as m
    u = _confirmed()
    units = {}
    for arm, x in u.items():
        units[f"t0_s7_{arm}.json"] = {**x, "tag": "t0", "seed": 7, "arm": arm, "commit": "abc", "model_id": "m",
                                      "L": L4, "steps": 240}
    v = m.verdict({7: {"units": units}}, {"params": ""}, {"res": str(tmp_path / "res")})
    assert v["verdict"] == "CONFIRMED" and v["summary"].startswith("CONFIRMED")
    assert m.verdict({8: {"units": units}}, {}, {"res": str(tmp_path / "r2")})["verdict"] == "ERROR"
    bad = {k: {**x, "commit": "zzz" if "v1" in k else "abc"} for k, x in units.items()}
    assert m.verdict({7: {"units": bad}}, {}, {"res": str(tmp_path / "r3")})["verdict"] == "ERROR"


# ------------------------------------------------------------------ review fixes (2026-11-28): sign, flat layers, bounds
@pytest.mark.parametrize("rule", ["v1", "v3"])
def test_strong_anti_alignment_is_not_confirmed(rule):
    u = _confirmed()
    _set_cos(u, f"{rule}|gradient", 0, -0.9)
    v = C.verdict(u, L4)
    assert v["verdict"] == "UNDEFINED" and v["anti_aligned"] == [rule] and "anti-aligned" in v["why"][0]
    u = _confirmed()
    _set_cos(u, f"{rule}|gradient", 0, -0.35)          # negative never falsifies (one-sided)
    v = C.verdict(u, L4)
    assert v["verdict"] == "UNDEFINED" and v["anti_aligned"] == [rule]


@pytest.mark.parametrize("val,word", [(-0.1, "UNDEFINED"), (-0.09, "CONFIRMED"), (0.09, "CONFIRMED"), (0.1, "UNDEFINED")])
def test_abs_cosine_bar_boundaries(val, word):
    u = _confirmed()
    _set_cos(u, "v1|gradient", 0, val)
    v = C.verdict(u, L4)
    assert v["verdict"] == word and v["anti_aligned"] == []


def test_flat_single_layer_night_is_a_reading_not_a_broken_run():
    u = _confirmed()
    u["grad_layer2"]["engagement"].update(loss_start=2.0, loss_end=2.0)
    u["grad_layer3"]["engagement"].update(loss_start=2.0, loss_end=2.1)
    u["grad_ans_layer1"]["engagement"].update(loss_start=2.0, loss_end=2.0)
    v = C.verdict(u, L4)
    assert v["verdict"] == "CONFIRMED" and v["engagement"]["report"] == []
    u["grad_layer2"]["engagement"]["loss_end"] = float("nan")       # non-finite loss still gates
    assert "grad_layer2 not engaged" in " ".join(C.verdict(u, L4)["why"])
    for arm in ("gradient", "grad_reseed", "grad_shuffled"):       # all-layer gradient arms stay strict
        u = _confirmed()
        u[arm]["engagement"].update(loss_start=2.0, loss_end=2.0)
        assert any(w.startswith(f"{arm} not engaged") for w in C.verdict(u, L4)["why"])


def test_missing_report_only_arm_is_undefined():
    u = _confirmed()
    u.pop("grad_ans_layer1")
    v = C.verdict(u, L4)
    assert v["verdict"] == "UNDEFINED" and "missing arms" in v["why"][0]


def test_carrier_fraction_and_gain_floor_boundaries():
    u = _confirmed()
    u["grad_layer1"]["result"]["graded"]["mean_logp"] = -9.5     # gain exactly 0.5 x best (1.0): a carrier
    _set_cos(u, "grad_reseed|gradient", 1, 0.8)
    assert C.verdict(u, L4)["carriers"] == [0, 1]
    u = _confirmed()
    u["no_sleep"]["result"]["graded"]["mean_logp"] = 0.0          # best exactly 0.1 nats (0.1 - 0.0 is exact): no map
    for i in range(L4):
        u[f"grad_layer{i}"]["result"]["graded"]["mean_logp"] = 0.1 if i == 0 else -0.5
    v = C.verdict(u, L4)
    assert v["best_gain"] == C.GAIN_MIN and v["verdict"] == "UNDEFINED" and "no carrier map" in " ".join(v["why"])


def test_recall_gate_boundaries():
    u = _confirmed()
    u["gradient"]["result"]["recall"] = 2                       # GRAD_MIN exactly: passes
    assert C.verdict(u, L4)["verdict"] == "CONFIRMED"
    u["gradient"]["result"]["recall"] = 1
    assert "too weak" in " ".join(C.verdict(u, L4)["why"])
    u = _confirmed()
    u["grad_shuffled"]["result"]["recall"] = u["no_sleep"]["result"]["recall"] + 1   # exactly no_sleep + 1: passes
    assert C.verdict(u, L4)["verdict"] == "CONFIRMED"


def test_positive_control_exactly_at_the_bar_passes():
    u = _confirmed()
    _set_cos(u, "grad_reseed|gradient", 0, 0.5)
    assert C.verdict(u, L4)["verdict"] == "CONFIRMED"
    _set_cos(u, "grad_reseed|gradient", 0, 0.4999)
    assert "positive control" in C.verdict(u, L4)["why"][0]


def test_half_the_layers_is_not_a_minority():
    u = _confirmed()
    u["grad_layer1"]["result"]["graded"]["mean_logp"] = -9.2     # R = [0, 1] = L/2 of L = 4
    _set_cos(u, "grad_reseed|gradient", 1, 0.8)
    v = C.verdict(u, L4)
    assert v["carriers"] == [0, 1] and v["minority"] is False and v["verdict"] == "UNDEFINED"
    assert "not concentrated" in " ".join(v["why"])
