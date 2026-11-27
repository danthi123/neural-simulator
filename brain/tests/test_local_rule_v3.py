# Capability: Sleep consolidation
# Record: brain/research/2026-11-27-local-rule-v3-DESIGN.md
# Expected-fail: none (plumbing on a tiny random cortex and synthetic records; no model weights)
"""Local rule v3: the recording forward is bit-identical to PlasticCortex, the lesion is exactly inert, top_only moves
only the last block, NLMS corrects a row at lr 1, G0 gates the nights, and the registered pick / verdict give the exact
label with each clause and each engagement gate failing alone. Engine specs render, chain and delegate."""
import copy
import json
import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from brain.experiments import local_rule_v3 as V
from brain.plastic_cortex import PlasticCortex
from brain.tests.test_source_tag import _Cortex

TEXT = "User: my cat is Pixel\n\nAssistant: ok\n\n"
TEXT2 = "User: my dog is Bolt\n\nAssistant: ok\n\n"
EPS = [(TEXT, 1.0), (TEXT2, 1.0)]
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
@pytest.mark.parametrize("adapters", [True, False])
def test_recording_forward_is_bit_identical_and_rebuilds_the_residual(adapters):
    pc = _pc(0.1)
    st, ids = _st(pc), pc.cortex.pipe.encode(TEXT)
    pc.enabled = adapters
    with torch.no_grad():
        ref = pc(ids, st)[0]
    pc.enabled = True
    lg, act = V.activity(V.adopt3(pc), ids, st, adapters=adapters)
    assert torch.equal(lg, ref)
    z, x = pc.z, act[pc.L - 1][3]
    assert torch.equal(F.layer_norm(x, (pc.C,), weight=z["ln_out.weight"], bias=z["ln_out.bias"]) @ z["head.weight"], ref)
    assert pc.enabled is True and V.adopt3(pc).rec is None


def test_teacher_reads_context_and_returns_the_telling_positions_only():
    pc = _pc(0.1)
    st = _st(pc)
    ids = pc.cortex.pipe.encode(TEXT)
    lg, t = V.teacher(V.adopt3(pc), ids, ids, st)
    assert lg.shape[0] == len(ids) and all(h.shape[0] == len(ids) and r.shape[0] == len(ids) for h, r in t.values())
    lg0, t0 = V.teacher(V.adopt3(pc), [], ids, st)
    pc.enabled = False
    with torch.no_grad():
        off = pc(ids, st)[0]
    assert torch.equal(lg0, off)                      # no context = the pretrained cortex on X (adapters off)
    assert not torch.allclose(lg, lg0)                # the first reading changes the second


def test_nlms_corrects_a_single_row_at_lr_one():
    torch.manual_seed(0)
    a, e, B = torch.randn(1, 3, dtype=torch.float64), torch.randn(1, 5, dtype=torch.float64), torch.randn(3, 5, dtype=torch.float64)
    B2 = B - V.nlms(a, e, eps=0.0)
    assert torch.allclose(a @ B2, a @ B - e)


def test_derangement_has_no_fixed_point_and_is_seeded():
    p = V.derangement(12, 42)
    assert sorted(p) == list(range(12)) and all(i != j for i, j in enumerate(p)) and p == V.derangement(12, 42)
    with pytest.raises(ValueError):
        V.derangement(1, 0)
    assert V.context_ids([[1], [2], [3]], 0, "sham", [2, 0, 1]) == [3] and V.context_ids([[1]], 0, "none") == []


def test_no_target_lesion_is_exactly_inert():
    pc = _pc()
    info = {}
    V.v3_sleep(pc, _st(pc), EPS, GEN, steps=10, lr=0.3, teacher_kind="none", info=info)
    assert pc.adapter_norm() == 0.0 and info["err_head"] == 0.0 and info["target_ctx_tokens"] == 0
    assert info["teacher"] == "none" and info["steps_run"] == 10


def test_v3_night_engages_and_top_only_moves_only_the_last_block():
    pc = _pc()
    info = {}
    V.v3_sleep(pc, _st(pc), EPS, GEN, steps=40, lr=0.3, info=info)
    assert pc.adapter_norm() > 0 and info["err_tail"] < info["err_head"] and info["layers_on"] == [0, 1]
    assert set(info["err_layer_first"]) == {"0", "1"}
    pc = _pc()
    info = {}
    V.v3_sleep(pc, _st(pc), EPS, GEN, steps=40, lr=0.3, top_only=True, info=info)
    assert info["layers_on"] == [1] and float((pc.B_k["0"].abs().sum() + pc.B_v["0"].abs().sum()).detach()) == 0.0
    assert float(pc.B_v["1"].detach().abs().sum()) > 0
    pc = _pc()
    info = {}
    V.v3_sleep(pc, _st(pc), EPS, GEN, steps=10, lr=0.3, teacher_kind="sham", seed=3, info=info)
    assert info["teacher"] == "sham" and info["sham_perm"] == [1, 0] and info["target_ctx_tokens"] > 0


def test_distill_night_trains_by_backprop():
    pc = _pc()
    info = {}
    V.distill_sleep(pc, _st(pc), EPS, GEN, steps=30, lr=1e-2, info=info)
    assert pc.adapter_norm() > 0 and info["kind"] == "grad_distill" and info["loss_tail"] < info["loss_head"]


def test_answer_index_and_g0_check_on_stub():
    pc = _pc()
    pipe = pc.cortex.pipe
    ids = pipe.encode(TEXT)
    assert V.answer_index(pipe, ids, TEXT, "pixel") == TEXT.index("Pixel")
    with pytest.raises(ValueError):
        V.answer_index(pipe, ids, TEXT, "volvo")
    r = V.g0_check(pc, _st(pc), [(TEXT, "pixel"), (TEXT2, "bolt")])
    assert r["n"] == 2 and set(r["items"][0]) >= {"answer_tok", "teacher_top1", "single_top1"}
    assert r["pass"] == V.g0_pass(r["n_ok"])
    assert V.g0_pass(10) and not V.g0_pass(9) and not V.g0_pass(None)


def test_guard():
    with pytest.raises(SystemExit, match="POOL_NODE"):
        V.guard({})
    V.guard({"POOL_NODE": "1"})
    V.guard({"BRAIN_ALLOW_W3090": "1"})


# ------------------------------------------------------------------ dev pick
REF = {"no_sleep": {"recall": 0}, "gradient": {"recall": 4, "generic_kl": 0.04, "heldout_kl_dev": 0.08}}
G0 = {"pass": True, "n": 12, "n_ok": 11}


def _row(lr=0.1, keep=1.0, recall=3, g=0.03, h=0.05, norm=1.0, eh=1.0, et=0.5):
    return {"lr": lr, "keep": keep, "recall": recall, "generic_kl": g, "heldout_kl_dev": h, "adapter_norm": norm,
            "err_head": eh, "err_tail": et}


def test_dev_pick_rule_and_each_disqualifier_alone():
    assert V.dev_pick(G0, REF, [_row()])["pick"] == {"lr": 0.1, "keep": 1.0}
    for bad in (dict(recall=1), dict(g=0.051), dict(h=0.101), dict(norm=0.0), dict(et=1.0), dict(recall=float("nan"))):
        assert V.dev_pick(G0, REF, [_row(**bad)])["verdict"] == "NEGATIVE-DEV", bad
    big = {"no_sleep": {"recall": 0}, "gradient": {"recall": 4, "generic_kl": 1.0, "heldout_kl_dev": 1.0}}
    assert V.dev_pick(G0, big, [_row(g=1.1)])["verdict"] == "NEGATIVE-DEV"          # no night above KL 1
    assert V.dev_pick(G0, big, [_row(h=1.1)])["verdict"] == "NEGATIVE-DEV"
    rows = [_row(lr=0.3, recall=4, h=0.05), _row(lr=0.1, recall=4, h=0.04), _row(lr=0.03, recall=4, h=0.04, g=0.01),
            _row(lr=0.03, keep=0.3, recall=3, h=0.0)]
    assert V.dev_pick(G0, REF, rows)["pick"] == {"lr": 0.03, "keep": 1.0}                # max recall, then drift, then lr
    assert V.dev_pick({"pass": False, "n": 12, "n_ok": 9}, REF, [_row()])["verdict"] == "STOP-G0"
    assert V.dev_pick(None, REF, [_row()])["verdict"] == "STOP-G0"
    nan = copy.deepcopy(REF)
    nan["gradient"]["heldout_kl_dev"] = float("nan")
    assert V.dev_pick(G0, nan, [_row()])["verdict"] == "UNDEFINED-DEV"


def test_dev_bar_binds_on_half_gradient():
    assert V.dev_bar(0, 4) == 2 and V.dev_bar(0, 6) == 3 and V.dev_bar(1, 2) == 3


# ------------------------------------------------------------------ verdict
LAYERS = list(range(12))


def _arm(recall, g=0.03, h=0.04, false=1, norm=1.0, eng=None):
    return {"recall": recall, "false_recall": false, "generic_kl": g, "heldout_kl": h, "adapter_norm": norm,
            "engagement": eng or {}}


def _go():
    loc = lambda t, lo, ctx: {"kind": "local", "teacher": t, "steps_run": 240, "layers_on": lo, "target_ctx_tokens": ctx,
                              "err_head": 1.0, "err_tail": 0.5}
    lossy = {"steps_run": 240, "loss_head": 2.0, "loss_tail": 0.5}
    a = {"no_sleep": _arm(0, 0, 0, 0, 0.0, {"kind": "none", "steps_run": 0}),
         "gradient": _arm(4, eng=lossy),
         "v3": _arm(4, eng=loc("double", LAYERS, 20)),
         "sham_ctx": _arm(0, eng=loc("sham", LAYERS, 20)),
         "no_target": _arm(0, 0, 0, 0, 0.0, loc("none", LAYERS, 0)),
         "top_only": _arm(2, eng=loc("double", [11], 20)),
         "grad_distill": _arm(4, eng=lossy)}
    return {arm: {s: copy.deepcopy(r) for s in V.SEEDS} for arm, r in a.items()}


def _set(recs, arm, seeds=V.SEEDS, **kw):
    for s in seeds:
        for k, v in kw.items():
            if k.startswith("eng_"):
                recs[arm][s]["engagement"][k[4:]] = v
            else:
                recs[arm][s][k] = v
    return recs


def _per_seed(recs, arm, key, vals):
    for s, v in zip(V.SEEDS, vals):
        recs[arm][s][key] = v
    return recs


def test_verdict_go_baseline():
    v = V.verdict(_go(), last_layer=11)
    assert v["verdict"] == "GO-V3" and all(v["clauses"].values()) and v["bar"] == 2


@pytest.mark.parametrize("name,edit,want", [
    ("a recall", lambda r: _set(r, "v3", recall=1), "NEGATIVE"),
    ("b wins", lambda r: _per_seed(_per_seed(r, "v3", "recall", [2, 2, 5, 5, 5, 5]), "sham_ctx", "recall", [3, 3, 0, 0, 0, 0]), "NEGATIVE"),
    ("b sham mean", lambda r: _set(r, "sham_ctx", recall=1.5), "NEGATIVE"),
    ("c generic", lambda r: _set(r, "v3", generic_kl=0.05), "NEGATIVE"),
    ("c heldout", lambda r: _set(r, "v3", heldout_kl=0.06), "NEGATIVE"),
    ("c night", lambda r: _per_seed(_set(r, "gradient", generic_kl=2.0), "v3", "generic_kl", [1.2, .03, .03, .03, .03, .03]), "NEGATIVE"),
    ("d recall", lambda r: _set(r, "no_target", recall=1), "UNDEFINED"),
    ("d norm", lambda r: _set(r, "no_target", seeds=[42], adapter_norm=0.01), "UNDEFINED"),
    ("e false", lambda r: _set(r, "v3", false_recall=2.5), "NEGATIVE"),
    ("shallow", lambda r: _set(r, "top_only", recall=3.5), "GO-SHALLOW"),
    ("gradient < 2", lambda r: _set(r, "gradient", recall=1), "UNDEFINED"),
])
def test_each_clause_failing_alone_gives_the_exact_verdict(name, edit, want):
    v = V.verdict(edit(_go()), last_layer=11)
    assert v["verdict"] == want, (name, v.get("clauses"), v.get("why"))
    if want == "NEGATIVE" and name[0] in "ace":
        failed = [k for k, ok in v["clauses"].items() if not ok]
        assert len(failed) == 1 and failed[0][0] == name[0], failed


@pytest.mark.parametrize("name,edit", [
    ("no_sleep norm", lambda r: _set(r, "no_sleep", seeds=[43], adapter_norm=0.1)),
    ("gradient loss flat", lambda r: _set(r, "gradient", seeds=[44], eng_loss_tail=3.0)),
    ("gradient norm 0", lambda r: _set(r, "gradient", seeds=[44], adapter_norm=0.0)),
    ("v3 err flat", lambda r: _set(r, "v3", seeds=[100], eng_err_tail=1.0)),
    ("v3 wrong teacher", lambda r: _set(r, "v3", seeds=[100], eng_teacher="sham")),
    ("v3 norm 0", lambda r: _set(r, "v3", seeds=[100], adapter_norm=0.0)),
    ("sham_ctx weak", lambda r: _set(r, "sham_ctx", seeds=[101], adapter_norm=0.4)),
    ("sham_ctx err flat", lambda r: _set(r, "sham_ctx", seeds=[101], eng_err_tail=2.0)),
    ("sham_ctx no context", lambda r: _set(r, "sham_ctx", seeds=[101], eng_target_ctx_tokens=0)),
    ("top_only norm 0", lambda r: _set(r, "top_only", seeds=[102], adapter_norm=0.0)),
    ("top_only all layers", lambda r: _set(r, "top_only", seeds=[102], eng_layers_on=LAYERS)),
    ("top_only wrong layer", lambda r: _set(r, "top_only", seeds=[102], eng_layers_on=[5])),
    ("no_target with context", lambda r: _set(r, "no_target", seeds=[42], eng_target_ctx_tokens=20, eng_teacher="double")),
    ("no_target never ran", lambda r: _set(r, "no_target", seeds=[42], eng_steps_run=0)),
])
def test_each_engagement_gate_alone_is_undefined_not_go(name, edit):
    v = V.verdict(edit(_go()), last_layer=11)
    assert v["verdict"] == "UNDEFINED", (name, v)


def test_a_core_failure_stays_negative_with_an_inert_control():
    r = _set(_set(_go(), "v3", recall=1), "sham_ctx", adapter_norm=0.0)
    assert V.verdict(r, 11)["verdict"] == "NEGATIVE"


def test_grad_distill_is_report_only_and_reads_a_negative():
    assert V.verdict(_set(_go(), "grad_distill", adapter_norm=0.0), 11)["verdict"] == "GO-V3"
    assert V.verdict(_set(_go(), "grad_distill", recall=float("nan")), 11)["verdict"] == "GO-V3"
    neg = lambda: _set(_go(), "v3", recall=1)
    assert V.verdict(neg(), 11)["reading"].startswith("rule-limit")
    assert V.verdict(_set(neg(), "grad_distill", recall=1), 11)["reading"].startswith("target-limit")
    assert V.verdict(_set(neg(), "grad_distill", heldout_kl=0.2), 11)["reading"].startswith("target-limit")
    assert V.verdict(_set(neg(), "grad_distill", eng_loss_tail=9.0), 11)["reading"].startswith("undetermined")


@pytest.mark.parametrize("arm", V.GATED_ARMS)
@pytest.mark.parametrize("key", V.GATED)
def test_any_nonfinite_gated_value_is_undefined(arm, key):
    r = _set(_go(), arm, seeds=[42], **{key: float("nan")})
    assert V.verdict(r, 11)["verdict"] == "UNDEFINED"


def test_verdict_needs_exactly_the_six_seeds_in_every_arm():
    r = _go()
    del r["top_only"][102]
    with pytest.raises(ValueError):
        V.verdict(r)
    r = _go()
    r["v3"][7] = r["v3"][42]
    with pytest.raises(ValueError):
        V.verdict(r)
    r = _go()
    del r["grad_distill"]
    with pytest.raises(ValueError):
        V.verdict(r)


def test_combine_two_scales():
    assert V.combine("GO-V3", "GO-V3") == "GO-V3"
    assert V.combine("GO-V3", "NEGATIVE") == "GO-V3-PARTIAL" and V.combine("NEGATIVE-DEV", "GO-V3") == "GO-V3-PARTIAL"
    assert V.combine("GO-SHALLOW", "NEGATIVE") == "GO-SHALLOW"
    assert V.combine("NEGATIVE", "NEGATIVE-DEV") == "NEGATIVE"
    for w in ("UNDEFINED", "UNDEFINED-DEV", "ERROR-SMOKE", "HOLD-DEV", None):
        assert V.combine("GO-V3", w) == "UNDEFINED", w


# ------------------------------------------------------------------ CLI on synthetic units
def _unit(tag, seed, arm, res, eng=None, lr=0.1, keep=1.0, model="m.pth", commit="abc"):
    loc = arm in V.LOCAL_ARMS
    return {"tag": tag, "seed": seed, "arm": arm, "model_id": model, "commit": commit, "prompts_sha256": "p",
            "lr": lr if loc else None, "keep": keep if loc else None, "last_layer": 11, "engagement": eng or {},
            "result": res}


def _write_six(d, recs, **kw):
    os.makedirs(d, exist_ok=True)
    for arm, by in recs.items():
        for s, r in by.items():
            r = dict(r)
            eng = r.pop("engagement")
            V.atomic_write(V.unit_path(d, "six", s, arm), _unit("six", s, arm, r, eng, **kw))


def _pickfile(tmp_path, lr=0.1, keep=1.0, model="m.pth", commit="abc"):
    p = str(tmp_path / "pick.json")
    V.atomic_write(p, {"verdict": "PICKED", "pick": {"lr": lr, "keep": keep}, "model_id": model, "commit": commit})
    return p


def test_score_cli_go_missing_and_mixed(tmp_path):
    d = str(tmp_path / "six")
    _write_six(d, _go())
    out = str(tmp_path / "v.json")
    V.main(["score", "--dir", d, "--tag", "six", "--out", out, "--pick", _pickfile(tmp_path)])
    assert json.load(open(out))["verdict"] == "GO-V3"
    with pytest.raises(SystemExit, match="pick"):
        V.main(["score", "--dir", d, "--tag", "six", "--pick", _pickfile(tmp_path, keep=3.0)])
    os.remove(V.unit_path(d, "six", 42, "top_only"))
    with pytest.raises(SystemExit, match="missing units"):
        V.main(["score", "--dir", d, "--tag", "six"])
    d2 = str(tmp_path / "mixed")
    _write_six(d2, _go())
    u = json.load(open(V.unit_path(d2, "six", 43, "v3")))
    u["commit"] = "zzz"
    V.atomic_write(V.unit_path(d2, "six", 43, "v3"), u)
    with pytest.raises(SystemExit, match="commit"):
        V.main(["score", "--dir", d2, "--tag", "six"])


def _write_dev(d, g0_pass=True, skip=None):
    os.makedirs(d, exist_ok=True)
    V.atomic_write(V.unit_path(d, "ref", 7, "g0"), _unit("ref", 7, "g0", {"pass": g0_pass, "n": 12, "n_ok": 11 if g0_pass else 8}))
    V.atomic_write(V.unit_path(d, "ref", 7, "no_sleep"), _unit("ref", 7, "no_sleep", REF["no_sleep"]))
    V.atomic_write(V.unit_path(d, "ref", 7, "gradient"), _unit("ref", 7, "gradient", REF["gradient"]))
    for lr in V.DEV_LRS:
        for k in V.DEV_KEEPS:
            if (lr, k) == skip:
                continue
            tag = f"lr{lr:g}_keep{k:g}"
            r = _row(lr, k, recall=3 if (lr, k) == (0.1, 3.0) else 1)
            eng = {"err_head": r.pop("err_head"), "err_tail": r.pop("err_tail")}
            V.atomic_write(V.unit_path(d, tag, 7, "v3"), _unit(tag, 7, "v3", r, eng, lr=lr, keep=k))


def test_pick_cli(tmp_path):
    d, out = str(tmp_path / "dev"), str(tmp_path / "pick.json")
    _write_dev(d)
    V.main(["pick", "--dir", d, "--out", out])
    p = json.load(open(out))
    assert p["verdict"] == "PICKED" and p["pick"] == {"lr": 0.1, "keep": 3.0} and p["g0"]["pass"] is True
    d2 = str(tmp_path / "dev2")
    _write_dev(d2, g0_pass=False)
    V.main(["pick", "--dir", d2, "--out", out])
    assert json.load(open(out))["verdict"] == "STOP-G0"
    d3 = str(tmp_path / "dev3")
    _write_dev(d3, skip=(0.3, 0.3))
    with pytest.raises(SystemExit, match="9 registered"):
        V.main(["pick", "--dir", d3])
    d4 = str(tmp_path / "dev4")
    _write_dev(d4)
    os.remove(V.unit_path(d4, "ref", 7, "g0"))
    with pytest.raises(SystemExit, match="G0"):
        V.main(["pick", "--dir", d4])


# ------------------------------------------------------------------ run_units on a stub cortex
def _run_args(tmp_path, **kw):
    a = dict(seed=42, arm_list=["g0"] + list(V.ARMS), out_dir=str(tmp_path / "six"), tag="six", model="/x/m.pth",
             steps=3, lr=0.1, keep=1.0, decay=1e-3, kl_weight=1.0, grad_lr=5e-4, device="cpu", pick=None,
             prompts_sha=V.PROMPTS_SHA256)
    a.update(kw)
    return SimpleNamespace(**a)


def _kw():
    from brain.experiments.mix_night_screen import PROMPTS
    from brain.experiments.sleep_probe import GENERIC, evaluate
    from brain.sleep import sleep
    return dict(PROMPTS=PROMPTS, GENERIC=GENERIC, build_arm_episodes=lambda s: {"sleep_true": EPS}, evaluate=evaluate,
                sleep=sleep, keys_all=["pixel", "bolt"], told=[(TEXT, "pixel"), (TEXT2, "bolt")],
                PlasticCortex=lambda cx: PlasticCortex(cx, rank=2))


def test_run_units_on_stub_cortex_resume_engagement_and_pick(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "G0_MIN", 0)
    monkeypatch.setenv("COMMIT", "abc")
    c = _Cortex()
    base = _st(_pc())
    a = _run_args(tmp_path, pick=_pickfile(tmp_path, lr=0.3, keep=3.0, model="m.pth", commit="abc"))
    done = V.run_units(a, c, base, **_kw())
    assert done == ["g0"] + list(V.ARMS)
    units = {arm: json.load(open(V.unit_path(a.out_dir, "six", 42, arm))) for arm in V.ARMS}
    for arm, u in units.items():
        assert {"host", "commit", "device", "precision", "model_id", "prompts_sha256", "engagement", "last_layer"} <= set(u)
        assert all(k in u["result"] for k in V.GATED + ("heldout_kl_dev", "adapter_norm")), arm
        assert u["engagement"].get("kind"), arm
    assert units["v3"]["lr"] == 0.3 and units["v3"]["keep"] == 3.0 and units["gradient"]["lr"] is None
    assert units["no_sleep"]["result"]["adapter_norm"] == 0.0 and units["no_target"]["result"]["adapter_norm"] == 0.0
    assert units["top_only"]["engagement"]["layers_on"] == [1] and units["sham_ctx"]["teacher"] == "sham"
    V.check_units(list(units.values()), V.read_pick(a.pick))
    assert V.run_units(a, c, base, **_kw()) == []                                  # resume
    with pytest.raises(SystemExit, match="model"):
        V.run_units(_run_args(tmp_path, model="/x/m04.pth", out_dir=str(tmp_path / "o2"), pick=a.pick), c, base, **_kw())
    monkeypatch.setenv("COMMIT", "zzz")
    with pytest.raises(SystemExit, match="commit"):
        V.run_units(_run_args(tmp_path, out_dir=str(tmp_path / "o3"), pick=a.pick), c, base, **_kw())


def test_failing_g0_stops_before_any_night(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "G0_MIN", 99)
    monkeypatch.setenv("COMMIT", "abc")
    c = _Cortex()
    base = _st(_pc())
    a = _run_args(tmp_path)
    with pytest.raises(SystemExit, match="G0 FAILED"):
        V.run_units(a, c, base, **_kw())
    assert sorted(os.listdir(a.out_dir)) == ["six_s42_g0.json"]
    with pytest.raises(SystemExit, match="G0 FAILED"):                              # resume does not skip the gate
        V.run_units(a, c, base, **_kw())
    assert sorted(os.listdir(a.out_dir)) == ["six_s42_g0.json"]


def test_dev_seed_never_computes_the_verdict_half(tmp_path, monkeypatch):
    monkeypatch.setenv("COMMIT", "abc")
    a = _run_args(tmp_path, seed=7, arm_list=["no_sleep", "v3"], out_dir=str(tmp_path / "dev"), tag="lr0.1_keep1")
    V.run_units(a, _Cortex(), _st(_pc()), **_kw())
    for arm in ("no_sleep", "v3"):
        r = json.load(open(V.unit_path(a.out_dir, "lr0.1_keep1", 7, arm)))["result"]
        assert r["heldout_kl"] is None and math.isfinite(r["heldout_kl_dev"])


# ------------------------------------------------------------------ engine specs
def test_engine_specs_render_chain_and_are_draft():
    from tools.engine import core
    a, b = core.Spec.find("local_rule_v3_01b"), core.Spec.find("local_rule_v3_04b")
    assert (a.status, b.status, a.memory_gb, b.memory_gb) == ("DRAFT", "DRAFT", 5, 10) and a.question == b.question
    assert a.model == "rwkv7-g1d-0.1b-20260129-ctx8192.pth" and b.model == "rwkv7-g1d-0.4b-20260210-ctx8192.pth"
    assert "{model}" in a.cmd and "{model}" in b.cmd and a.smoke["params"].startswith("--arms g0,")
    assert len(a.dev_grid) == 10 and a.dev_grid == b.dev_grid
    cells = {(float(p.split("--lr ")[1].split()[0]), float(p.split("--keep ")[1].split()[0]))
             for k, p in a.dev_grid.items() if k != "ref"}
    assert cells == {(lr, k) for lr in V.DEV_LRS for k in V.DEV_KEEPS}
    assert a.dev_grid["ref"] == "--arms g0,no_sleep,gradient --tag ref"
    c = a.render(7, a.dev_grid["lr0.1_keep1"], "/o.json", commit="c" * 12, job="lr0.1_keep1")
    assert "local_rule_v3_01b/cccccccccccc/lr0.1_keep1" in c and "--model $BRAIN_MODEL_DIR/rwkv7-g1d-0.1b" in c
    assert "-m brain.experiments.local_rule_v3 run" in c and "tools.engine.bundle $D /o.json --glob '*.json'" in c
    for v in ("GO-V3", "GO-SHALLOW", "NEGATIVE", "NEGATIVE-DEV"):
        assert a.next_for(v) == "local_rule_v3_04b", v
    for v in ("UNDEFINED", "ERROR", "HOLD-DEV"):
        assert a.next_for(v) is None, v
    assert b.next_for("GO-V3") is None


def test_engine_refuses_draft_without_allow_draft():
    from tools.engine import core
    from tools.engine.core import Engine, make_nodes
    s = core.Spec.find("local_rule_v3_01b")
    with pytest.raises(SystemExit, match="DRAFT"):
        Engine("local_rule_v3_01b", make_nodes("ssh"), dry=True, log=lambda *_: None).check_runnable(s)
    Engine("local_rule_v3_01b", make_nodes("ssh"), dry=True, log=lambda *_: None, allow_draft=True).check_runnable(s)


def test_engine_pick_and_verdict_delegate_to_the_harness(tmp_path):
    from tools.engine.specs import local_rule_v3 as m
    d = tmp_path / "src"
    _write_dev(str(d))
    dev = {"all": {"units": {f: json.load(open(d / f)) for f in os.listdir(d)}}}
    p = m.pick(dev, {"res": str(tmp_path / "res")})
    assert p["params"] == f"--arms {','.join(V.ARMS)} --lr 0.1 --keep 3.0 --tag six" and p["harness_pick"]["verdict"] == "PICKED"
    d2 = tmp_path / "src2"
    _write_dev(str(d2), g0_pass=False)
    p2 = m.pick({"all": {"units": {f: json.load(open(d2 / f)) for f in os.listdir(d2)}}}, {"res": str(tmp_path / "res2")})
    assert "STOP-G0" in p2["stop"]
    s = tmp_path / "six"
    _write_six(str(s), _go(), keep=3.0)
    six = {seed: {"units": {f: json.load(open(s / f)) for f in os.listdir(s) if f"_s{seed}_" in f}} for seed in V.SEEDS}
    hp = dict(p["harness_pick"])
    v = m.verdict(six, {"harness_pick": hp}, {"res": str(tmp_path / "res")})
    assert v["verdict"] == "GO-V3" and "failed clauses []" in v["summary"]
    del six[42]
    assert m.verdict(six, {"harness_pick": hp}, {"res": str(tmp_path / "res")})["verdict"] == "ERROR"
