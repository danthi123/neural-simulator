# Capability: Self and identity
# Record: brain/research/2026-11-27-voice-direction-ceiling-prereg.md
# Expected-fail: none (plumbing and the exact verdict rule on synthetic records and a tiny random cortex; no model weights)
"""Oracle source-direction ceiling: residual capture is exact, vectors leave the item out, the host guard refuses, the run
resumes per unit and writes atomically, and every clause of the verdict rule fails alone with the exact verdict."""
import copy
import json

import pytest
import torch

from brain import source_monitor as SM
from brain.experiments import voice_ceiling_oracle as VC
from brain.experiments import voice_tag as VT
from brain.tests.test_brain_key_switch import StubCortex
from brain.tests.test_source_tag import _plastic
from brain.tests.test_voice_tag import _life  # noqa: F401
from brain.tests.test_life import BKW, _env  # noqa: F401  (autouse fixture: fake night, 3 turns a day)

TEXT = "User: what is my cat called?\n\nAssistant:"


# ------------------------------------------------------------------ plumbing
def test_guard_refuses_without_a_node_flag():
    with pytest.raises(SystemExit):
        VC.guard({})
    with pytest.raises(SystemExit):
        VC.guard({"POOL_NODE": "0"})
    VC.guard({"POOL_NODE": "1"})
    VC.guard({"BRAIN_ALLOW_W3090": "1"})


def test_run_command_is_guarded(monkeypatch):
    monkeypatch.delenv("POOL_NODE", raising=False)
    monkeypatch.delenv("BRAIN_ALLOW_W3090", raising=False)
    with pytest.raises(SystemExit):
        VC.main(["run", "--life", "x", "--snaps", "x", "--trunk", "x", "--out", "x", "--work", "x"])


def test_capture_is_exact_and_the_footprint_adds_exactly_v_at_the_label():
    p = _plastic()
    pipe = p.cortex.pipe
    ids = pipe.encode(TEXT)
    sink = []
    with torch.no_grad(), VC.grab_residuals(p.L, sink):
        ref_logits, _ = p(ids)
    assert len(sink) == p.L and sink[0].shape == (p.C,)
    with torch.no_grad():
        again, _ = p(ids)
    assert torch.equal(again, ref_logits)                       # the patch is gone and changed nothing
    v = torch.randn(p.C)
    prov = SM.Provenance("own")
    SM.attach(p, v, 0, False, prov)
    sink2 = []
    with torch.no_grad(), VC.grab_residuals(p.L, sink2):
        p(ids)
    assert torch.allclose(sink2[0], sink[0] + v, atol=1e-5)     # label position tagged: block 0's residual shifts by v
    assert prov.stats["tagged"] > 0 and prov.stats["tagged"] < prov.stats["tokens"]
    SM.detach(p)


def test_capture_label_returns_every_block_and_refuses_a_tagged_or_inactive_brain():
    p = _plastic()

    class _C:
        active = True
        def read(self, text, state=None):
            return p(p.cortex.pipe.encode(text), state)

    class _B:
        cortex = _C()
        def _plastic(self):
            return p
    t = VC.capture_label(_B(), TEXT, None)
    assert t.shape == (p.L, p.C)
    SM.attach(p, torch.ones(p.C), 0, False, SM.Provenance("own"))
    with pytest.raises(RuntimeError):
        VC.capture_label(_B(), TEXT, None)
    SM.detach(p)
    _B.cortex.active = False
    with pytest.raises(RuntimeError):
        VC.capture_label(_B(), TEXT, None)


def test_vectors_leave_the_item_out_and_wrong_drops_its_source():
    qs = [{"item": i} for i in ("a", "a", "b", "c", "d")]
    d = [torch.full((2, 3), x) for x in (1.0, 3.0, 10.0, 100.0, 1000.0)]
    im = VC.item_means(qs, d)
    assert float(im["a"][0, 0]) == 2.0
    o = VC.loo_means(im)
    assert float(o["a"][0, 0]) == pytest.approx((10 + 100 + 1000) / 3)       # item a never in its own vector
    assert float(o["d"][0, 0]) == pytest.approx((2 + 10 + 100) / 3)
    wm = VC.wrong_map(["a", "b", "c", "d"])
    assert all(k != v for k, v in wm.items()) and set(wm) == set(wm.values())
    w = VC.loo_means(im, drop_of=lambda it: [k for k, x in wm.items() if x == it])
    for it in w:                                                       # nor the item whose unrelated turn WAS this item
        src = {k for k, x in wm.items() if x == it}
        rest = [float(im[k][0, 0]) for k in im if k != it and k not in src]
        assert float(w[it][0, 0]) == pytest.approx(sum(rest) / len(rest))
    with pytest.raises(ValueError):
        VC.wrong_map(["a", "b"])
    with pytest.raises(ValueError):
        VC.loo_means({"a": torch.zeros(1)})


def test_sham_is_norm_matched_deterministic_and_not_the_oracle():
    o = torch.randn(16)
    a, b = VC.sham_vector(319, 3, 12, o), VC.sham_vector(319, 3, 12, o)
    assert torch.equal(a, b) and float(a.norm()) == pytest.approx(float(o.norm()), rel=1e-5)
    assert not torch.equal(a, VC.sham_vector(319, 4, 12, o))
    assert abs(float(torch.nn.functional.cosine_similarity(a, o, dim=0))) < 0.99


def test_boot_diff_is_item_clustered_and_strict():
    a = {(1, "x", "q1"): 2.0, (1, "x", "q2"): 2.0, (1, "y", "q1"): 2.0, (2, "y", "q1"): 2.0}
    b = {k: 0.0 for k in a}
    r = VC.boot_diff(a, b, draws=200)
    assert r["diff"] == 2.0 and r["ci"] == [2.0, 2.0] and r["clusters"] == 2 and r["n"] == 4
    z = VC.boot_diff(b, b, draws=200)
    assert z["ci"] == [0.0, 0.0]                                   # lower bound 0 is not > 0
    assert VC.boot_diff({}, b)["ci"] is None


def test_reproduce_counts_identical_replies():
    rows = [{"you": "q1", "brain": "a"}, {"you": "q2", "brain": "b"}]
    ref = [{"you": "q1", "brain": "a"}, {"you": "q2", "brain": "x"}, {"you": "q3", "brain": "z"}]
    r = VC.reproduce(rows, ref)
    assert (r["n"], r["same"], r["frac"]) == (2, 1, 0.5)


def test_reproduce_gate_needs_coverage_replies_and_onsets():
    rows = [{"you": f"q{i}", "brain": "a", "onset": 0.1 * i} for i in range(10)]
    ref = [dict(r) for r in rows]
    r = VC.reproduce(rows, ref)
    assert (r["onset_frac"], r["frac"]) == (1.0, 1.0) and VC.repro_ok(r, 10)
    assert not VC.repro_ok(VC.reproduce(rows[:2], ref), 10)               # 2 of 10 questions matched: coverage fails
    off = [dict(r, onset=r["onset"] + 0.01 * (i < 1)) for i, r in enumerate(ref)]
    assert VC.repro_ok(VC.reproduce(rows, off), 10)                       # 9 of 10 onsets match: exactly the bar
    off = [dict(r, onset=r["onset"] + 0.01 * (i < 2)) for i, r in enumerate(ref)]
    assert not VC.repro_ok(VC.reproduce(rows, off), 10)                   # 8 of 10 onsets match: fails
    assert not VC.repro_ok(VC.reproduce(rows, [{k: v for k, v in x.items() if k != "onset"} for x in ref]), 10)
    assert not VC.repro_ok(None, 10) and not VC.repro_ok(r, None)


# ------------------------------------------------------------------ the verdict rule on synthetic records
ITEMS = {"pixel": "Pixel", "marguerite": "Marguerite", "volvo": "Volvo", "admin": "Admin"}
LED = {it: {"telling": f"My {it} is {k}.", "keys": [k], "told": True} for it, k in ITEMS.items()}
L = 12
MASK = {"calls": 9, "tokens": 90, "tagged": 30, "registered_hits": 0}


def _reply(kind, item):
    k = ITEMS[item]
    return {"uv": f"My {item} is called {k}.", "your": f"Your {item} is called {k}.", "none": "Hmm.",
            "refuse": f"Sorry, perhaps {k}."}[kind]


def _default(arm, seed, item, qi):
    if arm in ("C", "ORACLE"):
        return "your", 1.0
    return "uv", {"SHAM": -0.9, "ALL": -0.8, "WRONG": -0.7}.get(arm, -1.0)


def _outs(mod=None, seeds=(319, 320, 321), repro=1.0, mask=None, led_keys=True, nq=3, repro_n=None, onset_repro=1.0,
          label_tagged=True):
    outs = []
    nt = 4 * nq
    for s in seeds:
        o = {"seed": s, "layer": L, "report_layers": [], "n_questions": nt, "commit": "abc", "night": 14,
             "reproduction": {"vt": {"n": nt if repro_n is None else repro_n, "same": nt, "frac": repro,
                                     "onset_same": nt, "onset_frac": onset_repro}},
             "batteries": {}}
        for arm in ("H", "C", "ORACLE", "SHAM", "ALL", "WRONG"):
            name = arm if arm in ("H", "C") else f"{arm}@{L}"
            rows = []
            for it in ITEMS:
                for qi in range(nq):
                    kind, on = _default(arm, s, it, qi)
                    if mod:
                        kind, on = mod(arm, s, it, qi, kind, on)
                    rows.append({"kind": "H", "item": it, "you": f"{it} {qi}?", "brain": _reply(kind, it) if led_keys or arm != "H"
                                 else "Hmm.", "onset": on,
                                 "label_tagged": None if arm in ("H", "C") else label_tagged})
            o["batteries"][name] = {"rows": rows, "mask": None if arm in ("H", "C") else dict(mask or MASK,
                                                                                               **({"tagged": 90} if arm == "ALL" else {}))}
        outs.append(o)
    return outs


def _verdict(**kw):
    outs = _outs(**kw)
    return VC.table(outs, {s: LED for s in (319, 320, 321)})


def test_baseline_is_alive():
    t = _verdict()
    assert t["verdict"] == "CLASS-ALIVE", t
    assert t["failed"] == [] and t["clauses"]["ok"]["onset"]
    assert t["clauses"]["per_item"]["volvo"]["holds"] and t["C"]["knows"] == 36 and t["clauses"]["SHAM"]["p_uv"] == 1.0


def test_onset_clause_failing_closes_the_class():
    t = _verdict(mod=lambda a, s, i, q, k, o: (k, -1.5) if a == "ORACLE" else (k, o))
    assert t["verdict"] == "CLASS-CLOSED" and "onset" in t["why"]


def test_voice_clause_failing_closes_the_class():
    t = _verdict(mod=lambda a, s, i, q, k, o: ("uv", o) if a == "ORACLE" else (k, o))
    assert t["verdict"] == "CLASS-CLOSED" and t["failed"] == ["voice", "voice_sham", "seeds", "items"]


def _fails_alone(mod, clause):
    t = _verdict(mod=mod)
    assert t["verdict"] == "MIXED", t
    assert t["failed"] == [clause], t["failed"]


@pytest.mark.parametrize("arm,clause", [("SHAM", "sham"), ("ALL", "all"), ("WRONG", "wrong")])
def test_each_control_clause_fails_alone(arm, clause):
    _fails_alone(lambda a, s, i, q, k, o: (k, 1.0) if a == arm else (k, o), clause)


def test_knows_clause_fails_alone():
    _fails_alone(lambda a, s, i, q, k, o: ("none", o) if a == "ORACLE" and q == 0 else (k, o), "knows")


def test_refusal_clause_fails_alone():
    _fails_alone(lambda a, s, i, q, k, o: ("refuse", o) if a == "ORACLE" and q == 0 else (k, o), "refusals")


def test_seed_consistency_clause_fails_alone():
    _fails_alone(lambda a, s, i, q, k, o: ("uv", -0.5) if a == "ORACLE" and s != 319 else (k, o), "seeds")


@pytest.mark.parametrize("item", ["pixel", "marguerite"])
def test_primary_item_clause_fails_alone(item):
    _fails_alone(lambda a, s, i, q, k, o: ("uv", -0.5) if a == "ORACLE" and i == item else (k, o), "items")


def test_volvo_is_report_only():
    t = _verdict(mod=lambda a, s, i, q, k, o: ("uv", -0.5) if a == "ORACLE" and i == "volvo" else (k, o))
    assert t["verdict"] == "CLASS-ALIVE" and not t["clauses"]["per_item"]["volvo"]["holds"]


@pytest.mark.parametrize("n_uv,passes", [(18, True), (19, False)])
def test_p_uv_bar_is_inclusive_at_minus_point_one(n_uv, passes):
    """nq=5: 20 cells a seed, 60 knows pooled, H p_uv 1.0. ORACLE with 54 uv of 60 is exactly H - 0.10 (passes), 55 is not."""
    names = list(ITEMS)

    def mod(a, s, i, q, k, o):
        return ("uv", o) if a == "ORACLE" and names.index(i) * 5 + q < n_uv else (k, o)
    t = VC.table(_outs(mod=mod, nq=5), {s: LED for s in (319, 320, 321)})
    assert t["clauses"]["ORACLE"]["p_uv"] == pytest.approx(3 * n_uv / 60) and t["clauses"]["ok"]["voice"] is passes


def test_voice_control_sham_fails_alone():
    _fails_alone(lambda a, s, i, q, k, o: ("your", o) if a == "SHAM" else (k, o), "voice_sham")


def test_unevaluable_item_is_not_a_failure():
    """pixel has no knows rows under ORACLE and H: direction is not evaluable, so it is not counted as a failure."""
    def mod(a, s, i, q, k, o):
        return ("none", o) if i == "pixel" and a in ("H", "ORACLE") else (k, o)
    t = _verdict(mod=mod)
    assert t["clauses"]["per_item"]["pixel"]["holds"] is None and t["clauses"]["items_evaluable"] == ["marguerite"]
    assert t["clauses"]["ok"]["items"] and t["clauses"]["item_rows"]["pixel"]["knows_H"] == 0


@pytest.mark.parametrize("kw", [{"repro": 0.5}, {"led_keys": False}, {"seeds": (319, 320)}, {"repro_n": 2},
                                {"onset_repro": 0.5}, {"label_tagged": False},
                                {"mask": {"calls": 1, "tokens": 90, "tagged": 0, "registered_hits": 0}},
                                {"mask": {"calls": 1, "tokens": 90, "tagged": 90, "registered_hits": 0}}])
def test_undefined_cases(kw):
    t = _verdict(**kw)
    assert t["verdict"] == "UNDEFINED", t


def test_undefined_noop_arm_and_inconsistent_outputs():
    t = _verdict(mod=lambda a, s, i, q, k, o: (k, -1.0) if a == "ORACLE" else (k, o))     # ORACLE onset == H onset everywhere
    assert t["verdict"] == "UNDEFINED" and (319, "ORACLE") in t["noop_arms"]
    outs = _outs()
    outs[1]["commit"] = "zzz"
    assert VC.table(outs, {s: LED for s in (319, 320, 321)})["verdict"] == "UNDEFINED"
    outs = _outs()
    outs[1]["night"] = 13
    assert VC.table(outs, {s: LED for s in (319, 320, 321)})["verdict"] == "UNDEFINED"
    outs = _outs()
    outs[1] = copy.deepcopy(outs[0])                                  # the same seed twice must not clear "missing"
    t = VC.table(outs, {s: LED for s in (319, 320, 321)})
    assert t["verdict"] == "UNDEFINED" and "duplicate" in t["why"]


def test_undefined_without_a_reference_and_with_a_missing_battery():
    outs = _outs()
    for o in outs:
        o.pop("reproduction")
    assert VC.table(outs, {s: LED for s in (319, 320, 321)})["verdict"] == "UNDEFINED"
    outs = _outs()
    del outs[0]["batteries"][f"WRONG@{L}"]
    assert VC.table(outs, {s: LED for s in (319, 320, 321)})["verdict"] == "UNDEFINED"


def test_table_report_layers_and_handread():
    outs = _outs()
    for o in outs:
        o["report_layers"] = [6]
        for a in ("ORACLE", "SHAM"):
            o["batteries"][f"{a}@6"] = copy.deepcopy(o["batteries"][f"{a}@{L}"])
    t = VC.table(outs, {s: LED for s in (319, 320, 321)})
    assert "6" in t["report_layers"] and t["verdict"] == "CLASS-ALIVE"
    blind, key = VC.handread(outs, {s: LED for s in (319, 320, 321)}, n=40)
    assert len(blind) == len(key) == 40 and set(blind[0]) == {"id", "question", "reply"}
    assert {k["arm"] for k in key} <= {"H", "C", f"ORACLE@{L}"}


# ------------------------------------------------------------------ the run on a stub life: units, resume, atomic
def test_run_on_a_stub_life_resumes_per_unit(tmp_path, monkeypatch):
    res, snaps, trunk = _life(tmp_path)
    night = 2
    qs = VT.told_questions(res, night)
    if len({q["item"] for q in qs}) < 3:
        pytest.skip("the stub life tells fewer than 3 items")
    monkeypatch.setattr(VT, "_onset_ids", lambda pipe: ([1, 3], 2))
    monkeypatch.setattr(VC, "REQUIRE_ACTIVE", False)

    def fake(b, text, state):
        g = torch.Generator().manual_seed(sum(map(ord, text)) % 1000)
        return torch.randn(2, 8, generator=g) + 0.01 * float(sum(float(x.sum()) for x in state))
    attached = []
    monkeypatch.setattr(VC, "capture_label", fake)
    monkeypatch.setattr(VC, "set_vector", lambda b, v, layer, prov: attached.append((layer, float(v.norm()), prov.kind)))
    monkeypatch.setattr(VC, "clear_vector", lambda b: None)
    monkeypatch.setattr(VC, "_label_tagged", lambda b, kind, text: True)
    calls = []
    real = VC.battery
    monkeypatch.setattr(VC, "battery", lambda *a, **k: (calls.append(a[6]), real(*a, **k))[1])
    out, work = str(tmp_path / "o.json"), str(tmp_path / "w")
    R = VC.run(res, snaps, trunk, StubCortex(), work, out_path=out, night=night, log=lambda *a: None)
    assert R["layer"] == 1 and R["report_layers"] == [] and R["skipped_layers"] == [6, 18]
    assert sorted(calls) == sorted(["H", "C"] + [f"{a}@1" for a in VC.VEC_ARMS])
    assert set(R["batteries"]) == set(calls)
    for n, rec in R["batteries"].items():
        assert len(rec["rows"]) == len(qs) and all(r["onset"] is not None for r in rec["rows"])
    assert {k for _, _, k in attached} == {"own", "all"} and all(a[0] == 1 for a in attached)
    assert len(attached) == len(VC.VEC_ARMS) * len(qs)
    assert R["batteries"]["C"]["rows"][0]["told_first"] and not R["batteries"]["H"]["rows"][0]["told_first"]
    assert R["host"] and R["commit"] and R["device"]
    assert all(r["label_tagged"] is True and r["vec_norm"] > 0 for a in VC.VEC_ARMS for r in R["batteries"][f"{a}@1"]["rows"])
    assert all(r["label_tagged"] is None for r in R["batteries"]["H"]["rows"])
    saved = json.load(open(out))
    assert saved["spec"] == R["spec"] and not (tmp_path / "o.json.tmp").exists()
    n0 = len(calls)
    R2 = VC.run(res, snaps, trunk, StubCortex(), work, out_path=out, night=night, log=lambda *a: None)
    assert len(calls) == n0                                         # every unit resumed, none redone
    assert {k: v["rows"] for k, v in R2["batteries"].items()} == {k: v["rows"] for k, v in R["batteries"].items()}
    # the run records reproduction against a voice_tag-format reference of the same rows
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"batteries": {"A0snap": {"hoff": {"rows": R["batteries"]["H"]["rows"]}}}}))
    R3 = VC.run(res, snaps, trunk, StubCortex(), work, out_path=out, night=night, refs=[str(ref)], log=lambda *a: None)
    assert R3["reproduction"]["vt"]["frac"] == 1.0


# ------------------------------------------------------------------ the real tag path on a tiny cortex, ref formats, CLI
def test_real_tag_path_oracle_moves_the_onset_and_sham_has_the_same_norm():
    p = _plastic()
    pipe = p.cortex.pipe

    class _C:
        active = True
        def __init__(self):
            self.pipe = pipe
        def read(self, text, state=None):
            lg, st = p(pipe.encode(text), None)
            return (lg[-1] if lg.dim() > 1 else lg), st

    class _B:
        cortex = _C()
        _asst = "Assistant:"
        def _plastic(self):
            return p
    b, q = _B(), {"text": "what is my cat called?"}
    yid, mid = [1, 3], 2
    h = VC._onset(b, q, [], yid, mid)
    v = torch.randn(p.C) * 5
    sh = VC.sham_vector(319, 0, 0, v)
    assert VC._label_tagged(b, "own", f"User: {q['text']}\n\n{b._asst}")
    prov = SM.Provenance("own")
    VC.set_vector(b, v, 0, prov)
    o = VC._onset(b, q, [], yid, mid)
    assert prov.stats["tagged"] > 0 and o != pytest.approx(h, abs=1e-6)
    VC.clear_vector(b)
    assert VC._onset(b, q, [], yid, mid) == pytest.approx(h, abs=1e-6)       # cleared: back to H
    assert float(sh.norm()) == pytest.approx(float(v.norm()), rel=1e-5)
    VC.set_vector(b, sh, 0, SM.Provenance("own"))
    assert VC._onset(b, q, [], yid, mid) != pytest.approx(o, abs=1e-6)
    VC.clear_vector(b)


def test_load_ref_rows_reads_both_formats(tmp_path):
    rows = [{"you": "q", "brain": "a"}]
    vt, vs = tmp_path / "vt.json", tmp_path / "vs.json"
    vt.write_text(json.dumps({"batteries": {"A0snap": {"hoff": {"rows": rows}}}}))
    vs.write_text(json.dumps({"nights": [{"night": 14, "arms": {"hippocampus-off": {"B": {"rows": rows}}}}]}))
    assert VC.load_ref_rows(str(vt)) == ("vt", rows) and VC.load_ref_rows(str(vs), 14) == ("vs", rows)
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    with pytest.raises(ValueError):
        VC.load_ref_rows(str(bad))


def test_table_and_handread_cli(tmp_path, capsys):
    outs = _outs()
    paths, lives = [], []
    for o in outs:
        pth = tmp_path / f"vc_{o['seed']}.json"
        pth.write_text(json.dumps(o))
        paths.append(str(pth))
        lv = tmp_path / f"life_{o['seed']}.json"
        lv.write_text(json.dumps({"seed": o["seed"], "ledger": LED}))
        lives.append(str(lv))
    out = tmp_path / "verdict.json"
    VC.main(["table", *paths, "--lives", *lives, "--out", str(out)])
    assert json.load(open(out))["verdict"] == "CLASS-ALIVE"
    VC.main(["handread", *paths, "--lives", *lives, "--out-prefix", str(tmp_path / "hr"), "--n", "10"])
    blind, key = json.load(open(tmp_path / "hr_blind.json")), json.load(open(tmp_path / "hr_key.json"))
    assert len(blind) == len(key) == 10 and "arm" not in blind[0]
