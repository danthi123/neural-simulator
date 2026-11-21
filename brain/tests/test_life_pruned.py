# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md
# Expected-fail: none (mechanics of the L-prunedA savings branch; savings itself is expected to fail at 1.5B)
"""--pruned-a on the stub cortex: a 5-day mini life with the branch cut at L3 (A told L2 + L3, B told L3)."""
import json
import os
import re
import shutil
import subprocess
import importlib.util
import sys
from types import SimpleNamespace

import pytest
import torch

import brain.brain as bb
import brain.eval.life as life
from brain import clock
from brain.brain import Brain
from brain.eval import life_bank as LB
from brain.eval import life_table
from brain.hippocampus import Episode
from brain.tests.test_brain_key_switch import KW, StubCortex
from brain.tests.test_life import _hash, _scrub

BKW = {k: v for k, v in KW.items() if k != "sleep_on_restart"}
BD = 3
REAL_BUILD_LIFE = life.build_life     # captured before the autouse fixture swaps in mini_life
BASE = "28a3c4c3b"                    # the pre-Stage-1 life.py: flags-off outputs must equal it (HEAD would be vacuous once committed)
A_IDS, B_IDS = ["ab01", "ab02"], ["ab03", "ab04"]


def mini_life(seed, days, split="dev"):
    """L1 greet + H; L2 tells A (ab01, ab02) + same-day ab01; L3 re-tells A (A') + B (ab03, ab04) + same-day ab01;
    L4 a filler; L5 small talk. Ledger rows exactly as build_life.told writes them."""
    life.use_split(split)
    ledger = {}

    def told(it, d):
        ledger.setdefault(it.id, {"cls": it.cls, "told_days": [], "keys": list(it.keys), "telling": it.telling,
                                  "variant": None, "dup_of": it.dup_of})["told_days"].append(d)
    plan = {1: ["pixel"], 2: ["ab01", "ab02"], 3: ["ab01", "ab02", "ab03", "ab04"], 4: ["f01"], 5: []}
    sched = []
    for d in range(1, days + 1):
        turns = [{"kind": "greet", "text": "Good morning! How are you?", "item": None}]
        for i in plan.get(d, []):
            turns.append({"kind": "tell", "text": LB.ALL[i].telling, "item": i})
            told(LB.ALL[i], d)
        if d in (2, 3):
            turns.append({"kind": "sameday", "text": LB.ALL["ab01"].direct, "item": "ab01"})
        if d == 5:
            turns.append({"kind": "small", "text": LB.SMALL_TALK[0], "item": None})
        sched.append({"day": d, "turns": turns})
    return sched, ledger


def fake_lp(b, question, option, fetch=True):
    """Deterministic stand-in for option_lp (the stub has no .model): a length term plus 1.0 when an episode in the
    copy's store holds the option, so a pruned A item reads lower than an unpruned one."""
    held = any(option.lower() in e.text.lower() for e in b.hippocampus.episodes)
    return -0.1 * len(option) + (1.0 if held else 0.0)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(bb, "replay_into_cortex", lambda plastic, base, eps, generic, **k: [])
    monkeypatch.setattr(Brain, "_plastic", lambda self: type("P", (), {"parameters": lambda s: [], "B_k": {}, "B_v": {}})())
    monkeypatch.setattr(life, "build_life", mini_life)
    monkeypatch.setattr(life, "option_lp", fake_lp)
    yield
    clock.set(None)
    life.use_split("dev")


def _run(tmp_path, name, days=5, **kw):
    kw.setdefault("span", True)
    return life.run(StubCortex(), 7, days, "dev", str(tmp_path / name / "h"), str(tmp_path / name / "o" / "life_s7"),
                    kw.pop("full", {1}), brain_kw=BKW, log=lambda *_: None, **kw)


def _ep(text, **kw):
    return Episode(text=text, delta=[], key=torch.zeros(1), created=0.0, last_used=0.0, **kw)


def test_ab_sets_questions_and_sameday():
    sched, ledger = mini_life(7, 5)
    assert life.ab_sets(ledger, BD) == (A_IDS, B_IDS)
    qs = life.ab_questions(ledger, BD)
    assert [(q["item"], q["set"]) for q in qs] == [("ab01", "A"), ("ab02", "A"), ("ab03", "B"), ("ab04", "B")]
    assert all(q["kind"] == "ab" for q in qs)
    assert life.sameday_ab(sched, ledger, BD) == ["ab01"]
    # the real 15-day life: 8 + 8, and the same-day item is A[0] (asked on L2 and on L11)
    s, led = REAL_BUILD_LIFE(42, 15)
    a, b = life.ab_sets(led, 11)
    assert len(a) == 8 and len(b) == 8 and not set(a) & set(b)
    sd = life.sameday_ab(s, led, 11)
    assert len(sd) == 1 and sd[0] in a


def test_prune_rule_keeps_echoes_self_and_permanent_and_remaps_belief():
    tel = LB.ALL["ab01"].telling
    eps = [_ep("User: Good morning!"),
           _ep(f"User: {tel}\n\nAssistant: Nice."),                          # the A episode: deleted
           _ep(f"User: I like tea.\n\nAssistant: You said: {tel}"),          # an echo in a reply: kept
           _ep(f"User: {tel}", source="self"),                               # the brain's own words: kept
           _ep(f"User: {tel}", permanent=True),                              # permanent: kept
           _ep(f"User: {LB.ALL['ab02'].telling}"),                           # the other A episode (no reply): deleted
           _ep("User: The lesson.")]                                         # the self-belief episode
    meta = {"self_belief_unknown": 6}
    b = SimpleNamespace(hippocampus=SimpleNamespace(episodes=list(eps)), meta=meta,
                        self_belief=lambda: eps[meta["self_belief_unknown"]])
    _, ledger = mini_life(7, 5)
    rep = life.prune_episodes(b, ledger, A_IDS)
    assert rep["per_item"] == {"ab01": 1, "ab02": 1} and not rep["mismatch"]
    assert rep["pruned"] == 2 and rep["store_before"] == 7 and rep["store_after"] == 5
    assert rep["kept_mentions"] == ["ab01"]                    # the diagnostic sees the echo, the rule did not delete it
    assert b.hippocampus.episodes == [eps[0], eps[2], eps[3], eps[4], eps[6]]
    assert meta["self_belief_unknown"] == 4 and b.hippocampus.episodes[4] is eps[6]
    # a doubled telling is a mismatch (logged in a real run)
    b2 = SimpleNamespace(hippocampus=SimpleNamespace(episodes=[_ep(f"User: {tel}"), _ep(f"User: {tel}")]), meta={},
                         self_belief=lambda: None)
    assert life.prune_episodes(b2, ledger, ["ab01"])["mismatch"]


def test_savings_read_exact():
    span = [{"item": "ab01", "logodds": 1.0, "span_nll": 2.0}, {"item": "ab02", "logodds": 3.0, "span_nll": 4.0},
            {"item": "ab03", "logodds": 0.5, "span_nll": 5.0}, {"item": "ab04", "logodds": -0.5, "span_nll": 7.0}]
    rows = [{"kind": "ab", "item": "ab01", "ok": True}, {"kind": "ab", "item": "ab02", "ok": True},
            {"kind": "ab", "item": "ab03", "ok": False}, {"kind": "ab", "item": "ab04", "ok": True},
            {"kind": "H", "item": "pixel", "ok": True}]
    s = life.savings_read(span, rows, A_IDS, B_IDS, exclude=["ab01"])
    assert s["G"] == pytest.approx(2.0) and s["dR"] == 1 and (s["recall_A"], s["recall_B"]) == (2, 1)
    assert s["nll_A"] == pytest.approx(3.0) and s["nll_B"] == pytest.approx(6.0)
    assert s["logodds"] == {"A": {"ab01": 1.0, "ab02": 3.0}, "B": {"ab03": 0.5, "ab04": -0.5}}
    assert s["recall"]["B"] == {"ab03": False, "ab04": True}
    assert s["excluded"] == ["ab01"] and s["G_excl"] == pytest.approx(3.0) and s["dR_excl"] == 0
    e = life.savings_read([], [], A_IDS, [], exclude=())
    assert e["G"] is None and e["dR"] is None


def test_pruned_life(tmp_path):
    res = _run(tmp_path, "p", pruned_a=True, pruned_a_day=BD)
    pa = res["pruned_a"]
    assert pa["day"] == BD and pa["A"] == A_IDS and pa["B"] == B_IDS and pa["sameday_items"] == ["ab01"]
    assert pa["report"]["per_item"] == {"ab01": 1, "ab02": 1} and not pa["report"]["mismatch"]
    assert pa["report"]["store_after"] == pa["report"]["store_before"] - 2
    # the branch lived L3-L5 and slept each night with the trunk's night count
    lived = sorted(x["day"] for x in res["lived"] if x["line"] == "prunedA")
    assert lived == [3, 4, 5]
    homes = tmp_path / "p" / "h"
    with open(homes / "trunk" / "meta.json") as fh:
        tm = json.load(fh)
    with open(homes / "prunedA" / "meta.json") as fh:
        pm = json.load(fh)
    assert pm["sleeps"] == tm["sleeps"] == 5 and pm["last_sleep"] == pytest.approx(tm["last_sleep"])
    with open(homes / "prunedA" / "episodes.json") as fh:
        idx = json.load(fh)
    assert all(r["created"] <= pm["last_sleep"] for r in idx)
    for i in A_IDS + B_IDS:      # after L3 each A' and B' has exactly one episode in the branch
        assert sum(1 for r in idx if life._told_head(r["text"]) == f"User: {LB.ALL[i].telling}") == 1
    assert len(os.listdir(homes / "prunedA" / "episodes")) == len(idx)       # no orphan .pt
    assert not os.path.exists(str(homes / "prunedA") + ".build")
    # probes: branch hippocampus-off every night, both-off only at night BD, AB-only span and AB rows
    br = [p for p in res["probes"] if p["line"] == "prunedA"]
    assert sorted((p["night"], p["arm"]) for p in br) == [(3, "both-off"), (3, "hippocampus-off"), (4, "hippocampus-off"),
                                                          (5, "hippocampus-off")]
    for p in br:
        assert {r["cls"] for r in p["span"]} == {"AB"} and len(p["span"]) == 4
        assert [r["kind"] for r in p["rows"]] == ["ab"] * 4 and [r["set"] for r in p["rows"]] == list("AABB")
        assert p["arm"] != "intact"
    # the trunk's tagged unpruned AB recall on nights BD..BD+2, and its normal arms untouched
    tg = sorted((p["night"], p["arm"]) for p in res["probes"] if p.get("tag") == life.AB_TAG)
    assert tg == [(3, "hippocampus-off"), (4, "hippocampus-off"), (5, "hippocampus-off")]
    assert all(p["line"] == "trunk" for p in res["probes"] if p.get("tag"))
    # savings: the fake lp gives the pruned A items no episode bonus before L3 is slept... after L3 both have one
    sv = pa["savings"]
    assert set(sv) == {"3", "4", "5"}
    for n in sv:
        assert sv[n]["prunedA"]["both-off"]["from_night"] == BD and sv[n]["trunk"]["hippocampus-off"]["n_A"] == 2
        assert sv[n]["prunedA"]["hippocampus-off"]["G"] is not None
        assert set(sv[n]["prunedA"]["hippocampus-off"]["logodds"]["A"]) == set(A_IDS)
    md = open(str(tmp_path / "p" / "o" / "life_s7.md")).read()
    assert "## L-prunedA (trunk copy at L3 09:00" in md and "## Savings (L-prunedA)" in md
    assert "valid only if P0 (forgetting) is VALID" in md and "AB recall A" in md
    text = life_table.render([res])     # the table ignores the tagged probes and the branch line
    assert "trunk" in text


def test_branch_built_from_trunk_copy_and_trunk_untouched(tmp_path):
    res = _run(tmp_path, "t", days=2)     # a trunk slept through night 2 = the home at L3 09:00
    homes = tmp_path / "t" / "h"
    trunk = str(homes / "trunk")
    before = _hash(trunk)
    sim = life.SimClock()
    clock.set(sim.now)
    sched, ledger = mini_life(7, 5)
    dst = str(tmp_path / "t" / "branch")
    # ab01/ab02 are told on L2 only in the trunk: exactly one episode each
    rep = life.make_pruned_branch(trunk, dst, ledger, A_IDS, StubCortex(), {"sleep_seed": 7, **BKW}, sim, 7, BD)
    assert _hash(trunk) == before
    assert rep["per_item"] == {"ab01": 1, "ab02": 1} and not rep["mismatch"]
    with open(os.path.join(dst, "episodes.json")) as fh:
        idx = json.load(fh)
    assert not any(life._told_head(r["text"]) == f"User: {LB.ALL[i].telling}" for r in idx for i in A_IDS)
    with open(os.path.join(dst, "meta.json")) as fh, open(os.path.join(trunk, "meta.json")) as fh2:
        m, mt = json.load(fh), json.load(fh2)
    assert m["sleeps"] == mt["sleeps"] and m["last_sleep"] == mt["last_sleep"]
    assert all(r["created"] < m["last_sleep"] for r in idx)


def test_span_classes_keep_the_foils(tmp_path):
    _run(tmp_path, "s", days=4)
    trunk = str(tmp_path / "s" / "h" / "trunk")
    cp = str(tmp_path / "s" / "cp")
    shutil.copytree(trunk, cp)
    sim = life.SimClock()
    clock.set(sim.now)
    _, ledger = mini_life(7, 5)
    b = Brain(cp, cortex=StubCortex(), lock=True, **{**BKW, "sleep_seed": 7, "sleep_on_restart": False})
    try:
        full = life.span_read(b, ledger, 4, 7)
        ab = life.span_read(b, ledger, 4, 7, classes=("AB",))
    finally:
        b.release()
    assert ab == [r for r in full if r["cls"] == "AB"] and len(ab) == 4 and len(full) == 5


def test_flag_checks(tmp_path):
    with pytest.raises(ValueError):
        _run(tmp_path, "x", days=4, pruned_a=True, pruned_a_day=BD)       # days < bd + 2
    with pytest.raises(ValueError):
        _run(tmp_path, "y", span=False, pruned_a=True, pruned_a_day=BD)    # needs --span
    _run(tmp_path, "z", days=2)
    with pytest.raises(ValueError):                                         # resume of a life started without it
        _run(tmp_path, "z", resume=True, pruned_a=True, pruned_a_day=BD)
    _run(tmp_path, "q", days=5, pruned_a=True, pruned_a_day=BD)
    with pytest.raises(ValueError):                                         # resume with another day
        _run(tmp_path, "q", resume=True, pruned_a=True, pruned_a_day=2)


def _replies(res):
    lived = sorted((x["line"], x["day"], r["you"], r["brain"], str(r["recalled"])) for x in res["lived"] for r in x["rows"])
    probes = sorted((p["line"], p["night"], p["arm"], str(p.get("tag")), r["brain"]) for p in res["probes"] for r in p["rows"])
    span = sorted((p["line"], p["night"], p["arm"], str(p.get("tag")), json.dumps(p.get("span"))) for p in res["probes"])
    return lived, probes, span


def test_resume_after_cuts(tmp_path, monkeypatch):
    whole = _run(tmp_path, "w", pruned_a=True, pruned_a_day=BD)
    # a cut in a branch probe at night 4, then a resume that restores the flag config from res
    real = life.probe
    cut = {"on": True}

    def cutting(home, *a, **k):
        if cut["on"] and home.endswith("prunedA") and a[1] == 4:
            cut["on"] = False
            raise RuntimeError("cut")
        return real(home, *a, **k)
    monkeypatch.setattr(life, "probe", cutting)
    with pytest.raises(RuntimeError):
        _run(tmp_path, "c", pruned_a=True, pruned_a_day=BD)
    res = _run(tmp_path, "c", resume=True)          # flags restored from res
    assert res["pruned_a"]["report"] == whole["pruned_a"]["report"]
    assert _replies(res) == _replies(whole)
    assert res["pruned_a"]["savings"] == whole["pruned_a"]["savings"]
    # a cut inside the branch build: the trunk stays idle at bd-1 and the resume rebuilds the branch
    real_prune = life.prune_episodes
    cut2 = {"on": True}

    def cut_prune(*a, **k):
        if cut2["on"]:
            cut2["on"] = False
            raise RuntimeError("cut")
        return real_prune(*a, **k)
    monkeypatch.setattr(life, "prune_episodes", cut_prune)
    with pytest.raises(RuntimeError):
        _run(tmp_path, "b", pruned_a=True, pruned_a_day=BD)
    with open(tmp_path / "b" / "h" / "life_state.json") as fh:
        st = json.load(fh)
    assert "prunedA" not in st["arms"] and st["arms"]["trunk"]["done"] == BD - 1 and st["arms"]["trunk"]["phase"] == "idle"
    res2 = _run(tmp_path, "b", resume=True, pruned_a=True, pruned_a_day=BD)
    assert os.path.isdir(tmp_path / "b" / "h" / "prunedA") and not os.path.exists(str(tmp_path / "b" / "h" / "prunedA") + ".build")
    assert _replies(res2) == _replies(whole) and res2["pruned_a"]["report"] == whole["pruned_a"]["report"]


def _md_scrub(text):
    return re.sub(r"[\d.]+ s(?![A-Za-z])", "S", re.sub(r"host: [^;]+;", "", text))


def test_flags_off_equal_head_code(tmp_path, monkeypatch):
    """Flags off: the JSON and md equal those of the pre-Stage-1 life.py (commit BASE), not just of this code."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(life.__file__))))
    try:
        src = subprocess.run(["git", "-C", root, "show", f"{BASE}:brain/eval/life.py"], capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("base commit not available")
    path = tmp_path / "life_head.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("life_head", path)
    head = importlib.util.module_from_spec(spec)
    sys.modules["life_head"] = head
    try:
        spec.loader.exec_module(head)
        monkeypatch.setattr(head, "build_life", mini_life)
        monkeypatch.setattr(head, "option_lp", fake_lp)
        outs = {}
        for name, mod in (("new", life), ("head", head)):
            res = mod.run(StubCortex(), 7, 4, "dev", str(tmp_path / name / "h"), str(tmp_path / name / "o" / "l"), {1},
                          nosleep_arm=True, brain_kw=BKW, log=lambda *_: None, span=True)
            outs[name] = (json.dumps(_scrub(res), sort_keys=True),
                          _md_scrub(open(str(tmp_path / name / "o" / "l.md")).read()))
        assert outs["new"] == outs["head"]
        assert "pruned_a" not in json.loads(outs["new"][0]) and "prunedA" not in outs["new"][1]
    finally:
        sys.modules.pop("life_head", None)
