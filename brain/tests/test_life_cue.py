# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md
# Expected-fail: none (mechanics of the --cue-ladder read-out; cued recall itself is expected to fail at 1.5B under S0)
"""--cue-ladder on the stub cortex: the plan, the cue texts and their hash, the independent cue block on its own copy,
flag-on lives equal to flag-off lives outside the cue keys, resume, and the R_cue read-outs."""
import json
import os

import pytest

import brain.brain as bb
import brain.eval.life as life
from brain import clock
from brain.brain import Brain
from brain.eval import life_bank, life_bank_test, life_cues, life_ladder, life_table
from brain.tests.test_brain_key_switch import KW, StubCortex
from brain.tests.test_life import _hash, _scrub

BKW = {k: v for k, v in KW.items() if k != "sleep_on_restart"}
REAL_BUILD_LIFE = life.build_life
CUE_KEYS = ("cue_ladder", "cue_probes", "cue_summary", "todo")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(bb, "replay_into_cortex", lambda plastic, base, eps, generic, **k: [])
    monkeypatch.setattr(Brain, "_plastic", lambda self: type("P", (), {"parameters": lambda s: [], "B_k": {}, "B_v": {}})())

    def short(seed, days, split="dev"):
        sched, ledger = REAL_BUILD_LIFE(seed, days, split)
        for d in sched:
            d["turns"] = d["turns"][:3]
        return sched, ledger
    monkeypatch.setattr(life, "build_life", short)
    yield
    clock.set(None)
    life.use_split("dev")


def _run(tmp_path, name, days=3, **kw):
    return life.run(StubCortex(), 7, days, "dev", str(tmp_path / name / "h"), str(tmp_path / name / "o" / "life_s7"),
                    kw.pop("full", {1}), brain_kw=BKW, log=lambda *_: None, **kw)


def _plan(seed, days, split):
    sched, ledger = REAL_BUILD_LIFE(seed, days, split)
    return sched, ledger, life_ladder.ladder_plan(seed, days, sched, ledger, split)


@pytest.mark.parametrize("split,seeds,days", [("dev", (310, 311, 312), 7), ("dev", (310, 7), 15), ("test", (42, 101), 15)])
def test_plan_properties(split, seeds, days):
    for seed in seeds:
        sched, ledger, p = _plan(seed, days, split)
        assert p == _plan(seed, days, split)[2]                                 # deterministic
        items = [c["item"] for c in p["cells"]]
        assert len(items) == len(set(items))                                     # each item in exactly one cell
        late = life_ladder.live_asked_late(sched, ledger)
        eligible = {i for i, e in ledger.items() if e["cls"] in ("E", "F") and e["told_days"] and min(e["told_days"]) >= 2}
        assert set(items) | {u["item"] for u in p["unplaced"]} == eligible     # every eligible item is used or explained
        for c in p["cells"]:
            assert c["delay"] in life_ladder.DELAYS and c["told_day"] == c["night"] - c["delay"] + 1 >= 2
            assert c["night"] <= days and c["item"] not in late
            rot = {q["item"] for q in life.battery(seed, c["night"], ledger) if q["kind"] == "rotating"}
            assert c["item"] not in rot                                          # never the battery's rotating item
            assert c["rungs"] == list(life_ladder.RUNGS[c["track"]])
        if days == 7:
            assert all(c["delay"] in (1, 3) for c in p["cells"])
        assert sum(p["counts"].values()) == len(items)
    if split == "dev" and days == 15:
        assert p["counts"]["F+14"] >= 1 and p["counts"]["E+7"] >= 1
    life.use_split("dev")


def test_cue_texts_have_no_keys_and_matched_overlap():
    for split, bank in (("dev", life_bank), ("test", life_bank_test)):
        cues = life_cues.CUES[split]
        assert set(cues) == {x.id for x in bank.E + bank.F}
        ov = life_cues.overlaps(split, bank)
        allkeys = [k for x in bank.ALL.values() for k in x.keys]
        for p in life_cues.C4_PROMPTS:
            assert not life._has(p, allkeys) and p.endswith("?")
        for iid, (c2, c3) in cues.items():
            it = bank.ALL[iid]
            for t in (c2, c3):
                assert t.endswith(life_cues.TAIL) and not life._has(t, it.keys), (iid, t)
            assert abs(ov[iid][0] - ov[iid][1]) <= 1 and ov[iid][0] >= 1, (iid, ov[iid])
            assert ov[iid][2] == 0
            assert content_diff(c2, c3)                                         # a different situation, not a copy
        for x in bank.E:
            assert x.gist_q and x.gist_keys


def content_diff(a, b):
    return life_cues.content(a) != life_cues.content(b)


def test_test_cues_hash_checked(tmp_path, monkeypatch):
    assert life_ladder.load_cues("test") is life_cues.TEST
    f = tmp_path / "c.sha256"
    f.write_text("0" * 64 + "  life_cues.py\n")
    monkeypatch.setattr(life_ladder, "CUES_HASH_FILE", str(f))
    with pytest.raises(RuntimeError):
        life_ladder.load_cues("test")
    assert life_ladder.load_cues("dev") is life_cues.DEV                         # DEV is not frozen


def test_flag_on_life_equals_flag_off_outside_the_cue_block(tmp_path):
    off = _run(tmp_path, "off", 3)
    on = _run(tmp_path, "on", 3, cue_ladder=True)
    strip = lambda r: json.dumps(_scrub({k: v for k, v in r.items() if k not in CUE_KEYS}), sort_keys=True)  # noqa: E731
    assert strip(on) == strip(off)                       # same lived life, same battery replies, same probes
    assert not any(k in off for k in ("cue_ladder", "cue_probes", "cue_summary"))
    assert "gist slots (E gist_q)" in off["todo"] and not any("gist" in t for t in on["todo"])
    cells = on["cue_ladder"]["plan"]["cells"]
    assert cells
    got = sorted((p["night"], p["arm"]) for p in on["cue_probes"])
    want = sorted((n, a) for n in {c["night"] for c in cells}
                  for a in life.LIGHT_ARMS + (("both-off",) if n in {1} else ()))
    assert got == want
    for p in on["cue_probes"]:
        mine = [c for c in cells if c["night"] == p["night"]]
        assert [(r["item"], r["rung"]) for r in p["rows"]] == [(c["item"], g) for c in mine for g in c["rungs"]]
        if p["arm"] != "intact":
            assert all(r["recalled"] is None and not r["right_episode"] for r in p["rows"])
        assert all(isinstance(r["brain"], str) for r in p["rows"])
    md = open(str(tmp_path / "on" / "o" / "life_s7.md")).read()
    assert "## Cue ladder" in md and "PROVISIONAL" in md and "pooled item-level permutation" in md
    assert "cue block, morning after night" in md
    assert on["cue_summary"]["cells"] and all("R_cue" in c for c in on["cue_summary"]["cells"])
    text = life_table.render([on])
    assert "Cue ladder, pooled over seeds" in text


def test_cue_block_independent_own_copy_and_trunk_untouched(tmp_path):
    res = _run(tmp_path, "b", 3)
    home = str(tmp_path / "b" / "h" / "trunk")
    before = _hash(home)
    ledger = res["ledger"]
    sched, _, plan = _plan(7, 3, "dev")
    cells = [c for c in plan["cells"] if c["night"] == 3]
    assert len(cells) >= 2
    sim = life.SimClock()
    clock.set(sim.now)
    cues = life_ladder.load_cues("dev")
    work = str(tmp_path / "b" / "w")
    a = life_ladder.cue_probe(home, StubCortex(), 3, ("intact", "hippocampus-off"), cells, ledger, sim, 7, BKW, work, cues)
    b = life_ladder.cue_probe(home, StubCortex(), 3, ("intact",), cells[::-1], ledger, sim, 7, BKW, work, cues)
    assert _hash(home) == before and not os.listdir(work)                      # copies deleted, trunk byte-identical
    key = lambda rows: {(r["item"], r["rung"]): (r["brain"], r["recalled"], r["ok"]) for r in rows}  # noqa: E731
    assert key(a[0]["rows"]) == key(b[0]["rows"])         # order does not matter: every turn starts fresh and restored


def test_cue_turn_restores_strengths(tmp_path):
    _run(tmp_path, "s", 2)
    home = str(tmp_path / "s" / "h" / "trunk")
    sim = life.SimClock(life.day_start(3))
    clock.set(sim.now)
    b = Brain(home, cortex=StubCortex(), lock=True, **{**BKW, "sleep_seed": 7, "sleep_on_restart": False})
    try:
        eps = b.hippocampus.episodes
        assert eps
        before = [(e.strength, e.uses, e.last_used) for e in eps]
        it = life.LB.ALL["pixel"]
        cell = {"track": "F", "delay": 1, "night": 2}
        b.hippocampus.FAMILIARITY = -1.0    # force a fetch so the turn would boost an episode
        row = life_ladder.cue_turn(b, it, "C0", it.direct, cell, sim, life.day_start(3), 7)
        assert row["fetched"] and row["use_restored"]
        assert [(e.strength, e.uses, e.last_used) for e in b.hippocampus.episodes] == before
        assert b.hippocampus.episodes == eps and len(b._recent_replies) == 0
    finally:
        b.release()


def test_resume_and_flag_checks(tmp_path, monkeypatch):
    whole = _run(tmp_path, "w", 3, cue_ladder=True)
    real = life_ladder.cue_probe
    cut = {"on": True}

    def cutting(home, cortex, night, *a, **k):
        if cut["on"] and night == 3:
            cut["on"] = False
            raise RuntimeError("cut")
        return real(home, cortex, night, *a, **k)
    monkeypatch.setattr(life_ladder, "cue_probe", cutting)
    with pytest.raises(RuntimeError):
        _run(tmp_path, "c", 3, cue_ladder=True)
    res = _run(tmp_path, "c", 3, resume=True)                                   # the flag is restored from res
    rows = lambda r: sorted((p["night"], p["arm"], json.dumps(p["rows"])) for p in r["cue_probes"])  # noqa: E731
    assert rows(res) == rows(whole) and res["cue_ladder"] == whole["cue_ladder"]
    strip = lambda r: json.dumps(_scrub(r), sort_keys=True)  # noqa: E731
    assert strip(res) == strip(whole)
    with pytest.raises(ValueError):                                             # another horizon = another plan
        _run(tmp_path, "c", 4, resume=True)
    _run(tmp_path, "z", 2)
    with pytest.raises(ValueError):                                             # a life started without the flag
        _run(tmp_path, "z", 2, resume=True, cue_ladder=True)


def test_cli_flag(tmp_path):
    res = life.main(["--seed", "7", "--days", "2", "--full-probe-nights", "1", "--homes", str(tmp_path / "h"),
                     "--out", str(tmp_path / "o" / "life_s7"), "--cue-ladder"], cortex=StubCortex(), brain_kw=BKW)
    assert "cue_ladder" in res
    res2 = life.main(["--seed", "7", "--days", "2", "--full-probe-nights", "1", "--homes", str(tmp_path / "h2"),
                      "--out", str(tmp_path / "o2" / "life_s7")], cortex=StubCortex(), brain_kw=BKW)
    assert "cue_ladder" not in res2


def _row(item, rung, ok, track="E", delay=1, right=False):
    return {"item": item, "track": track, "delay": delay, "rung": rung, "ok": ok, "right_episode": right,
            "wrong_fetch": False, "use_restored": False, "gist_hit": False}


def test_r_cue_summary_and_pooled():
    def life_(seed, hits):
        rows = [_row(i, g, ok) for i, d in hits.items() for g, ok in d.items()]
        return {"seed": seed, "cue_probes": [{"night": 2, "arm": "intact", "rows": rows}]}
    a = life_(1, {"e1": {"C2": True, "C3": False, "C4": False}, "e2": {"C2": True, "C3": True, "C4": False}})
    s = life_ladder.cue_summary(a)["cells"][0]
    assert s["p"] == {"C2": 1.0, "C3": 0.5, "C4": 0.0} and s["R_cue"] == 0.5 and s["headline"]
    assert life_ladder.r_cue({"C2": 1.0, "C3": 0.0}) is None                     # a missing rung gives no R_cue
    b = life_(2, {"e1": {"C2": False, "C3": False, "C4": True}})
    pr = life_ladder.pooled([a, b], n_perm=200)[0]
    assert pr["n_items"] == 3 and pr["R_cue"] == pytest.approx(2 / 3 - 1 / 3, abs=1e-4)
    assert pr["per_seed"] == {1: 0.5, 2: -1.0} and pr["seeds_positive"] == 1 and pr["n_seeds"] == 2
    assert 0 < pr["p_perm"] <= 1
    strong = life_(3, {f"e{i}": {"C2": True, "C3": False, "C4": False} for i in range(12)})
    assert life_ladder.pooled([strong], n_perm=500)[0]["p_perm"] < 0.01


def test_plan_uses_the_split_bank_even_if_the_imported_life_module_is_on_dev():
    """Under `python -m brain.eval.life` the running life is __main__ and life_ladder's `life` is another module object
    whose LB stays DEV: the plan must still read the TEST bank (overlaps, battery) for a TEST life."""
    sched, ledger = REAL_BUILD_LIFE(42, 15, "test")
    life.LB = life_bank                       # what the imported module holds when the life runs as __main__
    p = life_ladder.ladder_plan(42, 15, sched, ledger, "test")
    ov = life_cues.overlaps("test", life_bank_test)
    assert life.LB is life_bank_test
    assert all((c["ov_C2"], c["ov_C3"], c["ov_C4"]) == ov[c["item"]] for c in p["cells"])
    life.use_split("dev")
