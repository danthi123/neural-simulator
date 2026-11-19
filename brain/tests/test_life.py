# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md
"""brain.eval.life on the stub cortex: 2 days of 3 turns, a fake night (no model weights)."""
import hashlib
import json
import os

import pytest

import brain.brain as bb
import brain.eval.life as life
from brain import clock
from brain.brain import Brain
from brain.eval import life_bank as LB
from brain.eval import life_table
from brain.eval.weekly import SESSION_1
from brain.tests.test_brain_key_switch import KW, StubCortex

BKW = {k: v for k, v in KW.items() if k != "sleep_on_restart"}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    nights = []

    def fake_replay(plastic, base, eps, generic, **k):
        nights.append(len(eps))
        return []
    monkeypatch.setattr(bb, "replay_into_cortex", fake_replay)
    monkeypatch.setattr(Brain, "_plastic", lambda self: type("P", (), {"parameters": lambda s: [], "B_k": {}, "B_v": {}})())
    real = life.build_life

    def short(seed, days, split="dev"):
        sched, ledger = real(seed, days, split)
        for d in sched:
            d["turns"] = d["turns"][:3]     # 3 turns a day keeps the stub life fast
        return sched, ledger
    monkeypatch.setattr(life, "build_life", short)
    yield nights
    clock.set(None)


def _run(tmp_path, name, days, **kw):
    return life.run(StubCortex(), 7, days, "dev", str(tmp_path / name / "h"), str(tmp_path / name / "o" / "life_s7"),
                    kw.pop("full", {1}), brain_kw=BKW, log=lambda *_: None, **kw)


def _hash(root):
    out = {}
    for d, _, fs in sorted(os.walk(root)):
        for f in sorted(fs):
            if f != ".lock":
                with open(os.path.join(d, f), "rb") as fh:
                    out[os.path.relpath(os.path.join(d, f), root)] = hashlib.sha256(fh.read()).hexdigest()
    return out


def _replies(res, last_nosleep_night=2):
    """Lived and probed replies, ordered by (line, day, arm); a nosleep probe exists only at the horizon's last night."""
    lived = sorted((x["line"], x["day"], r["you"], r["brain"], str(r["recalled"])) for x in res["lived"] for r in x["rows"])
    probes = sorted((p["line"], p["night"], p["arm"], r["brain"], str(r["recalled"])) for p in res["probes"] for r in p["rows"]
                    if p["line"] == "trunk" or p["night"] == last_nosleep_night)
    return lived, probes


def test_day_one_is_weekly_session_one_and_build_is_deterministic():
    s1, l1 = life.build_life(310, 3)
    assert [t["text"] for t in s1[0]["turns"]] == SESSION_1[:3]
    stats = LB.split_stats()
    assert 75 <= stats["items"] <= 100


def test_full_build_deterministic_and_variants_opposite():
    import importlib
    m = importlib.reload(life)      # un-patched build_life
    try:
        s1, l1 = m.build_life(310, 7)
        s2, l2 = m.build_life(310, 7)
        assert s1 == s2 and l1 == l2
        assert [t["text"] for t in s1[0]["turns"]] == SESSION_1
        assert m.build_life(311, 7)[0] != s1
        inc = {k: v["variant"] for k, v in l1.items() if k.startswith("inc:")}
        sys_v = {v for k, v in inc.items() if "sysadmin" in k}
        home_v = {v for k, v in inc.items() if "home" in k}
        assert len(sys_v) == len(home_v) == 1 and sys_v != home_v
        assert m.build_life(310, 3)[0] == s1[:3]        # day d does not depend on the horizon
    finally:
        importlib.reload(life)


def test_two_days_night_counter_clock_and_probes(tmp_path, _env):
    res = _run(tmp_path, "a", 2)
    home = str(tmp_path / "a" / "h" / "trunk")
    with open(os.path.join(home, "meta.json")) as fh:
        meta = json.load(fh)
    assert meta["sleeps"] == 2 == len([n for n in res["nights"] if n["line"] == "trunk" and n["slept"]])
    assert len(_env) == 2                                    # the night replayed twice
    assert meta["last_sleep"] == pytest.approx(life.day_start(2) + 14 * 3600, abs=1)
    with open(os.path.join(home, "episodes.json")) as fh:
        idx = json.load(fh)
    assert all(r["created"] <= meta["last_sleep"] for r in idx)           # nothing pending after the night
    day2 = [r for r in idx if r["created"] >= life.day_start(2)]
    assert day2 and all(r["created"] < life.day_start(2) + 14 * 3600 for r in day2)
    arms = {(p["night"], p["arm"]) for p in res["probes"]}
    assert arms == {(1, a) for a in life.ARMS} | {(2, "intact"), (2, "hippocampus-off")}   # full nights {1}
    p1 = next(p for p in res["probes"] if p["night"] == 2 and p["arm"] == "intact")
    kinds = [r["kind"] for r in p1["rows"]]
    assert kinds.count("H") == 6 and kinds.count("never") == 3 and kinds.count("neardup") == 2
    assert sum(k.startswith("open-") for k in kinds) == 6
    off = next(p for p in res["probes"] if p["night"] == 2 and p["arm"] == "hippocampus-off")
    assert all(r["recalled"] is None for r in off["rows"])             # no hippocampus, no recall
    assert os.path.exists(str(tmp_path / "a" / "o" / "life_s7.md")) and "morning after night 2" in open(str(tmp_path / "a" / "o" / "life_s7.md")).read()
    life_table.render([res])                                           # the table reads it


def test_probes_leave_the_trunk_byte_identical(tmp_path):
    res = _run(tmp_path, "b", 2)
    home = str(tmp_path / "b" / "h" / "trunk")
    before = _hash(home)
    sim = life.SimClock(life.day_start(3))
    clock.set(sim.now)
    ledger = res["ledger"]
    out = life.probe(home, StubCortex(), 2, tuple(life.ARMS), life.battery(7, 2, ledger), ledger, sim, 7, BKW,
                     str(tmp_path / "b" / "w"))
    assert len(out) == 4 and _hash(home) == before
    assert not os.path.exists(str(tmp_path / "b" / "w" / "probe_L2_intact"))      # the copy is deleted after use


def test_resume_equals_non_resume(tmp_path):
    whole = _run(tmp_path, "w", 2, nosleep_arm=True)
    # a life of 1 day, then resumed to 2
    _run(tmp_path, "r", 1, nosleep_arm=True)
    resumed = _run(tmp_path, "r", 2, nosleep_arm=True, resume=True)
    assert _replies(whole) == _replies(resumed)
    key = lambda r: sorted((n["line"], n["night"]) for n in r["nights"])  # noqa: E731
    assert key(resumed) == key(whole)


def test_resume_after_a_cut_between_save_and_night(tmp_path):
    whole = _run(tmp_path, "w2", 2)
    real = life.live_day
    state = {"cut": True}

    def cutting(home, cortex, day, ledger, sim, seed, kw, sleeps, on_lived=None, weekend=False):
        def hook(rows, live_s):
            on_lived(rows, live_s)
            if day["day"] == 2 and state["cut"]:
                state["cut"] = False
                raise RuntimeError("cut")
        return real(home, cortex, day, ledger, sim, seed, kw, sleeps, on_lived=hook, weekend=weekend)
    life.live_day = cutting
    try:
        with pytest.raises(RuntimeError):
            _run(tmp_path, "c", 2)
    finally:
        life.live_day = real
    res = _run(tmp_path, "c", 2, resume=True)
    with open(str(tmp_path / "c" / "h" / "trunk" / "meta.json")) as fh:
        assert json.load(fh)["sleeps"] == 2                       # the cut night was slept on resume
    assert _replies(whole) == _replies(res)
