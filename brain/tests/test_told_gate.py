"""confab_m2 --told-gate (default OFF): the ontop condition reinstates the lesson BEFORE normal recall, open does not,
no lesson = open; the report's bars; the default run path is untouched. Stub cortex, no model weights."""
import json
import types

from brain.brain import Brain
from brain.experiments import confab_m2 as m
from brain.experiments.abstain_v2 import keyed, reset
from brain.experiments.facts import FACTS
from brain.tests.test_brain_key_switch import KW, StubCortex


def _brain(tmp_path, lesson):
    b = keyed(Brain(str(tmp_path / ("l" if lesson else "n")), cortex=StubCortex(), **KW))
    for s, _, _ in FACTS[:3]:
        b.hear(s)
    if lesson:
        b.live_self_belief(m.LESSON_Q, m.GUIDED)
    return b


def _ask(b, cond, log):
    snap = [(e.strength, e.uses, e.last_used) for e in b.hippocampus.episodes]
    base = [x.clone() for x in b.base]
    orig = b.hippocampus.reinstate

    def spy(cortex, state, ep):
        log.append(ep.text)
        return orig(cortex, state, ep)
    b.hippocampus.reinstate = spy
    reset(b, snap, base)
    q, key, told = m.TOLD_SETS["told"][0]
    try:
        return m.ask_told(b, q, key, told, cond, 42, 0, b.self_belief())
    finally:
        b.hippocampus.reinstate = orig


def test_ontop_reinstates_lesson_first(tmp_path):
    b = _brain(tmp_path, lesson=True)
    log_open, log_top = [], []
    _ask(b, "open", log_open)
    _ask(b, "ontop", log_top)
    belief = b.self_belief().text
    assert belief not in log_open[:1]                      # open: the lesson is not put in by the harness
    assert log_top[0] == belief                            # ontop: lesson first, then normal recall (if any)
    assert log_top[1:] == log_open


def test_no_lesson_ontop_equals_open(tmp_path):
    b = _brain(tmp_path, lesson=False)
    assert _ask(b, "open", []) == _ask(b, "ontop", [])


def test_sets():
    assert len(m.TOLD_SETS["told"]) == len(m.TOLD_SETS["told_para"]) == 12
    assert all(k and s for _, k, s in m.TOLD_SETS["told"] + m.TOLD_SETS["told_para"])


def _file(d, seed, correct):
    arms = {}
    for arm, nc in correct.items():
        rows = []
        for cond in m.TOLD_CONDS:
            for st in m.TOLD_SETS:
                for i in range(12):
                    c = i < nc[cond]
                    rows.append({"cond": cond, "set": st, "q": f"q{i}", "reply": f"r{i}{c}", "correct": c,
                                 "dk": bool(not c and arm != "none" and cond == "ontop" and nc.get("dk_all")),
                                 "right_episode": i < 6, "lesson_recalled": False, "verbatim_guided": False})
        summ = {f"{cond}/{st}": {k: sum(bool(r[k]) for r in rows if r["cond"] == cond and r["set"] == st) for k in
                                 ("correct", "dk", "right_episode", "lesson_recalled", "verbatim_guided")}
                for cond in m.TOLD_CONDS for st in m.TOLD_SETS}
        arms[arm] = {"summary": summ, "rows": rows}
    with open(d / f"told_s{seed}.json", "w") as fh:
        json.dump({"seed": seed, "arms": arms}, fh)


def test_report_pass_and_fail(tmp_path):
    ok, bad = tmp_path / "ok", tmp_path / "bad"
    ok.mkdir()
    bad.mkdir()
    for s in m.SEEDS:
        _file(ok, s, {"none": {"open": 4, "ontop": 4}, "lived_self": {"open": 4, "ontop": 4}})
        _file(bad, s, {"none": {"open": 4, "ontop": 4}, "lived_self": {"open": 4, "ontop": 1, "dk_all": True}})
    r_ok = m.told_report(str(ok), arms=("none", "lived_self"))
    r_bad = m.told_report(str(bad), arms=("none", "lived_self"))
    assert r_ok["criteria"]["T0"]["powered"] and r_ok["criteria"]["lived_self"]["PASS"]
    assert r_bad["criteria"]["lived_self"]["FAIL"] and not r_bad["criteria"]["lived_self"]["ontop/told"]["T1"]
    assert not r_bad["criteria"]["lived_self"]["T3_dk_known"]["pass"]


def test_default_path_untouched(monkeypatch):
    calls = []
    monkeypatch.setattr(m, "run_arm", lambda a, c, arm: calls.append(("old", arm)) or {"rows": []})
    monkeypatch.setattr(m, "run_told_arm", lambda a, c, arm: calls.append(("told", arm)) or {"rows": []})
    res = m.run_phase(types.SimpleNamespace(seed=42, arms="none", max_q=0), None)
    assert calls == [("old", "none")] and "told_gate" not in res
    calls.clear()
    m.run_phase(types.SimpleNamespace(seed=42, arms="none", max_q=0, told_gate=True), None)
    assert calls == [("told", "none")]
