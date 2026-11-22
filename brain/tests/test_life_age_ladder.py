# Capability: Lifelong learning
# Record: brain/research/2026-11-22-fresh-test-split-design.md
# Expected-fail: none (mechanics of the age-ladder sub-battery and its flag-off identity)
"""brain/eval/life_age_ladder.py and life.py --age-ladder (design 2026-11-22 sections 2, 3 and 5). The v2 bank is
written by another agent: the plan tests run on the v1 TEST bank and on a synthetic v2-shaped bank (v1 TEST content,
D slots set to name, a synthetic 48-tuple NEVER_LADDER), with life.use_split monkeypatched for "test2"."""
import dataclasses
import hashlib
import json
import os
import types

import pytest

import brain.brain as bb
import brain.eval.life as life
from brain import clock
from brain.brain import Brain
from brain.eval import life_age_ladder as LA
from brain.eval import life_bank as DEV
from brain.eval import life_bank_test as T1
from brain.eval import life_ladder
from brain.tests import life_base_tree as BT
from brain.tests.test_brain_key_switch import KW, StubCortex

TEST_SEEDS = (42, 43, 44, 100, 101, 102)
DEV_SEEDS = (310, 311, 312, 319, 320, 321)
BKW = {k: v for k, v in KW.items() if k != "sleep_on_restart"}


def synth_v2():
    """A v2-shaped bank: v1 TEST content, D all name (section 4 slot rule), NEVER_LADDER thing 22 / name 14 / place 12."""
    m = types.ModuleType("synth_v2")
    for k in dir(T1):
        if not k.startswith("__"):
            setattr(m, k, getattr(T1, k))
    m.D = [dataclasses.replace(x, slot="name") for x in T1.D]
    m.F = [x if x.slot != "job" else dataclasses.replace(x, slot="thing") for x in T1.F]
    m.ALL = {x.id: x for x in T1.H + T1.AB + T1.E + m.F + m.D}
    nl = [(f"nl_t{i:02d}", f"What is the synthetic thing number {i}?", "thing") for i in range(22)]
    nl += [(f"nl_n{i:02d}", f"Who is synthetic person number {i}?", "name") for i in range(14)]
    nl += [(f"nl_p{i:02d}", f"Where is synthetic place number {i}?", "place") for i in range(12)]
    m.NEVER_LADDER = nl
    return m


@pytest.fixture
def v2(monkeypatch):
    bank = synth_v2()
    real = life.use_split

    def use(split):
        if split == "test2":
            life.LB = bank
            return bank
        return real(split)
    monkeypatch.setattr(life, "use_split", use)
    yield bank
    real("dev")


def _plan(seed, days, split):
    sched, led = life.build_life(seed, days, split)
    return LA.age_ladder_plan(seed, days, sched, led, life.LB), sched, led


# ---------------------------------------------------------------- the plan
@pytest.mark.parametrize("split,seeds", [("test", TEST_SEEDS), ("test2", TEST_SEEDS)])
def test_15_day_plans_on_test_sized_pools(split, seeds, v2):
    for s in seeds:
        p, sched, led = _plan(s, 15, split)
        late = life_ladder.live_asked_late(sched, led)
        assert len(p["items"]) == 12, (s, len(p["items"]))
        assert p["n_items_age10p"] >= 8 and sum(1 for x in p["items"] for a in x["ages"] if a >= 10) == 22
        assert all(len(x["ages"]) >= 3 for x in p["items"])
        assert sorted(c["told_day"] for c in p["controls"]) == [11, 11, 12, 13, 13, 14, 15, 15]
        assert not {x["item"] for x in p["items"] + p["controls"]} & late
        sd = {t["item"] for d in sched for t in d["turns"] if t["kind"] == "sameday"}
        assert not {x["item"] for x in p["items"] + p["controls"]} & sd
        assert all(led[x["item"]]["cls"] in ("F", "D") and len(led[x["item"]]["told_days"]) == 1
                   for x in p["items"] + p["controls"])
        for x in p["items"]:
            t = x["told_day"]
            assert 2 <= t <= 6 and x["nights"][0] == t and x["nights"][-1] == 15
            assert x["ages"] == [n - t + 1 for n in x["nights"]]
            assert len(x["ages"]) == (4 if t == 6 else 5)
    life.use_split("dev")


def test_dev_plans_have_items_and_no_controls():
    for s in DEV_SEEDS:
        p, _, _ = _plan(s, 15, "dev")
        assert len(p["items"]) == 12 and p["controls"] == [] and p["n_items_age10p"] == 12
        assert p["never"] == {"A": [], "B": [], "quotas": {}, "slot_counts": p["never"]["slot_counts"]}   # no NEVER_LADDER


def test_clipping_at_days_3_and_7():
    p3, _, _ = _plan(42, 3, "test")
    assert {x["told_day"] for x in p3["items"]} == {2, 3} and p3["controls"] == []
    assert all(set(x["ages"]) <= {1, 2} for x in p3["items"])
    assert {tuple(x["nights"]) for x in p3["items"]} == {(2, 3), (3,)}
    assert p3["arm_nights"] == {"hippocampus-off": [2, 3], "intact": [], "both-off": []}
    p7, _, _ = _plan(42, 7, "test")
    want = {2: [2, 5, 7], 3: [3, 6, 7], 4: [4, 7], 5: [5, 7], 6: [6, 7]}
    assert {x["told_day"]: x["nights"] for x in p7["items"]} == want
    assert p7["arm_nights"]["intact"] == [5] and p7["arm_nights"]["both-off"] == [5]
    life.use_split("dev")


def test_never_questions_quotas_halves_and_disjoint(v2):
    assert LA._quotas({"name": 2, "place": 1, "thing": 9}, {"thing": 22, "name": 14, "place": 12}) == \
        {"name": 4, "place": 4, "thing": 16}
    assert LA._quotas({"name": 5, "thing": 7}, {"thing": 22, "name": 14, "place": 12}) == {"name": 10, "thing": 14}
    assert LA._quotas({"job": 2, "thing": 10}, {"thing": 22, "name": 14, "place": 12}) == {"thing": 22}
    slot = {n[0]: n[2] for n in v2.NEVER_LADDER}
    for s in TEST_SEEDS:
        p, _, _ = _plan(s, 15, "test2")
        a, b = p["never"]["A"], p["never"]["B"]
        assert len(a) == len(b) == 12 and not set(a) & set(b)
        q = p["never"]["quotas"]
        assert sum(q.values()) == 24 and all(v >= 4 for v in q.values())
        assert set(q) == set(p["never"]["slot_counts"])
        got = {}
        for i in a + b:
            got[slot[i]] = got.get(slot[i], 0) + 1
        assert got == q
        assert not set(a + b) & {n[0] for n in v2.NEVER}
        assert _plan(s, 15, "test2")[0] == p               # deterministic
    p, _, _ = _plan(42, 15, "test2")
    assert [q["half"] for q in LA.due(p, 12, v2) if q["kind"] == "never"] == ["A"] * 12
    assert [q["half"] for q in LA.due(p, 13, v2) if q["kind"] == "never"] == ["B"] * 12


def test_never_pool_clash_is_refused(v2):
    v2.NEVER_LADDER = v2.NEVER_LADDER + [(v2.NEVER[0][0], "Another text?", "thing")]
    sched, led = life.build_life(42, 15, "test2")
    with pytest.raises(ValueError):
        LA.age_ladder_plan(42, 15, sched, led, v2)


def test_due_counts_match_the_budget(v2):
    p, _, _ = _plan(42, 15, "test2")
    hoff = sum(len(LA.due(p, n, v2)) for n in p["arm_nights"]["hippocampus-off"])
    assert hoff == 58 + 8 + 12 * 14          # ladder 58 + controls 8 + never 12 x 14 (design 6 budget)
    assert sum(1 for n in (5, 10, 15) for q in LA.due(p, n, v2) if q["kind"] != "never") <= 24


# ---------------------------------------------------------------- the battery under the flag
@pytest.mark.parametrize("split", ["test", "test2"])
def test_no_ladder_item_in_rotating_or_dpool_draws_under_the_flag(split, v2):
    hit_off = 0
    for s in TEST_SEEDS:
        p, _, led = _plan(s, 15, split)
        ex = set(p["exclude"])
        for n in range(1, 16):
            on = life.battery(s, n, led, exclude=p["exclude"])
            assert not {q["item"] for q in on if q["kind"] in ("rotating", "neardup")} & ex
            assert sum(q["kind"] == "neardup" for q in on) == 2
            off = life.battery(s, n, led)
            assert off == life.battery(s, n, led, exclude=None)
            hit_off += len({q["item"] for q in off if q["kind"] in ("rotating", "neardup")} & ex)
            assert [q for q in on if q["kind"] not in ("rotating", "neardup")] == \
                [q for q in off if q["kind"] not in ("rotating", "neardup")]
    assert hit_off > 0       # without the flag the draws do hit ladder items (the exclusion is not vacuous)
    life.use_split("dev")


# ---------------------------------------------------------------- scorers (section 3 controls)
def _led():
    e = lambda i, d: {"cls": DEV.ALL[i].cls, "told_days": [d], "keys": list(DEV.ALL[i].keys),  # noqa: E731
                      "telling": DEV.ALL[i].telling}
    return {"dan": e("dan", 1), "nova": e("nova", 1), "marguerite": e("marguerite", 1), "pixel": e("pixel", 1),
            "ab03": e("ab03", 2), "ab07": e("ab07", 20)}


def test_slot_false_positive_and_negative_controls():
    led, q = _led(), "Who is my landlord?"
    assert LA.slot_false("Your landlord is called Okonkwo.", q, "name", led, 5, DEV)        # same-slot told name
    assert LA.slot_false("Your landlord is called Marguerite.", q, "name", led, 5, DEV)     # other H keys count
    assert not LA.slot_false("I'm Nova.", q, "name", led, 5, DEV)                           # STOCK
    assert not LA.slot_false("Dan is a name.", q, "name", led, 5, DEV)                      # STOCK
    assert not LA.slot_false("My sister is called Marguerite.", q, "name", led, 5, DEV)    # echo / wrong subject
    assert not LA.slot_false("Your landlord is called Okonkwo.", q, "place", led, 5, DEV)  # another slot
    assert not LA.slot_false("Your landlord is called Okonkwo.", q, "name", led, 1, DEV)   # not told by that morning
    assert not LA.slot_false("Your landlord is called Fitzgerald.", q, "name", led, 5, DEV)   # told on day 20
    assert LA.lures(led, 5, "name", DEV) and not [k for k, _ in LA.lures(led, 5, "name", DEV) if k in ("dan", "nova")]


def test_fm_v2_and_v1_style():
    led, q = _led(), "Who is my landlord?"
    assert not LA.any_key_false("I'm Nova.", q, led, 5) and LA.any_key_false("I'm Nova.", q, led, 5, stock=True)
    assert LA.any_key_false("My sister is called Marguerite.", q, led, 5)       # any told key, any subject
    assert not LA.any_key_false("Pixel?", "Is Pixel my landlord?", led, 5)       # a key in the question never counts
    assert not LA.any_key_false("Fitzgerald.", q, led, 5)                        # not told yet
    assert LA.H_TELLING["admin"].startswith("Nice to meet you.")               # H: the SESSION_1 line


# ---------------------------------------------------------------- the life on the stub cortex
@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr(bb, "replay_into_cortex", lambda *a, **k: [])
    monkeypatch.setattr(Brain, "_plastic", lambda self: type("P", (), {"parameters": lambda s: [], "B_k": {}, "B_v": {}})())
    real = life.build_life

    def short(seed, days, split="dev"):
        sched, ledger = real(seed, days, split)
        for d in sched:
            d["turns"] = d["turns"][:4]
        return sched, ledger
    monkeypatch.setattr(life, "build_life", short)
    yield
    clock.set(None)
    life.use_split("dev")


def _strip(res):
    out = {k: v for k, v in res.items() if k != "age_ladder"}
    out["probes"] = [{k: v for k, v in p.items() if k != "age_ladder"} for p in res["probes"]]
    return out


def _run(tmp_path, name, days, split="test", full=(5,), **kw):
    return life.run(StubCortex(), 42, days, split, str(tmp_path / name / "h"), str(tmp_path / name / "o" / "life_s42"),
                    set(full), brain_kw=BKW, log=lambda *_: None, **kw)


def test_ladder_rows_go_to_their_own_field_never_rows(tmp_path, stub, v2):
    res = _run(tmp_path, "on", 5, "test2", age_ladder=True)
    assert res["age_ladder"]["days"] == 5
    by = {(p["night"], p["arm"]): p for p in res["probes"]}
    for (n, arm), p in by.items():
        kinds = {r["kind"] for r in p["rows"]}
        assert not kinds & {"ladder", "control"}
        assert all(r.get("kind") != "never" or "slot_false" not in r for r in p["rows"])
        want = arm in LA.arms_for(res["age_ladder"], n)
        assert ("age_ladder" in p) == want, (n, arm)
    lad = by[(5, "hippocampus-off")]["age_ladder"]
    assert len(lad) == len(LA.due(res["age_ladder"], 5, v2)) and {r["kind"] for r in lad} >= {"ladder", "never"}
    assert all(r["recalled"] is None for r in lad)                       # hippocampus off: no recall
    assert "age_ladder" in by[(5, "intact")] and "age_ladder" in by[(5, "both-off")]
    assert "age_ladder" not in by[(1, "hippocampus-off")]
    md = open(str(tmp_path / "on" / "o" / "life_s42.md")).read()
    assert "## Age ladder" in md
    from brain.eval import life_bars, life_table
    assert "ladder recall off" in life_table.render([res]) and "ladder recall" not in life_table.render([_strip(res)])
    A = life_bars.analyse([res])
    assert A["p0"]["instrument"] == "age-ladder" and A["bars"][1]["id"] == "FM-v2"
    off = _run(tmp_path, "off", 5, "test2")
    assert "age_ladder" not in off and all("age_ladder" not in p for p in off["probes"])
    # the flag changes only the battery's rotating / near-dup draws, never the rest of a probe's rows
    for p in off["probes"]:
        q = next(x for x in res["probes"] if x["night"] == p["night"] and x["arm"] == p["arm"])
        assert [r["you"] for r in p["rows"] if r["kind"] not in ("rotating", "neardup")] == \
            [r["you"] for r in q["rows"] if r["kind"] not in ("rotating", "neardup")]


def test_resume_keeps_the_ladder_and_refuses_a_new_one(tmp_path, stub, v2):
    whole = _run(tmp_path, "r", 3, "test2", age_ladder=True)
    res = _run(tmp_path, "r", 3, "test2", resume=True)          # the saved flag wins
    assert res["age_ladder"] == whole["age_ladder"] and any("age_ladder" in p for p in res["probes"] if p["night"] == 3)
    with pytest.raises(ValueError):                              # like --cue-ladder: the plan holds the horizon
        _run(tmp_path, "r", 4, "test2", resume=True)
    _run(tmp_path, "n", 2, "test2")
    with pytest.raises(ValueError):
        _run(tmp_path, "n", 3, "test2", resume=True, age_ladder=True)


def _hash(root):
    out = {}
    for d, _, fs in sorted(os.walk(root)):
        for f in sorted(fs):
            if f != ".lock":
                with open(os.path.join(d, f), "rb") as fh:
                    out[os.path.relpath(os.path.join(d, f), root)] = hashlib.sha256(fh.read()).hexdigest()
    return out


def test_trunk_byte_identical_after_ladder_probes(tmp_path, stub, v2):
    res = _run(tmp_path, "b", 5, "test2")
    home = str(tmp_path / "b" / "h" / "trunk")
    before = _hash(home)
    sched, led = life.build_life(42, 5, "test2")
    plan = LA.age_ladder_plan(42, 5, sched, led, life.LB)
    sim = life.SimClock(life.day_start(6))
    clock.set(sim.now)
    out = life.probe(home, StubCortex(), 5, tuple(life.ARMS), life.battery(42, 5, led, exclude=plan["exclude"]), led,
                     sim, 42, BKW, str(tmp_path / "b" / "w"), age_ladder=plan)
    assert _hash(home) == before and res["ledger"] == led
    assert {p["arm"] for p in out if "age_ladder" in p} == {"hippocampus-off", "intact", "both-off"}
    assert not os.path.exists(str(tmp_path / "b" / "w" / "probe_L5_intact"))


# ---------------------------------------------------------------- flag-off identity with the pre-change code
DRIVER = r'''
import json, sys
mode, split, seed, days, homes, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
import brain.brain as bb
import brain.eval.life as life
from brain.brain import Brain
if mode in ("stub", "stub-cue"):
    from brain.tests.test_brain_key_switch import KW, StubCortex
    bb.replay_into_cortex = lambda *a, **k: []
    Brain._plastic = lambda self: type("P", (), {"parameters": lambda s: [], "B_k": {}, "B_v": {}})()
    cortex, kw = StubCortex(), {k: v for k, v in KW.items() if k != "sleep_on_restart"}
    life.run(cortex, seed, days, split, homes, out, {days}, brain_kw=kw, log=lambda *_: None,
             **({"cue_ladder": True} if mode == "stub-cue" else {}))
else:
    life.main(["--model", "0.1b", "--seed", str(seed), "--days", str(days), "--split", split,
               "--full-probe-nights", str(days), "--homes", homes, "--out", out])
'''


def _identity(tmp_path, mode, split, seed, days):
    base = BT.export_base(str(tmp_path / "base"))
    outs = {}
    for tag, root in (("old", base), ("new", BT.REPO)):
        o = str(tmp_path / tag / "life")
        BT.run_driver(root, DRIVER, [mode, split, str(seed), str(days), str(tmp_path / tag / "h"), o])
        with open(o + ".json") as fh:
            j = json.load(fh)
        with open(o + ".md") as fh:
            outs[tag] = (json.dumps(BT.scrub(j), indent=1), BT.scrub_md(fh.read()))
    assert outs["old"][0] == outs["new"][0]
    assert outs["old"][1] == outs["new"][1]
    return outs


@pytest.mark.parametrize("split,seed", [("dev", 310), ("test", 42)])
def test_flag_off_life_identical_to_pre_change_code_stub(tmp_path, split, seed):
    """Flag-off life JSON and md equal the pre-change code's (timings scrubbed), stub cortex, 3 days, full night 3."""
    _identity(tmp_path, "stub", split, seed, 3)


@pytest.mark.parametrize("split,seed", [("dev", 310), ("test", 42)])
def test_flag_off_cue_ladder_life_identical_to_pre_change_code_stub(tmp_path, split, seed):
    """--cue-ladder with --age-ladder off: the life JSON (incl. cue_ladder.config.cues_sha256 and the cue plan) and md
    equal the pre-change code's, and the recorded cue hash is the registered v1 file hash."""
    outs = _identity(tmp_path, "stub-cue", split, seed, 3)
    j = json.loads(outs["new"][0])
    assert j["cue_ladder"]["config"]["cues_sha256"] == "2055dc92678c7481cac99d067a24f079126b80feb5d7a799e467cf3a23b3247e"
    assert j["cue_ladder"]["plan"]["cells"], "the 3-day cue plan must have cells for the identity check to bite"


@pytest.mark.skipif(os.environ.get("LIFE_0P1B_IDENTITY") != "1",
                    reason="0.1B 3-day lives take ~15-20 min each on a loaded CPU; set LIFE_0P1B_IDENTITY=1 to run")
@pytest.mark.parametrize("split,seed", [("dev", 310), ("test", 42)])
def test_flag_off_life_identical_to_pre_change_code_0p1b(tmp_path, split, seed):
    _identity(tmp_path, "0.1b", split, seed, 3)
