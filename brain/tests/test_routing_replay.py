# Capability: Episodic recall
# Record: brain/research/2026-11-24-routing-screen-prereg.md
# Expected-fail: none (harness plumbing on the stub cortex; no model weights)
"""routing_replay on the stub cortex: arm A is Hippocampus.recall exactly (plain and frozen paths) and changes no
episode; D+G equals A whenever the raw and weighted winners agree; on a stub life with --night-snapshots the replay
reproduces every banked intact-frozen fetch from the snapshots alone."""
import copy
import json

import pytest

from brain import clock
from brain.brain import BIRTH
from brain.experiments import routing_replay as RR
from brain.hippocampus import Hippocampus
from brain.tests.test_brain_key_switch import StubCortex
from brain.tests.test_life import BKW, _env  # noqa: F401  (autouse fixture: fake night, 3 turns a day)

T = 1_800_000_000.0
TELLINGS = ["My cat is called Pixel.", "I drive a red Volvo.", "My sister's name is Marguerite.",
            "My dog is called Bramble.", "The bus was late again this morning.", "My barber is called Rashid."]
CUES = ["What is my cat called?", "What car do I drive?", "Who is my sister?", "What is my dog called?",
        "Was the bus late?", "Who cuts my hair?", "Tell me a story about a fox.", "Good morning! How are you?"]


@pytest.fixture
def store():
    clock.set(lambda: T)
    cx = StubCortex()
    _, base = cx.read(BIRTH, None)
    hc = Hippocampus()
    for i, t in enumerate(TELLINGS):
        e = hc.store(cx, base, t)
        e.strength, e.last_used = [1.0, 9.0, 0.5, 8.5, 1.0, 2.0][i], T - 86400.0 * [10, 1, 6, 0, 3, 12][i]
    yield hc, cx, base
    clock.set(None)


def _snap(hc):
    return [(e.strength, e.uses, e.last_used) for e in hc.episodes]


@pytest.mark.parametrize("bar", [None, 0.0, -1.0])
def test_arm_A_is_recall(store, bar):
    hc, cx, base = store
    if bar is not None:   # lower bars so the stub's small similarities fire too (both sides see the same bar)
        hc.FAMILIARITY = hc.FAMILIARITY_SPONTANEOUS = bar
    before = _snap(hc)
    for cue in CUES:
        d = RR.decide(hc, cx, base, cue)
        assert _snap(hc) == before and hc.FREEZE_USE is False and hc.last_fetch is None   # nothing changed
        plain = copy.deepcopy(hc)
        ep, s = plain.recall(cx, base, cue, question=cue.endswith("?"))           # the default (plain) path
        assert d["A"]["idx"] == (None if ep is None else plain.episodes.index(ep)) and d["A"]["score"] == pytest.approx(s)
        frz = copy.deepcopy(hc)
        frz.FREEZE_USE = True
        ep2, s2 = frz.recall(cx, base, cue, question=cue.endswith("?"))          # the intact-frozen arm's path
        assert frz.last_fetch == d["A_fetch"] and s2 == d["A"]["score"]
        raw = copy.deepcopy(hc)
        raw.RANK_ON_RAW = True
        ep3, _ = raw.recall(cx, base, cue, question=cue.endswith("?"))            # arm D = the existing switch
        assert d["D"]["idx"] == (None if ep3 is None else raw.episodes.index(ep3))


def test_DG_differs_only_when_winners_differ(store):
    hc, cx, base = store
    hc.FAMILIARITY = hc.FAMILIARITY_SPONTANEOUS = -1.0     # every cue fires in A, so only the choice can differ
    hc.MARGIN = 0.0
    differ = 0
    for cue in CUES + TELLINGS:
        d = RR.decide(hc, cx, base, cue)
        if d["raw_winner"] == d["weighted_winner"]:
            assert d["DG"]["idx"] == d["A"]["idx"]
        else:
            assert d["A"]["idx"] == d["weighted_winner"] and d["DG"]["idx"] == d["raw_winner"]
            differ += 1
    assert differ > 0, "the store's strengths should make raw and weighted winners differ on some cue"
    # with the bar back, D+G never fires where A does not
    hc.FAMILIARITY, hc.FAMILIARITY_SPONTANEOUS, hc.MARGIN = 0.12, 0.45, 0.02
    for cue in CUES:
        d = RR.decide(hc, cx, base, cue)
        assert d["A"]["idx"] is not None or d["DG"]["idx"] is None


def test_SC_caps_only_non_permanent(store):
    hc, cx, base = store
    for e in hc.episodes:
        e.strength = min(e.strength, RR.SC_CAP)
        e.last_used = T
    for cue in CUES:   # every effective strength <= the cap already: SC is A
        d = RR.decide(hc, cx, base, cue)
        assert d["SC"]["idx"] == d["A"]["idx"]


def test_replay_reproduces_a_stub_life(tmp_path):
    from brain.experiments import life_kw

    snaps = tmp_path / "snaps"
    argv = ["--seed", "7", "--days", "3", "--full-probe-nights", "1", "--frozen-probe", "--homes", str(tmp_path / "h"),
            "--out", str(tmp_path / "o" / "life_s7"), "--night-snapshots", str(snaps)]
    res = life_kw.main(argv, cortex=StubCortex(), brain_kw=BKW)
    res = json.load(open(str(tmp_path / "o" / "life_s7") + ".json"))
    assert RR.validate(res, res) == []
    out = RR.replay(res, str(snaps), StubCortex(), str(tmp_path / "work"), weekend=False, nights=range(1, 4),
                    trunk=str(tmp_path / "h" / "trunk"), log=lambda *a: None)
    assert [n["night"] for n in out["nights"]] == [1, 2, 3]
    for n in out["nights"]:
        assert n["void_rows"] == [] and not n["void"], n["void_rows"]
        assert n["rows"] and all(r["A"]["idx"] == r["banked_idx"] for r in n["rows"])
    v = RR.score([dict(out, seed=s) for s in (1, 2, 3)])
    assert v["status"] == "READ" and set(v["arms"]) == set(RR.BARRED)


def test_replay_voids_a_wrong_snapshot(tmp_path):
    """A snapshot that is not the battery's store (one episode's strength changed) must not pass silently."""
    from brain.experiments import life_kw

    snaps = tmp_path / "snaps"
    argv = ["--seed", "7", "--days", "2", "--full-probe-nights", "1", "--frozen-probe", "--homes", str(tmp_path / "h"),
            "--out", str(tmp_path / "o" / "life_s7"), "--night-snapshots", str(snaps)]
    life_kw.main(argv, cortex=StubCortex(), brain_kw=BKW)
    res = json.load(open(str(tmp_path / "o" / "life_s7") + ".json"))
    for p in res["probes"]:   # corrupt the banked record instead: the replay must flag the row
        if p["arm"] == RR.FROZEN and p["night"] == 2:
            p["rows"][-1]["fetch"]["idx"] = 999
    out = RR.replay(res, str(snaps), StubCortex(), str(tmp_path / "work"), weekend=False, nights=[2],
                    log=lambda *a: None)
    assert out["nights"][0]["void"] and out["nights"][0]["void_rows"] == [len(out["nights"][0]["rows"]) - 1]


def test_reinstate_arm_A_is_the_frozen_probe(tmp_path):
    """--reinstate: the battery through Brain.hear with arm A's decision gives the banked frozen replies, row for row."""
    from brain.experiments import life_kw

    snaps = tmp_path / "snaps"
    argv = ["--seed", "7", "--days", "2", "--full-probe-nights", "1", "--frozen-probe", "--homes", str(tmp_path / "h"),
            "--out", str(tmp_path / "o" / "life_s7"), "--night-snapshots", str(snaps)]
    life_kw.main(argv, cortex=StubCortex(), brain_kw=BKW)
    res = json.load(open(str(tmp_path / "o" / "life_s7") + ".json"))
    out = RR.replay(res, str(snaps), StubCortex(), str(tmp_path / "work"), weekend=False, nights=[2],
                    trunk=str(tmp_path / "h" / "trunk"), reinstate=True, log=lambda *a: None)
    n = out["nights"][0]
    assert n["replies"]["A_reproduces_banked_replies"] == len(n["rows"])
    fetched = sum(r["A"]["idx"] is not None for r in n["rows"])
    print("stub fetches", fetched, "of", len(n["rows"]))
