# Capability: Episodic recall
# Record: brain/research/2026-11-25-routing-screen2-prereg.md
# Expected-fail: none (harness plumbing on the stub cortex; no model weights)
"""routing_screen2 on the stub cortex: arm K is KeyedHippocampus(S_all, slept bar).recall under FREEZE_USE exactly and
changes no episode; on a stub life with night snapshots, arm A reproduces every banked fetch and the scorer reads."""
import copy
import json

import pytest

from brain import clock
from brain.brain import BIRTH
from brain.experiments import routing_screen2 as S2
from brain.hippocampus import Hippocampus
from brain.hippocampus_keys import KeyedHippocampus
from brain.tests.test_brain_key_switch import StubCortex
from brain.tests.test_life import BKW, _env  # noqa: F401  (autouse fixture: fake night, 3 turns a day)
from brain.tests.test_routing_replay import CUES, TELLINGS, T


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


@pytest.mark.parametrize("bar", [None, 0.0, -1.0])
def test_K_is_keyed_recall_frozen(store, bar):
    hc, cx, base = store
    k = S2.keyed_from(hc)
    if bar is not None:
        k.FAMILIARITY = k.FAMILIARITY_SPONTANEOUS = bar
    before = [(e.strength, e.uses, e.last_used) for e in hc.episodes]
    for cue in CUES + TELLINGS:
        d = S2.decide_K(k, cx, base, cue, T)
        assert [(e.strength, e.uses, e.last_used) for e in hc.episodes] == before
        ref = KeyedHippocampus(key="S_all", familiarity=k.FAMILIARITY, margin=k.MARGIN)
        ref.FAMILIARITY_SPONTANEOUS = k.FAMILIARITY_SPONTANEOUS
        ref.episodes = copy.deepcopy(hc.episodes)
        ref.FREEZE_USE = True
        ep, s = ref.recall(cx, base, cue, question=cue.strip().endswith("?"))
        assert d["idx"] == (None if ep is None else ref.episodes.index(ep))
        assert d["score"] == pytest.approx(s)
        plain = copy.deepcopy(ref)            # the live (non-frozen) path picks the same episode
        plain.FREEZE_USE = False
        ep2, _ = plain.recall(cx, base, cue, question=cue.strip().endswith("?"))
        assert d["idx"] == (None if ep2 is None else plain.episodes.index(ep2))


def test_route_delta_episodes_are_read(store):
    """A completion episode keeps a plain routing change; arm K reads it as the live key does."""
    hc, cx, base = store
    hc.store(cx, base, "What's your name? My name is Nova.", completion_of="What's your name?")
    assert getattr(hc.episodes[-1], "route_delta", None) is not None
    k = S2.keyed_from(hc)
    k.FAMILIARITY = k.FAMILIARITY_SPONTANEOUS = -1.0
    ref = KeyedHippocampus(key="S_all", familiarity=-1.0, margin=k.MARGIN)
    ref.FAMILIARITY_SPONTANEOUS = -1.0
    ref.episodes = copy.deepcopy(hc.episodes)
    ref.FREEZE_USE = True
    for cue in ("What's your name?", "Who are you?") + tuple(CUES):
        ep, _ = ref.recall(cx, base, cue, question=True)
        assert S2.decide_K(k, cx, base, cue, T)["idx"] == (None if ep is None else ref.episodes.index(ep))


def test_keyed_from_uses_the_slept_bar(store):
    hc, _, _ = store
    k = S2.keyed_from(hc)
    assert (k.FAMILIARITY, k.MARGIN) == S2.SALL_SLEPT and k.FAMILIARITY_SPONTANEOUS == Hippocampus.FAMILIARITY_SPONTANEOUS
    assert k.episodes is hc.episodes and k.FREEZE_USE


def test_screen2_on_a_stub_life(tmp_path):
    from brain.experiments import life_kw

    snaps = tmp_path / "snaps"
    argv = ["--seed", "7", "--days", "3", "--full-probe-nights", "1", "--frozen-probe", "--homes", str(tmp_path / "h"),
            "--out", str(tmp_path / "o" / "life_s7"), "--night-snapshots", str(snaps)]
    life_kw.main(argv, cortex=StubCortex(), brain_kw=BKW)
    res = json.load(open(str(tmp_path / "o" / "life_s7") + ".json"))
    out = S2.run(res, str(snaps), str(tmp_path / "h" / "trunk"), StubCortex(), str(tmp_path / "w"), weekend=False,
                 nights=range(1, 4), log=lambda *a: None, size=None)
    assert [n["night"] for n in out["nights"]] == [1, 2, 3]
    for n in out["nights"]:
        assert not n["void"], n["void_rows"]
        assert all(r["A"]["idx"] == r["banked_idx"] for r in n["rows"])
    v = S2.score([dict(out, seed=s) for s in (1, 2, 3)], {})
    assert v["status"] == "READ" and v["K"]["status"] and v["per_night"]
