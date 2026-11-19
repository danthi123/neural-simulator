# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md
"""brain.clock: the default is time.time; a set clock reaches episode created/last_used, meta last_sleep and decay."""
import json
import os
import time

import pytest

import brain.brain as bb
from brain import clock
from brain.brain import Brain
from brain.hippocampus import Episode, Hippocampus
from brain.tests.test_brain_key_switch import KW, StubCortex


@pytest.fixture(autouse=True)
def _restore_clock():
    yield
    clock.set(None)


def _fake_sleep(monkeypatch):
    monkeypatch.setattr(bb, "replay_into_cortex", lambda *a, **k: [])
    monkeypatch.setattr(Brain, "_plastic", lambda self: type("P", (), {"parameters": lambda s: [], "B_k": {}, "B_v": {}})())


def test_default_is_wall_clock():
    t0 = time.time()
    assert t0 - 1 <= clock.now() <= time.time() + 1
    e = Episode(text="x", delta=[], key=None)
    assert t0 - 1 <= e.created <= time.time() + 1 and e.created == pytest.approx(e.last_used, abs=1)


def test_set_clock_reaches_new_episodes_and_decay():
    clock.set(lambda: 1000.0)
    e = Episode(text="x", delta=[], key=None)
    assert e.created == e.last_used == 1000.0          # default_factory is not bound to time.time at import
    h = Hippocampus()
    assert h.effective_strength(e) == pytest.approx(1.0)
    clock.set(lambda: 1000.0 + 3 * 86400)
    assert h.effective_strength(e) == pytest.approx((1 + 3.0) ** -0.5)
    clock.set(None)
    assert clock.now() != 1000.0


def test_pending_follows_the_world_clock(tmp_path, monkeypatch):
    _fake_sleep(monkeypatch)
    t = [10_000.0]
    clock.set(lambda: t[0])
    kw = {k: v for k, v in KW.items() if k != "sleep_on_restart"}
    b = Brain(str(tmp_path / "h"), cortex=StubCortex(), **kw)
    assert b.meta["last_sleep"] == 10_000.0 and b.meta["born"] == 10_000.0 and not b.pending()
    t[0] += 60.0
    b.hear("My cat is called Pixel.")
    ep = b.hippocampus.episodes[-1]
    assert ep.created == ep.last_used == 10_060.0
    assert len(b.pending()) == 1                       # a lived day leaves something pending
    t[0] += 3600.0
    b.close()                                          # the night
    assert not b.pending() and b.meta["last_sleep"] == 10_060.0 + 3600.0 and b.meta["sleeps"] == 1
    t[0] += 86400.0                                    # the next day: new episodes are later than the last night
    b2 = Brain(str(tmp_path / "h"), cortex=StubCortex(), **kw)
    assert not b2.pending()
    b2.hear("I drive a red Volvo.")
    new = b2.hippocampus.episodes[-1]
    assert new.created > b2.meta["last_sleep"] and len(b2.pending()) == 1
    b2.save()
    with open(os.path.join(str(tmp_path / "h"), "episodes.json")) as fh:
        idx = json.load(fh)
    assert max(r["created"] for r in idx) == new.created
