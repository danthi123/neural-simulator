# Capability: Lifelong learning
# Record: brain/research/2026-11-20-hub-rank-raw-prereg.md
# Expected-fail: none (mechanics only; the prereg's screen criteria are measured on banked homes, not here)
"""R-raw (Hippocampus.RANK_ON_RAW / Brain(rank="raw")): mechanics only, on a monkeypatched similarity. No model weights."""
import random

import pytest
import torch

from brain import clock
from brain.hippocampus import Episode, Hippocampus
from brain.tests.test_use_scale import DAY, T0, _ep, _fake, _recall, _ref_module, _state


@pytest.fixture(autouse=True)
def _clock():
    clock.set(lambda: T0)
    yield
    clock.set(None)


def test_rank_off_is_byte_identical_to_the_old_recall():
    ref = _ref_module()
    rng = random.Random(1)
    old, new = _fake(ref.Hippocampus), _fake(Hippocampus)
    for _ in range(6):
        s, age = rng.choice([0.5, 1.0, 2.0, 6.0]), rng.uniform(0, 5) * DAY
        old.episodes.append(_ep(ref.Episode, s, T0 - age))
        new.episodes.append(_ep(Episode, s, T0 - age))
    assert not new.RANK_ON_RAW
    for t in range(200):
        now = T0 + t * 1800.0
        clock.set(lambda now=now: now)
        sims = [rng.uniform(-0.2, 0.6) for _ in range(6)]
        q = rng.random() < 0.7
        a, b = _recall(old, sims, question=q), _recall(new, sims, question=q)
        assert (a[0] is None) == (b[0] is None) and a[1] == b[1] and _state(old) == _state(new)


def _hub():
    h = _fake(Hippocampus)
    h.episodes = [_ep(Episode, 6.0, T0), _ep(Episode, 1.0, T0 - 2 * DAY)]   # hub (strong) and a weak, older fact
    return h


def test_strength_cannot_make_the_winner():
    h = _hub()
    ep, _ = _recall(h, [0.20, 0.30])
    assert ep is h.episodes[0]                                   # off: the hub wins on strength
    h = _hub()
    h.RANK_ON_RAW = True
    ep, score = _recall(h, [0.20, 0.30])
    assert ep is h.episodes[1]                                   # on: the raw winner is fetched
    assert ep.strength == 1.5 and ep.uses == 1 and h.episodes[0].strength == 6.0   # use boost unchanged, goes to it
    lf = h.last_fetch
    assert lf["idx"] == 1 and lf["raw_winner"] == 1 and lf["weighted_winner"] == 0 and lf["weighted_only_winner"]
    assert lf["weighted_would_fire"]


def test_strength_still_gates_firing_and_margin_is_raw():
    h = _fake(Hippocampus)
    h.RANK_ON_RAW = True
    h.episodes = [_ep(Episode, 0.05, T0), _ep(Episode, 6.0, T0)]
    ep, _ = _recall(h, [0.50, 0.10])          # raw winner too weak: 0.5 * tanh(0.05) = 0.025 < bar
    assert ep is None and h.episodes[1].uses == 0 and h.episodes[0].uses == 0
    h = _fake(Hippocampus)
    h.RANK_ON_RAW = True
    h.episodes = [_ep(Episode, 6.0, T0), _ep(Episode, 6.0, T0)]
    ep, _ = _recall(h, [0.40, 0.39])          # raw margin 0.01 < MARGIN 0.02: no fetch
    assert ep is None
    ep, _ = _recall(h, [0.40, 0.37])
    assert ep is h.episodes[0]


def test_frozen_rank_raw_changes_nothing():
    h = _hub()
    h.RANK_ON_RAW = h.FREEZE_USE = True
    before = _state(h)
    ep, _ = _recall(h, [0.20, 0.30])
    assert ep is h.episodes[1] and _state(h) == before


def test_brain_rank_switch(tmp_path):
    from brain.brain import Brain
    from brain.tests.test_brain_key_switch import KW, TALK, StubCortex
    c = StubCortex()
    with pytest.raises(ValueError):
        Brain(str(tmp_path / "x"), cortex=c, **KW, rank="weighted")
    b = Brain(str(tmp_path / "r"), cortex=c, **KW, key="S_all", rank="raw")
    assert b.hippocampus.RANK_ON_RAW and b.hippocampus.key_name == "S_all"
    for t in TALK:
        b.hear(t)
    b.save()
    b2 = Brain(str(tmp_path / "r"), cortex=c, **KW)                  # not saved: pass it each time
    assert not b2.hippocampus.RANK_ON_RAW
    off = Brain(str(tmp_path / "o"), cortex=c, **KW)
    assert not off.hippocampus.RANK_ON_RAW and Hippocampus.RANK_ON_RAW is False
