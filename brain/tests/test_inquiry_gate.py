"""Brain(inquiry=...) novelty gate: off is unchanged; on without a lesson is unchanged; the gate fires on a coined word."""
import random

import numpy as np
import pytest
import torch

from brain.brain import BIRTH, Brain
from brain.cortex import Cortex, resolve_model
from brain.novelty import novelty, word_surprisal

LESSON = "My friend keeps a zorbit in her kitchen."
GUIDED = "I don't know that word. What is it?"


@pytest.fixture(scope="module")
def cortex():
    return Cortex(resolve_model("0.1b"))


def _reply(cortex, home, text, **kw):
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    b = Brain(str(home), cortex=cortex, **kw)
    return b, b.hear(text)


def test_word_surprisal_marks_the_coined_word(cortex):
    _, base = cortex.read(BIRTH)
    w = word_surprisal(cortex, base, "My sister works as a plinketer.")
    assert max(w, key=lambda x: x[1])[0].startswith("plinketer")
    assert novelty(cortex, base, "My sister works as a plinketer.") > novelty(cortex, base, "My sister works as a nurse.")


def test_inquiry_without_lesson_is_unchanged(cortex, tmp_path):
    _, (r0, i0) = _reply(cortex, tmp_path / "a", "My sister works as a plinketer.")
    _, (r1, i1) = _reply(cortex, tmp_path / "b", "My sister works as a plinketer.", inquiry=0.0)
    assert r0 == r1 and "inquiry" not in i1


def test_gate_fires_on_coined_not_common(cortex, tmp_path):
    b = Brain(str(tmp_path / "c"), cortex=cortex, inquiry=12.4)
    b.live_inquiry(LESSON, GUIDED)
    assert b.inquiry_episode() is not None
    _, info_n = b.hear("My sister works as a plinketer.")
    b2 = Brain(str(tmp_path / "d"), cortex=cortex, inquiry=12.4)
    b2.live_inquiry(LESSON, GUIDED)
    _, info_c = b2.hear("My sister works as a nurse.")
    assert info_n.get("inquiry") is True
    assert not info_c.get("inquiry")


def test_inquiry_episode_is_not_ordinary_recall(cortex, tmp_path):
    b = Brain(str(tmp_path / "e"), cortex=cortex)          # gate OFF: the lesson lives in memory only
    b.live_inquiry(LESSON, GUIDED)
    b.state = [x.clone() for x in b.base]
    _, info = b.hear("My friend keeps a zorbit in her garage.")   # the closest possible cue to the lesson
    assert info["recalled"] is None or "zorbit in her kitchen" not in info["recalled"]


def test_auto_without_size_row_is_off(cortex, tmp_path):
    b = Brain(str(tmp_path / "f"), cortex=cortex, inquiry="auto")      # 0.1B has no INQUIRY_BAR_BY_SIZE row
    assert b.inquiry is None and b.inquiry_episode() is None
    _, (r0, _) = _reply(cortex, tmp_path / "g", "My sister works as a plinketer.")
    _, (r1, _) = _reply(cortex, tmp_path / "h", "My sister works as a plinketer.", inquiry="auto")
    assert r0 == r1


def test_auto_lives_the_lesson_at_birth_only(cortex, tmp_path, monkeypatch):
    monkeypatch.setitem(Brain.INQUIRY_BAR_BY_SIZE, "0.1b", 12.4)
    b = Brain(str(tmp_path / "i"), cortex=cortex, inquiry="auto")
    assert b.inquiry == 12.4 and b.inquiry_episode() is not None
    assert "zorbit" in b.inquiry_episode().text
    assert b.meta["turns"] == 0 and all(torch.equal(x, y) for x, y in zip(b.state, b.base))
    assert sum("zorbit" in e.text for e in b.hippocampus.episodes) == 1
    n = len(b.hippocampus.episodes)
    b.save(); b.release()
    b2 = Brain(str(tmp_path / "i"), cortex=cortex, inquiry="auto", sleep_on_restart=False)   # reload: not lived again
    assert len(b2.hippocampus.episodes) == n and b2.inquiry_episode() is not None


def test_inquiry_lesson_is_not_replayed_at_night(cortex, tmp_path):
    b = Brain(str(tmp_path / "j"), cortex=cortex)
    n0 = len(b._night_episodes())
    b.live_inquiry(LESSON, GUIDED)
    assert b.inquiry_episode() is not None and len(b.hippocampus.episodes) == n0 + 1
    assert len(b._night_episodes()) == n0 and all("zorbit" not in e.text for e in b._night_episodes())
