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
