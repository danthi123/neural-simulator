# Capability: Self and identity
# Record: brain/research/2026-11-25-cascade-screen-prereg.md (prereg); brain/research/2026-11-24-routing-cascade-research.md
# Expected-fail: none (switch plumbing on the stub cortex; no model weights)
"""Brain(own_decay): older own replies fade from the conversation state by beta per later turn; the reply just spoken
stays at full weight; beta 1 is the split read at full gain (own_gain=1.0) bit for bit; a state reset restarts the
no-reply track; default OFF."""
import pytest
import torch

from brain.brain import Brain
from brain.tests.test_brain_key_switch import KW, StubCortex

TURNS = [("Hello there.", "Hi, I'm glad you came."), ("What is my cat called?", "My cat is called Pixel."),
         ("What car do I drive?", "I drive a red Volvo."), ("Where do I live?", "I live in Leeds.")]


def _run(b, turns=TURNS):
    st = [s.clone() for s in b.state]
    out = []
    for text, reply in turns:
        st = b.commit_exchange(text, reply, st)
        out.append(st)
    return out


def test_default_off_and_exclusions(tmp_path):
    c = StubCortex()
    assert Brain(str(tmp_path / "o"), cortex=c, **KW).own_decay is None
    with pytest.raises(ValueError):
        Brain(str(tmp_path / "a"), cortex=c, **KW, own_decay=0.5, own_gain=0.5)


def test_beta_one_is_the_full_gain_split_read(tmp_path):
    c = StubCortex()
    a = _run(Brain(str(tmp_path / "a"), cortex=c, **KW, own_gain=1.0))
    d = _run(Brain(str(tmp_path / "d"), cortex=c, **KW, own_decay=1.0))
    for x, y in zip(a, d):
        assert all(torch.allclose(p, q) for p, q in zip(x, y))   # own_gain 1.0 is su + (sp - su)


def test_first_turn_is_unchanged_and_later_turns_follow_the_recursion(tmp_path):
    c = StubCortex()
    beta = 0.5
    b = Brain(str(tmp_path / "d"), cortex=c, **KW, own_decay=beta)
    full = _run(Brain(str(tmp_path / "a"), cortex=c, **KW, own_gain=1.0))
    got = _run(b)
    assert all(torch.allclose(p, q) for p, q in zip(full[0], got[0]))     # no earlier reply: nothing to fade
    # the recursion by hand: N = no-reply track, X = conversation state
    X = [s.clone() for s in b.base]
    N = [s.clone() for s in b.base]
    for (text, reply), g in zip(TURNS, got):
        su = c.read(f"User: {text}\n\nAssistant:", X)[1]
        sn = c.read(f"User: {text}\n\nAssistant:", N)[1]
        mix = [sn[i] + beta * (su[i] - sn[i]) if i % 3 == 1 else su[i] for i in range(len(su))]
        X = c.read(f" {reply}\n\n", mix)[1]
        N = c.read("\n\n", sn)[1]
        assert all(torch.allclose(p, q) for p, q in zip(X, g))
    assert any(not torch.allclose(p, q) for p, q in zip(full[-1], got[-1]))   # it changes the state


def test_beta_zero_keeps_only_the_last_reply(tmp_path):
    """beta 0: the state is the heard turns alone plus the reply just spoken."""
    c = StubCortex()
    b = Brain(str(tmp_path / "d"), cortex=c, **KW, own_decay=0.0)
    got = _run(b)
    N = [s.clone() for s in b.base]
    for text, reply in TURNS[:-1]:
        N = c.read("\n\n", c.read(f"User: {text}\n\nAssistant:", N)[1])[1]
    sn = c.read(f"User: {TURNS[-1][0]}\n\nAssistant:", N)[1]
    su = c.read(f"User: {TURNS[-1][0]}\n\nAssistant:", got[-2])[1]
    ref = c.read(f" {TURNS[-1][1]}\n\n", [sn[i] if i % 3 == 1 else su[i] for i in range(len(su))])[1]
    assert all(torch.allclose(p, q) for p, q in zip(ref, got[-1]))


def test_a_state_reset_restarts_the_track(tmp_path):
    c = StubCortex()
    b = Brain(str(tmp_path / "d"), cortex=c, **KW, own_decay=0.25)
    b.hear("Hello there.")
    b.hear("What is my cat called?")
    b.state = [x.clone() for x in b.base]          # what run_battery does for an open prompt
    r1, _ = b.hear("What car do I drive?")
    fresh = Brain(str(tmp_path / "f"), cortex=c, **KW, own_decay=0.25)
    r2, _ = fresh.hear("What car do I drive?")
    assert all(torch.equal(p, q) for p, q in zip(b._noown, fresh._noown))
    assert b._noown_of is b.state

