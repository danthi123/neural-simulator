"""BRAIN_OPEN_ENDED_GENERATE_ROUTE + BRAIN_OPEN_ENDED_ACQUIRE_ROUTE (both default-OFF) -- webapp/server.py
::_open_ended_generate_route / _open_ended_acquire_route / _open_ended_brain_route.

Pins (a) flag OFF -> returns False WITHOUT touching `chat` (so the BRAIN_OPEN_ENDED free-talk block runs exactly as
before), (b) flag ON -> routes ONLY an explicit generation prompt (the ChatBrain's own `_parse_open_ended`), and (c)
the caller in brain_chat is short-circuited behind BRAIN_OPEN_ENDED (never evaluated on the default path).
Lane research/open-ended-production-turn-lb (2026-09-23).
"""
from __future__ import annotations

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Untouchable:
    def __getattr__(self, name):
        raise AssertionError("chat.%s touched while the route flag is OFF" % name)


class _FakeChat:
    def __init__(self, sentinel):
        self._s = sentinel

    def _parse_open_ended(self, msg):
        return ("dog", "chase") if msg.startswith("what might") else self._s


@pytest.fixture
def S(monkeypatch):
    pytest.importorskip("fastapi")
    import webapp.server as S
    return S


def test_flag_off_never_touches_chat(S, monkeypatch):
    monkeypatch.delenv("BRAIN_OPEN_ENDED_GENERATE_ROUTE", raising=False)
    assert S._open_ended_generate_route(_Untouchable(), "what might a dog chase") is False
    monkeypatch.setenv("BRAIN_OPEN_ENDED_GENERATE_ROUTE", "0")
    assert S._open_ended_generate_route(_Untouchable(), "what might a dog chase") is False


def test_flag_on_routes_only_generation_prompts(S, monkeypatch):
    from research.runners.brain_chat_tui import _NOT_OPEN_ENDED
    monkeypatch.setenv("BRAIN_OPEN_ENDED_GENERATE_ROUTE", "1")
    chat = _FakeChat(_NOT_OPEN_ENDED)
    assert S._open_ended_generate_route(chat, "what might a dog chase") is True
    assert S._open_ended_generate_route(chat, "tell me about canada") is False
    # a broken chat degrades to the unchanged free-talk path, never raises
    assert S._open_ended_generate_route(object(), "what might a dog chase") is False


def test_caller_is_short_circuited_behind_brain_open_ended():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "webapp", "server.py"), encoding="utf-8").read()
    # The ONE clause allowed between the two is BRAIN_OPEN_ENDED_GATED's `and not _oeg_on()` (lane
    # research/open-ended-gated-turn, default OFF): a pure env read, evaluated BEFORE _open_ended_brain_route on purpose
    # so the gated turn never touches the route helper (which may read `chat`). Any other intervening clause, or the
    # route helper moved ahead of the BRAIN_OPEN_ENDED check, still fails this pin.
    m = re.search(r'if \(os\.environ\.get\("BRAIN_OPEN_ENDED", "0"\)\.strip\(\)\.lower\(\) in '
                  r'\("1", "true", "on", "yes"\)[ \t]*\n'
                  r'(?:[ \t]*and not _oeg_on\(\)[ \t]*(?:#[^\n]*)?\n)?'
                  r'[ \t]*and not _open_ended_brain_route\(chat, msg\)\):', src)
    assert m is not None, "the route must be AND-ed AFTER the BRAIN_OPEN_ENDED check (short-circuit on default path)"
    if "def _oeg_on()" in src:
        # the gated-turn reader, when present, must stay a pure env read defaulting OFF (it runs on the default path)
        body = src.split("def _oeg_on()", 1)[1].split("\ndef ", 1)[0]
        assert 'os.environ.get("BRAIN_OPEN_ENDED_GATED", "0")' in body and "chat" not in body


# ── fix round (2026-09-23): the ACQUIRE route -- open-ended mode did not learn from being told ─────────────────────
def test_acquire_flag_off_never_touches_chat(S, monkeypatch):
    monkeypatch.delenv("BRAIN_OPEN_ENDED_ACQUIRE_ROUTE", raising=False)
    monkeypatch.delenv("BRAIN_OPEN_ENDED_GENERATE_ROUTE", raising=False)
    assert S._open_ended_acquire_route(_Untouchable(), "the wolf chase the rabbit") is False
    # BOTH flags off -> the combined route never touches chat either
    assert S._open_ended_brain_route(_Untouchable(), "the wolf chase the rabbit") is False
    assert S._open_ended_brain_route(_Untouchable(), "what might a dog chase") is False


class _FakeAcqChat:
    def _is_acquisition_candidate(self, msg):
        return not msg.endswith("?")


def test_acquire_flag_on_routes_only_assertions(S, monkeypatch):
    monkeypatch.setenv("BRAIN_OPEN_ENDED_ACQUIRE_ROUTE", "1")
    assert S._open_ended_acquire_route(_FakeAcqChat(), "the wolf chase the rabbit") is True
    assert S._open_ended_acquire_route(_FakeAcqChat(), "what does the wolf chase?") is False
    assert S._open_ended_acquire_route(object(), "the wolf chase the rabbit") is False   # broken chat -> free-talk


class _FakeInner:
    def __init__(self):
        self.heard = []

    def hear(self, text, polarity="AFFIRM"):
        self.heard.append((text, polarity))


_ACQ_PROBES = ["the wolf chase the rabbit", "wolf chase rabbit", "the dog does not eat grass",
               "what does the wolf chase", "what does the wolf chase?", "is the wolf big", "hello there",
               "tell me about the ocean please", "the big bad wolf chase the small rabbit", "cats eat fish.",
               "guess", "do dogs bark", "the wolf chase", ""]


@pytest.mark.parametrize("b3", ["1", "0"])
def test_acquisition_candidate_mirrors_maybe_acquire(monkeypatch, b3):
    """`_is_acquisition_candidate` must accept EXACTLY the inputs `_maybe_acquire` tries to teach (B3 on and off),
    and must itself have no side effect (it never calls inner.hear)."""
    monkeypatch.setenv("BRAIN_NONCONTRADICTION_GATE", b3)
    from research.runners.brain_chat_tui import ChatBrain
    cb = object.__new__(ChatBrain)
    cb.inner = _FakeInner()
    cb._refresh_facts = lambda: None
    for q in _ACQ_PROBES:
        n_before = len(cb.inner.heard)
        pred = cb._is_acquisition_candidate(q)
        assert len(cb.inner.heard) == n_before, "predicate must be side-effect-free"
        acquired = cb._maybe_acquire(q) is not None
        assert pred == acquired, (b3, q, pred, acquired)
