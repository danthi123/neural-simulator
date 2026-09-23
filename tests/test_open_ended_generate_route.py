"""BRAIN_OPEN_ENDED_GENERATE_ROUTE (default-OFF) -- webapp/server.py::_open_ended_generate_route.

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
    m = re.search(r'if \(os\.environ\.get\("BRAIN_OPEN_ENDED", "0"\)\.strip\(\)\.lower\(\) in '
                  r'\("1", "true", "on", "yes"\)\s*\n\s*and not _open_ended_generate_route\(chat, msg\)\):', src)
    assert m is not None, "the route must be AND-ed AFTER the BRAIN_OPEN_ENDED check (short-circuit on default path)"
