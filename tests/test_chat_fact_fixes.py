"""Branch research/chat-fact-fixes (2026-09-30): the two defects the Stage-1 exit packet logged on 2026-09-29
(research/FAILURE_LOG.md).

(1) `ChatBrain._maybe_acquire` stored a listener-directed REQUEST as an SVO fact: "tell me about the sky" was taught
    as [tell, about, sky] (Stage-1 dev smoke, group `lived`, dev seed 7). A request, like a question, is not an
    assertion; it must never reach `inner.hear`. Every genuine assertion must still be acquired exactly as before.
(2) The corpus-learned production organs degraded SILENTLY without data/corpus/ (a git worktree, a pool revision):
    every turn still answered and nothing in the reply said the brain was degraded. The server now records it on
    every /api/brain-chat reply (`corpus_present: false` + `corpus_missing`) and refuses to build under
    BRAIN_REQUIRE_CORPUS=1; with the corpus present the reply is untouched (byte-identical).
"""
from __future__ import annotations

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


class _FakeInner:
    def __init__(self):
        self.heard = []

    def hear(self, text, polarity="AFFIRM"):
        self.heard.append((text, polarity))


def _chat():
    from research.runners.brain_chat_tui import ChatBrain
    cb = object.__new__(ChatBrain)
    cb.inner = _FakeInner()
    cb._refresh_facts = lambda: None
    return cb


# Requests the B3 (default) path used to ACQUIRE: their function-word-stripped content is exactly three tokens.
REQUESTS_B3 = [
    "tell me about the sky",            # the Stage-1 dev-smoke case -> was [tell, about, sky]
    "Tell me about the sky.",
    "tell me about wolves",
    "tell me about the ocean please",   # an existing probe in tests/test_open_ended_generate_route.py
    "now tell me about dogs",
    "show me the red ball",             # -> was [show, red, ball]
    "give me the big apple",            # -> was [give, big, apple]
    "talk about the moon",              # -> was [talk, about, moon]
    "please remember the red ball",     # -> was [remember, red, ball]
    "can you describe the moon",        # -> was [can, describe, moon]
    "could you tell the story",         # -> was [could, tell, story]
    "can dogs fly",                     # a modal-inverted question without '?' -> was [can, dogs, fly]
    "let's talk about dogs",
    "don't tell me about the sky",      # a negative imperative -> was stored as NEGATE [tell, about, sky]
    "never show me the red ball",       # -> was NEGATE [show, red, ball]
]
# Requests the legacy path (BRAIN_NONCONTRADICTION_GATE=0: exactly three whitespace tokens) used to ACQUIRE.
REQUESTS_LEGACY = ["tell me more", "show us cats", "give me fish", "can dogs fly", "please remember dogs",
                   "explain the sky"]
# Genuine assertions: acquired before the fix and after it, with the SAME triple and polarity.
ASSERTIONS_B3 = {
    "the wolf hunts the deer": ("wolf hunt deer", "AFFIRM"),
    "the dog does not eat grass": ("dog eat grass", "NEGATE"),
    "cats eat fish.": ("cats eat fish", "AFFIRM"),
    "the cat sat on the mat": ("cat sat mat", "AFFIRM"),
    "the teacher tells the story": ("teacher tell story", "AFFIRM"),
    "mom shows the picture": ("mom show picture", "AFFIRM"),
    "dogs tell tales": ("dogs tell tales", "AFFIRM"),
}
ASSERTIONS_LEGACY = {"wolf chase rabbit": ("wolf chase rabbit", "AFFIRM"), "cats eat fish.": ("cats eat fish", "AFFIRM")}


@pytest.mark.parametrize("q", REQUESTS_B3)
def test_request_is_not_acquired_b3(monkeypatch, q):
    monkeypatch.delenv("BRAIN_NONCONTRADICTION_GATE", raising=False)      # the production default (B3 on)
    cb = _chat()
    assert cb._maybe_acquire(q) is None, q
    assert cb.inner.heard == [], "a request reached inner.hear: %r -> %r" % (q, cb.inner.heard)
    assert cb._is_acquisition_candidate(q) is False


@pytest.mark.parametrize("q", REQUESTS_LEGACY)
def test_request_is_not_acquired_legacy(monkeypatch, q):
    monkeypatch.setenv("BRAIN_NONCONTRADICTION_GATE", "0")
    cb = _chat()
    assert cb._maybe_acquire(q) is None, q
    assert cb.inner.heard == []
    assert cb._is_acquisition_candidate(q) is False


@pytest.mark.parametrize("q,expected", sorted(ASSERTIONS_B3.items()))
def test_assertion_still_acquired_b3(monkeypatch, q, expected):
    monkeypatch.delenv("BRAIN_NONCONTRADICTION_GATE", raising=False)
    cb = _chat()
    assert cb._maybe_acquire(q) is not None, q
    assert cb.inner.heard == [expected]
    assert cb._is_acquisition_candidate(q) is True


@pytest.mark.parametrize("q,expected", sorted(ASSERTIONS_LEGACY.items()))
def test_assertion_still_acquired_legacy(monkeypatch, q, expected):
    monkeypatch.setenv("BRAIN_NONCONTRADICTION_GATE", "0")
    cb = _chat()
    assert cb._maybe_acquire(q) is not None, q
    assert cb.inner.heard == [expected]


def test_request_shape_predicate_is_narrow():
    """The predicate rejects the request shapes and nothing a declarative SVO assertion starts with."""
    from research.runners.brain_chat_tui import _is_request_speech_act as R
    for q in REQUESTS_B3 + REQUESTS_LEGACY:
        assert R(q), q
    for q in list(ASSERTIONS_B3) + list(ASSERTIONS_LEGACY) + [
            "the dog and the cat walked in", "that was wonderful, I am so happy", "i like the sky",
            "the book is about dogs", "mom told me about dogs", "he tells us stories", "the wolf hunts the deer",
            "the dog does not eat grass", "dogs never eat grass", "the cat doesn't chase dogs", ""]:
        assert not R(q), q


# ── (2) corpus guard ─────────────────────────────────────────────────────────────────────────────────────────────

def _fake_root(tmp_path, files):
    d = tmp_path / "data" / "corpus"
    d.mkdir(parents=True)
    for f in files:
        (d / f).write_text("x")
    return str(tmp_path)


def test_corpus_status_names_the_missing_files(tmp_path):
    from webapp import corpus_guard as G
    root = _fake_root(tmp_path, [])
    st = G.corpus_status(root)
    assert st["corpus_present"] is False and st["corpus_missing"] == list(G.CORPUS_FILES)
    root2 = _fake_root(tmp_path / "ok", G.CORPUS_FILES)
    assert G.corpus_status(root2) == {"corpus_present": True, "corpus_missing": []}


def test_degraded_marker_only_when_missing(tmp_path):
    pytest.importorskip("fastapi")
    from starlette.responses import JSONResponse
    from webapp import corpus_guard as G
    body = {"answer": "wolf hunt deer.", "abstained": False, "recalled_svo": ["wolf", "hunt", "deer"]}
    ok = JSONResponse(body)
    same = G.mark_response(ok, root=_fake_root(tmp_path / "ok", G.CORPUS_FILES))
    assert same is ok and same.body == JSONResponse(body).body          # untouched -> byte-identical
    bad = G.mark_response(JSONResponse(body, status_code=200), root=_fake_root(tmp_path / "bad", []))
    d = json.loads(bad.body)
    assert d["corpus_present"] is False and d["corpus_missing"] == list(G.CORPUS_FILES)
    assert {k: d[k] for k in body} == body and bad.status_code == 200
    assert int(bad.headers["content-length"]) == len(bad.body)


def test_require_corpus_refuses_loudly(tmp_path, monkeypatch, capsys):
    from webapp import corpus_guard as G
    root = _fake_root(tmp_path, [])
    monkeypatch.setenv("BRAIN_REQUIRE_CORPUS", "1")
    with pytest.raises(G.CorpusMissingError) as ei:
        G.check_at_build(root=root)
    assert "tinystories.txt" in str(ei.value)
    monkeypatch.delenv("BRAIN_REQUIRE_CORPUS")
    G._WARNED.clear()
    st = G.check_at_build(root=root)                                        # default: builds, but LOUDLY
    assert st["corpus_present"] is False
    assert "CORPUS MISSING" in capsys.readouterr().err


def test_handler_routes_every_reply_through_the_marker():
    """`/api/brain-chat` has ONE exit (`return brain_reply(...)`); it must pass through `mark_response`, and the brain
    build must run `check_at_build` first (source-level pin: a real build takes minutes)."""
    src = open(os.path.join(REPO, "webapp", "server.py"), encoding="utf-8").read()
    i = src.index("def brain_chat(req: BrainChatRequest)")
    j = src.index("\ndef brain_reply(", i)
    handler = src[i:j]
    assert "_corpus_guard.mark_response(brain_reply(chat, req, source, cache_key))" in handler
    assert "return brain_reply(" not in handler
    k = src.index("def _build_chat_brain(brain: str, renderer: str):")
    assert "_corpus_guard.check_at_build()" in src[k:k + 1500]
