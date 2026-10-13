"""A question is never stored as a fact (Stage-2 HANDOFF item 18a, 2026-10-13).

The defect (dev seed 7, stage2 `infer` shard): the bare yes/no `does crow chase moth` reached the store as an
ASSERTION. `surprise_production_organ.extract_assertion` strips "does" as a function word, the D2 surprise fired
against the held "crow chase wren", and the reconsolidation organ rewrote that fact in place, so the next turn
volunteered "the crow chases the moth". `_maybe_acquire`'s B3 path had the same exposure for negative / connective-led
yes/no questions ("doesn't the crow chase the moth" was TAUGHT as a NEGATE fact; "can a robin breathe" as
[can, robin, breathe]).

The fix corrects the HOST acquisition scaffold only (`brain_chat_tui.is_interrogative_turn` /
`interrogative_blocks_store`): interrogatives are not acquired or reconsolidated; declaratives are byte-identical;
`BRAIN_ACQUIRE_QUESTIONS=1` restores the pre-fix behaviour. The unit tests use a fake `inner` (no brain); the handler
test builds the tiny-demo brain through the real `/api/brain-chat` handler on numpy (~6.3 GB RSS per session).
"""
from __future__ import annotations

import pytest

QUESTIONS = [
    "does crow chase moth", "does crow chase moth?", "Does the crow chase the moth", "do dogs bark",
    "is the wolf big", "did the cat eat the fish", "can crows chase moths", "will the dog eat grass",
    "could crow chase moth", "has the crow chased the moth", "doesn't the crow chase the moth",
    "don't dogs eat meat", "isn't the dog eating meat", "doesn’t the crow chase the moth",
    "so does crow chase moth", "and is the dog big", "can a robin breathe", "what does the fox chase",
    "where does the deer sleep", "the fox chases what", "who chases the moth", "crow chase moth?",
]
DECLARATIVES = [
    "the crow chases the moth", "crow chase moth", "the wolf chase the rabbit", "wolf chase rabbit",
    "the dog does not eat grass", "the dog doesn't eat grass", "cats eat fish.", "the heron hunts the frog",
    "the big bad wolf chase the small rabbit", "dog chase mouse", "hello there", "tell me about the ocean please",
    "the wolf chase", "", "the eel eats the grass", "a cat never eats grass",
]


def test_predicate_reads_form():
    from research.runners.brain_chat_tui import is_interrogative_turn
    for q in QUESTIONS:
        assert is_interrogative_turn(q), q
    for d in DECLARATIVES:
        assert not is_interrogative_turn(d), d


def test_escape_flag(monkeypatch):
    from research.runners.brain_chat_tui import interrogative_blocks_store
    monkeypatch.delenv("BRAIN_ACQUIRE_QUESTIONS", raising=False)
    assert interrogative_blocks_store("does crow chase moth")
    monkeypatch.setenv("BRAIN_ACQUIRE_QUESTIONS", "1")
    assert not interrogative_blocks_store("does crow chase moth")
    monkeypatch.setenv("BRAIN_ACQUIRE_QUESTIONS", "0")
    assert interrogative_blocks_store("does crow chase moth")


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


def _acquire_all(msgs):
    cb = _chat()
    out = [(m, cb._maybe_acquire(m), cb._is_acquisition_candidate(m)) for m in msgs]
    return out, list(cb.inner.heard)


@pytest.mark.parametrize("b3", ["1", "0"])
def test_questions_never_acquired(monkeypatch, b3):
    monkeypatch.setenv("BRAIN_NONCONTRADICTION_GATE", b3)
    monkeypatch.delenv("BRAIN_ACQUIRE_QUESTIONS", raising=False)
    out, heard = _acquire_all(QUESTIONS)
    assert heard == [], heard
    for m, acq, cand in out:
        assert acq is None and cand is False, (b3, m, acq, cand)


@pytest.mark.parametrize("b3", ["1", "0"])
def test_declaratives_acquired_byte_identically(monkeypatch, b3):
    """The fix touches no declarative: same return, same predicate, same `inner.hear` calls (text + polarity) as the
    pre-fix behaviour (`BRAIN_ACQUIRE_QUESTIONS=1`)."""
    monkeypatch.setenv("BRAIN_NONCONTRADICTION_GATE", b3)
    monkeypatch.setenv("BRAIN_ACQUIRE_QUESTIONS", "1")
    before = _acquire_all(DECLARATIVES)
    monkeypatch.delenv("BRAIN_ACQUIRE_QUESTIONS", raising=False)
    after = _acquire_all(DECLARATIVES)
    assert after == before
    assert any(acq is not None for _, acq, _ in after[0])   # the corpus does exercise acquisition


def test_escape_restores_prefix_question_acquisition(monkeypatch):
    """Pins what the fix removes on the default (B3) path: pre-fix, these questions were TAUGHT."""
    monkeypatch.setenv("BRAIN_NONCONTRADICTION_GATE", "1")
    monkeypatch.setenv("BRAIN_ACQUIRE_QUESTIONS", "1")
    _, heard = _acquire_all(["doesn't the crow chase the moth", "so does crow chase moth", "can a robin breathe"])
    assert heard == [("crow chase moth", "NEGATE"), ("crow chase moth", "AFFIRM"), ("can robin breathe", "AFFIRM")]


def test_surprise_extractor_still_reads_the_bare_question():
    """Documents WHY the reconsolidation guard is needed: the D2 extractor (unchanged) still reads the bare yes/no
    as an (agent, action, patient) triple, so only the store-side guard keeps it out of memory."""
    import research.runners.surprise_production_organ as SO
    from research.runners.brain_chat_tui import interrogative_blocks_store
    assert SO.extract_assertion("does crow chase moth") == ("crow", "chase", "moth")
    assert interrogative_blocks_store("does crow chase moth")
    assert SO.extract_assertion("the crow chases the moth") == ("crow", "chases", "moth")
    assert not interrogative_blocks_store("the crow chases the moth")


# ── the production handler (tiny-demo brain, stub renderer, numpy, dev seed 7) ──────────────────────────────────
# One tiny-demo session peaks ~6.4 GB RSS and a second session in the same process does not fit under a 9 GB memcap
# (measured 2026-10-13: OOM-killed at 9 GB), so each arm is its own test with ONE session. Run each alone:
#   tools/memcap.sh 9 -- env SIM_BACKEND=numpy .venv/bin/python -m pytest tests/test_interrogative_no_acquire.py -k <arm>
# IQ_HANDLER_DUMP_DIR=<dir> writes each arm's per-turn read-out to <dir>/<arm>.json (for the cross-arm comparison).
@pytest.fixture
def client(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    pytest.importorskip("numpy")
    monkeypatch.setenv("SIM_BACKEND", "numpy")
    monkeypatch.setenv("BRAIN_CHAT_SEED", "7")          # a dev seed, never a gate seed
    for k in ("BRAIN_SURPRISE", "BRAIN_RECONSOLIDATION", "BRAIN_RECONSOLIDATION_LESION", "BRAIN_SURPRISE_LESION",
              "BRAIN_ACQUIRE_QUESTIONS"):
        monkeypatch.delenv(k, raising=False)
    try:
        import research.runners.brain_chat_tui  # noqa: F401
    except Exception as e:
        pytest.skip(f"brain_chat_tui not importable here: {e}")
    from fastapi.testclient import TestClient
    from webapp.server import app
    return TestClient(app)


# the stage-2 `infer` group's teach turns, a declarative contradiction, then HANDOFF 18a's bare yes/no and a recall
_HANDLER_TURNS = [
    "the heron hunts the frog", "the crow chases the wren", "the wren chases the moth",   # 0-2 teach
    "dog chase mouse", "what does the dog chase",                                         # 3-4 declarative revision
    "does crow chase moth", "what does the crow chase",                                   # 5-6 the repro
]


def _ans(turn):
    return str(turn.get("answer") or "").lower()


def _rc(turn):
    return turn.get("reconsolidation") or {}


def _run_arm(client, arm):
    import json
    import os
    sess = "pytest-iq-%s" % arm
    out = []
    try:
        for m in _HANDLER_TURNS:
            r = client.post("/api/brain-chat", json={"session": sess, "brain": "tiny-demo", "renderer": "stub",
                                                     "message": m})
            assert r.status_code == 200, r.text
            out.append(r.json())
    finally:
        client.post("/api/brain-chat/reset", json={"session": sess, "brain": "tiny-demo", "renderer": "stub"})
    dump = os.environ.get("IQ_HANDLER_DUMP_DIR")
    if dump:
        os.makedirs(dump, exist_ok=True)
        rows = [{"message": m, "answer": t.get("answer"), "abstained": t.get("abstained"),
                 "reconsolidation": t.get("reconsolidation"),
                 "surprised": (t.get("surprise") or {}).get("surprised")} for m, t in zip(_HANDLER_TURNS, out)]
        with open(os.path.join(dump, "%s.json" % arm), "w") as fh:
            json.dump(rows, fh, indent=1, default=str)
    return out


@pytest.mark.slow
def test_handler_fixed_question_never_rewrites(client):
    t = _run_arm(client, "fixed")
    # declaratives: an ordinary contradicting assertion still revises the held fact
    assert _rc(t[3]).get("action") == "rewrite", _rc(t[3])
    assert "mouse" in _ans(t[4]), t[4].get("answer")
    # the bare yes/no opens no store write, and the held fact survives
    assert _rc(t[5]).get("action") != "rewrite", _rc(t[5])
    assert _rc(t[5]).get("reason") == "interrogative_turn", _rc(t[5])
    assert "revised it in place" not in _ans(t[5]), t[5].get("answer")
    assert "wren" in _ans(t[6]) and "moth" not in _ans(t[6]), t[6].get("answer")


@pytest.mark.slow
def test_handler_escape_reproduces_the_defect(client, monkeypatch):
    monkeypatch.setenv("BRAIN_ACQUIRE_QUESTIONS", "1")
    t = _run_arm(client, "escape")
    assert _rc(t[3]).get("action") == "rewrite", _rc(t[3])
    assert _rc(t[5]).get("action") == "rewrite", _rc(t[5])         # the question rewrote crow->wren ...
    assert "moth" in _ans(t[6]), t[6].get("answer")                 # ... into crow->moth (HANDOFF 18a)
