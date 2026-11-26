# Capability: Memory selection (pattern completion and separation)
# Record: brain/research/2026-11-26-routing-screen3-prereg.md (part X); results/2026-11-25-sall-live-dev-read-1.5b.md
# Expected-fail: none (switch plumbing on the stub cortex; no model weights)
"""Brain(recall_skip_permanent): default OFF; when on, a heard turn's recall never fetches a permanent episode, and with
no permanent episode stored it is the same recall as OFF."""
from brain.brain import Brain
from brain.tests.test_brain_key_switch import KW, StubCortex


def _brain(tmp_path, name, **kw):
    return Brain(str(tmp_path / name), cortex=StubCortex(), **KW, **kw)


def test_default_off(tmp_path):
    assert _brain(tmp_path, "o").recall_skip_permanent is False


def test_permanent_never_fetched_when_on(tmp_path):
    seen = {}
    for on in (False, True):
        b = _brain(tmp_path, f"b{on}", recall_skip_permanent=on)
        hc = b.hippocampus
        hc.store(b.cortex, b.base, "What's your name?\n\nAssistant: My name is Nova.", permanent=True, salience=5.0)
        calls = []
        orig = hc.recall

        def spy(*a, **k):
            calls.append(k.get("exclude", ()))
            return orig(*a, **k)
        hc.recall = spy
        b.hear("What's your name?")
        perm = tuple(i for i, e in enumerate(hc.episodes) if e.permanent)
        assert perm, "a permanent episode is stored"
        seen[on] = calls[0]
        if on:
            assert set(perm) <= set(calls[0])
    assert seen[False] == ()


def test_same_as_off_without_permanent(tmp_path):
    outs = []
    for on in (False, True):
        b = _brain(tmp_path, f"n{on}", recall_skip_permanent=on)
        b.hear("My cat is called Pixel.")
        r, info = b.hear("What is my cat called?")
        outs.append((r, info.get("recalled"), info.get("similarity")))
    assert outs[0] == outs[1]


def test_reply_level_fetch_changes(tmp_path):
    """The permanent episode OFF fetches for its own cue is not fetched ON (the switch changes the fetch)."""
    got = {}
    for on in (False, True):
        b = _brain(tmp_path, f"f{on}", recall_skip_permanent=on)
        b.hippocampus.store(b.cortex, b.base, "User: What's your name?\n\nAssistant: My name is Nova.", permanent=True,
                            salience=5.0)
        b.hippocampus.store(b.cortex, b.base, "User: My cat is called Pixel.")
        hc = b.hippocampus
        hc.FAMILIARITY = hc.FAMILIARITY_SPONTANEOUS = -10.0   # every best episode fires on the stub
        hc.MARGIN = -10.0
        _, info = b.hear("What's your name?")
        got[on] = info.get("recalled") or ""
    assert "Nova" in got[False]
    assert "Nova" not in got[True]
