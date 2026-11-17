"""Brain(episode=...) (switch, default OFF = "tell"): "lived" stores the user's telling with the brain's own reply."""
import pytest

from brain.brain import Brain
from brain.tests.test_brain_key_switch import KW, StubCortex, _talk


def _told(b):
    return [e.text for e in b.hippocampus.episodes if not getattr(e, "permanent", False)]


def test_default_stores_the_telling_only(tmp_path):
    b = Brain(str(tmp_path / "a"), cortex=StubCortex(), **KW)
    _talk(b)
    told = [t for t in _told(b) if "Pixel" in t or "Volvo" in t]
    assert told and all("Assistant:" not in t for t in told)


def test_lived_stores_the_exchange(tmp_path):
    b = Brain(str(tmp_path / "a"), cortex=StubCortex(), **KW, episode="lived")
    out = _talk(b)
    told = [t for t in _told(b) if t.startswith("User: My cat") or t.startswith("User: I drive")]
    assert told and all("\n\nAssistant: " in t for t in told)
    assert out[0][0].strip() in told[0]


def test_bad_value(tmp_path):
    with pytest.raises(ValueError):
        Brain(str(tmp_path / "a"), cortex=StubCortex(), **KW, episode="both")


def test_lived_key_indexes_by_the_telling(tmp_path):
    from brain.hippocampus import Hippocampus
    a = Brain(str(tmp_path / "a"), cortex=StubCortex(), **KW, episode="lived_key")
    b = Brain(str(tmp_path / "b"), cortex=StubCortex(), **KW, episode="tell")
    _talk(a), _talk(b)
    ea = [e for e in a.hippocampus.episodes if e.text.startswith("User: My cat")][0]
    eb = [e for e in b.hippocampus.episodes if e.text.startswith("User: My cat")][0]
    assert "\n\nAssistant: " in ea.text and "Assistant:" not in eb.text
    import torch
    assert torch.equal(ea.key, eb.key)                  # same index as the telling alone
    assert not all(torch.equal(x, y) for x, y in zip(ea.delta, eb.delta) if x is not None)   # richer content


def test_speaker_contrast_changes_content_not_key(tmp_path):
    import torch
    a = Brain(str(tmp_path / "a"), cortex=StubCortex(), **KW, speaker_contrast=3.0)
    b = Brain(str(tmp_path / "b"), cortex=StubCortex(), **KW)
    _talk(a), _talk(b)
    ea = [e for e in a.hippocampus.episodes if e.text == "User: My cat is called Pixel."][0]
    eb = [e for e in b.hippocampus.episodes if e.text == "User: My cat is called Pixel."][0]
    assert torch.equal(ea.key, eb.key)
    assert not all(torch.equal(x, y) for x, y in zip(ea.delta, eb.delta) if x is not None)
    assert b.speaker_contrast == 0.0


def test_speaker_contrast_with_sall_keeps_routing_and_persists(tmp_path):
    import torch
    from brain.tests.test_key_bar_switch import Stub15
    kw = dict(KW, key="S_all", key_bar="size")
    a = Brain(str(tmp_path / "a"), cortex=Stub15(), **kw, speaker_contrast=3.0)
    b = Brain(str(tmp_path / "b"), cortex=Stub15(), **kw)
    _talk(a), _talk(b)
    q = "What is my cat called?"
    sa = a.hippocampus.similarities(a.cortex, a.base, q)
    sb = b.hippocampus.similarities(b.cortex, b.base, q)
    assert torch.allclose(sa, sb, atol=1e-5)          # routing unchanged
    ea = [e for e in a.hippocampus.episodes if e.text == "User: My cat is called Pixel."][0]
    assert getattr(ea, "route_delta", None) is not None
    a.save()
    a.release()
    c = Brain(str(tmp_path / "a"), cortex=Stub15(), **kw, speaker_contrast=3.0)
    ec = [e for e in c.hippocampus.episodes if e.text == "User: My cat is called Pixel."][0]
    assert all((x is None and y is None) or torch.equal(x, y) for x, y in zip(ec.route_delta, ea.route_delta))
    assert torch.allclose(c.hippocampus.similarities(c.cortex, c.base, q), sb, atol=1e-5)


def test_self_completion_changes_self_content_not_key_or_tellings(tmp_path):
    import torch
    a = Brain(str(tmp_path / "a"), cortex=StubCortex(), **KW, self_completion=True)
    b = Brain(str(tmp_path / "b"), cortex=StubCortex(), **KW, self_completion=False)
    assert b.self_completion is False
    assert Brain(str(tmp_path / "c"), cortex=StubCortex(), **KW).self_completion is True   # default ON since 11-15
    sa = [e for e in a.hippocampus.episodes if e.permanent]
    sb = [e for e in b.hippocampus.episodes if e.permanent]
    assert len(sa) == len(sb) == 3
    for ea, eb in zip(sa, sb):
        assert ea.text == eb.text and torch.equal(ea.key, eb.key)            # same index, same replayed text
        assert not all(torch.equal(x, y) for x, y in zip(ea.delta, eb.delta) if x is not None)   # completion only
        na = torch.sqrt(sum((x.float() ** 2).sum() for x in ea.delta if x is not None))
        nb = torch.sqrt(sum((y.float() ** 2).sum() for y in eb.delta if y is not None))
        assert torch.allclose(na, nb, rtol=1e-2)                             # norm-matched
    _talk(a), _talk(b)
    ta = [e for e in a.hippocampus.episodes if e.text == "User: My cat is called Pixel."][0]
    tb = [e for e in b.hippocampus.episodes if e.text == "User: My cat is called Pixel."][0]
    assert all(torch.equal(x, y) for x, y in zip(ta.delta, tb.delta) if x is not None)   # tellings unchanged


def test_self_completion_with_sall_keeps_routing(tmp_path):
    import torch
    from brain.tests.test_key_bar_switch import Stub15
    kw = dict(KW, key="S_all", key_bar="size")
    a = Brain(str(tmp_path / "a"), cortex=Stub15(), **kw, self_completion=True)
    b = Brain(str(tmp_path / "b"), cortex=Stub15(), **kw, self_completion=False)
    q = "What's your name?"
    assert torch.allclose(a.hippocampus.similarities(a.cortex, a.base, q),
                          b.hippocampus.similarities(b.cortex, b.base, q), atol=1e-5)


def test_self_key_question_keys_self_by_question_only(tmp_path):
    import torch
    a = Brain(str(tmp_path / "a"), cortex=StubCortex(), **KW, self_key="question")
    b = Brain(str(tmp_path / "b"), cortex=StubCortex(), **KW)
    assert b.self_key == "exchange"
    sa = [e for e in a.hippocampus.episodes if e.permanent]
    sb = [e for e in b.hippocampus.episodes if e.permanent]
    assert len(sa) == len(sb) == 3
    for ea, eb in zip(sa, sb):
        q = ea.text.split("User: ", 1)[1].split("\n\nAssistant:")[0]
        _, st = a.raw_cortex.read(f"User: {q}\n\n", a.base)
        assert torch.equal(ea.key, a.raw_cortex.key_of(a.raw_cortex.matrix_delta(st, a.base)).cpu())   # the question's key
        assert not torch.equal(ea.key, eb.key)
        assert ea.text == eb.text and all(torch.equal(x, y) for x, y in zip(ea.delta, eb.delta) if x is not None)
    a.save()
    a.release() if hasattr(a, "release") else None
    r = Brain(str(tmp_path / "a"), cortex=StubCortex(), **KW)     # the key survives a restart (birth-only switch)
    ra = [e for e in r.hippocampus.episodes if e.permanent]
    assert all(torch.equal(x.key, y.key) for x, y in zip(ra, sa))
    with pytest.raises(ValueError):
        Brain(str(tmp_path / "c"), cortex=StubCortex(), **KW, self_key="answer")


def test_sleep_rehearse_guards_replies_to_heard_turns(tmp_path, monkeypatch):
    import brain.brain as bb
    seen = {}
    def fake(plastic, base, eps, generic, **k):
        seen["guard"] = list(k.get("guard", []))
        return []
    monkeypatch.setattr(bb, "replay_into_cortex", fake)
    monkeypatch.setattr(Brain, "_plastic", lambda self: type("P", (), {"parameters": lambda s: []})())
    b = Brain(str(tmp_path / "a"), cortex=StubCortex(), **KW, sleep_rehearse=True)
    b.hear("My cat is called Pixel.")
    b.sleep()
    assert any(g.startswith("User: My cat is called Pixel.\n\nAssistant:") for g in seen["guard"])
    assert not any("What's your name?" in g for g in seen["guard"])          # permanent self episodes not rehearsed
    c = Brain(str(tmp_path / "c"), cortex=StubCortex(), **KW)
    assert c.sleep_rehearse is False
    c.hear("My cat is called Pixel.")
    c.sleep()
    assert seen["guard"] == []


def test_habituation_blocks_recent_ngrams(tmp_path):
    from brain.cortex import Cortex
    c = StubCortex()
    blk = Cortex.ngram_block(c, ["abcdef"], n=4)
    ids = c.pipe.encode("abcdef")
    assert ids[3] in blk[tuple(ids[:3])]
    b = Brain(str(tmp_path / "a"), cortex=c, **KW, habituation=3)
    assert b.habituation == 3
    b.hear("Hello there.")
    b.hear("Hello there.")
    assert len(b._recent_replies) == 2
    assert Brain(str(tmp_path / "b"), cortex=c, **KW).habituation == 0


def test_sleep_rehearse_reply_passes_prefix_reply_pairs(tmp_path, monkeypatch):
    import brain.brain as bb
    seen = {}
    def fake(plastic, base, eps, generic, **k):
        seen["guard"] = list(k.get("guard", []))
        return []
    monkeypatch.setattr(bb, "replay_into_cortex", fake)
    monkeypatch.setattr(Brain, "_plastic", lambda self: type("P", (), {"parameters": lambda s: []})())
    b = Brain(str(tmp_path / "a"), cortex=StubCortex(), **KW, sleep_rehearse="reply")
    b.hear("My cat is called Pixel.")
    b.sleep()
    pairs = [g for g in seen["guard"] if isinstance(g, tuple)]
    assert ("User: My cat is called Pixel.\n\nAssistant:" in [p for p, _ in pairs])
    with pytest.raises(ValueError):
        Brain(str(tmp_path / "c"), cortex=StubCortex(), **KW, sleep_rehearse="all")


def test_sleep_reply_only_kl_masks_the_prefix():
    """A (prefix, reply) guard KL covers only reply positions: an empty prefix equals the whole-text guard, and a
    non-empty prefix changes the losses (the prefix positions leave the KL)."""
    import torch
    import torch.nn as nn
    from brain.sleep import sleep

    class Pipe:
        def encode(self, t):
            return [ord(ch) % 50 for ch in t]

    class Plastic(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.emb, self.out, self.delta = nn.Embedding(50, 8), nn.Linear(8, 50), nn.Linear(8, 50)
            for p in list(self.emb.parameters()) + list(self.out.parameters()):
                p.requires_grad_(False)
            self.cortex, self.enabled = type("C", (), {"pipe": Pipe()})(), True
        def forward(self, ids, state):
            h = self.emb(torch.tensor(ids))
            return self.out(h) + (self.delta(h) if self.enabled else 0), state

    def run(guard):
        return sleep(Plastic(), None, [("abcdefgh", 1.0)], ["hello there"], steps=5, seed=1, guard=guard)
    text = "User: hi\n\nAssistant: yo\n\n"
    assert run([text]) == run([("", text)])
    assert run([text]) != run([("User: hi\n\nAssistant:", " yo\n\n")])


def test_reply_monitor_withholds_copies(tmp_path, monkeypatch):
    from brain.brain import _is_copy
    assert _is_copy("What's the name of your hedgehog?", ["What's the name of my hedgehog?"])
    assert not _is_copy("Your hedgehog is called Quill.", ["What is my hedgehog called?"])
    assert not _is_copy("Hi Sam.", ["Hi Sam."])                          # under 3 words: never a copy
    c = StubCortex()
    b = Brain(str(tmp_path / "a"), cortex=c, **KW, reply_monitor=3)
    drafts = iter(["I went for a long walk by the river.", "I went for a long walk by the river.", "How lovely, where?"])
    monkeypatch.setattr(b.cortex, "speak", lambda lg, st, **k: (next(drafts), st))
    reply, info = b.hear("I went for a long walk by the river.")
    assert reply == "How lovely, where?" and info["withheld"] == 2 and b.monitor_log == [2]
    drafts2 = iter(["How lovely, where?"] * 5)
    monkeypatch.setattr(b.cortex, "speak", lambda lg, st, **k: (next(drafts2), st))
    reply, info = b.hear("Something else entirely today.")                # every draft repeats: the last is kept
    assert reply == "How lovely, where?" and info["withheld"] == 3
    off = Brain(str(tmp_path / "b"), cortex=c, **KW)
    assert off.reply_monitor == 0
    _, info = off.hear("I went for a long walk by the river.")
    assert "withheld" not in info and off.monitor_log == []


def test_own_gain_scales_the_reply_write(tmp_path):
    import torch
    c = StubCortex()
    off = Brain(str(tmp_path / "o"), cortex=c, **KW)
    assert off.own_gain is None
    b = Brain(str(tmp_path / "a"), cortex=c, **KW, own_gain=0.0)
    st = [s.clone() for s in b.state]
    su = b.cortex.read("User: Hello there.\n\nAssistant:", st)[1]
    sp = b.cortex.read(" Hi.\n\n", su)[1]
    out0 = b.commit_exchange("Hello there.", "Hi.", st)
    b.own_gain = 1.0
    out1 = b.commit_exchange("Hello there.", "Hi.", st)
    b.own_gain = 0.5
    outh = b.commit_exchange("Hello there.", "Hi.", st)
    for i in range(len(sp)):
        if i % 3 == 1:
            assert torch.allclose(out0[i], su[i]) and torch.allclose(out1[i], sp[i])
            assert torch.allclose(outh[i], (su[i] + sp[i]) / 2)
        else:
            assert torch.equal(out0[i], sp[i])
    reply, _ = b.hear("Hello there.")      # hear commits through the same path


def test_own_gain_copies_only_spares_ordinary_replies(tmp_path):
    import torch
    c = StubCortex()
    b = Brain(str(tmp_path / "a"), cortex=c, **KW, own_gain=0.0, own_gain_copies=True)
    st = [s.clone() for s in b.state]
    full = b.cortex.read("User: I went for a long walk by the river.\n\nAssistant: How lovely, where?\n\n", st)[1]
    su = b.cortex.read("User: I went for a long walk by the river.\n\nAssistant:", st)[1]
    sp = b.cortex.read(" How lovely, where?\n\n", su)[1]
    ordinary = b.commit_exchange("I went for a long walk by the river.", "How lovely, where?", st)
    copy = b.commit_exchange("I went for a long walk by the river.", "I went for a long walk by the river.", st)
    for i in range(1, len(sp), 3):
        assert torch.allclose(ordinary[i], sp[i]) and torch.allclose(copy[i], su[i])
    assert Brain(str(tmp_path / "b"), cortex=c, **KW).own_gain_copies is False


def test_carry_penalty_keeps_counts_across_turns(tmp_path):
    c = StubCortex()
    b = Brain(str(tmp_path / "a"), cortex=c, **KW, carry_penalty=True)
    b.hear("Hello there.")
    n1 = sum(b._occ.values())
    b.hear("Hello again.")
    assert n1 > 0 and len(b._occ) >= 1 and sum(b._occ.values()) != n1
    off = Brain(str(tmp_path / "b"), cortex=c, **KW)
    off.hear("Hello there.")
    assert off.carry_penalty is False and off._occ == {}


def test_fake_think_prompts_with_the_empty_think_tag(tmp_path):
    c = StubCortex()
    b = Brain(str(tmp_path / "a"), cortex=c, **KW, fake_think=True)
    seen = []
    real = b.cortex.read
    b.cortex.read = lambda t, s=None: (seen.append(t), real(t, s))[1]
    b.cortex.speak = lambda lg, st, **k: (">Hello there, friend.", st)
    reply, _ = b.hear("Hi.")
    assert reply == "Hello there, friend."
    assert any(t.endswith("Assistant: <think></think") for t in seen)
    assert any("Assistant: <think></think>Hello there, friend.\n\n" in t for t in seen)
    off = Brain(str(tmp_path / "b"), cortex=c, **KW)
    assert off.fake_think is False and off._spoken(">x") == ">x"


def test_teacher_context_triple_kl():
    """(prefix, reply, context) guard items (teacher night): an empty context equals the (prefix, reply) pair; a
    context the reference reads first changes the target (recurrent toy, so earlier tokens matter)."""
    import torch
    import torch.nn as nn
    from brain.sleep import sleep

    class Pipe:
        def encode(self, t):
            return [ord(ch) % 50 for ch in t]

    class Plastic(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.emb, self.out, self.delta = nn.Embedding(50, 8), nn.Linear(8, 50), nn.Linear(8, 50)
            for p in list(self.emb.parameters()) + list(self.out.parameters()):
                p.requires_grad_(False)
            self.cortex, self.enabled = type("C", (), {"pipe": Pipe()})(), True
        def forward(self, ids, state):
            h = torch.cumsum(self.emb(torch.tensor(ids)), 0)
            return self.out(h) + (self.delta(h) if self.enabled else 0), state

    def run(guard):
        return sleep(Plastic(), None, [("abcdefgh", 1.0)], ["hello there"], steps=5, seed=1, guard=guard)
    pre, rep = "User: what?\n\nAssistant:", " no idea\n\n"
    assert run([(pre, rep)]) == run([(pre, rep, "")])
    assert run([(pre, rep)]) != run([(pre, rep, "User: My cat is called Pixel.\n\n")])


def test_teacher_context_switch_passes_tellings(tmp_path, monkeypatch):
    import brain.brain as bb
    import brain.near_miss as nm
    seen = {}
    def fake(plastic, base, eps, generic, **k):
        seen["guard"] = list(k.get("guard", []))
        return []
    monkeypatch.setattr(bb, "replay_into_cortex", fake)
    monkeypatch.setattr(Brain, "_plastic", lambda self: type("P", (), {"parameters": lambda s: []})())
    monkeypatch.setattr(nm, "frame_cues", lambda *a, **k: ["What is my horse called?"])
    b = Brain(str(tmp_path / "a"), cortex=StubCortex(), **KW, near_miss=1, near_miss_frame="F", teacher_context=True)
    b.hear("My cat is called Pixel.")
    b.sleep()
    trip = [g for g in seen["guard"] if isinstance(g, tuple) and len(g) == 3]
    assert trip and trip[0][0] == "User: What is my horse called?\n\nAssistant:"
    assert "My cat is called Pixel." in trip[0][2] and "What's your name?" not in trip[0][2]
    off = Brain(str(tmp_path / "b"), cortex=StubCortex(), **KW, near_miss=1, near_miss_frame="F")
    assert off.teacher_context is False
    off.hear("My cat is called Pixel.")
    off.sleep()
    assert all(isinstance(g, str) for g in seen["guard"])


def test_paired_night_holds_own_and_rewrites_of_told_episodes_only(tmp_path, monkeypatch):
    import brain.brain as bb
    import brain.near_miss as nm
    seen, calls = {}, []
    def fake(plastic, base, eps, generic, **k):
        seen["guard"] = list(k.get("guard", []))
        return []
    def cues(cortex, base, text, k=4, frame="N1", seed=0):
        calls.append((text, frame))
        return "What is my cat called?", ["What colour is my cat?"]
    monkeypatch.setattr(bb, "replay_into_cortex", fake)
    monkeypatch.setattr(Brain, "_plastic", lambda self: type("P", (), {"parameters": lambda s: []})())
    monkeypatch.setattr(nm, "near_miss_cues", cues)
    b = Brain(str(tmp_path / "a"), cortex=StubCortex(), **KW, near_miss=1, near_miss_frame="P:NA", teacher_context=True)
    b.hear("My cat is called Pixel.")
    b.sleep()
    assert calls and all(f == "NA" for _, f in calls)
    assert all(t.startswith("User: ") and "Pixel" in t for t, _ in calls)   # told episodes only, no permanent ones
    pre = [g[0] for g in seen["guard"]]
    assert pre == ["User: What is my cat called?\n\nAssistant:", "User: What colour is my cat?\n\nAssistant:"]
    assert all(len(g) == 3 and "My cat is called Pixel." in g[2] for g in seen["guard"])
    import pytest
    with pytest.raises(AssertionError):
        Brain(str(tmp_path / "c"), cortex=StubCortex(), **KW, near_miss=1, near_miss_frame="P:NA")._near_miss_guard()
