"""Weekly eval: strict scoring, weekly rotation, header, sleep arm. No model weights (mock cortex)."""
import sys
import types

import torch
import pytest

try:  # brain.cortex imports the rwkv package at module level; the tests never load a model, so a stub is enough
    import rwkv.model  # noqa: F401
except ImportError:
    for _n in ("rwkv", "rwkv.model", "rwkv.utils"):
        sys.modules[_n] = types.ModuleType(_n)
    sys.modules["rwkv.model"].RWKV = object
    sys.modules["rwkv.utils"].PIPELINE = object

from brain.eval import weekly  # noqa: E402


class _Pipe:
    def __init__(self):
        self.vocab = {}

    def encode(self, text):
        return [self.vocab.setdefault(w, len(self.vocab) + 1) for w in text.split()]

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv[i] for i in ids)

    def sample_logits(self, logits, **kw):
        return int(torch.argmax(logits))


class MockCortex:
    """Word-level pipe; read returns the prompt's token ids as 'logits'; speak replies via `replier(prompt)`."""

    def __init__(self, replier=None):
        self.pipe = _Pipe()
        self.replier = replier or (lambda prompt: "You said: " + prompt.split("\n\n")[0].replace("User: ", ""))

    def read(self, text, state=None):
        st = [x.clone() for x in state] if state is not None else [torch.zeros(4), torch.zeros(1, 2, 2), torch.zeros(4)]
        ids = self.pipe.encode(text)
        st[1] = st[1] + float(sum(ids) % 7) * 0.1 + torch.arange(4.0).view(1, 2, 2) * 0.01 * len(ids)
        return torch.tensor([float(i) for i in ids]), st

    def speak(self, logits, state, **kw):
        prompt = self.pipe.decode([int(i) for i in logits])
        return self.replier(prompt), state

    def matrix_delta(self, state, base):
        return [(state[i] - base[i]).clone() if i % 3 == 1 else None for i in range(len(base))]

    @staticmethod
    def key_of(delta, layer=0):
        return delta[layer * 3 + 1].flatten().float()

    @staticmethod
    def add_delta(state, delta, alpha):
        return [x + alpha * d if d is not None else x.clone() for x, d in zip(state, delta)]


class FakePlastic:
    def __init__(self, cortex):
        self.cortex = cortex
        self.calls = 0

    def __call__(self, ids, state=None):
        self.calls += 1
        logits = torch.zeros(len(ids), 8)
        return logits, state if state is not None else [torch.zeros(4), torch.zeros(1, 2, 2), torch.zeros(4)]


@pytest.mark.parametrize("reply,key,expected", [
    ("Pixel", "pixel", True), ("The cat is called Pixel.", "pixel", True), ("My name is Nova.", "nova", True),
    ("Nova.", "nova", False), ("Nova, Nova.", "nova", False), ("", "pixel", False), ("   ", "pixel", False),
    ("pixel pixel pixel", "pixel", False), ("I don't know.", "pixel", False),
])
def test_counts_as_recall(reply, key, expected):
    assert weekly.counts_as_recall(reply, key) is expected


def test_rotation():
    a, b = weekly.draw_rotating("2026-W45"), weekly.draw_rotating("2026-W45")
    assert a == b and len(a) == 6
    assert {f[2] for f in weekly.draw_rotating("2026-W45")} != {f[2] for f in weekly.draw_rotating("2026-W46")}
    assert len(weekly.POOL) >= 18
    fixed_keys = {k for _, _, k in weekly.FIXED}
    for wk in ("2026-W45", "2026-W46", "2027-W01"):
        facts = weekly.scored_facts(wk)
        assert len(facts) == 12 and len({f[2] for f in facts}) == 12
        assert fixed_keys == {f[2] for f in facts if not f[3]}
        assert not fixed_keys & {f[2] for f in facts if f[3]}


def _patch(monkeypatch, events):
    monkeypatch.setattr(weekly, "PlasticCortex", FakePlastic)
    monkeypatch.setattr(weekly, "sleep", lambda plastic, base, eps, generic, steps=0, **kw:
                        (events.append(("sleep", list(eps), steps)), [(1.0, 0.1), (0.5, 0.05)])[1])
    instances = []

    class Rec(weekly.Brain):
        def __init__(self, home, **kw):
            super().__init__(home, **kw)
            instances.append(self)

        def hear(self, text):
            events.append(("hear", self.home, self.hippocampus.enabled, text))
            return super().hear(text)

    monkeypatch.setattr(weekly, "Brain", Rec)
    return instances


def test_header_and_sleep_arm(monkeypatch, tmp_path):
    events = []
    instances = _patch(monkeypatch, events)
    out = tmp_path / "t.md"
    scores, text = weekly.run_week(MockCortex(), str(out), "2026-W45", sleep_steps=7, home_root=str(tmp_path / "bd"))
    head = out.read_text().splitlines()[:12]
    for line in ("- intact: ", "- hippocampus off: ", "- sleep + hippocampus off: ",
                 "- difference (intact − hippocampus off): ", "- difference (sleep − hippocampus off): "):
        assert any(l.startswith(line) for l in head), line
    assert any(l.startswith("- intact: ") and l.endswith(" / 12") for l in head)
    assert "(this week's draw)" in text
    assert "7 steps" in text and "replay loss 1.000" in text

    sl = [i for i, e in enumerate(events) if e[0] == "sleep"]
    assert len(sl) == 1
    i = sl[0]
    home = lambda e: e[1].endswith("_weekly_sleep")
    before = [e for e in events[:i] if e[0] == "hear" and home(e)]
    after = [e for e in events[i + 1:] if e[0] == "hear" and home(e)]
    assert before and after
    assert all(e[2] for e in before) and not any(e[2] for e in after)   # session 1 ON, session 2 OFF
    assert events[i][2] == 7
    s1_brain = next(b for b in instances if b.home.endswith("_weekly_sleep"))
    stored = [ep.text for ep in s1_brain.hippocampus.episodes]
    assert [t for t, _ in events[i][1]] == stored
    s2_brain = [b for b in instances if b.home.endswith("_weekly_sleep")][-1]
    assert isinstance(s2_brain.cortex, weekly._PlasticReader) and not s2_brain.hippocampus.enabled


def test_degenerate_never_earns_check(monkeypatch, tmp_path):
    _patch(monkeypatch, [])
    bad = ["Nova.", "Nova, Nova.", "", "pixel pixel pixel volvo volvo volvo"]
    n = {"i": 0}

    def replier(prompt):
        n["i"] += 1
        return bad[n["i"] % len(bad)] if "?" in prompt else "ok"

    out = tmp_path / "t.md"
    scores, text = weekly.run_week(MockCortex(replier), str(out), "2026-W45", home_root=str(tmp_path / "bd"))
    assert "✅" not in text
    assert all(h == 0 for h, _ in scores.values())
