"""Sleep inside the live brain: replies go through the consolidated cortex, restart = sleep, adapters persist, lesions.

Most tests use a tiny random RWKV-7-shaped mock (no weights needed). The golden test replays a conversation recorded
with the pre-sleep brain (origin/main c231bf752) and needs the 0.1B weights.
"""
import json
import os
import types

import pytest
import torch

from brain.brain import ADAPTERS, Brain, LiveCortex
from brain.cortex import DEFAULT_MODEL, Cortex
from brain.plastic_cortex import PlasticCortex

V, C, H, F, D, L = 128, 8, 2, 16, 4, 2
N = C // H


def _tiny_z(seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) * 0.5  # noqa: E731
    z = {"emb.weight": r(V, C), "ln_out.weight": torch.ones(C), "ln_out.bias": torch.zeros(C), "head.weight": r(C, V)}
    for i in range(L):
        b, a, f = f"blocks.{i}.", f"blocks.{i}.att.", f"blocks.{i}.ffn."
        for n in ("ln1", "ln2"):
            z[b + n + ".weight"], z[b + n + ".bias"] = torch.ones(C), torch.zeros(C)
        for n in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "k_k", "k_a", "a0", "v0", "w0", "r_k"):
            z[a + n] = r(C)
        for n in ("receptance.weight", "key.weight", "value.weight", "output.weight"):
            z[a + n] = r(C, C) / C ** 0.5
        for n in ("w1", "a1", "g1", "v1"):
            z[a + n] = r(C, D)
        for n in ("w2", "a2", "g2", "v2"):
            z[a + n] = r(D, C)
        z[a + "ln_x.weight"], z[a + "ln_x.bias"] = torch.ones(C), torch.zeros(C)
        z[f + "x_k"], z[f + "key.weight"], z[f + "value.weight"] = r(C), r(C, F) / C ** 0.5, r(F, C) / F ** 0.5
    return z


class _Pipe:
    def encode(self, text):
        return [max(1, min(ord(ch), V - 1)) for ch in text]

    def decode(self, toks):
        return "".join(chr(t) for t in toks)

    def sample_logits(self, logits, temperature=1.0, top_p=0.85, top_k=0):
        return int(torch.argmax(logits))


class _Model:
    """Stands in for the rwkv package model: same z layout, forward = the plastic math with adapters off."""

    def __init__(self):
        self.z = _tiny_z()
        self.n_head, self.head_size, self.n_embd, self.n_layer = H, N, C, L
        self.args = types.SimpleNamespace(n_layer=L)
        self._ref = PlasticCortex(types.SimpleNamespace(model=self), rank=1)
        self._ref.enabled = False

    def forward(self, tokens, state):
        with torch.no_grad():
            logits, st = self._ref(tokens, state)
        return logits[-1], st


@pytest.fixture()
def cortex():
    c = Cortex.__new__(Cortex)
    c.model, c.pipe, c.n_layers = _Model(), _Pipe(), L
    return c


def _brain(home, cortex, **kw):
    kw.setdefault("sleep_steps", 4)
    kw.setdefault("sleep_lr", 1e-2)
    kw.setdefault("sleep_kl_weight", 1.0)   # the mock cortex has no size: SLEEP_DEFAULTS has no row for it
    kw.setdefault("adapter_rank", 2)
    return Brain(str(home), cortex=cortex, **kw)


def _talk(b, turns=("Hi! What's your name?", "My cat is Pixel.", "What is my cat called?")):
    return [b.hear(t)[0] for t in turns]


def test_zero_adapters_are_the_pretrained_brain(tmp_path, cortex):
    plain = _talk(_brain(tmp_path / "a", cortex))
    b = _brain(tmp_path / "b", cortex)
    b._plastic()            # owns a plastic cortex, but nothing consolidated yet
    b.cortex.refresh()
    assert b.cortex.plastic is not None and not b.cortex.active
    assert _talk(b) == plain


def test_restart_sleeps_once_and_adapters_survive(tmp_path, cortex):
    b = _brain(tmp_path, cortex)
    assert b.meta["sleeps"] == 0 and not b.pending()       # birth is not a night
    b.hear("My cat is called Pixel.")
    assert len(b.pending()) == 1
    b.save()                                                 # crash/plain save: the next start sleeps
    b2 = _brain(tmp_path, cortex)
    assert b2.meta["sleeps"] == 1 and not b2.pending() and b2.cortex.active
    assert os.path.exists(tmp_path / ADAPTERS)
    saved = {k: v.clone() for k, v in b2.cortex.plastic.state_dict().items()}
    b3 = _brain(tmp_path, cortex)                            # nothing new lived: no sleep, same synapses
    assert b3.meta["sleeps"] == 1
    for k, v in b3.cortex.plastic.state_dict().items():
        assert torch.equal(v, saved[k])


def test_close_sleeps_only_when_something_was_lived(tmp_path, cortex):
    b = _brain(tmp_path, cortex)
    b.close()
    assert b.meta["sleeps"] == 0 and not os.path.exists(tmp_path / ADAPTERS)
    b = _brain(tmp_path, cortex)
    b.hear("I drive a red Volvo.")
    b.close()
    assert b.meta["sleeps"] == 1 and os.path.exists(tmp_path / ADAPTERS)
    assert _brain(tmp_path, cortex).meta["sleeps"] == 1      # shutdown already slept: the restart does not


def test_no_sleep_flag(tmp_path, cortex):
    b = _brain(tmp_path, cortex, sleep_on_restart=False)
    b.hear("My cat is called Pixel.")
    b.close()
    assert _brain(tmp_path, cortex, sleep_on_restart=False).meta["sleeps"] == 0


def test_consolidated_synapses_change_replies_and_the_lesion_removes_them(tmp_path, cortex):
    b = _brain(tmp_path, cortex, sleep_steps=30, sleep_lr=5e-2)
    b.hear("My cat is called Pixel.")
    b.sleep()
    ids = cortex.pipe.encode("User: What is my cat called?\n\nAssistant:")
    on, _ = b.cortex.forward(ids, b.base)
    ref, _ = cortex.forward(ids, b.base)
    assert not torch.equal(on, ref)                          # replies run through what sleep consolidated
    b.set_lesions(consolidated_off=True)
    off, _ = b.cortex.forward(ids, b.base)
    assert torch.equal(off, ref)                             # lesion: the pretrained cortex exactly
    b.set_lesions(hippocampus_off=True)
    assert not b.hippocampus.enabled and b.cortex.consolidated_on
    assert b.hear("What is my cat called?")[1]["recalled"] is None


def test_replay_is_strength_weighted_and_leaves_rng_alone(tmp_path, cortex, monkeypatch):
    import brain.brain as bb
    seen = {}

    def fake_sleep(plastic, base, eps, generic, **kw):
        seen["eps"] = eps
        torch.manual_seed(999)
        return []
    monkeypatch.setattr(bb, "replay_into_cortex", fake_sleep)
    b = _brain(tmp_path, cortex)
    b.hear("My cat is called Pixel.")
    torch.manual_seed(5)
    before = torch.random.get_rng_state()
    b.sleep()
    assert torch.equal(torch.random.get_rng_state(), before)
    texts = dict(seen["eps"])
    assert texts["User: My cat is called Pixel.\n\n"] == pytest.approx(1.0, abs=1e-3)
    assert min(s for t, s in seen["eps"] if "What's your name?" in t) >= 3.0   # identity replays most


def test_live_cortex_delegates_when_inactive(cortex):
    live = LiveCortex(cortex)
    logits, st = live.read("hello", None)
    ref, rst = cortex.read("hello", None)
    assert torch.equal(logits, ref) and all(torch.equal(a, b) for a, b in zip(st, rst))


GOLDEN = os.path.join(os.path.dirname(__file__), "fixtures", "brain_golden_0.1b.json")


@pytest.mark.skipif(not os.path.exists(DEFAULT_MODEL), reason="cortex weights not on this machine")
def test_zero_adapter_brain_is_byte_equal_to_the_pre_sleep_brain(tmp_path):
    from brain.experiments.golden_chat import record
    got = record(str(tmp_path / "home"), Cortex(device="cpu"))
    with open(GOLDEN) as fh:
        want = json.load(fh)
    # replies, recalled episode and storing decision byte-equal; the recall score carries the memory's wall-clock decay
    # (hippocampus.effective_strength), so it may move in the third decimal between recording and replay
    assert [[r[0], r[1], r[2], r[4]] for r in got] == [[r[0], r[1], r[2], r[4]] for r in want]
    assert all(abs(g[3] - w[3]) <= 2e-3 for g, w in zip(got, want))
