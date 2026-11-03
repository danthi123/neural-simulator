"""Brain(key=...) switch: default OFF is the old Brain (byte-equal), S_all uses KeyedHippocampus with its own bar and is
persisted in the home's meta.json. Runs on a tiny deterministic stand-in cortex (no model weights)."""
import importlib.util
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from brain.brain import Brain
from brain.cortex import Cortex
from brain.hippocampus import Hippocampus
from brain.hippocampus_keys import KeyedHippocampus

REF = "research/landing-1103"   # the Brain before the switch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Pipe:
    V = 128

    def encode(self, text):
        return [ord(c) % self.V for c in text]

    def decode(self, ids):
        return "".join(chr(i) if 32 <= i < 127 or i == 10 else "?" for i in ids)

    def sample_logits(self, logits, temperature=1.0, top_p=0.3):
        return int(torch.argmax(logits))


class StubCortex(Cortex):
    """Deterministic 2-layer recurrent stand-in with the Cortex state layout (token-shift, matrix memory, token-shift)."""
    C, H, N = 8, 2, 4

    def __init__(self):
        self.pipe, self.n_layers = _Pipe(), 2
        g = torch.Generator().manual_seed(0)
        self.emb = torch.randn(_Pipe.V, self.C, generator=g)
        self.out = torch.randn(self.C, _Pipe.V, generator=g)

    def forward(self, tokens, state):
        if state is None:
            state = [torch.zeros(self.C) if i % 3 != 1 else torch.zeros(self.H, self.N, self.N)
                     for i in range(3 * self.n_layers)]
        st = [s.clone() for s in state]
        for t in tokens:
            x = self.emb[t]
            for l in range(self.n_layers):
                k = torch.tanh(x * (l + 1)).view(self.H, self.N)
                st[3 * l + 1] = 0.9 * st[3 * l + 1] + k[:, :, None] * k[:, None, :]
                st[3 * l] = x.clone()
                x = torch.tanh(x + st[3 * l + 1].sum(-1).flatten())
                st[3 * l + 2] = x.clone()
        logits = x @ self.out
        logits[10] += 3.0 if len(tokens) == 1 and tokens[0] != 10 and float(x[0]) > 0.5 else 0.0
        return logits, st


# the stub has no model name, so sleep settings are explicit (SLEEP_DEFAULTS has no silent fallback)
KW = dict(sleep_on_restart=False, sleep_steps=4, sleep_lr=1e-3, sleep_kl_weight=1.0, adapter_rank=2)

TALK = ["My cat is called Pixel.", "I drive a red Volvo.", "What is my cat called?", "What car do I drive?",
        "My sister's name is Marguerite.", "Who is my sister?"]


def _talk(b):
    torch.manual_seed(0)
    return [b.hear(t) for t in TALK]


def _old_brain_module():
    try:
        src = subprocess.run(["git", "-C", ROOT, "show", f"{REF}:brain/brain.py"], capture_output=True, text=True,
                             check=True).stdout
    except Exception:
        pytest.skip(f"{REF} not available")
    path = os.path.join(ROOT, "brain", "_ref_brain_for_test.py")
    with open(path, "w") as fh:
        fh.write(src)
    try:
        spec = importlib.util.spec_from_file_location("brain._ref_brain_for_test", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    finally:
        os.remove(path)
    return mod


def _meta(home):
    import json
    with open(os.path.join(home, "meta.json")) as fh:
        return json.load(fh)


def test_default_matches_the_old_brain(tmp_path):
    """Default OFF = the old Brain: same replies and info, and the same store. The on-disk layout is NOT compared
    byte for byte any more: adopt-1.5b moved storage to one file per episode (episodes/ + episodes.json), so the old
    memory.pt bytes no longer exist for the new Brain. The old home is loaded through the new Brain's legacy
    (memory.pt) migration and its episodes are compared tensor for tensor with the new home's."""
    old = _old_brain_module()
    c = StubCortex()
    b_old = old.Brain(str(tmp_path / "old"), cortex=c, **KW)
    b_new = Brain(str(tmp_path / "new"), cortex=c, **KW)
    assert type(b_new.hippocampus) is Hippocampus and b_new.key == "S0"
    r_old, r_new = _talk(b_old), _talk(b_new)
    assert r_old == r_new
    b_old.save()
    b_new.save()
    assert "key" not in _meta(str(tmp_path / "new"))
    o = Brain(str(tmp_path / "old"), cortex=c, **KW).hippocampus.episodes   # legacy layout
    n = Brain(str(tmp_path / "new"), cortex=c, **KW).hippocampus.episodes
    assert len(o) == len(n) == len(b_new.hippocampus.episodes) > 0
    for eo, en in zip(o, n):
        for k in ("text", "strength", "uses", "permanent", "source"):
            assert getattr(eo, k) == getattr(en, k)
        assert torch.equal(eo.key, en.key)
        assert all((x is None and y is None) or torch.equal(x, y) for x, y in zip(eo.delta, en.delta))


def test_source_is_saved_and_loaded(tmp_path):
    home = str(tmp_path / "h")
    c = StubCortex()
    b = Brain(home, cortex=c, **KW)
    _talk(b)
    b.hippocampus.episodes[0].source = "self"
    b.save()
    b2 = Brain(home, cortex=c, **KW)
    assert [e.source for e in b2.hippocampus.episodes] == [e.source for e in b.hippocampus.episodes]
    assert b2.hippocampus.episodes[0].source == "self"


def test_sall_switch_bar_and_no_stored_key(tmp_path):
    b = Brain(str(tmp_path / "h"), cortex=StubCortex(), **KW, key="S_all")
    h = b.hippocampus
    assert isinstance(h, KeyedHippocampus) and h.key_name == "S_all" and b.key == "S_all"
    assert (h.FAMILIARITY, h.MARGIN) == (0.185, 0.05)
    assert (Hippocampus.FAMILIARITY, Hippocampus.MARGIN) == (0.12, 0.02)   # class defaults untouched
    _talk(b)
    assert all(e.key is None for e in h.episodes)


def test_sall_persisted_with_the_store(tmp_path):
    home = str(tmp_path / "h")
    c = StubCortex()
    b = Brain(home, cortex=c, **KW, key="S_all")
    _talk(b)
    b.save()
    assert _meta(home)["key"] == "S_all"
    b2 = Brain(home, cortex=c, **KW)            # no key given: the stored one
    assert b2.key == "S_all" and b2.hippocampus.FAMILIARITY == 0.185
    assert len(b2.hippocampus.episodes) == len(b.hippocampus.episodes)
    b3 = Brain(home, cortex=c, **KW, key="S0")  # explicit key wins; S0 keys rebuilt from deltas
    assert type(b3.hippocampus) is Hippocampus
    for e in b3.hippocampus.episodes:
        assert torch.equal(e.key, e.delta[1].float().flatten())
    b3.save()
    assert "key" not in _meta(home)


def test_s0_home_opened_with_sall(tmp_path):
    home = str(tmp_path / "h")
    c = StubCortex()
    b = Brain(home, cortex=c, **KW)
    _talk(b)
    b.save()
    b2 = Brain(home, cortex=c, **KW, key="S_all")
    assert b2.key == "S_all" and all(e.key is None for e in b2.hippocampus.episodes)
    reply, info = b2.hear("What is my cat called?")
    assert isinstance(reply, str) and "similarity" in info


def test_sall_similarity_is_centred_cosine_of_all_layer_deltas(tmp_path):
    c = StubCortex()
    b = Brain(str(tmp_path / "h"), cortex=c, **KW, key="S_all")
    _talk(b)
    h = b.hippocampus
    cue = "Which car is mine?"
    _, st = c.read(f"User: {cue}\n\n", b.base)
    blk = lambda d: torch.cat([F.normalize(d[3 * l + 1].flatten().float(), dim=0) for l in range(c.n_layers)])
    K = torch.stack([blk(e.delta) for e in h.episodes])
    q = blk([st[i] - b.base[i] if i % 3 == 1 else None for i in range(len(st))])
    mu = K.mean(0)
    want = F.normalize(K - mu, dim=1) @ F.normalize(q - mu, dim=0)
    assert torch.allclose(h.similarities(b.cortex, b.base, cue), want, atol=2e-3)
