# Capability: Self and identity
# Record: brain/research/2026-11-26-source-tag-prereg-DRAFT.md
# Expected-fail: none (plumbing on a tiny random cortex; no model weights)
"""Source tag: the mask covers "User:" turns only; no tag (or tag off, or a zero tag) leaves the plastic cortex's output
unchanged; a tag changes only tagged positions and what follows; sleep trains a learned tag and leaves a fixed one."""
import torch

from brain import source_tag as ST
from brain.plastic_cortex import PlasticCortex
from brain.sleep import sleep
from brain.tests.test_brain_key_switch import _Pipe


class _Model:
    def __init__(self, C=8, H=2, L=2, hid=16, V=128, r=3, seed=0):
        g = torch.Generator().manual_seed(seed)
        R = lambda *s: torch.randn(*s, generator=g) * 0.3
        z = {"emb.weight": R(V, C), "ln_out.weight": torch.ones(C), "ln_out.bias": torch.zeros(C), "head.weight": R(C, V)}
        for i in range(L):
            b, a, f = f"blocks.{i}.", f"blocks.{i}.att.", f"blocks.{i}.ffn."
            for n in ("ln1", "ln2"):
                z[b + n + ".weight"], z[b + n + ".bias"] = torch.ones(C), torch.zeros(C)
            for n in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "a0", "k_k", "k_a", "v0", "w0", "r_k"):
                z[a + n] = R(C)
            for n in ("receptance.weight", "key.weight", "value.weight", "output.weight"):
                z[a + n] = R(C, C)
            for n in ("w1", "a1", "g1", "v1"):
                z[a + n] = R(C, r)
            for n in ("w2", "a2", "g2", "v2"):
                z[a + n] = R(r, C)
            z[a + "ln_x.weight"], z[a + "ln_x.bias"] = torch.ones(C), torch.zeros(C)
            z[f + "x_k"], z[f + "key.weight"], z[f + "value.weight"] = R(C), R(C, hid), R(hid, C)
        self.z, self.n_head, self.head_size, self.n_embd, self.n_layer = z, H, C // H, C, L


class _Cortex:
    def __init__(self):
        self.pipe, self.model = _Pipe(), _Model()


def _plastic():
    torch.manual_seed(1)
    p = PlasticCortex(_Cortex(), rank=2)
    with torch.no_grad():
        for d in (p.B_k, p.B_v):
            for v in d.values():
                v.normal_(0, 0.1)
    return p


TEXT = "Hi there\n\nUser: my cat\n\nAssistant: ok"


def test_mask_covers_user_turns_only():
    pipe = _Pipe()
    m = ST.user_mask(pipe, pipe.encode(TEXT))
    got = "".join(c for c, t in zip(TEXT, m) if t)
    assert got == "User: my cat\n\n"
    assert ST.user_mask(pipe, pipe.encode("User: x")) == [True] * 7
    assert not any(ST.user_mask(pipe, pipe.encode("Assistant: hi\n\n")))


def test_no_tag_zero_tag_and_lesion_leave_output_unchanged():
    p = _plastic()
    ids = p.cortex.pipe.encode(TEXT)
    with torch.no_grad():
        ref = p(ids)[0]
        ST.attach(p, torch.zeros(8), layer=0, learn=False)
        assert torch.equal(p(ids)[0], ref)
        ST.attach(p, torch.ones(8), layer=0, learn=False)
        on = p(ids)[0]
        first = TEXT.index("User:")
        assert torch.equal(on[:first], ref[:first])          # before the user turn nothing changes
        assert not torch.allclose(on[first:], ref[first:])
        p.enabled = False
        off_tag = p(ids)[0]
        ST.detach(p)
        assert torch.equal(off_tag, p(ids)[0])               # the tag is part of the slow store: off with it
        p.enabled = True
        assert torch.equal(p(ids)[0], ref)                   # detached = no tag


def test_sleep_trains_a_learned_tag_not_a_fixed_one():
    eps, gen = [("User: my cat is Pixel\n\n", 1.0)], ["User: hello\n\nAssistant: hi\n\n"]
    base = None
    a = _plastic()
    la = sleep(a, base, eps, gen, steps=4, seed=3)
    b = _plastic()
    assert sleep(b, base, eps, gen, steps=4, seed=3) == la      # deterministic with no tag
    c = _plastic()
    t = ST.attach(c, torch.zeros(8), layer=1, learn=True)
    sleep(c, base, eps, gen, steps=4, seed=3)
    assert float(t.detach().abs().sum()) > 0
    d = _plastic()
    v = torch.full((8,), 0.5)
    ST.attach(d, v, layer=1, learn=False)
    sleep(d, base, eps, gen, steps=4, seed=3)
    assert torch.equal(d.source_tag, v)
