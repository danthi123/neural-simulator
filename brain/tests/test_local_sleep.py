# Capability: Sleep consolidation by a local plasticity rule
# Record: brain/research/2026-11-27-local-sleep-rule-prereg-DRAFT.md
# Expected-fail: none (plumbing on a tiny random cortex; no model weights)
"""Local sleep: no autograd, adapters change, pretrained weights do not, replay loss falls, off path == stock."""
import torch

from brain.experiments.local_sleep import Feedback, LocalCortex, adopt, local_sleep, local_update, _forward
from brain.tests.test_source_tag import _Cortex
from brain.plastic_cortex import PlasticCortex

TEXT = "User: my cat is Pixel\n\nAssistant: ok\n\n"
GEN = ["User: hi there\n\nAssistant: hello\n\n"]


def _pc():
    torch.manual_seed(1)
    return PlasticCortex(_Cortex(), rank=2)


def _state(pc):
    return [s.float() for s in pc.zero_state()]


def test_update_is_gradient_free_and_changes_only_adapters():
    pc = _pc()
    z0 = {k: v.clone() for k, v in pc.z.items()}
    before = [p.detach().clone() for p in pc.parameters()]
    lc = adopt(pc)
    fb = Feedback(lc, "kp", 0)
    ids = lc.cortex.pipe.encode(TEXT)
    logits = _forward(lc, ids, _state(pc))
    assert not logits.requires_grad and all(not t.requires_grad for r in lc.rec for t in r[1:])
    err = torch.softmax(logits, -1)
    err[range(len(ids) - 1), ids[1:]] -= 1
    err[-1] = 0
    local_update(lc, fb, err, lr=0.1)
    assert all(p.grad is None for p in pc.parameters())
    assert any(not torch.equal(b, p) for b, p in zip(before, pc.parameters()))
    assert all(torch.equal(z0[k], pc.z[k]) for k in z0)


def test_replay_loss_decreases_and_off_path_is_stock():
    pc = _pc()
    st = _state(pc)
    stock = pc(pc.cortex.pipe.encode(TEXT), st)[0].detach()
    losses = local_sleep(pc, st, [(TEXT, 1.0)], GEN, steps=80, lr=0.3, decay=0.0, seed=3, feedback="transport")
    assert sum(l[0] for l in losses[-10:]) < sum(l[0] for l in losses[:10]) * 0.9
    assert pc.adapter_norm() > 0
    pc.enabled = False
    assert torch.allclose(pc(pc.cortex.pipe.encode(TEXT), st)[0], stock, atol=1e-5)
    fresh = _pc()   # zero B: adopted LocalCortex with rec off equals stock
    lc = adopt(fresh)
    assert isinstance(lc, LocalCortex) and torch.equal(lc(fresh.cortex.pipe.encode(TEXT), st)[0],
                                                       fresh(fresh.cortex.pipe.encode(TEXT), st)[0])


def _randomise(pc):
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for p in pc.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)


def test_recording_forward_matches_stock_with_random_adapters():
    pc = _pc()
    _randomise(pc)
    st = _state(pc)
    ids = pc.cortex.pipe.encode(TEXT)
    with torch.no_grad():
        stock = pc(ids, st)[0].clone()
    lc = adopt(pc)
    rec_logits = _forward(lc, ids, st)
    assert len(lc.rec) == len(pc.layers)
    assert torch.allclose(rec_logits, stock, atol=1e-5, rtol=1e-4)


def test_zero_error_step_leaves_adapters_unchanged_and_decay_acts_separately():
    pc = _pc()
    _randomise(pc)
    lc = adopt(pc)
    ids = lc.cortex.pipe.encode(TEXT)
    st = _state(pc)
    for mode in ("kp", "fa"):
        fb = Feedback(lc, mode, 0)
        zeros = torch.zeros_like(_forward(lc, ids, st))
        before = [p.detach().clone() for p in pc.parameters()]
        fb_before = [fb.v[i].clone() for i in pc.layers]
        local_update(lc, fb, zeros, lr=0.5, decay=0.0)
        assert all(torch.equal(b, p) for b, p in zip(before, pc.parameters()))
        assert all(torch.equal(b, fb.v[i]) for b, i in zip(fb_before, pc.layers))
        # decay alone shrinks B (and only B) by (1 - decay)
        A0 = {i: pc.A_k[str(i)].detach().clone() for i in pc.layers}
        B0 = {i: (pc.B_k[str(i)].detach().clone(), pc.B_v[str(i)].detach().clone()) for i in pc.layers}
        local_update(lc, fb, zeros, lr=0.5, decay=0.1)
        for i in pc.layers:
            s = str(i)
            assert torch.allclose(pc.B_k[s], B0[i][0] * 0.9) and torch.allclose(pc.B_v[s], B0[i][1] * 0.9)
            assert torch.equal(pc.A_k[s], A0[i])
