# Capability: Language
# Record: docs/plans/2026-11-27-spiking-cortex-route-memo.md
# Expected-fail: none (plumbing on a tiny random cortex; no model weights)
"""Spiking twin: spikes off == PlasticCortex; batched == per-sequence; counts are integers in [-K, K] with a clipped
straight-through gradient; spikes on change the output; a frozen twin lets sleep() change only the adapters; weights
round-trip; random init is seeded; checkpointing leaves gradients unchanged; TwinCortex runs the Cortex API."""
import torch

from brain.plastic_cortex import PlasticCortex
from brain.sleep import sleep
from brain.spiking_twin import SpikingTwin, TwinCortex, spike_count
from brain.tests.test_source_tag import _Cortex

TEXT = "User: my cat is Pixel\n\nAssistant: ok\n\n"


def _ids(c):
    return c.pipe.encode(TEXT)


def _calibrated(c, **kw):
    tw = SpikingTwin(c, **kw)
    tw.calibrate([torch.tensor([_ids(c)])])
    return tw


def test_spikes_off_equals_plastic_cortex_with_and_without_adapters():
    c = _Cortex()
    torch.manual_seed(3)
    pc = PlasticCortex(c, rank=2)
    torch.manual_seed(3)
    tw = SpikingTwin(c, rank=2, spikes=False)
    ids = _ids(c)
    with torch.no_grad():
        for d_pc, d_tw in ((pc.B_k, tw.B_k), (pc.B_v, tw.B_v)):
            for k in d_pc:
                d_pc[k].normal_(0, 0.1)
                d_tw[k].copy_(d_pc[k])
        a, sa = pc(ids)
        b, sb = tw(ids)
        assert torch.allclose(a, b, atol=1e-5)
        assert all(torch.allclose(x, y, atol=1e-5) for x, y in zip(sa, sb))
        a2, _ = pc(ids[:5], sa)    # carrying state
        b2, _ = tw(ids[:5], sb)
        assert torch.allclose(a2, b2, atol=1e-5)


def test_batched_equals_per_sequence():
    c = _Cortex()
    tw = _calibrated(c)
    ids = _ids(c)[:12]
    other = list(reversed(ids))
    with torch.no_grad():
        lb, _ = tw.forward_batch(torch.tensor([ids, other]))
        assert torch.allclose(lb[0], tw(ids)[0], atol=1e-5)
        assert torch.allclose(lb[1], tw(other)[0], atol=1e-5)


def test_counts_are_integers_in_range_with_clipped_ste():
    x = torch.tensor([-10.0, -0.6, 0.2, 0.7, 1.4, 10.0], requires_grad=True)
    th = torch.tensor(0.5, requires_grad=True)
    y, cnt = spike_count(x, th, 4)
    assert torch.equal(cnt, torch.tensor([-4.0, -1.0, 0.0, 1.0, 3.0, 4.0]))
    assert torch.allclose(y, cnt * 0.5)
    y.sum().backward()
    assert torch.equal(x.grad, torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 0.0]))
    assert th.grad is not None and float(th.grad) != 0.0


def test_spikes_on_change_output_and_record_stats():
    c = _Cortex()
    tw = _calibrated(c)
    ids = _ids(c)
    with torch.no_grad():
        tw.spikes = False
        off = tw(ids)[0]
        tw.spikes = True
        tw.stats = {}
        on = tw(ids)[0]
    assert not torch.allclose(on, off, atol=1e-4)
    s = tw.spike_summary()
    assert s["sites"] == len(tw.sites) - 2          # layer 0 has no v1/v2 gate
    assert 0 < s["mean_count"] <= 4 and 0 <= s["silent_share"] < 1


def test_frozen_twin_sleep_changes_only_adapters():
    c = _Cortex()
    torch.manual_seed(1)
    tw = _calibrated(c, rank=2)
    z0 = {k: v.clone() for k, v in tw.z.items()}
    th0 = {k: v.clone() for k, v in tw.theta.items()}
    st = [s.float() for s in tw.zero_state()]
    names = {n for n, _ in tw.named_parameters()}
    assert all(n.split(".")[0] in ("A_k", "B_k", "A_v", "B_v") for n in names)
    sleep(tw, st, [(TEXT, 1.0)], ["User: hi\n\nAssistant: hello\n\n"], steps=5, lr=1e-2, seed=0)
    assert tw.adapter_norm() > 0
    assert all(torch.equal(z0[k], tw.z[k]) for k in z0) and all(torch.equal(th0[k], tw.theta[k]) for k in th0)


def test_trainable_gets_gradients_in_weights_and_thetas():
    c = _Cortex()
    tw = SpikingTwin(c, trainable=True)
    tw.calibrate([torch.tensor([_ids(c)])])
    lg, _ = tw.forward_batch(torch.tensor([_ids(c)]))
    lg.float().pow(2).mean().backward()
    assert tw.W["head|weight"].grad is not None and tw.W["blocks|1|att|key|weight"].grad.abs().sum() > 0
    assert tw.Th["0|r"].grad is not None


def test_checkpointing_leaves_gradients_unchanged():
    c = _Cortex()
    grads = []
    for ck in (False, True):
        tw = SpikingTwin(c, trainable=True, ckpt=ck)
        lg, _ = tw.forward_batch(torch.tensor([_ids(c)[:10]]))
        lg.pow(2).mean().backward()
        grads.append(tw.W["blocks|0|att|value|weight"].grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_weights_roundtrip_and_random_init():
    c = _Cortex()
    tw = _calibrated(c, trainable=True)
    tw.random_init(5)
    blob = tw.weights_state()
    fresh = SpikingTwin(c)
    fresh.load_weights(blob)
    ids = _ids(c)
    with torch.no_grad():
        assert torch.allclose(fresh(ids)[0], tw(ids)[0], atol=1e-6)
    teacher = SpikingTwin(c)
    assert not torch.allclose(blob["z"]["blocks.0.att.key.weight"], teacher.z["blocks.0.att.key.weight"])
    assert torch.equal(blob["z"]["blocks.0.ln1.weight"], torch.ones_like(blob["z"]["blocks.0.ln1.weight"]))
    again = SpikingTwin(c)
    again.random_init(5)
    assert torch.equal(again.z["blocks.1.ffn.key.weight"], blob["z"]["blocks.1.ffn.key.weight"])


def test_twin_cortex_api():
    c = _Cortex()
    tc = TwinCortex(_calibrated(c))
    lg, st = tc.read("User: hi\n\n")
    assert lg.shape == (128,) and len(st) == 6
    text, _ = tc.speak(lg, st, max_tokens=5, temperature=0)
    assert isinstance(text, str)
    d = tc.matrix_delta(tc.read("User: my cat\n\n", st)[1], st)
    assert d[1] is not None and d[0] is None and tc.key_of(d).numel() == 2 * 4 * 4
    _, _, nll = tc.read_nll("User: hi\n\n")
    assert nll.shape == (len("User: hi\n\n") - 1,)


def test_rwkv_init_recipe_is_structured_and_trains():
    c = _Cortex()
    tw = SpikingTwin(c, trainable=True)
    tw.random_init(3)                                   # default recipe: RWKV-LM-style (arm b)
    assert tw.init_fallback == []                       # every stub key is named by the recipe
    z = tw.z
    for k in ("blocks.0.att.output.weight", "blocks.1.ffn.value.weight", "blocks.0.att.w1", "blocks.1.att.a0"):
        assert torch.count_nonzero(z[k]) == 0, k
    assert torch.allclose(z["blocks.0.att.k_k"], torch.full_like(z["blocks.0.att.k_k"], 0.85))
    w2 = z["blocks.0.att.w2"].detach()
    assert torch.allclose(w2 @ w2.T, 0.01 * torch.eye(w2.shape[0]), atol=1e-5)   # orthogonal, gain 0.1
    tw.calibrate([torch.tensor([_ids(c)])])
    # the w2 site's input is tanh(x @ 0) = 0 at init: theta falls back to 1/K, not the 1e-4 floor
    assert torch.allclose(tw.theta["0.w2"].detach(), torch.full_like(tw.theta["0.w2"], 1 / 4))
    lg, _ = tw.forward_batch(torch.tensor([_ids(c)[:12]]))
    assert torch.isfinite(lg).all()
    lg.pow(2).mean().backward()
    # zero output projections (RWKV-LM): step 1 trains them; once they move, the zero-init LoRA downs learn too
    assert tw.W["blocks|0|att|output|weight"].grad.abs().sum() > 0 and tw.W["blocks|1|ffn|value|weight"].grad.abs().sum() > 0
    with torch.no_grad():
        for p in tw.parameters():
            if p.grad is not None:
                p -= 0.1 * p.grad
    tw.zero_grad()
    lg, _ = tw.forward_batch(torch.tensor([_ids(c)[:12]]))
    lg.pow(2).mean().backward()
    assert tw.W["blocks|0|att|w1"].grad.abs().sum() > 0
    again, ref = SpikingTwin(c, trainable=True), SpikingTwin(c, trainable=True)
    again.random_init(3)
    ref.random_init(3)
    assert all(torch.equal(again.z[k], ref.z[k]) for k in ref.z)
    mom = SpikingTwin(c)
    mom.random_init(3, recipe="moment")
    assert torch.count_nonzero(mom.z["blocks.0.att.output.weight"]) > 0
