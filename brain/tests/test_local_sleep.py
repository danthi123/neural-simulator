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


# ---- SHY downscale after the night (prereg 2026-11-27-local-sleep-shy-prereg-DRAFT.md) ----
import pytest

from brain.experiments.local_sleep import commit_id, eval_downscaled, parse_grid, provenance


def _slept():
    pc = _pc()
    _randomise(pc)
    return pc


def _snap(pc):
    return [p.detach().clone() for p in pc.parameters()]


def test_downscale_none_leaves_adapters_and_row_identical():
    pc = _slept()
    before = _snap(pc)
    ref = {"adapter_norm": pc.adapter_norm()}
    [(g, r)] = eval_downscaled(pc, lambda m: {"x": 1}, [None])
    assert g is None and "downscale_g" not in r and r["adapter_norm"] == ref["adapter_norm"]
    assert all(torch.equal(b, p) for b, p in zip(before, pc.parameters()))
    [(g1, r1)] = eval_downscaled(pc, lambda m: {"x": 1}, [1.0])   # g = 1 is the plain evaluation
    assert r1["adapter_norm"] == ref["adapter_norm"] and r1["downscale_g"] == 1.0


def test_downscale_scales_norm_by_g_for_B_and_AB():
    for factors in ("B", "AB"):
        pc = _slept()
        n0 = pc.adapter_norm()
        rows = eval_downscaled(pc, lambda m: {}, [0.5, 0.25], factors)
        for g, r in rows:
            assert r["adapter_norm"] == pytest.approx(n0 * g, rel=1e-4) and r["downscale_factors"] == factors


def test_grid_restores_adapters_between_evaluations_and_at_the_end():
    pc = _slept()
    before = _snap(pc)
    seen = []
    def ev(m):
        seen.append(m.adapter_norm())
        return {}
    eval_downscaled(pc, ev, [1.0, 0.6, 0.4], "B")
    n0 = seen[0]
    assert seen[1] == pytest.approx(0.6 * n0, rel=1e-4) and seen[2] == pytest.approx(0.4 * n0, rel=1e-4)   # not 0.4 x 0.6
    assert all(torch.equal(b, p) for b, p in zip(before, pc.parameters()))
    def boom(m):
        raise RuntimeError
    with pytest.raises(RuntimeError):
        eval_downscaled(pc, boom, [0.3])
    assert all(torch.equal(b, p) for b, p in zip(before, pc.parameters()))   # restored even on error


def test_parse_grid_and_provenance(monkeypatch):
    assert parse_grid("1,0.8, 0.25") == [1.0, 0.8, 0.25] and parse_grid(None) == []
    for bad in ("0", "1.5", "-0.1"):
        with pytest.raises(ValueError):
            parse_grid(bad)
    monkeypatch.setenv("COMMIT", "abc123")
    p = provenance()
    assert p["commit"] == "abc123" and isinstance(p["host"], str) and p["host"]
    monkeypatch.delenv("COMMIT")
    monkeypatch.setenv("HOME", "/nonexistent-home")
    assert commit_id() is None and "commit" in provenance()


def test_off_path_output_has_exactly_the_old_keys(monkeypatch):
    from types import SimpleNamespace
    from brain.experiments.local_sleep import record_arm, result_header
    monkeypatch.setenv("HOST", "somehost")
    a = SimpleNamespace(model="m", seed=1, steps=2, lr=0.1, mode="kp", feedback="mixed", decay=1e-3, downscale_factors="B")
    res = result_header(a, [])
    assert set(res) == {"model", "seed", "steps", "lr", "mode", "feedback", "decay", "arms", "host", "host_env", "commit"}
    old_row = {"recall": 1, "paraphrase": 0, "false_recall": 0, "generic_kl": 0.0, "recall_items": [], "paraphrase_items": [],
               "adapter_norm": 1.0}
    [(g, r)] = eval_downscaled(_slept(), lambda m: dict(old_row), [None])
    key = record_arm(res, "local", g, r, final_loss=0.5, fbk="mixed")
    assert key == "local" and "rows" not in res
    assert set(res["arms"]["local"]) == set(old_row) | {"final_replay_loss", "feedback"}
    res2 = result_header(a, [1.0, 0.5])
    assert res2["rows"] == [] and res2["downscale_grid"] == [1.0, 0.5]
