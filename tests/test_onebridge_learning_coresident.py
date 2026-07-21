"""CI guard: the render-LEARNING (delta-rule read-out over cp_ssm_state) co-resides with the composer on ONE bridge
and LEARNS while the composer binds/queries; frozen = load-bearing (2026-07-20). GPU-only; skips on numpy."""
import numpy as np
import pytest

from sim.backend import get_backend, is_gpu_backend, to_host

pytestmark = pytest.mark.skipif(not is_gpu_backend(), reason="GPU-only (RF ops + ssm read-out)")


def _run(frozen, seed=42, n_steps=150, lr=0.02):
    from research.runners._gap_onebridge_capstone_derisk import _build_capstone_bridge, SharedBridgeComposer
    xp, _ = get_backend(); rng = np.random.default_rng(seed)
    D_wkv, D_cmp = 64, 64
    facts = [("dog", "chase", "cat"), ("owl", "eat", "mouse")]
    vocab = sorted({w for f in facts for w in f})
    mb, mchan, enc_idx, cmp_idx = _build_capstone_bridge(D_wkv, D_cmp, seed, 0.9)
    n = int(mb.core_config.num_neurons)
    read_idx = np.concatenate([np.asarray(g) for g in mchan]).astype(np.int64)
    off = np.setdiff1d(np.arange(n), read_idx)
    OUT = 8
    W = (rng.standard_normal((OUT, n)) * 0.05).astype(np.float32); W[:, off] = 0.0
    mb.cp_ssm_readout_w = xp.asarray(W); mb.cp_ssm_readout_out = None
    cmp = SharedBridgeComposer(seed=seed, D=D_cmp, vocab=vocab); cmp.bind_to_shared(mb, cmp_idx)
    for a, v, p in facts:
        cmp.store(a, v, p)
    inj = rng.standard_normal(len(read_idx)) * 0.5; target = rng.standard_normal(OUT)
    losses = []
    for step in range(n_steps):
        cur = np.zeros(n, np.float32); cur[read_idx] = inj.astype(np.float32)
        mb.cp_ssm_inject[:] = xp.asarray(cur); mb.cp_ssm_shunt[:] = 0.0
        mb._run_one_simulation_step()
        out = np.asarray(to_host(mb.cp_ssm_readout_out)).astype(np.float64)
        err = out - target; losses.append(float(np.mean(err ** 2)))
        if not frozen:
            st = np.asarray(to_host(mb.cp_ssm_state)).astype(np.float64)
            dW = np.zeros((OUT, n)); dW[:, read_idx] = np.outer(err, st[read_idx])
            W = (W - lr * dW).astype(np.float32); W[:, off] = 0.0
            mb.cp_ssm_readout_w = xp.asarray(W)
        if step % 40 == 20:
            cmp.query_patient("dog", "chase")
    ans = [cmp.query_patient(a, v) for a, v, _ in facts]
    abstain = cmp.query_patient("lion", "roar")
    return losses, ans, abstain


def test_learning_coresident_with_composer():
    losses, ans, abstain = _run(frozen=False)
    assert losses[-1] < 0.2 * losses[0], f"read-out did not learn on shared bridge: {losses[0]} -> {losses[-1]}"
    assert ans == ["cat", "mouse"], f"composer recall wrong while learning: {ans}"
    assert abstain is None, "no-confab moat broken while learning"


def test_frozen_readout_does_not_learn():
    losses, _, _ = _run(frozen=True)
    assert losses[-1] > 0.2 * losses[0], "frozen read-out learned (delta update not load-bearing)"
