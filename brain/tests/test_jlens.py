# Capability: Memory selection (pattern completion and separation)
# Record: brain/research/2026-11-26-routing-screen3-prereg.md (part J: J matrices, identity + finite-difference checks)
# Expected-fail: none (a random tiny RWKV-7-shaped cortex; no model weights)
"""jlens: residuals() reproduces PlasticCortex.forward; J of the last layer is the identity; J matches a finite
difference of the mean future sum; batched and looped vector-Jacobian products agree."""
import types

import torch

from brain.experiments import jlens
from brain.plastic_cortex import PlasticCortex

C, H, N, L, V, D, Fh = 8, 2, 4, 3, 13, 4, 16


def _cortex(seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: 0.3 * torch.randn(*s, generator=g, dtype=torch.float32)   # noqa: E731
    z = {"emb.weight": r(V, C), "ln_out.weight": 1 + r(C), "ln_out.bias": r(C), "head.weight": r(C, V)}
    for i in range(L):
        a, f, b = f"blocks.{i}.att.", f"blocks.{i}.ffn.", f"blocks.{i}."
        for k in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "a0", "w0", "v0", "k_k", "k_a", "r_k", "ln_x.bias"):
            z[a + k] = r(C)
        z[a + "ln_x.weight"] = 1 + r(C)
        for k in ("receptance.weight", "key.weight", "value.weight", "output.weight"):
            z[a + k] = r(C, C)
        for k in ("w", "a", "g", "v"):
            z[a + f"{k}1"], z[a + f"{k}2"] = r(C, D), r(D, C)
        z[f + "x_k"], z[f + "key.weight"], z[f + "value.weight"] = r(C), r(C, Fh), r(Fh, C)
        for n in ("ln1", "ln2"):
            z[b + n + ".weight"], z[b + n + ".bias"] = 1 + r(C), r(C)
    model = types.SimpleNamespace(z=z, n_head=H, head_size=N, n_embd=C, n_layer=L)
    p = PlasticCortex(types.SimpleNamespace(model=model), rank=2, dtype=torch.float32)
    p.enabled = False
    return p


def _zero_state(p):
    return [torch.zeros(C, dtype=torch.float32), torch.zeros(H, N, N, dtype=torch.float32),
            torch.zeros(C, dtype=torch.float32)] * L


TOK = [1, 4, 2, 7, 3]


def test_residuals_match_forward():
    p = _cortex()
    st = _zero_state(p)
    logits, s1 = p(TOK, st)
    hs, _, s2 = jlens.residuals(p, TOK, st)
    assert len(hs) == L
    assert torch.allclose(jlens.readout(p, hs[-1]), logits)
    assert all(torch.allclose(a, b) for a, b in zip(s1, s2))


def test_last_layer_is_identity():
    p = _cortex()
    p.zero_state = lambda: _zero_state(p)
    J = jlens.jacobians_one(p, TOK, [L - 1], chunk=4)
    assert torch.allclose(J[L - 1], torch.eye(C, dtype=torch.float32), atol=1e-6)


def test_finite_difference_and_batched():
    p = _cortex(1)
    p.zero_state = lambda: _zero_state(p)
    l = 0
    Jb = jlens.jacobians_one(p, TOK, [l], chunk=3, batched=True)[l]
    Jl = jlens.jacobians_one(p, TOK, [l], chunk=3, batched=False)[l]
    assert torch.allclose(Jb, Jl, atol=1e-5)
    # finite difference: perturb h_l at every source position t by eps * e_k in turn; sum_t sum_t' dh_L,t' / T
    T, eps = len(TOK), 1e-3

    def hL_sum_after(t, k, sgn):
        z = p.z
        st = [s.clone() for s in _zero_state(p)]
        x = z["emb.weight"][TOK]
        v_first = torch.empty_like(x)
        import torch.nn.functional as F
        for i in range(p.L):
            b = f"blocks.{i}."
            xx = F.layer_norm(x, (C,), weight=z[b + "ln1.weight"], bias=z[b + "ln1.bias"])
            xx, st[i * 3], st[i * 3 + 1], v_first = p._tmix(i, xx, st[i * 3], v_first, st[i * 3 + 1])
            x = x + xx
            xx = F.layer_norm(x, (C,), weight=z[b + "ln2.weight"], bias=z[b + "ln2.bias"])
            xx, st[i * 3 + 2] = p._cmix(i, xx, st[i * 3 + 2])
            x = x + xx
            if i == l:
                x = x.clone()
                x[t, k] += sgn * eps
        return x.sum(0)

    Jfd = torch.zeros(C, C, dtype=torch.float32)
    for k in range(C):
        col = sum((hL_sum_after(t, k, 1) - hL_sum_after(t, k, -1)) / (2 * eps) for t in range(T)) / T
        Jfd[:, k] = col
    assert torch.allclose(Jb, Jfd, atol=2e-3, rtol=1e-2)
