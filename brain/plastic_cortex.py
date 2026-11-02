"""Plastic cortex: the same RWKV-7 cortex with gradients on, plus slow low-rank synapses that sleep can change.

The rwkv package runs inference only (every forward is under no_grad), so consolidation needs its own forward. This
module reuses the package's loaded weights (Cortex.model.z) and re-implements its v7 sequence math without in-place
writes, so autograd can flow. The pretrained weights stay frozen; learning goes into per-layer low-rank adapters on the
channel-mix (feed-forward) key and value weights: the cortex's slow store. With the adapters at zero the output equals
the package's (tests/test_plastic_cortex.py).

Scaffold note (SCAFFOLDS.md): the adapter update is gradient descent, the declared pretraining-style trade, applied
during sleep only; what is replayed and how often is the hippocampus's job (brain/sleep.py, next).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class PlasticCortex(torch.nn.Module):
    def __init__(self, cortex, rank: int = 8, layers=None, dtype=torch.float32):
        super().__init__()
        self.cortex = cortex
        self.z = {k: v.to(dtype) for k, v in cortex.model.z.items()}  # frozen copy in a trainable dtype
        self.H, self.N = cortex.model.n_head, cortex.model.head_size
        self.C, self.L = cortex.model.n_embd, cortex.model.n_layer
        self.layers = list(range(self.L)) if layers is None else list(layers)
        dev = self.z["emb.weight"].device
        hidden = self.z["blocks.0.ffn.key.weight"].shape[1]
        self.A_k, self.B_k, self.A_v, self.B_v = (torch.nn.ParameterDict() for _ in range(4))
        for i in self.layers:
            # delta = A @ B; B starts at zero so a fresh cortex is exactly the pretrained one
            self.A_k[str(i)] = torch.nn.Parameter(torch.randn(self.C, rank, device=dev, dtype=dtype) / self.C ** 0.5)
            self.B_k[str(i)] = torch.nn.Parameter(torch.zeros(rank, hidden, device=dev, dtype=dtype))
            self.A_v[str(i)] = torch.nn.Parameter(torch.randn(hidden, rank, device=dev, dtype=dtype) / hidden ** 0.5)
            self.B_v[str(i)] = torch.nn.Parameter(torch.zeros(rank, self.C, device=dev, dtype=dtype))
        self.enabled = True  # lesion switch: False = the pretrained cortex alone

    def zero_state(self):
        dev, dt = self.z["emb.weight"].device, self.z["emb.weight"].dtype
        st = []
        for _ in range(self.L):
            st += [torch.zeros(self.C, device=dev, dtype=dt), torch.zeros(self.H, self.N, self.N, device=dev),
                   torch.zeros(self.C, device=dev, dtype=dt)]
        return st

    def _tmix(self, i, x, x_prev, v_first, state):
        z, H, N = self.z, self.H, self.N
        a_ = f"blocks.{i}.att."
        T = x.shape[0]
        xx = torch.cat((x_prev.unsqueeze(0), x[:-1, :])) - x
        xr, xw, xk = x + xx * z[a_ + "x_r"], x + xx * z[a_ + "x_w"], x + xx * z[a_ + "x_k"]
        xv, xa, xg = x + xx * z[a_ + "x_v"], x + xx * z[a_ + "x_a"], x + xx * z[a_ + "x_g"]
        r = xr @ z[a_ + "receptance.weight"]
        w = torch.tanh(xw @ z[a_ + "w1"]) @ z[a_ + "w2"]
        k = xk @ z[a_ + "key.weight"]
        v = xv @ z[a_ + "value.weight"]
        a = torch.sigmoid(z[a_ + "a0"] + (xa @ z[a_ + "a1"]) @ z[a_ + "a2"])
        g = torch.sigmoid(xg @ z[a_ + "g1"]) @ z[a_ + "g2"]
        kk = F.normalize((k * z[a_ + "k_k"]).view(T, H, N), dim=-1, p=2.0).view(T, H * N)
        k = k * (1 + (a - 1) * z[a_ + "k_a"])
        if i == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(z[a_ + "v0"] + (xv @ z[a_ + "v1"]) @ z[a_ + "v2"])
        w = torch.exp(-0.606531 * torch.sigmoid((z[a_ + "w0"] + w).float()))
        outs = []
        for t in range(T):
            vk = v[t].view(H, N, 1) @ k[t].view(H, 1, N)
            ab = (-kk[t]).view(H, N, 1) @ (kk[t] * a[t]).view(H, 1, N)
            state = state * w[t].view(H, 1, N) + state @ ab.float() + vk.float()
            outs.append((state.to(dtype=x.dtype) @ r[t].view(H, N, 1)).view(H * N))
        y = torch.stack(outs)
        y = F.group_norm(y, num_groups=H, weight=z[a_ + "ln_x.weight"], bias=z[a_ + "ln_x.bias"], eps=64e-5)
        y = y + ((r * k * z[a_ + "r_k"]).view(T, H, N).sum(dim=-1, keepdim=True) * v.view(T, H, N)).view(T, H * N)
        return (y * g) @ z[a_ + "output.weight"], x[-1, :], state, v_first

    def _cmix(self, i, x, x_prev):
        f = f"blocks.{i}.ffn."
        xx = torch.cat((x_prev.unsqueeze(0), x[:-1, :])) - x
        k_in = x + xx * self.z[f + "x_k"]
        K, V = self.z[f + "key.weight"], self.z[f + "value.weight"]
        s = str(i)
        if self.enabled and s in self.A_k:
            h = torch.relu(k_in @ K + (k_in @ self.A_k[s]) @ self.B_k[s]) ** 2
            return h @ V + (h @ self.A_v[s]) @ self.B_v[s], x[-1, :]
        return (torch.relu(k_in @ K) ** 2) @ V, x[-1, :]

    def forward(self, tokens, state=None):
        """Logits for every position (T, vocab) and the new state. Gradients flow into the adapters only."""
        z = self.z
        state = [s.clone() for s in (state if state is not None else self.zero_state())]
        x = z["emb.weight"][tokens]
        v_first = torch.empty_like(x)
        for i in range(self.L):
            b = f"blocks.{i}."
            xx = F.layer_norm(x, (self.C,), weight=z[b + "ln1.weight"], bias=z[b + "ln1.bias"])
            xx, state[i * 3], state[i * 3 + 1], v_first = self._tmix(i, xx, state[i * 3], v_first, state[i * 3 + 1])
            x = x + xx
            xx = F.layer_norm(x, (self.C,), weight=z[b + "ln2.weight"], bias=z[b + "ln2.bias"])
            xx, state[i * 3 + 2] = self._cmix(i, xx, state[i * 3 + 2])
            x = x + xx
        x = F.layer_norm(x, (self.C,), weight=z["ln_out.weight"], bias=z["ln_out.bias"])
        return x @ z["head.weight"], state

    def adapter_norm(self) -> float:
        with torch.no_grad():
            return float(sum((self.A_k[s] @ self.B_k[s]).norm() + (self.A_v[s] @ self.B_v[s]).norm()
                             for s in self.A_k))
