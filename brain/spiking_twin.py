# Capability: Language
# Record: docs/plans/2026-11-27-spiking-cortex-route-memo.md
# Expected-fail: at 0.1B a fluency gap of about 0.3 nats to the teacher (SpikingMamba-130M, cited in the record) and weak
#   in-brain memory (in-context 6/12 for the teacher itself); a miss there is a scale result. A reinstatement miss is a
#   verdict on this method (L1 counts, STE, this distillation recipe), not on spiking cortices.
"""Spiking twin of the RWKV-7 cortex, level L1 (memo section 2): every projection input is a signed integer spike count.

A count is c = clamp(round(x / theta), -K, K) per channel (K = 4, SpikingMamba's range) and the projection receives
c * theta: theta is the synaptic scale of that input channel, learnable, so the projection weights see integer spike
counts times a per-channel efficacy. Gradient: straight-through on the rounding, zero outside the clamp (clipped STE),
and through theta as in LSQ. The recurrent state (matrix memory, token shifts) stays graded: synaptic variables, as in
SpikeGPT / SpikingBrain / SpikingMamba (no published fluent model holds its state in spikes; memo section 2, L3).

Sites quantised (all projections of the v7 block plus the readout): att receptance/key/value inputs, the four low-rank
gates (w1, a1, g1, v1) and their up projections (w2, a2, g2, v2), the attention output, channel-mix key and value, head.
The embedding is a lookup, not a projection: unchanged.

SpikingTwin subclasses PlasticCortex (nothing frozen is patched, like local_sleep.LocalCortex): with spikes=False and
teacher weights its output equals PlasticCortex's (tests/test_spiking_twin.py), with the same state layout, so the
hippocampus, sleep() and sleep_probe.evaluate run on it unchanged. It also runs batched (B, T) for distillation, with
per-block activation checkpointing. trainable=True turns every weight and theta into a Parameter (distillation);
trainable=False (default) keeps them plain tensors so sleep() trains only the slow-store adapters.

Training kernel (switch wkv, default "loop" = the per-token Python WKV loop, unchanged): wkv="fla" computes the
time-mix state recurrence with flash-linear-attention's chunk_rwkv7 (fla-core 0.6.0, off the shelf, CUDA/Triton only;
prereg brain/research/2026-11-27-spiking-twin-e1-prereg.md section 7 item 1). The mapping onto fla's DPLR form
S_fla <- S_fla * exp(gk) + b (a^T S_fla) + k v^T, o = scale * S_fla^T q, is (see _wkv_fla): S_fla = our state
transposed (ours is [H, V, K]), q = r, gk = log w = -0.606531 sigmoid(w0 + lora) (computed in log space, not log(exp)),
a = -kk, b = kk * a_gate, scale = 1. fla's fp32 Triton dots run in IEEE fp32, not TF32 (fla_f32_precision: TF32
biased the fp32 path, run/parity/bias_diagnosis.md). The package is imported lazily (fla_import) so nothing else
depends on it; it is a training path only: single-token steps (T == 1, sampling) and non-CUDA tensors always take the loop. Spikes, a ragged
T, checkpointing (the block is recomputed through self._wkv_fla) and lower_bound all take the kernel; nothing else
bypasses it. With spikes on, a kernel difference smaller than a count's rounding margin is absorbed by the integer
counts of the "o" site and the head, so the logits can be bit-equal to the loop although the kernel ran (the graded
state is not): seen on the tiny parity cortex, 11-27, and reproduced on CPU by tests/test_spiking_twin_fla.py.

TwinCortex wraps a twin in the Cortex API (read, read_nll, speak, matrix_delta, add_delta, key_of) so the scale-ladder
memory rows run on it.

Scaffold note (SCAFFOLDS.md, to be declared with the result): spike counts with graded state = level L1; the training
is surrogate-gradient distillation from the teacher, the declared pretraining-style trade (ARCHITECTURE.md rule 4).
"""
from __future__ import annotations

import copy
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from brain.cortex import Cortex
from brain.plastic_cortex import PlasticCortex

MAX_COUNT = 4
WKV_KERNELS = ("loop", "fla")
FLA_VERSION = "0.6.0"                     # the fla-core release the parity test was written against
FLA_LIB = "/home/dant123/Projects/sim/_scratch/pylib_fla"   # installed with --no-deps; on PYTHONPATH for the twin only
# gk = -0.606531 * sigmoid(.) lies in (LOG_DECAY_MIN, 0): fp32 rounds the product to -0.60653102 at saturation, so the
# bound handed to fla (wkv_lower_bound=True only) sits a little below -0.606531.
LOG_DECAY_MIN = -0.6066


def fla_version():
    """The installed fla-core version, read from package metadata only: nothing is imported, so CUDA is not touched
    (importing fla probes the GPU at import, fla/utils/hardware.py). Raises ImportError if absent or not FLA_VERSION."""
    import importlib.metadata as md
    try:
        ver = md.version("fla-core")
    except md.PackageNotFoundError as e:  # pragma: no cover - environment
        raise ImportError(f"wkv='fla' needs fla-core {FLA_VERSION} on PYTHONPATH (e.g. {FLA_LIB}): not installed") from e
    if ver != FLA_VERSION:
        raise ImportError(f"wkv='fla' was validated against fla-core {FLA_VERSION}; found {ver}")
    return ver


FLA_F32_PRECISION = "ieee"   # Triton's precision for fp32 tl.dot inside fla's kernels (Triton's own default is tf32)


def fla_f32_precision():
    """Make every fp32 tl.dot in fla's kernels IEEE fp32 instead of Triton's default TF32 (TRITON_F32_DEFAULT).

    Why (run/parity/bias_diagnosis.md, 11-27): with TF32, chunk_rwkv7 on fp32 inputs is not fp32-accurate (relative
    error 7e-4 to 9e-4 per layer on the 0.1B teacher) and it is biased: its output is shrunk toward zero (slope of the
    error on y -3e-4 to -9e-4 in all 12 layers) and 63% of the o-site count flips lower |count|. With IEEE the
    error is fp32 rounding (4e-7 to 1.7e-6, the size of an fp64 recurrence against the loop) and the shrink is gone.
    Triton reads the setting when it compiles a kernel; its in-process kernel cache is not keyed on it (the on-disk
    cache is), so this must run before the first fla kernel compiles in the process: fla_import calls it first.
    A different explicit setting (environment or triton.knobs) is refused, not overridden."""
    cur = os.environ.get("TRITON_F32_DEFAULT")
    if cur not in (None, "", FLA_F32_PRECISION):
        raise RuntimeError(f"wkv='fla' runs fla with TRITON_F32_DEFAULT={FLA_F32_PRECISION}; found {cur!r}")
    os.environ["TRITON_F32_DEFAULT"] = FLA_F32_PRECISION
    triton = sys.modules.get("triton")
    knobs = getattr(triton, "knobs", None) if triton is not None else None
    if knobs is not None:
        got = knobs.language.fp32_default        # a Python-level override wins over the environment
        if got != FLA_F32_PRECISION:
            raise RuntimeError(f"wkv='fla' needs triton.knobs.language.fp32_default={FLA_F32_PRECISION}; found {got!r}")
    return FLA_F32_PRECISION


def fla_import():
    """Lazily import fla's rwkv7 ops (chunk_rwkv7, fused_recurrent_rwkv7) after checking the installed version and
    fixing fla's fp32 dot precision to IEEE (fla_f32_precision). Importing fla initialises CUDA (device name /
    capability probe), so call this only where a GPU run is intended."""
    fla_version()
    fla_f32_precision()
    try:
        from fla.ops.rwkv7 import chunk_rwkv7, fused_recurrent_rwkv7
    except ImportError as e:  # pragma: no cover - environment
        raise ImportError(f"wkv='fla' needs fla-core {FLA_VERSION} on PYTHONPATH (e.g. {FLA_LIB}): {e}") from e
    return chunk_rwkv7, fused_recurrent_rwkv7


def wkv_fla_args(r, logw, k, v, kk, a, S):
    """Our (B, T, H*N) time-mix tensors and (B, H, N_v, N_k) state -> chunk_rwkv7's keyword arguments.

    Ours (PlasticCortex._tmix): S <- S * w[k] + (S @ -kk)(kk * a)^T + v k^T, out = S @ r, with S[v, k].
    fla (generalized_delta_rule/dplr): S' <- S' * exp(gk)[k] + b (a^T S') + k v^T, out = scale S'^T q, with S'[k, v].
    So S' = S^T, q = r, gk = log w, a_fla = -kk, b_fla = kk * a, scale = 1; layout [B, T, H, K]; state [B, H, K, V]."""
    B, T, C = r.shape
    H, N = S.shape[1], S.shape[-1]
    sh = lambda t: t.reshape(B, T, H, N)
    return dict(r=sh(r), w=sh(logw), k=sh(k), v=sh(v), a=sh(-kk), b=sh(kk * a), scale=1.0,
                initial_state=S.transpose(-1, -2), output_final_state=True)
ATT_SITES = ("r", "k", "v", "w1", "w2", "a1", "a2", "g1", "g2", "v1", "v2", "o")
FFN_SITES = ("fk", "fv")
NORM_KEYS = ("ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias", "ln_x.weight", "ln_x.bias", "ln_out.weight",
             "ln_out.bias")


def spike_count(x, theta, max_count: int = MAX_COUNT):
    """(signal sent to the projection, integer counts). Forward: theta * clamp(round(x / theta), -K, K). Backward:
    straight-through inside the clamp, zero outside, and LSQ-style into theta."""
    xs = x / theta
    xc = torch.clamp(xs, -max_count, max_count)
    c = torch.round(xc)
    return theta * (xc + (c - xc).detach()), c.detach()


class SpikingTwin(PlasticCortex):
    def __init__(self, cortex, *, spikes: bool = True, max_count: int = MAX_COUNT, trainable: bool = False,
                 rank: int = 8, layers=None, sites=None, ckpt: bool = False, dtype=torch.float32,
                 wkv: str = "loop", wkv_dtype=torch.float32, wkv_lower_bound: bool = False):
        super().__init__(cortex, rank=rank, layers=layers, dtype=dtype)
        if wkv not in WKV_KERNELS:
            raise ValueError(f"wkv must be one of {WKV_KERNELS}, not {wkv!r}")
        self.wkv = wkv                       # switch: "loop" (default, unchanged) or "fla" (training on CUDA only)
        self.wkv_dtype = wkv_dtype           # dtype of the tensors handed to the kernel (its state is fp32 inside)
        self.wkv_lower_bound = wkv_lower_bound   # pass lower_bound=LOG_DECAY_MIN (licenses fla's 64-chunk scheme)
        if wkv == "fla":
            fla_version()                    # metadata only; the import (which probes CUDA) waits for _wkv_fla
        self.spikes = spikes                 # switch: False = graded inputs (the teacher's math, the lesion read)
        self.max_count = max_count
        self.ckpt = ckpt                     # activation checkpointing per block (training only)
        self.trainable = trainable
        all_sites = [f"{i}.{s}" for i in range(self.L) for s in ATT_SITES + FFN_SITES] + ["head"]
        self.sites = set(all_sites if sites is None else sites)
        self.theta = {}
        for name in all_sites:
            self.theta[name] = torch.ones(self._site_dim(name), device=self.z["emb.weight"].device, dtype=dtype)
        self.stats = None                    # dict site -> [sum |count| per channel, n tokens, nonzero per channel]
        if trainable:
            self._make_trainable()

    # ---- parameters -------------------------------------------------------------------------------------------------
    def _site_dim(self, name):
        if name == "head":
            return self.C
        i, s = name.split(".")
        a, f = f"blocks.{i}.att.", f"blocks.{i}.ffn."
        key = {"r": a + "receptance.weight", "k": a + "key.weight", "v": a + "value.weight", "o": a + "output.weight",
               "w1": a + "w1", "w2": a + "w2", "a1": a + "a1", "a2": a + "a2", "g1": a + "g1", "g2": a + "g2",
               "v1": a + "v1", "v2": a + "v2", "fk": f + "key.weight", "fv": f + "value.weight"}[s]
        if s in ("v1", "v2") and int(i) == 0:
            return 1                         # layer 0 has no v gate (the package stores empty tensors there)
        return self.z[key].shape[0]

    def _make_trainable(self):
        self.W = torch.nn.ParameterDict({k.replace(".", "|"): torch.nn.Parameter(v.detach().clone())
                                         for k, v in self.z.items() if v.numel() > 0})
        self.Th = torch.nn.ParameterDict({k.replace(".", "|"): torch.nn.Parameter(v.detach().clone())
                                          for k, v in self.theta.items()})
        self.z = {k: (self.W[k.replace(".", "|")] if v.numel() > 0 else v) for k, v in self.z.items()}
        self.theta = {k: self.Th[k.replace(".", "|")] for k in self.theta}

    def weights_state(self) -> dict:
        """Plain CPU tensors of every weight and theta (what a trained twin is)."""
        return {"z": {k: v.detach().cpu().clone() for k, v in self.z.items()},
                "theta": {k: v.detach().cpu().clone() for k, v in self.theta.items()},
                "max_count": self.max_count, "sites": sorted(self.sites)}

    def load_weights(self, blob: dict):
        dev = self.z["emb.weight"].device
        with torch.no_grad():
            for k, v in blob["z"].items():
                if self.trainable and v.numel() > 0:
                    self.z[k].copy_(v.to(dev))
                else:
                    self.z[k] = v.to(dev, self.z[k].dtype)
            for k, v in blob["theta"].items():
                if self.trainable:
                    self.theta[k].copy_(v.to(dev))
                else:
                    self.theta[k] = v.to(dev, self.theta[k].dtype)
        self.max_count = blob.get("max_count", self.max_count)
        self.sites = set(blob.get("sites", self.sites))

    def random_init(self, seed: int, recipe: str = "rwkv"):
        """Route-B control: a seeded random twin, thetas recalibrated afterwards by the caller.

        recipe "rwkv" (E1 arm b, review 11-27 item 7): the RWKV-LM x070 init as recalled from its model.py (RWKV-LM is
        not on disk; not checked against the source [E]), in this package's parameterisation (decay
        exp(-e^-0.5 sigmoid(w0 + lora)) is RWKV-LM's exp(-exp(-softplus(-(w0 + lora)) - 0.5)), so w0 maps 1:1):
        token-shift mixes as layer-ratio ramps, w0 = per-channel decay-speed ramp + 0.5, LoRA down zeros / up orthogonal
        0.1, a0 0, v0 1, k_k 0.85, k_a 1, r_k 0, r/v uniform +-0.5/sqrt(C), k +-0.05/sqrt(C), att output and ffn value
        zero, ffn key +-0.5/sqrt(C), ln_x ((1+i)/L)^0.7, other norms 1 / 0, head orthogonal 0.5 sqrt(V/C), emb
        uniform +-1e-4 through a unit LayerNorm (RWKV-LM's ln0). Keys the recipe does not name fall back to "moment"
        and are listed in self.init_fallback.
        recipe "moment": every tensor from N(mean, std) of the teacher's own tensor; norms at 1 / 0 (not a trainable
        RWKV init: kept for the record)."""
        g = torch.Generator().manual_seed(seed)
        C, L = self.C, self.L
        self.init_fallback = []

        def moment(v):
            sd = float(v.float().std()) if v.numel() > 1 else 0.0
            return torch.randn(v.shape, generator=g) * sd + float(v.float().mean())

        def unif(v, a):
            return (torch.rand(v.shape, generator=g) * 2 - 1) * a

        def orth(v, gain):
            t = torch.empty(v.shape)
            torch.nn.init.orthogonal_(t, gain=gain, generator=g)
            return t

        def rwkv(k, v):
            if any(k.endswith(n) for n in NORM_KEYS):
                return torch.ones(v.shape) if k.endswith("weight") else torch.zeros(v.shape)
            if k == "emb.weight":
                return F.layer_norm(unif(v, 1e-4), (v.shape[-1],))
            if k == "head.weight":
                V = v.shape[1] if v.shape[0] == C else v.shape[0]
                return orth(v, 0.5 * (V / C) ** 0.5 if V > C else 0.5)
            parts = k.split(".")
            if parts[0] != "blocks":
                return None
            i, kind, name = int(parts[1]), parts[2], ".".join(parts[3:])
            r01 = i / max(L - 1, 1)
            r1 = 1.0 - i / L
            ddd = torch.arange(C, dtype=torch.float32) / C
            vec = lambda x: x.reshape(v.shape) if x.numel() == v.numel() else None
            if kind == "ffn":
                if name == "x_k":
                    return vec(1.0 - ddd ** (r1 ** 4))
                if name == "key.weight":
                    return unif(v, 0.5 / C ** 0.5)
                if name == "value.weight":
                    return torch.zeros(v.shape)
                return None
            mix = {"x_r": 0.2, "x_w": 0.9, "x_k": 0.7, "x_v": 0.7, "x_a": 0.9, "x_g": 0.2}
            if name in mix:
                return vec(1.0 - ddd ** (mix[name] * r1))
            if name == "w0":
                n = torch.arange(C, dtype=torch.float32)
                return vec(-7 + 5 * (n / max(C - 1, 1)) ** (0.85 + 1.0 * r01 ** 0.5) + 0.5)
            if name in ("w1", "a1", "g1", "v1", "a0", "r_k", "output.weight"):
                return torch.zeros(v.shape)
            if name in ("w2", "a2", "g2", "v2"):
                return orth(v, 0.1)
            if name == "v0":
                return torch.ones(v.shape)
            if name == "k_k":
                return torch.full(v.shape, 0.85)
            if name == "k_a":
                return torch.ones(v.shape)
            if name in ("receptance.weight", "value.weight"):
                return unif(v, 0.5 / C ** 0.5)
            if name == "key.weight":
                return unif(v, 0.05 / C ** 0.5)
            if name == "ln_x.weight":
                return torch.full(v.shape, ((1 + i) / L) ** 0.7)
            return None

        with torch.no_grad():
            for k in sorted(self.z):
                v = self.z[k]
                if v.numel() == 0:
                    continue
                if recipe == "moment":
                    new = torch.ones(v.shape) if (any(k.endswith(n) for n in NORM_KEYS) and k.endswith("weight")) \
                        else torch.zeros(v.shape) if any(k.endswith(n) for n in NORM_KEYS) else moment(v)
                elif recipe == "rwkv":
                    new = rwkv(k, v)
                    if new is None:
                        self.init_fallback.append(k)
                        new = moment(v)
                else:
                    raise ValueError(recipe)
                new = new.to(v.device, v.dtype)
                if self.trainable:
                    self.z[k].copy_(new)
                else:
                    self.z[k] = new

    @torch.no_grad()
    def calibrate(self, batches, quantile: float = 0.99):
        """theta per channel = q-quantile of |x| over the calibration tokens / K, read with spikes off (graded)."""
        rec = {}
        was, self.spikes = self.spikes, False
        self._calib = rec
        try:
            for toks in batches:
                self.forward_batch(toks)
        finally:
            self._calib = None
            self.spikes = was
        for name, xs in rec.items():
            x = torch.cat(xs, 0).abs().float()
            if x.shape[0] > 200_000:
                x = x[torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(0))[:200_000].to(x.device)]
            th = torch.quantile(x, quantile, dim=0) / self.max_count
            # a channel that is exactly zero on every calibration token (a zero-initialised LoRA down projection in
            # the random arm b) gets theta 1/K (counts span +-1) instead of the 1e-4 floor, which would clip it at
            # +-4e-4 and zero its straight-through gradient as soon as it moves. Teacher-weight channels are not zero.
            th = torch.where(th > 0, th, torch.full_like(th, 1.0 / self.max_count))
            th = torch.clamp(th, min=1e-4)
            if self.trainable:
                self.theta[name].copy_(th.to(self.theta[name].dtype))
            else:
                self.theta[name] = th.to(self.theta[name].dtype)

    _calib = None

    # ---- the spike function on one projection input --------------------------------------------------------------
    def _q(self, name, x):
        if self._calib is not None:
            self._calib.setdefault(name, []).append(x.detach().reshape(-1, x.shape[-1]))
            return x
        if not self.spikes or name not in self.sites:
            return x
        y, c = spike_count(x, self.theta[name].to(x.dtype), self.max_count)
        if self.stats is not None:
            cc = c.reshape(-1, c.shape[-1]).abs()
            s = self.stats.setdefault(name, [torch.zeros(c.shape[-1], device=c.device), 0,
                                             torch.zeros(c.shape[-1], device=c.device)])
            s[0] += cc.sum(0).float()
            s[1] += cc.shape[0]
            s[2] += (cc > 0).sum(0).float()
        return y

    # ---- batched v7 math (B, T, C); B = 1 reproduces PlasticCortex -------------------------------------------------
    def _tmix_b(self, i, x, x_prev, v_first, S):
        z, H, N, q = self.z, self.H, self.N, self._q
        a_ = f"blocks.{i}.att."
        n = f"{i}."
        B, T, C = x.shape
        xx = torch.cat((x_prev.unsqueeze(1), x[:, :-1]), 1) - x
        xr, xw, xk = x + xx * z[a_ + "x_r"], x + xx * z[a_ + "x_w"], x + xx * z[a_ + "x_k"]
        xv, xa, xg = x + xx * z[a_ + "x_v"], x + xx * z[a_ + "x_a"], x + xx * z[a_ + "x_g"]
        r = q(n + "r", xr) @ z[a_ + "receptance.weight"]
        w = q(n + "w2", torch.tanh(q(n + "w1", xw) @ z[a_ + "w1"])) @ z[a_ + "w2"]
        k = q(n + "k", xk) @ z[a_ + "key.weight"]
        v = q(n + "v", xv) @ z[a_ + "value.weight"]
        a = torch.sigmoid(z[a_ + "a0"] + q(n + "a2", q(n + "a1", xa) @ z[a_ + "a1"]) @ z[a_ + "a2"])
        g = q(n + "g2", torch.sigmoid(q(n + "g1", xg) @ z[a_ + "g1"])) @ z[a_ + "g2"]
        kk = F.normalize((k * z[a_ + "k_k"]).view(B, T, H, N), dim=-1, p=2.0).view(B, T, H * N)
        k = k * (1 + (a - 1) * z[a_ + "k_a"])
        if i == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(z[a_ + "v0"] + q(n + "v2", q(n + "v1", xv) @ z[a_ + "v1"]) @ z[a_ + "v2"])
        if self.wkv == "fla" and x.is_cuda and T > 1:
            y, S = self._wkv_fla(r, -0.606531 * torch.sigmoid((z[a_ + "w0"] + w).float()), k, v, kk, a, S)
            y = y.to(x.dtype)
        else:
            if self.wkv == "fla" and not x.is_cuda and T > 1:
                raise RuntimeError("wkv='fla' runs on CUDA only (Triton); use wkv='loop' on CPU")
            y, S = self._wkv_loop(B, T, H, N, r, w, k, v, kk, a, S, z, a_, x)
        y = F.group_norm(y.reshape(B * T, C), num_groups=H, weight=z[a_ + "ln_x.weight"], bias=z[a_ + "ln_x.bias"],
                         eps=64e-5).view(B, T, C)
        y = y + ((r * k * z[a_ + "r_k"]).view(B, T, H, N).sum(dim=-1, keepdim=True) * v.view(B, T, H, N)).view(B, T, C)
        return q(n + "o", y * g) @ z[a_ + "output.weight"], x[:, -1], S, v_first

    def _wkv_fla(self, r, logw, k, v, kk, a, S):
        """The state recurrence through fla's chunk_rwkv7 (mapping: wkv_fla_args). Returns y (B, T, H*N), S ours."""
        chunk_rwkv7, _ = fla_import()
        B, T, C = r.shape
        dt = self.wkv_dtype
        with torch.autocast(device_type=r.device.type, enabled=False):
            kw = wkv_fla_args(r.to(dt), logw.to(dt), k.to(dt), v.to(dt), kk.to(dt), a.to(dt), S.float())
            kw["initial_state"] = kw["initial_state"].contiguous()
            if self.wkv_lower_bound:
                kw["lower_bound"] = LOG_DECAY_MIN
            o, ht = chunk_rwkv7(**kw)
        return o.reshape(B, T, C), ht.float().transpose(-1, -2)

    def _wkv_loop(self, B, T, H, N, r, w, k, v, kk, a, S, z, a_, x):
        w = torch.exp(-0.606531 * torch.sigmoid((z[a_ + "w0"] + w).float()))
        S = S.float()
        rf, kf, vf, kkf, af = r.float(), k.float(), v.float(), kk.float(), a.float()
        outs = []
        with torch.autocast(device_type=x.device.type, enabled=False):   # the graded state stays fp32 under bf16 AMP
            for t in range(T):
                kt = kkf[:, t].view(B, H, N)
                sa = S @ (-kt).unsqueeze(-1)                               # state @ (-kk): (B, H, N, 1)
                S = S * w[:, t].view(B, H, 1, N) + sa * (kt * af[:, t].view(B, H, N)).unsqueeze(-2) \
                    + vf[:, t].view(B, H, N, 1) * kf[:, t].view(B, H, 1, N)
                outs.append((S @ rf[:, t].view(B, H, N, 1)).view(B, H * N))
        return torch.stack(outs, 1).to(x.dtype), S

    def _cmix_b(self, i, x, x_prev):
        f = f"blocks.{i}.ffn."
        xx = torch.cat((x_prev.unsqueeze(1), x[:, :-1]), 1) - x
        k_in = self._q(f"{i}.fk", x + xx * self.z[f + "x_k"])
        K, V = self.z[f + "key.weight"], self.z[f + "value.weight"]
        s = str(i)
        if self.enabled and s in self.A_k:
            h = self._q(f"{i}.fv", torch.relu(k_in @ K + (k_in @ self.A_k[s]) @ self.B_k[s]) ** 2)
            return h @ V + (h @ self.A_v[s]) @ self.B_v[s], x[:, -1]
        h = self._q(f"{i}.fv", torch.relu(k_in @ K) ** 2)
        return h @ V, x[:, -1]

    def _block(self, i, x, s_att, S, s_ffn, v_first):
        b = f"blocks.{i}."
        xx = F.layer_norm(x, (self.C,), weight=self.z[b + "ln1.weight"], bias=self.z[b + "ln1.bias"])
        xx, s_att, S, v_first = self._tmix_b(i, xx, s_att, v_first, S)
        x = x + xx
        xx = F.layer_norm(x, (self.C,), weight=self.z[b + "ln2.weight"], bias=self.z[b + "ln2.bias"])
        xx, s_ffn = self._cmix_b(i, xx, s_ffn)
        return x + xx, s_att, S, s_ffn, v_first

    def zero_state_b(self, B):
        dev, dt = self.z["emb.weight"].device, self.z["emb.weight"].dtype
        st = []
        for _ in range(self.L):
            st += [torch.zeros(B, self.C, device=dev, dtype=dt), torch.zeros(B, self.H, self.N, self.N, device=dev),
                   torch.zeros(B, self.C, device=dev, dtype=dt)]
        return st

    def forward_batch(self, tokens, state=None, hiddens: bool = False):
        """tokens (B, T) long -> logits (B, T, V), state (list, batched), [residual after each block, if hiddens]."""
        if self.source_tag is not None and self.enabled:
            raise NotImplementedError("source tag is not supported on the spiking twin")
        z = self.z
        tokens = torch.as_tensor(tokens, device=z["emb.weight"].device)
        B = tokens.shape[0]
        state = list(state) if state is not None else self.zero_state_b(B)
        x = z["emb.weight"][tokens]
        v_first = torch.empty_like(x)
        hs = []
        for i in range(self.L):
            args = (x, state[i * 3], state[i * 3 + 1], state[i * 3 + 2], v_first)
            if self.ckpt and torch.is_grad_enabled():
                out = checkpoint(self._block, i, *args, use_reentrant=False)
            else:
                out = self._block(i, *args)
            x, state[i * 3], state[i * 3 + 1], state[i * 3 + 2], v_first = out
            if hiddens:
                hs.append(x)
        x = F.layer_norm(x, (self.C,), weight=z["ln_out.weight"], bias=z["ln_out.bias"])
        logits = self._q("head", x) @ z["head.weight"]
        return (logits, state, hs) if hiddens else (logits, state)

    def forward(self, tokens, state=None):
        """The PlasticCortex interface: tokens (T,) -> logits (T, V), state (unbatched list)."""
        toks = torch.as_tensor(list(tokens) if not torch.is_tensor(tokens) else tokens, dtype=torch.long)
        st = None if state is None else [s.unsqueeze(0) for s in state]
        logits, st = self.forward_batch(toks.view(1, -1), st)
        return logits[0], [s[0] for s in st]

    # ---- read-outs ------------------------------------------------------------------------------------------------
    def spike_summary(self) -> dict:
        """Mean |count| per channel per token and the share of channels that never fired, over the recorded tokens."""
        if not self.stats:
            return {"mean_count": None, "silent_share": None, "sites": 0}
        tot = n = silent = ch = 0.0
        for s_sum, s_n, s_nz in self.stats.values():
            tot += float(s_sum.sum())
            n += s_n * s_sum.numel()
            silent += float((s_nz == 0).sum())
            ch += s_nz.numel()
        return {"mean_count": tot / max(n, 1), "silent_share": silent / max(ch, 1), "sites": len(self.stats)}


class TwinCortex(Cortex):
    """A SpikingTwin behind the Cortex API (no package model): read, read_nll, speak, matrix_delta, add_delta."""

    def __init__(self, twin: SpikingTwin, name: str = "spiking-twin"):
        self.twin = twin
        self.pipe = twin.cortex.pipe
        self.n_layers = twin.L
        self.model_path = getattr(twin.cortex, "model_path", None)
        self.model_name = name
        self.strategy = "spiking-twin"

    def forward(self, tokens, state):
        with torch.no_grad():
            logits, st = self.twin(list(tokens), state)
        return logits[-1], st

    def read_nll(self, text, state=None):
        ids = self.pipe.encode(text)
        st = copy.deepcopy(state) if state is not None else None
        with torch.no_grad():
            logits, new = self.twin(ids, st)
        lp = torch.log_softmax(logits.float(), -1)
        nll = -lp[:-1].gather(1, torch.tensor(ids[1:], device=lp.device)[:, None])[:, 0]
        return logits[-1], new, nll
