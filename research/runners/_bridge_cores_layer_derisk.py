"""BRIDGE CO-RESIDENCE de-risk #2 (per the scoping ladder): port ONE FULL Qwen2.5-0.5B decoder layer (layer 12)
end-to-end onto the LIVE SimulationBridge RF (resonate-and-fire complex-synapse) substrate + verify it reproduces
the B-1 PyTorch-spiking layer (every linear's RF matvec bit-exact + the validated graded nonlinearities).

Scoping: research/findings/2026-06-23-bridge-coresidence-qwen-faculty-scoping.md -- de-risk #2 (the LLaMA-stack
BLOCK consolidation). De-risk #1 (`_bridge_cores_qproj_derisk.json`) confirmed ONE q_proj on the RF bridge is
BIT-EXACT (max-err 4.58e-7, the C1 matvec transfers to Qwen weights). THIS step does the WHOLE decoder layer:

  ATTENTION block:
    input_layernorm (RMSNorm; graded read, B-0's exact-RMS divisive form) -> q/k/v_proj as RF matvecs + biases
    (host-add) -> RoPE on q/k (fixed, bit-exact) -> GQA attention with the B-1 graded softmax (the WIDE exp grid)
    -> o_proj as an RF matvec -> residual add (host).
  MLP block:
    post_attention_layernorm (RMSNorm) -> gate_proj + up_proj as RF matvecs -> SiLU-graded on the gate, multiply
    by up -> down_proj as an RF matvec -> residual add (host).

The 7 learned linears (q/k/v/o_proj, gate/up/down_proj) all install via the de-risk-#1 pattern: W = weight.T
(shape [D_in,D_out]) as complex synapses; kick a row (real, phase 0); resonate RF_NSTEPS @ lam=0/omega~0 so the
complex accumulator computes Re(Z_out)=nsteps*(a@W) EXACTLY; read Re(Z)/nsteps + bias (host-add, NOT a matvec).
RoPE = a deterministic trig rotation of q/k (0 learned params, host-exact). The nonlinearities (RMSNorm/SiLU/
softmax) are the B-1 calibrated-graded reads (reuse-by-import of the B-1 module's banks + spiking forwards).

COMPARE the RF-layer output to:
  (a) the B-1 PyTorch-SPIKING layer (SAME graded nonlinearities, but linears = exact torch matmuls). RF-vs-B1 is
      the PURE matvec-transfer claim -- with the SAME pool-noise generator seed reset before both forwards, the
      ONLY difference is RF-matvec-vs-torch-matmul, so this diff ~= the de-risk-#1 bit-exactness residual (~1e-6).
  (b) the EXACT ANN layer (`layer.forward(...)`, fp16). RF-vs-ANN reflects the graded-read noise (the already-
      characterized T-graded-read SEM) + fp16 + the RMSNorm L1/graded residual the scoping flags.

BAR (the scoping's de-risk-#2 gate):
  - each linear's RF-matvec bit-exact (~1e-6, the C1 / de-risk-#1 level) -- MEASURED per-linear, not asserted;
  - the full RF layer == the B-1 PyTorch-spiking layer to ~matvec precision (cosine ~1.0; the only diffs are the
    already-characterized graded-read noise, which is SHARED between the two when the generator is seed-matched);
  - the RMSNorm graded-vs-exact residual NOTED (the scoping's +0.037 / host-read fallback if it bites): we run
    RMSNorm BOTH graded AND host-exact and report the delta, so the residual is quantified, not hidden.

ANTI-CHEAT: a LESION control -- row-permute (shuffle) every installed RF weight; the RF-layer output must
COLLAPSE away from the true layer (cosine drops far) while each shuffled matvec still EXACTLY reproduces ITS OWN
a@W_shuf (proving the RF carries the installed weights, not a trivial pass). (The C1 `full_genf` lesion pattern.)

NO `sim/` edit (the RF path rf_set_complex_weights/rf_kick/rf_resonate_steps already exists from the C1
generative arc; the B-1 spiking ops are reused-by-import). GPU (SIM_BACKEND=cupy). FOREGROUND/blocking by design.
Usage:
  SIM_BACKEND=cupy python -m research.runners._bridge_cores_layer_derisk
  SIM_BACKEND=cupy python -m research.runners._bridge_cores_layer_derisk --layer 12 --n-rows 16 --T 16
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Reuse the EXACT RF bridge + RF operating point from the C1 generative arc / de-risk #1 (reuse-by-import).
from research.runners.rf_phasor_composer import _build_rf_bridge  # noqa: E402
# Reuse the B-1 spiking ops VERBATIM (the calibrated graded reads + the spiking forwards / banks). The B-1 module
# is `_grounded_lang_p1b_stepB1_forward_derisk.py`; we import its SpikeState/GradedRead/bank builders + the
# spiking RMSNorm/SiLU/softmax forwards so this layer's nonlinearities ARE the B-1 nonlinearities (no re-impl).
import research.runners._grounded_lang_p1b_stepB1_forward_derisk as B1  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CORPUS = _REPO / "data" / "corpus" / "tinystories.txt"
OUT = _REPO / "research" / "findings" / "raw" / "_bridge_cores_layer_derisk.json"

# The C1 / de-risk-#1 RF operating point (identical): lam=0 (no magnitude decay) + a HUGE period so omega~0
# (rotation/step ~ identity) => the complex accumulator computes Re(Z_out)=nsteps*(a@W) EXACTLY (max-err ~5e-7).
RF_PERIOD = 100000
RF_NSTEPS = 8
RF_LAMBDA = 0.0
EXACT_BAR = 1e-5     # the C1 / de-risk-#1 per-matvec bit-exact bar (de-risk #1 measured 4.58e-7)


def log(msg):
    print(f"[layer-rf] {msg}", flush=True)


# =====================================================================================================
# The RF matvec (the de-risk-#1 pattern, reused VERBATIM): install W=weight.T as complex synapses on an
# n=D_in+D_out RF bridge; per row kick z=row (real, phase 0), resonate RF_NSTEPS @ lam=0/omega~0, read
# Re(Z)[D_in:]/RF_NSTEPS = row@W. A per-shape cached bridge (build the CSR once per weight, the C1
# `_WEIGHT_CSR_CACHE` optimization) -- keyed by the (D_in,D_out) shape so the 2 same-shaped projections (q/o,
# k/v, gate/up) reuse the same bridge object (re-installing weights each time).
# =====================================================================================================
class RFMatvec:
    """One LIVE RF bridge per unique (D_in,D_out) shape (reused across same-shaped linears by re-installing W)."""

    def __init__(self, seed=42):
        self.seed = int(seed)
        self._bridges = {}     # (D_in,D_out) -> bridge
        self.n_installs = 0
        self.install_seconds = 0.0
        self.matvec_seconds = 0.0

    def _bridge_for(self, D_in, D_out):
        key = (int(D_in), int(D_out))
        if key not in self._bridges:
            n = D_in + D_out
            t0 = time.time()
            self._bridges[key] = _build_rf_bridge(n, seed=self.seed)
            log(f"    built RF bridge for shape {key} (n={n} neurons) in {time.time()-t0:.2f}s")
        return self._bridges[key]

    def __call__(self, W, rows):
        """W: [D_in, D_out] real (the weight.T install orientation). rows: [N, D_in] real activations.
        Returns [N, D_out] = rows @ W computed by the RF complex-synapse matvec (Re(Z)/nsteps)."""
        import cupy as cp
        D_in, D_out = W.shape
        n = D_in + D_out
        bridge = self._bridge_for(D_in, D_out)
        # install W as complex synapses (W_im=0) -- REPLACES any prior weights on this bridge.
        t0 = time.time()
        conns = [(D_in + nn, m, complex(float(W[m, nn]), 0.0))
                 for m in range(D_in) for nn in range(D_out) if W[m, nn] != 0.0]
        bridge.rf_set_complex_weights(conns)
        self.install_seconds += time.time() - t0
        self.n_installs += 1
        out = np.zeros((rows.shape[0], D_out), dtype=np.float64)
        inv = 1.0 / float(RF_NSTEPS)
        t0 = time.time()
        for r in range(rows.shape[0]):
            kick = np.zeros(n, dtype=np.complex128)
            kick[:D_in] = np.asarray(rows[r], dtype=np.float64)
            bridge.rf_kick(kick, period=int(RF_PERIOD), lam=float(RF_LAMBDA))
            bridge.rf_resonate_steps(int(RF_NSTEPS))
            re = cp.asnumpy(bridge.cp_membrane_potential_v[D_in:]).astype(np.float64)
            out[r] = re * inv
        self.matvec_seconds += time.time() - t0
        return out


def _metrics(rf, ref):
    """max-abs-err + mean/min cosine (per-row) + mean/max relative error (per-row) of rf vs ref, both (N,D)."""
    rf = np.asarray(rf, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    max_abs = float(np.max(np.abs(rf - ref))) if rf.size else float("nan")
    cos_rows, rel_rows = [], []
    for i in range(rf.shape[0]):
        a, b = rf[i], ref[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        cos_rows.append(float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan"))
        denom = nb if nb > 0 else 1.0
        rel_rows.append(float(np.linalg.norm(a - b) / denom))
    cos = [c for c in cos_rows if not math.isnan(c)]
    return {
        "max_abs_err": max_abs,
        "mean_cosine": float(np.mean(cos)) if cos else float("nan"),
        "min_cosine": float(np.min(cos)) if cos else float("nan"),
        "mean_rel_err": float(np.mean(rel_rows)) if rel_rows else float("nan"),
        "max_rel_err": float(np.max(rel_rows)) if rel_rows else float("nan"),
    }


# =====================================================================================================
# The B-1 graded nonlinearities as numpy ops over (N, D) host arrays (the layer forward runs host-side; the
# B-1 module's reads are torch-on-GPU, so we mirror its math with the SAME calibrated banks + the SAME
# pool-SEM noise model -- a numpy generator seeded so the RF and B-1-spiking forwards draw IDENTICAL noise,
# isolating the RF-matvec residual). The bank COEFFICIENTS come from the B-1 module's fitters (reuse-by-import).
# =====================================================================================================
class HostGradedRead:
    """A numpy mirror of B1.GradedRead: fn(x) ~ c0 + sum_k a_k*clip((x-knot)/READ_SCALE,0,1), + rate-coded
    graded-pool SEM noise sqrt(a(1-a))/sqrt(pool) per basis (B-1's honesty model). Same math, numpy, seeded."""

    def __init__(self, c0, a_k, knots, read_scale=B1.READ_SCALE):
        self.c0 = float(c0)
        self.a_k = np.asarray(a_k, dtype=np.float64)
        self.knots = np.asarray(knots, dtype=np.float64)
        self.read_scale = float(read_scale)

    def __call__(self, x, pool, rng):
        xf = np.asarray(x, dtype=np.float64)
        shp = xf.shape
        flat = xf.reshape(-1)
        a_cont = np.clip((flat[:, None] - self.knots[None, :]) / self.read_scale, 0.0, 1.0)  # (M,K)
        if pool and pool > 0:
            sem = np.sqrt(np.clip(a_cont * (1.0 - a_cont), 1e-6, None)) / math.sqrt(pool)
            a_cont = np.clip(a_cont + rng.standard_normal(a_cont.shape) * sem, 0.0, 1.0)
        return (self.c0 + a_cont @ self.a_k).reshape(shp)


def build_host_banks(silu_range, device):
    """Build the B-1 SiLU + wide-exp banks (off-line fit over the SAME ranges/grids B-1 uses) and mirror them
    host-side. We call B-1's fitters so the coefficients are byte-identical to the B-1 module."""
    silu_bank_t, silu_fd = B1.make_silu_bank(silu_range, device)     # torch bank (for parity) -> mirror it
    exp_bank_t, exp_fd = B1.make_exp_bank(device)
    # pull the fitted coefficients back out of the torch banks for the host mirror.
    silu_host = HostGradedRead(silu_bank_t.c0, silu_bank_t.a_k.detach().cpu().numpy(),
                               silu_bank_t.knots.detach().cpu().numpy())
    exp_host = HostGradedRead(exp_bank_t.c0, exp_bank_t.a_k.detach().cpu().numpy(),
                              exp_bank_t.knots.detach().cpu().numpy())
    return silu_host, silu_fd, exp_host, exp_fd


def graded_rmsnorm(x, weight, eps, pool_div, rng):
    """The B-1 spiking RMSNorm (numpy mirror of B1.spiking_rmsnorm_forward): weight*(x/(sqrt(mean(x^2)+eps)+SEM)).
    The divisor is the EXACT RMS with a rate-coded graded-pool SEM (1/sqrt(pool_div)), floored at half true RMS."""
    h = np.asarray(x, dtype=np.float64)
    var = (h ** 2).mean(axis=-1, keepdims=True)
    rms = np.sqrt(var + eps)
    D = rms.copy()
    if pool_div and pool_div > 0:
        spread = h.std(axis=-1, keepdims=True) / math.sqrt(pool_div)
        D = D + rng.standard_normal(D.shape) * spread
        D = np.maximum(D, 0.5 * rms)
    return weight[None, :] * (h / D)


def exact_rmsnorm(x, weight, eps):
    h = np.asarray(x, dtype=np.float64)
    var = (h ** 2).mean(axis=-1, keepdims=True)
    return weight[None, :] * (h / np.sqrt(var + eps))


def graded_silu(x, bank, pool, rng):
    return bank(x, pool, rng)


# =====================================================================================================
# RoPE (host-exact) + GQA attention with the B-1 graded softmax.
# =====================================================================================================
def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(q, k, cos, sin):
    """q:(H,S,d) k:(Hkv,S,d), cos/sin:(S,d). Broadcast over heads. Bit-exact host RoPE (matches transformers)."""
    cos_b = cos[None, :, :]
    sin_b = sin[None, :, :]
    q_emb = q * cos_b + rotate_half(q) * sin_b
    k_emb = k * cos_b + rotate_half(k) * sin_b
    return q_emb, k_emb


def graded_softmax_lastdim(scores, exp_bank, pool_softmax, pool_div, rng):
    """B-1 spiking softmax over the last dim (numpy mirror of B1._spiking_softmax_lastdim): max-subtract -> the
    wide-grid graded exp read -> masked keys contribute 0 -> divisive-sum (with a pooled-denominator SEM).
    `scores` has -inf at causal-masked positions (replaced by a large-negative finite by the caller)."""
    aw = np.asarray(scores, dtype=np.float64)
    m = aw.max(axis=-1, keepdims=True)
    shifted = aw - m                                            # all <= 0
    masked = shifted < (B1.EXP_GRID_LO - 0.5)                   # causal-masked positions (far below the grid)
    e = exp_bank(shifted, pool_softmax, rng)
    e = np.clip(e, 0.0, None)
    e = np.where(masked, 0.0, e)
    s = e.sum(axis=-1, keepdims=True)
    if pool_div and pool_div > 0:
        s_noise = s * (rng.standard_normal(s.shape) / math.sqrt(pool_softmax))
        s = np.maximum(s + s_noise, 0.5 * np.maximum(s, 1e-30))
    s = np.maximum(s, 1e-30)
    w = e / s
    if not np.isfinite(w).all():
        bad = ~np.isfinite(w).all(axis=-1, keepdims=True)
        valid = (~masked).astype(np.float64)
        unif = valid / np.maximum(valid.sum(axis=-1, keepdims=True), 1.0)
        w = np.where(bad, unif, w)
    return w


def run_attention(q_flat, k_flat, v_flat, cos, sin, scaling, Hq, Hkv, head_dim, exp_bank,
                  pool_softmax, pool_div, rng):
    """q_flat:(S,Hq*d) k_flat/v_flat:(S,Hkv*d). Full GQA causal attention with the B-1 graded softmax.
    cos/sin:(S,d). Returns attn_output (S, Hq*d). All host numpy."""
    S = q_flat.shape[0]
    n_rep = Hq // Hkv
    q = q_flat.reshape(S, Hq, head_dim).transpose(1, 0, 2)        # (Hq, S, d)
    k = k_flat.reshape(S, Hkv, head_dim).transpose(1, 0, 2)       # (Hkv, S, d)
    v = v_flat.reshape(S, Hkv, head_dim).transpose(1, 0, 2)       # (Hkv, S, d)
    # RoPE on q/k (host-exact)
    q, k = apply_rope(q, k, cos, sin)
    # GQA: repeat k/v to Hq heads (== repeat_kv)
    k = np.repeat(k, n_rep, axis=0)                               # (Hq, S, d)
    v = np.repeat(v, n_rep, axis=0)                               # (Hq, S, d)
    # scores = q @ k^T * scaling  -> (Hq, S, S)
    scores = np.matmul(q, k.transpose(0, 2, 1)) * scaling
    # causal mask: position i attends to <= i
    causal = np.triu(np.ones((S, S), dtype=bool), k=1)            # True above diagonal = masked
    scores = np.where(causal[None, :, :], -np.inf, scores)
    # graded softmax over the key dim (last). +/-inf masked -> shifted far below grid -> 0 weight.
    # replace -inf with a large negative finite so the max-subtract is well-defined.
    scores = np.where(np.isneginf(scores), -1.0e9, scores)
    w = graded_softmax_lastdim(scores, exp_bank, pool_softmax, pool_div, rng)  # (Hq, S, S)
    out = np.matmul(w, v)                                         # (Hq, S, d)
    out = out.transpose(1, 0, 2).reshape(S, Hq * head_dim)       # (S, Hq*d)
    return out


# =====================================================================================================
# The full decoder layer forward (host), parameterized by the linear backend (RF matvec OR exact torch->numpy).
# =====================================================================================================
def layer_forward(hidden, weights, cfg, linear_fn, *, rmsnorm_mode, silu_bank, exp_bank,
                  pool_silu, pool_div, pool_softmax, rng, cos, sin):
    """hidden: (S, D) numpy (the layer input = layer-12 residual-stream input).
    weights: dict of numpy arrays (the layer's params, with the *_install = weight.T orientation for linears).
    linear_fn(name, rows): returns rows @ W_install[name] (+ bias handled by caller). The ONLY swappable piece.
    rmsnorm_mode: 'graded' (B-1 spiking RMSNorm) or 'exact' (exact sqrt-RMS) or 'host_exact' (alias of exact).
    Returns (S, D) numpy = the decoder-layer output."""
    eps = cfg["eps"]; Hq = cfg["Hq"]; Hkv = cfg["Hkv"]; head_dim = cfg["head_dim"]; scaling = cfg["scaling"]

    def rms(x, w):
        if rmsnorm_mode == "graded":
            return graded_rmsnorm(x, w, eps, pool_div, rng)
        return exact_rmsnorm(x, w, eps)

    # ---- ATTENTION block ----
    residual = hidden
    h = rms(hidden, weights["ln1_w"])                            # input_layernorm (RMSNorm)
    q = linear_fn("q", h) + weights["q_bias"][None, :]          # q_proj (+ bias, host-add)
    k = linear_fn("k", h) + weights["k_bias"][None, :]          # k_proj
    v = linear_fn("v", h) + weights["v_bias"][None, :]          # v_proj
    attn = run_attention(q, k, v, cos, sin, scaling, Hq, Hkv, head_dim, exp_bank,
                         pool_softmax, pool_div, rng)            # RoPE + GQA + graded softmax
    attn_out = linear_fn("o", attn)                             # o_proj (no bias)
    hidden = residual + attn_out                                # residual add (host)

    # ---- MLP block ----
    residual = hidden
    h = rms(hidden, weights["ln2_w"])                           # post_attention_layernorm (RMSNorm)
    gate = linear_fn("gate", h)                                 # gate_proj (no bias)
    up = linear_fn("up", h)                                     # up_proj   (no bias)
    act = graded_silu(gate, silu_bank, pool_silu, rng)          # SiLU-graded on the gate
    mlp_in = act * up                                           # SwiGLU element-wise product
    mlp_out = linear_fn("down", mlp_in)                         # down_proj (no bias)
    hidden = residual + mlp_out                                 # residual add (host)
    return hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=12, help="which decoder layer (B-0/B-1 used layer 12)")
    ap.add_argument("--n-rows", type=int, default=16, help="number of real sequence positions (S) to drive")
    ap.add_argument("--T", type=int, default=16, help="rate-code pool budget (B-1 lowest-feasible T=16)")
    args = ap.parse_args()

    t0 = time.time()
    backend = os.environ.get("SIM_BACKEND", "auto")
    log(f"SIM_BACKEND={backend}")

    import torch
    import torch.nn.functional as F  # noqa: F401
    log(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
        f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"loading {MODEL_ID} (fp16, eager attention) ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16,
                                                 attn_implementation="eager").cuda().eval()
    device = next(model.parameters()).device
    mcfg = model.config
    log(f"loaded; {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params on {device}")

    L = args.layer
    layer = model.model.layers[L]
    attn = layer.self_attn
    eps = float(mcfg.rms_norm_eps)
    Hq = int(mcfg.num_attention_heads)
    Hkv = int(mcfg.num_key_value_heads)
    head_dim = int(getattr(mcfg, "head_dim", None) or mcfg.hidden_size // Hq)
    scaling = head_dim ** -0.5
    D = int(mcfg.hidden_size)
    log(f"layer {L}: D={D}, Hq={Hq}, Hkv={Hkv}, head_dim={head_dim}, scaling={scaling:.6f}, eps={eps:.1e}")

    # ---- capture the REAL layer-L INPUT hidden state + the cos/sin + the EXACT ANN layer OUTPUT ----
    captured = {}

    def layer_pre_hook(mod, args_, kwargs_):
        # the decoder layer's forward(hidden_states, ..., position_embeddings=...)
        hs = args_[0] if args_ else kwargs_.get("hidden_states")
        captured["layer_in"] = hs.detach()
        pe = kwargs_.get("position_embeddings")
        if pe is None and len(args_) >= 7:
            pe = args_[6]
        captured["pos_emb"] = pe
        return None

    def layer_fwd_hook(mod, args_, output):
        out = output[0] if isinstance(output, tuple) else output
        captured["layer_out"] = out.detach()

    hp = layer.register_forward_pre_hook(layer_pre_hook, with_kwargs=True)
    hf = layer.register_forward_hook(layer_fwd_hook)

    if CORPUS.exists():
        with open(CORPUS, "r", encoding="utf-8") as f:
            text = f.read()
        line = text[:2000]
    else:
        line = "Once upon a time there was a little girl who loved to read books in the garden every day."
    enc = tok(line, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**{kk: vv[:, :128] for kk, vv in enc.items()})
    hp.remove(); hf.remove()

    layer_in_full = captured["layer_in"][0]                      # (S, D)
    layer_out_full = captured["layer_out"][0]                    # (S, D)
    pos_emb = captured["pos_emb"]                                # (cos, sin), each (1, S, d)
    S_total = layer_in_full.shape[0]
    n_rows = min(args.n_rows, S_total)
    # use the FIRST n_rows positions (so the causal attention has a well-formed lower-triangular window).
    sl = slice(0, n_rows)
    hidden_in = layer_in_full[sl].to(torch.float64).cpu().numpy()   # (S, D)
    layer_out_ann = layer_out_full[sl].to(torch.float64).cpu().numpy()
    cos = pos_emb[0][0, sl].to(torch.float64).cpu().numpy()        # (S, d)
    sin = pos_emb[1][0, sl].to(torch.float64).cpu().numpy()        # (S, d)
    log(f"captured layer-{L} input/output: seq_len={S_total}, using first n_rows={n_rows}; "
        f"in range [{hidden_in.min():.3f},{hidden_in.max():.3f}], out range [{layer_out_ann.min():.3f},"
        f"{layer_out_ann.max():.3f}]; cos/sin shape {cos.shape}")

    # ---- extract the 7 linears (weight + bias) in the W=weight.T install orientation + the RMSNorm affines ----
    def w_install(lin):
        return np.ascontiguousarray(lin.weight.detach().to(torch.float64).cpu().numpy().T)  # (D_in, D_out)

    def bias_of(lin, d_out):
        return (lin.bias.detach().to(torch.float64).cpu().numpy() if lin.bias is not None
                else np.zeros(d_out, dtype=np.float64))

    W = {
        "q": w_install(attn.q_proj), "k": w_install(attn.k_proj), "v": w_install(attn.v_proj),
        "o": w_install(attn.o_proj), "gate": w_install(layer.mlp.gate_proj),
        "up": w_install(layer.mlp.up_proj), "down": w_install(layer.mlp.down_proj),
    }
    weights = {
        "ln1_w": layer.input_layernorm.weight.detach().to(torch.float64).cpu().numpy(),
        "ln2_w": layer.post_attention_layernorm.weight.detach().to(torch.float64).cpu().numpy(),
        "q_bias": bias_of(attn.q_proj, Hq * head_dim),
        "k_bias": bias_of(attn.k_proj, Hkv * head_dim),
        "v_bias": bias_of(attn.v_proj, Hkv * head_dim),
    }
    lin_shapes = {nm: tuple(W[nm].shape) for nm in W}
    log(f"linear install shapes (D_in,D_out): {lin_shapes}")
    total_nnz = sum(int(np.count_nonzero(W[nm])) for nm in W)
    log(f"total learned nnz across the 7 linears = {total_nnz:,}")

    # ---- build the B-1 banks (off-line fit over the B-1-measured ranges; reuse-by-import the B-1 fitters) ----
    silu_range = (-7.34375, 5.4140625)   # the B-1 measured SiLU-input range (stepB1_forward.json)
    silu_host, silu_fd, exp_host, exp_fd = build_host_banks(silu_range, device)
    log(f"SiLU bank: grid {silu_fd['grid']} knots {silu_fd['n_knots']} fit-max-err {silu_fd['fit_max_err_grid']:.5f}")
    log(f"exp  bank: grid {exp_fd['grid']} knots {exp_fd['n_knots']} fit-max-err {exp_fd['fit_max_err_grid']:.5f} "
        f"(WIDE; B-1 logit-min -102.5 handled by the masked-key floor)")

    # ---- pool budget from T (B-1's T->pool mapping) ----
    T = args.T
    pool_silu = B1.POOL_BASE * T
    pool_div = B1.POOL_BASE * T
    pool_softmax = B1.POOL_BASE_SM * T
    log(f"T={T} -> pool_silu={pool_silu}, pool_div={pool_div}, pool_softmax={pool_softmax}")

    cfg = {"eps": eps, "Hq": Hq, "Hkv": Hkv, "head_dim": head_dim, "scaling": scaling}

    # =================================================================================================
    # PER-LINEAR bit-exactness (the de-risk-#1 claim, now for all 7 linears of the layer).
    # We exercise each on the SAME real activation rows it would see -- but the cheapest faithful probe is the
    # rms-normed layer input for q/k/v (their real input), and a separate real input for gate/up/down/o.
    # For a clean per-op number, drive each linear with the REAL pre-linear activation captured from the ANN
    # layer's own sub-forward. We reconstruct those host-side (exact-RMS for q/k/v's input; exact attention for
    # o's input; exact SiLU*up for down's input) so the per-linear RF read is measured on its true operand.
    # =================================================================================================
    rfmv = RFMatvec(seed=42)
    log("=== PER-LINEAR RF bit-exactness (each linear's RF matvec vs exact a@W on its REAL operand) ===")

    # exact-RMS layer input -> q/k/v operand
    h_attn_in = exact_rmsnorm(hidden_in, weights["ln1_w"], eps)              # (S, D)
    # exact q/k/v (float) for the exact attention -> o operand
    q_ex = h_attn_in @ W["q"] + weights["q_bias"][None, :]
    k_ex = h_attn_in @ W["k"] + weights["k_bias"][None, :]
    v_ex = h_attn_in @ W["v"] + weights["v_bias"][None, :]
    # exact attention (float softmax) -> o operand
    attn_ex = _exact_attention(q_ex, k_ex, v_ex, cos, sin, scaling, Hq, Hkv, head_dim)
    # exact MLP path -> down operand
    h_mlp_in = exact_rmsnorm(hidden_in + (attn_ex @ W["o"]), weights["ln2_w"], eps)
    gate_ex = h_mlp_in @ W["gate"]
    up_ex = h_mlp_in @ W["up"]
    silu_ex = gate_ex * (1.0 / (1.0 + np.exp(-gate_ex)))
    down_in_ex = silu_ex * up_ex

    per_linear_operands = {
        "q": h_attn_in, "k": h_attn_in, "v": h_attn_in, "o": attn_ex,
        "gate": h_mlp_in, "up": h_mlp_in, "down": down_in_ex,
    }
    per_linear = {}
    worst_linear_err = 0.0
    for nm in ("q", "k", "v", "o", "gate", "up", "down"):
        operand = per_linear_operands[nm]
        ref = operand @ W[nm]                                                # exact a@W (fp64)
        rf = rfmv(W[nm], operand)                                            # RF matvec
        m = _metrics(rf, ref)
        per_linear[nm] = {"shape": list(lin_shapes[nm]), **m}
        worst_linear_err = max(worst_linear_err, m["max_abs_err"])
        log(f"  {nm:>4}_proj {str(lin_shapes[nm]):>14}: RF vs a@W  max-abs {m['max_abs_err']:.3e}  "
            f"cos {m['mean_cosine']:.8f}  rel {m['mean_rel_err']:.3e}")
    all_linears_bitexact = worst_linear_err <= EXACT_BAR
    log(f"  -> worst per-linear max-abs-err = {worst_linear_err:.3e}  (C1/de-risk-#1 bar {EXACT_BAR:.0e}) -> "
        f"{'BIT-EXACT' if all_linears_bitexact else 'OVER BAR'}")

    # =================================================================================================
    # FULL-LAYER forwards. The pool-noise generator is seed-RESET before the RF and the B-1-spiking forwards so
    # they draw IDENTICAL graded-read noise -> RF-vs-B1 == the pure RF-matvec residual (the bit-exact claim);
    # both-vs-ANN == the graded-read fidelity (the already-characterized T-read noise + fp16).
    # =================================================================================================
    log("=== FULL-LAYER forwards (RF-linears vs B-1-spiking-linears vs exact ANN) ===")
    NOISE_SEED = 7

    def rf_linear_fn(name, rows):
        return rfmv(W[name], rows)

    def torch_linear_fn(name, rows):
        return rows @ W[name]      # exact numpy matmul == the B-1 PyTorch matmul at fp64

    # (1) RF layer, RMSNorm GRADED (the B-1 spiking RMSNorm)
    rng = np.random.default_rng(NOISE_SEED)
    t_rf = time.time()
    out_rf_graded = layer_forward(hidden_in, weights, cfg, rf_linear_fn, rmsnorm_mode="graded",
                                  silu_bank=silu_host, exp_bank=exp_host, pool_silu=pool_silu,
                                  pool_div=pool_div, pool_softmax=pool_softmax, rng=rng, cos=cos, sin=sin)
    rf_layer_seconds = time.time() - t_rf

    # (2) B-1 PyTorch-spiking layer (exact linears, SAME graded nonlinearities, SAME noise seed)
    rng = np.random.default_rng(NOISE_SEED)
    out_b1_graded = layer_forward(hidden_in, weights, cfg, torch_linear_fn, rmsnorm_mode="graded",
                                  silu_bank=silu_host, exp_bank=exp_host, pool_silu=pool_silu,
                                  pool_div=pool_div, pool_softmax=pool_softmax, rng=rng, cos=cos, sin=sin)

    # (3) RF layer, RMSNorm EXACT (host sqrt-RMS) -- the scoping resolution (ii) host-read fallback
    rng = np.random.default_rng(NOISE_SEED)
    out_rf_exactrms = layer_forward(hidden_in, weights, cfg, rf_linear_fn, rmsnorm_mode="exact",
                                    silu_bank=silu_host, exp_bank=exp_host, pool_silu=pool_silu,
                                    pool_div=pool_div, pool_softmax=pool_softmax, rng=rng, cos=cos, sin=sin)

    # (4) B-1 spiking layer, RMSNorm EXACT (the matched reference for (3))
    rng = np.random.default_rng(NOISE_SEED)
    out_b1_exactrms = layer_forward(hidden_in, weights, cfg, torch_linear_fn, rmsnorm_mode="exact",
                                    silu_bank=silu_host, exp_bank=exp_host, pool_silu=pool_silu,
                                    pool_div=pool_div, pool_softmax=pool_softmax, rng=rng, cos=cos, sin=sin)

    # ---- the load-bearing comparisons ----
    m_rf_vs_b1_graded = _metrics(out_rf_graded, out_b1_graded)       # PURE RF-matvec transfer (graded RMSNorm)
    m_rf_vs_b1_exact = _metrics(out_rf_exactrms, out_b1_exactrms)    # PURE RF-matvec transfer (exact RMSNorm)
    m_rf_vs_ann_graded = _metrics(out_rf_graded, layer_out_ann)      # RF layer vs the EXACT ANN layer
    m_b1_vs_ann_graded = _metrics(out_b1_graded, layer_out_ann)      # B-1 spiking layer vs the EXACT ANN layer
    m_rf_vs_ann_exact = _metrics(out_rf_exactrms, layer_out_ann)     # RF layer (exact RMSNorm) vs ANN
    m_b1_vs_ann_exact = _metrics(out_b1_exactrms, layer_out_ann)     # B-1 (exact RMSNorm) vs ANN
    # RMSNorm graded-vs-exact residual (the scoping's note): the B-1-spiking-layer output under graded vs exact RMS.
    m_rmsnorm_residual = _metrics(out_b1_graded, out_b1_exactrms)

    log(f"  RF-vs-B1 (graded RMSNorm)  : max-abs {m_rf_vs_b1_graded['max_abs_err']:.3e}  "
        f"cos {m_rf_vs_b1_graded['mean_cosine']:.8f}   [PURE matvec transfer; ~de-risk-#1 residual]")
    log(f"  RF-vs-B1 (exact  RMSNorm)  : max-abs {m_rf_vs_b1_exact['max_abs_err']:.3e}  "
        f"cos {m_rf_vs_b1_exact['mean_cosine']:.8f}")
    log(f"  RF-vs-ANN (graded RMSNorm) : max-abs {m_rf_vs_ann_graded['max_abs_err']:.3e}  "
        f"cos {m_rf_vs_ann_graded['mean_cosine']:.6f}   [graded-read fidelity + fp16]")
    log(f"  B1-vs-ANN (graded RMSNorm) : max-abs {m_b1_vs_ann_graded['max_abs_err']:.3e}  "
        f"cos {m_b1_vs_ann_graded['mean_cosine']:.6f}   [the B-1 ceiling; RF should ~match this]")
    log(f"  RF-vs-ANN (exact  RMSNorm) : max-abs {m_rf_vs_ann_exact['max_abs_err']:.3e}  "
        f"cos {m_rf_vs_ann_exact['mean_cosine']:.6f}")
    log(f"  B1-vs-ANN (exact  RMSNorm) : max-abs {m_b1_vs_ann_exact['max_abs_err']:.3e}  "
        f"cos {m_b1_vs_ann_exact['mean_cosine']:.6f}")
    log(f"  RMSNorm graded-vs-exact RESIDUAL (B-1 layer, graded vs exact RMS): "
        f"max-abs {m_rmsnorm_residual['max_abs_err']:.3e}  cos {m_rmsnorm_residual['mean_cosine']:.6f}  "
        f"(the scoping's L1/graded RMSNorm note, quantified)")

    # =================================================================================================
    # ANTI-CHEAT: LESION (shuffle every installed RF weight) -> the RF-layer output must DIVERGE from the true
    # layer, while each shuffled matvec still EXACTLY reproduces ITS OWN a@W_shuf. (The C1 full_genf lesion.)
    # =================================================================================================
    log("=== ANTI-CHEAT lesion (row-permute every installed RF weight) ===")
    rng_perm = np.random.default_rng(98765)
    W_shuf = {nm: np.ascontiguousarray(W[nm][rng_perm.permutation(W[nm].shape[0]), :]) for nm in W}

    def rf_linear_fn_shuf(name, rows):
        return rfmv(W_shuf[name], rows)

    rng = np.random.default_rng(NOISE_SEED)
    out_rf_lesion = layer_forward(hidden_in, weights, cfg, rf_linear_fn_shuf, rmsnorm_mode="exact",
                                  silu_bank=silu_host, exp_bank=exp_host, pool_silu=pool_silu,
                                  pool_div=pool_div, pool_softmax=pool_softmax, rng=rng, cos=cos, sin=sin)
    m_lesion_vs_true = _metrics(out_rf_lesion, out_b1_exactrms)      # lesioned RF layer (FULL output) vs true
    # The decoder layer has TWO residual adds: the (large) layer-input residual passes through UNCHANGED, so the
    # FULL output stays partly correlated even with scrambled weights (the residual IS supposed to pass through).
    # The load-bearing lesion signal is on the COMPUTED DELTA (output - residual_input) = the attn+mlp contribution
    # the weights actually drive -- that must collapse to ~chance, like the C1 full_genf lesion.
    delta_true = out_b1_exactrms - hidden_in
    delta_lesion = out_rf_lesion - hidden_in
    m_lesion_delta = _metrics(delta_lesion, delta_true)             # the attn+mlp delta -> must collapse to ~chance
    # also confirm one shuffled matvec still exactly reproduces its own a@W_shuf (RF not broken, weights are).
    ref_shuf_q = h_attn_in @ W_shuf["q"]
    rf_shuf_q = rfmv(W_shuf["q"], h_attn_in)
    m_shuf_self_q = _metrics(rf_shuf_q, ref_shuf_q)
    log(f"  lesioned RF layer vs TRUE B-1 layer (FULL output): max-abs {m_lesion_vs_true['max_abs_err']:.3e}  "
        f"cos {m_lesion_vs_true['mean_cosine']:.4f}  (residual-dominated; the large input passes through both)")
    log(f"  lesioned RF layer vs TRUE (attn+mlp DELTA, residual removed): max-abs {m_lesion_delta['max_abs_err']:.3e} "
        f" cos {m_lesion_delta['mean_cosine']:.4f}  (must COLLAPSE to ~chance -- the load-bearing lesion signal)")
    log(f"  shuffled q matvec vs its OWN a@W_shuf: max-abs {m_shuf_self_q['max_abs_err']:.3e}  "
        f"(must stay EXACT -> proves the RF carries the installed weights)")

    # =================================================================================================
    # VERDICT
    # =================================================================================================
    rf_transfers = (m_rf_vs_b1_graded["max_abs_err"] <= 1e-4          # full-layer RF==B1 to ~matvec precision
                    and m_rf_vs_b1_graded["mean_cosine"] >= 0.9999
                    and m_rf_vs_b1_exact["mean_cosine"] >= 0.9999)
    # RF should ~match the B-1 ceiling against the ANN (within the graded-read noise; allow a small slack).
    rf_matches_b1_ceiling = (abs(m_rf_vs_ann_graded["mean_cosine"] - m_b1_vs_ann_graded["mean_cosine"]) <= 0.02)
    # The load-bearing lesion signal is the attn+mlp DELTA collapse (the residual passthrough keeps the full
    # output correlated by construction). The delta must drop to ~chance AND the max-abs must blow up vs the
    # bit-exact true-layer residual.
    lesion_diverges = (m_lesion_delta["mean_cosine"] < 0.3
                       and m_lesion_vs_true["max_abs_err"] > 1000 * max(m_rf_vs_b1_exact["max_abs_err"], 1e-9))
    lesion_self_exact = (m_shuf_self_q["max_abs_err"] <= EXACT_BAR)

    if all_linears_bitexact and rf_transfers and lesion_diverges and lesion_self_exact:
        verdict = "GO"
        tail = (f"the FULL Qwen layer-{L} ports onto the LIVE RF bridge: all 7 linears bit-exact "
                f"(worst per-linear max-abs {worst_linear_err:.2e} <= {EXACT_BAR:.0e}); the RF-layer output == the "
                f"B-1 PyTorch-spiking layer to matvec precision (RF-vs-B1 graded max-abs "
                f"{m_rf_vs_b1_graded['max_abs_err']:.2e}, cos {m_rf_vs_b1_graded['mean_cosine']:.8f}); RF tracks the "
                f"B-1 ceiling vs the ANN (RF-vs-ANN cos {m_rf_vs_ann_graded['mean_cosine']:.4f} ~ B1-vs-ANN "
                f"{m_b1_vs_ann_graded['mean_cosine']:.4f}). RMSNorm graded-vs-exact residual = "
                f"{m_rmsnorm_residual['max_abs_err']:.2e}/cos {m_rmsnorm_residual['mean_cosine']:.5f} "
                f"(small; host-exact fallback available, resolution (ii)). Lesion: the attn+mlp DELTA collapses "
                f"(cos {m_lesion_delta['mean_cosine']:.3f}, full-output cos {m_lesion_vs_true['mean_cosine']:.3f} "
                f"stays high only via the residual passthrough) while the shuffled matvec stays exact -> the RF "
                f"carries the computation. => the full 24-layer port (de-risk #3, the wall-clock number) is "
                f"UNBLOCKED. NO `sim/` edit (reuse-by-import).")
    elif all_linears_bitexact and m_rf_vs_b1_graded["mean_cosine"] >= 0.999:
        verdict = "GO_WITH_CAVEAT"
        tail = (f"the 7 linears are bit-exact (worst {worst_linear_err:.2e}) and the full RF layer ~matches the B-1 "
                f"spiking layer (RF-vs-B1 cos {m_rf_vs_b1_graded['mean_cosine']:.6f}), but a gate is soft: "
                f"rf_transfers={rf_transfers}, rf_matches_b1_ceiling={rf_matches_b1_ceiling}, "
                f"lesion_diverges={lesion_diverges}. Inspect the flagged item; the matvec transfer itself holds.")
    else:
        verdict = "HONEST_RESIDUAL"
        tail = (f"a piece diverged. worst per-linear max-abs {worst_linear_err:.3e} "
                f"({'<=' if all_linears_bitexact else '>'} {EXACT_BAR:.0e}); RF-vs-B1 (graded) cos "
                f"{m_rf_vs_b1_graded['mean_cosine']:.6f}, max-abs {m_rf_vs_b1_graded['max_abs_err']:.3e}. If the "
                f"per-linear is exact but the full-layer diverges, it is an OP-COMPOSITION issue (RoPE placement / "
                f"residual / softmax mask / GQA repeat); if a per-linear is over bar it is a shape/scale/orientation "
                f"issue. RMSNorm residual {m_rmsnorm_residual['max_abs_err']:.3e} -- if THAT dominates, use the "
                f"host-exact RMSNorm (resolution ii). No `sim/` edit was added.")

    verdict_line = (
        f"bridge_cores_layer: FULL Qwen layer-{L} on the LIVE RF bridge -> "
        f"7 linears bit-exact (worst max-abs {worst_linear_err:.2e}, bar {EXACT_BAR:.0e}); "
        f"full-layer RF-vs-B1(spiking) cos {m_rf_vs_b1_graded['mean_cosine']:.8f} (max-abs "
        f"{m_rf_vs_b1_graded['max_abs_err']:.2e}); RF-vs-ANN cos {m_rf_vs_ann_graded['mean_cosine']:.5f} "
        f"(B1-vs-ANN {m_b1_vs_ann_graded['mean_cosine']:.5f}); RMSNorm graded-vs-exact residual cos "
        f"{m_rmsnorm_residual['mean_cosine']:.5f}; lesion attn+mlp-delta cos {m_lesion_delta['mean_cosine']:.3f} "
        f"(full-output {m_lesion_vs_true['mean_cosine']:.3f}; diverges={lesion_diverges}) -> {verdict}. {tail}")

    result = {
        "probe": "bridge_coresidence_derisk2_qwen_full_decoder_layer_rf",
        "resolves": "de-risk #2 (scoping 2026-06-23): port ONE FULL Qwen2.5-0.5B decoder layer (layer 12) onto the "
                    "LIVE SimulationBridge RF substrate (all 7 linears as RF matvecs + RMSNorm/SiLU/softmax as the "
                    "B-1 graded reads + RoPE host-exact) and verify it reproduces the B-1 PyTorch-spiking layer "
                    "(matvecs bit-exact + the graded nonlinearities at their validated fidelity).",
        "model_id": MODEL_ID,
        "layer": L,
        "arch": {"D": D, "Hq": Hq, "Hkv": Hkv, "head_dim": head_dim, "scaling": scaling, "eps": eps},
        "n_rows_tested": int(n_rows), "seq_len": int(S_total), "T": int(T),
        "pools": {"silu": pool_silu, "div": pool_div, "softmax": pool_softmax},
        "rf_operating_point": {"period": RF_PERIOD, "nsteps": RF_NSTEPS, "lambda": RF_LAMBDA,
                               "read": "Re(Z_out)/nsteps = a @ W (W=linear.weight.T); biases host-add on the read"},
        "exact_bar": EXACT_BAR,
        "linear_install_shapes": {nm: list(lin_shapes[nm]) for nm in lin_shapes},
        "total_linear_nnz": int(total_nnz),
        "silu_fit": silu_fd, "exp_fit": exp_fd,
        "mechanism": "reuse-by-import: the C1/de-risk-#1 RF exact-matvec (rf_set_complex_weights / rf_kick / "
                     "rf_resonate_steps + the Re(Z)/nsteps read; W=weight.T, biases host-add) for all 7 linears, "
                     "ONE RF bridge per unique (D_in,D_out) shape; the B-1 calibrated graded reads "
                     "(RMSNorm/SiLU/wide-exp-softmax) imported from _grounded_lang_p1b_stepB1_forward_derisk; "
                     "RoPE applied host-exact (a fixed trig rotation, 0 learned params). NO `sim/` edit.",
        "per_linear_bit_exactness": per_linear,
        "worst_per_linear_max_abs_err": worst_linear_err,
        "all_linears_bit_exact": bool(all_linears_bitexact),
        "full_layer": {
            "rf_vs_b1_spiking_graded_rms": m_rf_vs_b1_graded,
            "rf_vs_b1_spiking_exact_rms": m_rf_vs_b1_exact,
            "rf_vs_ann_graded_rms": m_rf_vs_ann_graded,
            "b1_vs_ann_graded_rms": m_b1_vs_ann_graded,
            "rf_vs_ann_exact_rms": m_rf_vs_ann_exact,
            "b1_vs_ann_exact_rms": m_b1_vs_ann_exact,
        },
        "rmsnorm_graded_vs_exact_residual": m_rmsnorm_residual,
        "rmsnorm_note": ("the scoping (1c) flags Qwen's RMSNorm needs exact sqrt(mean x^2) whereas the shipped "
                         "on-bridge divisive op is L1/mean-abs (+0.037 residual on a Gen-F LayerNorm). HERE the "
                         "RMSNorm is the B-1 EXACT-RMS graded read (divisor = exact sqrt with a rate-coded SEM), so "
                         "the graded-vs-exact residual is the SEM-noise-only delta reported above. The host-exact "
                         "RMSNorm fallback (resolution ii) is the zero-edit option if the SEM bites at 24 layers."),
        "anti_cheat_lesion": {
            "lesioned_rf_layer_vs_true_b1_layer_FULL_output": m_lesion_vs_true,
            "lesioned_rf_layer_vs_true_attn_mlp_DELTA": m_lesion_delta,
            "shuffled_q_matvec_vs_own_a_at_W_shuf": m_shuf_self_q,
            "lesion_diverges": bool(lesion_diverges),
            "lesion_self_exact": bool(lesion_self_exact),
            "note": "row-permuting every installed RF weight. The FULL output stays partly correlated by "
                    "construction (the decoder layer's TWO residual adds pass the large layer-input through "
                    "unchanged); the load-bearing signal is the attn+mlp DELTA (output - residual_input) = the "
                    "contribution the weights drive, which must COLLAPSE to ~chance. Each shuffled matvec still "
                    "exactly reproduces its own a@W_shuf -- proving the RF carries the installed weights, not a "
                    "trivial pass.",
        },
        "timing": {
            "rf_install_seconds": round(rfmv.install_seconds, 3),
            "rf_matvec_seconds": round(rfmv.matvec_seconds, 3),
            "rf_n_installs": rfmv.n_installs,
            "one_full_layer_rf_seconds": round(rf_layer_seconds, 3),
        },
        "sim_edit_needed": False,
        "verdict": verdict,
        "verdict_line": verdict_line,
        "total_seconds": round(time.time() - t0, 2),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False,
                  default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else o)

    print("\n" + "=" * 100, flush=True)
    print(verdict_line, flush=True)
    print("=" * 100, flush=True)
    log(f"wrote {OUT}")
    log(f"DONE in {result['total_seconds']:.1f}s")
    return result


def _exact_attention(q_flat, k_flat, v_flat, cos, sin, scaling, Hq, Hkv, head_dim):
    """Exact (float softmax) GQA attention -- used ONLY to reconstruct the per-linear operands for o_proj.
    Mirrors run_attention but with the exact softmax (no graded read, no pool noise)."""
    S = q_flat.shape[0]
    n_rep = Hq // Hkv
    q = q_flat.reshape(S, Hq, head_dim).transpose(1, 0, 2)
    k = k_flat.reshape(S, Hkv, head_dim).transpose(1, 0, 2)
    v = v_flat.reshape(S, Hkv, head_dim).transpose(1, 0, 2)
    q, k = apply_rope(q, k, cos, sin)
    k = np.repeat(k, n_rep, axis=0)
    v = np.repeat(v, n_rep, axis=0)
    scores = np.matmul(q, k.transpose(0, 2, 1)) * scaling
    causal = np.triu(np.ones((S, S), dtype=bool), k=1)
    scores = np.where(causal[None, :, :], -np.inf, scores)
    m = scores.max(axis=-1, keepdims=True)
    e = np.exp(scores - m)
    e = np.where(np.isnan(e), 0.0, e)
    w = e / e.sum(axis=-1, keepdims=True)
    out = np.matmul(w, v).transpose(1, 0, 2).reshape(S, Hq * head_dim)
    return out


if __name__ == "__main__":
    main()
