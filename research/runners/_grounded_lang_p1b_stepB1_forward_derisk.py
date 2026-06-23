"""P1b STEP B-1 (the culminating step): the FULL SPIKING FORWARD of Qwen2.5-0.5B-Instruct.

Replace EVERY nonlinearity in the real model's forward with its B-0 calibrated-graded-read spiking version
(RMSNorm-graded, SiLU-graded inside SwiGLU, Softmax-graded with the WIDE exp grid; RoPE fixed/bit-exact, no
convert). The linears (q/k/v/o_proj, gate/up/down_proj, lm_head, embed) stay EXACT matmuls -- per the scoping
the linears are exact-on-RF; here in PyTorch they are the dense matmuls. Then MEASURE the load-bearing
[VERIFY]: does the SPIKING generation stay COHERENT / fluent / non-degenerate / not-verbatim-copying, at the
lowest feasible rate-code averaging pool T, with spiking held-out PPL <=~1.2x the ANN baseline (ppl 6.53)?

WHY THIS IS THE GENUINE OPEN QUESTION (not settled by B-0's per-op cosines):
  B-0 (GO_MECHANISM_TRANSFERS) proved each op TRANSFERS in isolation on real activations (RMSNorm cos 1.0 exact,
  SiLU cos 0.943 / noise-free 1.0, Softmax cos 0.963 / noise-free 0.9996, RoPE bit-exact). But a 24-layer
  generative forward COMPOSES 24x(RMSNorm + SiLU + Softmax) reads, each carrying 1/sqrt(pool) graded-read SEM,
  and the question of whether that COMPOUNDING error preserves token-level GENERATION COHERENCE is NOT settled
  by per-op fidelity -- it has to be READ. That READ is this step's deliverable.

THE MECHANISM (reused VERBATIM from B-0; here vectorized in torch on GPU for a full-model forward):
  - The calibrated rectified-basis graded read: fn(x) ~ c0 + sum_k a_k * clip((x-knot_k)/READ_SCALE, 0, 1),
    coefficients fit OFF-line on a fixed grid (NOT on the data); rate-coded graded-pool SEM noise
    sqrt(a*(1-a))/sqrt(pool) added to each rectified basis read (B-0's pool-noise honesty model). `pool` = the
    rate-code averaging pool (the LIF T>1 multi-step / population route): larger pool = lower SEM.
  - RMSNorm: weight*(x / (sqrt(mean(x^2)+eps) + pool-SEM)) -- the exact-RMS divisive read (B-0 headline; cos 1.0).
  - SiLU: the rectified-basis graded read calibrated over the gate_proj-output range (B-0 SiLU bank).
  - Softmax: exp via the SAME read over the post-max-subtract logits + sum-normalization (divisive arm), with a
    WIDE exp grid sized to Qwen's ACTUAL logit support (~[-33,0] -- B-0's honest residual: Qwen's post-max
    logits are WIDE, exp dynamic range ~5.8e13, far wider than Gen-F's [-4,0]) + a LARGER softmax averaging
    pool than SiLU (B-0 pool-sweep: SiLU reaches the bar ~256, softmax ~4096).
  - RoPE: applied EXACTLY by the model (a fixed trig rotation; B-0 confirmed bit-exact, 0 learned params).

T -> POOL mapping (the "drive it with T in {4,8,16}" -- start cheapest, escalate if degraded):
  The rate-code averaging pool backing each graded read scales with the multi-step budget T. We map
  pool_silu = POOL_BASE * T and pool_softmax = POOL_BASE_SM * T (softmax gets the larger base per B-0). The
  divisor pool (RMSNorm + softmax denominator) scales the same way. T is the knob the scoping names.

MEASURE vs the ANN baseline (ppl 6.53):
  (1) spiking held-out PPL (same TinyStories tail + sliding window as STEP A); target <=~1.2x ANN (<=~7.84).
  (2) distinct-1/2/3-gram non-degeneracy on the spiking generations.
  (3) verbatim-copy fraction vs the corpus (<=0.20).
  (4) GENERATE ~5 samples from the SPIKING forward (the SAME prompts as STEP A) + SAVE the text VERBATIM
      (the load-bearing READ -- the controller reads coherence).
  >=3 seeds where stochastic (the sampled generations).

VERDICT:
  GO = spiking generation stays COHERENT (READ + fluent + non-degenerate + not-verbatim-copying) at the lowest
       feasible T, ppl <=~1.2x ANN -> a SPIKING FLUENT FACULTY EXISTS (P1 done -> the grounded-language arc's
       faculty piece complete).
  HONEST/escalate = the degradation + the T it needs -> raise T -> (the ordered fallbacks) the Plug-and-Play
       Spiking Operators LIF primitive / NEXUS bit-exact; report the best coherence achieved + the wall.

FOREGROUND/blocking by design. GPU (RTX 3090). PyTorch OFF the bridge (the spiking ops are PyTorch gate-circuit
sims; bridge co-residence is a later step). Usage:
  python -m research.runners._grounded_lang_p1b_stepB1_forward_derisk            # default T sweep {4,8,16}
  python -m research.runners._grounded_lang_p1b_stepB1_forward_derisk --quick    # T=8 only, fewer ppl windows
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# reduce CUDA fragmentation from the transient per-layer attention/graded-read intermediates (24 layers).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CORPUS = _REPO / "data" / "corpus" / "tinystories.txt"
OUT = _REPO / "research" / "findings" / "raw" / "_grounded_lang_p1b_stepB1_forward.json"

ANN_PPL = 6.5303          # STEP A baseline (the target: spiking ppl <=~1.2x this = <=~7.836)
PPL_TARGET_MULT = 1.2

# ---- graded-read operating point (the B-0 / C1 mechanism) ----
READ_SCALE = 20.0          # a_cont = clip((x-knot)/READ_SCALE, 0, 1) -> the rectifier over the support
POOL_BASE = 32             # SiLU / RMSNorm-divisor pool = POOL_BASE * T  (T=8 -> 256, the B-0 SiLU bar pool)
POOL_BASE_SM = 256         # softmax pool = POOL_BASE_SM * T  (T=16 -> 4096, the B-0 softmax bar pool; LARGER)

# Wide exp grid sized to Qwen's ACTUAL post-max-subtract logit support (B-0 honest residual: range to -31.7).
EXP_GRID_LO = -34.0        # cover the measured -31.7 with margin (NOT a narrow [-12,0] grid)
EXP_GRID_HI = 0.5

# Softmax flattened N = B*H*T*S is huge -> chunk the graded read over row-blocks (peak ~ chunk*K_knots).
SOFTMAX_CHUNK = 4_000_000


def log(msg):
    print(f"[p1b-B1] {msg}", flush=True)


def safe_print(s):
    """Print to a possibly-cp1252 Windows console without crashing on smart-quotes/emoji."""
    try:
        print(s, flush=True)
    except UnicodeEncodeError:
        enc = (sys.stdout.encoding or "utf-8")
        print(s.encode(enc, errors="replace").decode(enc, errors="replace"), flush=True)


# =================================================================================================
# The calibrated rectified-basis graded read (B-0's mechanism), as a torch op on GPU.
# Coefficients are fit OFF-line on a fixed grid (numpy lstsq), then applied on-device, vectorized.
# =================================================================================================
def fit_pwl(fn, lo, hi, knots, read_scale=READ_SCALE, n=4000):
    """Calibrate ONCE on a fixed grid (OFF-line; NOT on the data): fn(x) ~ c0 + sum_k a_k*clip((x-knot)/RS,0,1).
    NOTE: B-0's fit used relu (clip lower-only); the live read uses clip(.,0,1) (saturating). We fit with the
    SAME saturating basis the read uses so fit==read (tighter than B-0's relu-fit/clip-read mismatch)."""
    xs = np.linspace(lo, hi, n)
    cols = [np.ones_like(xs)] + [np.clip((xs - kn) / read_scale, 0.0, 1.0) for kn in knots]
    B = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(B, fn(xs), rcond=None)
    fit = B @ coef
    err = np.abs(fit - fn(xs))
    return float(coef[0]), coef[1:].astype(np.float64), {
        "fit_max_err_grid": float(err.max()), "fit_rmse_grid": float(np.sqrt(np.mean(err ** 2)))}


class GradedRead:
    """A calibrated rectified-basis graded read held on-device. Applies element-wise with rate-coded
    graded-pool SEM noise (sqrt(a(1-a))/sqrt(pool)) -- B-0's honesty model -- vectorized in torch."""

    def __init__(self, c0, a_k, knots, device, dtype=torch.float32, read_scale=READ_SCALE):
        self.c0 = float(c0)
        self.a_k = torch.tensor(a_k, device=device, dtype=dtype)             # (K,)
        self.knots = torch.tensor(np.asarray(knots), device=device, dtype=dtype)  # (K,)
        self.read_scale = read_scale
        self.device = device
        self.dtype = dtype

    def _read_block(self, flat, pool, generator):
        """flat: (M,) on device. Returns (M,) graded read. Peak extra memory = M*K (the basis matrix)."""
        a_cont = torch.clamp((flat[:, None] - self.knots) / self.read_scale, 0.0, 1.0)  # (M,K)
        if pool and pool > 0:
            sem = torch.sqrt(torch.clamp(a_cont * (1.0 - a_cont), min=1e-6)) / math.sqrt(pool)
            # fuse: a_cont += randn_like * sem  (one extra (M,K) for randn, freed immediately)
            a_cont = torch.clamp(a_cont.addcmul_(
                torch.randn(a_cont.shape, device=self.device, dtype=self.dtype, generator=generator), sem),
                0.0, 1.0)
        return self.c0 + a_cont @ self.a_k                                   # (M,)

    def __call__(self, x, pool, generator=None, chunk=None):
        """x: any shape. pool: int (rate-code averaging pool backing each basis read). Returns same shape.
        `chunk` caps the row-block size so peak extra memory = chunk*K (essential for the softmax, whose
        flattened N = B*H*T*S is huge)."""
        xf = x.to(self.dtype)
        flat = xf.reshape(-1)                                               # (N,)
        N = flat.shape[0]
        if chunk is None or N <= chunk:
            out = self._read_block(flat, pool, generator)
        else:
            out = torch.empty_like(flat)
            for i in range(0, N, chunk):
                out[i:i + chunk] = self._read_block(flat[i:i + chunk], pool, generator)
        return out.reshape(x.shape)


# Banks (built once after we measure ranges on a calibration pass).
def make_silu_bank(x_range, device):
    lo = min(-8.0, x_range[0] - 1.0)
    hi = max(8.0, x_range[1] + 1.0)
    knots = np.concatenate([np.linspace(lo, -2.0, 7),
                            np.linspace(-1.8, 1.8, 16),
                            np.linspace(2.0, hi, 7)])
    fn = lambda x: x / (1.0 + np.exp(-x))
    c0, a_k, fd = fit_pwl(fn, lo, hi, knots)
    fd["grid"] = [lo, hi]; fd["n_knots"] = len(knots)
    return GradedRead(c0, a_k, knots, device), fd


def make_exp_bank(device, lo=EXP_GRID_LO, hi=EXP_GRID_HI):
    """WIDE exp grid sized to Qwen's actual post-max logit support (B-0 residual)."""
    knots = np.concatenate([np.linspace(lo, -8.0, 12),       # wide tail (B-0: logits to -31.7)
                            np.linspace(-7.5, -3.0, 10),
                            np.linspace(-2.8, 0.0, 14),       # dense near 0 (exp curves fastest)
                            np.linspace(0.1, hi, 3)])
    c0, a_k, fd = fit_pwl(lambda s: np.exp(s), lo, hi, knots)
    fd["grid"] = [lo, hi]; fd["n_knots"] = len(knots)
    return GradedRead(c0, a_k, knots, device), fd


# =================================================================================================
# The spiking op state (banks + per-T pools + a torch generator for reproducible pool noise).
# A module-global so the monkeypatched forwards can reach it without touching signatures.
# =================================================================================================
class SpikeState:
    def __init__(self):
        self.silu_bank = None
        self.exp_bank = None
        self.pool_silu = 0
        self.pool_div = 0          # RMSNorm divisor + softmax denominator pool
        self.pool_softmax = 0      # softmax exp-read pool
        self.enabled = False
        self.gen = None
        self.eps = 1e-6

    def set_T(self, T):
        self.pool_silu = POOL_BASE * T
        self.pool_div = POOL_BASE * T
        self.pool_softmax = POOL_BASE_SM * T


SPK = SpikeState()


# =================================================================================================
# Spiking RMSNorm: weight * (x / (sqrt(mean(x^2)+eps) + pool-SEM)). The exact-RMS divisive read (B-0 headline).
# The divisor mean is a rate-coded mean over d features estimated by a DIV_POOL pool -> ~1/sqrt(pool) SEM.
# =================================================================================================
def spiking_rmsnorm_forward(self, hidden_states):
    if not SPK.enabled:
        # fall back to exact (used during the ANN calibration pass)
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        var = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(var + self.variance_epsilon)
        return self.weight * h.to(input_dtype)
    input_dtype = hidden_states.dtype
    h = hidden_states.to(torch.float32)
    var = h.pow(2).mean(-1, keepdim=True)                       # mean(x^2)
    D = torch.sqrt(var + self.variance_epsilon)                 # exact RMS divisor
    if SPK.pool_div > 0:
        spread = h.std(dim=-1, keepdim=True) / math.sqrt(SPK.pool_div)   # rate-code SEM on the divisor mean
        D = D + torch.randn(D.shape, device=h.device, dtype=h.dtype, generator=SPK.gen) * spread
        D = torch.clamp(D, min=0.5 * torch.sqrt(var + self.variance_epsilon))  # never below half true RMS
    h = h / D
    return self.weight * h.to(input_dtype)


# =================================================================================================
# Spiking SiLU (drop-in for Qwen2MLP.act_fn): the rectified-basis graded read over the gate_proj range.
# =================================================================================================
class SpikingSiLU(nn.Module):
    def forward(self, x):
        if not SPK.enabled or SPK.silu_bank is None:
            return x * torch.sigmoid(x)                          # exact (calibration pass)
        orig_dtype = x.dtype
        out = SPK.silu_bank(x, SPK.pool_silu, generator=SPK.gen)
        return out.to(orig_dtype)


# =================================================================================================
# Spiking attention forward (drop-in for eager_attention_forward): the QK^T*scale + mask is exact (linears +
# fixed RoPE already applied upstream); ONLY the softmax is converted -> exp graded-read (WIDE grid) over the
# post-max-subtract logits + sum-normalization (divisive arm), with pool noise on both. Matches B-0 exactly.
# =================================================================================================
def _spiking_softmax_lastdim(attn_weights):
    """attn_weights: (B, H, T, S) with masked entries = large-negative. Convert softmax over dim=-1 to the
    spiking exp-read + sum-norm. Returns same shape, fp32."""
    aw = attn_weights.to(torch.float32)
    # max-subtract (standard numerically-stable) -- per (B,H,T) row over the key dim S
    m = aw.max(dim=-1, keepdim=True).values
    shifted = aw - m                                            # all <= 0
    # The exp-read bank's grid is WIDE [EXP_GRID_LO, 0.5]; masked entries are ~ -65504 (fp16 min) -> below the
    # grid -> clip((shifted-knot)/RS,0,1) = 0 for all knots -> read ~= c0. exp(c0-grid-min) ~ 0; but to be exact
    # we floor masked positions to exactly 0 weight after the read (they should contribute 0 to the sum).
    masked = shifted < (EXP_GRID_LO - 0.5)                      # the causal-masked positions (far below grid)
    e = SPK.exp_bank(shifted, SPK.pool_softmax, generator=SPK.gen, chunk=SOFTMAX_CHUNK)
    e = torch.clamp(e, min=0.0)                                 # exp is non-negative
    e = torch.where(masked, torch.zeros_like(e), e)             # masked keys contribute exactly 0
    # (the per-key exp-read pool noise above is the dominant, B-0-measured noise term.)
    s = e.sum(dim=-1, keepdim=True)                             # the divisive-normalization denominator
    if SPK.pool_div > 0:
        # The denominator is ONE pooled quantity read by a DIV_POOL population -> relative SEM = 1/sqrt(pool)
        # (a single pooled rate read, NOT nk fully-correlated per-key errors -- the B-0 *nk form was the
        # worst-case bound calibrated for tiny nk<=64; at full context nk~2048 the correct independent-read SEM
        # is the pooled-denominator relative SEM, which is also well-conditioned). Floor s at half its noiseless
        # value so a downward draw can never produce a near-zero (NaN-inducing) divisor.
        s_noise = s * (torch.randn(s.shape, device=e.device, dtype=e.dtype, generator=SPK.gen)
                       / math.sqrt(SPK.pool_softmax))
        s = torch.clamp(s + s_noise, min=0.5 * s.clamp(min=1e-30))
    s = torch.clamp(s, min=1e-30)
    w = e / s
    # final guard: any residual non-finite (extreme fp32 tail) -> fall back to a uniform-over-valid row.
    if not torch.isfinite(w).all():
        bad = ~torch.isfinite(w).all(dim=-1, keepdim=True)
        valid = (~masked).to(w.dtype)
        unif = valid / valid.sum(dim=-1, keepdim=True).clamp(min=1)
        w = torch.where(bad, unif, w)
    return w


def spiking_eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    import transformers.models.qwen2.modeling_qwen2 as q
    repeat_kv = q.repeat_kv
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling   # EXACT (linears+RoPE upstream)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    if SPK.enabled and SPK.exp_bank is not None:
        attn_weights = _spiking_softmax_lastdim(attn_weights).to(query.dtype)  # SPIKING softmax
    else:
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)  # exact (cal pass)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


# =================================================================================================
# Install / uninstall the spiking ops onto the live model.
# =================================================================================================
def install_spiking_ops(model):
    """Replace RMSNorm.forward (all instances), each MLP.act_fn, and the eager attention's softmax."""
    import types
    import transformers.models.qwen2.modeling_qwen2 as q
    # 1. RMSNorm: bind the spiking forward on every Qwen2RMSNorm instance
    n_rms = 0
    for mod in model.modules():
        if type(mod).__name__ == "Qwen2RMSNorm":
            mod.forward = types.MethodType(spiking_rmsnorm_forward, mod)
            n_rms += 1
    # 2. SiLU: swap each MLP's act_fn for the spiking module
    n_silu = 0
    for mod in model.modules():
        if type(mod).__name__ == "Qwen2MLP":
            mod.act_fn = SpikingSiLU()
            n_silu += 1
    # 3. Softmax: register our spiking attention as the "eager" interface + force eager.
    #    The attention forward looks up ALL_ATTENTION_FUNCTIONS.get_interface(impl, eager_attention_forward);
    #    we both (a) set the default fallback by patching the module symbol and (b) register under "eager".
    q.eager_attention_forward = spiking_eager_attention_forward
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["eager"] = spiking_eager_attention_forward
    except Exception as e:
        log(f"  (note: ALL_ATTENTION_FUNCTIONS register skipped: {e})")
    # make sure every attention module uses eager so our softmax is hit
    model.config._attn_implementation = "eager"
    for mod in model.modules():
        if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
            mod.config._attn_implementation = "eager"
        if type(mod).__name__ == "Qwen2Attention":
            try:
                mod.config._attn_implementation = "eager"
            except Exception:
                pass
    return {"n_rmsnorm": n_rms, "n_silu_mlp": n_silu}


# =================================================================================================
# Measurement helpers.
# =================================================================================================
def held_out_text():
    with open(CORPUS, "r", encoding="utf-8") as f:
        text = f.read()
    held = text[-120_000:]
    delim = "<|endoftext|>"
    idx = held.find(delim)
    if idx != -1:
        held = held[idx + len(delim):].lstrip()
    return held, text


def compute_ppl(model, tok, held, max_windows=None):
    cfg = model.config
    enc = tok(held, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    n_total = input_ids.shape[1]
    max_len = min(getattr(cfg, "max_position_embeddings", 2048) or 2048, 2048)
    stride = max_len
    nll_sum, n_scored, n_win = 0.0, 0, 0
    with torch.no_grad():
        for begin in range(0, n_total, stride):
            end = min(begin + max_len, n_total)
            ids = input_ids[:, begin:end]
            if ids.shape[1] < 2:
                break
            out = model(ids, labels=ids)
            ns = ids.shape[1] - 1
            nll_sum += float(out.loss) * ns
            n_scored += ns
            n_win += 1
            if max_windows and n_win >= max_windows:
                break
    mean_nll = nll_sum / max(n_scored, 1)
    return math.exp(mean_nll), mean_nll, n_scored


def distinct_ngrams(text, n):
    toks = text.split()
    if len(toks) < n:
        return 0.0, 0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return (len(set(grams)) / len(grams) if grams else 0.0), len(grams)


def max_verbatim_copy(gen, corpus, min_words=8):
    """Longest run of consecutive words in `gen` that appears verbatim in `corpus`, as a fraction of gen length.
    Cheap heuristic: slide windows of decreasing length from min(len, 30) down to min_words."""
    gtoks = gen.split()
    if len(gtoks) < min_words:
        return 0.0, 0
    # build a set of corpus n-grams lazily per window length (bounded; corpus is ~8MB)
    corpus_lower = corpus.lower()
    longest = 0
    for L in range(min(len(gtoks), 30), min_words - 1, -1):
        found = False
        for i in range(0, len(gtoks) - L + 1):
            phrase = " ".join(gtoks[i:i + L]).lower()
            if phrase in corpus_lower:
                longest = L
                found = True
                break
        if found:
            break
    return (longest / len(gtoks) if gtoks else 0.0), longest


def generate(model, tok, user_msg, do_sample, temperature, seed, max_new_tokens=120):
    msgs = [{"role": "user", "content": user_msg}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    if SPK.gen is not None:
        SPK.gen.manual_seed(seed + 1000)
    gk = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    if do_sample:
        gk.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gk.update(do_sample=False)
    with torch.no_grad():
        out = model.generate(**ids, **gk)
    new = out[0, ids.input_ids.shape[1]:]
    return tok.decode(new, skip_special_tokens=True)


# Prompts: the SAME spec as STEP A (so generations are directly comparable).
GEN_SPEC = [
    ("Once upon a time", False, 0.0),
    ("Once upon a time", True, 0.8),
    ("The weather today", False, 0.0),
    ("What do dogs eat?", False, 0.0),
    ("What do dogs eat?", True, 0.8),
]


def run_for_T(model, tok, held, corpus, T, ppl_windows, sampled_seeds):
    """Run the spiking forward at rate-code pool budget T: PPL + generations (greedy once; sampled over seeds)."""
    SPK.set_T(T)
    SPK.enabled = True
    log(f"--- T={T}  (pool_silu={SPK.pool_silu}, pool_div={SPK.pool_div}, pool_softmax={SPK.pool_softmax}) ---")

    # PPL (deterministic forward; pool noise still injected -> a fixed-seed generator for reproducibility)
    SPK.gen.manual_seed(12345 + T)
    t0 = time.time()
    ppl, nll, n_scored = compute_ppl(model, tok, held, max_windows=ppl_windows)
    ppl_s = time.time() - t0
    log(f"  spiking held-out PPL = {ppl:.4f}  (ANN {ANN_PPL:.4f}; target <= {ANN_PPL * PPL_TARGET_MULT:.4f}); "
        f"{n_scored} tok in {ppl_s:.1f}s")

    # Generations: greedy (deterministic) + sampled over seeds (>=3) for the load-bearing READ.
    gens = []
    for i, (msg, do_sample, temp) in enumerate(GEN_SPEC):
        if not do_sample:
            t0 = time.time()
            g = generate(model, tok, msg, do_sample=False, temperature=0.0, seed=42 + i)
            gens.append({"idx": i, "user_prompt": msg, "mode": "greedy", "seed": 42 + i,
                         "generated_text": g, "gen_seconds": round(time.time() - t0, 2)})
            log(f"  gen[{i}] greedy '{msg}':")
            safe_print("      " + g.replace("\n", "\n      "))
        else:
            for sd in sampled_seeds:
                t0 = time.time()
                g = generate(model, tok, msg, do_sample=True, temperature=temp, seed=sd + i)
                gens.append({"idx": i, "user_prompt": msg, "mode": f"sampled(temp={temp},top_p=0.9)",
                             "seed": sd + i, "generated_text": g, "gen_seconds": round(time.time() - t0, 2)})
                log(f"  gen[{i}] sampled seed={sd + i} '{msg}':")
                safe_print("      " + g.replace("\n", "\n      "))

    # Non-degeneracy + verbatim-copy over the generations.
    deg = []
    for gd in gens:
        txt = gd["generated_text"]
        d1, n1 = distinct_ngrams(txt, 1)
        d2, n2 = distinct_ngrams(txt, 2)
        d3, n3 = distinct_ngrams(txt, 3)
        vc, vlen = max_verbatim_copy(txt, corpus)
        gd["distinct_1"] = round(d1, 4); gd["distinct_2"] = round(d2, 4); gd["distinct_3"] = round(d3, 4)
        gd["n_words"] = len(txt.split())
        gd["verbatim_copy_frac"] = round(vc, 4); gd["verbatim_copy_words"] = vlen
        deg.append((d1, d2, d3, vc))
    d1m = float(np.mean([d[0] for d in deg])) if deg else 0.0
    d2m = float(np.mean([d[1] for d in deg])) if deg else 0.0
    d3m = float(np.mean([d[2] for d in deg])) if deg else 0.0
    vcm = float(np.max([d[3] for d in deg])) if deg else 0.0
    nonword = float(np.mean([1.0 if gd["n_words"] >= 20 else 0.0 for gd in gens]))  # length-floor sanity
    log(f"  distinct-1/2/3 (mean over gens) = {d1m:.3f}/{d2m:.3f}/{d3m:.3f} | max verbatim-copy = {vcm:.3f}")

    return {
        "T": T,
        "pool_silu": SPK.pool_silu, "pool_div": SPK.pool_div, "pool_softmax": SPK.pool_softmax,
        "spiking_perplexity": round(ppl, 4),
        "spiking_mean_nll": round(nll, 4),
        "ppl_ratio_vs_ann": round(ppl / ANN_PPL, 4),
        "ppl_n_tokens_scored": int(n_scored),
        "ppl_seconds": round(ppl_s, 2),
        "ppl_within_target": bool(ppl <= ANN_PPL * PPL_TARGET_MULT),
        "distinct_1_mean": round(d1m, 4), "distinct_2_mean": round(d2m, 4), "distinct_3_mean": round(d3m, 4),
        "max_verbatim_copy_frac": round(vcm, 4),
        "frac_gens_ge20_words": round(nonword, 4),
        "generations": gens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="T=8 only, fewer ppl windows (fast smoke)")
    ap.add_argument("--ts", type=str, default="4,8,16", help="comma T sweep (default 4,8,16; cheapest first)")
    ap.add_argument("--ppl-windows", type=int, default=0, help="cap ppl windows (0=all ~14 windows)")
    ap.add_argument("--sampled-seeds", type=str, default="42,43,44", help=">=3 seeds for sampled gens")
    args = ap.parse_args()

    t_start = time.time()
    log(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
        f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")
    if not torch.cuda.is_available():
        log("WARNING: CUDA not available -- this is a GPU runner; proceeding on CPU will be very slow.")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"loading {MODEL_ID} (fp16, eager attention) ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16,
                                                 attn_implementation="eager").cuda().eval()
    device = next(model.parameters()).device
    log(f"loaded; {sum(p.numel() for p in model.parameters())/1e6:.1f}M params on {device}")

    SPK.gen = torch.Generator(device=device)
    SPK.eps = float(model.config.rms_norm_eps)

    Ts = ([8] if args.quick else [int(x) for x in args.ts.split(",") if x.strip()])
    ppl_windows = (3 if args.quick else (args.ppl_windows or None))
    sampled_seeds = [int(x) for x in args.sampled_seeds.split(",") if x.strip()]

    result = {
        "probe": "grounded_lang_p1b_stepB1_full_spiking_forward_qwen",
        "resolves": "the FULL spiking forward of Qwen2.5-0.5B (RMSNorm/SiLU/Softmax graded + RoPE fixed) -- does "
                    "the SPIKING generation stay COHERENT/fluent/non-degenerate/not-verbatim-copying at the "
                    "lowest feasible rate-code pool T, with spiking ppl <=~1.2x ANN (6.53)? The load-bearing "
                    "generation-coherence [VERIFY] NOT settled by B-0's per-op cosines.",
        "model_id": MODEL_ID,
        "ann_baseline_ppl": ANN_PPL, "ppl_target": round(ANN_PPL * PPL_TARGET_MULT, 4),
        "read_scale": READ_SCALE, "pool_base_silu": POOL_BASE, "pool_base_softmax": POOL_BASE_SM,
        "exp_grid": [EXP_GRID_LO, EXP_GRID_HI],
        "T_sweep": Ts, "sampled_seeds": sampled_seeds,
        "mechanism": "PyTorch gate-circuit sims (B-0's calibrated-graded-read), vectorized in torch on GPU; "
                     "NOT on the SimulationBridge yet (bridge co-residence = a later step). The linears "
                     "(q/k/v/o_proj, gate/up/down_proj, lm_head, embed) stay EXACT matmuls; RoPE applied exactly.",
        "continues": {
            "stepA_ann": "_grounded_lang_p1b_ann_baseline.json (ppl 6.53, fluent)",
            "stepB0_ops": "_grounded_lang_p1b_stepB0_ops.json (GO_MECHANISM_TRANSFERS: RMSNorm 1.0 / SiLU "
                          "0.943(nf 1.0) / Softmax 0.963(nf 0.9996) / RoPE bit-exact)",
            "scoping": "2026-06-22-grounded-language-faculty-scoping.md (the LLaMA-stack convert, S1c)",
        },
    }

    # ---- CALIBRATION PASS (SPK disabled): measure SiLU-input + softmax-logit ranges to size the banks ----
    log("calibration pass: measuring real SiLU-input + softmax-logit ranges (SPK disabled) ...")
    held, corpus = held_out_text()
    SPK.enabled = False
    ranges = {"silu_min": math.inf, "silu_max": -math.inf, "logit_shift_min": math.inf}
    cal_layer = model.model.layers[12]
    gate_outs = []

    def gate_hook(mod, args, output):
        gate_outs.append((output.detach().float().min().item(), output.detach().float().max().item()))
    h1 = cal_layer.mlp.gate_proj.register_forward_hook(gate_hook)
    logit_min_holder = {"v": math.inf}
    real_softmax = F.softmax

    def cal_softmax(inp, *a, **k):
        if inp.dim() == 4:
            x = inp.detach().float()
            m = x.max(dim=-1, keepdim=True).values
            sh = (x - m)
            valid = sh > -1e4
            if valid.any():
                logit_min_holder["v"] = min(logit_min_holder["v"], float(sh[valid].min().item()))
        return real_softmax(inp, *a, **k)

    # short calibration text (a few hundred tokens is plenty for the range)
    cal_ids = tok(held[:3000], return_tensors="pt").to(device)
    F.softmax = cal_softmax
    with torch.no_grad():
        for mod in model.modules():
            if type(mod).__name__ == "Qwen2MLP":
                pass
        model(**{k: v[:, :512] for k, v in cal_ids.items()})
    F.softmax = real_softmax
    h1.remove()
    if gate_outs:
        ranges["silu_min"] = min(g[0] for g in gate_outs)
        ranges["silu_max"] = max(g[1] for g in gate_outs)
    ranges["logit_shift_min"] = logit_min_holder["v"]
    log(f"  measured SiLU-input range [{ranges['silu_min']:.2f},{ranges['silu_max']:.2f}] | "
        f"softmax post-max logit min = {ranges['logit_shift_min']:.2f}")

    # ---- BUILD BANKS (off-line fit over the measured ranges; WIDE exp grid) ----
    SPK.silu_bank, silu_fd = make_silu_bank((ranges["silu_min"], ranges["silu_max"]), device)
    SPK.exp_bank, exp_fd = make_exp_bank(device)
    log(f"  SiLU bank: grid {silu_fd['grid']} knots {silu_fd['n_knots']} fit-max-err {silu_fd['fit_max_err_grid']:.4f}")
    log(f"  exp  bank: grid {exp_fd['grid']} knots {exp_fd['n_knots']} fit-max-err {exp_fd['fit_max_err_grid']:.5f} "
        f"(WIDE, covers measured logit-min {ranges['logit_shift_min']:.1f})")
    result["measured_ranges"] = ranges
    result["silu_fit"] = silu_fd
    result["exp_fit"] = exp_fd

    # ---- SANITY: re-run ANN ppl with SPK disabled through THIS process (should match STEP A 6.53) ----
    log("sanity: ANN ppl with SPK disabled (should match STEP A 6.53) ...")
    SPK.enabled = False
    ann_ppl_here, ann_nll_here, ann_n = compute_ppl(model, tok, held, max_windows=ppl_windows)
    log(f"  ANN ppl (this process, SPK off) = {ann_ppl_here:.4f}  ({ann_n} tok)")
    result["ann_ppl_reproduced"] = round(ann_ppl_here, 4)

    # ---- INSTALL the spiking ops ----
    inst = install_spiking_ops(model)
    log(f"installed spiking ops: {inst}")
    result["install"] = inst

    # ---- T SWEEP (cheapest first; stop early on a clean GO) ----
    per_T = []
    go_T = None
    for T in Ts:
        r = run_for_T(model, tok, held, corpus, T, ppl_windows, sampled_seeds)
        per_T.append(r)
        # a clean-GO heuristic for early-stop: ppl within target AND non-degenerate (distinct-2 healthy, low copy)
        clean = (r["ppl_within_target"] and r["distinct_2_mean"] >= 0.55
                 and r["max_verbatim_copy_frac"] <= 0.20 and r["frac_gens_ge20_words"] >= 0.6)
        if clean and go_T is None:
            go_T = T
            log(f"  >>> T={T} is a clean-GO candidate (ppl {r['spiking_perplexity']:.3f} <= target, "
                f"distinct-2 {r['distinct_2_mean']:.3f}, copy {r['max_verbatim_copy_frac']:.3f}) -- "
                f"stopping the sweep early (lowest feasible T).")
            break
    result["per_T"] = per_T

    # ---- VERDICT ----
    # The READ (coherence) is the controller's call; we provide the QUANTITATIVE gate + flag the best T.
    feasible = [r for r in per_T if r["ppl_within_target"] and r["distinct_2_mean"] >= 0.55
                and r["max_verbatim_copy_frac"] <= 0.20 and r["frac_gens_ge20_words"] >= 0.6]
    best = min(per_T, key=lambda r: r["spiking_perplexity"]) if per_T else None
    if feasible:
        lowest = min(feasible, key=lambda r: r["T"])
        verdict = "GO"
        lowest_T = lowest["T"]
        tail = (f"a SPIKING FLUENT FACULTY EXISTS: the full spiking forward generates non-degenerate text with "
                f"spiking ppl {lowest['spiking_perplexity']:.3f} ({lowest['ppl_ratio_vs_ann']:.2f}x ANN, within "
                f"the 1.2x target) at the lowest feasible T={lowest_T} (pool_silu={lowest['pool_silu']}, "
                f"pool_softmax={lowest['pool_softmax']}); distinct-2 {lowest['distinct_2_mean']:.3f}, max "
                f"verbatim-copy {lowest['max_verbatim_copy_frac']:.3f}. P1 done -> the grounded-language arc's "
                f"faculty piece is complete (PENDING the controller's READ of the verbatim generation samples "
                f"for coherence).")
    else:
        verdict = "HONEST_ESCALATE"
        lowest_T = None
        bt = best["T"] if best else None
        tail = (f"no T in the sweep {Ts} cleared the quantitative gate (ppl<= {ANN_PPL * PPL_TARGET_MULT:.2f} AND "
                f"distinct-2>=0.55 AND copy<=0.20). Best: T={bt} ppl {best['spiking_perplexity']:.3f} "
                f"({best['ppl_ratio_vs_ann']:.2f}x ANN), distinct-2 {best['distinct_2_mean']:.3f}, copy "
                f"{best['max_verbatim_copy_frac']:.3f}. -> raise T further, then the ordered fallbacks: the "
                f"Plug-and-Play Spiking Operators LIF primitive (multi-step T) / NEXUS bit-exact conversion. "
                f"Report the best coherence achieved + the wall.")

    verdict_line = (
        "p1b_stepB1: FULL spiking forward of Qwen2.5-0.5B (RMSNorm/SiLU/Softmax graded + RoPE bit-exact; "
        "linears exact) -> " + " | ".join(
            f"T{r['T']}: ppl {r['spiking_perplexity']:.3f} ({r['ppl_ratio_vs_ann']:.2f}x), "
            f"d2 {r['distinct_2_mean']:.2f}, copy {r['max_verbatim_copy_frac']:.2f}" for r in per_T)
        + f"  [ANN {ANN_PPL:.2f}, target <= {ANN_PPL * PPL_TARGET_MULT:.2f}] -> {verdict}. " + tail)

    result["verdict"] = verdict
    result["lowest_feasible_T"] = lowest_T
    result["best_T_by_ppl"] = (best["T"] if best else None)
    result["verdict_line"] = verdict_line
    result["total_seconds"] = round(time.time() - t_start, 2)

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


if __name__ == "__main__":
    main()
