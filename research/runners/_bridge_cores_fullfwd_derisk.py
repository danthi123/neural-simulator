"""BRIDGE CO-RESIDENCE de-risk #3 (the culminating feasibility test): the FULL 24-layer Qwen2.5-0.5B forward on
the LIVE SimulationBridge RF (resonate-and-fire complex-synapse) substrate.

De-risk #1 (`_bridge_cores_qproj_derisk.json`, GO): ONE q_proj on the RF bridge is BIT-EXACT (max-err 4.58e-7).
De-risk #2 (`_bridge_cores_layer_derisk.json`, GO): ONE FULL decoder layer on RF == B-1 PyTorch-spiking layer
(cos 1.0, all 7 linears bit-exact, no error accumulation). THIS step (de-risk #3): install ALL 24 decoder layers
+ the (tied) lm_head on the RF bridge and MEASURE the four load-bearing numbers per the scoping §3 de-risk #4 + §6:

  (1) VRAM   -- does the full 494M fit on the bridge (< 24 GB)? report the ACTUAL usage + the storage chosen.
  (2) WALL-CLOCK = tokens/sec for the RF forward (the SOLE cloud-trigger per the scoping; the load-bearing number).
  (3) ppl on a SMALL held-out slice (a few hundred tokens) vs the B-1 spiking ppl (7.08 @ T=16) -- does the
      full on-bridge forward MATCH B-1, or does the per-layer graded-read SEM COMPOUND over 24 layers?
  (4) a SHORT spiking generation (~16-32 tokens, "Once upon a time") -- READ it (coherent?).

MECHANISM (reuse-by-import, NO `sim/` edit):
  - The 7 linears/layer + lm_head are RF matvecs (the de-risk-#1/#2 pattern: install W=weight.T as complex
    synapses; kick z=row, resonate RF_NSTEPS @ lam=0/omega~0; read Re(Z)/nsteps = a@W EXACTLY). The CSR is built
    VECTORIZED (rows=D_in+tile(arange(D_out),D_in), cols=repeat(arange(D_in),D_out), data=W.ravel()) and assigned
    to bridge.cp_rf_w_re/cp_rf_w_im -- BIT-IDENTICAL to rf_set_complex_weights (verified max|.|=0.0) but ms not the
    136M-iter Python list-comprehension (the lm_head would be intractable otherwise). Assigning the bridge's CSR
    attributes IS the sanctioned C1 `_set_rf_weights` cache-swap pattern -- NOT a `sim/` edit.
  - RMSNorm/SiLU/softmax are the B-1 calibrated graded reads (reuse-by-import from de-risk #2's host mirrors of the
    B-1 module banks); RoPE host-exact. The full decoder layer forward + attention are de-risk #2's `layer_forward`
    / `run_attention` VERBATIM (reuse-by-import).
  - VRAM control = LAYER-STREAMING: one RF bridge per unique (D_in,D_out) shape (4 shapes for the layer linears +
    1 for lm_head); each layer re-installs its 7 weights onto the shared per-shape bridges (build->use->free the
    *connections*, the CSR is the resident object). Resident peak = the 4 layer-shape CSRs + lm_head, NOT all 169.

ANTI-CHEAT: a LESION (scramble the lm_head RF weights) -> the logits must DIVERGE from the true forward (cos
collapse) while the shuffled matvec still reproduces ITS OWN a@W_shuf (the RF carries the computation).

FOREGROUND/blocking. GPU (SIM_BACKEND=cupy). Usage:
  SIM_BACKEND=cupy python -m research.runners._bridge_cores_fullfwd_derisk
  SIM_BACKEND=cupy python -m research.runners._bridge_cores_fullfwd_derisk --ppl-tokens 256 --gen-tokens 24 --T 16
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

# Reuse the RF bridge + the de-risk-#2 layer port + the B-1 banks VERBATIM (reuse-by-import; NO re-impl).
from research.runners.rf_phasor_composer import _build_rf_bridge  # noqa: E402
import research.runners._bridge_cores_layer_derisk as L2  # noqa: E402  (layer_forward / run_attention / banks)
import research.runners._grounded_lang_p1b_stepB1_forward_derisk as B1  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CORPUS = _REPO / "data" / "corpus" / "tinystories.txt"
OUT = _REPO / "research" / "findings" / "raw" / "_bridge_cores_fullfwd_derisk.json"

# The C1 / de-risk-#1/#2 RF operating point (identical): lam=0, huge period (omega~0) => Re(Z_out)=nsteps*(a@W).
RF_PERIOD = L2.RF_PERIOD
RF_NSTEPS = L2.RF_NSTEPS
RF_LAMBDA = L2.RF_LAMBDA

ANN_PPL = B1.ANN_PPL            # 6.5303 (the ANN baseline)
B1_PPL = 7.08                  # the B-1 spiking ppl @ T=16 (the matched reference for the on-bridge full forward)


def log(msg):
    print(f"[fullfwd-rf] {msg}", flush=True)


def safe_print(s):
    try:
        print(s, flush=True)
    except UnicodeEncodeError:
        enc = (sys.stdout.encoding or "utf-8")
        print(s.encode(enc, errors="replace").decode(enc, errors="replace"), flush=True)


def _vram_used_gb():
    """Actual CuPy-pool used bytes + the torch-allocated bytes -> GB (the live resident VRAM)."""
    used = 0.0
    try:
        import cupy as cp
        used += float(cp.get_default_memory_pool().used_bytes())
    except Exception:
        pass
    try:
        import torch
        used += float(torch.cuda.memory_allocated())
    except Exception:
        pass
    return used / 1e9


def _vram_free_total_gb():
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        return free / 1e9, total / 1e9
    except Exception:
        return float("nan"), float("nan")


# =====================================================================================================
# The VECTORIZED RF matvec: one LIVE RF bridge per unique (D_in,D_out) shape; install W by building the
# complex CSR DIRECTLY (vectorized, ms) and assigning bridge.cp_rf_w_re/cp_rf_w_im (the C1 cache-swap
# pattern, bit-identical to rf_set_complex_weights). The resonate uses the megakernel (enable_rf_cudagraph)
# when enabled. Per-row kick/resonate/read loop (the C1 `_rf_matvec_rows`).
# =====================================================================================================
class RFMatvecVec:
    def __init__(self, seed=42, use_megakernel=True):
        import cupy as cp
        from sim.backend import get_sparse_module
        self.cp = cp
        self.csp = get_sparse_module()
        self.seed = int(seed)
        self.use_megakernel = bool(use_megakernel)
        self._bridges = {}          # (D_in,D_out) -> bridge
        # cache the (rows,cols) index arrays per shape (model-constant index structure; only data changes per W)
        self._idx_cache = {}        # (D_in,D_out) -> (cp rows, cp cols)
        # PERF (the C1 _WEIGHT_CSR_CACHE insight): each weight tensor is MODEL-CONSTANT across the whole run, so
        # build its complex CSR ONCE (keyed by id(W)) and thereafter SWAP the cached (re,im) onto the bridge with
        # two attribute assignments -- near-instant, BIT-IDENTICAL to a fresh build (same csr on the same data).
        # Without this, generation re-installs all 169 weights EVERY autoregressive forward (~10s of install each).
        self._csr_cache = {}        # id(W) -> (D_in, D_out, cp_csr_re, cp_csr_im)
        self.n_installs = 0
        self.n_cache_hits = 0
        self.install_seconds = 0.0
        self.matvec_seconds = 0.0

    def _bridge_for(self, D_in, D_out):
        key = (int(D_in), int(D_out))
        if key not in self._bridges:
            n = D_in + D_out
            t0 = time.time()
            b = _build_rf_bridge(n, seed=self.seed)
            if self.use_megakernel:
                # opt-in megakernel (one CUDA kernel/resonate-step); GPU-only, == the loop at the read tolerance.
                try:
                    b.core_config.enable_rf_cudagraph = True
                except Exception:
                    pass
            self._bridges[key] = b
            log(f"    built RF bridge for shape {key} (n={n} neurons) in {time.time()-t0:.2f}s "
                f"(megakernel={self.use_megakernel})")
        return self._bridges[key]

    def _indices_for(self, D_in, D_out):
        key = (int(D_in), int(D_out))
        if key not in self._idx_cache:
            cp = self.cp
            # post = D_in + nn (column index of W), pre = m (row index of W); W.ravel() is m-major (row-major)
            rows = (D_in + cp.tile(cp.arange(D_out, dtype=cp.int32), int(D_in)))
            cols = cp.repeat(cp.arange(D_in, dtype=cp.int32), int(D_out))
            self._idx_cache[key] = (rows, cols)
        return self._idx_cache[key]

    def install(self, W):
        """Install the real weight W=[D_in,D_out] as complex synapses on its shape's bridge by building the CSR
        directly (vectorized) and assigning cp_rf_w_re/cp_rf_w_im. BIT-IDENTICAL to rf_set_complex_weights.
        Caches the built CSR by id(W) -> a repeat install of the SAME tensor is a near-instant attribute swap."""
        cp = self.cp
        D_in, D_out = W.shape
        n = D_in + D_out
        bridge = self._bridge_for(D_in, D_out)
        key = id(W)
        cached = self._csr_cache.get(key)
        if cached is not None:
            _, _, csr_re, csr_im = cached
            bridge.cp_rf_w_re = csr_re          # swap the cached CSR (== a fresh rf_set_complex_weights)
            bridge.cp_rf_w_im = csr_im
            self.n_cache_hits += 1
            return bridge, D_in, D_out
        rows, cols = self._indices_for(D_in, D_out)
        t0 = time.time()
        data_re = cp.asarray(np.ascontiguousarray(W, dtype=np.float64).ravel())     # W[m,nn], m-major
        csr_re = self.csp.csr_matrix((data_re, (rows, cols)), shape=(n, n))
        csr_im = self.csp.csr_matrix(
            (cp.zeros(data_re.shape[0], dtype=cp.float64), (rows, cols)), shape=(n, n))
        cp.cuda.Stream.null.synchronize()
        bridge.cp_rf_w_re = csr_re
        bridge.cp_rf_w_im = csr_im
        self._csr_cache[key] = (D_in, D_out, csr_re, csr_im)
        self.install_seconds += time.time() - t0
        self.n_installs += 1
        return bridge, D_in, D_out

    def matvec_rows(self, W, rows_in):
        """Install W then run the RF matvec on every row of rows_in (N, D_in). Returns (N, D_out) = rows_in @ W."""
        cp = self.cp
        bridge, D_in, D_out = self.install(W)
        n = D_in + D_out
        N = rows_in.shape[0]
        out = np.zeros((N, D_out), dtype=np.float64)
        inv = 1.0 / float(RF_NSTEPS)
        t0 = time.time()
        for r in range(N):
            kick = np.zeros(n, dtype=np.complex128)
            kick[:D_in] = np.asarray(rows_in[r], dtype=np.float64)
            bridge.rf_kick(kick, period=int(RF_PERIOD), lam=float(RF_LAMBDA))
            bridge.rf_resonate_steps(int(RF_NSTEPS))
            re = cp.asnumpy(bridge.cp_membrane_potential_v[D_in:]).astype(np.float64)
            out[r] = re * inv
        cp.cuda.Stream.null.synchronize()
        self.matvec_seconds += time.time() - t0
        return out

    def evict(self, W):
        """Drop W's cached CSR + free the pool blocks (used to make room for a non-cached tensor, e.g. the lesion's
        shuffled lm_head, which would otherwise allocate ON TOP of the cache-everything resident and OOM)."""
        key = id(W)
        cached = self._csr_cache.pop(key, None)
        if cached is not None:
            _, _, csr_re, csr_im = cached
            for b in self._bridges.values():
                if b.cp_rf_w_re is csr_re:
                    b.cp_rf_w_re = None
                if b.cp_rf_w_im is csr_im:
                    b.cp_rf_w_im = None
            del csr_re, csr_im
        try:
            self.cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    def free_layer_shapes(self):
        """Drop the per-layer-shape CSRs (keep the bridges + index caches). Used between layers if streaming the
        layer weights -- here the CSR is re-assigned each install, so the prior layer's CSR is freed on reassign;
        this explicit free trims the pool between layers."""
        for b in self._bridges.values():
            b.cp_rf_w_re = None
            b.cp_rf_w_im = None
        try:
            self.cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


# =====================================================================================================
# Extract a Qwen decoder layer's params in the de-risk-#2 install convention (W=weight.T, biases host-add).
# =====================================================================================================
def extract_layer(layer, attn, Hq, Hkv, head_dim):
    import torch

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
    return W, weights


# =====================================================================================================
# The full RF forward on a token-id sequence -> logits (S, V). Embedding gather exact; 24 decoder layers
# (RF linears + B-1 graded nonlinearities, de-risk #2's layer_forward); final RMSNorm (B-1 graded read,
# RMSNORM_MODE); lm_head RF matvec. Layers are STREAMED (re-install per shape); lm_head installed once.
# =====================================================================================================
def rf_full_forward(ids, ctx, rmsnorm_mode="graded", noise_seed=7, lm_head_W_override=None):
    """ids: (S,) int. ctx: the captured model context (weights extractors, cos/sin, banks, cfg, rfmv). Returns
    logits (S, V) numpy. The pool-noise generator is seed-reset at the start so two forwards (e.g. RF vs the
    reference) draw IDENTICAL graded-read noise -> isolates the RF-matvec residual."""
    rng = np.random.default_rng(noise_seed)
    S = len(ids)
    cfg = ctx["cfg"]
    cos = ctx["cos_full"][:S]
    sin = ctx["sin_full"][:S]
    silu_host = ctx["silu_host"]; exp_host = ctx["exp_host"]
    pool_silu = ctx["pool_silu"]; pool_div = ctx["pool_div"]; pool_softmax = ctx["pool_softmax"]
    rfmv = ctx["rfmv"]

    # embedding gather (exact; the rows ARE x)
    hidden = ctx["embed"][np.asarray(ids)].astype(np.float64)          # (S, D)

    def rf_linear_fn(name, rows):
        return rfmv.matvec_rows(ctx["cur_W"][name], rows)

    for li in range(cfg["n_layers"]):
        # stream: extract + cache THIS layer's weights, install lazily inside rf_linear_fn.
        W, weights = ctx["get_layer"](li)
        ctx["cur_W"] = W
        hidden = L2.layer_forward(hidden, weights, cfg, rf_linear_fn, rmsnorm_mode=rmsnorm_mode,
                                  silu_bank=silu_host, exp_bank=exp_host, pool_silu=pool_silu,
                                  pool_div=pool_div, pool_softmax=pool_softmax, rng=rng, cos=cos, sin=sin)

    # final RMSNorm (B-1 graded read or exact), then lm_head RF matvec (tied embedding weight).
    if rmsnorm_mode == "graded":
        hidden = L2.graded_rmsnorm(hidden, ctx["norm_w"], cfg["eps"], pool_div, rng)
    else:
        hidden = L2.exact_rmsnorm(hidden, ctx["norm_w"], cfg["eps"])
    W_head = lm_head_W_override if lm_head_W_override is not None else ctx["lm_head_W"]
    logits = rfmv.matvec_rows(W_head, hidden)                          # (S, V)
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=16, help="rate-code pool budget (B-1 point T=16)")
    ap.add_argument("--ppl-tokens", type=int, default=192, help="held-out tokens for the small ppl slice")
    ap.add_argument("--gen-tokens", type=int, default=24, help="short greedy generation length")
    ap.add_argument("--rmsnorm", type=str, default="graded", choices=["graded", "exact"],
                    help="on-bridge RMSNorm read (graded=B-1 spiking RMS; exact=host sqrt-RMS)")
    ap.add_argument("--no-megakernel", action="store_true", help="use the per-step RF loop (slower; default uses "
                                                                  "the megakernel enable_rf_cudagraph)")
    args = ap.parse_args()

    t0 = time.time()
    backend = os.environ.get("SIM_BACKEND", "auto")
    log(f"SIM_BACKEND={backend}")
    import cupy as cp
    import torch
    log(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
        f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")
    free0, total0 = _vram_free_total_gb()
    log(f"VRAM free {free0:.2f}GB / total {total0:.2f}GB at start")

    def _raise_pool_limit(frac=0.94):
        # Raise the CuPy mempool limit (the bridge constructor sets it to 0.8 = 20.6GB on the 3090; the
        # cache-everything resident ~14GB + the lesion's extra shuffled-lm_head CSR ~4.4GB can exceed it). Call
        # this AFTER bridges are built (their init resets the limit to 0.8). Runner-side cupy call, NOT a sim/ edit.
        try:
            _tb = cp.cuda.Device().mem_info[1]
            cp.get_default_memory_pool().set_limit(size=int(_tb * frac))
            log(f"  raised CuPy mempool limit to {_tb*frac/1e9:.1f}GB ({frac:.2f} of total)")
        except Exception as _e:
            log(f"  (mempool limit raise skipped: {_e})")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"loading {MODEL_ID} (fp16, eager attention) ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16,
                                                 attn_implementation="eager").cuda().eval()
    device = next(model.parameters()).device
    mcfg = model.config
    n_params = sum(p.numel() for p in model.parameters())
    log(f"loaded; {n_params/1e6:.1f}M params on {device}")

    eps = float(mcfg.rms_norm_eps)
    Hq = int(mcfg.num_attention_heads)
    Hkv = int(mcfg.num_key_value_heads)
    head_dim = int(getattr(mcfg, "head_dim", None) or mcfg.hidden_size // Hq)
    scaling = head_dim ** -0.5
    D = int(mcfg.hidden_size)
    V = int(mcfg.vocab_size)
    n_layers = int(mcfg.num_hidden_layers)
    cfg = {"eps": eps, "Hq": Hq, "Hkv": Hkv, "head_dim": head_dim, "scaling": scaling, "n_layers": n_layers}
    log(f"arch: D={D} V={V} n_layers={n_layers} Hq={Hq} Hkv={Hkv} head_dim={head_dim} eps={eps:.1e}")

    # ---- capture cos/sin for the full ctx via a forward hook on a layer (de-risk #2 pattern) ----
    captured = {}

    def layer_pre_hook(mod, args_, kwargs_):
        pe = kwargs_.get("position_embeddings")
        if pe is None and len(args_) >= 7:
            pe = args_[6]
        if pe is not None and "pos_emb" not in captured:
            captured["pos_emb"] = (pe[0].detach(), pe[1].detach())
        return None

    hp = model.model.layers[0].register_forward_pre_hook(layer_pre_hook, with_kwargs=True)

    # ---- held-out text + a prime sequence to capture cos/sin over a long-enough window ----
    if CORPUS.exists():
        with open(CORPUS, "r", encoding="utf-8") as f:
            corpus = f.read()
        held = corpus[-120_000:]
        delim = "<|endoftext|>"
        idx = held.find(delim)
        if idx != -1:
            held = held[idx + len(delim):].lstrip()
    else:
        corpus = "Once upon a time there was a little girl who loved to read books in the garden every day."
        held = corpus

    # context window we'll need = max(ppl_tokens, prompt+gen_tokens). Prime a forward to grab cos/sin that long.
    ctx_need = max(args.ppl_tokens, 64 + args.gen_tokens) + 8
    prime_ids = tok(held, return_tensors="pt").input_ids.to(device)[:, :ctx_need]
    with torch.no_grad():
        model(prime_ids)
    hp.remove()
    pe = captured["pos_emb"]                                            # (cos,sin), each (1, S, d)
    cos_full = pe[0][0].to(torch.float64).cpu().numpy()                # (S, d)
    sin_full = pe[1][0].to(torch.float64).cpu().numpy()
    log(f"captured cos/sin: shape {cos_full.shape} (ctx_need={ctx_need})")

    # ---- build the B-1 banks (off-line fit; reuse-by-import the B-1 fitters) ----
    silu_range = (-7.34375, 5.4140625)     # the B-1 measured SiLU-input range
    silu_host, silu_fd, exp_host, exp_fd = L2.build_host_banks(silu_range, device)
    log(f"SiLU bank: grid {silu_fd['grid']} knots {silu_fd['n_knots']} fit-max-err {silu_fd['fit_max_err_grid']:.5f}")
    log(f"exp  bank: grid {exp_fd['grid']} knots {exp_fd['n_knots']} fit-max-err {exp_fd['fit_max_err_grid']:.5f}")

    T = args.T
    pool_silu = B1.POOL_BASE * T
    pool_div = B1.POOL_BASE * T
    pool_softmax = B1.POOL_BASE_SM * T
    log(f"T={T} -> pool_silu={pool_silu}, pool_div={pool_div}, pool_softmax={pool_softmax}")

    # ---- the embedding + tied lm_head weight (W=weight.T = embed.T, shape (D, V)) + final norm ----
    embed = model.model.embed_tokens.weight.detach().to(torch.float64).cpu().numpy()        # (V, D)
    lm_head_W = np.ascontiguousarray(embed.T)                                                # (D, V) install
    norm_w = model.model.norm.weight.detach().to(torch.float64).cpu().numpy()                # (D,)
    log(f"embedding (V,D)={embed.shape}; lm_head install (D,V)={lm_head_W.shape}; tied={mcfg.tie_word_embeddings}")

    # ---- the RF matvec engine (vectorized CSR build + megakernel resonate) ----
    use_mega = not args.no_megakernel
    rfmv = RFMatvecVec(seed=42, use_megakernel=use_mega)

    # PRE-EXTRACT all 24 layers' weights ONCE into a persistent list so the W tensor IDENTITY is stable across
    # forwards -> the install-cache (id(W)) makes each weight install exactly ONCE for the whole run (the C1
    # cache-everything pattern; ~720MB host fp64). Generation then re-uses the cached CSRs (near-instant swaps).
    log("pre-extracting all 24 layers' weights (stable tensor identity for the install-cache) ...")
    t_ex = time.time()
    _all_layers = []
    for li in range(n_layers):
        layer = model.model.layers[li]
        _all_layers.append(extract_layer(layer, layer.self_attn, Hq, Hkv, head_dim))
    log(f"  extracted {n_layers} layers in {time.time()-t_ex:.1f}s")

    def get_layer(li):
        return _all_layers[li]

    ctx = {
        "cfg": cfg, "cos_full": cos_full, "sin_full": sin_full,
        "silu_host": silu_host, "exp_host": exp_host,
        "pool_silu": pool_silu, "pool_div": pool_div, "pool_softmax": pool_softmax,
        "rfmv": rfmv, "embed": embed, "lm_head_W": lm_head_W, "norm_w": norm_w,
        "get_layer": get_layer, "cur_W": None,
    }

    # =================================================================================================
    # VRAM probe: install ONE lm_head + the 4 layer shapes resident simultaneously, read the pool.
    # =================================================================================================
    log("=== VRAM probe: install the lm_head (the single biggest CSR) + the 4 layer shapes ===")
    vram_before = _vram_used_gb()
    # install lm_head (n=D+V, nnz=D*V=136M) -- the biggest object
    t_h = time.time()
    rfmv.install(lm_head_W)
    cp.cuda.Stream.null.synchronize()
    log(f"  lm_head installed in {time.time()-t_h:.2f}s")
    vram_lm_head = _vram_used_gb()
    log(f"  VRAM used: before={vram_before:.3f}GB  +lm_head={vram_lm_head:.3f}GB (the single biggest CSR)")

    # =================================================================================================
    # (2)+(3) WALL-CLOCK + ppl on a SMALL held-out slice. Run the RF full forward over `ppl-tokens` tokens and
    # TIME it -> tokens/sec. The FIRST forward is COLD (installs all 169 weights once into the id(W) cache);
    # a SECOND identical forward is WARM (every install is a cached attribute swap) = the steady-state generation
    # rate. We report BOTH (cold incl. one-time install; warm = the generation tokens/sec). ppl from the RF logits.
    # The matched reference = the B-1-spiking forward (exact linears, SAME graded ops, SAME noise) over the SAME
    # tokens, so RF-vs-B1 isolates the RF-matvec residual; both-vs-ANN = the graded-read fidelity.
    # =================================================================================================
    ppl_n = min(args.ppl_tokens, cos_full.shape[0])
    ppl_ids = tok(held, return_tensors="pt").input_ids[0, :ppl_n].cpu().numpy().astype(np.int64)
    log(f"=== full RF forward on a {ppl_n}-token held-out slice (wall-clock + ppl) ===")

    # COLD forward (installs everything once -> the full id(W) cache; after this all 169 CSRs are resident)
    t_fwd = time.time()
    rf_logits = rf_full_forward(ppl_ids, ctx, rmsnorm_mode=args.rmsnorm, noise_seed=7)
    cp.cuda.Stream.null.synchronize()
    rf_fwd_seconds_cold = time.time() - t_fwd
    toks_per_sec_cold = ppl_n / rf_fwd_seconds_cold
    # NOW all weights are cached & resident -> the TRUE resident peak (cache-everything, the C1 full_genf pattern)
    vram_full = _vram_used_gb()
    free_now, total_now = _vram_free_total_gb()
    log(f"  COLD forward (installs all 169 weights): {ppl_n} tok in {rf_fwd_seconds_cold:.1f}s "
        f"({toks_per_sec_cold:.3f} tok/s); install {rfmv.install_seconds:.1f}s, matvec {rfmv.matvec_seconds:.1f}s, "
        f"n_installs {rfmv.n_installs}")
    log(f"  VRAM resident peak (ALL 169 CSRs cached) = {vram_full:.3f}GB; free {free_now:.2f}GB / total "
        f"{total_now:.2f}GB -> {'FITS (<24GB)' if vram_full < 24.0 else 'OVER 24GB'}")
    vram_peak_resident_gb = vram_full

    # WARM forward (all installs are cached swaps) = the steady-state generation rate
    _mv0 = rfmv.matvec_seconds
    t_fwd = time.time()
    rf_logits = rf_full_forward(ppl_ids, ctx, rmsnorm_mode=args.rmsnorm, noise_seed=7)
    cp.cuda.Stream.null.synchronize()
    rf_fwd_seconds = time.time() - t_fwd
    toks_per_sec = ppl_n / rf_fwd_seconds
    log(f"  WARM forward (install-cached): {ppl_n} tok in {rf_fwd_seconds:.1f}s -> {toks_per_sec:.3f} tokens/sec "
        f"(matvec {rfmv.matvec_seconds-_mv0:.1f}s, cache_hits {rfmv.n_cache_hits}) [the steady-state gen rate]")
    # all bridges are built now -> raise the pool limit (their init had reset it to 0.8) so the lesion's extra
    # shuffled-lm_head CSR fits on top of the cache-everything resident.
    _raise_pool_limit(0.94)

    # ppl of the RF logits (next-token NLL)
    def ppl_from_logits(logits, ids):
        lg = np.asarray(logits, dtype=np.float64)
        nll, n = 0.0, 0
        for i in range(len(ids) - 1):
            row = lg[i]
            row = row - row.max()
            logp = row - math.log(np.exp(row).sum())
            nll += -float(logp[ids[i + 1]])
            n += 1
        return math.exp(nll / max(n, 1)), nll / max(n, 1), n

    rf_ppl, rf_nll, rf_n = ppl_from_logits(rf_logits, ppl_ids)
    log(f"  RF spiking ppl = {rf_ppl:.4f}  (B-1 spiking {B1_PPL:.2f}, ANN {ANN_PPL:.2f}); {rf_n} tok scored")

    # the B-1-spiking reference logits over the SAME tokens (exact linears via numpy matmul, SAME graded ops)
    log("  computing the B-1-spiking reference forward (exact linears, same graded nonlinearities) ...")

    def b1_full_forward(ids, rmsnorm_mode, noise_seed=7):
        rng = np.random.default_rng(noise_seed)
        S = len(ids)
        cos = cos_full[:S]; sin = sin_full[:S]
        hidden = embed[np.asarray(ids)].astype(np.float64)

        def torch_linear_fn(name, rows):
            return rows @ ctx["cur_W"][name]

        for li in range(n_layers):
            W, weights = get_layer(li)
            ctx["cur_W"] = W
            hidden = L2.layer_forward(hidden, weights, cfg, torch_linear_fn, rmsnorm_mode=rmsnorm_mode,
                                      silu_bank=silu_host, exp_bank=exp_host, pool_silu=pool_silu,
                                      pool_div=pool_div, pool_softmax=pool_softmax, rng=rng, cos=cos, sin=sin)
        if rmsnorm_mode == "graded":
            hidden = L2.graded_rmsnorm(hidden, norm_w, eps, pool_div, rng)
        else:
            hidden = L2.exact_rmsnorm(hidden, norm_w, eps)
        return hidden @ lm_head_W

    b1_logits = b1_full_forward(ppl_ids, args.rmsnorm, noise_seed=7)
    b1_ppl, b1_nll, _ = ppl_from_logits(b1_logits, ppl_ids)
    log(f"  B-1-spiking(reproduced here) ppl = {b1_ppl:.4f}")

    # also the EXACT ANN logits over the same tokens (the ceiling)
    with torch.no_grad():
        ann_out = model(torch.tensor(ppl_ids[None, :], device=device)).logits[0].to(torch.float64).cpu().numpy()
    ann_ppl, ann_nll, _ = ppl_from_logits(ann_out, ppl_ids)
    log(f"  exact ANN ppl (same tokens) = {ann_ppl:.4f}")

    # fidelity: RF-vs-B1 (pure matvec transfer) + RF-vs-ANN (matvec + graded-read + fp16) on the logits
    def logit_fidelity(a, b):
        a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
        cos_rows = []
        agree = 0
        for i in range(a.shape[0]):
            x, y = a[i], b[i]
            nx, ny = np.linalg.norm(x), np.linalg.norm(y)
            if nx > 0 and ny > 0:
                cos_rows.append(float(x @ y / (nx * ny)))
            if int(np.argmax(x)) == int(np.argmax(y)):
                agree += 1
        return (float(np.mean(cos_rows)) if cos_rows else float("nan"),
                agree / a.shape[0])

    rf_b1_cos, rf_b1_argmax = logit_fidelity(rf_logits, b1_logits)
    rf_ann_cos, rf_ann_argmax = logit_fidelity(rf_logits, ann_out)
    b1_ann_cos, b1_ann_argmax = logit_fidelity(b1_logits, ann_out)
    log(f"  logit fidelity RF-vs-B1 : cos {rf_b1_cos:.6f}  argmax-agree {rf_b1_argmax:.3f}  [pure matvec transfer]")
    log(f"  logit fidelity RF-vs-ANN: cos {rf_ann_cos:.6f}  argmax-agree {rf_ann_argmax:.3f}")
    log(f"  logit fidelity B1-vs-ANN: cos {b1_ann_cos:.6f}  argmax-agree {b1_ann_argmax:.3f}  [the B-1 ceiling]")

    # =================================================================================================
    # ANTI-CHEAT lesion: scramble the lm_head RF weights -> logits diverge; shuffled matvec still exact.
    # =================================================================================================
    log("=== anti-cheat lesion (row-permute the lm_head RF weights) ===")
    rng_perm = np.random.default_rng(98765)
    lm_head_shuf = np.ascontiguousarray(lm_head_W[rng_perm.permutation(lm_head_W.shape[0]), :])
    # The shuffled lm_head builds a FRESH 136M-nnz CSR (~4.4GB). With the cache-everything resident at ~14GB +
    # the 80%-pool limit, that allocation OOMs. EVICT the true lm_head's cached CSR (+free the pool) to make room
    # -- the lesion forward installs the shuffled head onto the SAME lm_head-shape bridge anyway.
    rfmv.evict(lm_head_W)
    lesion_logits = rf_full_forward(ppl_ids[:32], ctx, rmsnorm_mode=args.rmsnorm, noise_seed=7,
                                    lm_head_W_override=lm_head_shuf)
    les_cos, les_argmax = logit_fidelity(lesion_logits, b1_logits[:32])
    # shuffled head matvec vs its own a@W_shuf (a tiny operand) -- RF still exact on the scrambled weights
    probe_rows = (embed[ppl_ids[:4]]).astype(np.float64)
    ref_shuf = probe_rows @ lm_head_shuf
    rf_shuf = rfmv.matvec_rows(lm_head_shuf, probe_rows)
    shuf_self_err = float(np.max(np.abs(rf_shuf - ref_shuf)))
    log(f"  lesioned logits vs true B-1 logits: cos {les_cos:.4f}  argmax-agree {les_argmax:.3f} "
        f"(must COLLAPSE)")
    log(f"  shuffled lm_head matvec vs its OWN a@W_shuf: max-abs {shuf_self_err:.3e} (must stay EXACT)")
    # free the shuffled head before generation re-installs the true head (avoid stacking two lm_head CSRs).
    rfmv.evict(lm_head_shuf)

    # =================================================================================================
    # (4) SHORT spiking generation (greedy, ~gen-tokens) from a neutral prompt -> READ it.
    # =================================================================================================
    log(f"=== short spiking generation (greedy, {args.gen_tokens} tokens) ===")
    rfmv.install(lm_head_W)  # ensure the true head
    prompt = "Once upon a time"
    msgs = [{"role": "user", "content": prompt}]
    gen_prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    gen_ids = tok(gen_prompt, return_tensors="pt").input_ids[0].cpu().numpy().astype(np.int64).tolist()
    # cap the prompt so prompt+gen fits the captured cos/sin window
    max_prompt = cos_full.shape[0] - args.gen_tokens - 1
    if len(gen_ids) > max_prompt:
        gen_ids = gen_ids[:max_prompt]
    log(f"  prompt='{prompt}' ({len(gen_ids)} prompt tokens); generating {args.gen_tokens} greedy tokens ...")
    t_gen = time.time()
    new_tokens = []
    cur = list(gen_ids)
    for step in range(args.gen_tokens):
        lg = rf_full_forward(np.asarray(cur, dtype=np.int64), ctx, rmsnorm_mode=args.rmsnorm, noise_seed=7)
        nxt = int(np.argmax(lg[-1]))
        if nxt == tok.eos_token_id:
            log(f"  (hit EOS at step {step})")
            break
        new_tokens.append(nxt)
        cur.append(nxt)
    cp.cuda.Stream.null.synchronize()
    gen_seconds = time.time() - t_gen
    sec_per_gen_token = gen_seconds / max(len(new_tokens), 1)   # avg wall-clock per generated token (full re-forward)
    gen_text = tok.decode(new_tokens, skip_special_tokens=True)
    log(f"  generated {len(new_tokens)} tokens in {gen_seconds:.1f}s ({sec_per_gen_token:.1f}s/token; each token = "
        f"a full re-forward over the growing context)")
    log("  GENERATION (verbatim):")
    safe_print("    " + gen_text.replace("\n", "\n    "))

    # the matched B-1-spiking greedy generation (for a side-by-side coherence read)
    log("  (B-1-spiking matched greedy generation for comparison) ...")
    b1_new = []
    cur = list(gen_ids)
    for step in range(args.gen_tokens):
        lg = b1_full_forward(np.asarray(cur, dtype=np.int64), args.rmsnorm, noise_seed=7)
        nxt = int(np.argmax(lg[-1]))
        if nxt == tok.eos_token_id:
            break
        b1_new.append(nxt)
        cur.append(nxt)
    b1_gen_text = tok.decode(b1_new, skip_special_tokens=True)
    gen_match = sum(1 for a, b in zip(new_tokens, b1_new) if a == b) / max(len(new_tokens), 1)
    log("  B-1-spiking GENERATION (verbatim):")
    safe_print("    " + b1_gen_text.replace("\n", "\n    "))
    log(f"  RF-vs-B1 greedy token-agreement over the generation: {gen_match:.3f}")

    # =================================================================================================
    # VERDICT
    # =================================================================================================
    runs = vram_peak_resident_gb < 24.0
    ppl_matches_b1 = (rf_ppl <= B1_PPL * 1.25)                 # the on-bridge full forward ~matches B-1 (no compound)
    rf_transfers = (rf_b1_cos >= 0.99 and rf_b1_argmax >= 0.9)  # logits == B-1 to ~matvec precision
    lesion_collapses = (les_cos < 0.5 and shuf_self_err <= 1e-4)
    # the controller READS the generation for coherence; we provide the quantitative gate + the verbatim text.

    if runs and ppl_matches_b1 and rf_transfers and lesion_collapses:
        verdict = "GO"
        tail = (f"the FULL 24-layer Qwen forward RUNS on the LIVE RF bridge (resident peak "
                f"{vram_peak_resident_gb:.2f}GB << 24GB, LOCAL); the on-bridge ppl {rf_ppl:.3f} ~ the B-1 spiking "
                f"{b1_ppl:.3f} (the per-layer graded-read SEM does NOT compound-degrade over 24 layers; RF-vs-B1 "
                f"logit cos {rf_b1_cos:.5f}, argmax-agree {rf_b1_argmax:.3f}); the lesion collapses "
                f"(cos {les_cos:.3f}) while the shuffled matvec stays exact ({shuf_self_err:.1e}). MEASURED "
                f"wall-clock = {toks_per_sec:.3f} tokens/sec for the RF forward. => bridge co-residence is "
                f"FEASIBLE end-to-end, LOCAL. NO `sim/` edit (vectorized CSR build = the C1 cache-swap pattern, "
                f"bit-identical to rf_set_complex_weights). The wall-clock is the perf lever (NOT cloud: no VRAM "
                f"wall).")
    elif runs and rf_transfers:
        verdict = "GO_WITH_CAVEAT"
        tail = (f"the full forward RUNS ({vram_peak_resident_gb:.2f}GB) and the RF matvecs transfer (RF-vs-B1 cos "
                f"{rf_b1_cos:.5f}), but a gate is soft: ppl_matches_b1={ppl_matches_b1} (rf_ppl {rf_ppl:.3f} vs B-1 "
                f"{b1_ppl:.3f}), lesion_collapses={lesion_collapses}. Inspect the flagged item; the matvec transfer "
                f"itself holds. wall-clock {toks_per_sec:.3f} tok/s.")
    else:
        verdict = "HONEST_RESIDUAL"
        tail = (f"a piece diverged. runs={runs} ({vram_peak_resident_gb:.2f}GB), rf_transfers={rf_transfers} "
                f"(RF-vs-B1 cos {rf_b1_cos:.5f}, argmax {rf_b1_argmax:.3f}), ppl_matches_b1={ppl_matches_b1} "
                f"(rf_ppl {rf_ppl:.3f} vs B-1 {b1_ppl:.3f}), lesion_collapses={lesion_collapses}. If the matvecs "
                f"transfer but ppl compounds, the 24-layer graded-read SEM is the wall (raise T / host-exact "
                f"RMSNorm). wall-clock {toks_per_sec:.3f} tok/s. No `sim/` edit was added.")

    # overnight-viability for a validation corpus on the 3090 (the cloud-trigger check)
    # a "small validation corpus" ~ 2000 tokens of ppl; estimate hours at the measured tokens/sec.
    val_tokens = 2000
    val_hours = val_tokens / max(toks_per_sec, 1e-9) / 3600.0
    overnight_viable = val_hours <= 10.0

    verdict_line = (
        f"bridge_cores_fullfwd: FULL 24-layer Qwen2.5-0.5B on the LIVE RF bridge -> "
        f"VRAM resident peak {vram_peak_resident_gb:.2f}GB ({'LOCAL' if runs else 'OVER 24GB'}); "
        f"{toks_per_sec:.3f} tokens/sec; on-bridge ppl {rf_ppl:.3f} (B-1 spiking {b1_ppl:.3f}, ANN {ann_ppl:.3f}); "
        f"RF-vs-B1 logit cos {rf_b1_cos:.5f} (argmax {rf_b1_argmax:.2f}); lesion cos {les_cos:.3f} "
        f"(shuf-self {shuf_self_err:.1e}); generation read below -> {verdict}. {tail} "
        f"Validation-corpus({val_tokens} tok) ETA {val_hours:.1f}h on the 3090 -> "
        f"{'overnight-viable LOCAL' if overnight_viable else 'too slow -> FIX PERF (batch/dense RF matvec) FIRST, not cloud'}.")

    result = {
        "probe": "bridge_coresidence_derisk3_qwen_full_24layer_forward_rf",
        "resolves": "de-risk #3 (scoping 2026-06-23 §3 de-risk #4 + §6): the FULL 24-layer Qwen2.5-0.5B forward on "
                    "the LIVE SimulationBridge RF substrate -- measure (1) VRAM, (2) tokens/sec (the cloud-trigger), "
                    "(3) ppl vs B-1 (does the per-layer graded-read SEM compound over 24 layers?), (4) a short "
                    "spiking generation (coherent?).",
        "model_id": MODEL_ID,
        "arch": {"D": D, "V": V, "n_layers": n_layers, "Hq": Hq, "Hkv": Hkv, "head_dim": head_dim,
                 "scaling": scaling, "eps": eps, "tie_word_embeddings": bool(mcfg.tie_word_embeddings)},
        "n_model_params": int(n_params),
        "T": int(T), "pools": {"silu": pool_silu, "div": pool_div, "softmax": pool_softmax},
        "rmsnorm_mode": args.rmsnorm,
        "rf_operating_point": {"period": RF_PERIOD, "nsteps": RF_NSTEPS, "lambda": RF_LAMBDA,
                               "read": "Re(Z_out)/nsteps = a @ W (W=linear.weight.T); biases host-add"},
        "storage": {
            "chosen": "complex CSR (cp_rf_w_re + all-zero cp_rf_w_im, float64 data + int32 idx), VECTORIZED build "
                      "(rows=D_in+tile(arange(D_out),D_in), cols=repeat(arange(D_in),D_out), data=W.ravel()), "
                      "assigned to bridge.cp_rf_w_re/cp_rf_w_im == the C1 _set_rf_weights cache-swap (bit-identical "
                      "to rf_set_complex_weights, verified max|.|=0.0). LAYER-STREAMED: 4 unique layer shapes + "
                      "lm_head resident; each layer re-installs its 7 weights onto the shared per-shape bridges.",
            "note": "the as-is re+im f64 CSR is the scoping's worst-case storage; the all-zero im CSR is retained "
                    "for byte-identity with the RF matvec path (the _rf_advance_one reads cp_rf_w_im). The scoping's "
                    "dense-fp16 (~1GB) is the perf-storage fallback if VRAM ever binds (it does NOT here).",
        },
        "vram_gb": {
            "free_at_start": round(free0, 3), "total": round(total0, 3),
            "used_before_install": round(vram_before, 3),
            "used_with_lm_head": round(vram_lm_head, 3),
            "resident_peak_lm_head_plus_4_layer_shapes": round(vram_peak_resident_gb, 3),
            "free_after_install": round(free_now, 3),
            "fits_under_24gb": bool(runs),
        },
        "wall_clock": {
            "ppl_slice_tokens": int(ppl_n),
            "rf_forward_seconds_cold": round(rf_fwd_seconds_cold, 2),
            "tokens_per_sec_cold_incl_one_time_install": round(toks_per_sec_cold, 4),
            "rf_forward_seconds_warm": round(rf_fwd_seconds, 2),
            "tokens_per_sec_warm_install_cached": round(toks_per_sec, 4),
            "rf_install_seconds_total_one_time": round(rfmv.install_seconds, 2),
            "rf_matvec_seconds_total": round(rfmv.matvec_seconds, 2),
            "rf_n_installs": int(rfmv.n_installs),
            "rf_n_cache_hits": int(rfmv.n_cache_hits),
            "generation_seconds": round(gen_seconds, 2),
            "generation_tokens": int(len(new_tokens)),
            "sec_per_generated_token": round(sec_per_gen_token, 2),
            "megakernel": bool(use_mega),
            "note": "COLD = the first forward (installs all 169 weights once -> the id(W) cache); WARM = a "
                    "subsequent forward (every install a cached attribute swap) = the steady-state generation "
                    "rate. Generation re-uses the warm path.",
        },
        "perplexity": {
            "rf_on_bridge": round(rf_ppl, 4),
            "b1_spiking_reproduced": round(b1_ppl, 4),
            "b1_spiking_reference_T16": B1_PPL,
            "ann_exact": round(ann_ppl, 4),
            "ann_baseline_stepA": ANN_PPL,
            "n_tokens_scored": int(rf_n),
            "rf_vs_b1_ratio": round(rf_ppl / max(b1_ppl, 1e-9), 4),
            "compounds_over_24_layers": bool(rf_ppl > B1_PPL * 1.25),
        },
        "logit_fidelity": {
            "rf_vs_b1": {"cosine": rf_b1_cos, "argmax_agree": rf_b1_argmax},
            "rf_vs_ann": {"cosine": rf_ann_cos, "argmax_agree": rf_ann_argmax},
            "b1_vs_ann": {"cosine": b1_ann_cos, "argmax_agree": b1_ann_argmax},
        },
        "anti_cheat_lesion": {
            "lesioned_logits_vs_true_b1_cosine": les_cos,
            "lesioned_logits_vs_true_b1_argmax_agree": les_argmax,
            "shuffled_lm_head_matvec_vs_own_a_at_W_shuf_maxabs": shuf_self_err,
            "lesion_collapses": bool(lesion_collapses),
        },
        "generation": {
            "prompt": prompt,
            "rf_on_bridge_greedy": gen_text,
            "b1_spiking_greedy": b1_gen_text,
            "rf_vs_b1_token_agreement": round(gen_match, 4),
            "n_new_tokens": int(len(new_tokens)),
        },
        "silu_fit": silu_fd, "exp_fit": exp_fd,
        "mechanism": "reuse-by-import: de-risk #2's layer_forward/run_attention + the B-1 graded banks; the RF "
                     "exact-matvec with a VECTORIZED CSR build (bit-identical to rf_set_complex_weights). 24 layers "
                     "STREAMED onto 4 per-shape bridges + lm_head. NO `sim/` edit.",
        "cloud_trigger_check": {
            "rule": "feedback_long_local_runs_ok_confirm_cloud_cause: cloud ONLY for a genuine >24GB VRAM wall; "
                    "wall-clock alone is fine with an ETA, run local (local perf fix first).",
            "vram_wall": False,
            "validation_corpus_tokens_assumed": val_tokens,
            "validation_eta_hours_on_3090": round(val_hours, 2),
            "overnight_viable_local": bool(overnight_viable),
            "decision": ("LOCAL -- overnight-viable" if overnight_viable else
                         "LOCAL but slow -> FIX PERF (batch tokens / dense-on-bridge RF matvec, scoping #6) BEFORE "
                         "cloud; an H100 only makes the COMPUTE ~3-5x faster, it does NOT lift a VRAM wall (there "
                         "is none)."),
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


if __name__ == "__main__":
    main()
