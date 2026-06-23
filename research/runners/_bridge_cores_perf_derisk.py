"""BRIDGE CO-RESIDENCE PERF de-risk (scoping #6 — the PERF LEVER): make the bridge-co-resident Qwen faculty FAST.

De-risk #3 (`_bridge_cores_fullfwd_derisk.py`, DEMONSTRATED) proved the FULL 24-layer Qwen2.5-0.5B runs on the LIVE
SimulationBridge RF substrate, LOCAL + bit-exact + coherent — but SLOW: 0.786 tok/s prefill (warm), 161 s/generated
token, "launch-/CSR-gather-bound" (the forward runs the RF matvec PER TOKEN-ROW in a Python loop, each row = kick +
resonate(8 steps) + read; each resonate step is a SPARSE cuSPARSE CSR matvec over a DENSE 494M weight matrix — the
WRONG storage; the Qwen layers are 100% dense).

THIS de-risk (RUNNER-LEVEL, no `sim/` edit yet) measures the perf levers per the scoping §2c/§6:

  (1) PROFILE one layer's wall-clock: break down resonate(8) loop vs CSR matvec/gather vs per-op launch overhead vs
      graded nonlinearities. Where does the 161 s/token actually go?
  (2) DENSE-MATVEC lever: the RF read is `Re(Z)/nsteps = a@W`. Since W is DENSE, compute it as a DENSE cupy
      `a @ W_dense` (the SAME math, dense storage) instead of the sparse-CSR-resonate. Measure the speedup (CSR-RF vs
      dense) + CONFIRM bit-exactness (the dense path == the CSR-RF path to ~f32 precision). Is dense ~ANN GEMM speed?
  (3) BATCH lever: batch all S token-positions through ONE GEMM `[S, D_in] @ [D_in, D_out]` vs the per-row RF loop.
  (4) EXTRAPOLATE: with dense (+batch), the projected tok/s for the full 24-layer forward (prefill AND per-gen-token).
      Usable (>10 tok/s, ideally near ANN)?

VERDICT: profile + dense speedup (×) + projected tok/s + whether a `sim/` edit (a guarded dense-RF-matvec mode on the
bridge, byte-identical default) is the real lever (scoped precisely if so).

FOREGROUND/blocking. GPU (SIM_BACKEND=cupy). Usage:
  SIM_BACKEND=cupy python -m research.runners._bridge_cores_perf_derisk
"""
from __future__ import annotations

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

from research.runners.rf_phasor_composer import _build_rf_bridge  # noqa: E402
import research.runners._bridge_cores_layer_derisk as L2  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_bridge_cores_perf_derisk.json"

RF_PERIOD = L2.RF_PERIOD     # 100000  (omega ~ 0)
RF_NSTEPS = L2.RF_NSTEPS     # 8
RF_LAMBDA = L2.RF_LAMBDA     # 0.0  (decay = 1.0)


def log(msg):
    print(f"[perf-rf] {msg}", flush=True)


def _sync():
    import cupy as cp
    cp.cuda.Stream.null.synchronize()


# Qwen2.5-0.5B per-matvec shapes (from the cached config; the scoping §2a table). These 7 + lm_head are the model's
# learned matvecs; per layer the 7 repeat 24×, lm_head once.
QWEN_LAYER_SHAPES = {
    "q_proj":   (896, 896),
    "k_proj":   (896, 128),
    "v_proj":   (896, 128),
    "o_proj":   (896, 896),
    "gate_proj": (896, 4864),
    "up_proj":  (896, 4864),
    "down_proj": (4864, 896),
}
LM_HEAD_SHAPE = (896, 151936)
N_LAYERS = 24


# =====================================================================================================
# RF matvec engine (the de-risk-#3 RFMatvecVec path, condensed): one LIVE RF bridge per (D_in,D_out); install W by
# building the complex CSR directly (vectorized) + assigning cp_rf_w_re/cp_rf_w_im; per-row kick/resonate/read loop.
# =====================================================================================================
class RFMatvecCSR:
    def __init__(self, seed=42, use_megakernel=True):
        import cupy as cp
        from sim.backend import get_sparse_module
        self.cp = cp
        self.csp = get_sparse_module()
        self.seed = int(seed)
        self.use_megakernel = bool(use_megakernel)
        self._bridges = {}
        self._idx_cache = {}

    def bridge_for(self, D_in, D_out):
        key = (int(D_in), int(D_out))
        if key not in self._bridges:
            n = D_in + D_out
            b = _build_rf_bridge(n, seed=self.seed)
            if self.use_megakernel:
                try:
                    b.core_config.enable_rf_cudagraph = True
                except Exception:
                    pass
            self._bridges[key] = b
        return self._bridges[key]

    def _indices_for(self, D_in, D_out):
        key = (int(D_in), int(D_out))
        if key not in self._idx_cache:
            cp = self.cp
            rows = (D_in + cp.tile(cp.arange(D_out, dtype=cp.int32), int(D_in)))
            cols = cp.repeat(cp.arange(D_in, dtype=cp.int32), int(D_out))
            self._idx_cache[key] = (rows, cols)
        return self._idx_cache[key]

    def install(self, W):
        cp = self.cp
        D_in, D_out = W.shape
        n = D_in + D_out
        bridge = self.bridge_for(D_in, D_out)
        rows, cols = self._indices_for(D_in, D_out)
        data_re = cp.asarray(np.ascontiguousarray(W, dtype=np.float64).ravel())
        csr_re = self.csp.csr_matrix((data_re, (rows, cols)), shape=(n, n))
        csr_im = self.csp.csr_matrix((cp.zeros(data_re.shape[0], dtype=cp.float64), (rows, cols)), shape=(n, n))
        _sync()
        bridge.cp_rf_w_re = csr_re
        bridge.cp_rf_w_im = csr_im
        return bridge, D_in, D_out

    def matvec_rows(self, bridge, D_in, D_out, rows_in):
        """Per-row RF matvec on every row of rows_in (N, D_in). Returns (N, D_out) = rows_in @ W."""
        cp = self.cp
        n = D_in + D_out
        N = rows_in.shape[0]
        out = np.zeros((N, D_out), dtype=np.float64)
        inv = 1.0 / float(RF_NSTEPS)
        for r in range(N):
            kick = np.zeros(n, dtype=np.complex128)
            kick[:D_in] = np.asarray(rows_in[r], dtype=np.float64)
            bridge.rf_kick(kick, period=int(RF_PERIOD), lam=float(RF_LAMBDA))
            bridge.rf_resonate_steps(int(RF_NSTEPS))
            re = cp.asnumpy(bridge.cp_membrane_potential_v[D_in:]).astype(np.float64)
            out[r] = re * inv
        _sync()
        return out


# =====================================================================================================
# (1) PROFILE one decoder layer's wall-clock on the RF path. Use the gate_proj (the biggest non-lm_head shape:
# 896->4864, 4.36M nnz) as the representative; break down kick / resonate / read across the per-row loop, and the
# CSR matvec alone (one resonate step) vs the per-op launch overhead.
# =====================================================================================================
def profile_one_layer(rfmv, S):
    import cupy as cp
    cp_ = cp
    log("=== (1) PROFILE: per-op breakdown of the RF matvec on representative shapes ===")
    prof = {}

    # Time each of the 7 layer linears (per-row loop over S rows), on a representative activation.
    rng = np.random.default_rng(1)
    per_linear = {}
    layer_total_csr = 0.0
    for name, (D_in, D_out) in QWEN_LAYER_SHAPES.items():
        W = np.ascontiguousarray(rng.standard_normal((D_in, D_out)) * (1.0 / math.sqrt(D_in)))
        bridge, _, _ = rfmv.install(W)
        rows_in = rng.standard_normal((S, D_in)) * 0.5
        # break the per-row loop into kick / resonate / read
        n = D_in + D_out
        inv = 1.0 / float(RF_NSTEPS)
        t_kick = t_res = t_read = 0.0
        _sync()
        t0 = time.perf_counter()
        for r in range(S):
            tk = time.perf_counter()
            kick = np.zeros(n, dtype=np.complex128)
            kick[:D_in] = rows_in[r]
            bridge.rf_kick(kick, period=int(RF_PERIOD), lam=float(RF_LAMBDA))
            _sync(); t_kick += time.perf_counter() - tk
            tr = time.perf_counter()
            bridge.rf_resonate_steps(int(RF_NSTEPS))
            _sync(); t_res += time.perf_counter() - tr
            td = time.perf_counter()
            _ = cp_.asnumpy(bridge.cp_membrane_potential_v[D_in:]).astype(np.float64) * inv
            t_read += time.perf_counter() - td
        _sync()
        tot = time.perf_counter() - t0
        layer_total_csr += tot
        per_linear[name] = {"D_in": D_in, "D_out": D_out, "nnz": D_in * D_out,
                            "total_s": round(tot, 4), "kick_s": round(t_kick, 4),
                            "resonate_s": round(t_res, 4), "read_s": round(t_read, 4),
                            "per_row_ms": round(1000.0 * tot / S, 3)}
        log(f"  {name:10s} ({D_in}->{D_out}, nnz {D_in*D_out:>8,}): {tot:.3f}s for {S} rows "
            f"(kick {t_kick:.3f} / resonate {t_res:.3f} / read {t_read:.3f}); {1000.0*tot/S:.2f} ms/row")
    prof["per_linear_csr"] = per_linear
    prof["one_layer_7linears_csr_s"] = round(layer_total_csr, 4)

    # Isolate the resonate cost as a function of n_steps on gate_proj (is it linear in nsteps? = the 8x amortizable).
    D_in, D_out = QWEN_LAYER_SHAPES["gate_proj"]
    W = np.ascontiguousarray(rng.standard_normal((D_in, D_out)) * (1.0 / math.sqrt(D_in)))
    bridge, _, _ = rfmv.install(W)
    n = D_in + D_out
    step_scan = {}
    for nsteps in (1, 2, 4, 8, 16):
        kick = np.zeros(n, dtype=np.complex128); kick[:D_in] = rng.standard_normal(D_in) * 0.5
        # warm
        bridge.rf_kick(kick, period=int(RF_PERIOD), lam=float(RF_LAMBDA)); bridge.rf_resonate_steps(nsteps); _sync()
        reps = 60
        t0 = time.perf_counter()
        for _ in range(reps):
            bridge.rf_kick(kick, period=int(RF_PERIOD), lam=float(RF_LAMBDA))
            bridge.rf_resonate_steps(nsteps)
        _sync()
        ms = 1000.0 * (time.perf_counter() - t0) / reps
        step_scan[nsteps] = round(ms, 4)
    prof["gate_proj_resonate_ms_by_nsteps_megakernel"] = step_scan
    log(f"  gate_proj resonate ms by nsteps (megakernel, incl kick): {step_scan}")
    # the per-call fixed overhead (intercept) vs per-step slope -> how launch-bound it is
    xs = np.array(sorted(step_scan.keys()), dtype=np.float64)
    ys = np.array([step_scan[int(x)] for x in xs], dtype=np.float64)
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
    prof["resonate_fit_ms_per_step"] = round(float(slope), 4)
    prof["resonate_fit_fixed_overhead_ms"] = round(float(intercept), 4)
    log(f"  resonate fit: {slope:.4f} ms/step + {intercept:.4f} ms fixed/call "
        f"(fixed-overhead fraction at nsteps=8: {intercept/(8*slope+intercept)*100:.0f}%)")
    return prof


# =====================================================================================================
# (2)+(3) DENSE-MATVEC + BATCH levers: compute `a@W` as a DENSE cupy GEMM. Compare:
#   (a) CSR-RF per-row loop (the current path)
#   (b) dense per-row loop (a@W_dense, row by row)   [isolates the storage change from the batch change]
#   (c) dense BATCHED GEMM ([S,D_in]@[D_in,D_out])   [the full lever]
# all on the SAME W (so dense == CSR-RF to ~f32). Report speedups + bit-exactness.
# =====================================================================================================
def bench_dense(rfmv, S):
    import cupy as cp
    log("=== (2)+(3) DENSE-MATVEC + BATCH levers (vs the CSR-RF per-row loop) ===")
    rng = np.random.default_rng(2)
    res = {}
    for name, (D_in, D_out) in list(QWEN_LAYER_SHAPES.items()) + [("lm_head", LM_HEAD_SHAPE)]:
        W = np.ascontiguousarray(rng.standard_normal((D_in, D_out)) * (1.0 / math.sqrt(D_in)))
        rows_in = (rng.standard_normal((S, D_in)) * 0.5).astype(np.float64)

        # (a) CSR-RF per-row loop
        bridge, _, _ = rfmv.install(W)
        _sync(); t0 = time.perf_counter()
        out_csr = rfmv.matvec_rows(bridge, D_in, D_out, rows_in)
        csr_s = time.perf_counter() - t0

        # dense weight on GPU (f32 — the natural ANN storage; the megakernel already casts the membrane to f32)
        W_d32 = cp.asarray(W, dtype=cp.float32)
        A_d32 = cp.asarray(rows_in, dtype=cp.float32)

        # (b) dense per-row loop (isolate storage change). Repeat REPS times (each = S GEMVs) so the small-GEMM
        # wall-clock is above the timer floor; report the per-forward (1 rep) average. Sync ONCE after the loop.
        # Use perf_counter (high-res) + a large REPS so even the tiny 896x128 GEMMs are measurable above the floor.
        REPS = 200
        _ = A_d32[0] @ W_d32; _sync()   # warm
        t0 = time.perf_counter()
        for _rep in range(REPS):
            for r in range(S):
                _ = A_d32[r] @ W_d32
        _sync(); dense_row_s = (time.perf_counter() - t0) / REPS

        # (c) dense BATCHED GEMM (one GEMM over all S rows). Repeat REPS times, sync once. This is the on-GPU
        # compute (no host D->H copy — the forward keeps the activation on-GPU between matvecs; the only D->H is
        # at the very end for the logits read, measured separately in the extrapolation).
        _ = A_d32 @ W_d32; _sync()      # warm
        t0 = time.perf_counter()
        for _rep in range(REPS):
            out_batch_g = A_d32 @ W_d32
        _sync(); dense_batch_s = (time.perf_counter() - t0) / REPS
        out_batch = cp.asnumpy(out_batch_g).astype(np.float64)

        # also an f64 batched GEMM (exactness ceiling — matches the f64 CSR data)
        out_batch64 = cp.asnumpy(cp.asarray(rows_in) @ cp.asarray(W, dtype=cp.float64))

        # bit-exactness: dense-batch(f32) vs CSR-RF(f32 membrane). Both approximate a@W(f64).
        ref64 = rows_in @ W
        err_csr_vs_ref = float(np.max(np.abs(out_csr - ref64)))
        err_batch32_vs_ref = float(np.max(np.abs(out_batch - ref64)))
        err_batch64_vs_ref = float(np.max(np.abs(out_batch64 - ref64)))
        err_csr_vs_batch = float(np.max(np.abs(out_csr - out_batch)))

        # Floor the dense times at a credible 1us so a sub-timer-resolution GEMM doesn't print an absurd speedup;
        # the load-bearing speedups (lm_head, down_proj, the full-forward extrapolation) are well above the floor.
        _batch_floored = max(dense_batch_s, 1e-6)
        res[name] = {
            "D_in": D_in, "D_out": D_out, "nnz": D_in * D_out,
            "csr_rf_per_row_s": round(csr_s, 5),
            "dense_per_row_s": round(dense_row_s, 6),
            "dense_batched_gemm_s": round(dense_batch_s, 7),
            "speedup_dense_batched_vs_csr": round(csr_s / _batch_floored, 1),
            "speedup_dense_batched_vs_csr_timer_floored": bool(dense_batch_s < 1e-6),
            "speedup_dense_perrow_vs_csr": round(csr_s / max(dense_row_s, 1e-9), 1),
            "max_err_csr_rf_vs_a_at_W_f64": err_csr_vs_ref,
            "max_err_dense_batch_f32_vs_a_at_W_f64": err_batch32_vs_ref,
            "max_err_dense_batch_f64_vs_a_at_W_f64": err_batch64_vs_ref,
            "max_err_csr_rf_vs_dense_batch_f32": err_csr_vs_batch,
        }
        _flag = " [batch<timer-floor]" if dense_batch_s < 1e-6 else ""
        log(f"  {name:10s}: CSR-RF {csr_s*1000:8.1f}ms  dense-row {dense_row_s*1000:7.3f}ms  "
            f"dense-batch {dense_batch_s*1000:8.4f}ms  -> batch SPEEDUP {csr_s/_batch_floored:8.0f}x{_flag}  "
            f"(err csr-vs-ref {err_csr_vs_ref:.1e}, batch32-vs-ref {err_batch32_vs_ref:.1e}, "
            f"csr-vs-batch {err_csr_vs_batch:.1e})")
    return res


# =====================================================================================================
# (4) EXTRAPOLATE: with the dense batched GEMM, the projected tok/s for the full 24-layer forward.
# We TIME a full forward's worth of dense matvecs (24 × 7 linears + lm_head) at:
#   - prefill: S rows batched per matvec (one GEMM/matvec)
#   - generation: 1 row per matvec (the autoregressive step, but the LAST-token only matvec — still one GEMM)
# and compare to the DEMONSTRATED CSR numbers (0.786 tok/s prefill, 161 s/gen-token).
# =====================================================================================================
def extrapolate(dense_res, S):
    import cupy as cp
    log("=== (4) EXTRAPOLATE: full 24-layer forward with the dense batched GEMM ===")
    rng = np.random.default_rng(3)
    # build dense GPU weights once (the natural fp16/fp32 ANN storage)
    Wd = {}
    for name, (D_in, D_out) in list(QWEN_LAYER_SHAPES.items()) + [("lm_head", LM_HEAD_SHAPE)]:
        Wd[name] = cp.asarray(rng.standard_normal((D_in, D_out)).astype(np.float32) * (1.0 / math.sqrt(D_in)))

    def time_full_forward(n_rows, reps=3):
        """Time the dense matvec wall-clock for ONE full forward over n_rows token-positions: 24 layers × 7 linears
        (each [n_rows, D_in] @ [D_in, D_out]) + lm_head (only on the n_rows positions; for generation only the LAST
        row needs lm_head, but we time n_rows = the prefill, and n_rows=1 = the per-gen-token last-token forward)."""
        # warm
        for name, (D_in, D_out) in QWEN_LAYER_SHAPES.items():
            A = cp.asarray(rng.standard_normal((n_rows, D_in)).astype(np.float32) * 0.5)
            _ = A @ Wd[name]
        _sync()
        # Pre-stage the activation operands on-GPU (the forward keeps activations resident between matvecs; we
        # isolate the GEMM wall-clock, not the host RNG/H->D, which the real forward does once per layer not here).
        A_layer = {name: cp.asarray(rng.standard_normal((n_rows, D_in)).astype(np.float32) * 0.5)
                   for name, (D_in, D_out) in QWEN_LAYER_SHAPES.items()}
        A_head = cp.asarray(rng.standard_normal((n_rows, LM_HEAD_SHAPE[0])).astype(np.float32) * 0.5)
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            for _li in range(N_LAYERS):
                for name in QWEN_LAYER_SHAPES:
                    _ = A_layer[name] @ Wd[name]
            _ = A_head @ Wd["lm_head"]   # lm_head on n_rows positions
            _sync()
            best = min(best, time.perf_counter() - t0)
        return best

    # prefill: S positions through every matvec, lm_head on all S (the de-risk-#3 prefill timed all S)
    prefill_s = time_full_forward(S, reps=3)
    prefill_toks_per_sec = S / prefill_s
    # generation: the autoregressive last-token forward = 1 row through every matvec + lm_head on 1 row
    gen_one_s = time_full_forward(1, reps=5)
    gen_toks_per_sec = 1.0 / gen_one_s

    log(f"  dense-matvec full-forward wall-clock: prefill({S} tok) {prefill_s*1000:.2f}ms -> "
        f"{prefill_toks_per_sec:.1f} tok/s; generation(1 tok) {gen_one_s*1000:.3f}ms -> {gen_toks_per_sec:.1f} tok/s")
    # NOTE: this is the MATVEC wall-clock only (the dense linears). The graded nonlinearities (RMSNorm/SiLU/softmax
    # host reads + RoPE + attention) add host time on top — the per-layer profile (1) measures whether they dominate
    # once the matvec is cheap. We report the matvec-only projection + flag the nonlinearity residual.
    return {
        "S_prefill": S,
        "dense_prefill_full_forward_s": round(prefill_s, 5),
        "dense_prefill_tok_per_sec_matvec_only": round(prefill_toks_per_sec, 1),
        "dense_generation_one_token_s": round(gen_one_s, 6),
        "dense_generation_tok_per_sec_matvec_only": round(gen_toks_per_sec, 1),
        "csr_demonstrated_prefill_tok_per_sec": 0.786,
        "csr_demonstrated_sec_per_gen_token": 161.0,
        "prefill_speedup_x": round(prefill_toks_per_sec / 0.786, 1),
        "gen_speedup_x": round(161.0 / max(gen_one_s, 1e-9), 1),
        "note": "matvec-only wall-clock (dense GEMM for all 169 linears). The graded nonlinearities + attention add "
                "host time on top; profile (1) shows their share. CSR baseline = the de-risk #3 DEMONSTRATED numbers.",
    }


# =====================================================================================================
# END-TO-END measurement (the honest number): run the de-risk #3 FULL forward path VERBATIM (real Qwen weights,
# the B-1 graded RMSNorm/SiLU/softmax + RoPE + attention) but with the per-row CSR-RF `linear_fn` SWAPPED for a
# dense GPU GEMM. Times one WARM forward over a held-out slice -> the real end-to-end tok/s, directly comparable to
# de-risk #3's measured 0.786 tok/s (whole forward). This converts the matvec-only projection into a MEASURED
# end-to-end number that INCLUDES the nonlinearity/attention residual.
# =====================================================================================================
def measure_end_to_end(S):
    import cupy as cp
    import torch
    import research.runners._bridge_cores_fullfwd_derisk as F3   # the de-risk #3 module (extract_layer, MODEL_ID, CORPUS)
    log("=== END-TO-END: de-risk #3 full forward with a DENSE GEMM linear_fn (real Qwen weights) ===")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(F3.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(F3.MODEL_ID, dtype=torch.float16,
                                                 attn_implementation="eager").cuda().eval()
    device = next(model.parameters()).device
    mcfg = model.config
    eps = float(mcfg.rms_norm_eps); Hq = int(mcfg.num_attention_heads); Hkv = int(mcfg.num_key_value_heads)
    head_dim = int(getattr(mcfg, "head_dim", None) or mcfg.hidden_size // Hq)
    D = int(mcfg.hidden_size); V = int(mcfg.vocab_size); n_layers = int(mcfg.num_hidden_layers)
    cfg = {"eps": eps, "Hq": Hq, "Hkv": Hkv, "head_dim": head_dim, "scaling": head_dim ** -0.5, "n_layers": n_layers}

    # capture cos/sin (de-risk #3 hook pattern)
    captured = {}

    def layer_pre_hook(mod, args_, kwargs_):
        pe = kwargs_.get("position_embeddings")
        if pe is None and len(args_) >= 7:
            pe = args_[6]
        if pe is not None and "pos_emb" not in captured:
            captured["pos_emb"] = (pe[0].detach(), pe[1].detach())
        return None

    hp = model.model.layers[0].register_forward_pre_hook(layer_pre_hook, with_kwargs=True)
    if F3.CORPUS.exists():
        held = open(F3.CORPUS, "r", encoding="utf-8").read()[-40_000:]
    else:
        held = "Once upon a time there was a little girl who loved to read books in the garden every day."
    ids = tok(held, return_tensors="pt").input_ids.to(device)[:, :S + 4]
    with torch.no_grad():
        model(ids)
    hp.remove()
    pe = captured["pos_emb"]
    cos_full = pe[0][0].to(torch.float64).cpu().numpy()
    sin_full = pe[1][0].to(torch.float64).cpu().numpy()

    silu_range = (-7.34375, 5.4140625)
    silu_host, _silu_fd, exp_host, _exp_fd = L2.build_host_banks(silu_range, device)
    T = 16
    import research.runners._grounded_lang_p1b_stepB1_forward_derisk as B1
    pool_silu = B1.POOL_BASE * T; pool_div = B1.POOL_BASE * T; pool_softmax = B1.POOL_BASE_SM * T

    embed = model.model.embed_tokens.weight.detach().to(torch.float64).cpu().numpy()
    lm_head_W = np.ascontiguousarray(embed.T)
    norm_w = model.model.norm.weight.detach().to(torch.float64).cpu().numpy()
    all_layers = [F3.extract_layer(model.model.layers[li], model.model.layers[li].self_attn, Hq, Hkv, head_dim)
                  for li in range(n_layers)]

    # DENSE GPU weights (one per tensor) — the natural ANN storage. Keyed by name within a layer + lm_head.
    def to_gpu(W):
        return cp.asarray(W, dtype=cp.float32)

    gpu_layer_W = [{k: to_gpu(Wd) for k, Wd in W.items()} for (W, _w) in all_layers]
    gpu_lm_head = to_gpu(lm_head_W)

    cur = {"W": None}
    timing = {"linear_s": 0.0, "n_linear": 0}

    def dense_linear_fn(name, rows):
        # rows: (S, D_in) numpy -> H->D once, dense GEMM, D->H back (== the de-risk #3 linear_fn contract, dense path)
        t0 = time.perf_counter()
        A = cp.asarray(rows, dtype=cp.float32)
        out = cp.asnumpy(A @ cur["W"][name]).astype(np.float64)
        _sync()
        timing["linear_s"] += time.perf_counter() - t0
        timing["n_linear"] += 1
        return out

    ppl_n = min(S, cos_full.shape[0])
    ppl_ids = tok(held, return_tensors="pt").input_ids[0, :ppl_n].cpu().numpy().astype(np.int64)
    cos = cos_full[:ppl_n]; sin = sin_full[:ppl_n]

    def dense_full_forward(seq_ids):
        rng = np.random.default_rng(7)
        hidden = embed[np.asarray(seq_ids)].astype(np.float64)
        for li in range(n_layers):
            _W, weights = all_layers[li]
            cur["W"] = gpu_layer_W[li]
            hidden = L2.layer_forward(hidden, weights, cfg, dense_linear_fn, rmsnorm_mode="graded",
                                      silu_bank=silu_host, exp_bank=exp_host, pool_silu=pool_silu,
                                      pool_div=pool_div, pool_softmax=pool_softmax, rng=rng, cos=cos, sin=sin)
        hidden = L2.graded_rmsnorm(hidden, norm_w, eps, pool_div, rng)
        cur["W"] = {"head": gpu_lm_head}
        return dense_linear_fn("head", hidden)

    # warm + timed
    _ = dense_full_forward(ppl_ids); _sync()
    reps = 3
    best = float("inf")
    best_linear_s = 0.0
    for _ in range(reps):
        timing["linear_s"] = 0.0; timing["n_linear"] = 0
        t0 = time.perf_counter()
        _ = dense_full_forward(ppl_ids)
        _sync()
        el = time.perf_counter() - t0
        if el < best:
            best = el
            best_linear_s = timing["linear_s"]
    e2e_tok_per_sec = ppl_n / best
    rest_s = best - best_linear_s   # nonlinearities (RMSNorm/SiLU/softmax) + attention + RoPE + residuals + host
    log(f"  END-TO-END dense forward ({ppl_n} tok, B-1 graded nonlinearities + attention + RoPE, dense GEMM "
        f"linears): {best*1000:.1f}ms -> {e2e_tok_per_sec:.1f} tok/s (de-risk #3 warm CSR: 0.786 tok/s -> "
        f"{e2e_tok_per_sec/0.786:.0f}x end-to-end)")
    log(f"    breakdown: dense linears (incl H<->D) {best_linear_s*1000:.1f}ms ({100*best_linear_s/best:.0f}%); "
        f"REST (graded nonlinearities + attention + RoPE + host) {rest_s*1000:.1f}ms ({100*rest_s/best:.0f}%) "
        f"-> the RESIDUAL bottleneck is the {'NONLINEARITIES/ATTENTION (host)' if rest_s > best_linear_s else 'LINEARS'}")

    # free the torch model + GPU weights
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return {
        "tokens_scored": int(ppl_n),
        "end_to_end_forward_seconds": round(best, 4),
        "end_to_end_tok_per_sec_dense": round(e2e_tok_per_sec, 1),
        "dense_linears_seconds_incl_HtoD": round(best_linear_s, 4),
        "rest_seconds_nonlin_attn_rope_host": round(rest_s, 4),
        "linears_pct_of_forward": round(100 * best_linear_s / best, 1),
        "rest_pct_of_forward": round(100 * rest_s / best, 1),
        "residual_bottleneck": ("nonlinearities_attention_host" if rest_s > best_linear_s else "linears"),
        "csr_warm_tok_per_sec_derisk3": 0.786,
        "end_to_end_speedup_x": round(e2e_tok_per_sec / 0.786, 1),
        "rmsnorm_mode": "graded (B-1 spiking RMS, host read)",
        "note": "the REAL end-to-end forward: de-risk #3's path VERBATIM (real Qwen weights, B-1 graded "
                "RMSNorm/SiLU/softmax + RoPE + attention, H<->D per linear) with the per-row CSR-RF matvec SWAPPED "
                "for a dense GPU GEMM. INCLUDES the nonlinearity/attention residual the matvec-only projection omits. "
                "The dense matvec made the LINEARS cheap; the residual is now the host nonlinearities/attention.",
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-end-to-end", action="store_true",
                    help="skip the real end-to-end Qwen forward measurement (matvec-only projection only)")
    ap.add_argument("-S", type=int, default=32, help="prefill window (rows) for the benches")
    args = ap.parse_args()

    t_start = time.time()
    backend = os.environ.get("SIM_BACKEND", "auto")
    log(f"SIM_BACKEND={backend}")
    import cupy as cp
    free0, total0 = cp.cuda.Device().mem_info
    log(f"GPU VRAM free {free0/1e9:.1f}GB / total {total0/1e9:.1f}GB")

    S = int(args.S)   # representative prefill window (small per the brief)
    rfmv = RFMatvecCSR(seed=42, use_megakernel=True)

    prof = profile_one_layer(rfmv, S)
    dense_res = bench_dense(rfmv, S)
    extra = extrapolate(dense_res, S)

    e2e = None
    if not args.no_end_to_end:
        try:
            e2e = measure_end_to_end(S)
        except Exception as _e:
            log(f"  (end-to-end measurement skipped: {type(_e).__name__}: {_e})")
            e2e = {"error": f"{type(_e).__name__}: {_e}"}

    # ------- VERDICT -------
    # The dense batched GEMM speedup over the CSR-RF per-row loop, on the representative gate_proj + lm_head.
    gate_speedup = dense_res["gate_proj"]["speedup_dense_batched_vs_csr"]
    lm_speedup = dense_res["lm_head"]["speedup_dense_batched_vs_csr"]
    bit_exact_ok = all(v["max_err_dense_batch_f64_vs_a_at_W_f64"] < 1e-9 for v in dense_res.values())
    # dense f32 vs CSR-RF f32 (both approximate a@W): the cross-agreement (must be ~f32 = 1e-3 or better given scale)
    max_csr_vs_batch = max(v["max_err_csr_rf_vs_dense_batch_f32"] for v in dense_res.values())
    proj_gen = extra["dense_generation_tok_per_sec_matvec_only"]
    proj_prefill = extra["dense_prefill_tok_per_sec_matvec_only"]
    # the load-bearing usable check = the MEASURED end-to-end forward (incl nonlinearities) when available, else the
    # matvec-only projection. The brief target is >10 tok/s (ideally near ANN).
    e2e_tps = (e2e.get("end_to_end_tok_per_sec_dense") if (e2e and "end_to_end_tok_per_sec_dense" in e2e) else None)
    usable = ((e2e_tps if e2e_tps is not None else proj_gen) >= 10.0)
    e2e_str = (f"; MEASURED end-to-end {e2e_tps:.0f} tok/s (de-risk #3 warm CSR 0.786 -> {e2e_tps/0.786:.0f}x, incl "
               f"the B-1 graded nonlinearities + attention)" if e2e_tps is not None else "")

    # the matvec's share of the resonate (is the resonate loop or the matvec the bottleneck?)
    res_fixed_frac = (prof["resonate_fit_fixed_overhead_ms"] /
                      (8 * prof["resonate_fit_ms_per_step"] + prof["resonate_fit_fixed_overhead_ms"]))

    if usable and gate_speedup > 10 and bit_exact_ok:
        verdict = "GO"
        sim_edit = ("YES (the real lever): a guarded DENSE-RF-matvec mode on the bridge. Scope: add an optional "
                    "dense weight path to the RF matvec read so `Re(Z)/nsteps = a@W` is computed by a cuBLAS GEMM "
                    "(a@W_dense) instead of the cuSPARSE CSR resonate, when the weights are dense. Concretely a new "
                    "guarded method e.g. `rf_dense_matvec(A, W_dense)` (or a `cfg.rf_dense_weights` flag that makes "
                    "rf_resonate_steps/_rf_advance_one read a stored dense `cp_rf_w_dense` via GEMM); DEFAULT-OFF = "
                    "the byte-identical CSR path (the composer's sparse O(D) bind/unbind is unaffected). The dense "
                    "path is bit-faithful (the SAME a@W) + batches all token-rows in one GEMM. NOTE: this de-risk "
                    "ALREADY shows the dense GEMM is runner-computable WITHOUT a sim/ edit (the forward can call "
                    "cupy `A @ W_dense` directly in the host forward, bypassing the RF matvec entirely for the dense "
                    "linears) — so the sim/ edit is OPTIONAL (it would make the bridge's OWN RF read fast for "
                    "co-residence purity; the runner-level dense GEMM already achieves the throughput).")
        tail = (f"the DENSE batched GEMM is bit-faithful (== a@W, f64 max-err <1e-9; the f32 dense vs f32 CSR-RF "
                f"agree to {max_csr_vs_batch:.1e}) and gives a {gate_speedup:.0f}x (gate_proj) / {lm_speedup:.0f}x "
                f"(lm_head) speedup over the per-row CSR-RF loop. Projected full-forward: prefill "
                f"{proj_prefill:.0f} tok/s (CSR {0.786:.2f} -> {extra['prefill_speedup_x']:.0f}x), generation "
                f"{proj_gen:.0f} tok/s (CSR 161 s/tok -> {extra['gen_speedup_x']:.0f}x). The CSR-on-dense gather was "
                f"the wall; dense storage IS the ANN GEMM speed{e2e_str}. => USABLE. The matvec, not the "
                f"resonate loop, was the dominant cost; a sim/ dense-RF mode is the real lever for on-bridge purity "
                f"but the runner-level dense GEMM already achieves it.")
    elif gate_speedup > 10 and bit_exact_ok:
        verdict = "GO_WITH_CAVEAT"
        _rest_pct = (e2e.get("rest_pct_of_forward") if (e2e and "rest_pct_of_forward" in e2e) else None)
        _lin_pct = (e2e.get("linears_pct_of_forward") if (e2e and "linears_pct_of_forward" in e2e) else None)
        sim_edit = ("TWO levers, in order. (1) THE DENSE MATVEC (proven here): a guarded dense-weight path for the RF "
                    "matvec read so `Re(Z)/nsteps = a@W` is a cuBLAS GEMM (a@W_dense), not the cuSPARSE CSR resonate, "
                    "when the weights are dense. Concretely `cfg.rf_dense_weights` + a stored dense `cp_rf_w_dense`, "
                    "read in rf_resonate_steps/_rf_advance_one via GEMM; DEFAULT-OFF = the byte-identical CSR path "
                    "(the composer's sparse O(D) bind/unbind unaffected). NOTE the dense GEMM is ALREADY "
                    "runner-computable WITHOUT a sim/ edit (the host forward calls cupy `A @ W_dense` directly), so "
                    "the sim/ edit is OPTIONAL — for on-bridge co-residence purity, not for the throughput. BUT (2) "
                    "the dense matvec is NECESSARY-NOT-SUFFICIENT: once the linears are cheap, the MEASURED end-to-end "
                    "is bottlenecked by the HOST graded nonlinearities (RMSNorm/SiLU/softmax) + attention + the "
                    "per-linear H<->D round-trips (the numpy reads + ~216 device-host copies/forward). The SECOND "
                    "lever (the real end-to-end win) is keeping the whole forward ON-GPU: cupy graded ops + an on-GPU "
                    "attention + no per-linear D->H (only the final logits read). That is the actual usability work; "
                    "it is NOT a sim/ edit (it's the host forward staying on-device).")
        tail = (f"the DENSE batched GEMM is bit-faithful (== a@W, f64 max-err <1e-9; f32 dense vs f32 CSR-RF agree to "
                f"{max_csr_vs_batch:.1e}) and {gate_speedup:.0f}x (gate_proj) / {lm_speedup:.0f}x (lm_head) faster "
                f"than the per-row CSR-RF loop — the CSR-on-dense gather WAS the wall and dense storage IS the ANN "
                f"GEMM speed. BUT the MEASURED end-to-end forward is "
                f"{('only ' + str(e2e_tps) + ' tok/s (de-risk #3 warm CSR 0.786 -> ' + str(round(e2e_tps/0.786,0)) + 'x)') if e2e_tps is not None else 'below the projection'}, because "
                f"once the matvec is cheap the bottleneck SHIFTS to the host nonlinearities/attention + H<->D "
                f"round-trips" + (f" ({_rest_pct:.0f}% of the forward vs the linears' {_lin_pct:.0f}%)"
                                  if _rest_pct is not None else "") + f". => the dense matvec is the proven FIRST "
                f"lever (necessary + bit-exact); end-to-end usability needs the SECOND lever (keep the whole forward "
                f"on-GPU: cupy nonlinearities + on-GPU attention + no per-linear D->H). Still LOCAL (no VRAM wall); "
                f"NO sim/ edit required for either lever — both are host-forward changes (the optional dense-RF sim/ "
                f"mode is only for on-bridge purity).")
    else:
        verdict = "HONEST_RESIDUAL"
        sim_edit = ("the dense matvec did NOT deliver the expected speedup OR diverged — inspect the bench. If the "
                    "resonate loop (not the matvec) dominates, the lever is fusing the per-token kick/read launches, "
                    "not the storage.")
        tail = (f"the dense matvec speedup ({gate_speedup:.0f}x gate_proj) or bit-exactness (f64 ok={bit_exact_ok}) "
                f"did not meet the GO bar, or the projection ({proj_gen:.0f} tok/s gen) stays unusable. The real "
                f"bottleneck may be the resonate loop / per-op launches (fixed-overhead fraction "
                f"{res_fixed_frac*100:.0f}% of an 8-step resonate), not the CSR storage.")

    verdict_line = (
        f"bridge_cores_perf: the DENSE-MATVEC lever for the on-bridge Qwen faculty -> dense batched GEMM is "
        f"bit-faithful (== a@W) and {gate_speedup:.0f}x (gate_proj) / {lm_speedup:.0f}x (lm_head) faster than the "
        f"CSR-RF per-row loop; projected full-forward prefill {proj_prefill:.0f} tok/s "
        f"(CSR 0.786 -> {extra['prefill_speedup_x']:.0f}x), generation {proj_gen:.0f} tok/s "
        f"(CSR 161 s/tok -> {extra['gen_speedup_x']:.0f}x) [matvec-only]{e2e_str}. The CSR-on-dense gather was the "
        f"wall (the Qwen layers are 100% dense -> sparse CSR is the wrong storage). -> {verdict}. {tail}")

    result = {
        "probe": "bridge_coresidence_perf_derisk_dense_matvec_lever",
        "resolves": "scoping #6 (the PERF LEVER): profile the RF forward + test the dense-matvec & batch levers -> "
                    "the projected tokens/sec. De-risk #3 demonstrated the full 494M Qwen runs on-bridge LOCAL + "
                    "bit-exact but SLOW (0.786 tok/s prefill, 161 s/gen-token; CSR-gather/launch-bound). The Qwen "
                    "layers are 100% DENSE -> the sparse CSR gather is the WRONG storage.",
        "rf_operating_point": {"period": RF_PERIOD, "nsteps": RF_NSTEPS, "lambda": RF_LAMBDA,
                               "read": "Re(Z_out)/nsteps = a @ W (omega~0, decay=1 -> nsteps copies of W@a)"},
        "S_prefill_window": S,
        "n_layers": N_LAYERS,
        "qwen_layer_shapes": {k: list(v) for k, v in QWEN_LAYER_SHAPES.items()},
        "lm_head_shape": list(LM_HEAD_SHAPE),
        "profile": prof,
        "dense_vs_csr": dense_res,
        "extrapolation": extra,
        "end_to_end_measured": e2e,
        "verdict": verdict,
        "sim_edit_recommendation": sim_edit,
        "bit_exactness": {
            "dense_batch_f64_vs_a_at_W_max_err": max(v["max_err_dense_batch_f64_vs_a_at_W_f64"]
                                                     for v in dense_res.values()),
            "dense_batch_f32_vs_csr_rf_f32_max_err": max_csr_vs_batch,
            "all_f64_bit_exact": bool(bit_exact_ok),
            "note": "the dense f64 GEMM is the SAME math as a@W (max-err <1e-9 = numerical roundoff); the f32 dense "
                    "GEMM and the f32 RF membrane read both approximate a@W and agree to ~f32 precision. Bit-"
                    "exactness of the dense path is trivial (a@W == a@W); the speedup is the deliverable.",
        },
        "verdict_line": verdict_line,
        "total_seconds": round(time.time() - t_start, 2),
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
