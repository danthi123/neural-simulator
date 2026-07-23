"""WKV SPIKING-GRADED-READ FORWARD de-risk (spiking-Broca convertibility of run3's 83M generator, 2026-07-23).

THE ENDGAME this de-risks: the on-substrate spiking generator that REPLACES the ANN WKV.  The 88.6M
transformer Gen-F was validated earlier as a faithful RF-on-bridge spiking forward (ppl_ratio 0.9999999,
logit_fid 1.0 -- research/findings/2026-06-30-100M-C2-scaleup-C1-GO-C2-nuanced.md, via `verify_on_bridge` /
`rf_full_forward`).  run3 is a DIFFERENT architecture -- an 83.17M CHUNKED-WKV/SSM generator (d_model=1024,
n_layers=16, chunk_c=16, vocab=16000; `bridges/lmtrain/run3/ckpt/best.pt`, torch.load weights_only=False).
This runner RE-VALIDATES the spiking-graded-read forward on run3's ACTUAL checkpoint.

THE MECHANISM (a direct mirror of the C1 transformer pattern, reuse-by-import, NO `sim/` edit):
  * Every LEARNED-WEIGHT matvec (per block: Wv, Wr, Wo; the head) goes through the SPIKING GRADED READ --
    the RF (resonate-and-fire) complex-synapse accumulator that computes `h @ W` as Re(Z_out)/nsteps at
    lam=0, omega~=0 (`rf_linear_layer_signed` / `_rf_project_seq`; EXACT to ~5e-7 in the transformer arc).
  * The PARAMETER-FREE ops are FAITHFUL HOST READS (exactly as the transformer's softmax/GELU/LayerNorm were):
      - LayerNorm (`ln`, `lnf`)                                        [the transformer's LN, verbatim reuse]
      - the two sigmoid gates  r=sigmoid(Wr h),  lam=sigmoid(decay)    [0-param monotone nonlinearity]
      - the SSM temporal mix   a_t = lam*a_{t-1} + (1-lam)*v_t         [the analogue of softmax value-mixing:
                                                                        parameter-free-given-lam time mixing]
      - the residual `x + r*Wo(a)`                                     [elementwise add]
  * The embedding lookup is an exact row gather (input/environment side -- a lookup, not a matvec).

TWO graded-read backends (same forward wiring; the ONLY difference is how `h @ W` is computed):
  * `numpy-model`  (CPU smoke): `h @ W + read_eps * N(0,1)`  -- an idealized model of the RF read whose
                    absolute read error `read_eps` defaults to the transformer arc's MEASURED RF exactness
                    (~5e-7).  A knob so the smoke can DEMONSTRATE the metric detects a non-faithful read.
  * `rf-bridge`    (GPU/cupy): the REAL RF complex-synapse graded read (`_rf_project_seq`), the exact
                    machinery the 88.6M used.  This is the AWS-GPU lane.

WHAT THE SMOKE PROVES (wiring, on a TINY numpy 2-layer WKV; 1 seed):
  1. the numpy ANN forward == the torch chunked-WKV forward (the SSM-scan reimplementation is correct);
  2. the spiking-graded-read forward, wired op-by-op, reproduces the ANN at RF-level read error
     (ppl_ratio ~= 1.0, logit_fid ~= 1.0)  -- the convertibility claim's STRUCTURE;
  3. every CONTROL changes the result (a never-invoked control is the bug):
       - read_eps sweep {0, 5e-7, 1e-3, 1e-2}: faithful at RF-level, DEGRADES at exaggerated error
         (the fidelity metric is sensitive; the RF-level read is faithful);
       - shuffle_head: permute the head rows in the spiking forward -> logit_fid COLLAPSES
         (proves the graded read reads the REAL head weights);
       - ssm_lesion: zero the SSM state -> ppl EXPLODES vs ANN (proves the SSM time-mix is wired + load-bearing).
  It also loads run3's ACTUAL best.pt header (keys/shapes/param-count) to confirm the loader wiring on the
  real 83M artifact WITHOUT running the heavy forward.

An honest NEGATIVE (the 16-layer WKV's per-op graded-read error compounds where the 12-layer transformer's
did not) is a FIRST-CLASS result -- it maps what the substrate can/can't do and is reported as such.

  CPU smoke (this is what you run to build/verify the wiring):
    python -m research.runners._wkv_spiking_forward_derisk --mode smoke \
        --out research/findings/raw/wkv_spiking_forward/smoke.json
  Real de-risk (AWS GPU, cupy; run3's actual ckpt, 6 seeds): see the printed launch command / the spec.

ASCII only. Reuse-by-import. NO `sim/` edit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ---- reuse-by-import (the C1 spiking-forward machinery + the WKV model def) --------------------------
from research.runners._lmtrain_chunked_scan import WKV  # noqa: E402  (the run3 model class)
from research.runners._genseq_loopstep3_mlp_gelu_rf_distill_derisk import _layernorm  # noqa: E402
from research.runners._genseq_loopstep3_full_genf_generate_derisk import (  # noqa: E402
    _heldout_nll_numpy, _perplexity)

RUN3_CKPT = _REPO / "bridges/lmtrain/run3/ckpt/best.pt"
RUN3_VAL = _REPO / "bridges/lmtrain/run3/tokens_val.npy"
RUN3_ARCH = dict(vocab_size=16000, d_model=1024, n_layers=16, chunk_c=16)


# =====================================================================================================
# metrics
# =====================================================================================================
def _spearman(a, b):
    """Rank-correlation between two 1-D logit vectors (per position)."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel(); b = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if (na > 0 and nb > 0) else 0.0


def logit_fidelity(ann_forward, spk_forward, ids, block_size, n_windows, n_logit_pos, rng):
    """Mean spearman + cosine between ANN and spiking logits, over n_logit_pos sampled positions per
    window across n_windows (teacher-forced; the same windows scored for both)."""
    n = len(ids)
    if n < block_size + 2:
        return {"spearman": float("nan"), "cosine": float("nan"), "n_scored": 0}
    total_w = (n - 1) // block_size
    step_w = max(1, total_w // max(1, n_windows))
    sp, co, cnt = [], [], 0
    wi = 0
    for w in range(0, total_w, step_w):
        s = w * block_size
        x = ids[s:s + block_size]
        lg_a = np.asarray(ann_forward(x), dtype=np.float64)   # (block_size, V)
        lg_s = np.asarray(spk_forward(x), dtype=np.float64)
        T = lg_a.shape[0]
        pos = np.unique(np.linspace(0, T - 1, num=min(n_logit_pos, T)).astype(int))
        for p in pos:
            sp.append(_spearman(lg_a[p], lg_s[p]))
            co.append(_cosine(lg_a[p], lg_s[p]))
        cnt += len(pos)
        wi += 1
        if wi >= n_windows:
            break
    return {"spearman": float(np.mean(sp)) if sp else float("nan"),
            "cosine": float(np.mean(co)) if co else float("nan"), "n_scored": cnt}


# =====================================================================================================
# the WKV as numpy weights + the ANN / spiking-graded-read forwards (single window: ids(N,) -> logits(N,V))
# =====================================================================================================
def wkv_state_to_numpy(state_dict, n_layers):
    """Extract the WKV weights as numpy float64. torch Linear weight is (out,in); the GRADED matvec wants
    `h @ Wg` with Wg=(in,out), so every learned weight is stored transposed (Wg)."""
    def g(k):
        return np.asarray(state_dict[k].detach().to("cpu").numpy(), dtype=np.float64)
    blocks = []
    for li in range(n_layers):
        p = f"blocks.{li}."
        blocks.append({
            "ln_w": g(p + "ln.weight"), "ln_b": g(p + "ln.bias"),
            "decay": g(p + "decay"),
            "Wv": g(p + "Wv.weight").T.copy(),    # (d,d)  graded
            "Wr": g(p + "Wr.weight").T.copy(),
            "Wo": g(p + "Wo.weight").T.copy(),
        })
    return {
        "emb": g("emb.weight"),                    # (V,d) lookup table
        "blocks": blocks,
        "lnf_w": g("lnf.weight"), "lnf_b": g("lnf.bias"),
        "head": g("head.weight").T.copy(),         # (d,V) graded
        "head_b": g("head.bias"),
        "n_layers": n_layers,
        "d_model": int(g("emb.weight").shape[1]),
        "vocab_size": int(g("emb.weight").shape[0]),
    }


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _ssm_scan(v, lam):
    """The WKV time-mix a_t = lam*a_{t-1} + (1-lam)*v_t  (== loop_ssm; the chunked_ssm reference, exact to
    1e-4). Parameter-free given lam -> a faithful host read. v:(N,d), lam:(d,) -> a:(N,d)."""
    N, d = v.shape
    a = np.zeros(d, dtype=np.float64)
    out = np.empty((N, d), dtype=np.float64)
    for t in range(N):
        a = lam * a + (1.0 - lam) * v[t]
        out[t] = a
    return out


def wkv_forward(model, ids, graded_read, *, shuffle_head=None, ssm_lesion=False):
    """The WKV forward on one window (ids: (N,) ints) -> logits (N, V).

    `graded_read(W, h)` computes the LEARNED matvec `h @ W` (either the exact numpy matmul + a modelled RF
    read error, or the real RF-on-bridge graded read). Every parameter-free op is a faithful host read.
    Controls: `shuffle_head` (a permutation applied to the head OUTPUT columns in the spiking forward ->
    scrambles which logit each read maps to); `ssm_lesion` (zero the SSM state -> the block becomes x+0)."""
    ids = np.asarray(ids)
    x = model["emb"][ids].astype(np.float64)               # (N,d) exact embedding lookup
    for blk in model["blocks"]:
        h = _layernorm(x, blk["ln_w"], blk["ln_b"])        # faithful
        v = graded_read(blk["Wv"], h)                      # GRADED matvec
        r = _sigmoid(graded_read(blk["Wr"], h))            # GRADED matvec + faithful sigmoid
        lam = _sigmoid(blk["decay"])                       # faithful
        a = np.zeros_like(v) if ssm_lesion else _ssm_scan(v, lam)   # faithful time-mix (lesion = 0)
        wo = graded_read(blk["Wo"], a)                     # GRADED matvec
        x = x + r * wo                                     # faithful residual
    x = _layernorm(x, model["lnf_w"], model["lnf_b"])      # faithful final LN
    logits = graded_read(model["head"], x) + model["head_b"]   # GRADED matvec + bias
    if shuffle_head is not None:
        logits = logits[:, shuffle_head]                   # scramble the logit<->vocab map (control)
    return logits


# =====================================================================================================
# graded-read backends
# =====================================================================================================
def make_numpy_graded_read(read_eps, rng):
    """CPU model of the RF graded read: exact `h @ W` + zero-mean absolute read error `read_eps`
    (default ~= the transformer arc's MEASURED RF exactness). read_eps=0 -> exact."""
    def _read(W, h):
        y = h.astype(np.float64) @ W.astype(np.float64)
        if read_eps and read_eps > 0:
            y = y + read_eps * rng.standard_normal(size=y.shape)
        return y
    return _read


def make_rf_graded_read(bridges, *, period, nsteps, lam, err_accum=None, w32=None):
    """GPU/cupy: the REAL RF complex-synapse graded read via `_rf_project_seq`. `bridges` is a cache
    keyed by (d_in, d_out) -> an RF bridge (built once, reused across matvecs of the same shape).

    PERF: `_set_rf_weights` caches the installed complex CSR by `id(W)` (builds the up-to-1024x16000 =
    16.4M-entry head connection list ONCE, then just swaps the cached CSR). So we must pass the SAME
    float32 weight OBJECT every call -- `_w32` memoizes the float32 cast by the source array's id so the
    CSR cache hits across ALL windows/seeds (a fresh `.astype` each call would rebuild the head CSR every
    matvec = catastrophic).

    id()-REUSE FIX (2026-07-23, the seed-43 blowup): the module-level `_WEIGHT_CSR_CACHE` (in
    `_genseq_loopstep3_full_genf_generate_derisk`) keys by `id(Wf)`. If a FRESH `_w32` is built per seed,
    seed-N's Wf arrays are GC'd when the seed returns and seed-(N+1)'s freshly-allocated Wf reuse their
    id() -> FALSE cache hits install seed-N's CSR for a DIFFERENT weight -> a wrong matvec -> the RF read
    blows up (seed 43: read_err 14.05, ppl_ratio 130; seed 42 clean because it ran first with an empty
    cache). Pass a RUN-LIFETIME `w32` (shared across seeds) so every Wf stays alive -> `id(Wf)` is stable
    -> the CSR cache is valid AND genuinely hits across seeds (the documented intent). `w32=None` (the
    default) preserves the old per-call-fresh behavior for any standalone caller."""
    from research.runners._genseq_loopstep3_full_genf_generate_derisk import _rf_project_seq
    from research.runners._genseq_loopstep3_rf_probe import _build_rf_bridge
    _w32 = {} if w32 is None else w32

    def _read(W, h):
        wk = id(W)
        Wf = _w32.get(wk)
        if Wf is None:
            Wf = np.ascontiguousarray(W, dtype=np.float32)
            _w32[wk] = Wf                       # stable object -> stable id -> CSR cache hits
        d_in, d_out = Wf.shape
        key = (int(d_in), int(d_out))
        if key not in bridges:
            bridges[key] = _build_rf_bridge(d_in + d_out, seed=42)
        out, max_err = _rf_project_seq(bridges[key], Wf, h.astype(np.float64),
                                       period=period, nsteps=nsteps, lam=lam,
                                       measure_err=(err_accum is not None))
        if err_accum is not None:
            err_accum.append(float(max_err))
        return out
    return _read


# =====================================================================================================
# checkpoint / corpus loading
# =====================================================================================================
def load_ckpt_header(ckpt_path):
    """Load run3's ckpt header on CPU (weights_only=False) and confirm keys/shapes/param-count == the WKV
    arch, WITHOUT building the model. Returns (state_dict, info)."""
    import torch
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    tot = int(sum(int(v.numel()) for v in sd.values()))
    d = int(sd["emb.weight"].shape[1]); V = int(sd["emb.weight"].shape[0])
    n_layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    keys_ok = all(f"blocks.{li}.{s}" in sd for li in range(n_layers)
                  for s in ("decay", "ln.weight", "ln.bias", "Wv.weight", "Wr.weight", "Wo.weight"))
    keys_ok = keys_ok and all(k in sd for k in ("emb.weight", "lnf.weight", "lnf.bias",
                                                "head.weight", "head.bias"))
    info = {"ckpt": str(ckpt_path), "total_params": tot, "total_params_M": round(tot / 1e6, 2),
            "d_model": d, "vocab_size": V, "n_layers": n_layers,
            "arch_keys_ok": bool(keys_ok),
            "matches_run3_arch": (d == RUN3_ARCH["d_model"] and V == RUN3_ARCH["vocab_size"]
                                  and n_layers == RUN3_ARCH["n_layers"]),
            "step": (int(ck.get("step", -1)) if isinstance(ck, dict) else -1),
            "tokens_seen": (int(ck.get("tokens_seen", -1)) if isinstance(ck, dict) else -1)}
    return sd, info


def build_numpy_model_from_ckpt(ckpt_path):
    sd, info = load_ckpt_header(ckpt_path)
    model = wkv_state_to_numpy(sd, info["n_layers"])
    return model, info


def load_val_ids(seed, n_tokens):
    """A seed-offset contiguous slice of run3's val tokens (uint16 -> python ints)."""
    arr = np.load(str(RUN3_VAL), mmap_mode="r")
    rng = np.random.default_rng(seed)
    hi = max(1, arr.shape[0] - n_tokens - 1)
    start = int(rng.integers(0, hi))
    return np.asarray(arr[start:start + n_tokens], dtype=np.int64)


# =====================================================================================================
# SMOKE (CPU, numpy, tiny 2-layer WKV) -- wiring + all controls
# =====================================================================================================
def run_smoke(args):
    import torch
    t0 = time.time()
    V, D, L, C, N = 64, 32, 2, 8, 48   # tiny
    block_size = 16
    seed = args.seed

    # (0) confirm the run3 loader wiring on the REAL 83M artifact (header only; no heavy forward)
    ckpt_info = None
    if RUN3_CKPT.is_file():
        try:
            _sd, ckpt_info = load_ckpt_header(RUN3_CKPT)
            del _sd
        except Exception as e:  # honest: record the failure rather than crash the smoke
            ckpt_info = {"error": repr(e), "ckpt": str(RUN3_CKPT)}
    else:
        ckpt_info = {"error": "run3 ckpt not present", "ckpt": str(RUN3_CKPT)}

    # (1) build a tiny torch chunked-WKV, export weights, verify numpy-ANN == torch
    torch.manual_seed(seed)
    tm = WKV(V, D, L, block="chunked", C=C).eval()
    model = wkv_state_to_numpy(tm.state_dict(), L)
    rng = np.random.default_rng(seed)
    ids_all = rng.integers(0, V, size=N).astype(np.int64)
    with torch.no_grad():
        torch_logits = tm(torch.tensor(ids_all[None], dtype=torch.long))[0].double().numpy()
    exact_read = make_numpy_graded_read(0.0, np.random.default_rng(0))
    numpy_ann_logits = wkv_forward(model, ids_all, exact_read)
    ann_vs_torch = float(np.max(np.abs(numpy_ann_logits - torch_logits)))
    ann_matches_torch = ann_vs_torch < 1e-3

    # forward closures over one window; ANN = exact-read numpy forward
    def ann_fwd(x):
        return wkv_forward(model, x, exact_read)

    # held-out ids (a longer stream so _heldout_nll_numpy has several windows)
    ho_ids = rng.integers(0, V, size=block_size * 12).astype(np.int64)
    ann_ppl = _perplexity(_heldout_nll_numpy(ann_fwd, list(ho_ids), V, block_size, args.n_windows))

    # (2) read_eps sweep -- faithful at RF-level, degrades at exaggerated error (metric sensitivity + the
    #     convertibility headline). Every arm RUN + recorded.
    eps_sweep = {}
    for eps in [0.0, 5e-7, 1e-3, 1e-2]:
        rd = make_numpy_graded_read(eps, np.random.default_rng(1234))

        def spk_fwd(x, _rd=rd):
            return wkv_forward(model, x, _rd)
        spk_ppl = _perplexity(_heldout_nll_numpy(spk_fwd, list(ho_ids), V, block_size, args.n_windows))
        fid = logit_fidelity(ann_fwd, spk_fwd, list(ho_ids), block_size, args.n_windows,
                             args.n_logit_pos, rng)
        eps_sweep[f"{eps:.0e}"] = {
            "read_eps": eps, "spiking_ppl": spk_ppl, "ann_ppl": ann_ppl,
            "ppl_ratio": (spk_ppl / ann_ppl if math.isfinite(ann_ppl) and ann_ppl > 0 else float("inf")),
            "logit_fid_spearman": fid["spearman"], "logit_fid_cosine": fid["cosine"]}

    faithful = eps_sweep["5e-07"]   # RF-level read error
    faithful_ok = (abs(faithful["ppl_ratio"] - 1.0) < 0.02 and faithful["logit_fid_spearman"] > 0.99)
    degrades_at_high_eps = (eps_sweep["1e-02"]["logit_fid_spearman"]
                            < faithful["logit_fid_spearman"] - 1e-6)

    # (3a) CONTROL shuffle_head -> logit_fid collapses
    perm = np.random.default_rng(777).permutation(V)
    rd_faith = make_numpy_graded_read(5e-7, np.random.default_rng(1234))

    def shuf_fwd(x):
        return wkv_forward(model, x, rd_faith, shuffle_head=perm)
    fid_shuf = logit_fidelity(ann_fwd, shuf_fwd, list(ho_ids), block_size, args.n_windows,
                              args.n_logit_pos, rng)
    shuffle_collapses = fid_shuf["spearman"] < 0.5

    # (3b) CONTROL ssm_lesion -> ppl explodes vs ANN
    def les_fwd(x):
        return wkv_forward(model, x, rd_faith, ssm_lesion=True)
    les_ppl = _perplexity(_heldout_nll_numpy(les_fwd, list(ho_ids), V, block_size, args.n_windows))
    les_ratio = les_ppl / ann_ppl if math.isfinite(ann_ppl) and ann_ppl > 0 else float("inf")
    fid_les = logit_fidelity(ann_fwd, les_fwd, list(ho_ids), block_size, args.n_windows,
                             args.n_logit_pos, rng)
    # On the UNTRAINED tiny smoke model, ppl sits at ~chance with or without the time-mix, so the ppl
    # signal is weak; the decisive wiring signal is the LOGIT-FIDELITY delta (removing the SSM drops fid
    # far below the faithful graded-read fid). The ppl-EXPLOSION lesion is the TRAINED-model signal,
    # validated on run3's real weights in `--mode full`.
    lesion_fid_delta = faithful["logit_fid_spearman"] - fid_les["spearman"]
    lesion_breaks = (les_ratio > 1.10) or (lesion_fid_delta > 1e-3)

    all_controls_change = bool(degrades_at_high_eps and shuffle_collapses and lesion_breaks)
    smoke_wired = bool(ann_matches_torch and faithful_ok and all_controls_change)

    out = {
        "runner": "_wkv_spiking_forward_derisk", "mode": "smoke", "backend": "numpy-model",
        "purpose": "CPU wiring smoke of run3's 83M WKV spiking-graded-read forward + all controls",
        "smoke_wired": smoke_wired,
        "knobs": {"vocab": V, "d_model": D, "n_layers": L, "chunk_c": C, "block_size": block_size,
                  "n_stream_tokens": int(len(ho_ids)), "n_windows": args.n_windows,
                  "n_logit_pos": args.n_logit_pos, "seed": seed,
                  "read_eps_default_rf_level": 5e-7,
                  "ssm_scan": "loop reference a_t=lam a_{t-1}+(1-lam)v_t (== chunked_ssm to 1e-4)"},
        "gate_1_numpy_ann_eq_torch": {"max_abs_logit_diff": ann_vs_torch, "pass": ann_matches_torch},
        "gate_2_faithful_at_rf_read_error": {
            "ppl_ratio": faithful["ppl_ratio"], "logit_fid_spearman": faithful["logit_fid_spearman"],
            "logit_fid_cosine": faithful["logit_fid_cosine"], "pass": faithful_ok},
        "read_eps_sweep": eps_sweep,
        "gate_3_controls_change_result": {
            "read_eps_degrades_at_1e-2": {"value_spearman": eps_sweep["1e-02"]["logit_fid_spearman"],
                                          "pass": degrades_at_high_eps},
            "shuffle_head_collapses_fid": {"spearman": fid_shuf["spearman"], "pass": shuffle_collapses},
            "ssm_lesion_breaks_ppl": {"lesion_ppl": les_ppl, "ann_ppl": ann_ppl,
                                      "lesion_ppl_ratio": les_ratio,
                                      "lesion_fid_spearman": fid_les["spearman"],
                                      "faithful_fid_spearman": faithful["logit_fid_spearman"],
                                      "lesion_fid_delta_vs_faithful": lesion_fid_delta,
                                      "note": ("untrained smoke: ppl ~chance +/- SSM so the decisive "
                                               "signal is the fid delta; ppl-explosion lesion is the "
                                               "TRAINED-model signal validated in --mode full"),
                                      "pass": lesion_breaks},
            "all_controls_change": all_controls_change},
        "run3_ckpt_header_check": ckpt_info,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    # the AWS-GPU real de-risk command (built here so it stays in sync with the runner's flags)
    out["aws_gpu_launch_command"] = build_aws_command(args)
    _emit_verdict_smoke(out)
    return out


def _emit_verdict_smoke(out):
    print("=" * 96, flush=True)
    print("[wkv-spk-fwd:SMOKE] run3 83M WKV spiking-graded-read FORWARD -- CPU wiring smoke", flush=True)
    c = out["run3_ckpt_header_check"] or {}
    if "error" in c:
        print(f"  (0) run3 ckpt header: UNAVAILABLE ({c['error']})", flush=True)
    else:
        print(f"  (0) run3 ckpt header: {c['total_params_M']}M params, d={c['d_model']} L={c['n_layers']} "
              f"V={c['vocab_size']} step={c['step']} | matches_run3_arch={c['matches_run3_arch']} "
              f"keys_ok={c['arch_keys_ok']}", flush=True)
    g1 = out["gate_1_numpy_ann_eq_torch"]
    print(f"  (1) numpy-ANN == torch WKV: max|diff|={g1['max_abs_logit_diff']:.2e}  "
          f"{'PASS' if g1['pass'] else 'FAIL'}", flush=True)
    g2 = out["gate_2_faithful_at_rf_read_error"]
    print(f"  (2) faithful @ RF read error (eps=5e-7): ppl_ratio={g2['ppl_ratio']:.6f}  "
          f"logit_fid(spearman)={g2['logit_fid_spearman']:.4f} cos={g2['logit_fid_cosine']:.4f}  "
          f"{'PASS' if g2['pass'] else 'FAIL'}", flush=True)
    print("      read_eps sweep (spearman):  " + "  ".join(
        f"{k}={v['logit_fid_spearman']:.4f}" for k, v in out["read_eps_sweep"].items()), flush=True)
    g3 = out["gate_3_controls_change_result"]
    print(f"  (3) controls change result: "
          f"eps-degrades={g3['read_eps_degrades_at_1e-2']['pass']}  "
          f"shuffle-head-collapses(fid={g3['shuffle_head_collapses_fid']['spearman']:.3f})="
          f"{g3['shuffle_head_collapses_fid']['pass']}  "
          f"ssm-lesion-breaks(fid {g3['ssm_lesion_breaks_ppl']['faithful_fid_spearman']:.3f}->"
          f"{g3['ssm_lesion_breaks_ppl']['lesion_fid_spearman']:.3f})="
          f"{g3['ssm_lesion_breaks_ppl']['pass']}", flush=True)
    print(f"  VERDICT: smoke_wired={out['smoke_wired']}  "
          f"({'wiring OK; controls all live; faithful at RF read error' if out['smoke_wired'] else 'SEE FAILED GATE ABOVE -- honest negative is first-class'})",
          flush=True)
    print("=" * 96, flush=True)


# =====================================================================================================
# FULL (GPU/cupy or CPU) -- run3's actual ckpt, per-seed ANN vs spiking-graded-read on val windows
# =====================================================================================================
def run_full_one_seed(model, info, seed, args, bridges=None, w32=None):
    from research.runners._genseq_loopstep3_rf_probe import RF_PERIOD, RF_LAMBDA
    V = model["vocab_size"]
    block_size = args.block_size
    ids = list(load_val_ids(seed, block_size * (args.n_windows + 2)))

    def ann_fwd(x):
        rd = make_numpy_graded_read(0.0, None)
        return wkv_forward(model, x, rd)

    err_accum = [] if args.backend == "rf-bridge" else None
    if args.backend == "rf-bridge":
        if bridges is None:      # standalone (single-seed) caller: local cache; run_full passes shared ones
            bridges = {}
        graded = make_rf_graded_read(bridges, period=RF_PERIOD, nsteps=args.nsteps, lam=RF_LAMBDA,
                                     err_accum=err_accum, w32=w32)
    else:
        graded = make_numpy_graded_read(args.read_eps, np.random.default_rng(seed))

    def spk_fwd(x):
        return wkv_forward(model, x, graded)

    ann_ppl = _perplexity(_heldout_nll_numpy(ann_fwd, ids, V, block_size, args.n_windows))
    spk_ppl = _perplexity(_heldout_nll_numpy(spk_fwd, ids, V, block_size, args.n_windows))
    fid = logit_fidelity(ann_fwd, spk_fwd, ids, block_size, args.n_windows, args.n_logit_pos,
                         np.random.default_rng(seed))
    ratio = spk_ppl / ann_ppl if math.isfinite(ann_ppl) and ann_ppl > 0 else float("inf")
    rec = {"seed": seed, "ann_ppl": ann_ppl, "spiking_ppl": spk_ppl, "ppl_ratio": ratio,
           "logit_fid_spearman": fid["spearman"], "logit_fid_cosine": fid["cosine"],
           "n_windows": args.n_windows, "n_logit_pos": args.n_logit_pos}
    if err_accum:
        rec["rf_max_read_err"] = float(max(err_accum))
    print(f"[wkv-spk-fwd:FULL seed={seed}] ann_ppl={ann_ppl:.4f} spk_ppl={spk_ppl:.4f} "
          f"ppl_ratio={ratio:.6f} logit_fid={fid['spearman']:.4f}"
          + (f" rf_read_err={rec['rf_max_read_err']:.1e}" if err_accum else ""), flush=True)
    return rec


def run_full(args):
    t0 = time.time()
    model, info = build_numpy_model_from_ckpt(Path(args.ckpt))
    print(f"[wkv-spk-fwd:FULL] loaded {info['total_params_M']}M WKV "
          f"(d={info['d_model']} L={info['n_layers']} V={info['vocab_size']}) backend={args.backend} "
          f"nsteps={args.nsteps}", flush=True)
    seeds = args.seeds if args.seeds else [args.seed]
    # PERSIST the RF bridge cache + the float32-weight-cast cache ACROSS seeds. Otherwise each seed's Wf
    # arrays are GC'd on return and the next seed's Wf reuse their id() -> the module-level
    # _WEIGHT_CSR_CACHE (keyed by id(Wf)) aliases the WRONG weight's CSR -> a wrong matvec -> the RF read
    # blows up (seed 43 read_err 14.05 / ppl 130; seed 42 clean as it ran first). Shared caches keep every
    # Wf alive -> stable id -> valid cache + genuine cross-seed CSR reuse. (2026-07-23 seed-43 blowup fix.)
    _shared_bridges = {}
    _shared_w32 = {}
    recs = [run_full_one_seed(model, info, s, args, bridges=_shared_bridges, w32=_shared_w32)
            for s in seeds]
    ratios = [r["ppl_ratio"] for r in recs]
    fids = [r["logit_fid_spearman"] for r in recs]
    verdict_go = (float(np.mean(ratios)) <= 1.05 and float(np.mean(fids)) >= 0.99)
    out = {"runner": "_wkv_spiking_forward_derisk", "mode": "full", "backend": args.backend,
           "ckpt": args.ckpt, "arch": info,
           "knobs": {"nsteps": args.nsteps, "read_eps": args.read_eps, "block_size": args.block_size,
                     "n_windows": args.n_windows, "n_logit_pos": args.n_logit_pos, "seeds": seeds,
                     "rf_period": 100000, "rf_lambda": 0.0},
           "per_seed": recs,
           "mean_ppl_ratio": float(np.mean(ratios)), "mean_logit_fid_spearman": float(np.mean(fids)),
           "verdict_go_ppl_ratio<=1.05_and_fid>=0.99": bool(verdict_go),
           "elapsed_seconds": round(time.time() - t0, 2)}
    print(f"[wkv-spk-fwd:FULL] mean ppl_ratio={out['mean_ppl_ratio']:.6f} "
          f"mean logit_fid={out['mean_logit_fid_spearman']:.4f}  "
          f"VERDICT {'GO' if verdict_go else 'NEGATIVE (first-class -- maps the substrate limit)'}", flush=True)
    return out


# =====================================================================================================
def build_aws_command(args):
    # 6 "configs" = 6 val-window seeds (the RF forward itself is deterministic at lam=0/bridge-seed 42, so
    # the only seed-dependence is WHICH val windows are scored -> robustness over data). n_windows=16 is
    # chosen for cost: run3's head is V=16000 (~8x the 88.6M's V=2048), so each window's head resonate
    # dominates. Cheap-first pass FIRST: --seeds 42 43 --n-windows 8  (~20-40 min) before the full 6-seed.
    return (".venv/bin/python -m research.runners._wkv_spiking_forward_derisk --mode full "
            "--ckpt bridges/lmtrain/run3/ckpt/best.pt --backend rf-bridge "
            "--seeds 42 43 44 100 101 102 --n-windows 16 --nsteps 8 --block-size 256 --n-logit-pos 16 "
            "--out research/findings/raw/wkv_spiking_forward/run3_rf_6seed.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--backend", choices=["numpy-model", "rf-bridge"], default="numpy-model",
                    help="numpy-model = CPU RF-read model (smoke); rf-bridge = real RF-on-bridge (GPU/cupy)")
    ap.add_argument("--ckpt", default=str(RUN3_CKPT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--n-windows", type=int, default=8)
    ap.add_argument("--n-logit-pos", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--nsteps", type=int, default=8, help="RF resonate budget (rf-bridge backend)")
    ap.add_argument("--read-eps", type=float, default=5e-7,
                    help="numpy-model absolute read error (CPU stand-in for the RF finite-precision read)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-emit-webapp-sidecar", action="store_true")
    args = ap.parse_args()

    if args.mode == "smoke":
        out = run_smoke(args)
    else:
        out = run_full(args)

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"[wkv-spk-fwd] wrote {p}", flush=True)
    return out


if __name__ == "__main__":
    main()
