"""BRIDGE CO-RESIDENCE de-risk #1 (cheapest-first) -- install ONE Qwen2.5-0.5B decoder layer's q_proj
weight onto the LIVE SimulationBridge RF (resonate-and-fire complex-synapse) path and verify the RF matvec
is BIT-EXACT vs the B-1 PyTorch matmul.

Scoping: research/findings/2026-06-23-bridge-coresidence-qwen-faculty-scoping.md -- de-risk #1 (the
foundational feasibility check). The scoping found bridge co-residence FEASIBLE + LOCAL (the C1 RF exact-matvec
mechanism transfers to the LLaMA stack with NO new mechanism; VRAM ~11.9 GB worst-case < 24 GB). This de-risk
confirms the matvec transfers at the OP level on a REAL Qwen weight tensor BEFORE the full-layer/full-model port.

WHAT THIS PROVES (vs the C1 Gen-F result):
  C1 proved Re(Z)/nsteps = a@W exact (max-err 4.9e-7) for Gen-F's 256x256 / 256x1024 vanilla-GPT matvecs
  (`_genseq_loopstep3_fullblock_rf.json`). This step confirms the SAME exact-matvec at 896-wide on a REAL Qwen
  q_proj weight + its bias, with the activation = a REAL hidden state from a TinyStories forward through the
  cached Qwen model -- the LLaMA-stack op-level transfer. A GREEN here unblocks de-risk #2 (the full decoder layer).

THE MECHANISM (reused VERBATIM from the C1 RF-distill / full-genf consolidation -- NO new `sim/` edit):
  * Extract layer-L's `q_proj` (a torch nn.Linear, weight [D_out,D_in], optional bias [D_out]).
  * PyTorch q_proj forward is `y = a @ weight.T + bias`. The RF install convention
    (`_set_rf_weights` / `rf_linear_layer_signed`) is `(post=D_in+nn, pre=m, w=W[m,nn]+0j)` so the matvec reads
    `Re(Z_out)/nsteps = a @ W` for W in [D_in,D_out] orientation. Therefore install **W = weight.T**
    (shape [D_in,D_out]) -> the RF read `a @ W = a @ weight.T` == the linear's pre-bias output; ADD the bias
    `b` on the read (exact, host-side -- like C1's bias adds / RoPE / residuals; a bias is NOT a learned matvec).
  * Kick z_in = a (REAL, magnitude=activation, phase 0); resonate RF_NSTEPS with lam=0 + a HUGE period
    (omega~0 -> rotation/step ~= identity) so the complex accumulator computes Re(Z)=nsteps*(a@W) with NO clip,
    NO g*(V-E), NO refractory ceiling (`_rf_advance_one`, sim/bridge.py:5710). Read Re(Z)/nsteps + b.

COMPARE the RF result to the EXACT PyTorch q_proj `a @ weight.T + b`:
  max-abs-err, cosine, relative error (per-row mean), over a batch of REAL activation rows.
  BAR (the C1 level): max-abs-err ~1e-6 -> the matvec transfers to a Qwen-shaped weight bit-exactly.

ANTI-CHEAT (scoping #1: bit-exactness MEASURED, not asserted): we report the measured max|Re(Z)/nsteps - a@W|
  (pre-bias, the pure RF-vs-float-matvec claim) AND the post-bias max|RF+b - (a@W.T+b)| AND a SHUFFLED-weight
  control (row-permute the installed W -> the RF read must DIVERGE from the true q_proj, ruling out a trivial
  pass). Bias is exercised only if q_proj has one (Qwen2.5 q/k/v_proj DO have bias).

NO `sim/` edit (the RF path `rf_set_complex_weights`/`rf_kick`/`rf_resonate_steps`/`rf_read_phases` already
exists from the C1 generative arc; reuse-by-import). GPU (SIM_BACKEND=cupy). FOREGROUND/blocking by design. Usage:
  SIM_BACKEND=cupy python -m research.runners._bridge_cores_qproj_derisk
  SIM_BACKEND=cupy python -m research.runners._bridge_cores_qproj_derisk --layer 12 --n-rows 16
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

# Reuse the EXACT RF bridge + the RF operating point from the C1 generative arc (reuse-by-import; no new mechanism).
from research.runners.rf_phasor_composer import _build_rf_bridge  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CORPUS = _REPO / "data" / "corpus" / "tinystories.txt"
OUT = _REPO / "research" / "findings" / "raw" / "_bridge_cores_qproj_derisk.json"

# The C1 RF operating point (identical to `_genseq_loopstep3_rf_probe.py` / `_set_rf_weights`):
# lam=0 (no magnitude decay) + a HUGE period so omega=2pi/period~0 (rotation/step ~ identity) => the complex
# accumulator computes Re(Z_out)=nsteps*(a@W) EXACTLY (C1 measured max-err 4.9e-7).
RF_PERIOD = 100000
RF_NSTEPS = 8
RF_LAMBDA = 0.0

# The C1 exactness bar (the full Gen-F block reported 4.9e-7). We pass at ~1e-5 with margin for the wider 896-dim
# + fp16->fp32 activation cast; report the actual number either way.
EXACT_BAR = 1e-5


def log(msg):
    print(f"[qproj-rf] {msg}", flush=True)


def _rf_matvec_rows(bridge, W, rows, *, period, nsteps, lam):
    """Run the RF COMPLEX-SYNAPSE matvec on EVERY row of `rows` (N, D_in): install W (real, [D_in,D_out]) as
    complex synapses (post=D_in+nn <- pre=m, w=W[m,nn]+0j); per row kick z_in=row, resonate nsteps, read
    Re(Z)/nsteps = row @ W. Returns (N, D_out). Identical mechanism to C1's `_set_rf_weights` + `_rf_matvec_rows`
    and `rf_linear_layer_signed` -- the only difference is W is now a Qwen q_proj.weight.T."""
    import cupy as cp
    D_in, D_out = W.shape
    n = D_in + D_out
    # install W as complex synapses (W_im=0) -- REPLACES any prior weights on the bridge.
    conns = [(D_in + nn, m, complex(float(W[m, nn]), 0.0))
             for m in range(D_in) for nn in range(D_out) if W[m, nn] != 0.0]
    bridge.rf_set_complex_weights(conns)
    out = np.zeros((rows.shape[0], D_out), dtype=np.float64)
    inv = 1.0 / float(nsteps)
    for r in range(rows.shape[0]):
        kick = np.zeros(n, dtype=np.complex128)
        kick[:D_in] = np.asarray(rows[r], dtype=np.float64)
        bridge.rf_kick(kick, period=int(period), lam=float(lam))
        bridge.rf_resonate_steps(int(nsteps))
        re = cp.asnumpy(bridge.cp_membrane_potential_v[D_in:]).astype(np.float64)
        out[r] = re * inv
    return out


def _metrics(rf, ref):
    """max-abs-err + mean cosine (per-row) + mean relative error (per-row) of rf vs ref, both (N,D)."""
    rf = np.asarray(rf, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    max_abs = float(np.max(np.abs(rf - ref)))
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
        "mean_rel_err": float(np.mean(rel_rows)),
        "max_rel_err": float(np.max(rel_rows)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=12, help="which decoder layer's q_proj (B-1 used layer 12)")
    ap.add_argument("--n-rows", type=int, default=16, help="number of real activation rows to test the matvec on")
    args = ap.parse_args()

    t0 = time.time()
    backend = os.environ.get("SIM_BACKEND", "auto")
    log(f"SIM_BACKEND={backend}")

    import torch
    import torch.nn.functional as F  # noqa: F401  (kept for parity with B-1 forward env)
    log(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
        f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"loading {MODEL_ID} (fp16, eager attention) ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16,
                                                 attn_implementation="eager").cuda().eval()
    device = next(model.parameters()).device
    log(f"loaded; {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params on {device}")

    # ---- pick layer-L's q_proj; capture its INPUT activation (the attention input) on a real TinyStories line ----
    layer = model.model.layers[args.layer]
    qproj = layer.self_attn.q_proj
    D_out, D_in = qproj.weight.shape   # nn.Linear: weight is [out_features, in_features]
    has_bias = qproj.bias is not None
    log(f"layer {args.layer} q_proj: weight {tuple(qproj.weight.shape)} (D_out={D_out}, D_in={D_in}) "
        f"bias={'present' if has_bias else 'NONE'}")

    captured = {}

    def qproj_input_hook(mod, inputs, output):
        # inputs[0] is the activation fed to q_proj = the attention input (post input_layernorm hidden state).
        captured["a"] = inputs[0].detach()
        captured["y"] = output.detach()    # the model's OWN q_proj output (a@W.T+b) at fp16 -- a cross-check
    h = qproj.register_forward_hook(qproj_input_hook)

    if CORPUS.exists():
        with open(CORPUS, "r", encoding="utf-8") as f:
            text = f.read()
        line = text[:2000]
    else:
        line = "Once upon a time there was a little girl who loved to read books in the garden every day."
    enc = tok(line, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**{k: v[:, :128] for k, v in enc.items()})
    h.remove()

    a_full = captured["a"]              # (1, S, D_in) fp16
    a_seq = a_full[0]                   # (S, D_in)
    S = a_seq.shape[0]
    n_rows = min(args.n_rows, S)
    # take the LAST n_rows positions (richest context) as the test activations.
    a_rows = a_seq[-n_rows:].to(torch.float32).cpu().numpy().astype(np.float64)   # (n_rows, D_in)
    log(f"captured real q_proj input activations: seq_len={S}, testing last n_rows={n_rows}; "
        f"a range [{a_rows.min():.3f}, {a_rows.max():.3f}]")

    # ---- the EXACT PyTorch reference: y = a @ weight.T + b  (computed in fp64 for an exact reference) ----
    W_torch = qproj.weight.detach().to(torch.float64).cpu().numpy()   # (D_out, D_in)
    b_torch = (qproj.bias.detach().to(torch.float64).cpu().numpy() if has_bias
               else np.zeros(D_out, dtype=np.float64))                # (D_out,)
    # install orientation: W_install[D_in, D_out] = W_torch.T so the RF read a @ W_install == a @ weight.T.
    W_install = np.ascontiguousarray(W_torch.T)                       # (D_in, D_out)
    ref_prebias = a_rows @ W_install                                  # (n_rows, D_out) == a @ weight.T (exact fp64)
    ref_postbias = ref_prebias + b_torch[None, :]                     # == a @ weight.T + b  (the q_proj forward)
    # cross-check: the model's OWN fp16 q_proj output (sanity that we captured the right tensor / orientation).
    y_model = captured["y"][0, -n_rows:].to(torch.float64).cpu().numpy()
    model_vs_fp64 = _metrics(y_model, ref_postbias)
    log(f"sanity: model fp16 q_proj output vs our fp64 (a@W.T+b): max-abs {model_vs_fp64['max_abs_err']:.4e}, "
        f"cos {model_vs_fp64['mean_cosine']:.6f} (fp16 rounding only -> confirms orientation/tensor)")

    # ---- VRAM pre-flight (the scoping's load-bearing number, at the single-matvec scale) ----
    n_neurons = D_in + D_out
    nnz = int(np.count_nonzero(W_install))   # dense q_proj -> ~D_in*D_out
    # bridge stores re+im complex CSR: 2 * (8B f64 data + 4B int32 idx) per nnz.
    est_gb = (nnz * 2 * (8 + 4) + n_neurons * 64) / 1e9
    log(f"VRAM pre-flight: RF bridge n={n_neurons} neurons, nnz={nnz:,} -> ~{est_gb:.4f} GB (re+im f64 CSR; "
        f"the single q_proj matvec)")

    # ---- build the LIVE RF bridge (n = D_in + D_out neurons; RESONATE_AND_FIRE) ----
    log(f"building live RF SimulationBridge ({n_neurons} neurons) ...")
    bridge = _build_rf_bridge(n_neurons, seed=42)

    # ---- run the RF matvec on the real activation rows ----
    log(f"installing q_proj.weight.T as complex synapses + running RF matvec (period={RF_PERIOD}, "
        f"nsteps={RF_NSTEPS}, lam={RF_LAMBDA}) on {n_rows} rows ...")
    t_rf = time.time()
    rf_prebias = _rf_matvec_rows(bridge, W_install, a_rows, period=RF_PERIOD, nsteps=RF_NSTEPS, lam=RF_LAMBDA)
    rf_postbias = rf_prebias + b_torch[None, :]
    rf_seconds = time.time() - t_rf

    # ---- metrics: RF vs PyTorch ----
    m_prebias = _metrics(rf_prebias, ref_prebias)      # the PURE RF-vs-float-matvec claim (a@W, no bias)
    m_postbias = _metrics(rf_postbias, ref_postbias)   # the full q_proj forward (a@W.T+b)
    log(f"RF vs PyTorch q_proj | PRE-bias  (a@W.T): max-abs {m_prebias['max_abs_err']:.4e}, "
        f"cos {m_prebias['mean_cosine']:.8f}, rel {m_prebias['mean_rel_err']:.4e}")
    log(f"RF vs PyTorch q_proj | POST-bias (a@W.T+b): max-abs {m_postbias['max_abs_err']:.4e}, "
        f"cos {m_postbias['mean_cosine']:.8f}, rel {m_postbias['mean_rel_err']:.4e}")

    # ---- ANTI-CHEAT: shuffled-weight control (row-permute W_install -> RF must DIVERGE from true q_proj) ----
    rng = np.random.default_rng(12345)
    perm = rng.permutation(D_in)
    W_shuf = np.ascontiguousarray(W_install[perm, :])
    rf_shuf = _rf_matvec_rows(bridge, W_shuf, a_rows, period=RF_PERIOD, nsteps=RF_NSTEPS, lam=RF_LAMBDA)
    m_shuf_vs_true = _metrics(rf_shuf, ref_prebias)    # RF(shuffled W) vs the TRUE a@W -> must collapse
    # also confirm the shuffled RF still EXACTLY reproduces ITS OWN matvec (proves the divergence is the weights,
    # not an RF failure): RF(shuf) vs a@W_shuf must be ~exact.
    ref_shuf = a_rows @ W_shuf
    m_shuf_self = _metrics(rf_shuf, ref_shuf)
    log(f"anti-cheat shuffled-weight: RF(shufW) vs TRUE a@W -> max-abs {m_shuf_vs_true['max_abs_err']:.3e}, "
        f"cos {m_shuf_vs_true['mean_cosine']:.4f} (must DIVERGE); RF(shufW) vs a@W_shuf -> "
        f"max-abs {m_shuf_self['max_abs_err']:.3e} (must stay EXACT)")

    # ---- VERDICT ----
    bit_exact = (m_prebias["max_abs_err"] <= EXACT_BAR)
    shuffled_diverges = (m_shuf_vs_true["mean_cosine"] < 0.5
                         or m_shuf_vs_true["max_abs_err"] > 10 * m_prebias["max_abs_err"])
    shuffled_self_exact = (m_shuf_self["max_abs_err"] <= EXACT_BAR)
    if bit_exact and shuffled_diverges and shuffled_self_exact:
        verdict = "GO"
        tail = (f"the RF complex-synapse matvec reproduces the Qwen q_proj `a@W.T+b` BIT-EXACTLY "
                f"(pre-bias max-abs {m_prebias['max_abs_err']:.2e} <= C1 bar {EXACT_BAR:.0e}, "
                f"cos {m_prebias['mean_cosine']:.8f}) on the LIVE bridge with a REAL 896-wide Qwen weight + its "
                f"bias -> the C1 matvec mechanism TRANSFERS to the LLaMA stack at the op level. The full-layer "
                f"port (de-risk #2) is UNBLOCKED. NO `sim/` edit (reuse-by-import). VRAM for this matvec "
                f"~{est_gb:.3f} GB (the scoping's per-layer streaming pattern keeps the full model local).")
    elif bit_exact:
        verdict = "GO_WITH_CAVEAT"
        tail = (f"the matvec is bit-exact (pre-bias max-abs {m_prebias['max_abs_err']:.2e}) but an anti-cheat is "
                f"soft (shuffled_diverges={shuffled_diverges}, shuffled_self_exact={shuffled_self_exact}) -- "
                f"report + inspect.")
    else:
        verdict = "HONEST_RESIDUAL"
        tail = (f"the RF read did NOT reach the C1 bit-exact bar: pre-bias max-abs {m_prebias['max_abs_err']:.3e} "
                f"> {EXACT_BAR:.0e}. Diagnose: shape/dtype/scale/bias/orientation. The pure-matvec cosine is "
                f"{m_prebias['mean_cosine']:.6f}; if cosine~1 but max-abs large it is a SCALE issue (nsteps / a "
                f"magnitude), if cosine low it is an ORIENTATION issue (W vs W.T). No `sim/` edit was added.")

    verdict_line = (
        f"bridge_cores_qproj: layer {args.layer} q_proj [{D_out}x{D_in}] (bias={'Y' if has_bias else 'N'}) "
        f"installed on the LIVE RF bridge ({n_neurons} neurons); RF Re(Z)/nsteps vs PyTorch a@W.T -> "
        f"PRE-bias max-abs {m_prebias['max_abs_err']:.3e} cos {m_prebias['mean_cosine']:.8f}; "
        f"POST-bias (+b) max-abs {m_postbias['max_abs_err']:.3e} cos {m_postbias['mean_cosine']:.8f}; "
        f"shuffled-control RF-vs-true cos {m_shuf_vs_true['mean_cosine']:.3f} (diverges={shuffled_diverges}); "
        f"[C1 bar {EXACT_BAR:.0e}; C1 Gen-F ref 4.9e-7] -> {verdict}. {tail}")

    result = {
        "probe": "bridge_coresidence_derisk1_qwen_qproj_rf_bitexact",
        "resolves": "de-risk #1 (scoping 2026-06-23): install ONE Qwen2.5-0.5B q_proj weight on the LIVE "
                    "SimulationBridge RF path + verify the RF matvec is BIT-EXACT vs the B-1 PyTorch matmul -- "
                    "confirm the C1 exact-matvec mechanism TRANSFERS to the LLaMA stack at the OP level before "
                    "the full-layer/full-model port.",
        "model_id": MODEL_ID,
        "layer": args.layer,
        "q_proj_shape": [int(D_out), int(D_in)],
        "has_bias": bool(has_bias),
        "n_neurons": int(n_neurons),
        "nnz": int(nnz),
        "n_rows_tested": int(n_rows),
        "seq_len": int(S),
        "rf_operating_point": {"period": RF_PERIOD, "nsteps": RF_NSTEPS, "lambda": RF_LAMBDA,
                               "read": "Re(Z_out)/nsteps = a @ W (W=q_proj.weight.T); bias added host-side on the read"},
        "exact_bar": EXACT_BAR,
        "c1_reference_max_err": 4.9e-7,
        "vram_est_gb_this_matvec": round(est_gb, 4),
        "mechanism": "reuse-by-import of the C1 RF exact-matvec (rf_set_complex_weights / rf_kick / "
                     "rf_resonate_steps + the Re(Z)/nsteps read); install W = q_proj.weight.T as complex synapses "
                     "(W_im=0); the bias is a host-side add on the read (NOT a learned matvec). NO `sim/` edit.",
        "sanity_model_fp16_vs_fp64_ref": model_vs_fp64,
        "rf_vs_pytorch_prebias_a_at_W": m_prebias,
        "rf_vs_pytorch_postbias_a_at_W_plus_b": m_postbias,
        "anti_cheat_shuffled_weight": {
            "rf_shuf_vs_true_a_at_W": m_shuf_vs_true,
            "rf_shuf_vs_own_a_at_W_shuf": m_shuf_self,
            "shuffled_diverges_from_true": bool(shuffled_diverges),
            "shuffled_still_self_exact": bool(shuffled_self_exact),
            "note": "row-permuting the installed W must make the RF read DIVERGE from the true q_proj (cos<0.5) "
                    "while STILL exactly reproducing its own a@W_shuf -- proving the RF carries the installed "
                    "weights, not a trivial pass.",
        },
        "rf_seconds": round(rf_seconds, 3),
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
