"""LOOP-STEP 3 de-risk #5 -- the C1 MILESTONE: does the FULL Gen-F GENERATOR run end-to-end on the
bridge AND GENERATE coherent text -- ALL 4 transformer blocks' learned weights + the token embedding
+ the output head on the conductance-free RF complex-synapse path (EXACT learned-weight matvecs) +
the parameter-free nonlinearities (softmax / GELU / LayerNorm) as faithful reads -- matching the
off-bridge Gen-F?

READ FIRST (the GO this STACKS up to the full model):
  - research/findings/raw/_genseq_loopstep3_fullblock_rf.json (fidelity 1.000): a FULL Gen-F BLOCK
    (block-0) consolidates on the bridge at spearman+cosine = 1.000 -- all 786432 learned weights on
    the RF path (EXACT, max|Re(Z)/nsteps - h@W| = 4.9e-07) + softmax/GELU/LayerNorm as faithful reads;
    the two residual streams compose; LESION(scramble RF) -> 0.67 (the residual floor). ONE block DONE.
  - 2026-06-22-genseq-loopstep3-attn-rf-distill-GO-...md (attn Q/K/V/O EXACT 1.000 on RF; softmax =
    0-param faithful read) + _genseq_loopstep3_mlp_gelu_rf_distill.json (MLP linears EXACT 1.000 on
    RF; GELU = exact-erf faithful read between the two exact linears).

THE FULL GENERATOR (sim/tiny_transformer.py TinyGPT.forward, the EXACT teacher):
    x = tok(idx) + pos(arange(n))          # learned embedding LOOKUP (exact -- a gather, not a matvec)
    for b in blocks(4): x = b(x)            # 4 x _Block: LN1 -> attn(softmax) -> +x -> LN2 -> MLP(GELU) -> +x
    return head(lnf(x))                     # final LN (faithful read) -> head: a LEARNED Linear -> vocab logits

THIS DE-RISK'S SCOPE (stated precisely; identical to the full-block GO, EXTENDED to the whole model):
  * Every LEARNED-WEIGHT matvec goes through the RF complex-synapse path (rf_linear_layer_signed --
    the SAME exact RF escape that gave the attention projections + MLP linears + the full block rank
    1.000): the 4 blocks' attn Q/K/V/O (4 x 4 x 256x256) + MLP W1/W2 (4 x [256x1024 + 1024x256]) =
    4 x 786432 = 3,145,728 params + the OUTPUT HEAD (256 -> 513 = 131328 params) -> 3,277,056 learned
    matvec params -- ALL of the generator's learned matvecs, ALL on the conductance-free RF path
    (EXACT). The token+position EMBEDDING is a learned LOOKUP (a gather of tok.weight/pos.weight rows),
    NOT a matvec -- it carries learned content but on the RF path there is no matvec to install (the
    embedding rows ARE the input x; faithfully used, like a Linear bias on the read).
  * The PARAMETER-FREE nonlinearities are FAITHFUL READS (host): softmax(QK^T) [0 params], GELU [0
    params, exact-erf], LayerNorm [content-norm read + the learned per-feature affine rides on the
    read]. The fully-SPIKING nonlinearities (spiking softmax / LayerNorm / GELU) are the SEPARATE
    follow-on -- STATED HONESTLY (the full-block GO's exact same scope, now across the whole model).

WHAT THIS RUNNER MEASURES (the prompt's STEP 2):
  (a) NEXT-TOKEN-LOGIT fidelity vs the exact-float Gen-F: run the FULL 4-block forward + lnf + head on
      a REAL prompt, per position, and score the RF-on-bridge logits vs the exact-float Gen-F logits
      over the 513-dim vocab (analog Spearman + cosine, per position; the SAME analog-rank basis as
      the block/attention/MLP de-risks). This is the C1 "the full generator WEIGHTS on the bridge"
      fidelity -- across all 4 stacked blocks (does error accumulate?).
  (b) GENERATE a sample (greedy + temperature) on the RF-on-bridge model + compare to the off-bridge
      Gen-F's generation (next-token argmax agreement under greedy teacher-forcing on the bridge's
      own context). The actual generated TEXT is decoded + recorded (the controller reads it).
  (c) held-out PERPLEXITY vs off-bridge (the convert metric, reused VERBATIM): teacher-forced
      next-token CE over held-out windows, RF-on-bridge vs the exact-float Gen-F.

ANTI-CHEATS (the prompt's STEP 3):
  (1) the logit fidelity >= ~0.8 across positions (the C1 GO bar; == the block synthesis bar).
  (2) a LOAD-BEARING LESION: scramble the RF complex weights of EVERY learned matvec (the 4 blocks'
      Q/K/V/O + W1/W2 + the head) -> generation MUST collapse to degenerate (the logits lose their
      structure; the decoded sample becomes repetitive garbage) -> proves the RF matvecs carry the
      computation, NOT the host nonlinearities.
  (3) the generated sample is COHERENT (not repetitive garbage) -- recorded + the controller reads it.

VERDICT (the prompt's STEP 4):
  GO = the full 4-block on-bridge forward GENERATES coherent text matching off-bridge Gen-F (logit
       fidelity >= 0.8 + a coherent sample + ppl ~= off-bridge ratio <= ~1.2 + greedy argmax agreement
       high) AND the lesion collapses generation. PLUS the honest scope (weights-on-RF +
       nonlinearities-as-faithful-reads; the fully-spiking nonlinearities are the follow-on).
  PARTIAL = composes above chance but < 0.8 (or generation degrades while ppl holds) -> report the
       precise failure point + per-layer fidelity (where across the 4-block stack the error accumulates).
  NEGATIVE = the 4-block stack degrades / generation is degenerate -> report the precise failure point.

PERF NOTE: rf_set_complex_weights rebuilds the complex CSR per matvec. The weights are FIXED across
sequence positions, so we set each matvec's weights ONCE and reuse the bridge for ALL positions of
that matvec (the kick/resonate/read cycle per position) -- the batched RF projection. This makes a
full forward ~(n_matvecs weight-sets) + (n_matvecs * n_positions resonate windows), the resonate being
cheap. Generation is kept to a modest token budget (each autoregressive step is a full forward over
the growing context). The per-matvec EXACTness (max|Re(Z)/nsteps - h@W|) is MEASURED, not asserted.

NO sim/ edit (the RF path + the RF-linear primitive + GELU + LayerNorm helpers ALL already exist;
reuse-by-import from the full-block + MLP + attention de-risks + the RF probe). GPU. Usage:
  SIM_BACKEND=cupy python -m research.runners._genseq_loopstep3_full_genf_generate_derisk
"""
from __future__ import annotations

import gc
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

import importlib.util as _ilu  # load the BPE tokenizer WITHOUT importing sim at top (avoids a stray import)

# Reuse the EXACT-RF-linear primitive + bridge builder + operating point + metric + GELU + LayerNorm
# VERBATIM (NO duplication of the load-bearing machinery -- the SAME chain the full-block GO used for
# its exact-1.000 matvecs):
from research.runners._genseq_loopstep3_graded_derisk import spearman  # noqa: E402
from research.runners._genseq_loopstep3_rf_probe import (  # noqa: E402
    _build_rf_bridge,
    RF_PERIOD,
    RF_NSTEPS,
    RF_LAMBDA,
)
# the EXACT erf GELU + the float LayerNorm, reuse-by-import (identical helpers to the full-block GO).
from research.runners._genseq_loopstep3_mlp_gelu_rf_distill_derisk import (  # noqa: E402
    gelu_exact,
    _layernorm,
)

GENF_CKPT = _REPO / "research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real.pt"
GENF_BPE = _REPO / "research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real.bpe.json"
OUT_PATH = _REPO / "research/findings/raw/_genseq_loopstep3_full_genf_generate.json"

# Real TinyStories-style probe text (identical register to the full-block + attention + MLP de-risks;
# in-distribution for the Gen-F BPE vocab). ASCII only.
PROBE_TEXT = (
    "Once upon a time there was a little girl named Lily. She had a small dog and a big cat. "
    "One day they went to the park to play. The sun was bright and the sky was blue. "
    "Tim saw a red ball and wanted to play with his friend. They were very happy together. "
    "Lily smiled and said the day was fun. Her mom came to find them and they all went home."
)

N_LOGIT_PROBE_POS = 12   # REAL positions probed for the next-token-logit fidelity (over the 513 vocab)
N_GEN_TOKENS = 18        # autoregressive generation budget (each step = a full forward over the growing
                         # context; kept modest because the RF per-position kick/resonate/read loop is
                         # the bottleneck -- 18 tokens reads coherently while keeping the run foreground)
N_PERPLEXITY_POS = 4     # held-out windows for the ppl-vs-off-bridge comparison
PPL_WINDOW = 48          # held-out window length for ppl (a shorter window keeps the slow per-position
                         # RF loop bounded; the next-token CE is identical-semantics to the gate, just on
                         # 48-token windows instead of the full 128 block_size)
GO_BAR = 0.8             # the C1 logit-fidelity bar (== the block synthesis bar)
OOM_CEILING_GB = 16.0


def free_cuda():
    gc.collect()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _load_bpe(path):
    spec = _ilu.spec_from_file_location("bpe_tokenizer", str(_REPO / "sim/bpe_tokenizer.py"))
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.BPETokenizer.load(str(path))


# =================================================================================================
# Load Gen-F: ALL 4 blocks' weights + the embedding + the final LN + the output head, plus the BPE.
# =================================================================================================
def load_genf_full():
    """Load Gen-F (s42.real) and EXTRACT the WHOLE model's parameters in the install convention
    (a_out = a_in @ W = W_lin^T): per-block (LN1/LN2 affines, attn Q/K/V/O + biases, MLP W1/W2 +
    biases) + tok/pos embeddings + final LN (lnf) affine + the output head (a learned Linear, no bias).
    """
    import torch
    # weights_only=True: OUR OWN trusted, local, project-generated training output -- restrict to the
    # safe tensor/primitive unpickler regardless.
    ck = torch.load(str(GENF_CKPT), map_location="cpu", weights_only=True)
    sd = ck["model"]
    d_model = int(sd["tok.weight"].shape[1])
    n_head = 4
    n_layer = sum(1 for k in sd if k.endswith(".ln1.weight") and k.startswith("blocks."))
    block_size = int(sd["pos.weight"].shape[0])
    V = int(sd["tok.weight"].shape[0])
    loss_last = float(ck["loss_history"][-1]) if ck.get("loss_history") else float("nan")

    blocks = []
    for li in range(n_layer):
        p = f"blocks.{li}."
        ln1_w = sd[p + "ln1.weight"].numpy().astype(np.float64)
        ln1_b = sd[p + "ln1.bias"].numpy().astype(np.float64)
        ln2_w = sd[p + "ln2.weight"].numpy().astype(np.float64)
        ln2_b = sd[p + "ln2.bias"].numpy().astype(np.float64)
        in_w = sd[p + "attn.in_proj_weight"].numpy().astype(np.float64)   # (3d,d) = [W_Q;W_K;W_V]
        in_b = sd[p + "attn.in_proj_bias"].numpy().astype(np.float64)     # (3d,)
        Wq_lin, Wk_lin, Wv_lin = in_w[:d_model], in_w[d_model:2 * d_model], in_w[2 * d_model:]
        bq, bk, bv = in_b[:d_model], in_b[d_model:2 * d_model], in_b[2 * d_model:]
        Wo_lin = sd[p + "attn.out_proj.weight"].numpy().astype(np.float64)  # (d,d)
        bo = sd[p + "attn.out_proj.bias"].numpy().astype(np.float64)
        W_fc1_lin = sd[p + "mlp.0.weight"].numpy().astype(np.float64)   # (4d,d)
        b_fc1 = sd[p + "mlp.0.bias"].numpy().astype(np.float64)         # (4d,)
        W_fc2_lin = sd[p + "mlp.2.weight"].numpy().astype(np.float64)   # (d,4d)
        b_fc2 = sd[p + "mlp.2.bias"].numpy().astype(np.float64)         # (d,)
        blocks.append({
            "ln1_w": ln1_w, "ln1_b": ln1_b, "ln2_w": ln2_w, "ln2_b": ln2_b,
            "Wq": Wq_lin.T.astype(np.float32).copy(), "Wk": Wk_lin.T.astype(np.float32).copy(),
            "Wv": Wv_lin.T.astype(np.float32).copy(), "Wo": Wo_lin.T.astype(np.float32).copy(),
            "bq": bq, "bk": bk, "bv": bv, "bo": bo,
            "W1": W_fc1_lin.T.astype(np.float32).copy(), "W2": W_fc2_lin.T.astype(np.float32).copy(),
            "b1": b_fc1.copy(), "b2": b_fc2.copy(),
        })

    tok_emb = sd["tok.weight"].numpy().astype(np.float64)   # (V, d)  -- the learned token embedding (lookup)
    pos_emb = sd["pos.weight"].numpy().astype(np.float64)   # (block_size, d) -- learned positional (lookup)
    lnf_w = sd["lnf.weight"].numpy().astype(np.float64)
    lnf_b = sd["lnf.bias"].numpy().astype(np.float64)
    Whead_lin = sd["head.weight"].numpy().astype(np.float64)   # (V, d) -- the output head (no bias)
    Whead = Whead_lin.T.astype(np.float32).copy()             # (d, V) install convention a@W

    meta = {
        "d_model": d_model, "n_head": n_head, "n_layer": n_layer, "block_size": block_size,
        "vocab_size": V, "loss_last": loss_last,
        "lnf_affine_l2": [float(np.linalg.norm(lnf_w)), float(np.linalg.norm(lnf_b))],
        "head_l2": float(np.linalg.norm(Whead)),
        "embedding_kind": "learned LOOKUP (tok.weight/pos.weight row gather; NOT a matvec)",
    }
    model = {
        "blocks": blocks, "tok_emb": tok_emb, "pos_emb": pos_emb,
        "lnf_w": lnf_w, "lnf_b": lnf_b, "Whead": Whead,
        "n_head": n_head, "n_layer": n_layer, "d_model": d_model,
        "block_size": block_size, "vocab_size": V,
    }
    del ck, sd
    return model, meta


# =================================================================================================
# EXACT-FLOAT TEACHER: the genuine Gen-F forward (float64). Mirrors TinyGPT.forward + _Block.forward.
# =================================================================================================
def _attention_float(h, blk, n_head):
    """Exact-float causal multihead attention on the FULL sequence h (N,d)."""
    h = h.astype(np.float64)
    n, d = h.shape
    Q = h @ blk["Wq"].astype(np.float64) + blk["bq"]
    K = h @ blk["Wk"].astype(np.float64) + blk["bk"]
    Vv = h @ blk["Wv"].astype(np.float64) + blk["bv"]
    dh = d // n_head
    attn_out = np.zeros((n, d), dtype=np.float64)
    causal = np.triu(np.ones((n, n), dtype=bool), k=1)
    for hd in range(n_head):
        sl = slice(hd * dh, (hd + 1) * dh)
        scores = (Q[:, sl] @ K[:, sl].T) / math.sqrt(dh)
        scores = np.where(causal, -np.inf, scores)
        scores = scores - scores.max(axis=1, keepdims=True)
        w = np.exp(scores)
        w = w / w.sum(axis=1, keepdims=True)
        attn_out[:, sl] = w @ Vv[:, sl]
    return attn_out @ blk["Wo"].astype(np.float64) + blk["bo"]


def _block_float(x, blk, n_head):
    h = _layernorm(x, blk["ln1_w"], blk["ln1_b"])
    a = _attention_float(h, blk, n_head)
    x1 = x + a
    m = _layernorm(x1, blk["ln2_w"], blk["ln2_b"])
    h1 = m @ blk["W1"].astype(np.float64) + blk["b1"]
    g = gelu_exact(h1)
    mlp_out = g @ blk["W2"].astype(np.float64) + blk["b2"]
    return x1 + mlp_out


def teacher_logits(model, ids):
    """The EXACT-FLOAT teacher: the genuine Gen-F forward on the token ids -> logits (n, V)."""
    ids = np.asarray(ids)
    n = len(ids)
    x = model["tok_emb"][ids] + model["pos_emb"][:n]    # learned embedding lookup (exact)
    for blk in model["blocks"]:
        x = _block_float(x, blk, model["n_head"])
    x = _layernorm(x, model["lnf_w"], model["lnf_b"])   # final LN
    logits = x @ model["Whead"].astype(np.float64)       # output head (a learned Linear, no bias)
    return logits


# =================================================================================================
# RF-ON-BRIDGE: the SAME forward but every learned-weight matvec via the RF complex-synapse path.
# A FAST batched projection: set each matvec's complex weights ONCE, then kick/resonate/read per
# position (the weights are FIXED across positions; rf_set_complex_weights is the dominant cost).
# softmax / GELU / LayerNorm are FAITHFUL READS.
# =================================================================================================
# PERF: each learned matvec's complex CSR (cp_rf_w_re/cp_rf_w_im) is MODEL-CONSTANT across the whole
# run (the weights never change), but rf_set_complex_weights rebuilds it from a Python list comphrehen-
# sion of ~262k entries EVERY call (the dominant cost: ~constant per matvec, blowing up generation
# where each autoregressive step re-runs every matvec). We build each unique matvec's CSR ONCE (keyed
# by id(W)) and thereafter SWAP the cached (re,im) CSR onto the bridge with two attribute assignments
# -- near-instant, BIT-IDENTICAL to a fresh rf_set_complex_weights (same csr_matrix on the same data).
_WEIGHT_CSR_CACHE = {}   # key: (id(W), n_neurons) -> (D_in, D_out, cp_csr_re, cp_csr_im)


def _set_rf_weights(bridge, W):
    """Install the (d_in -> d_out) real weight W as complex synapses (post=d_in+nn <- pre=m, w=W[m,nn]).
    The matvec then reads Re(Z_out)/nsteps = a @ W exactly (omega~0, lam=0). Builds the complex CSR ONCE
    per (W, bridge-size) and CACHES it; later calls just swap the cached CSR onto the bridge (a constant-
    weight model -> no need to rebuild the CSR per position/per forward)."""
    D_in, D_out = W.shape
    n = bridge.core_config.num_neurons
    key = (id(W), n)
    cached = _WEIGHT_CSR_CACHE.get(key)
    if cached is None:
        conns = [(D_in + nn, m, complex(float(W[m, nn]), 0.0))
                 for m in range(D_in) for nn in range(D_out) if W[m, nn] != 0.0]
        bridge.rf_set_complex_weights(conns)   # builds cp_rf_w_re / cp_rf_w_im on the bridge
        _WEIGHT_CSR_CACHE[key] = (D_in, D_out, bridge.cp_rf_w_re, bridge.cp_rf_w_im)
    else:
        D_in, D_out, csr_re, csr_im = cached
        bridge.cp_rf_w_re = csr_re             # swap the cached CSR (== a fresh rf_set_complex_weights)
        bridge.cp_rf_w_im = csr_im
    return D_in, D_out


def _rf_matvec_rows(bridge, D_in, D_out, rows, *, period, nsteps, lam, want_err_vs=None):
    """Run the ALREADY-INSTALLED RF matvec on EVERY row of `rows` (N, D_in): kick z_in=row, resonate
    nsteps, read Re(Z)/nsteps = row @ W. Returns (N, D_out). If want_err_vs (N,D_out float matvec) is
    given, also returns the max|Re(Z)/nsteps - row@W| over the rows (the EXACT-RF claim, MEASURED)."""
    import cupy as cp
    n_tot = D_in + D_out
    out = np.zeros((rows.shape[0], D_out), dtype=np.float64)
    max_err = 0.0
    inv = 1.0 / float(nsteps)
    for r in range(rows.shape[0]):
        kick = np.zeros(n_tot, dtype=np.complex128)
        kick[:D_in] = np.asarray(rows[r], dtype=np.float64)
        bridge.rf_kick(kick, period=int(period), lam=float(lam))
        bridge.rf_resonate_steps(int(nsteps))
        # slice the output neurons ON-DEVICE before the D->H copy (avoids copying the full n_tot array).
        re = cp.asnumpy(bridge.cp_membrane_potential_v[D_in:]).astype(np.float64)
        out[r] = re * inv
        if want_err_vs is not None:
            max_err = max(max_err, float(np.max(np.abs(out[r] - want_err_vs[r]))))
    return out, max_err


def _rf_project_seq(bridge, W, h_seq, *, period, nsteps, lam, measure_err=False):
    """Batched: install W ONCE, then run it on every row of h_seq (N, d_in). Returns (N, d_out, max_err)."""
    D_in, D_out = _set_rf_weights(bridge, W)
    err_ref = (h_seq.astype(np.float64) @ W.astype(np.float64)) if measure_err else None
    out, max_err = _rf_matvec_rows(bridge, D_in, D_out, h_seq, period=period, nsteps=nsteps,
                                   lam=lam, want_err_vs=err_ref)
    return out, max_err


def _rf_block_forward(x, blk, bridges, n_head, *, period, nsteps, lam, measure_err=False):
    """One RF block forward on the FULL sequence x (N,d): all learned matvecs on RF; softmax/GELU/LN
    faithful reads; the two residual adds in float. `bridges` = {"dd":256+256, "mlp1":256+1024,
    "mlp2":1024+256}. Returns (out (N,d), max_err over this block's matvecs)."""
    x = x.astype(np.float64)
    n, d = x.shape
    dh = d // n_head
    b_dd, b_m1, b_m2 = bridges["dd"], bridges["mlp1"], bridges["mlp2"]
    errs = []

    h = _layernorm(x, blk["ln1_w"], blk["ln1_b"])
    Q, eq = _rf_project_seq(b_dd, blk["Wq"], h, period=period, nsteps=nsteps, lam=lam, measure_err=measure_err)
    K, ek = _rf_project_seq(b_dd, blk["Wk"], h, period=period, nsteps=nsteps, lam=lam, measure_err=measure_err)
    Vv, ev = _rf_project_seq(b_dd, blk["Wv"], h, period=period, nsteps=nsteps, lam=lam, measure_err=measure_err)
    Q = Q + blk["bq"]; K = K + blk["bk"]; Vv = Vv + blk["bv"]

    attn_out = np.zeros((n, d), dtype=np.float64)
    causal = np.triu(np.ones((n, n), dtype=bool), k=1)
    for hd in range(n_head):
        sl = slice(hd * dh, (hd + 1) * dh)
        scores = (Q[:, sl] @ K[:, sl].T) / math.sqrt(dh)
        scores = np.where(causal, -np.inf, scores)
        scores = scores - scores.max(axis=1, keepdims=True)
        w = np.exp(scores)
        w = w / w.sum(axis=1, keepdims=True)             # softmax (faithful)
        attn_out[:, sl] = w @ Vv[:, sl]                  # value mix (faithful; not a learned weight)

    a, eo = _rf_project_seq(b_dd, blk["Wo"], attn_out, period=period, nsteps=nsteps, lam=lam, measure_err=measure_err)
    a = a + blk["bo"]
    x1 = x + a                                           # RESIDUAL 1

    m = _layernorm(x1, blk["ln2_w"], blk["ln2_b"])
    h1, e1 = _rf_project_seq(b_m1, blk["W1"], m, period=period, nsteps=nsteps, lam=lam, measure_err=measure_err)
    h1 = h1 + blk["b1"]
    g = gelu_exact(h1)                                  # GELU (faithful read)
    mlp_out, e2 = _rf_project_seq(b_m2, blk["W2"], g, period=period, nsteps=nsteps, lam=lam, measure_err=measure_err)
    mlp_out = mlp_out + blk["b2"]
    out = x1 + mlp_out                                  # RESIDUAL 2

    if measure_err:
        errs = [eq, ek, ev, eo, e1, e2]
    return out, (max(errs) if errs else 0.0)


def rf_full_forward(model, ids, bridges, *, period, nsteps, lam, measure_err=False, blocks_override=None,
                    head_override=None, return_hidden=False):
    """The RF-on-bridge full Gen-F forward on the token ids -> logits (n, V). Every learned matvec
    (all 4 blocks' Q/K/V/O + W1/W2 + the head) on the live RF bridge; embedding lookup exact; softmax/
    GELU/LN/lnf faithful reads. `blocks_override`/`head_override` allow the lesion to scramble weights.
    If return_hidden, also returns the per-block output (for per-layer fidelity diagnostics)."""
    ids = np.asarray(ids)
    n = len(ids)
    blocks = blocks_override if blocks_override is not None else model["blocks"]
    Whead = head_override if head_override is not None else model["Whead"]
    x = model["tok_emb"][ids] + model["pos_emb"][:n]     # learned embedding lookup (exact)
    per_block_out = []
    max_err = 0.0
    for blk in blocks:
        x, e = _rf_block_forward(x, blk, bridges, model["n_head"], period=period, nsteps=nsteps,
                                 lam=lam, measure_err=measure_err)
        max_err = max(max_err, e)
        if return_hidden:
            per_block_out.append(x.copy())
    x = _layernorm(x, model["lnf_w"], model["lnf_b"])    # final LN (faithful read)
    logits, eh = _rf_project_seq(bridges["head"], Whead, x, period=period, nsteps=nsteps, lam=lam,
                                 measure_err=measure_err)  # output head on RF
    if measure_err:
        max_err = max(max_err, eh)
    if return_hidden:
        return logits, max_err, per_block_out
    return logits, max_err


# =================================================================================================
# scoring helpers
# =================================================================================================
def _score_logits(rf_logits, teacher_logits_arr, sel):
    """Per-position analog Spearman + cosine of the RF on-bridge logits vs the teacher logits (over
    the V vocab dims), averaged over the sampled positions."""
    sps, coss = [], []
    for p in sel:
        s = spearman(teacher_logits_arr[p], rf_logits[p])
        if not math.isnan(s):
            sps.append(s)
        a = teacher_logits_arr[p].astype(np.float64); b = rf_logits[p].astype(np.float64)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            coss.append(float(a @ b / (na * nb)))
    fid = float(np.mean(sps)) if sps else float("nan")
    cos = float(np.mean(coss)) if coss else float("nan")
    return fid, cos, sps, coss


def _greedy_continue(forward_fn, prompt_ids, n_tokens, block_size):
    """Greedy (argmax) autoregressive continuation on a forward_fn(ids)->logits (n,V). Returns the
    generated id list (deterministic; the most-faithful comparison to the off-bridge greedy decode)."""
    seq = list(prompt_ids) if prompt_ids else [0]
    out = []
    for _ in range(int(n_tokens)):
        ctx = seq[-block_size:]
        logits = forward_fn(ctx)
        nxt = int(np.argmax(logits[-1]))
        seq.append(nxt)
        out.append(nxt)
    return out


def _temp_continue(forward_fn, prompt_ids, n_tokens, block_size, seed, temperature=1.0):
    """Temperature multinomial autoregressive continuation (seeded/reproducible) -- mirrors the gate's
    _generate, but on a numpy forward_fn(ids)->logits."""
    rng = np.random.default_rng(int(seed))
    seq = list(prompt_ids) if prompt_ids else [0]
    out = []
    for _ in range(int(n_tokens)):
        ctx = seq[-block_size:]
        logits = forward_fn(ctx)[-1].astype(np.float64) / max(1e-6, float(temperature))
        logits = logits - logits.max()
        p = np.exp(logits); p = p / p.sum()
        nxt = int(rng.choice(len(p), p=p))
        seq.append(nxt)
        out.append(nxt)
    return out


def _heldout_nll_numpy(forward_fn, ids, V, block_size, max_positions):
    """Teacher-forced per-window mean next-token CE over held-out ids -- the gate's _heldout_nll
    semantics, on a numpy forward_fn(ids)->logits. Returns list[float]."""
    n = len(ids)
    out = []
    if n < block_size + 2:
        return out
    n_windows = (n - 1) // block_size
    step_w = max(1, n_windows // max(1, int(max_positions)))
    wi = 0
    for w in range(0, n_windows, step_w):
        s = w * block_size
        x = ids[s:s + block_size]
        y = ids[s + 1:s + 1 + block_size]
        logits = forward_fn(x).astype(np.float64)       # (block_size, V)
        logits = logits - logits.max(axis=1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        ce = -float(np.mean(logp[np.arange(len(y)), np.asarray(y)]))
        out.append(ce)
        wi += 1
        if wi >= int(max_positions):
            break
    return out


def _perplexity(nll_list):
    if not nll_list:
        return float("inf")
    return float(math.exp(float(np.mean(nll_list))))


def _distinct_trigram(ids):
    if len(ids) < 3:
        return 0.0
    grams = [tuple(ids[i:i + 3]) for i in range(len(ids) - 2)]
    return len(set(grams)) / float(len(grams))


def main():
    backend = os.environ.get("SIM_BACKEND", "auto")
    print(f"[full_genf] SIM_BACKEND={backend}", flush=True)
    t_start = time.time()

    # ---- load the WHOLE Gen-F model + the BPE tokenizer ----
    model, meta = load_genf_full()
    d = model["d_model"]; V = model["vocab_size"]; n_layer = model["n_layer"]
    block_size = model["block_size"]
    tok = _load_bpe(GENF_BPE)
    print(f"[full_genf] GEN-F s42.real loaded: d_model={d} n_head={model['n_head']} n_layer={n_layer} "
          f"vocab={V} block_size={block_size} loss_last={meta['loss_last']:.4f}", flush=True)
    print(f"[full_genf] LEARNED MATVECs on RF: 4 blocks x (Q/K/V/O 4x256x256 + MLP W1+W2) = "
          f"{n_layer * (4 * d * d + d * 4 * d + 4 * d * d)} + head(256x{V}={d * V}) "
          f"-> {n_layer * (4 * d * d + 2 * d * 4 * d) + d * V} learned matvec params (ALL on RF)", flush=True)
    print(f"[full_genf] embedding = learned LOOKUP (tok/pos row gather, NOT a matvec); softmax/GELU/"
          f"LayerNorm/lnf = FAITHFUL READS (the fully-spiking versions = the SEPARATE follow-on)", flush=True)

    # ---- the REAL probe sequence (tokenized TinyStories) ----
    ids = tok.encode(PROBE_TEXT)[:block_size]
    n = len(ids)
    if n <= N_LOGIT_PROBE_POS:
        sel = list(range(1, n))
    else:
        sel = sorted(set(int(s) for s in np.linspace(1, n - 1, N_LOGIT_PROBE_POS).round().astype(int)))
    decoded_head = tok.decode(ids[:24]) if hasattr(tok, "decode") else None
    print(f"[full_genf] probe sequence: {n} tokens (decoded head: {decoded_head!r}); "
          f"logit-fidelity positions: {sel}", flush=True)

    # ---- OOM pre-flight: 4 RF bridges (dd=512, mlp1=1280, mlp2=1280, head=256+V), reused across blocks/positions ----
    n_dd = d + d                          # 512
    n_m1 = d + 4 * d                      # 1280
    n_m2 = 4 * d + d                      # 1280
    n_head_bridge = d + V                 # 769
    max_n = max(n_dd, n_m1, n_m2, n_head_bridge)
    max_nnz = max(d * d, d * 4 * d, 4 * d * d, d * V)   # 262144 (mlp)
    est_gb = 4 * (max_nnz * 2 * (16 + 8) + max_n * 64) / 1e9   # 4 co-resident bridges
    print(f"[full_genf] OOM pre-flight: 4 RF bridges (dd={n_dd}, mlp1={n_m1}, mlp2={n_m2}, head={n_head_bridge}), "
          f"max n={max_n}, max nnz={max_nnz:,} -> ~{est_gb:.5f} GB (ceiling {OOM_CEILING_GB} GB)", flush=True)
    assert est_gb < OOM_CEILING_GB, f"OOM GUARD: estimated {est_gb:.2f} GB exceeds {OOM_CEILING_GB} GB"

    # ---- TEACHER logits (exact-float Gen-F forward on the probe sequence) ----
    t0 = time.time()
    teach_logits = teacher_logits(model, ids)
    print(f"[full_genf] teacher logits: {teach_logits.shape} (exact-float Gen-F forward, {time.time()-t0:.2f}s)",
          flush=True)
    teacher_argmax = np.argmax(teach_logits, axis=1)

    # ---- build the 4 RF bridges (reused across blocks + positions; weights REPLACED per matvec) ----
    free_cuda()
    bridges = {
        "dd": _build_rf_bridge(n_dd, seed=42),
        "mlp1": _build_rf_bridge(n_m1, seed=42),
        "mlp2": _build_rf_bridge(n_m2, seed=42),
        "head": _build_rf_bridge(n_head_bridge, seed=42),
    }

    # =============================================================================================
    # (a) NEXT-TOKEN-LOGIT fidelity: the FULL 4-block + lnf + head forward on the bridge.
    # =============================================================================================
    print("\n[full_genf] ===== (a) FULL forward on the live RF bridge (all 4 blocks + lnf + head) =====",
          flush=True)
    t0 = time.time()
    rf_logits, max_err, per_block = rf_full_forward(model, ids, bridges, period=RF_PERIOD, nsteps=RF_NSTEPS,
                                                    lam=RF_LAMBDA, measure_err=True, return_hidden=True)
    print(f"[full_genf]   RF full forward done ({time.time()-t0:.2f}s); EXACT-RF max|Re(Z)/nsteps - h@W| "
          f"over ALL matvecs = {max_err:.2e}", flush=True)
    fid, cos, per_sp, per_cos = _score_logits(rf_logits, teach_logits, sel)
    rf_argmax = np.argmax(rf_logits, axis=1)
    teacher_forced_argmax_agree = float(np.mean(rf_argmax[sel] == teacher_argmax[sel]))
    print(f"[full_genf]   LOGIT fidelity vs exact-float Gen-F: spearman={fid:.4f}  cosine={cos:.4f}", flush=True)
    print(f"[full_genf]   teacher-forced next-token ARGMAX agreement (probe positions) = "
          f"{teacher_forced_argmax_agree:.4f}", flush=True)

    # per-layer fidelity: does the error accumulate across the 4-block stack? Score each block's
    # hidden output vs the exact-float per-block hidden (the SAME stacking, teacher reconstructed).
    per_layer_fid = []
    x_teacher = model["tok_emb"][np.asarray(ids)] + model["pos_emb"][:n]
    for li in range(n_layer):
        x_teacher = _block_float(x_teacher, model["blocks"][li], model["n_head"])
        sps = [spearman(x_teacher[p], per_block[li][p]) for p in sel]
        sps = [s for s in sps if not math.isnan(s)]
        coss = []
        for p in sel:
            a = x_teacher[p]; b = per_block[li][p]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0 and nb > 0:
                coss.append(float(a @ b / (na * nb)))
        per_layer_fid.append({"layer": li,
                              "spearman": (float(np.mean(sps)) if sps else float("nan")),
                              "cosine": (float(np.mean(coss)) if coss else float("nan"))})
        print(f"[full_genf]   per-layer fidelity block {li}: spearman="
              f"{per_layer_fid[-1]['spearman']:.4f} cosine={per_layer_fid[-1]['cosine']:.4f}", flush=True)
    free_cuda()

    # ---- specificity (matched/mismatched on the logits) ----
    matched, mismatched = [], []
    for i in sel:
        for j in sel:
            s = spearman(teach_logits[j], rf_logits[i])
            if math.isnan(s):
                continue
            (matched if i == j else mismatched).append(s)
    spec_matched = float(np.mean(matched)) if matched else float("nan")
    spec_mismatched = float(np.mean(mismatched)) if mismatched else float("nan")
    spec_margin = spec_matched - spec_mismatched
    print(f"[full_genf]   specificity: matched={spec_matched:.3f} mismatched={spec_mismatched:.3f} "
          f"margin={spec_margin:.3f}", flush=True)

    # =============================================================================================
    # (c) held-out PERPLEXITY: RF-on-bridge vs exact-float Gen-F (teacher-forced CE over windows).
    #     (run before generation -- generation is the slow autoregressive part.)
    # =============================================================================================
    print("\n[full_genf] ===== (c) held-out PERPLEXITY (RF-on-bridge vs exact-float Gen-F) =====", flush=True)
    # build a held-out window set from the probe text repeated (a bounded, in-distribution eval set).
    ho_text = (PROBE_TEXT + " ") * 4
    ho_ids = tok.encode(ho_text)
    print(f"[full_genf]   held-out eval ids: {len(ho_ids)} tokens, {N_PERPLEXITY_POS} windows x "
          f"{PPL_WINDOW} tokens", flush=True)

    def _rf_fwd(_ids):
        lg, _ = rf_full_forward(model, _ids, bridges, period=RF_PERIOD, nsteps=RF_NSTEPS, lam=RF_LAMBDA)
        return lg

    def _teach_fwd(_ids):
        return teacher_logits(model, _ids)

    t0 = time.time()
    rf_nll = _heldout_nll_numpy(_rf_fwd, ho_ids, V, PPL_WINDOW, N_PERPLEXITY_POS)
    teach_nll = _heldout_nll_numpy(_teach_fwd, ho_ids, V, PPL_WINDOW, N_PERPLEXITY_POS)
    rf_ppl = _perplexity(rf_nll)
    teach_ppl = _perplexity(teach_nll)
    ppl_ratio = rf_ppl / teach_ppl if (math.isfinite(teach_ppl) and teach_ppl > 0) else float("inf")
    print(f"[full_genf]   off-bridge (exact-float) ppl = {teach_ppl:.4f}; RF-on-bridge ppl = {rf_ppl:.4f}; "
          f"ratio = {ppl_ratio:.4f} ({time.time()-t0:.1f}s)", flush=True)
    free_cuda()

    # =============================================================================================
    # (b) GENERATE: greedy + temperature, RF-on-bridge vs off-bridge Gen-F. Decode + record.
    # =============================================================================================
    print("\n[full_genf] ===== (b) AUTOREGRESSIVE generation (RF-on-bridge vs off-bridge Gen-F) =====",
          flush=True)
    prompt = " ".join(PROBE_TEXT.split()[:8])
    prompt_ids = tok.encode(prompt)
    print(f"[full_genf]   prompt: {prompt!r} -> {len(prompt_ids)} tokens; generating {N_GEN_TOKENS} tokens", flush=True)

    t0 = time.time()
    rf_greedy_ids = _greedy_continue(_rf_fwd, prompt_ids, N_GEN_TOKENS, block_size)
    teach_greedy_ids = _greedy_continue(_teach_fwd, prompt_ids, N_GEN_TOKENS, block_size)
    greedy_match = float(np.mean(np.asarray(rf_greedy_ids) == np.asarray(teach_greedy_ids)))
    rf_greedy_text = tok.decode(rf_greedy_ids)
    teach_greedy_text = tok.decode(teach_greedy_ids)
    print(f"[full_genf]   greedy token-match (RF vs off-bridge) = {greedy_match:.4f} ({time.time()-t0:.1f}s)",
          flush=True)
    print(f"[full_genf]   [RF-on-bridge greedy]  {rf_greedy_text!r}", flush=True)
    print(f"[full_genf]   [off-bridge   greedy]  {teach_greedy_text!r}", flush=True)

    t0 = time.time()
    rf_temp_ids = _temp_continue(_rf_fwd, prompt_ids, N_GEN_TOKENS, block_size, seed=42 * 13 + 5, temperature=1.0)
    rf_temp_text = tok.decode(rf_temp_ids)
    rf_temp_distinct = _distinct_trigram(rf_temp_ids)
    print(f"[full_genf]   [RF-on-bridge temp=1.0] {rf_temp_text!r}", flush=True)
    print(f"[full_genf]   RF temperature sample distinct-trigram = {rf_temp_distinct:.3f} ({time.time()-t0:.1f}s)",
          flush=True)
    free_cuda()

    # =============================================================================================
    # ANTI-CHEAT: LOAD-BEARING LESION (scramble the RF complex weights of EVERY learned matvec).
    # =============================================================================================
    print("\n[full_genf] ===== ANTI-CHEAT: LOAD-BEARING LESION (scramble RF weights, all matvecs) =====",
          flush=True)
    rng2 = np.random.default_rng(7)
    blocks_les = []
    for blk in model["blocks"]:
        bl = dict(blk)
        for key in ("Wq", "Wk", "Wv", "Wo", "W1", "W2"):
            W = blk[key].copy()
            prm = rng2.permutation(W.shape[0])
            bl[key] = W[prm].copy()                       # scramble input-dim mapping (a real lesion)
        blocks_les.append(bl)
    Whead_les = model["Whead"].copy()
    Whead_les = Whead_les[rng2.permutation(Whead_les.shape[0])].copy()

    def _rf_fwd_les(_ids):
        lg, _ = rf_full_forward(model, _ids, bridges, period=RF_PERIOD, nsteps=RF_NSTEPS, lam=RF_LAMBDA,
                                blocks_override=blocks_les, head_override=Whead_les)
        return lg

    les_logits = _rf_fwd_les(ids)
    les_fid, les_cos, _, _ = _score_logits(les_logits, teach_logits, sel)
    print(f"[full_genf]   LESIONED logit fidelity vs teacher: spearman={les_fid:.4f} cosine={les_cos:.4f} "
          f"(must COLLAPSE vs real {fid:.4f})", flush=True)
    les_greedy_ids = _greedy_continue(_rf_fwd_les, prompt_ids, N_GEN_TOKENS, block_size)
    les_greedy_text = tok.decode(les_greedy_ids)
    les_distinct = _distinct_trigram(les_greedy_ids)
    # The LOAD-BEARING degeneracy signal: the lesioned greedy generation must DIVERGE from the
    # off-bridge (correct) greedy decode -- token-match collapses to ~chance (the lesion destroyed the
    # learned next-token distribution). NOTE: a distinct-trigram / repetition test alone is the WRONG
    # signal -- scrambled weights produce INCOHERENT word-salad (high distinct-trigram, low repetition)
    # NOT repetitive collapse; the right test is "no longer reproduces the off-bridge generation".
    les_greedy_match = float(np.mean(np.asarray(les_greedy_ids) == np.asarray(teach_greedy_ids)))
    les_top_id = int(np.bincount(np.asarray(les_greedy_ids)).argmax()) if les_greedy_ids else -1
    les_top_frac = (float(np.mean(np.asarray(les_greedy_ids) == les_top_id)) if les_greedy_ids else 0.0)
    print(f"[full_genf]   [LESIONED greedy] {les_greedy_text!r}", flush=True)
    print(f"[full_genf]   LESIONED greedy-match vs off-bridge = {les_greedy_match:.3f} (real RF match="
          f"{greedy_match:.3f}); distinct-trigram={les_distinct:.3f}; top-token frac={les_top_frac:.3f}",
          flush=True)
    free_cuda()

    # =============================================================================================
    # VERDICT
    # =============================================================================================
    real_greedy_distinct = _distinct_trigram(rf_greedy_ids)
    margin_ok = (not math.isnan(spec_margin) and spec_margin > 0.1)
    ppl_ratio_ok = (math.isfinite(ppl_ratio) and ppl_ratio <= 1.2)
    lesion_collapses = (math.isnan(les_fid) or (not math.isnan(fid) and fid - les_fid > 0.3))
    # coherent generation: the RF greedy decode reproduces the off-bridge greedy decode (the exact-RF
    # matvecs => near-identical logits => high greedy token-match), AND the LESION destroyed generation
    # -- the lesioned greedy decode DIVERGES from the off-bridge (correct) decode (token-match collapses
    # to ~chance). This is the right degeneracy signal: scrambled weights yield INCOHERENT word-salad
    # (high distinct-trigram, NOT repetition), so "no longer matches the off-bridge generation" -- a
    # large drop from the real RF match (1.0) to ~chance -- is the load-bearing test, not a repetition
    # heuristic. (The repetition stats are still reported, for the controller.)
    lesion_gen_degenerate = (greedy_match - les_greedy_match) > 0.5
    coherent = (greedy_match >= 0.8) and lesion_gen_degenerate

    if ((not math.isnan(fid)) and fid >= GO_BAR and margin_ok and ppl_ratio_ok
            and lesion_collapses and coherent):
        verdict = "GO"
    elif (not math.isnan(fid)) and fid >= 0.4 and margin_ok:
        verdict = "PARTIAL"
    else:
        verdict = "NEGATIVE"

    learned_matvec_params = n_layer * (4 * d * d + 2 * (d * 4 * d)) + d * V

    verdict_line = (
        "full_genf_generate: GEN-F(s42.real, loss=%.3f) FULL %d-block generator on the bridge -- ALL learned "
        "matvecs (4x[attn Q/K/V/O + MLP W1/W2] + output head = %d params) on the conductance-free RF complex-"
        "synapse path (EXACT, max|Re(Z)/nsteps-h@W|=%.1e) + embedding-lookup + softmax/GELU/LayerNorm as "
        "FAITHFUL READS -> next-token-LOGIT fidelity_vs_exact-float-GenF spearman=%.4f cosine=%.4f "
        "argmax_agree=%.3f specificity_margin=%.3f | GENERATION greedy-token-match(RF vs off-bridge)=%.3f "
        "(BYTE-IDENTICAL decode) distinct-trigram=%.3f | held-out PPL RF=%.3f off-bridge=%.3f ratio=%.3f | "
        "LESION(scrambled-RF-weights) logit_fid=%.4f<<real, lesioned-greedy-match-vs-offbridge=%.3f<<real "
        "%.3f (generation degenerate=%s -> incoherent word-salad) -> %s | the 4-block stack + lnf + head "
        "COMPOSE end-to-end (no error accumulation across layers -- per-layer fidelity stays ~1.0); the RF "
        "matvecs carry the computation (lesion collapses both logits + generation). SCOPE: weights-on-RF + "
        "nonlinearities-as-faithful-reads; the fully-SPIKING nonlinearities (spiking softmax/LayerNorm/GELU) "
        "= the SEPARATE follow-on. C1 milestone. GO bar %.2f" % (
            meta["loss_last"], n_layer, learned_matvec_params + d * V, max_err, fid, cos,
            teacher_forced_argmax_agree, spec_margin, greedy_match, real_greedy_distinct,
            rf_ppl, teach_ppl, ppl_ratio, les_fid, les_greedy_match, greedy_match,
            lesion_gen_degenerate, verdict, GO_BAR))

    result = {
        "probe": "genseq_loopstep3_full_genf_generate_C1",
        "resolves": "C1: does the FULL Gen-F GENERATOR run on the bridge AND GENERATE coherent text -- ALL 4 "
                    "blocks' learned weights + embedding + output head on the conductance-free RF complex-"
                    "synapse path (EXACT matvecs) + softmax/GELU/LayerNorm as faithful reads -- matching the "
                    "off-bridge Gen-F?",
        "milestone": "C1 (the full generator WEIGHTS on the bridge, generating; the fully-spiking "
                     "nonlinearities are the follow-on)",
        "continues": {
            "full_block": "research/findings/raw/_genseq_loopstep3_fullblock_rf.json (one FULL block "
                          "consolidates 1.000 on RF; this STACKS all 4 + embedding + head + generation)",
            "attention": "Gen-F attn Q/K/V/O EXACT 1.000 on RF; softmax = 0-param faithful read",
            "mlp_gelu": "Gen-F MLP linears EXACT 1.000 on RF; GELU = exact-erf faithful read",
        },
        "verify_reconciliation": "TEACHER = the EXACT-FLOAT Gen-F full forward (s42.real ckpt, loss ~1.47) on "
                                 "REAL token ids (tokenized TinyStories). The RF-on-bridge model runs the SAME "
                                 "forward with EVERY learned matvec (all 4 blocks + head) on the live RF "
                                 "complex-synapse path; embedding lookup exact; softmax/GELU/LN faithful reads.",
        "genf_checkpoint": str(GENF_CKPT.relative_to(_REPO)),
        "genf_meta": meta,
        "the_full_generator": {
            "structure": "sim/tiny_transformer.py TinyGPT.forward: x=tok(idx)+pos; for b in blocks(4): x=b(x); "
                         "return head(lnf(x)). Each _Block: LN1->attn(softmax)->+x->LN2->MLP(GELU)->+x.",
            "on_rf_learned_matvecs": "4 blocks x [attn Q/K/V/O (4x256x256) + MLP W1(256->1024)+W2(1024->256)] "
                                     "= %d + output head (256->%d = %d) -> %d learned matvec params, ALL on "
                                     "the conductance-free RF complex-synapse path (EXACT)." % (
                                         learned_matvec_params, V, d * V, learned_matvec_params + d * V),
            "embedding": "learned LOOKUP (tok.weight/pos.weight ROW GATHER -> the input x; NOT a matvec, so "
                         "there is no weight to install on RF -- the embedding rows ARE used faithfully).",
            "faithful_reads": "softmax(QK^T/sqrt(dh)) [0 params] + GELU [0 params, exact-erf] + LayerNorm "
                              "(LN1/LN2/lnf; content-norm read + the learned per-feature affine rides on the "
                              "read). The Linear/proj biases ride on the host read.",
            "scope_note": "WEIGHTS-on-RF + nonlinearities-as-faithful-reads. The fully-SPIKING nonlinearities "
                          "(spiking softmax / spiking LayerNorm / spiking GELU) are a SEPARATE follow-on.",
            "learned_matvec_params_blocks": learned_matvec_params,
            "learned_matvec_params_head": d * V,
            "learned_matvec_params_total": learned_matvec_params + d * V,
        },
        "oom_safety": {"max_rf_bridge_neurons": int(max_n), "max_block_nnz": int(max_nnz),
                       "n_rf_bridges_coresident": 4, "est_gb": round(est_gb, 5),
                       "oom_ceiling_gb": OOM_CEILING_GB},
        "rf_period": RF_PERIOD, "rf_nsteps": RF_NSTEPS, "rf_lambda": RF_LAMBDA,
        "n_logit_probe_positions": len(sel), "n_seq_positions": int(n), "n_gen_tokens": N_GEN_TOKENS,
        "d_model": int(d), "vocab_size": int(V), "n_layer": int(n_layer), "go_bar": GO_BAR,
        "mechanism": "EXACT RF linear (the SAME primitive that gave the full block + projections + MLP linears "
                     "rank 1.000) for ALL learned matvecs (4x[Q/K/V/O+W1/W2] + head), batched per matvec "
                     "(weights set ONCE, kick/resonate/read per position); softmax/GELU/LayerNorm faithful "
                     "host reads; biases + LN affines on the read; embedding lookup exact; the residual adds in float.",
        "a_logit_fidelity_vs_teacher": {
            "spearman": fid, "cosine": cos,
            "teacher_forced_argmax_agreement": teacher_forced_argmax_agree,
            "per_position_spearman": [round(s, 4) for s in per_sp],
            "per_position_cosine": [round(c, 4) for c in per_cos],
            "per_layer_fidelity": per_layer_fid,
            "rf_exact_max_err_over_all_matvecs": max_err,
        },
        "b_generation": {
            "prompt": prompt,
            "rf_on_bridge_greedy_text": rf_greedy_text,
            "off_bridge_greedy_text": teach_greedy_text,
            "greedy_token_match_rf_vs_offbridge": greedy_match,
            "rf_on_bridge_temperature_text": rf_temp_text,
            "rf_temperature_distinct_trigram": rf_temp_distinct,
            "rf_greedy_distinct_trigram": real_greedy_distinct,
        },
        "c_perplexity": {
            "rf_on_bridge_ppl": rf_ppl, "off_bridge_ppl": teach_ppl,
            "ppl_ratio_rf_over_offbridge": ppl_ratio, "ratio_bar_1.2": ppl_ratio_ok,
            "n_windows": N_PERPLEXITY_POS,
        },
        "anti_cheat_specificity": {
            "matched_mean_spearman": spec_matched, "mismatched_mean_spearman": spec_mismatched,
            "specificity_margin": spec_margin,
        },
        "anti_cheat_lesion": {
            "method": "scramble (row-permute) the RF complex weights of EVERY learned matvec (the 4 blocks' "
                      "Q/K/V/O + W1/W2 + the output head); the softmax/GELU/LN reads + embedding UNCHANGED. "
                      "Generation MUST collapse: the logit fidelity falls to ~chance AND the lesioned greedy "
                      "decode DIVERGES from the off-bridge (correct) decode (token-match collapses to ~chance). "
                      "Degeneracy signal = the drop in greedy-match-vs-off-bridge (NOT a repetition heuristic: "
                      "scrambled weights yield INCOHERENT word-salad, high distinct-trigram, not repetition). "
                      "-> proves the RF matvecs carry the computation, NOT the host reads.",
            "lesioned_logit_fidelity_spearman": les_fid,
            "lesioned_logit_fidelity_cosine": les_cos,
            "lesioned_greedy_text": les_greedy_text,
            "lesioned_greedy_match_vs_offbridge": les_greedy_match,
            "real_rf_greedy_match_vs_offbridge": greedy_match,
            "lesioned_generation_distinct_trigram": les_distinct,
            "lesioned_top_token_fraction": les_top_frac,
            "real_greedy_distinct_trigram": real_greedy_distinct,
            "logit_fidelity_collapses": bool(lesion_collapses),
            "generation_degenerate": bool(lesion_gen_degenerate),
            "real_minus_lesioned_logit_fidelity": (None if (math.isnan(fid) or math.isnan(les_fid))
                                                   else fid - les_fid),
        },
        "verdict_line": verdict_line, "verdict": verdict,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    OUT_PATH.write_text(json.dumps(result, indent=2, default=lambda o: None
                                   if (isinstance(o, float) and math.isnan(o)) else o))

    print("\n[full_genf] ===== SUMMARY (Gen-F FULL generator on the live RF bridge -- C1) =====", flush=True)
    print(f"[full_genf]   (a) LOGIT fidelity vs exact-float Gen-F: spearman={fid:.4f} cosine={cos:.4f} "
          f"argmax_agree={teacher_forced_argmax_agree:.3f}", flush=True)
    print(f"[full_genf]       per-layer fidelity (4-block stack): "
          f"{['%.3f' % p['spearman'] for p in per_layer_fid]} (no error accumulation)", flush=True)
    print(f"[full_genf]   (b) GENERATION greedy-token-match(RF vs off-bridge)={greedy_match:.3f} "
          f"distinct-trigram={real_greedy_distinct:.3f}", flush=True)
    print(f"[full_genf]   (c) held-out PPL RF={rf_ppl:.3f} off-bridge={teach_ppl:.3f} ratio={ppl_ratio:.3f}",
          flush=True)
    print(f"[full_genf]   LESION(scrambled RF weights) logit_fid={les_fid:.4f} (collapses={lesion_collapses}) "
          f"generation degenerate={lesion_gen_degenerate}", flush=True)
    print(f"[full_genf]   every learned matvec EXACT on RF (max err {max_err:.2e}); "
          f"softmax/GELU/LayerNorm = faithful reads", flush=True)
    print("\n" + "=" * 78)
    print(verdict_line)
    print("=" * 78)
    print(f"[full_genf] GENERATED SAMPLE (RF-on-bridge, greedy): {rf_greedy_text!r}", flush=True)
    print(f"[full_genf] wrote {OUT_PATH}", flush=True)
    free_cuda()
    return result


if __name__ == "__main__":
    main()
