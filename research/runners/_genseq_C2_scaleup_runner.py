"""GENERATIVE-SEQUENCE FRONTIER (Spine A) -- C2 SCALE-UP runner: does a BIGGER generator DEMONSTRATE
the grow-without-forget LOOP that the 3.4M toy could not (a model-CAPACITY wall)?

WHY THIS RUNNER EXISTS (read research/findings/2026-06-23-C2-scaleup-scoping.md +
research/findings/2026-06-23-C2-moderate-shift-NEGATIVE-scale-wall.md first):
  The C2 grow-no-forget MECHANISM (generative self-replay, dose-monotone) AND the C1 RF consolidation
  are BOTH validated on the 3.4M toy. The ONLY thing that fails is the SCALE: at a moderate learnable
  SH_FRAC=0.45 shift the 3.4M best-replay arm retained only 53.9% (bar >=85%) -- it cannot hold two
  distributions at once because it is near its capacity ceiling for ONE. The scoping sizes the
  smallest demonstrating model at ~100M, with 30M the CHEAP FIRST PROBE (~4 h local, likely PARTIAL
  -- retention ~75-85%, learns-new + clears the forgetting contrast -- which already confirms the
  scale hypothesis directionally and de-risks the 100M decisive run).

THE BUILD (reuse the existing pipeline VERBATIM; ONLY arch hyperparams + FT_BATCH change, per the
scoping's "the larger model is a drop-in to the same loop -- no sim/ edit, no new mechanism"):
  Gen-F = sim/tiny_transformer.py TinyGPT. The 3.4M toy is d=256/L=4/H=4/V=513/block=128. For ~30M the
  scoping's config is d=512/L=8/H=8/V=2048/block=128 = 27.38M params (the EXACT count is computed +
  printed by this runner; tune --d-model/--n-layers to land near 30M). KEEP block=128, TinyStories.

  Three stages, each gated on a completion marker (RESUMABLE -- a long run that is interrupted resumes
  from the last completed stage; stage 1's TRAIN is itself atomic-checkpoint resumable inside
  train_tiny_gpt, the dominant ~4 h cost):
    (1) TRAIN the bigger Gen-F on TinyStories -> checkpoint (research.runners.tiny_transformer_train
        .train_tiny_gpt, reuse-by-import; it does atomic .pt + os.replace + resume).
    (2) C1-CONSOLIDATE the trained model onto the RF complex-synapse bridge + VERIFY generation: the
        EXACT C1 install (rf_full_forward) -> next-token-logit fidelity + greedy-token-match vs the
        off-bridge model + held-out ppl_ratio (reuse-by-import of the C1 derisk's machinery).
    (3) the C2 grow+no-forget MODERATE-SHIFT loop (research.runners._genseq_C2_moderate_shift_derisk
        .run_c2_loop, reuse-by-import) on the trained bigger frozen Gen-F -> the dose-response /
        no-replay control / on-bridge verify / GO|PARTIAL|NEGATIVE + the SCALE-issue flag.

SMOKE (--smoke; the ONLY GPU run intended here, a wiring smoke NOT a measurement): instantiate the
bigger model, print the EXACT param count + the VRAM used, run ~5 TRAIN steps (confirm loss
decreasing), and a 1-step DRY-RUN of the C1 + C2 stages (confirm wired) -- NO full training. The full
~4 h run is launched SEPARATELY by the controller (see the printed command).

NO sim/ edit (the whole pipeline is reuse-by-import; the only code touch is this orchestrator + the
additive parameterization of the C2 runner's load_genf/run_c2_loop -- behaviourally unchanged for the
original 3.4M main()). GPU. Usage (FULL run, controller-launched):
  SIM_BACKEND=cupy python -m research.runners._genseq_C2_scaleup_runner \
      --d-model 512 --n-layers 8 --n-heads 8 --steps 30000 \
      --out research/findings/raw/_genseq_C2_scaleup_30M.json
Usage (1-step wiring SMOKE only):
  SIM_BACKEND=cupy python -m research.runners._genseq_C2_scaleup_runner --smoke \
      --d-model 512 --n-layers 8 --n-heads 8
"""
from __future__ import annotations

import argparse
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

# ---- reuse-by-import: the trainer, the C1 install + scoring, the C2 loop body, the gate generate ----
from research.runners.tiny_transformer_train import train_tiny_gpt  # noqa: E402
from research.runners.corpus_fetch import fetch_corpus, split_corpus  # noqa: E402
from research.runners._genseq_C2_moderate_shift_derisk import run_c2_loop, load_genf  # noqa: E402

# default arch (the scoping's "30M" probe; the EXACT param count is computed + printed at runtime).
DEF_D_MODEL = 512
DEF_N_LAYERS = 8
DEF_N_HEADS = 8
DEF_VOCAB = 2048
DEF_BLOCK = 128
DEF_STEPS = 3500          # ~6 epochs of the ~1.8M-token TinyStories-valid corpus at B=24/T=128 (~10.8M
                          # tok). The corpus is DATA-bound (only ~1.8M unique tokens) -> the budget is set
                          # by epochs, NOT Chinchilla's 20x-param token count (which would be ~50 epochs of
                          # REPETITION = memorization; see the FIRST-run diagnosis below). 3000-5000 is the
                          # convergence band; with dropout+wd the held-out ppl bottoms ~6 around here.
DEF_BATCH = 24            # TRAIN batch (OOM-safe on a 3090 at d=512; auto-halves on OOM in the trainer)
DEF_FT_BATCH = 32         # C2 fine-tune batch (the scoping's 30M FT_BATCH; < 3.4M's 48 for VRAM headroom)
DEF_LR = 3e-4
# --- regularization / schedule defaults (size-aware) -------------------------------------------------
# WHY (2026-06-23 diagnosis of the failed 30M run): the FIRST 30M run reused the 3.4M toy's config --
# dropout=0, no warmup, 30000 steps. At batch 24 x block 128 that is ~92M tokens = ~50 EPOCHS over the
# ~1.8M-token TinyStories-valid corpus. A 27M-param model with ZERO regularization MEMORIZED the train
# set (train loss 0.16 ~ ppl 1.18) while held-out ppl was 95.4 -- a classic overfit, NOT undertraining
# (the 3.4M reached held-out 6.1 only because it LACKS the capacity to memorize). The fix is to (a) cap
# the epoch budget (DEF_STEPS below = ~6 epochs, not ~50), (b) add dropout, (c) add weight decay, and
# (d) warm the LR up. All ADDITIVE; the 3.4M toy path (train_tiny_gpt defaults) is byte-unchanged.
DEF_DROPOUT = 0.1
DEF_WEIGHT_DECAY = 0.1
DEF_WARMUP = 300
DEF_HELDOUT_EVERY = 500   # in-loop overfit probe cadence (0=off)

# stage artifacts (all under one run-id dir so the run is RESUMABLE + self-contained).
RAW = _REPO / "research/findings/raw"


def free_cuda():
    gc.collect()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def tinygpt_param_count(vocab_size, d_model, n_layer, block_size):
    """EXACT TinyGPT parameter count (matches sim/tiny_transformer.py module-for-module):
      tok emb V*d + pos emb block*d
      per block: ln1(2d)+ln2(2d) + attn[in_proj 3d*d + 3d, out_proj d*d + d] + mlp[(d*4d+4d)+(4d*d+d)]
      lnf 2d + head V*d (no bias)."""
    d = int(d_model); V = int(vocab_size); L = int(n_layer); blk = int(block_size)
    emb = V * d + blk * d
    per_block = (2 * d + 2 * d) + (3 * d * d + 3 * d) + (d * d + d) + (d * 4 * d + 4 * d) + (4 * d * d + d)
    lnf = 2 * d
    head = V * d
    return emb + L * per_block + lnf + head


def measured_param_count(model):
    return int(sum(p.numel() for p in model.parameters()))


# =================================================================================================
# STAGE 1: train the bigger Gen-F on TinyStories -> checkpoint. RESUMABLE (train_tiny_gpt atomic-
# checkpoints every print_every steps + resumes from the .pt; we additionally write a stage marker).
# =================================================================================================
def stage1_train(run_dir, *, seed, d_model, n_layer, n_head, vocab_size, block_size, steps, batch_size,
                 lr, device, corpus_path, dropout=0.1, weight_decay=0.1, warmup_steps=300,
                 heldout_path=None, heldout_every=0):
    ckpt_path = str(run_dir / "genf.ckpt")
    bpe_path = str(run_dir / "genf.bpe.json")
    marker = run_dir / "stage1_train.DONE.json"
    if marker.exists():
        print(f"[scaleup] STAGE 1 already DONE (marker {marker.name}); skipping train.", flush=True)
        return ckpt_path, bpe_path, json.loads(marker.read_text())
    print("\n[scaleup] ===== STAGE 1: TRAIN the bigger Gen-F on TinyStories =====", flush=True)
    print(f"[scaleup] reg/sched: dropout={dropout} weight_decay={weight_decay} warmup_steps={warmup_steps} "
          f"lr={lr} steps={steps} batch={batch_size}", flush=True)
    t0 = time.time()
    rr = train_tiny_gpt(
        seed=seed, corpus_path=corpus_path, vocab_size=vocab_size, d_model=d_model,
        n_layer=n_layer, n_head=n_head, block_size=block_size, steps=steps, batch_size=batch_size,
        lr=lr, ckpt_path=ckpt_path, bpe_path=bpe_path, device=device, print_every=500, verbose=True,
        dropout=dropout, weight_decay=weight_decay, warmup_steps=warmup_steps,
        heldout_path=heldout_path, heldout_every=heldout_every)
    info = {"final_loss": rr.get("final_loss"), "initial_loss": rr.get("initial_loss"),
            "vocab_size": rr.get("vocab_size"), "steps": steps, "interrupted": rr.get("interrupted"),
            "dropout": dropout, "weight_decay": weight_decay, "warmup_steps": warmup_steps, "lr": lr,
            "final_heldout_ppl": rr.get("final_heldout_ppl"),
            "heldout_history": rr.get("heldout_history"),
            "elapsed_seconds": round(time.time() - t0, 1)}
    if rr.get("interrupted"):
        print("[scaleup] STAGE 1 INTERRUPTED (checkpoint flushed); re-run to resume (NO marker written).",
              flush=True)
        return ckpt_path, bpe_path, info
    marker.write_text(json.dumps(info, indent=2, default=str))
    print(f"[scaleup] STAGE 1 DONE: final_loss={info['final_loss']} ({info['elapsed_seconds']}s)", flush=True)
    return ckpt_path, bpe_path, info


# =================================================================================================
# STAGE 2: C1-consolidate the trained model onto the RF complex-synapse bridge + verify generation.
# Reuse the C1 derisk's EXACT install + scoring (rf_full_forward / greedy / heldout-ppl), with the
# bigger model's parameters extracted in the install convention (a_out = a_in @ W). Arch-agnostic.
# =================================================================================================
def stage2_c1_consolidate(run_dir, *, ckpt_path, bpe_path, device, n_head, n_logit_pos, n_gen_tokens,
                          ppl_windows, dry_run=False):
    import torch
    from research.runners._genseq_loopstep3_full_genf_generate_derisk import (
        rf_full_forward, teacher_logits, _greedy_continue, _heldout_nll_numpy, _perplexity, _score_logits,
        _distinct_trigram)
    from research.runners._genseq_loopstep3_rf_probe import _build_rf_bridge, RF_PERIOD, RF_NSTEPS, RF_LAMBDA

    marker = run_dir / "stage2_c1.DONE.json"
    if marker.exists() and not dry_run:
        print(f"[scaleup] STAGE 2 already DONE (marker {marker.name}); skipping.", flush=True)
        return json.loads(marker.read_text())

    print("\n[scaleup] ===== STAGE 2: C1-consolidate the trained Gen-F on the RF bridge + verify gen =====",
          flush=True)
    t0 = time.time()
    # ---- load the trained TinyGPT (frozen) + BPE; extract the whole model in the RF install convention ----
    frozen, tok, V, loss_last = load_genf(device, ckpt_path=ckpt_path + ".pt", bpe_path=bpe_path,
                                          n_head=n_head)
    d = frozen.cfg["d_model"]; n_layer = frozen.cfg["n_layer"]; block_size = frozen.cfg["block_size"]
    print(f"[scaleup]   trained Gen-F loaded: d={d} L={n_layer} H={frozen.cfg['n_head']} V={V} "
          f"block={block_size} loss_last={loss_last:.4f}", flush=True)

    sd = {k: v.detach().to("cpu") for k, v in frozen.state_dict().items()}
    blocks = []
    for li in range(n_layer):
        p = f"blocks.{li}."
        in_w = sd[p + "attn.in_proj_weight"].numpy().astype(np.float64)
        in_b = sd[p + "attn.in_proj_bias"].numpy().astype(np.float64)
        Wq, Wk, Wv = in_w[:d], in_w[d:2 * d], in_w[2 * d:]
        bq, bk, bv = in_b[:d], in_b[d:2 * d], in_b[2 * d:]
        Wo = sd[p + "attn.out_proj.weight"].numpy().astype(np.float64)
        bo = sd[p + "attn.out_proj.bias"].numpy().astype(np.float64)
        W1 = sd[p + "mlp.0.weight"].numpy().astype(np.float64)
        b1 = sd[p + "mlp.0.bias"].numpy().astype(np.float64)
        W2 = sd[p + "mlp.2.weight"].numpy().astype(np.float64)
        b2 = sd[p + "mlp.2.bias"].numpy().astype(np.float64)
        blocks.append({
            "ln1_w": sd[p + "ln1.weight"].numpy().astype(np.float64),
            "ln1_b": sd[p + "ln1.bias"].numpy().astype(np.float64),
            "ln2_w": sd[p + "ln2.weight"].numpy().astype(np.float64),
            "ln2_b": sd[p + "ln2.bias"].numpy().astype(np.float64),
            "Wq": Wq.T.astype(np.float32).copy(), "Wk": Wk.T.astype(np.float32).copy(),
            "Wv": Wv.T.astype(np.float32).copy(), "Wo": Wo.T.astype(np.float32).copy(),
            "bq": bq, "bk": bk, "bv": bv, "bo": bo,
            "W1": W1.T.astype(np.float32).copy(), "W2": W2.T.astype(np.float32).copy(),
            "b1": b1, "b2": b2})
    model_b = {
        "blocks": blocks, "tok_emb": sd["tok.weight"].numpy().astype(np.float64),
        "pos_emb": sd["pos.weight"].numpy().astype(np.float64),
        "lnf_w": sd["lnf.weight"].numpy().astype(np.float64), "lnf_b": sd["lnf.bias"].numpy().astype(np.float64),
        "Whead": sd["head.weight"].numpy().T.astype(np.float32).copy(),
        "n_head": frozen.cfg["n_head"], "n_layer": n_layer, "d_model": d,
        "block_size": block_size, "vocab_size": V}

    # ---- the 4 RF bridges (sizes scale with d, V; OOM pre-flight) ----
    n_dd = d + d; n_m1 = d + 4 * d; n_m2 = 4 * d + d; n_head_bridge = d + V
    max_n = max(n_dd, n_m1, n_m2, n_head_bridge)
    max_nnz = max(d * d, d * 4 * d, 4 * d * d, d * V)
    est_gb = 4 * (max_nnz * 2 * (16 + 8) + max_n * 64) / 1e9
    print(f"[scaleup]   RF bridges dd={n_dd} m1={n_m1} m2={n_m2} head={n_head_bridge} | max_n={max_n} "
          f"max_nnz={max_nnz:,} -> ~{est_gb:.4f} GB", flush=True)
    free_cuda()
    bridges = {"dd": _build_rf_bridge(n_dd, seed=42), "mlp1": _build_rf_bridge(n_m1, seed=42),
               "mlp2": _build_rf_bridge(n_m2, seed=42), "head": _build_rf_bridge(n_head_bridge, seed=42)}

    # in-distribution TinyStories probe text (matches the C1 derisk register).
    probe_text = ("Once upon a time there was a little girl named Lily. She had a small dog and a big "
                  "cat. One day they went to the park to play. The sun was bright and the sky was blue.")
    ids = tok.encode(probe_text)[:block_size]
    n = len(ids)
    n_pos = min(int(n_logit_pos), max(1, n - 1))
    sel = sorted(set(int(s) for s in np.linspace(1, n - 1, n_pos).round().astype(int))) if n > 1 else [0]

    def _rf_fwd(_ids):
        lg, _ = rf_full_forward(model_b, _ids, bridges, period=RF_PERIOD, nsteps=RF_NSTEPS, lam=RF_LAMBDA)
        return lg

    def _teach_fwd(_ids):
        return teacher_logits(model_b, _ids)

    # (a) next-token-logit fidelity (RF-on-bridge vs the exact-float teacher) ----
    teach_logits = teacher_logits(model_b, ids)
    rf_logits, max_err = rf_full_forward(model_b, ids, bridges, period=RF_PERIOD, nsteps=RF_NSTEPS,
                                         lam=RF_LAMBDA, measure_err=True)
    fid, cos, _, _ = _score_logits(rf_logits, teach_logits, sel)
    argmax_agree = float(np.mean(np.argmax(rf_logits, axis=1)[sel] == np.argmax(teach_logits, axis=1)[sel]))
    print(f"[scaleup]   (a) logit fidelity spearman={fid:.4f} cosine={cos:.4f} argmax_agree={argmax_agree:.3f} "
          f"EXACT-RF max|Re(Z)/nsteps-h@W|={max_err:.2e}", flush=True)
    free_cuda()

    # (b) generate (greedy) RF-on-bridge vs off-bridge ----
    prompt_ids = tok.encode(" ".join(probe_text.split()[:8]))
    rf_greedy = _greedy_continue(_rf_fwd, prompt_ids, n_gen_tokens, block_size)
    teach_greedy = _greedy_continue(_teach_fwd, prompt_ids, n_gen_tokens, block_size)
    greedy_match = float(np.mean(np.asarray(rf_greedy) == np.asarray(teach_greedy)))
    rf_greedy_text = tok.decode(rf_greedy)
    print(f"[scaleup]   (b) greedy-token-match(RF vs off-bridge)={greedy_match:.3f} "
          f"distinct-trigram={_distinct_trigram(rf_greedy):.3f}", flush=True)
    print(f"[scaleup]       [RF-on-bridge greedy] {rf_greedy_text!r}", flush=True)
    free_cuda()

    # (c) held-out ppl_ratio (RF-on-bridge vs off-bridge) ----
    ho_text = (probe_text + " ") * 3
    ho_ids = tok.encode(ho_text)
    ppl_w = min(48, block_size)
    rf_ppl = _perplexity(_heldout_nll_numpy(_rf_fwd, ho_ids, V, ppl_w, ppl_windows))
    off_ppl = _perplexity(_heldout_nll_numpy(_teach_fwd, ho_ids, V, ppl_w, ppl_windows))
    ppl_ratio = rf_ppl / off_ppl if (math.isfinite(off_ppl) and off_ppl > 0) else float("inf")
    print(f"[scaleup]   (c) held-out ppl RF={rf_ppl:.4f} off-bridge={off_ppl:.4f} ratio={ppl_ratio:.6f}",
          flush=True)
    del bridges
    free_cuda()

    info = {
        "d_model": int(d), "n_layer": int(n_layer), "vocab_size": int(V), "loss_last": loss_last,
        "logit_fidelity_spearman": fid, "logit_fidelity_cosine": cos,
        "argmax_agreement": argmax_agree, "rf_exact_max_err": max_err,
        "greedy_token_match_rf_vs_offbridge": greedy_match,
        "rf_greedy_text": rf_greedy_text, "off_bridge_greedy_text": tok.decode(teach_greedy),
        "rf_on_bridge_ppl": rf_ppl, "off_bridge_ppl": off_ppl, "ppl_ratio": ppl_ratio,
        "rf_bridges": {"dd": n_dd, "mlp1": n_m1, "mlp2": n_m2, "head": n_head_bridge, "est_gb": round(est_gb, 4)},
        "n_logit_positions": len(sel), "n_gen_tokens": int(n_gen_tokens), "ppl_windows": int(ppl_windows),
        "dry_run": bool(dry_run), "elapsed_seconds": round(time.time() - t0, 1),
    }
    del frozen, sd, model_b
    free_cuda()
    if not dry_run:
        marker.write_text(json.dumps(info, indent=2, default=str))
        print(f"[scaleup] STAGE 2 DONE: logit_fid={fid:.4f} greedy_match={greedy_match:.3f} "
              f"ppl_ratio={ppl_ratio:.6f} ({info['elapsed_seconds']}s)", flush=True)
    else:
        print(f"[scaleup] STAGE 2 DRY-RUN wired OK ({info['elapsed_seconds']}s) -- NO marker written.",
              flush=True)
    return info


# =================================================================================================
# STAGE 3: the C2 grow+no-forget MODERATE-SHIFT loop on the trained bigger frozen Gen-F (reuse the C2
# runner's run_c2_loop verbatim; ONLY ft_batch + out_path + arch_label differ).
# =================================================================================================
def stage3_c2_loop(run_dir, *, ckpt_path, bpe_path, device, n_head, ft_batch, out_path, arch_label,
                   dry_run=False, c2_original="tinystories"):
    marker = run_dir / "stage3_c2.DONE.json"
    if marker.exists() and not dry_run:
        print(f"[scaleup] STAGE 3 already DONE (marker {marker.name}); skipping.", flush=True)
        return json.loads(marker.read_text())
    print(f"\n[scaleup] ===== STAGE 3: the C2 grow+no-forget MODERATE-SHIFT loop (run_c2_loop) "
          f"[c2_original={c2_original}] =====", flush=True)
    frozen, tok, V, loss_last = load_genf(device, ckpt_path=ckpt_path + ".pt", bpe_path=bpe_path,
                                          n_head=n_head)
    res = run_c2_loop(frozen, tok, V, loss_last, device, out_path=out_path, ft_batch=ft_batch,
                      arch_label=arch_label, dry_run=dry_run, c2_original=c2_original)
    del frozen
    free_cuda()
    if not dry_run:
        marker.write_text(json.dumps({"verdict": res.get("verdict"),
                                      "verdict_line": res.get("verdict_line"),
                                      "out_path": str(out_path)}, indent=2, default=str))
    return res


# =================================================================================================
# SMOKE: instantiate the bigger model, print the EXACT param count + VRAM, ~5 train steps (loss down),
# 1-step DRY-RUN of the C1 + C2 stages (confirm wired). NO full training. The ONLY GPU run intended here.
# =================================================================================================
def smoke(args, run_dir, device, corpus_path):
    import torch
    import torch.nn.functional as F
    from sim.tiny_transformer import TinyGPT
    from sim.bpe_tokenizer import BPETokenizer

    print("\n[scaleup] =================== 1-STEP WIRING SMOKE (NO full training) ===================",
          flush=True)
    # ---- BPE: reuse the cached run BPE if present, else train it once on the corpus (needed for stages) ----
    bpe_path = str(run_dir / "genf.bpe.json")
    corpus = Path(corpus_path).read_text(encoding="utf-8")
    if os.path.exists(bpe_path):
        tok = BPETokenizer.load(bpe_path)
        print(f"[scaleup:smoke] loaded cached BPE (vocab {tok.vocab_size}) {bpe_path}", flush=True)
    else:
        tok = BPETokenizer()
        tok.train(corpus, vocab_size=args.vocab_size)
        Path(bpe_path).parent.mkdir(parents=True, exist_ok=True)
        tok.save(bpe_path)
        print(f"[scaleup:smoke] trained BPE (vocab {tok.vocab_size}) -> {bpe_path}", flush=True)
    V = tok.vocab_size

    # ---- EXACT param count (formula + measured) ----
    formula_params = tinygpt_param_count(V, args.d_model, args.n_layers, args.block_size)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    model = TinyGPT(vocab_size=V, d_model=args.d_model, n_layer=args.n_layers, n_head=args.n_heads,
                    block_size=args.block_size, dropout=0.0).to(device)
    measured = measured_param_count(model)
    assert measured == formula_params, (f"param-count formula {formula_params} != measured {measured} "
                                        f"-- the formula is wrong; fix before trusting the sizing.")
    print(f"[scaleup:smoke] arch V={V} d={args.d_model} L={args.n_layers} H={args.n_heads} "
          f"block={args.block_size}", flush=True)
    print(f"[scaleup:smoke] EXACT param count = {measured:,} ({measured/1e6:.2f}M)  "
          f"(formula == measured: {measured == formula_params})", flush=True)
    vram_after_build = (torch.cuda.memory_allocated() / 1e9) if device == "cuda" else 0.0

    # ---- ~5 TRAIN steps (confirm loss decreasing) ----
    data = torch.tensor(tok.encode(corpus[:2_000_000]), dtype=torch.long, device=device)
    nd = data.numel()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    losses = []
    bs = min(args.batch_size, 8)   # tiny smoke batch (the full run uses --batch-size); keep VRAM low
    n_smoke_steps = 5
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    t0 = time.time()
    for step in range(n_smoke_steps):
        ix = torch.randint(0, nd - args.block_size - 1, (bs,), generator=g).to(device)
        x = torch.stack([data[i:i + args.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + args.block_size] for i in ix])
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
        print(f"[scaleup:smoke]   train step {step+1}/{n_smoke_steps} loss={losses[-1]:.4f}", flush=True)
    loss_decreasing = losses[-1] < losses[0]
    vram_peak = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0
    print(f"[scaleup:smoke] 5-step loss: {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"(decreasing={loss_decreasing}, {time.time()-t0:.1f}s)", flush=True)
    print(f"[scaleup:smoke] VRAM: after-build={vram_after_build:.3f} GB  peak(train)={vram_peak:.3f} GB", flush=True)

    # ---- save a TINY checkpoint so the C1+C2 dry-runs have a real .pt to load (resume-shaped) ----
    smoke_ckpt = str(run_dir / "genf_smoke.ckpt")
    Path(smoke_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optim": opt.state_dict(),
                "loss_history": losses, "step": n_smoke_steps}, smoke_ckpt + ".pt")
    del model, opt, data
    free_cuda()

    # ---- 1-step DRY-RUN of STAGE 2 (C1 install) + STAGE 3 (C2 loop) -- confirm WIRED (not measured) ----
    print("\n[scaleup:smoke] --- dry-run STAGE 2 (C1 install + verify gen), 1 window / 2 positions ---",
          flush=True)
    stage2_dry = "n/a"
    try:
        s2 = stage2_c1_consolidate(run_dir, ckpt_path=smoke_ckpt, bpe_path=bpe_path, device=device,
                                   n_head=args.n_heads, n_logit_pos=2, n_gen_tokens=2, ppl_windows=1,
                                   dry_run=True)
        stage2_dry = {"wired": True, "logit_fidelity_spearman": s2["logit_fidelity_spearman"],
                      "greedy_token_match": s2["greedy_token_match_rf_vs_offbridge"],
                      "ppl_ratio": s2["ppl_ratio"], "elapsed_seconds": s2["elapsed_seconds"]}
        print(f"[scaleup:smoke]   STAGE 2 WIRED (logit_fid={s2['logit_fidelity_spearman']:.4f} "
              f"greedy_match={s2['greedy_token_match_rf_vs_offbridge']:.3f} "
              f"ppl_ratio={s2['ppl_ratio']:.4f})", flush=True)
    except Exception as e:
        stage2_dry = {"wired": False, "error": f"{type(e).__name__}: {e}"}
        print(f"[scaleup:smoke]   STAGE 2 DRY-RUN FAILED: {type(e).__name__}: {e}", flush=True)

    print("\n[scaleup:smoke] --- dry-run STAGE 3 (C2 loop), 1 window / 2 arms / 3 FT steps ---", flush=True)
    stage3_dry = "n/a"
    try:
        smoke_out = run_dir / "genf_smoke_c2_dryrun.json"
        s3 = stage3_c2_loop(run_dir, ckpt_path=smoke_ckpt, bpe_path=bpe_path, device=device,
                            n_head=args.n_heads, ft_batch=min(args.ft_batch, 8),
                            out_path=smoke_out, arch_label="SMOKE", dry_run=True,
                            c2_original=args.c2_original)
        stage3_dry = {"wired": True, "verdict": s3.get("verdict"),
                      "arms": list(s3.get("arms", {}).keys()),
                      "elapsed_seconds": s3.get("elapsed_seconds")}
        print(f"[scaleup:smoke]   STAGE 3 WIRED (dry-run verdict={s3.get('verdict')} "
              f"arms={list(s3.get('arms', {}).keys())})", flush=True)
    except Exception as e:
        stage3_dry = {"wired": False, "error": f"{type(e).__name__}: {e}"}
        print(f"[scaleup:smoke]   STAGE 3 DRY-RUN FAILED: {type(e).__name__}: {e}", flush=True)

    smoke_result = {
        "probe": "genseq_C2_scaleup_SMOKE",
        "arch": {"vocab_size": int(V), "d_model": int(args.d_model), "n_layer": int(args.n_layers),
                 "n_head": int(args.n_heads), "block_size": int(args.block_size)},
        "exact_param_count": int(measured), "exact_param_count_millions": round(measured / 1e6, 4),
        "param_formula_equals_measured": bool(measured == formula_params),
        "vram_gb": {"after_build": round(vram_after_build, 4), "peak_train": round(vram_peak, 4)},
        "five_step_loss": {"initial": losses[0], "final": losses[-1], "decreasing": bool(loss_decreasing),
                           "history": [round(x, 4) for x in losses]},
        "stage2_c1_dry_run": stage2_dry, "stage3_c2_dry_run": stage3_dry,
        "stages_wired": bool(isinstance(stage2_dry, dict) and stage2_dry.get("wired")
                             and isinstance(stage3_dry, dict) and stage3_dry.get("wired")),
        "device": device,
    }
    smoke_path = run_dir / "smoke_result.json"
    smoke_path.write_text(json.dumps(smoke_result, indent=2, default=str))
    print("\n" + "=" * 78, flush=True)
    print(f"[scaleup:smoke] SMOKE DONE: {measured:,} params ({measured/1e6:.2f}M) | "
          f"VRAM peak={vram_peak:.2f} GB | 5-step loss {losses[0]:.3f}->{losses[-1]:.3f} "
          f"(decreasing={loss_decreasing}) | stages_wired={smoke_result['stages_wired']}", flush=True)
    print(f"[scaleup:smoke] wrote {smoke_path}", flush=True)
    print("=" * 78, flush=True)
    return smoke_result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d-model", type=int, default=DEF_D_MODEL)
    ap.add_argument("--n-layers", type=int, default=DEF_N_LAYERS)
    ap.add_argument("--n-heads", type=int, default=DEF_N_HEADS)
    ap.add_argument("--vocab-size", type=int, default=DEF_VOCAB)
    ap.add_argument("--block-size", type=int, default=DEF_BLOCK)
    ap.add_argument("--steps", type=int, default=DEF_STEPS, help="STAGE-1 train steps (the ~4 h cost)")
    ap.add_argument("--batch-size", type=int, default=DEF_BATCH, help="STAGE-1 train batch")
    ap.add_argument("--ft-batch", type=int, default=DEF_FT_BATCH, help="STAGE-3 C2 fine-tune batch")
    ap.add_argument("--lr", type=float, default=DEF_LR)
    # --- regularization / schedule (size-aware: a 10x model on a SMALL corpus overfits without these) ---
    ap.add_argument("--dropout", type=float, default=DEF_DROPOUT,
                    help="STAGE-1 attn+MLP+emb dropout (default %g; a 27M model on ~1.8M tokens MUST "
                         "regularize or it memorizes the train set)" % DEF_DROPOUT)
    ap.add_argument("--weight-decay", type=float, default=DEF_WEIGHT_DECAY,
                    help="STAGE-1 AdamW weight decay (default %g)" % DEF_WEIGHT_DECAY)
    ap.add_argument("--warmup-steps", type=int, default=DEF_WARMUP,
                    help="STAGE-1 linear LR warmup steps then cosine decay (default %d)" % DEF_WARMUP)
    ap.add_argument("--heldout-every", type=int, default=DEF_HELDOUT_EVERY,
                    help="print STAGE-1 held-out TinyStories ppl every N steps (overfit probe; "
                         "default %d; 0=off)" % DEF_HELDOUT_EVERY)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpus", type=str, default="tinystories")
    ap.add_argument("--c2-original", type=str, default="tinystories", choices=["tinystories", "simplewiki"],
                    help="STAGE-3 C2 ORIGINAL (retention-measured) domain. 'tinystories' (default) is the "
                         "byte-unchanged behaviour; 'simplewiki' makes the C2 retain the model's ACTUAL "
                         "training domain (use this for the SimpleWiki-trained 100M -- 'retain TinyStories' "
                         "is a confounded test for it since it never saw TinyStories). The model's BPE "
                         "tokenizer is unchanged; only the original/retention corpus text is re-pointed.")
    ap.add_argument("--out", type=str, default=str(RAW / "_genseq_C2_scaleup_30M.json"),
                    help="STAGE-3 C2 result JSON (the loop verdict)")
    ap.add_argument("--run-dir", type=str, default=None,
                    help="resumable run dir for stage checkpoints/markers (default derived from --out)")
    ap.add_argument("--smoke", action="store_true",
                    help="1-step wiring SMOKE ONLY (param count + VRAM + 5 train steps + C1/C2 dry-run); "
                         "NO full training")
    a = ap.parse_args()

    import torch
    backend = os.environ.get("SIM_BACKEND", "auto")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert a.d_model % a.n_heads == 0, f"--d-model {a.d_model} must be divisible by --n-heads {a.n_heads}"

    out_path = Path(a.out)
    run_dir = Path(a.run_dir) if a.run_dir else (RAW / ("c2_scaleup_" + out_path.stem))
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[scaleup] SIM_BACKEND={backend} device={device} | run_dir={run_dir}", flush=True)

    # corpus (cached, offline-safe; degrades to tinyshakespeare with degraded=True noted).
    cinfo = fetch_corpus(name=a.corpus, max_bytes=8_000_000)
    train_text, heldout_text = split_corpus(cinfo["text"], heldout_frac=0.1)
    corpus_file = str(run_dir / "train_corpus.txt")
    Path(corpus_file).write_text(train_text, encoding="utf-8")
    # the held-out tail (same split the C2 stage scores on) -> the in-loop overfit probe.
    heldout_file = str(run_dir / "heldout_corpus.txt")
    Path(heldout_file).write_text(heldout_text, encoding="utf-8")
    print(f"[scaleup] corpus={cinfo['corpus_used']} degraded={cinfo['degraded']} "
          f"train_chars={len(train_text)} heldout_chars={len(heldout_text)}", flush=True)

    # the EXACT param count (printed up front regardless of mode).
    formula_params = tinygpt_param_count(a.vocab_size, a.d_model, a.n_layers, a.block_size)
    print(f"[scaleup] target arch V={a.vocab_size} d={a.d_model} L={a.n_layers} H={a.n_heads} "
          f"block={a.block_size} -> ~{formula_params:,} params ({formula_params/1e6:.2f}M) "
          f"(BPE may shift V by +1 -> recomputed exactly in the smoke)", flush=True)

    if a.smoke:
        smoke(a, run_dir, device, corpus_file)
        full_cmd = (f"SIM_BACKEND=cupy python -m research.runners._genseq_C2_scaleup_runner "
                    f"--d-model {a.d_model} --n-layers {a.n_layers} --n-heads {a.n_heads} "
                    f"--steps {a.steps} --batch-size {a.batch_size} --ft-batch {a.ft_batch} "
                    f"--lr {a.lr} --dropout {a.dropout} --weight-decay {a.weight_decay} "
                    f"--warmup-steps {a.warmup_steps} --heldout-every {a.heldout_every} "
                    f"--out {a.out}")
        print(f"\n[scaleup] FULL RUN (controller launches separately):\n  {full_cmd}", flush=True)
        return 0

    # ---- FULL run: the 3 stages end-to-end (each resumable via its marker) ----
    t_start = time.time()
    ckpt_path, bpe_path, s1 = stage1_train(
        run_dir, seed=a.seed, d_model=a.d_model, n_layer=a.n_layers, n_head=a.n_heads,
        vocab_size=a.vocab_size, block_size=a.block_size, steps=a.steps, batch_size=a.batch_size,
        lr=a.lr, device=device, corpus_path=corpus_file, dropout=a.dropout,
        weight_decay=a.weight_decay, warmup_steps=a.warmup_steps,
        heldout_path=heldout_file, heldout_every=a.heldout_every)
    if s1.get("interrupted"):
        print("[scaleup] stage 1 interrupted -> exiting (re-run to resume).", flush=True)
        return 0
    s2 = stage2_c1_consolidate(run_dir, ckpt_path=ckpt_path, bpe_path=bpe_path, device=device,
                               n_head=a.n_heads, n_logit_pos=12, n_gen_tokens=18, ppl_windows=4)
    arch_label = f"d{a.d_model}_L{a.n_layers}_H{a.n_heads}_~{formula_params//1_000_000}M"
    s3 = stage3_c2_loop(run_dir, ckpt_path=ckpt_path, bpe_path=bpe_path, device=device, n_head=a.n_heads,
                        ft_batch=a.ft_batch, out_path=out_path, arch_label=arch_label,
                        c2_original=a.c2_original)
    print("\n" + "=" * 78, flush=True)
    print(f"[scaleup] FULL LOOP DONE ({round(time.time()-t_start,1)}s). C2 verdict: {s3.get('verdict')}",
          flush=True)
    print(f"[scaleup]   {s3.get('verdict_line')}", flush=True)
    print(f"[scaleup]   stage1 final_loss={s1.get('final_loss')} | stage2 logit_fid="
          f"{s2.get('logit_fidelity_spearman')} ppl_ratio={s2.get('ppl_ratio')}", flush=True)
    print(f"[scaleup] wrote {out_path}", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
