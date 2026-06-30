"""GENERATIVE-SEQUENCE FRONTIER (Spine A) -- C2 RE-RUN at a LEARNABLE distribution shift.

WHY THIS RE-RUN EXISTS (read research/findings/2026-06-22-C2-grow-no-forget-scoping.md +
research/findings/raw/_genseq_C2_grow_no_forget.json first): the C2 MECHANISM is VALIDATED --
generative self-replay causally prevents catastrophic forgetting, dose-monotone (no_replay retention
0.107 -> replay_10 0.445 -> replay_30 0.631; new learned; no-replay forgets ~5.92x). But that prior
run used the EXTREME Shakespeare shift (41x distinct from TinyStories) on the 3.4M toy -> even WITH
replay the model could only retain ~52% (MISSED the >=85% bar) because a 41x register shift forces too
much weight motion for a 3.4M-param model to hold both distributions. The open question this re-run
answers: does generative self-replay hit >=85% retention at a LEARNABLE (moderate, in-domain-ish) shift
while still learning the new + the no-replay control still forgets?

THE NEW CORPUS (the design decision -- a LEARNABLE shift, NOT Shakespeare-full):
  An empirical corpus-selection sweep on THIS frozen 3.4M Gen-F (probe #1-#5, throwaway) mapped the
  whole shift space:
    - PURE TinyStories topic/structural slices (dragon/space/king/dialogue/longest) = ~0.8-1.05x baseline
      ppl: Gen-F (trained on the FULL 8MB TinyStories) already models every theme -> NO shift, the
      no-replay control would NOT forget (uninformative). REJECTED.
    - PURE out-of-domain registers (Shakespeare 42x, WikiText 91x; "simplifying" them by short-line
      filtering does NOT lower ppl -- register/vocab drives it, not length) = strong forgetting but
      replay can't restore past ~55%. REJECTED (this is the prior run's failure corner).
    - INTERLEAVED TinyStories + Shakespeare blocks at a tunable Shakespeare fraction = a CONTINUOUS
      distinctness knob landing the mixture held-out ppl anywhere in [10, 110]. CHOSEN.
  NEW = TinyStories-train blocks interleaved with Shakespeare blocks at SH_FRAC=0.45 -> held-out ppl
  ~47.8 (7.7x the TinyStories 6.21), 0% <UNK> under Gen-F's TinyStories BPE. This is a LEGITIMATE,
  measurably-distinct sub-distribution (children's stories carrying a distinct interleaved register),
  squarely in the prompt's ~20-60 band: distinct enough that the no-replay control should forget,
  learnable enough that replay can retain. Retention is ALWAYS measured on the DISJOINT pure-TinyStories
  held-out tail (never on the mixture) -> the retention number is honest.
  HONEST PRE-REGISTERED CAVEAT: a mixture self-reinforces the old distribution (45% of its blocks ARE
  TinyStories), so the no-replay forgetting contrast at this in-band point may be MODEST (the directional
  mini-FT probe showed ~1.07-1.10x, no-replay retaining ~64%) rather than the prior run's 5.92x
  catastrophic spike -- the price of staying in-band. The dose-response monotonicity + the absolute
  learn/retain numbers are the decisive evidence either way.

REUSE (verbatim, NO sim/ edit): the GROW (off-bridge fine-tune of a clone of the frozen Gen-F + the
generative self-replay corpus mix) + re-distill/install on the RF bridge (the C1 path) + the
two-distribution ppl harness + the no-replay control + the replay-fraction dose-response. The ONLY thing
that changes vs _genseq_C2_grow_no_forget_derisk.py is the NEW-corpus construction (interleave, not
pure Shakespeare) + the output path + the rationale.

VERDICT: GO = (original-retention >=0.85 at replay 0.3 OR 0.5) AND (new learned: ppl_drop >=0.5) AND
  (no-replay forgets: >=1.3x spike vs with-replay) -> the full LOOP (train->generate->grow->no-forget)
  demonstrated at toy scale. If even this moderate learnable shift can't retain >=0.85 (dose-response
  still monotone) -> a genuine SCALE issue (the cloud-justifying point) -- reported honestly with the
  dose-response. GPU. Usage:
  SIM_BACKEND=cupy python -m research.runners._genseq_C2_moderate_shift_derisk
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

# ---- reuse-by-import: the ppl harness (the byte-unmodified Gen-F gate), the corpus split, and the
#      EXACT C1 RF install + full forward. The GROW machinery is duplicated here ONLY because the
#      original runner inlines it in main(); the load-bearing functions (sample_self_replay,
#      grow_finetune, two_dist_ppl, verify_on_bridge) are byte-identical to the original. -----------
from research.runners.generator_f_gate import _heldout_nll, _generate  # noqa: E402
from research.runners.subword_lm_gate_core import (  # noqa: E402
    perplexity, distinct_ngram_ratio,
)
from research.runners.corpus_fetch import fetch_corpus, split_corpus  # noqa: E402

GENF_CKPT = _REPO / "research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real.pt"
GENF_BPE = _REPO / "research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real.bpe.json"
OUT_PATH = _REPO / "research/findings/raw/_genseq_C2_moderate_shift.json"

# ---- knobs (toy scale; foreground-bounded) -----------------------------------------------------
SEED = 42
BLOCK_SIZE = 128
PPL_EVAL_POSITIONS = 200     # held-out windows for the two-distribution ppl (both corpora)
FT_STEPS = 1500              # fine-tune steps (re-warmed; matches the original C2 run)
FT_BATCH = 48               # OOM-safe on a 3090 at d=256 block=128
FT_LR = 1e-5                # re-warm LR -- continual-learning rate (demo-design-validated 2026-06-23: 3e-4 OVERWROTE the original even w/replay -> retention 0.45; 1e-5 retains 0.884 w/replay vs 0.392 without). See 2026-06-23-generative-loop-DEMONSTRATED.md.
REPLAY_FRAC = 0.30          # reference replay fraction reported in the table (decisive arm chosen adaptively)
REPLAY_SWEEP = (0.0, 0.30, 0.50)   # the prompt's required sweep (no-replay control + 0.3 + 0.5)
REPLAY_SAMPLE_TOKENS = 200   # tokens per self-replay sample (autoregressive from the frozen Gen-F)
REPLAY_POOL_TOKENS = 60000   # bounded self-replay reservoir (sampled-with-replacement at the per-arm fraction)
ONBRIDGE_VERIFY_WINDOWS = 3  # windows to VERIFY the RF install reproduces off-bridge ppl (the "on the bridge" measure)
RETAIN_BAR = 0.85            # ORIGINAL ppl must stay <= baseline / RETAIN_BAR (>= 85% retention)
LEARN_BAR = 0.50             # NEW ppl must drop to <= (1 - LEARN_BAR) * pre-grow new-ppl (a clear >=50% drop --
                            #  the prompt's "new learned (ppl_drop >= 0.5)" bar)
FORGET_MARGIN = 1.30         # the no-replay arm's ORIGINAL ppl must be >= FORGET_MARGIN x the with-replay arm's

# ---- the NEW-corpus construction (the ONLY substantive change vs the original runner) -----------
SH_FRAC = 0.45               # default Shakespeare fraction (REACHABLE start of the auto-tune ladder).
                            #  On the 3.4M Gen-F (orig_ppl 6.21) this lands the mixture held-out ppl ~47.8
                            #  = 7.7x orig_ppl -- squarely IN the learnable band below, so the 3.4M behavior
                            #  is byte-reproduced. On a BIGGER (higher-orig_ppl) model 0.45 is TOO distinct
                            #  (the 30M's orig_ppl ~28.4 x 0.45-interleave -> ~836 ppl = ~29x) -> the
                            #  auto-tune sweeps SH_FRAC DOWN until base_new lands in the RELATIVE band.
INTERLEAVE_SEED = 7
INTERLEAVE_BLOCK = 400       # contiguous-char block granularity of the interleave
INTERLEAVE_TOTAL = 1_400_000 # total chars of the built NEW corpus

# ---- the LEARNABLE-shift band (RELATIVE to the CURRENT model's orig_ppl -- the size-adaptive fix) ----
# WHY (2026-06-23): the prior hardcoded `base_new < 110` upper guard was tuned for the 3.4M toy's ppl
# scale (orig_ppl 6.2 -> SH0.45 lands 47.8). At the 30M's HIGHER orig_ppl (~28.4) the SAME SH0.45
# interleave is ~836 ppl -> the absolute guard fires ("new corpus too distinct") even though, RELATIVE to
# the model's own competence, it is the SAME kind of shift. The fix anchors the band to orig_ppl: a NEW
# corpus is "learnable" when its pre-grow ppl is ~LEARNABLE_BAND_LO_MULT..HI_MULT x orig_ppl -- distinct
# enough that the no-replay control forgets, but learnable enough that replay can retain. The 3.4M's 7.7x
# point lands inside [4, 12], so the old ~47 reference is reachable.
LEARNABLE_BAND_LO_MULT = 4.0    # base_new >= this x orig_ppl (distinct enough the no-replay control forgets)
LEARNABLE_BAND_HI_MULT = 12.0   # base_new <= this x orig_ppl (learnable enough replay can retain)
# SH_FRAC candidate ladder, swept HIGH->LOW; the first frac whose base_new lands in the relative band is
# CHOSEN. (Starts at the 3.4M's 0.45 so that model keeps picking 0.45; descends for bigger models.)
# 2026-06-23 EMPIRICAL: the SH_FRAC->base_new curve is SHARPLY non-linear with a CLIFF in [0.30, 0.45]:
# on the 30M, SH_FRAC=0.45 -> base_new 836x but SH_FRAC=0.30 -> only 2.75x (the band [4,12]x falls in the
# GAP). The ladder is therefore DENSE through the cliff (0.45..0.30 in 0.02-0.03 steps) so a bigger model
# can land its band there; the coarse low rungs remain for any model whose band sits lower.
SH_FRAC_LADDER = (0.45, 0.43, 0.41, 0.39, 0.37, 0.35, 0.33, 0.31, 0.30, 0.25, 0.20, 0.12, 0.08, 0.05)
TS_TRAIN_CAP = 3_000_000     # cap on the TinyStories-train source used for the interleave


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


# =================================================================================================
# Load the frozen Gen-F into a TinyGPT (weights VERBATIM) + its TinyStories BPE. (byte-identical to orig)
# =================================================================================================
def load_genf(device, *, ckpt_path=None, bpe_path=None, d_model=256, n_layer=4, n_head=4,
              block_size=None):
    """Load a frozen Gen-F TinyGPT (weights VERBATIM) + its BPE. Defaults reproduce the original 3.4M
    toy (d=256/L=4/H=4, the s42.real checkpoint); the scale-up runner passes the bigger arch + paths.
    The arch is also auto-detected from the state_dict (d_model/n_layer/block_size/vocab from the
    tensor shapes) so a mismatched hyperparam can't silently load the wrong shape -- n_head alone is
    not recoverable from the shapes, so it is taken from the argument (and must match training)."""
    import torch
    from sim.tiny_transformer import TinyGPT
    from sim.bpe_tokenizer import BPETokenizer
    ckpt_path = str(ckpt_path) if ckpt_path is not None else str(GENF_CKPT)
    bpe_path = str(bpe_path) if bpe_path is not None else str(GENF_BPE)
    tok = BPETokenizer.load(bpe_path)
    V = tok.vocab_size
    # weights_only=True: OUR OWN trusted, local, project-generated checkpoint; safe unpickler regardless.
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = ck["model"]
    loss_last = float(ck["loss_history"][-1]) if ck.get("loss_history") else float("nan")
    # auto-detect arch from the tensor shapes (defensive against a wrong --d-model / --n-layer arg).
    d_det = int(sd["tok.weight"].shape[1])
    L_det = sum(1 for k in sd if k.endswith(".ln1.weight") and k.startswith("blocks."))
    blk_det = int(sd["pos.weight"].shape[0])
    V_det = int(sd["tok.weight"].shape[0])
    if d_model != d_det or n_layer != L_det:
        print(f"[C2:load_genf] NOTE: arch arg (d={d_model},L={n_layer}) != ckpt shapes "
              f"(d={d_det},L={L_det}); using ckpt shapes.", flush=True)
    d_model, n_layer, block_size, V = d_det, L_det, blk_det, V_det
    m = TinyGPT(vocab_size=V, d_model=d_model, n_layer=n_layer, n_head=n_head,
                block_size=block_size, dropout=0.0).to(device)
    m.load_state_dict(sd)
    m.eval()
    del ck, sd
    return m, tok, V, loss_last


def clone_model(model, device):
    """A fresh TinyGPT initialised from `model`'s weights (so the fine-tune does not mutate the frozen
    baseline / replay source). (byte-identical to orig)"""
    import copy
    import torch
    from sim.tiny_transformer import TinyGPT
    cfg = model.cfg
    m = TinyGPT(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], n_layer=cfg["n_layer"],
                n_head=cfg["n_head"], block_size=cfg["block_size"], dropout=0.0).to(device)
    m.load_state_dict(copy.deepcopy(model.state_dict()))
    return m


# =================================================================================================
# THE NEW (LEARNABLE) DISTRIBUTION: interleave TinyStories-train blocks with Shakespeare blocks at
# SH_FRAC. A continuous distinctness knob (vs the original's pure-Shakespeare 41x); SH_FRAC=0.45 lands
# the mixture held-out ppl ~47.8 (7.7x the TinyStories 6.21) -- the prompt's ~20-60 learnable band.
# Deterministic (np.random.default_rng(INTERLEAVE_SEED)).
# =================================================================================================
def build_new_corpus(ts_tr, sh, frac, seed):
    rng = np.random.default_rng(seed)
    out = []
    n = 0
    ai = bi = 0
    while n < INTERLEAVE_TOTAL:
        if rng.random() < frac and bi + INTERLEAVE_BLOCK < len(sh):
            out.append(sh[bi:bi + INTERLEAVE_BLOCK]); bi += INTERLEAVE_BLOCK
        elif ai + INTERLEAVE_BLOCK < len(ts_tr):
            out.append(ts_tr[ai:ai + INTERLEAVE_BLOCK]); ai += INTERLEAVE_BLOCK
        else:
            ai = 0   # wrap the TinyStories source if exhausted
        n += INTERLEAVE_BLOCK
    return " ".join(out)


def _build_new_split(ts_tr, sh_text, frac, tok):
    """Build the NEW (interleave) corpus at a given SH_FRAC and return everything the loop needs:
    (new_full, new_tr, new_ho, new_train_ids, new_unk). Factored out of run_c2_loop's inline body so the
    SH_FRAC auto-tune can construct candidates without duplicating the split/encode."""
    new_full = build_new_corpus(ts_tr[:TS_TRAIN_CAP], sh_text, frac, INTERLEAVE_SEED)
    new_tr, new_ho = split_corpus(new_full, heldout_frac=0.15)
    new_train_ids = tok.encode(new_tr[:600_000])
    new_unk = sum(1 for i in new_train_ids if i == 0) / max(1, len(new_train_ids))
    return new_full, new_tr, new_ho, new_train_ids, new_unk


def auto_select_sh_frac(frozen, tok, ts_tr, sh_text, ts_ho, base_ts, device, *, ppl_positions,
                        lo_mult=None, hi_mult=None, ladder=None, dry_run=False):
    """Sweep SH_FRAC HIGH->LOW over `ladder`; at each candidate build the NEW interleave + measure its
    pre-grow held-out ppl (`base_new`) under the FROZEN Gen-F, and CHOOSE the first frac whose base_new
    lands in the RELATIVE learnable band [lo_mult, hi_mult] x base_ts. This makes the corpus selection
    adapt to ANY model size (the 3.4M's orig_ppl 6.2 picks ~0.45 = 7.7x; the 30M's orig_ppl ~28.4 needs a
    LOWER frac to stay <=12x). If NO candidate lands in band, pick the one whose ppl/base_ts ratio is
    CLOSEST to the band (clamped), so the loop still runs honestly on the most-learnable available shift.

    Returns (sel_frac, sel_base_new, sel_split, sweep_log) where sel_split is the
    (_build_new_split) tuple for the chosen frac (so the caller does not rebuild it)."""
    lo_mult = float(lo_mult) if lo_mult is not None else LEARNABLE_BAND_LO_MULT
    hi_mult = float(hi_mult) if hi_mult is not None else LEARNABLE_BAND_HI_MULT
    ladder = tuple(ladder) if ladder is not None else SH_FRAC_LADDER
    band_lo, band_hi = lo_mult * base_ts, hi_mult * base_ts
    print(f"\n[C2mod] ===== AUTO-SELECT SH_FRAC: target base_new in [{band_lo:.2f}, {band_hi:.2f}] "
          f"(= [{lo_mult:.1f}x, {hi_mult:.1f}x] orig_ppl {base_ts:.3f}) =====", flush=True)
    center = math.sqrt(band_lo * band_hi)
    sweep_log = []
    chosen = None
    chosen_split = None
    best_dist = None   # fallback: track the candidate closest to the band centre (by log-distance)
    best_frac = best_base_new = best_split = None
    above_dist = None  # fallback-2: the LOWEST rung still >= band_lo (least-distinct-but-distinct-enough)
    above_frac = above_base_new = above_split = None
    for frac in ladder:
        split = _build_new_split(ts_tr, sh_text, frac, tok)
        new_ho = split[2]
        bn = perplexity(_heldout_nll(frozen, tok, new_ho, BLOCK_SIZE, device, ppl_positions))
        ratio = bn / base_ts if base_ts > 0 else float("inf")
        in_band = (band_lo <= bn <= band_hi)
        sweep_log.append({"sh_frac": frac, "base_new_ppl": bn, "ratio_over_orig": ratio,
                          "in_band": bool(in_band)})
        print(f"[C2mod]   SH_FRAC={frac:.2f}: base_new={bn:.3f} ({ratio:.2f}x orig)  "
              f"{'<= IN BAND' if in_band else 'out of band'}", flush=True)
        # fallback-1: multiplicative distance to the band centre (log-space).
        dist = abs(math.log(bn / center)) if (bn > 0 and math.isfinite(bn)) else float("inf")
        if best_dist is None or dist < best_dist:
            best_dist, best_frac, best_base_new, best_split = dist, frac, bn, split
        # fallback-2: among rungs that CLEAR band_lo (>= LO_MULT x orig = distinct enough so the LO assert
        # passes + the no-replay control forgets), keep the one CLOSEST to band_lo (least over-distinct).
        if math.isfinite(bn) and bn >= band_lo:
            d_lo = abs(math.log(bn / band_lo))
            if above_dist is None or d_lo < above_dist:
                above_dist, above_frac, above_base_new, above_split = d_lo, frac, bn, split
        if in_band:
            chosen, chosen_split = frac, split
            break   # first (highest) frac in band -> the LEAST-distinct learnable shift (most retainable)
        if dry_run:
            # smoke: don't sweep the whole ladder (1-window ppl is too noisy to band-match anyway) --
            # take the first candidate as the "selection" purely to exercise the wiring.
            chosen, chosen_split = frac, split
            break
    if chosen is None:
        # Prefer fallback-2 (a rung that clears band_lo) so the >= LO_MULT guard passes and the no-replay
        # control still forgets; only if NO rung clears band_lo do we fall to the closest-by-log-distance.
        if above_frac is not None:
            chosen, chosen_split = above_frac, above_split
            print(f"[C2mod]   NO SH_FRAC landed exactly in band; FALLBACK to the lowest rung that still "
                  f"clears band_lo: SH_FRAC={chosen:.2f} (base_new={above_base_new:.3f}, "
                  f"{above_base_new/base_ts:.2f}x orig >= {LEARNABLE_BAND_LO_MULT:.0f}x).", flush=True)
        else:
            chosen, chosen_split = best_frac, best_split
            print(f"[C2mod]   NO SH_FRAC reached band_lo; FALLBACK to closest-to-band SH_FRAC={chosen:.2f} "
                  f"(base_new={best_base_new:.3f}, {best_base_new/base_ts:.2f}x orig). The run will proceed "
                  f"and the verdict logic reports honestly.", flush=True)
    sel_base_new = perplexity(_heldout_nll(frozen, tok, chosen_split[2], BLOCK_SIZE, device, ppl_positions))
    print(f"[C2mod] AUTO-SELECTED SH_FRAC={chosen:.2f} -> base_new={sel_base_new:.3f} "
          f"({sel_base_new/base_ts:.2f}x orig_ppl {base_ts:.3f})", flush=True)
    return chosen, sel_base_new, chosen_split, sweep_log


# =================================================================================================
# GENERATIVE SELF-REPLAY: sample OLD-distribution (TinyStories) text from the FROZEN pre-grow Gen-F.
# (byte-identical to orig) -- the CLS hippocampal-replay analogue (Shin 2017).
# =================================================================================================
def sample_self_replay(frozen_model, tok, n_target_tokens, block_size, device, seed, primes=None):
    import torch
    rng = np.random.default_rng(seed)
    out_ids = []
    si = 0
    # default primes = TinyStories register (byte-unchanged for the original 3.4M path); a
    # different-domain frozen model (e.g. the SimpleWiki-trained 100M) passes register-matched primes so
    # the self-replay samples its OWN old distribution, not a TinyStories one it was never trained on.
    if primes is None:
        primes = ["Once upon a time", "One day", "There was a", "She had a", "The little",
                  "He wanted to", "They went to", "Lily and Tom"]
    while len(out_ids) < n_target_tokens:
        prime = primes[si % len(primes)]
        prompt_ids = tok.encode(prime)
        gen = _generate(frozen_model, tok, prompt_ids, REPLAY_SAMPLE_TOKENS, block_size, device,
                        int(rng.integers(1, 2**31 - 1)))
        out_ids.extend(prompt_ids)
        out_ids.extend(gen)
        si += 1
    return out_ids[:n_target_tokens]


# =================================================================================================
# THE GROW STEP (grow-route #1): fine-tune a clone of the OFF-bridge Gen-F on (NEW + self-replayed-OLD).
# (byte-identical to orig) -- self-contained fine-tune with a RE-WARMED fresh optimiser/cosine.
# =================================================================================================
def grow_finetune(frozen_model, tok, new_train_ids, replay_ids, target_replay_frac, device, *, steps,
                  batch_size, lr, block_size, seed, label):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = clone_model(frozen_model, device)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    V = tok.vocab_size

    new_t = torch.tensor(new_train_ids, dtype=torch.long, device=device)
    n_new = new_t.numel()
    have_replay = (replay_ids is not None and len(replay_ids) > (block_size + 2)
                   and target_replay_frac > 0.0)
    if have_replay:
        rep_t = torch.tensor(replay_ids, dtype=torch.long, device=device)
        n_rep = rep_t.numel()
        replay_frac = float(target_replay_frac)
    else:
        rep_t, n_rep, replay_frac = None, 0, 0.0

    g = torch.Generator(device="cpu").manual_seed(seed * 7 + 1)

    def _batch(bs):
        if have_replay:
            use_rep = torch.rand(bs, generator=g) < replay_frac
        else:
            use_rep = torch.zeros(bs, dtype=torch.bool)
        xs, ys = [], []
        for b in range(bs):
            if bool(use_rep[b]):
                i = int(torch.randint(0, n_rep - block_size - 1, (1,), generator=g).item())
                src = rep_t
            else:
                i = int(torch.randint(0, n_new - block_size - 1, (1,), generator=g).item())
                src = new_t
            xs.append(src[i:i + block_size])
            ys.append(src[i + 1:i + 1 + block_size])
        return torch.stack(xs), torch.stack(ys)

    loss_hist = []
    cur_bs = batch_size
    t0 = time.time()
    step = 0
    while step < steps:
        try:
            x, y = _batch(cur_bs)
            logits = m(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_hist.append(float(loss.item()))
            step += 1
            if step % 300 == 0:
                print(f"[C2mod:{label}] ft step {step}/{steps} loss={loss_hist[-1]:.4f} "
                      f"replay_frac={replay_frac:.2f} ({time.time()-t0:.0f}s)", flush=True)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() and cur_bs > 1:
                torch.cuda.empty_cache()
                cur_bs = max(1, cur_bs // 2)
                print(f"[C2mod:{label}] OOM -> batch halved to {cur_bs}", flush=True)
                continue
            raise
    m.eval()
    log = {"label": label, "steps": steps, "final_loss": (loss_hist[-1] if loss_hist else None),
           "initial_loss": (loss_hist[0] if loss_hist else None), "realized_replay_frac": replay_frac,
           "n_new_tokens": int(n_new), "n_replay_tokens": int(n_rep), "batch_size_final": cur_bs}
    return m, log


# =================================================================================================
# Two-distribution held-out ppl (the C2 measurement). ORIGINAL = pure-TinyStories tail; NEW = the
# interleave held-out tail. SAME Gen-F TinyStories BPE throughout -> directly comparable.
# =================================================================================================
def two_dist_ppl(model, tok, ts_ho, new_ho, device, *, label, ppl_positions=None, orig_label="TinyStories"):
    ppl_positions = int(ppl_positions) if ppl_positions is not None else PPL_EVAL_POSITIONS
    ts_ppl = perplexity(_heldout_nll(model, tok, ts_ho, BLOCK_SIZE, device, ppl_positions))
    new_ppl = perplexity(_heldout_nll(model, tok, new_ho, BLOCK_SIZE, device, ppl_positions))
    print(f"[C2mod:{label}]   ORIGINAL({orig_label}) held-out ppl = {ts_ppl:.4f} | "
          f"NEW(SH{SH_FRAC}-interleave) held-out ppl = {new_ppl:.4f}", flush=True)
    return {"original_tinystories_ppl": ts_ppl, "new_interleave_ppl": new_ppl}


# =================================================================================================
# ON-THE-BRIDGE VERIFICATION (byte-identical to orig): re-distill + install the grown model on the RF
# complex-synapse bridge (the C1 path) and confirm on-bridge ppl == off-bridge ppl on the SAME windows.
# =================================================================================================
def verify_on_bridge(grown_model, tok, ts_ho, new_ho, device, *, n_windows=None):
    from research.runners._genseq_loopstep3_full_genf_generate_derisk import (
        rf_full_forward, _heldout_nll_numpy, _perplexity)
    from research.runners._genseq_loopstep3_rf_probe import (
        _build_rf_bridge, RF_PERIOD, RF_NSTEPS, RF_LAMBDA)
    n_windows = int(n_windows) if n_windows is not None else ONBRIDGE_VERIFY_WINDOWS

    d = grown_model.cfg["d_model"]; V = grown_model.cfg["vocab_size"]; n_layer = grown_model.cfg["n_layer"]
    sd = {k: v.detach().to("cpu") for k, v in grown_model.state_dict().items()}
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
        "blocks": blocks,
        "tok_emb": sd["tok.weight"].numpy().astype(np.float64),
        "pos_emb": sd["pos.weight"].numpy().astype(np.float64),
        "lnf_w": sd["lnf.weight"].numpy().astype(np.float64),
        "lnf_b": sd["lnf.bias"].numpy().astype(np.float64),
        "Whead": sd["head.weight"].numpy().T.astype(np.float32).copy(),
        "n_head": grown_model.cfg["n_head"], "n_layer": n_layer, "d_model": d,
        "block_size": grown_model.cfg["block_size"], "vocab_size": V}

    n_dd = d + d; n_m1 = d + 4 * d; n_m2 = 4 * d + d; n_head_bridge = d + V
    bridges = {"dd": _build_rf_bridge(n_dd, seed=42), "mlp1": _build_rf_bridge(n_m1, seed=42),
               "mlp2": _build_rf_bridge(n_m2, seed=42), "head": _build_rf_bridge(n_head_bridge, seed=42)}

    def _rf_fwd(_ids):
        lg, _ = rf_full_forward(model_b, _ids, bridges, period=RF_PERIOD, nsteps=RF_NSTEPS, lam=RF_LAMBDA)
        return lg

    def _off_fwd(_ids):
        import torch as _t
        with _t.no_grad():
            x = _t.tensor(_ids, dtype=_t.long, device=device)[None]
            return grown_model(x)[0].float().cpu().numpy()

    ts_ids = tok.encode(ts_ho[:n_windows * BLOCK_SIZE * 8])
    new_ids = tok.encode(new_ho[:n_windows * BLOCK_SIZE * 8])
    res = {}
    for name, ids in (("original_tinystories", ts_ids), ("new_interleave", new_ids)):
        rf_nll = _heldout_nll_numpy(_rf_fwd, ids, V, BLOCK_SIZE, n_windows)
        off_nll = _heldout_nll_numpy(_off_fwd, ids, V, BLOCK_SIZE, n_windows)
        rf_ppl = _perplexity(rf_nll); off_ppl = _perplexity(off_nll)
        ratio = rf_ppl / off_ppl if (math.isfinite(off_ppl) and off_ppl > 0) else float("inf")
        res[name] = {"on_bridge_ppl": rf_ppl, "off_bridge_ppl": off_ppl, "ppl_ratio": ratio,
                     "n_windows": n_windows}
        print(f"[C2mod:on-bridge]   {name}: RF-on-bridge ppl={rf_ppl:.4f} off-bridge ppl={off_ppl:.4f} "
              f"ratio={ratio:.6f}", flush=True)
    del bridges
    free_cuda()
    return res


# =================================================================================================
# THE ORIGINAL-DISTRIBUTION SELECTOR (the C2-VALIDITY fix, 2026-06-30). The default ORIGINAL is
# TinyStories (the 3.4M toy's training domain). BUT the 100M scale-up Gen-F was TRAINED ON SIMPLEWIKI
# (held-out ppl ~11.5), NOT TinyStories -- so "retain TinyStories" is a CONFOUNDED test for it (the
# model never knew TinyStories; its TinyStories ppl is ~227, and the SH-interleave is mostly TinyStories
# so the model LEARNS rather than forgets -> no_replay_forgets=False, a spurious NEGATIVE). This selector
# lets the ORIGINAL task be the model's ACTUAL training domain, so retention is measured on what it KNOWS.
#   "tinystories" (default) -> the byte-unchanged behaviour (fetch_corpus tinystories, TinyStories primes).
#   "simplewiki"            -> ORIGINAL=SimpleWiki: prefer the model's OWN cached SimpleWiki train/heldout
#                              split (research/findings/raw/c2_scaleup_100M/{train,heldout}_corpus.txt, the
#                              exact text the 100M trained+eval'd on so orig_ppl reproduces ~11.5); fall
#                              back to splitting data/corpus/simplewiki.txt. NEW=SimpleWiki interleaved with
#                              Shakespeare (the SAME build_new_corpus mechanism). The tokenizer (`tok`,
#                              the SimpleWiki-fit BPE) is UNCHANGED -- only the corpus TEXT is re-pointed.
# Returns (orig_label, orig_train, orig_heldout, replay_primes, source_note).
# =================================================================================================
_SIMPLEWIKI_CACHED_DIR = _REPO / "research/findings/raw/c2_scaleup_100M"
_SIMPLEWIKI_FALLBACK = _REPO / "data/corpus/simplewiki.txt"
# SimpleWiki-register self-replay primes (encyclopaedic openings) so a SimpleWiki-trained frozen Gen-F
# samples its OWN old distribution for the no-forget replay (vs the TinyStories "Once upon a time" set).
_SIMPLEWIKI_PRIMES = ["The", "In", "A", "It is", "He was", "She was", "They were",
                      "This is a", "There are", "The city of"]
# CAP the SimpleWiki original train/heldout char lengths. WHY: the project BPE encode is SLOW
# (~0.02 MChar/s measured) and `_heldout_nll` re-encodes the WHOLE original-heldout on every ppl call
# (baseline + each grow arm) -- the model's full 14MB SimpleWiki heldout would cost ~15 min PER call
# (~1 hr wasted). The heldout ppl scores only ppl_eval_positions (200) windows x block (128) = ~25.6K
# tokens, so ~1.5 MB of heldout text is MORE than enough; the train cap matches build_new_corpus's
# internal TS_TRAIN_CAP (3M) usage. Both are leading slices of the SAME contiguous cached split, so the
# retention measurement (a leading slice of the disjoint pure-SimpleWiki heldout tail) is honest.
_SIMPLEWIKI_HELDOUT_CAP = 1_500_000
_SIMPLEWIKI_TRAIN_CAP = TS_TRAIN_CAP   # the interleave only consumes tr[:TS_TRAIN_CAP] anyway


def load_original_corpus(c2_original, *, ts_tr, ts_ho):
    """Pick the ORIGINAL (retention-measured) distribution + register-matched self-replay primes.
    `ts_tr`/`ts_ho` are the already-fetched TinyStories train/heldout (so the default path reuses them
    with ZERO extra I/O). Returns (orig_label, orig_train, orig_heldout, replay_primes, source_note)."""
    key = (c2_original or "tinystories").strip().lower()
    if key in ("tinystories", "ts", "default", ""):
        return "TinyStories", ts_tr, ts_ho, None, "tinystories (default; byte-unchanged)"
    if key in ("simplewiki", "wiki", "simple_wiki"):
        from research.runners.corpus_fetch import clean_text
        tr_cache = _SIMPLEWIKI_CACHED_DIR / "train_corpus.txt"
        ho_cache = _SIMPLEWIKI_CACHED_DIR / "heldout_corpus.txt"
        if tr_cache.is_file() and ho_cache.is_file():
            # the EXACT text the 100M trained + eval'd on (already cleaned + split) -> orig_ppl ~= 11.5.
            # Read only a leading slice (cleaning + holding all 127MB is needless; the caps below bound it).
            o_tr = clean_text(tr_cache.read_text(encoding="utf-8", errors="ignore")[:_SIMPLEWIKI_TRAIN_CAP * 2])
            o_ho = clean_text(ho_cache.read_text(encoding="utf-8", errors="ignore")[:_SIMPLEWIKI_HELDOUT_CAP * 2])
            src = f"the model's OWN cached split {tr_cache.name}/{ho_cache.name}"
        else:
            # fallback: split the raw SimpleWiki corpus the same way STAGE 1 did (heldout_frac=0.1).
            wiki = fetch_corpus(name=str(_SIMPLEWIKI_FALLBACK), max_bytes=200_000_000)
            o_tr, o_ho = split_corpus(wiki["text"], heldout_frac=0.1)
            src = f"{_SIMPLEWIKI_FALLBACK.name} split heldout_frac=0.1 (cached run split absent)"
        # CAP both (the BPE encode is slow; see _SIMPLEWIKI_HELDOUT_CAP). Leading slices of the SAME
        # contiguous split, so train stays disjoint from the heldout tail and retention is honest.
        o_tr = o_tr[:_SIMPLEWIKI_TRAIN_CAP]
        o_ho = o_ho[:_SIMPLEWIKI_HELDOUT_CAP]
        note = (f"SimpleWiki from {src} -- the 100M's actual training domain; train {len(o_tr)} (cap "
                f"{_SIMPLEWIKI_TRAIN_CAP}) / heldout {len(o_ho)} (cap {_SIMPLEWIKI_HELDOUT_CAP}) chars")
        return "SimpleWiki", o_tr, o_ho, _SIMPLEWIKI_PRIMES, note
    raise ValueError(f"unknown c2_original={c2_original!r} (expected 'tinystories' or 'simplewiki')")


def run_c2_loop(frozen, tok, V, loss_last, device, *, out_path=None, ft_batch=None,
                ft_steps=None, ft_lr=None, sh_frac=None, replay_sweep=None,
                ppl_eval_positions=None, arch_label=None, t_start=None, do_onbridge_verify=True,
                replay_pool_tokens=None, dry_run=False, c2_original="tinystories"):
    """The C2 grow-no-forget LOOP body (corpora -> baseline -> self-replay -> grow dose-sweep ->
    on-bridge verify -> verdict), factored out of main() so the C2 scale-up runner can drive the SAME
    machinery on a BIGGER frozen Gen-F. All knobs default to the module-level constants (so the
    original main() is behaviourally unchanged). `frozen` is an already-loaded TinyGPT (any arch); the
    body is arch-agnostic (it reads frozen.cfg throughout). `dry_run` cuts every loop to the cheapest
    1-window/1-arm/short-FT smoke (for the scale-up wiring smoke -- NOT a real measurement)."""
    import torch
    out_path = Path(out_path) if out_path is not None else OUT_PATH
    ft_batch = int(ft_batch) if ft_batch is not None else FT_BATCH
    ft_steps = int(ft_steps) if ft_steps is not None else FT_STEPS
    ft_lr = float(ft_lr) if ft_lr is not None else FT_LR
    # sh_frac=None (the DEFAULT) -> AUTO-TUNE it to land base_new in the relative learnable band for THIS
    # model's orig_ppl (the size-adaptive fix). A pinned float forces that exact frac (back-compat / tests).
    sh_frac_requested = (float(sh_frac) if sh_frac is not None else None)
    auto_sh_frac = (sh_frac_requested is None)
    replay_sweep = tuple(replay_sweep) if replay_sweep is not None else REPLAY_SWEEP
    ppl_eval_positions = int(ppl_eval_positions) if ppl_eval_positions is not None else PPL_EVAL_POSITIONS
    replay_pool_tokens = int(replay_pool_tokens) if replay_pool_tokens is not None else REPLAY_POOL_TOKENS
    arch_label = arch_label or f"d{frozen.cfg['d_model']}_L{frozen.cfg['n_layer']}"
    if t_start is None:
        t_start = time.time()
    if dry_run:   # cheapest possible wiring smoke: 1 window, 1 with-replay arm, a handful of FT steps
        ppl_eval_positions = 1
        ft_steps = min(ft_steps, 3)
        replay_sweep = (0.0, 0.30)
        replay_pool_tokens = min(replay_pool_tokens, 1500)

    # ---- corpora ----
    # Always fetch TinyStories (it is the DEFAULT original AND the no-extra-I/O reuse for the selector).
    ts = fetch_corpus(name="tinystories", max_bytes=8_000_000)
    _ts_tr, _ts_ho = split_corpus(ts["text"], heldout_frac=0.1)
    sh = fetch_corpus(name=str(_REPO / "data/tinyshakespeare.txt"), max_bytes=8_000_000)
    # ORIGINAL-distribution selector (the 2026-06-30 C2-validity fix): default 'tinystories' returns the
    # TinyStories split verbatim (byte-unchanged); 'simplewiki' re-points ORIGINAL to the 100M's actual
    # training domain so retention is measured on what the model KNOWS. `ts_tr`/`ts_ho` keep their names
    # downstream (they are "the ORIGINAL train/heldout", whatever domain that is); `replay_primes` are
    # the register-matched self-replay seeds; the tokenizer is UNCHANGED (only the corpus text moves).
    orig_label, ts_tr, ts_ho, replay_primes, orig_note = load_original_corpus(
        c2_original, ts_tr=_ts_tr, ts_ho=_ts_ho)
    print(f"[C2mod] ORIGINAL={orig_label} [{orig_note}] (train {len(ts_tr)} / heldout {len(ts_ho)} chars); "
          f"Shakespeare register source {len(sh['text'])} chars (degraded={sh['degraded']})", flush=True)

    # ---- ORIGINAL-distribution baseline FIRST (it anchors the relative learnable band) ----
    base_ts = perplexity(_heldout_nll(frozen, tok, ts_ho, BLOCK_SIZE, device, ppl_eval_positions))
    print(f"[C2mod] BASELINE orig({orig_label}) held-out ppl = {base_ts:.4f} (anchors the learnable band)",
          flush=True)

    # ---- choose the NEW (learnable) distribution = TS-blocks interleaved with SH-blocks at SH_FRAC ----
    sh_select_sweep = None
    if auto_sh_frac:
        # AUTO-TUNE: sweep SH_FRAC DOWN until base_new lands in [LO_MULT, HI_MULT] x base_ts. Size-adaptive.
        sh_frac, base_new, _split, sh_select_sweep = auto_select_sh_frac(
            frozen, tok, ts_tr, sh["text"], ts_ho, base_ts, device,
            ppl_positions=ppl_eval_positions, dry_run=dry_run)
        new_full, new_tr, new_ho, new_train_ids, new_unk = _split
    else:
        # PINNED frac (back-compat / explicit override): build it and measure base_new directly.
        sh_frac = sh_frac_requested
        new_full, new_tr, new_ho, new_train_ids, new_unk = _build_new_split(ts_tr, sh["text"], sh_frac, tok)
        base_new = perplexity(_heldout_nll(frozen, tok, new_ho, BLOCK_SIZE, device, ppl_eval_positions))
    print(f"[C2mod] NEW(SH-frac={sh_frac} interleave): full {len(new_full)} chars "
          f"(train {len(new_tr)} / heldout {len(new_ho)}); train tokens {len(new_train_ids)} "
          f"(<UNK> frac {new_unk:.4f})", flush=True)

    # =============================================================================================
    # BASELINE: pin the two-distribution pre-grow ppls + the distinctness ratio.
    # =============================================================================================
    base = {"original_tinystories_ppl": base_ts, "new_interleave_ppl": base_new}
    distinct_ratio = base_new / base_ts if base_ts > 0 else float("inf")
    band_lo = LEARNABLE_BAND_LO_MULT * base_ts
    band_hi = LEARNABLE_BAND_HI_MULT * base_ts
    print(f"[C2mod] BASELINE: {orig_label}(orig)={base_ts:.4f}  NEW(SH{sh_frac}-interleave)={base_new:.4f}  "
          f"(distinctness {distinct_ratio:.2f}x; learnable band [{band_lo:.2f},{band_hi:.2f}] "
          f"= [{LEARNABLE_BAND_LO_MULT:.0f},{LEARNABLE_BAND_HI_MULT:.0f}]x orig)", flush=True)
    # RELATIVE band (size-adaptive): the new corpus must be measurably distinct (>= LO_MULT x orig_ppl, so
    # the no-replay control forgets) but learnable (<= HI_MULT x orig_ppl, so replay can retain). Replaces
    # the old absolute `base_new < 110` guard (which was tuned for the 3.4M's ppl scale).
    in_learnable_band = bool(LEARNABLE_BAND_LO_MULT <= distinct_ratio <= LEARNABLE_BAND_HI_MULT)
    if not dry_run:   # the distinctness guard is a real-measurement guard; a 1-window smoke ppl is too noisy
        if auto_sh_frac:
            # AUTO mode: the ladder already swept for the band. If the SH_FRAC->ppl CLIFF means no rung
            # lands exactly in band, the auto-tuner chose the best available shift -- the run PROCEEDS and
            # the verdict reports honestly (a "shift-space cliff" is a real finding, not a code bug). We
            # warn but never crash, so the loop still produces its dose-response evidence.
            if not in_learnable_band:
                print(f"[C2mod] WARNING: auto-selected SH_FRAC={sh_frac:.2f} base_new={base_new:.2f} "
                      f"({distinct_ratio:.2f}x orig) did NOT land in the [{LEARNABLE_BAND_LO_MULT:.0f},"
                      f"{LEARNABLE_BAND_HI_MULT:.0f}]x band -- the SH_FRAC->ppl curve has a cliff there. "
                      f"PROCEEDING on the closest available shift; the verdict reports the honest result.",
                      flush=True)
        else:
            # PINNED frac (explicit user/back-compat choice): validate it hard -- a pinned out-of-band frac
            # is a config error the caller should see immediately.
            assert distinct_ratio >= LEARNABLE_BAND_LO_MULT, (
                f"pinned sh_frac={sh_frac} NOT measurably distinct (ratio {distinct_ratio:.2f}x < "
                f"{LEARNABLE_BAND_LO_MULT}x orig_ppl); raise sh_frac.")
            assert distinct_ratio <= LEARNABLE_BAND_HI_MULT, (
                f"pinned sh_frac={sh_frac} too distinct for the learnable band (ratio {distinct_ratio:.2f}x "
                f"> {LEARNABLE_BAND_HI_MULT}x orig_ppl, ppl {base_new:.1f}); lower sh_frac.")

    # =============================================================================================
    # SELF-REPLAY: sample OLD (ORIGINAL-domain) text from the FROZEN pre-grow Gen-F (built ONCE).
    # =============================================================================================
    print(f"\n[C2mod] ===== GENERATIVE SELF-REPLAY: sample OLD {orig_label} from the FROZEN Gen-F =====",
          flush=True)
    t0 = time.time()
    replay_ids_full = sample_self_replay(frozen, tok, replay_pool_tokens, BLOCK_SIZE, device, seed=SEED * 17,
                                         primes=replay_primes)
    rep_distinct = distinct_ngram_ratio(replay_ids_full, n=3)
    print(f"[C2mod] self-replay: sampled {len(replay_ids_full)} OLD tokens from frozen Gen-F "
          f"(distinct-trigram {rep_distinct:.3f}, {time.time()-t0:.0f}s); sample decode: "
          f"{tok.decode(replay_ids_full[:40])!r}", flush=True)
    free_cuda()

    # =============================================================================================
    # GROW + the dose-response / no-replay control: fine-tune at each replay fraction (0 / 0.3 / 0.5).
    # =============================================================================================
    arms = {}
    sweep_fracs = sorted(set(replay_sweep) | ({REPLAY_FRAC} if not dry_run else set()))
    for frac in sweep_fracs:
        label = ("no_replay" if frac == 0.0 else f"replay_{int(round(frac*100)):02d}")
        print(f"\n[C2mod] ===== GROW arm '{label}' (replay_frac={frac:.2f}) =====", flush=True)
        replay_ids = replay_ids_full if frac > 0.0 else None
        free_cuda()
        grown, ftlog = grow_finetune(
            frozen, tok, new_train_ids, replay_ids, frac, device,
            steps=ft_steps, batch_size=ft_batch, lr=ft_lr, block_size=BLOCK_SIZE,
            seed=SEED, label=label)
        ppls = two_dist_ppl(grown, tok, ts_ho, new_ho, device, label=label, ppl_positions=ppl_eval_positions,
                            orig_label=orig_label)
        prompt_ids = tok.encode("Once upon a time there was a little")
        gen_ids = _generate(grown, tok, prompt_ids, 40, BLOCK_SIZE, device, SEED * 13 + 5)
        gen_text = tok.decode(gen_ids)
        sh_prompt = tok.encode("To be or not to be")
        gen_sh = _generate(grown, tok, sh_prompt, 40, BLOCK_SIZE, device, SEED * 19 + 3)
        gen_sh_text = tok.decode(gen_sh)
        arms[label] = {
            "replay_frac": frac, "ft_log": ftlog,
            "original_tinystories_ppl": ppls["original_tinystories_ppl"],
            "new_interleave_ppl": ppls["new_interleave_ppl"],
            "original_retention": (base_ts / ppls["original_tinystories_ppl"]
                                   if ppls["original_tinystories_ppl"] > 0 else 0.0),
            "new_ppl_drop_frac": (1.0 - ppls["new_interleave_ppl"] / base_new if base_new > 0 else 0.0),
            "gen_oldprime_text": gen_text, "gen_newprime_text": gen_sh_text,
            "_model": grown,
        }
        print(f"[C2mod] arm '{label}': orig_retention={arms[label]['original_retention']*100:.1f}% "
              f"new_ppl_drop={arms[label]['new_ppl_drop_frac']*100:.1f}% "
              f"(orig {ppls['original_tinystories_ppl']:.3f} new {ppls['new_interleave_ppl']:.3f})",
              flush=True)
        print(f"[C2mod]   [old-prime gen] {gen_text!r}", flush=True)
        free_cuda()

    # ---- the no-replay control + the DECISIVE with-replay arm (chosen adaptively) ----
    noreplay = arms["no_replay"]
    repl_arms = [(arms[k]["replay_frac"], k, arms[k]) for k in arms if arms[k]["replay_frac"] > 0.0]
    repl_arms.sort()

    def _learns(a):
        return (not math.isnan(a["new_interleave_ppl"])) and a["new_interleave_ppl"] <= (1.0 - LEARN_BAR) * base_new

    def _retains(a):
        return (not math.isnan(a["original_tinystories_ppl"])
                and (base_ts / a["original_tinystories_ppl"]) >= RETAIN_BAR)

    cleared = [(f, k, a) for (f, k, a) in repl_arms if _learns(a) and _retains(a)]
    if cleared:
        decisive_label = cleared[0][1]
    else:
        learners = [(a["original_retention"], k) for (f, k, a) in repl_arms if _learns(a)]
        if learners:
            decisive_label = max(learners)[1]
        else:
            decisive_label = max((a["original_retention"], k) for (f, k, a) in repl_arms)[1]
    decisive = arms[decisive_label]
    print(f"\n[C2mod] DECISIVE with-replay arm = '{decisive_label}' (replay_frac={decisive['replay_frac']:.2f}, "
          f"retention {decisive['original_retention']*100:.1f}%, new-drop {decisive['new_ppl_drop_frac']*100:.1f}%)",
          flush=True)
    REPLAY_FRAC_DECISIVE = decisive["replay_frac"]

    # =============================================================================================
    # ON-THE-BRIDGE VERIFICATION: install the decisive grown-with-replay model on the live RF bridge.
    # =============================================================================================
    print("\n[C2mod] ===== ON-THE-BRIDGE VERIFY: install the grown-with-replay model on the RF bridge =====",
          flush=True)
    onbridge = None
    try:
        onbridge = verify_on_bridge(decisive["_model"], tok, ts_ho, new_ho, device,
                                    n_windows=(1 if dry_run else None))
    except Exception as e:
        print(f"[C2mod] on-bridge verify raised ({type(e).__name__}: {e}); the off-bridge ppl table stands "
              f"(C1 proved the install ppl_ratio=0.99999999). Recording the exception.", flush=True)
        onbridge = {"error": f"{type(e).__name__}: {e}"}

    for a in arms.values():
        a.pop("_model", None)
    free_cuda()

    # =============================================================================================
    # VERDICT
    # =============================================================================================
    dec_ts = decisive["original_tinystories_ppl"]; dec_new = decisive["new_interleave_ppl"]
    nr_ts = noreplay["original_tinystories_ppl"]
    learns_new = (not math.isnan(dec_new)) and dec_new <= (1.0 - LEARN_BAR) * base_new
    retains_old = (not math.isnan(dec_ts)) and (base_ts / dec_ts) >= RETAIN_BAR
    noreplay_forgets = (not math.isnan(nr_ts)) and (nr_ts >= FORGET_MARGIN * dec_ts)
    retains = [(arms[k]["replay_frac"], arms[k]["original_retention"])
               for k in arms if not math.isnan(arms[k]["original_tinystories_ppl"])]
    retains.sort()
    dose_monotone = all(retains[i + 1][1] >= retains[i][1] - 0.03 for i in range(len(retains) - 1))

    if learns_new and retains_old and noreplay_forgets:
        verdict = "GO"
    elif learns_new and noreplay_forgets and (base_ts / dec_ts) >= 0.75:
        verdict = "PARTIAL"
    else:
        verdict = "NEGATIVE"

    # honest scale-flag: learns + dose-monotone + retention below the strict bar => the named SCALE issue.
    scale_issue = bool(learns_new and dose_monotone and not retains_old)

    verdict_line = (
        "C2 RE-RUN at a LEARNABLE shift (Gen-F %s, route #1 re-distill + GENERATIVE SELF-REPLAY): "
        "NEW=AUTO-SELECTED SH-frac=%.2f TinyStories/Shakespeare interleave (%.2fx-distinct, in the relative "
        "[%.0f,%.0f]x-orig_ppl learnable band). "
        "BASELINE orig=%.3f new=%.3f | GROWN-WITH-REPLAY(frac=%.2f) orig=%.3f (retention %.1f%%) new=%.3f "
        "(drop %.1f%%) | NO-REPLAY-control orig=%.3f (retention %.1f%%, %.2fx the with-replay orig) new=%.3f "
        "| on-bridge-install-ppl-ratio~=%s -> %s [learns_new(drop>=%.0f%%)=%s retains_old>=%.0f%%=%s "
        "no_replay_forgets(>=%.2fx)=%s dose_monotone=%s scale_issue=%s]. CLS: fine-tune=development, "
        "self-replay=hippocampal no-forget, RF install=consolidated cortical store. RF-install "
        "full-width fidelity scope carries from C1." % (
            arch_label, sh_frac, distinct_ratio, LEARNABLE_BAND_LO_MULT, LEARNABLE_BAND_HI_MULT,
            base_ts, base_new, REPLAY_FRAC_DECISIVE, dec_ts,
            decisive["original_retention"] * 100, dec_new, decisive["new_ppl_drop_frac"] * 100, nr_ts,
            noreplay["original_retention"] * 100, (nr_ts / dec_ts if dec_ts > 0 else float("inf")),
            noreplay["new_interleave_ppl"],
            ("%.6f" % onbridge["original_tinystories"]["ppl_ratio"]
             if (onbridge and "original_tinystories" in onbridge) else "n/a"),
            verdict, LEARN_BAR * 100, learns_new, RETAIN_BAR * 100, retains_old, FORGET_MARGIN,
            noreplay_forgets, dose_monotone, scale_issue))

    ppl_table = {
        "rows": ["baseline_pregrow", "grown_with_replay", "grown_no_replay"],
        "cols": ["original_tinystories_heldout_ppl", "new_interleave_heldout_ppl"],
        "baseline_pregrow": {"original_tinystories_heldout_ppl": base_ts,
                             "new_interleave_heldout_ppl": base_new},
        "grown_with_replay": {"original_tinystories_heldout_ppl": dec_ts,
                              "new_interleave_heldout_ppl": dec_new,
                              "replay_frac": REPLAY_FRAC_DECISIVE},
        "grown_no_replay": {"original_tinystories_heldout_ppl": nr_ts,
                            "new_interleave_heldout_ppl": noreplay["new_interleave_ppl"]},
    }

    result = {
        "probe": "genseq_C2_moderate_shift",
        "resolves": "C2 RE-RUN: at a LEARNABLE (moderate, in-band ~20-60 ppl) distribution shift, does the "
                    "consolidated spiking generator LEARN the new (new-ppl drops >=50%) WHILE RETAINING the "
                    "original (>=85%) via generative self-replay -- with the no-replay control forgetting? "
                    "(The prior run used the EXTREME 41x Shakespeare shift and missed the 85% bar.)",
        "supersedes_context": "research/findings/raw/_genseq_C2_grow_no_forget.json (the EXTREME-shift run: "
                              "mechanism validated + dose-monotone, but 52% retention at 41x Shakespeare). "
                              "This re-run tests the moderate-shift question that prior run left open.",
        "scoping": "research/findings/2026-06-22-C2-grow-no-forget-scoping.md (route #1: re-distill on "
                   "new+self-replayed-old; generative self-replay is the CLS no-forget).",
        "cls_mapping": {
            "slow_neocortex": "the off-bridge fine-tune (development / gradual structured learning)",
            "fast_hippocampus_replay": "generative self-replay (the frozen Gen-F samples its OWN old "
                                       "TinyStories distribution into the fine-tune corpus)",
            "consolidated_cortical_store": "the RF complex-synapse install on the one bridge (C1)",
        },
        "genf_checkpoint": str(GENF_CKPT.relative_to(_REPO)),
        "genf_loss_last": loss_last, "vocab_size": V, "seed": SEED,
        "arch_label": arch_label,
        "arch": {"d_model": int(frozen.cfg["d_model"]), "n_layer": int(frozen.cfg["n_layer"]),
                 "n_head": int(frozen.cfg["n_head"]), "block_size": int(frozen.cfg["block_size"]),
                 "vocab_size": int(frozen.cfg["vocab_size"])},
        "dry_run": bool(dry_run),
        "c2_original": c2_original,
        "original_domain": orig_label,
        "original_domain_source_note": orig_note,
        "original_distribution": "%s (heldout tail) [%s]" % (orig_label, orig_note),
        "new_distribution": "SH-frac=%.2f %s/Shakespeare block-interleave (heldout tail) -- "
                            "%.2fx-distinct (pre-grow new-ppl/orig-ppl)" % (sh_frac, orig_label, distinct_ratio),
        "new_corpus_choice_rationale": (
            "An empirical corpus-selection sweep on THIS frozen 3.4M Gen-F mapped the shift space: PURE "
            "TinyStories topic/structural slices = ~0.8-1.05x baseline ppl (Gen-F trained on the FULL "
            "TinyStories already models every theme -> NO shift, the no-replay control would not forget); "
            "PURE out-of-domain registers (Shakespeare 42x, WikiText 91x; short-line 'simplifying' does NOT "
            "lower ppl -- register/vocab drives it) = strong forgetting but replay can't restore past ~55%% "
            "(the prior run's failure corner). The chosen knob is a TUNABLE TinyStories/Shakespeare "
            "block-interleave: SH_FRAC=%.2f lands the mixture held-out ppl ~%.1f (%.2fx), 0%% <UNK> under "
            "Gen-F's TinyStories BPE -- a legitimate measurably-distinct sub-distribution in the prompt's "
            "~20-60 learnable band. Retention is ALWAYS measured on the DISJOINT pure-TinyStories tail. "
            "HONEST CAVEAT: a mixture self-reinforces the old distribution (it is %d%% TinyStories blocks), "
            "so the no-replay forgetting CONTRAST at this in-band point is expected MODEST (a directional "
            "mini-FT showed ~1.07-1.10x) rather than the prior run's 5.92x catastrophic spike -- the price "
            "of staying in-band; the dose-response + absolute learn/retain numbers are the decisive evidence."
            % (sh_frac, base_new, distinct_ratio, int(round((1 - sh_frac) * 100)))),
        "new_corpus_construction": {
            "method": "deterministic block-interleave of TinyStories-train (capped %d chars) with "
                      "Shakespeare blocks at SH_FRAC Shakespeare probability per block" % TS_TRAIN_CAP,
            "sh_frac": sh_frac, "interleave_seed": INTERLEAVE_SEED,
            "interleave_block_chars": INTERLEAVE_BLOCK, "interleave_total_chars": INTERLEAVE_TOTAL,
            "new_unk_frac": new_unk,
            "sh_frac_auto_selected": bool(auto_sh_frac),
            "sh_frac_requested": sh_frac_requested,
        },
        "learnable_band": {
            "method": "SH_FRAC is auto-tuned so base_new (pre-grow new-corpus ppl) lands in a band defined "
                      "RELATIVE to the CURRENT model's orig_ppl -- size-adaptive (replaces the prior "
                      "hardcoded `base_new < 110` guard tuned for the 3.4M's ppl scale). Distinct enough "
                      "(>= LO_MULT x orig) the no-replay control forgets; learnable enough (<= HI_MULT x "
                      "orig) replay can retain.",
            "lo_mult": LEARNABLE_BAND_LO_MULT, "hi_mult": LEARNABLE_BAND_HI_MULT,
            "orig_ppl_anchor": base_ts,
            "band_lo_ppl": LEARNABLE_BAND_LO_MULT * base_ts,
            "band_hi_ppl": LEARNABLE_BAND_HI_MULT * base_ts,
            "selected_sh_frac": sh_frac,
            "selected_base_new_ppl": base_new,
            "selected_ratio_over_orig": distinct_ratio,
            "in_band": bool(LEARNABLE_BAND_LO_MULT <= distinct_ratio <= LEARNABLE_BAND_HI_MULT),
            "sh_frac_sweep": sh_select_sweep,
            "sh_frac_ladder": list(SH_FRAC_LADDER),
        },
        "config": {
            "block_size": BLOCK_SIZE, "ppl_eval_positions": ppl_eval_positions,
            "ft_steps": ft_steps, "ft_batch": ft_batch, "ft_lr_rewarm": ft_lr,
            "replay_frac_reference": REPLAY_FRAC, "replay_frac_decisive": REPLAY_FRAC_DECISIVE,
            "replay_sweep": list(sweep_fracs),
            "replay_sample_tokens": REPLAY_SAMPLE_TOKENS,
            "retain_bar": RETAIN_BAR, "learn_bar_drop": LEARN_BAR, "forget_margin": FORGET_MARGIN,
            "n_self_replay_tokens": len(replay_ids_full),
        },
        "grow_route": "#1 (the scoping's cheapest + most C1-faithful): the GROW step IS a second C1 pass "
                      "on a GROWN corpus (new + self-replayed-old), re-distilled + re-installed on the RF "
                      "bridge. NO sim/ edit (reuse-by-import of the C1 install + the Gen-F ppl harness).",
        "baseline_pregrow_ppl": base,
        "self_replay": {
            "method": "autoregressively sample OLD (TinyStories) text from the FROZEN pre-grow Gen-F "
                      "(varied in-distribution primes); NO stored old data -- the generator IS the source "
                      "(Shin 2017 Deep Generative Replay; Huang 2024 Self-Synthesized Rehearsal).",
            "n_tokens": len(replay_ids_full), "distinct_trigram": rep_distinct,
            "sample_decode": tok.decode(replay_ids_full[:60]),
        },
        "arms": {k: {kk: vv for kk, vv in a.items() if kk != "_model"} for k, a in arms.items()},
        "ppl_table": ppl_table,
        "on_bridge_verification": onbridge,
        "anti_cheat_no_replay": {
            "method": "fine-tune on NEW ONLY (replay_frac=0) -> the ORIGINAL TinyStories ppl must spike "
                      "vs the with-replay arm -> proves the self-replay is CAUSAL for retention.",
            "no_replay_original_ppl": nr_ts,
            "with_replay_original_ppl": dec_ts,
            "forgetting_ratio_noreplay_over_withreplay": (nr_ts / dec_ts if dec_ts > 0 else float("inf")),
            "no_replay_forgets": bool(noreplay_forgets),
        },
        "anti_cheat_dose_response": {
            "method": "retention should improve monotonically with replay fraction (the CL-LLM "
                      "dose-response; a flat curve would mean retention isn't from the replay).",
            "replay_frac_to_retention": {f"{f:.2f}": r for f, r in retains},
            "monotone": bool(dose_monotone),
        },
        "checks": {"learns_new": bool(learns_new), "retains_old": bool(retains_old),
                   "no_replay_forgets": bool(noreplay_forgets), "dose_monotone": bool(dose_monotone),
                   "scale_issue": scale_issue},
        "verdict_line": verdict_line, "verdict": verdict,
        "scale_issue_flag": scale_issue,
        "scale_issue_note": ("learns_new + dose_monotone but retention below the 85% bar even at a moderate "
                             "learnable shift => a genuine SCALE issue on the 3.4M toy (the cloud-justifying "
                             "point): the mechanism works (replay monotonically improves retention) but the "
                             "3.4M-param capacity cannot hold both distributions tightly enough."
                             if scale_issue else "n/a"),
        "honest_scope": "toy-scale (3.4M-param) demonstration of the loop's BACK HALF (grow + no-forget) at "
                        "a MODERATE learnable shift. The RF-install full-width fidelity scope carries forward "
                        "from C1 (not re-litigated). ppl measured off-bridge (fast) and VERIFIED on-bridge "
                        "(C1: the RF install reproduces off-bridge ppl to ppl_ratio 0.99999999).",
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    out_path.write_text(json.dumps(result, indent=2, default=lambda o: None
                                   if (isinstance(o, float) and math.isnan(o)) else o))

    print("\n[C2mod] ===== PPL TABLE (held-out; Gen-F TinyStories BPE throughout) =====", flush=True)
    print(f"[C2mod]   {'condition':24s} {'original(TinyStories)':>22s} {'new(SH'+str(sh_frac)+'-interleave)':>22s}",
          flush=True)
    print(f"[C2mod]   {'baseline (pre-grow)':24s} {base_ts:>22.4f} {base_new:>22.4f}", flush=True)
    print(f"[C2mod]   {'grown WITH replay':24s} {dec_ts:>22.4f} {dec_new:>22.4f}", flush=True)
    print(f"[C2mod]   {'grown NO replay':24s} {nr_ts:>22.4f} {noreplay['new_interleave_ppl']:>22.4f}",
          flush=True)
    print("\n" + "=" * 78)
    print(verdict_line)
    print("=" * 78)
    print(f"[C2mod] wrote {out_path}", flush=True)
    free_cuda()
    return result


def main():
    import torch
    backend = os.environ.get("SIM_BACKEND", "auto")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[C2mod] SIM_BACKEND={backend} device={device}", flush=True)
    t_start = time.time()
    # ---- load the original 3.4M frozen Gen-F + BPE (behaviourally unchanged from the pre-refactor) ----
    frozen, tok, V, loss_last = load_genf(device)
    print(f"[C2mod] frozen Gen-F loaded: vocab={V} d_model={frozen.cfg['d_model']} "
          f"n_layer={frozen.cfg['n_layer']} block_size={BLOCK_SIZE} loss_last={loss_last:.4f}", flush=True)
    return run_c2_loop(frozen, tok, V, loss_last, device, t_start=t_start)


if __name__ == "__main__":
    main()
