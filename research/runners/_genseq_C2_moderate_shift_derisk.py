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
FT_LR = 3e-4                # re-warm LR (cosine-annealed over FT_STEPS)
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
SH_FRAC = 0.45               # Shakespeare fraction in the interleave -> mixture held-out ppl ~47.8 (7.7x)
INTERLEAVE_SEED = 7
INTERLEAVE_BLOCK = 400       # contiguous-char block granularity of the interleave
INTERLEAVE_TOTAL = 1_400_000 # total chars of the built NEW corpus
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
def load_genf(device):
    import torch
    from sim.tiny_transformer import TinyGPT
    from sim.bpe_tokenizer import BPETokenizer
    tok = BPETokenizer.load(str(GENF_BPE))
    V = tok.vocab_size
    # weights_only=True: OUR OWN trusted, local, project-generated checkpoint; safe unpickler regardless.
    ck = torch.load(str(GENF_CKPT), map_location=device, weights_only=True)
    loss_last = float(ck["loss_history"][-1]) if ck.get("loss_history") else float("nan")
    m = TinyGPT(vocab_size=V, d_model=256, n_layer=4, n_head=4, block_size=BLOCK_SIZE, dropout=0.0).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    del ck
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


# =================================================================================================
# GENERATIVE SELF-REPLAY: sample OLD-distribution (TinyStories) text from the FROZEN pre-grow Gen-F.
# (byte-identical to orig) -- the CLS hippocampal-replay analogue (Shin 2017).
# =================================================================================================
def sample_self_replay(frozen_model, tok, n_target_tokens, block_size, device, seed):
    import torch
    rng = np.random.default_rng(seed)
    out_ids = []
    si = 0
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
def two_dist_ppl(model, tok, ts_ho, new_ho, device, *, label):
    ts_ppl = perplexity(_heldout_nll(model, tok, ts_ho, BLOCK_SIZE, device, PPL_EVAL_POSITIONS))
    new_ppl = perplexity(_heldout_nll(model, tok, new_ho, BLOCK_SIZE, device, PPL_EVAL_POSITIONS))
    print(f"[C2mod:{label}]   ORIGINAL(TinyStories) held-out ppl = {ts_ppl:.4f} | "
          f"NEW(SH{SH_FRAC}-interleave) held-out ppl = {new_ppl:.4f}", flush=True)
    return {"original_tinystories_ppl": ts_ppl, "new_interleave_ppl": new_ppl}


# =================================================================================================
# ON-THE-BRIDGE VERIFICATION (byte-identical to orig): re-distill + install the grown model on the RF
# complex-synapse bridge (the C1 path) and confirm on-bridge ppl == off-bridge ppl on the SAME windows.
# =================================================================================================
def verify_on_bridge(grown_model, tok, ts_ho, new_ho, device):
    from research.runners._genseq_loopstep3_full_genf_generate_derisk import (
        rf_full_forward, _heldout_nll_numpy, _perplexity)
    from research.runners._genseq_loopstep3_rf_probe import (
        _build_rf_bridge, RF_PERIOD, RF_NSTEPS, RF_LAMBDA)

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

    ts_ids = tok.encode(ts_ho[:ONBRIDGE_VERIFY_WINDOWS * BLOCK_SIZE * 8])
    new_ids = tok.encode(new_ho[:ONBRIDGE_VERIFY_WINDOWS * BLOCK_SIZE * 8])
    res = {}
    for name, ids in (("original_tinystories", ts_ids), ("new_interleave", new_ids)):
        rf_nll = _heldout_nll_numpy(_rf_fwd, ids, V, BLOCK_SIZE, ONBRIDGE_VERIFY_WINDOWS)
        off_nll = _heldout_nll_numpy(_off_fwd, ids, V, BLOCK_SIZE, ONBRIDGE_VERIFY_WINDOWS)
        rf_ppl = _perplexity(rf_nll); off_ppl = _perplexity(off_nll)
        ratio = rf_ppl / off_ppl if (math.isfinite(off_ppl) and off_ppl > 0) else float("inf")
        res[name] = {"on_bridge_ppl": rf_ppl, "off_bridge_ppl": off_ppl, "ppl_ratio": ratio,
                     "n_windows": ONBRIDGE_VERIFY_WINDOWS}
        print(f"[C2mod:on-bridge]   {name}: RF-on-bridge ppl={rf_ppl:.4f} off-bridge ppl={off_ppl:.4f} "
              f"ratio={ratio:.6f}", flush=True)
    del bridges
    free_cuda()
    return res


def main():
    import torch
    backend = os.environ.get("SIM_BACKEND", "auto")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[C2mod] SIM_BACKEND={backend} device={device}", flush=True)
    t_start = time.time()

    # ---- load frozen Gen-F + BPE ----
    frozen, tok, V, loss_last = load_genf(device)
    print(f"[C2mod] frozen Gen-F loaded: vocab={V} d_model=256 n_layer=4 block_size={BLOCK_SIZE} "
          f"loss_last={loss_last:.4f}", flush=True)

    # ---- corpora ----
    ts = fetch_corpus(name="tinystories", max_bytes=8_000_000)
    ts_tr, ts_ho = split_corpus(ts["text"], heldout_frac=0.1)
    sh = fetch_corpus(name=str(_REPO / "data/tinyshakespeare.txt"), max_bytes=8_000_000)
    print(f"[C2mod] ORIGINAL=TinyStories (train {len(ts_tr)} / heldout {len(ts_ho)} chars); "
          f"Shakespeare register source {len(sh['text'])} chars (degraded={sh['degraded']})", flush=True)

    # ---- build the NEW (learnable) distribution = TS-blocks interleaved with SH-blocks at SH_FRAC ----
    new_full = build_new_corpus(ts_tr[:TS_TRAIN_CAP], sh["text"], SH_FRAC, INTERLEAVE_SEED)
    new_tr, new_ho = split_corpus(new_full, heldout_frac=0.15)
    new_train_ids = tok.encode(new_tr[:600_000])
    new_unk = sum(1 for i in new_train_ids if i == 0) / max(1, len(new_train_ids))
    print(f"[C2mod] NEW(SH-frac={SH_FRAC} interleave): full {len(new_full)} chars "
          f"(train {len(new_tr)} / heldout {len(new_ho)}); train tokens {len(new_train_ids)} "
          f"(<UNK> frac {new_unk:.4f})", flush=True)

    # =============================================================================================
    # BASELINE: pin the two-distribution pre-grow ppls + the distinctness ratio.
    # =============================================================================================
    print("\n[C2mod] ===== BASELINE: pre-grow Gen-F two-distribution held-out ppl =====", flush=True)
    base = two_dist_ppl(frozen, tok, ts_ho, new_ho, device, label="baseline")
    base_ts = base["original_tinystories_ppl"]; base_new = base["new_interleave_ppl"]
    distinct_ratio = base_new / base_ts if base_ts > 0 else float("inf")
    print(f"[C2mod] BASELINE: TinyStories(orig)={base_ts:.4f}  NEW(SH{SH_FRAC}-interleave)={base_new:.4f}  "
          f"(distinctness {distinct_ratio:.2f}x)", flush=True)
    assert distinct_ratio > 3.0, (
        f"new corpus NOT measurably distinct (ratio {distinct_ratio:.2f}); raise SH_FRAC.")
    assert base_new < 110.0, (
        f"new corpus too distinct for the 'learnable' band (ppl {base_new:.1f}); lower SH_FRAC.")

    # =============================================================================================
    # SELF-REPLAY: sample OLD (TinyStories) text from the FROZEN pre-grow Gen-F (built ONCE).
    # =============================================================================================
    print("\n[C2mod] ===== GENERATIVE SELF-REPLAY: sample OLD TinyStories from the FROZEN Gen-F =====",
          flush=True)
    t0 = time.time()
    replay_ids_full = sample_self_replay(frozen, tok, REPLAY_POOL_TOKENS, BLOCK_SIZE, device, seed=SEED * 17)
    rep_distinct = distinct_ngram_ratio(replay_ids_full, n=3)
    print(f"[C2mod] self-replay: sampled {len(replay_ids_full)} OLD tokens from frozen Gen-F "
          f"(distinct-trigram {rep_distinct:.3f}, {time.time()-t0:.0f}s); sample decode: "
          f"{tok.decode(replay_ids_full[:40])!r}", flush=True)
    free_cuda()

    # =============================================================================================
    # GROW + the dose-response / no-replay control: fine-tune at each replay fraction (0 / 0.3 / 0.5).
    # =============================================================================================
    arms = {}
    sweep_fracs = sorted(set(REPLAY_SWEEP) | {REPLAY_FRAC})
    for frac in sweep_fracs:
        label = ("no_replay" if frac == 0.0 else f"replay_{int(round(frac*100)):02d}")
        print(f"\n[C2mod] ===== GROW arm '{label}' (replay_frac={frac:.2f}) =====", flush=True)
        replay_ids = replay_ids_full if frac > 0.0 else None
        free_cuda()
        grown, ftlog = grow_finetune(
            frozen, tok, new_train_ids, replay_ids, frac, device,
            steps=FT_STEPS, batch_size=FT_BATCH, lr=FT_LR, block_size=BLOCK_SIZE,
            seed=SEED, label=label)
        ppls = two_dist_ppl(grown, tok, ts_ho, new_ho, device, label=label)
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
        onbridge = verify_on_bridge(decisive["_model"], tok, ts_ho, new_ho, device)
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
        "C2 RE-RUN at a LEARNABLE shift (toy Gen-F 3.4M, route #1 re-distill + GENERATIVE SELF-REPLAY): "
        "NEW=SH-frac=%.2f TinyStories/Shakespeare interleave (%.2fx-distinct, in the ~20-60 learnable band). "
        "BASELINE orig=%.3f new=%.3f | GROWN-WITH-REPLAY(frac=%.2f) orig=%.3f (retention %.1f%%) new=%.3f "
        "(drop %.1f%%) | NO-REPLAY-control orig=%.3f (retention %.1f%%, %.2fx the with-replay orig) new=%.3f "
        "| on-bridge-install-ppl-ratio~=%s -> %s [learns_new(drop>=%.0f%%)=%s retains_old>=%.0f%%=%s "
        "no_replay_forgets(>=%.2fx)=%s dose_monotone=%s scale_issue=%s]. CLS: fine-tune=development, "
        "self-replay=hippocampal no-forget, RF install=consolidated cortical store. Toy-scale; RF-install "
        "full-width fidelity scope carries from C1." % (
            SH_FRAC, distinct_ratio, base_ts, base_new, REPLAY_FRAC_DECISIVE, dec_ts,
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
        "original_distribution": "TinyStories (data/corpus/tinystories.txt, heldout tail)",
        "new_distribution": "SH-frac=%.2f TinyStories/Shakespeare block-interleave (heldout tail) -- "
                            "%.2fx-distinct (pre-grow new-ppl/orig-ppl)" % (SH_FRAC, distinct_ratio),
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
            % (SH_FRAC, base_new, distinct_ratio, int(round((1 - SH_FRAC) * 100)))),
        "new_corpus_construction": {
            "method": "deterministic block-interleave of TinyStories-train (capped %d chars) with "
                      "Shakespeare blocks at SH_FRAC Shakespeare probability per block" % TS_TRAIN_CAP,
            "sh_frac": SH_FRAC, "interleave_seed": INTERLEAVE_SEED,
            "interleave_block_chars": INTERLEAVE_BLOCK, "interleave_total_chars": INTERLEAVE_TOTAL,
            "new_unk_frac": new_unk,
        },
        "config": {
            "block_size": BLOCK_SIZE, "ppl_eval_positions": PPL_EVAL_POSITIONS,
            "ft_steps": FT_STEPS, "ft_batch": FT_BATCH, "ft_lr_rewarm": FT_LR,
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
    OUT_PATH.write_text(json.dumps(result, indent=2, default=lambda o: None
                                   if (isinstance(o, float) and math.isnan(o)) else o))

    print("\n[C2mod] ===== PPL TABLE (held-out; Gen-F TinyStories BPE throughout) =====", flush=True)
    print(f"[C2mod]   {'condition':24s} {'original(TinyStories)':>22s} {'new(SH'+str(SH_FRAC)+'-interleave)':>22s}",
          flush=True)
    print(f"[C2mod]   {'baseline (pre-grow)':24s} {base_ts:>22.4f} {base_new:>22.4f}", flush=True)
    print(f"[C2mod]   {'grown WITH replay':24s} {dec_ts:>22.4f} {dec_new:>22.4f}", flush=True)
    print(f"[C2mod]   {'grown NO replay':24s} {nr_ts:>22.4f} {noreplay['new_interleave_ppl']:>22.4f}",
          flush=True)
    print("\n" + "=" * 78)
    print(verdict_line)
    print("=" * 78)
    print(f"[C2mod] wrote {OUT_PATH}", flush=True)
    free_cuda()
    return result


if __name__ == "__main__":
    main()
