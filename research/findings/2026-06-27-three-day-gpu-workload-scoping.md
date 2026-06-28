# The single best ~3-day GPU workload — SCOPING (read-only; NO edits, NO GPU launch) (2026-06-27)

**VERDICT (one line):** run the **C2 generative grow-without-forget LOOP at 100M params with the CLEAN
recipe** — a properly-regularized, early-stopped 100M Gen-F trained on the **143 MB SimpleWiki** corpus
(not the data-bound 8 MB TinyStories), then C1-consolidated onto the RF bridge, then the C2 grow+no-forget
loop **with the now-default fixed FT_LR=1e-5**. This is the decisive demonstration of the full generative
loop at the scale the project has been pointing at since the 3.4M capacity wall — and **the prior "scale
wall" verdicts (3.4M + 30M) are now BOTH invalid for two independent, confirmed reasons** (below), so this
run resolves a genuinely-open question rather than re-confirming a known result. It saturates ~3 days
productively (the TRAIN stage is inherently long-horizon and compute-bound at 100M, unlike the
hours-maturing develop loop), fits 24 GB (~10-11 GB real), and is launched by ONE resumable command.

The exact command, the why-it-saturates-3-days, the artifact, the VRAM/throughput, and the one cheap
de-risk are in §4 (VERDICT). The ranked alternatives are in §3.

---

## 0. Context recap (why this scoping exists)

The owner wants ~3 days of a 24 GB RTX 3090 used PRODUCTIVELY. The longitudinal develop loop (the live run)
MATURES in ~hours — CYCLE 692 confirms it cannot fill 3 days with NEW learning: the curriculum learns
~24 concepts/day and the composer's recall caps at ~hundreds of concepts at feasible D; after the vocab is
learned it just consolidates (repetitive). The leading candidate the owner flagged is a long
generative-training run (the free-generation frontier — `project_generative_sequence_frontier`), because
backprop pretraining is *inherently* long-horizon.

This doc is a deep read of the generative harness + the develop infrastructure + the alternatives, and a
ranked verdict. **READ-ONLY: no code/sim/GPU edit; no GPU job launched (the GPU is busy with the live
320-concept develop run).**

---

## 1. The generative-sequence frontier — what is the READY harness, and what's its true state?

### 1.1 The harness is built, validated, and resumable

The full generative LOOP harness exists and is reuse-by-import (NO `sim/` edit anywhere in the arc):

- **`research/runners/_genseq_C2_scaleup_runner.py`** — the **3-stage, RESUMABLE, near-one-command**
  pipeline, purpose-built for exactly this long run:
  - **Stage 1 — TRAIN** the Gen-F (`research.runners.tiny_transformer_train.train_tiny_gpt`, a
    `sim.tiny_transformer.TinyGPT`). Kill-safe atomic resume (`tmp + os.replace`, resumes from the `.pt`,
    flushes on Ctrl-C); size-aware regularization knobs (`--dropout/--weight-decay/--warmup-steps`); an
    in-loop held-out overfit probe (`--heldout-every`). This is the ~days-scale cost.
  - **Stage 2 — C1 CONSOLIDATE** the trained model onto the RF complex-synapse bridge + VERIFY generation
    (logit fidelity / greedy-token-match / held-out ppl_ratio vs off-bridge). Arch-agnostic; reuse of the
    C1 derisk machinery. Validated bit-exact at 3.4M AND 30M (ppl_ratio 1.0000000, logit spearman 1.0,
    greedy-match 1.0 — see the 30M smoke).
  - **Stage 3 — C2 GROW + NO-FORGET LOOP** (`_genseq_C2_moderate_shift_derisk.run_c2_loop`) — generative
    self-replay, dose-sweep (replay 0.0/0.3/0.5), no-replay forgetting control, on-bridge verify, GO/
    PARTIAL/NEGATIVE verdict.
  - **Each stage is gated on a `.DONE.json` completion marker** under one `run_dir`, so an interrupted
    multi-day run resumes from the last completed stage; Stage 1's train is itself atomic-resumable inside.
- The **C2 LOOP itself is DEMONSTRATED end-to-end at toy scale** (`2026-06-23-generative-loop-DEMONSTRATED.md`,
  CYCLE 478): train → generate → grow → confirm-no-catastrophic-forgetting, multi-seed GO (3/3), with the
  fully-spiking-C1 nonlinearities (LayerNorm/GELU/softmax) all GO. **Both load-bearing gates are MET at toy
  scale: C1 (one fully-spiking bridge) + C2 (no catastrophic forgetting).**

### 1.2 THE KEY FINDING — the prior "scale wall" verdicts are BOTH invalid (two confirmed, independent faults)

The standing narrative (`2026-06-23-C2-moderate-shift-NEGATIVE-scale-wall.md` + the 30M run) is "the 3.4M is
a capacity wall; scale up to ~100M." But reading the actual artifacts reveals the 30M scale-up NEGATIVE
(`research/findings/raw/_genseq_C2_scaleup_30M.json`, retention **48.3%**) is contaminated by **two
independent bugs, neither of which is a capacity wall** — confirmed from git ordering + the saved artifacts:

**Fault A — the 30M C2 loop used the OLD broken FT_LR=3e-4.** The 30M artifact records
`"ft_lr_rewarm": 0.0003`. The FT_LR=1e-5 continual-learning fix (the entire point of CYCLE 478's
`2026-06-23-generative-loop-DEMONSTRATED.md`) was wired into the shared `run_c2_loop` default in commit
**`05c107b0`**, which is AFTER the 30M loop launched (CYCLE 471, `9cf61c06`, with the OLD LR). At FT_LR=3e-4
the fine-tune overwrites the original distribution in ~1500 steps **regardless of replay** — exactly the
~48% retention the 30M shows. This is the demonstrated demo-design bug, NOT 27M capacity.

**Fault B — the 30M base model was BADLY OVERFIT.** Its `stage1_train.DONE.json` shows it trained for
**40,000 steps** on the 8 MB TinyStories (~1.8M tokens ≈ ~50 epochs). The held-out ppl is a textbook
U-curve: it bottomed at **9.66 @ step 7000**, then climbed back to **26.37 @ step 40000** (final). Train
loss 0.34 (≈ ppl 1.4 = memorized). So the frozen Gen-F that fed the C2 loop had held-out ppl **26.4 — ~4×
worse than its own best (9.66) AND ~4× worse than the 3.4M's 6.1**. A model that is overfit on ONE
distribution has no slack to hold a second; the retention deficit is partly this, not capacity.

**The fix for Fault B is already validated:** `_genseq_scaleup_fix_validate.json` ran the corrected recipe
(dropout 0.1 / weight-decay 0.1 / warmup 300 / **~3500 steps ≈ 6 epochs**) and the "fixed" arm bottoms
held-out ppl at **13.08 and is still descending** (no U-turn — not overfit). The scaleup runner's CURRENT
defaults already encode this fix (`DEF_DROPOUT=0.1`, `DEF_WEIGHT_DECAY=0.1`, `DEF_WARMUP=300`,
`DEF_STEPS=3500`, with an explanatory comment block diagnosing the exact 40k-step overfit). And the
scaleup runner's `stage3_c2_loop` calls `run_c2_loop(...)` **without passing `ft_lr`** → it now inherits the
fixed `FT_LR=1e-5` default automatically.

**⇒ The decisive C2 scale-up experiment — "does a bigger, CLEANLY-trained generator clear the ≥85%-retain /
learns-new / no-replay-forgets bars with the corrected loop?" — has NEVER actually been run.** Both prior
NEGATIVEs were artifacts. This is a genuinely-open, high-value question, and it is precisely what a ~3-day
run answers.

### 1.3 The other readiness facts

- **Corpus matters for a bigger model.** TinyStories (8 MB, ~1.8M tokens) is DATA-bound: a 100M model
  Chinchilla-wants ~1.8B tokens but TinyStories has ~1.8M unique → ~1000 epochs = memorization (this is
  literally what bit the 30M). The repo ALSO has **`data/corpus/simplewiki.txt` = 143 MB** (~30M+ tokens)
  and `wikitext.txt` = 4 MB. **SimpleWiki is the right base corpus for 100M** — ~30M tokens supports several
  clean epochs at 100M without the TinyStories memorization trap, and it is a richer register (closer to
  the owner's deep-knowledge goal). (Honest caveat: a register shift from TinyStories changes the C2
  "new-distribution" choice; the C2 loop auto-tunes its shift relative to the base model's own ppl, so it
  adapts — but the shift-selection should be sanity-checked in the smoke.)
- **C1 install is NOT the VRAM concern at any scale** — even at 500M the 4 RF bridges total 0.19 GB; the
  install is per-matvec-*shape* (4 shapes), not per-layer. The binding VRAM constraint is the Stage-1/3
  fine-tune (AdamW 4×params + activations).

---

## 2. Quantify the ~3-day run (100M, the decisive size)

Param sizing (formula verified against the known 3.45M and 27.38M):

| nominal | d | L | H | V | block | exact params |
|---|---|---|---|---|---|---|
| 3.4M (toy) | 256 | 4 | 4 | 513 | 128 | 3,454,976 |
| 30M (stale run) | 512 | 8 | 8 | 2049 | 128 | 27,383,808 |
| **100M (decisive)** | **768** | **12** | **12** | **2048** | **512** | **88,594,944 (88.6M)** |
| 200M (safety net) | 1024 | 16 | 16 | 2048 | 512 | 206,260,224 |

**VRAM (measured + estimated):** 30M smoke measured peak-train **0.78 GB** (weights only; the C2 fine-tune
adds AdamW+grads+activations). Scaling the scoping's verified breakdown: 100M fine-tune ≈ **8 GB estimate +
~2-3 GB CUDA/torch context = ~10-11 GB real**, comfortably under 24 GB. The VRAM wall is ~500M; **cloud is
NOT justified** (per `feedback_long_local_runs_ok_confirm_cloud_cause`: cloud only for a genuine >24 GB
wall).

**Throughput (measured anchors):**
- Stage-1 TRAIN at 30M: `_genseq_scaleup_fix_validate.json` = **611 s / 3500 steps at B=24 → ~175 ms/step**
  (~94k tok/s at B=24×T=128). (A second 30M train logged 71 ms/step at 40k steps — likely a faster path /
  resume timer; I use the conservative 175 ms/step anchor.)
- Stage-2 C1 consolidate at 3.4M = 2102 s; scales gently (overhead-heavy, per-position RF loop).
- Stage-3 C2 loop at 30M = ~1198 s (3 arms × 1500 ft-steps + replay-sample + on-bridge verify).

**Wallclock ETA at 100M (this is the load-bearing "fills 3 days" calc):** the TRAIN stage dominates and is
the one component that grows with the token budget. A 100M model is ~6× the 30M FLOPs/token; at the measured
~175 ms/step (30M, fp32) scaled FLOP-linearly → **~1.0 s/step at 100M, B=16**. The token budget is the lever
that fills 3 days:

| TRAIN budget | steps (B=16, T=512) | tokens | Stage-1 wall @ ~1 s/step | + C1 (~4 h) + C2 (~0.5 h) | **TOTAL** |
|---|---|---|---|---|---|
| ~3 clean SimpleWiki epochs | ~90M tok / (16·512) ≈ **11k steps** | ~90M | ~3 h | ~4.5 h | ~7.5 h (TOO SHORT) |
| Chinchilla-ish ~1.8B tok | ~220k steps | ~1.8B | **~61 h** | ~4.5 h | **~65 h ≈ 2.7 days** |
| ~2.0B tok (round up) | ~245k steps | ~2.0B | **~68 h** | ~4.5 h | **~72 h = 3.0 days** |

**⇒ A ~2B-token TRAIN budget (~245k steps) at 100M lands the full loop at ~3 days.** This is genuinely
compute-bound work (a 100M model at ~1 s/step is far past the 3.4M's 89%-overhead regime — the FLOPs are
real), so the 3 days are *productive training*, not idle overhead. SimpleWiki's ~30M tokens supports ~60+
epochs at this budget — more repetition than ideal, so **the cleaner framing is to set the budget by
wallclock (`--steps` for ~3 days) with strong regularization + the held-out probe watching for the overfit
U-turn**, early-stopping/best-checkpointing on held-out ppl. (See the de-risk in §4.)

Note: the runners are **fp32**. Enabling TF32/AMP (a small, safe, separate runner change) would ~3-4× the
throughput — which would let 100M finish a clean ~3-epoch run in ~1-2 h, OR (the point here) let a ~3-day
budget train a 200M model or many more tokens. fp32 is assumed throughout (the code path); TF32 is the
obvious accelerator lever, not assumed.

---

## 3. Alternatives — ranked by (project value) × (genuinely-uses-3-days) × (ready/self-driven)

| # | workload | project value | fills 3 days? | ready / self-driven? | net |
|---|---|---|---|---|---|
| **1** | **100M C2 loop, clean recipe + fixed LR** (the VERDICT) | **HIGH** — resolves the now-invalidated "scale wall"; demonstrates the generative loop at the target scale; advances `project_generative_sequence_frontier` (an owner-approved frontier) | **YES** — TRAIN is compute-bound at 100M; ~2B-tok budget = ~3 days of real training | **YES** — ONE resumable command, `.DONE.json` stage markers, atomic-resume train | **BEST** |
| 2 | scale the brain3000pos stream-cortex training to far more windows (better codes → directly improves the develop run's recall) | MEDIUM-HIGH — the CYCLE-691 finding says per-day recall is window-budget-limited (0.33-0.67); more windows → better codes → directly lifts the deep-knowledge develop run | PARTIAL — a window sweep is many short runs, not one 3-day burn; saturates GPU only if batched as a big multi-(window-budget × seed) sweep | YES but bespoke — would need a sweep launcher; the develop loop itself matures (the limit this whole scoping is about) | strong #2, but it's a *fix for the develop run*, not a standalone 3-day artifact |
| 3 | 200M C2 loop (the safety-net size) | HIGH if 100M misses the retain bar; otherwise redundant | YES — ~3 days is roughly the 200M TRAIN alone | YES (same runner, bigger `--d-model`) | do this ONLY if 100M misses (contingent), per the staged cheapest-first plan |
| 4 | big multi-seed validation sweep of a close-out item (e.g. a 6-seed re-confirm of an already-GO result) | LOW-MEDIUM — confirmation, not new capability; most close-outs are already multi-seed | YES if enough seeds × configs | YES (existing runners + aggregators) | fills time but low new value; the project's bar is "honest negatives/new capability ARE the deliverable," not re-confirmation |
| 5 | 320-concept develop run with far more `--max-windows-per-day` | MEDIUM — gives a better developmental artifact (the owner's current pick) | NO — still matures in ~hours then consolidates (the exact problem this scoping addresses) | YES (the live run) | already running as the meanwhile-artifact; not a 3-day *fill* |

**Why #1 over #2:** #2 is genuinely valuable (and is the right *next* GPU job after the develop arc), but it
is a **fix for the develop run's recall**, realized as a sweep of many short runs — it doesn't naturally form
one coherent 3-day artifact, and the develop loop it feeds is the very thing that matures early. #1 is the
only candidate that is BOTH (a) a single, coherent, inherently-long-horizon run that *productively* consumes
3 days of compute, AND (b) resolves a high-value, genuinely-open question (the invalidated scale wall) on an
owner-approved frontier. #2 should be the staged follow-on (it's cheap and directly lifts the deep-knowledge
develop run).

---

## 4. VERDICT — the single best ~3-day GPU workload

### The run: the 100M C2 generative grow-without-forget LOOP, clean recipe + the fixed FT_LR

**The exact command** (one resumable command; the controller launches it when the owner frees the GPU):

```bash
SIM_BACKEND=cupy python -m research.runners._genseq_C2_scaleup_runner \
    --d-model 768 --n-layers 12 --n-heads 12 --vocab-size 2048 --block-size 512 \
    --batch-size 16 --ft-batch 8 \
    --steps 245000 \
    --dropout 0.1 --weight-decay 0.1 --warmup-steps 1000 --heldout-every 1000 \
    --corpus simplewiki \
    --out research/findings/raw/_genseq_C2_scaleup_100M.json \
    --run-dir research/findings/raw/c2_scaleup_100M
```

- `--steps 245000` at B=16×T=512 ≈ **2.0B tokens ≈ ~3 days of TRAIN** (the load-bearing knob; tune to the
  available window). The 3-stage pipeline (TRAIN → C1 → C2) then runs to completion; Stage-1 dominates.
- `--corpus simplewiki` — the 143 MB corpus (not the data-bound 8 MB TinyStories that overfit the 30M).
  *(Verify `fetch_corpus` accepts `"simplewiki"` in the smoke; if the name isn't wired, pass the explicit
  path. See de-risk.)*
- `--ft-batch 8` for the 100M Stage-3 fine-tune VRAM headroom (auto-halves on OOM).
- Resumable: re-running the SAME command continues from the last completed stage / the atomic train
  checkpoint. The owner can PAUSE for gaming by killing it (the train flushes); re-run resumes.

### Why it saturates 3 days PRODUCTIVELY (not maturing early)

Unlike the develop loop (which learns its finite curriculum in ~hours then just consolidates), **generative
pretraining is inherently long-horizon and, at 100M, genuinely compute-bound** (~1 s/step of real FLOPs, far
past the 3.4M's 89%-overhead regime). The 3 days are spent on *new gradient descent over billions of
tokens*, continuously improving the model — the held-out ppl keeps dropping (with regularization preventing
the overfit U-turn). The work does not plateau the way a fixed-vocabulary curriculum does.

### The expected artifact

- A **trained 100M spiking-consolidatable Gen-F** (the largest in the project; held-out ppl target ~6-10 on
  SimpleWiki) — a reusable asset for the grounded-language faculty (`project_grounded_language_faculty`) and
  future generation work.
- **The decisive C2 verdict**: GO (retain ≥85% + learns-new + no-replay-forgets, with the fixed LR) would
  **demonstrate the full generative loop at scale** and retire the "scale wall." A PARTIAL (retention
  75-85%) directionally confirms scale and de-risks 200M. A clean NEGATIVE *with the corrected recipe* would
  be a genuine, much stronger capacity finding than the two contaminated priors — an honest deliverable
  either way.
- On-bridge C1 verification (ppl_ratio ≈ 1.0) re-confirming the 100M install holds the grown generation.

### VRAM / throughput

- **VRAM: ~10-11 GB real** (well under 24 GB; cloud NOT justified — the VRAM wall is ~500M).
- **Throughput: ~1 s/step at 100M B=16 fp32** (scaled FLOP-linearly from the measured 30M ~175 ms/step); TF32
  would ~3-4× this (optional accelerator).

### The de-risk needed FIRST (cheap, ~5-10 min GPU — do when the GPU is free, BEFORE the 3-day commit)

Run the runner's built-in **1-step wiring SMOKE at the 100M arch on SimpleWiki**:

```bash
SIM_BACKEND=cupy python -m research.runners._genseq_C2_scaleup_runner --smoke \
    --d-model 768 --n-layers 12 --n-heads 12 --vocab-size 2048 --block-size 512 \
    --corpus simplewiki
```

It prints the EXACT param count + measured **VRAM peak** (confirm < ~12 GB at this arch/block/batch — block
512 at d768 raises activation memory vs the 30M's block 128, so this measurement is load-bearing), runs 5
train steps (loss decreasing), and dry-runs C1 + C2 (confirm `stages_wired=True`). It also surfaces whether
`--corpus simplewiki` resolves and whether the C2 auto-shift-selection finds an in-band shift for a
SimpleWiki-trained base (if not, fall back to `--corpus tinystories` with strong regularization + a capped
~3500-step train, accepting the data-bound ceiling — still a valid clean re-run of the decisive question,
just at the 30M-arch-comparable corpus). If the 100M block-512 smoke OOMs, drop to `--block-size 256`
(88.4M params, lower activation memory) or `--batch-size 8`.

**One-line owner-facing framing:** *"The '30M scale wall' was a mirage — that run had the old 3e-4
fine-tune-LR bug AND a 40k-step overfit base model (held-out ppl 26 vs its own best 9.6). With both fixed
(the LR is now the default; the regularized ~3500-step recipe is validated), a clean 100M run is the
decisive, never-actually-run test of whether scale demonstrates the generative grow-without-forget loop —
and a ~2B-token budget fills ~3 days of genuine compute, fits 24 GB, one resumable command."*

---

## 5. Honest scope / caveats

- The 100M sizing is a reasoned extrapolation from the (now-corrected) 3.4M + 30M points + the
  scale-vs-forgetting literature, anchored on two real timings — **not a measured curve**. The smoke
  resolves VRAM/wiring; the 3-day run resolves the science.
- The SimpleWiki register shift vs the prior TinyStories base changes the C2 "new-distribution" auto-shift;
  the loop adapts (it tunes relative to the base ppl) but the smoke must confirm an in-band shift is found.
- Throughput is fp32 (the code path); the ~1 s/step 100M figure is FLOP-linear from the measured 30M and
  carries ±30%. The `--steps` budget is the dial — set it to the available wall-clock and let the held-out
  probe + best-checkpoint guard against overfit. The run is resumable, so an over- or under-shoot of the
  3-day window is harmless.
- This is NOT a `sim/` edit (reuse-by-import); the only inputs are the runner CLI flags. The no-confab moat
  is not touched (this is the generative axis, separate from the conversational composer).
- Staged cheapest-first, per the existing scoping: if 100M GOes, done; if PARTIAL, it de-risks 200M; 200M
  (the ~3-day safety net) only if 100M misses. Cloud remains unjustified until ~500M (>24 GB).
