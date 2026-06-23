# C2 scale-up scoping — SIZE the bigger generator that can DEMONSTRATE the grow-without-forget loop (2026-06-23)

**Read-only scoping (NO edits, NO GPU runs — every number ESTIMATED from the model arch + corpus + the
known 3.4M timings). Prepares the owner's morning scale-up decision after
`2026-06-23-C2-moderate-shift-NEGATIVE-scale-wall.md` mapped the whole shift axis and concluded the 3.4M
Gen-F is a model-CAPACITY wall (it can't hold two distributions tightly at ANY learnable-and-forgetting
shift). The C2 self-replay MECHANISM and the C1 consolidation are both validated; only the SCALE to
*demonstrate* the loop remains.**

## TL;DR verdict
- **Target size: ~100M params** is the smallest that plausibly clears the ≥85%-retain / learns-new /
  no-replay-forgets bars simultaneously (justification below). 30M is the cheap first probe (likely
  PARTIAL); 200M is the safety margin if 100M still misses.
- **VRAM (RTX 3090, 24 GB): LOCAL up to ~200-300M; the wall is ~500M.** The fine-tune dominates VRAM, NOT
  the RF complex-synapse install (which is < 0.2 GB even at 500M). 100M fine-tune ≈ 8 GB, 200M ≈ 9-12 GB,
  500M ≈ 24 GB (tight, batch-1-ish).
- **Wallclock ETA (full loop = train Gen-F + C1 consolidate + C2 grow-no-forget): 30M ≈ 4 h, 100M ≈ 15 h,
  200M ≈ 60 h.** The 3.4M is ~89% launch/overhead-bound, so wallclock scales *sublinearly* until ~100M.
- **LOCAL-vs-CLOUD: the demonstrating size (100M, ~15 h) is a LOCAL long-wallclock run.** Per
  `feedback_long_local_runs_ok_confirm_cloud_cause`, cloud is justified ONLY by a genuine VRAM wall
  (>24 GB), which does not occur until ~500M. **Recommendation: run 100M locally; do NOT go to cloud.**

---

## 1. What model size demonstrates the loop?

The 3.4M result is not a mechanism failure — it is a **capacity** failure: at the moderate SH_FRAC=0.45
interleave (7.7×-distinct, the in-band sweet spot) the best with-replay arm retained only **53.9%** of the
original distribution (bar = 85%), and even the no-replay control only forgot 1.16× (bar = 1.3×). The whole
shift axis was mapped: pure-distinct shifts (41× Shakespeare) forget hard but cap replay-retention at
~52-55%; in-band mixtures self-reinforce the old distribution so they don't forget enough. **No corpus on
the 3.4M gives BOTH a clean forgetting contrast AND ≥85% retention** — the 3.4M cannot hold two
distributions at once.

**Why a bigger model fixes this (two converging arguments):**

1. **In-domain capacity (the direct argument).** Holding two distributions tightly requires the model to
   have spare representational capacity beyond fitting either one alone. The 3.4M Gen-F reaches TinyStories
   held-out ppl 6.1 — it is near its capacity ceiling *for one distribution*. Adding a second register
   (even a moderate 7.7× shift) forces weight motion that overwrites the first because there is no slack.
   The TinyStories/TinyStories-2 generation of work (Eldan & Li 2023; the GPT-Neo TinyStories models)
   shows coherent small-story modeling is comfortable at ~30M and that *headroom* (the ability to carry
   extra structure — dialogue, multiple registers, instruction-style variation) appears in the
   **30M → 100M+** band, not at single-digit-M. A 3.4M model is ~10× below even the smallest comfortable
   TinyStories model.

2. **The continual-learning-LLM (CL-LLM) scaling literature.** Replay-based continual learning retention is
   strongly capacity-dependent: larger models forget less per unit of new learning at a fixed replay
   fraction (Scialom 2022 fine-tuning-with-replay; Ibrahim 2024 *Simple and Scalable Strategies to
   Continually Pre-train LLMs* — re-warm + small replay matches full retrain, but the **margin widens with
   model size**; Ramasesh 2022 *Effect of scale on catastrophic forgetting* — forgetting decreases roughly
   log-linearly with parameter count, with a notable knee in the 10s-of-M to ~100M range for small-corpus
   LMs). The mechanism the C2 runner uses (generative self-replay, dose-monotone) is exactly the
   replay-based family these results cover.

**Sizing call:** the retention deficit to close is **53.9% → ≥85%** at a 7.7× shift. Ramasesh-style
log-linear forgetting-vs-scale, anchored at the measured 3.4M point, predicts crossing the 85% bar at
roughly **30-100M** params — with 30M the optimistic edge (likely PARTIAL: learns-new + clears the
forgetting contrast, retention ~75-85%) and **100M the conservative size that clears all three bars with
margin**. 200M is the safety net. This is a reasoned extrapolation from the single measured 3.4M capacity
point + the scale-vs-forgetting literature, **not a guarantee** — hence the staged 30M→100M plan below.

---

## 2. VRAM at the target sizes (RTX 3090, 24 GB)

Canonical TinyGPT configs (param formula verified against the known 3.45M at d=256/L=4/V=513/blk=128):

| nominal | d | L | V | block | params |
|---|---|---|---|---|---|
| 3.4M (baseline) | 256 | 4 | 513 | 128 | 3.5M |
| **30M** | 512 | 8 | 2048 | 256 | 27.4M |
| **100M** | 768 | 12 | 2048 | 512 | 88.6M |
| **200M** | 1024 | 16 | 2048 | 512 | 206.3M |
| **500M** | 1408 | 22 | 4096 | 1024 | 536.8M |

**VRAM breakdown — the GROW fine-tune is the binding constraint** (it carries AdamW m+v + grads + weights =
4× params in fp32, plus training activations). The RF install is negligible by comparison.

| nominal | FT batch | weights+grad+Adam (4×params, fp32) | activations (fp32, no checkpt) | RF complex-CSR install (re+im, 4 bridges) | **total FT VRAM** | fits 24 GB? |
|---|---|---|---|---|---|---|
| 3.4M | 48 | 0.06 GB | 0.43 GB | 0.006 GB | **0.5 GB** | YES (measured: ~3-5 GB used incl. CUDA ctx) |
| 30M | 32 | 0.44 GB | 2.6 GB | 0.027 GB | **3.0 GB** | YES |
| 100M | 16 | 1.4 GB | 6.8 GB | 0.055 GB | **8.2 GB** | YES (comfortable) |
| 200M | 8 | 3.3 GB | 6.0 GB | 0.092 GB | **9.3 GB** | YES |
| 500M | 4 | 8.6 GB | 15.4 GB | 0.189 GB | **24.0 GB** | **MARGINAL — the wall** |

Notes:
- The measured 3.4M C2 fine-tune used ~3-5 of 24 GB (≈ the 0.5 GB estimate + CUDA/cuPy/torch context
  overhead ~2-3 GB). The same fixed ~2-3 GB context overhead applies at every size, so add it to the table:
  100M ≈ 10-11 GB real, 200M ≈ 11-13 GB real, 500M would exceed 24 GB at any usable batch.
- **The RF complex-synapse install is NOT the VRAM concern.** Even at 500M the four install bridges
  (dd / mlp1 / mlp2 / head, each holding re+im fp32 weights) total **0.189 GB**. The install is per-matvec-
  *shape* (4 shapes), not per-layer, so it does not grow with depth. C1's on-bridge consolidation is
  memory-cheap; it is *time*-bound (the per-position RF resonate loop), not VRAM-bound.
- **WHERE the VRAM wall is: ~500M.** 200-300M is the practical local ceiling with comfortable batch sizes;
  500M is achievable only at batch ~1-2 (slow, fragile to OOM). Above ~600M-1B the fine-tune optimizer
  state alone (4× params) crosses into not-fitting territory — that is the genuine cloud-justifying regime,
  and it is far above the size needed to demonstrate the loop.

---

## 3. Throughput + wallclock ETA (RTX 3090)

**Two measured anchors:**
- **TRAIN:** `generator_f_gate.json` = 3464 s for 3 seeds × 2 trains (real + word-shuffle control) = **6 full
  12000-step trains** → **577 s per 12000-step 3.4M train** (B=64, T=128) = 48 ms/step ≈ **170k tok/s**.
- **C1 consolidate:** `_genseq_loopstep3_full_genf_generate.json` = **2102 s** for the full 4-block
  generator on the RF bridge (3.4M, ppl_ratio 0.99999999 — the install is exact).
- **C2 grow-no-forget:** `_genseq_C2_moderate_shift.json` = **448 s** for 3 arms × 1500 ft-steps (B=48) +
  replay-sample + on-bridge verify (3.4M).

**Scaling model (the load-bearing observation): the 3.4M is ~89% launch/overhead-bound.** Decomposing the
measured 3.4M per-token time into a fixed kernel-launch/python-loop floor + an asymptotic compute term
(6·N/token at a conservative 30 TFLOP/s fp32 sustained) gives overhead = 5.24 µs/tok vs compute = 0.63
µs/tok at 3.4M → the compute fraction is only ~11%. **Consequence: wallclock scales SUBLINEARLY in params
until ~100M** (the fixed overhead amortizes as the per-step compute grows), then approaches FLOP-linear
above ~100-200M. (fp32 assumed throughout — the runners use fp32; TF32/AMP would cut the compute term ~3-4×
and shorten the large-model ETAs, but I do not assume it since the code path is fp32.)

**Token budgets:** TRAIN tokens sized Chinchilla-ish (~20 tok/param, floored at the proven 98M the 3.4M
converged on — we only need to *demonstrate* the loop, not hit SOTA ppl). C1 + C2 are fixed small budgets
(per-position RF loop + 3×1500 ft-steps); they scale gently (overhead-heavy).

| nominal | params | train tokens | **TRAIN** | **C1 consolidate** | **C2 grow-no-forget** | **TOTAL wallclock** |
|---|---|---|---|---|---|---|
| 3.4M (baseline) | 3.5M | 98M | 0.16 h | 0.58 h | 0.09 h | **0.84 h** (matches measured) |
| **30M** | 27.4M | 549M | 1.6 h | 2.0 h | 0.2 h | **~4 h** |
| **100M** | 88.6M | 1.77B | 11 h | 4.1 h | 0.5 h | **~15 h** |
| **200M** | 206M | 4.1B | 52 h | 6.8 h | 0.6 h | **~60 h** |
| 500M | 537M | 10.7B | 328 h | 12 h | 1.2 h | ~340 h (impractical locally) |

- The TRAIN step dominates the total at ≥100M (it is the one component that grows with the token budget).
  If the Chinchilla token budget is relaxed (a smaller "demonstrate-the-loop" corpus pass, e.g. cap at
  ~500M-1B tokens since the toy converged at 98M), the 100M TRAIN drops to ~3-6 h and the **100M TOTAL to
  ~8-10 h**. The table is the conservative (full-budget) end.
- **Smallest demonstrating size (100M): ~15 h conservative / ~8-10 h with a relaxed token budget — a single
  overnight-to-one-day LOCAL run.**

---

## 4. Local-vs-cloud verdict

**LOCAL.** Per `feedback_long_local_runs_ok_confirm_cloud_cause` (long local runs are fine with an ETA;
cloud is reserved for a genuine VRAM wall, >24 GB / ~1B+ params):

- The demonstrating size (**100M**) fits in **~10-11 GB real VRAM** (8 GB estimate + ~2-3 GB context) — far
  under 24 GB. Its wallclock is **~15 h** (conservative) to ~8-10 h (relaxed budget) — a long but ordinary
  overnight local run.
- The VRAM wall does not appear until **~500M** (≈ 24 GB, batch-1-marginal); the cloud-justifying regime
  (>24 GB, ~600M-1B+) is **5-10× larger than the size needed to demonstrate the loop**.
- **There is no genuine VRAM cause for cloud here.** Cloud would only be justified if 100M *also* missed
  the retention bar and the next step demanded ~500M-1B (then the >24 GB wall bites). That is a *contingent
  future* decision, not the current one.

---

## Recommended next step for the owner (staged, cheapest-first)

1. **First probe: 30M (~4 h local).** Cheap. If it already clears all three bars (learns-new, retain ≥85%,
   no-replay forgets ≥1.3×) at the SH_FRAC=0.45 moderate shift → the loop is demonstrated and we are done
   far cheaper than expected. Most likely outcome: PARTIAL (retention ~75-85%, forgetting contrast clears)
   — which already *confirms the scale hypothesis directionally* and de-risks the 100M run.
2. **Decisive run: 100M (~15 h local, single overnight/day).** The conservative size that clears the bars
   with margin. This is the recommended decisive demonstration of the full grow-without-forget loop.
3. **Safety net: 200M (~60 h local)** only if 100M still misses — a 2-3 day local run, still no cloud.
4. **Cloud: NOT recommended** unless 200M misses and the next step is ~500M-1B (the only genuine
   >24 GB VRAM wall) — a contingent future decision, not now.

All three stages reuse the existing, validated machinery verbatim (the Gen-F trainer
`tiny_transformer_train.py`, the C1 RF install `_genseq_loopstep3_full_genf_generate_derisk.py`, the C2
loop `_genseq_C2_moderate_shift_derisk.py`) with only the arch hyperparameters (d / L / V / block) and the
fine-tune batch size changed — **no `sim/` edit, no new mechanism**; the larger model is a drop-in to the
same loop. The only code touch is bumping the TinyGPT config + lowering FT_BATCH for VRAM headroom.

### Honesty / caveats
- The size sizing is a **reasoned extrapolation** from a SINGLE measured capacity point (3.4M at 53.9%
  retention) + the scale-vs-forgetting literature, not a measured curve. 30M could under-deliver (→ 100M)
  or 100M could already over-deliver (→ 30M sufficed). The staged plan is built precisely to resolve this
  cheaply.
- Wallclock assumes **fp32** (the code path). Enabling TF32/AMP (a small, safe runner change) would cut the
  TRAIN ETAs ~3-4× and is the obvious throughput lever if the 100M full-budget run feels long — bringing
  100M to ~4-5 h total. Worth flagging as a cheap accelerator, not assumed in the table.
- Throughput scaling is anchored on two real timings and a conservative 30 TFLOP/s sustained-fp32 figure;
  the true large-model wallclock sits inside the [compute-bound, FLOP-linear-from-3.4M] band the model
  brackets. The 100M / 200M numbers are the realistic middle; treat them as ±30%.
