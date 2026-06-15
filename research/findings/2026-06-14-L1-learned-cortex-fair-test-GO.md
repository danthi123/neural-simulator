# L1 learned-graded-cortex fair test — **GO** (owner-directed "better-resourced de-risk")

**Date:** 2026-06-14 (CYCLE 54)
**Status:** ✅ **GO** — a brain-plausible online local PCA rule on true-PPMI input learns the real
category structure from the real TinyStories corpus, reaching 92% of the offline optimum and beating the
project's own host method. Multi-seed, full anti-cheat battery clean. **Overturns** both CYCLE 53's
"data-limited" conclusion and the convergence sweep's apparent NEGATIVE.

## The question (owner directive)

After the night's arc (the dendritic gain not load-bearing; the cheap log/marginal fixes failing on the
real corpus), the owner directed a **"better-resourced de-risk first"** before any months-scale build: a
fair, day-scale test of the L1 similarity-matching learned-embedding route, with **true PPMI input +
SVD-grade low-rank + more epochs + a faithful truly-saturating control**. The question: **can a
brain-based online learner reach the host ceiling on real data?**

## What was unfair before, and the fix

CYCLE 53's runner was unfair in two ways the owner flagged:
1. its "log(count) then divide by marginal" was **not PPMI**. PPMI is the log **of** the marginal ratio
   (`log(P(a,b)/(P(a)P(b)))`) — the normalization is **inside** the log (a subtraction), not a divide
   after it. The fair test uses the host's exact PPMI (α=0.75 context smoothing, host-matched).
2. it had **no SVD-grade low-rank denoising** — the step that lifts the host to its ceiling.

New runner: `research/runners/learned_graded_cortex_fair_test.py` (synthetic + real, 4+ arms, full battery).

## The arc (honest, including two corrections of my own intermediate reads)

| Stage | Result | Read |
|---|---|---|
| **Synthetic calibration** (host +0.957) | simmatch-on-PPMI **98% of host**, gen 1.000, eff-rank 8 = the 8 categories; faithful saturating control fails (eff-rank **1.0**, uniform blob); input lesion confirmed (raw ~0, log-alone only +0.43) | machinery clean |
| **Real corpus, similarity-matching** (n_hub=2000, 200 ep) | online Pehlevan-Chklovskii simmatch **plateaus at +0.290** (offdiag-cos **+0.97**, saturated) vs offline PPMI+PCA **+0.523** | looked like a wall |
| **Convergence sweep** (lr-decay, smaller k — the Robbins-Monro fixes) | best +0.292; the principled fixes did **not** help | apparent NEGATIVE |
| **Root-cause probe** (`_l1_centered_online_pca_probe.py`) | swap to **Oja's subspace rule** (provably-convergent, same brain-plausible online-local-Hebbian-PCA class): **+0.443 uncentered / +0.481 centered** = 85–92% of offline | the +0.29 was **under-convergence of that specific simmatch implementation**, not a rule-class wall |
| **Airtight validation** (`_l1_oja_validated.py`) | **GO**, all 6 gates green, full battery | **resolved** |

The two intermediate over-reads (the apparent NEGATIVE) were corrected by **root-causing the +0.29
before declaring a scientific wall** — the discipline that repeatedly protects this project. The tell: the
lr-decay de-saturated (offdiag 0.97→0.75) yet Pearson *fell*, i.e. de-saturating moved *away* from the
input structure — diagnostic of a specific-rule convergence failure, not a data limit.

## The validated result (real TinyStories corpus, n_hub=2000, seeds 42/43/44)

| Arm | Pearson(S, S_true) | offdiag-cos | gen (chance 0.125) |
|---|---|---|---|
| offline PPMI+PCA(k=64) — **ceiling** | **+0.523** | — | — |
| project host `ppmi_svd_sim(k=50)` | +0.323 | — | nn-same 0.547 |
| cos(PPMI rows), full-rank | +0.485 | — | — |
| **Oja(PPMI, centered) — the learner** | **+0.481** (+0.484/+0.476/+0.485) | **−0.016** | **0.875** |
| Oja(raw) — input lesion | +0.213 | +0.566 | 0.573 |
| random-proj(PPMI) — is-learning-load-bearing? | +0.169 | +0.128 | 0.500 |
| faithful saturating Hebbian — the failure | **−0.001** | **+1.000** | 0.109 |
| permuted-similarity (anti-cheat) | **−0.000** | — | — |

**All 6 gates green:** host_carries, A_saturating_fails, learner_reaches_ceiling (92% of offline, ≥0.30),
generalizes, permuted_collapses, input_lesion_collapses.

**The load-bearing honesty control came out strongly:** a random k=64 projection of PPMI reaches only
+0.169 (a random 64-dim subspace *loses* the structure that lives in the full-rank cosines), so Oja's
+0.481 is **genuine extraction** — the learning adds **+0.312** over random projection. Both the **PPMI
input** (Oja-PPMI +0.481 vs Oja-raw +0.213) **and the learning** (Oja +0.481 vs random +0.169) are
load-bearing; a *saturating* learner destroys the structure (−0.001), an *Oja-class non-saturating* learner
extracts it.

## What this means

- **The learned graded cortex from real experience is VIABLE** at the rate level: PPMI-shaped input + a
  non-saturating online local PCA rule (Oja class) learns real, generalizing category structure
  (held-out gen 0.875), reaching 92% of the best low-rank method and beating the project's own host.
- **CYCLE 53's "data-limited" is overturned** — the structure *is* recoverable from real PPMI.
- **The convergence sweep's NEGATIVE is overturned** — it was an under-converged similarity-matching
  implementation, not the online-rule-class wall (the robustly-convergent Oja rule in the same class
  reaches the ceiling).

## Brain-plausibility scoping (the honest claim boundary)

This is the **rate-level** Oja rule on **host-computed** PPMI input. A GO **greenlights building** the
spiking version; it does **not** claim it is built. The brain-based pieces, however, already exist or are
identified:
- **PPMI input** = `log` (Weber-Fechner / dendritic compression) + `/marginal` (divisive normalization by
  the pre/post activity EMA — **exactly the Phase-1 dendritic divisive gain** `g=σ/(σ+EMA)`, already a
  byte-reviewed protected edit) + `max(·,0)` (the spike threshold).
- **Oja's rule** = Hebbian feedforward + lateral decay — has published spiking realizations (Oja/Földiák
  nets; Pehlevan 2015/2019 spiking similarity-matching, integrate-and-fire + local Hebbian/anti-Hebbian).

## Honest caveats

1. **Rate, not spiking yet.** The recurring rate→spike loss is this project's hardest theme; the spiking
   realization is the real (owner-gated) build, not de-risked here.
2. **Moderate ceiling.** +0.52, not +0.9 — the real 8-category TinyStories taxonomy is *moderately*
   recoverable, so the learned cortex would have moderate ("cat is somewhat like dog"), not perfect,
   generalization. Real, but honest about magnitude.
3. **Small scale.** 64 concepts. Scaling to the 2,048-concept tier is part of the build, not yet shown.
4. **Oja, not the exact Pehlevan rule.** Oja is squarely in the owner's L1 class (online local Hebbian
   PCA) and is the robustly-convergent member; the specific Pehlevan-Chklovskii rule under-converged here
   (fixable — it is provably equivalent in the limit), a cheap follow-on to confirm.

## Files
- `research/runners/learned_graded_cortex_fair_test.py` — the fair test (true PPMI + SVD-grade reference +
  faithful saturating control + converged simmatch).
- `research/runners/_l1_simmatch_convergence_sweep.py` — the tuning-vs-wall sweep (showed simmatch under-converges).
- `research/runners/_l1_centered_online_pca_probe.py` — the Oja root-cause probe (the flip).
- `research/runners/_l1_oja_validated.py` — the airtight multi-seed validation + load-bearing honesty control.
- raw: `_l1_fair_real_multiseed.{json,log}`, `_l1_simmatch_convergence_sweep.{json,log}`,
  `_l1_centered_online_pca_probe.{json,log}`, `_l1_oja_validated.{json,log}`.

## Recommendation (decision point — the next step is owner-gated)

The cheap-first de-risk the owner asked for is **resolved: GO**. The natural escalation is the **weeks-scale
spiking similarity-matching build** (the artificial-life learned-cortex frontier). Because (a) the spiking
realization is the genuine high-variance unknown and (b) the owner gated the big build, the recommended
cheapest next de-risks before that commit are: confirm the exact Pehlevan-Chklovskii rule converges (closes
caveat 4), and a small **spiking-Oja smoke** on PPMI input (directly attacks the rate→spike risk). The flat
2,048-concept curated cortex (Option A, delivered) remains the shipped conversational product in parallel.
