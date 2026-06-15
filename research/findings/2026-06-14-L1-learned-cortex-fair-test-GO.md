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
4. ~~**Oja, not the exact Pehlevan rule.**~~ **CLOSED (same cycle).** The exact Pehlevan-Chklovskii
   similarity-matching rule (the owner's *named* L1 rule), with the same fixes (centered input + a
   properly-settled fixed point + faster lateral), reaches **+0.515** multi-seed = 98% of the offline
   optimum, *above* Oja, de-saturated (offdiag −0.012), permuted clean (−0.007). The +0.29 was definitively
   under-convergence, not a rule limit. `_l1_simmatch_converges_check.py`.

## Follow-on de-risks (same cycle, both GO)

**(a) Spiking-input axis — GO and graceful** (`_l1_spiking_oja_smoke.py`). The single highest-variance
unknown before a spiking build is this project's recurring rate→spike loss. Delivering the PPMI input as
**Poisson spike counts** (re-sampled per epoch for realistic trial-to-trial noise, and at test time),
sweeping the spike budget:

| spike budget | ~spikes/hub/concept | spiking-Oja Pearson | gen | controls (sat / perm) |
|---|---|---|---|---|
| gain 3 | ~2 | +0.377 (78% of rate) | 0.812 | +0.000 / −0.009 |
| gain 10 | ~6 | +0.430 (89%) | 0.859 | +0.000 / +0.011 |
| gain 100 | ~63 | +0.475 (99%) | 0.854 | +0.000 / +0.000 |

At **biologically-reasonable spike counts (~2–6 spikes/hub/concept)** the structure largely survives
(78–89% of the rate ceiling), controls clean at every budget. The rate→spike loss on the **input axis** —
historically this project's wall (threshold-silencing) — is **graceful given log-PPMI input**, not fatal.
(The *learning*-spiking gap — spike-timing-driven Hebbian — is kept exact here and is the build-time follow-on.)

**(b) Learning-spiking + non-negativity axis — GO** (`_l1_nonneg_simmatch_check.py`). Real spiking neurons
emit **non-negative** spike counts; the signed Oja/Pehlevan decorrelation relies on negative output
correlations a rectified rate lacks. The brain-correct **non-negative** similarity-matching (rectified
output + lateral inhibition — Pehlevan's NSM):

| variant | Pearson | gen | perm |
|---|---|---|---|
| SIGNED (reference) | +0.515 | 0.776 | −0.007 |
| NONNEG rate | +0.461 (90%) | 0.724 | −0.002 |
| NONNEG + spike output (Poisson on the Hebbian terms) | +0.466 (90%) | 0.719 | −0.001 |

Non-negativity costs only ~10%, and the full spike-driven output barely changes it (+0.461→+0.466). The
learning-spiking axis is de-risked: rectified firing does **not** break the learner.

**(c) Scale-capacity axis — GO** (`_l1_scale_capacity_check.py`). Online local rules can degrade as the
number of categories grows. At realistic difficulty (synthetic tuned to host ~+0.40), the non-negative
online rule's extraction fraction (learner / offline PCA) **holds across 64→256 concepts**:

| concepts | learner | offline | host | fraction | gen |
|---|---|---|---|---|---|
| 64 (8 cat) | +0.421 | +0.439 | +0.404 | 96% | 0.651 |
| 128 (16 cat) | +0.518 | +0.516 | +0.404 | 100% | 0.818 |
| 256 (32 cat) | +0.398 | +0.454 | +0.355 | 88% | 0.703 |

The fraction stays ≥88% (min) with no collapse trend, and the learner matches/beats the host's own
`ppmi_svd_sim` at every scale. Capacity headroom is present in this range. (Real-data-noise at the full
2048 scale remains a build concern, not cheaply testable.)

**(d) Common-mode removal is the enabling operation.** The fix that converged both Oja and the exact
Pehlevan rule was **centering** (subtract the column mean = remove the common mode). This is the *same*
whitening/common-mode theme the project has hit 5+ times — and it is brain-plausible as a slow
**subtractive-inhibition EMA** (feedforward inhibition). The local online rule *can* extract the structure,
**given** common-mode removal; without it the dominant common mode saturates the codes.

## Files
- `research/runners/learned_graded_cortex_fair_test.py` — the fair test (true PPMI + SVD-grade reference +
  faithful saturating control + converged simmatch).
- `research/runners/_l1_simmatch_convergence_sweep.py` — the tuning-vs-wall sweep (showed simmatch under-converges).
- `research/runners/_l1_centered_online_pca_probe.py` — the Oja root-cause probe (the flip).
- `research/runners/_l1_oja_validated.py` — the airtight multi-seed validation + load-bearing honesty control.
- raw: `_l1_fair_real_multiseed.{json,log}`, `_l1_simmatch_convergence_sweep.{json,log}`,
  `_l1_centered_online_pca_probe.{json,log}`, `_l1_oja_validated.{json,log}`.

## Phase-A capstone: end-to-end spiking composition — GO, and it SIMPLIFIES the build

Do the four axes **compose**? An end-to-end spiking net — spiking input **+** spiking output **+**
spike-driven learning **+** recurrent settle, every signal a spike (`_l1_phaseA_end_to_end_spiking.py`,
numpy, no sim/ edits). Multi-seed at a realistic spike budget (in_gain 30):

| recipe | Pearson | gen | perm |
|---|---|---|---|
| full SM (with anti-Hebbian lateral) | +0.386 | 0.693 | +0.001 |
| **subtractive-inhibition centering + bounded Hebbian (no lateral)** | **+0.545** (106% of rate ceiling) | 0.708 | −0.010 |
| random projection (no learning) | +0.121 | 0.375 | +0.013 |

The structure **is recovered end-to-end in full spikes** (+0.545 ≈ the offline optimum), learning is
load-bearing (vs random +0.121), permuted clean. **Build-simplifying finding:** the anti-Hebbian recurrent
lateral — the *highest-risk protected edit* in the build proposal — **hurts under end-to-end spike noise**
(+0.386 < +0.545). Common-mode removal done **explicitly** via subtractive inhibition (feedforward
inhibition) is more spike-robust than doing it **implicitly** via the recurrent lateral. ⇒ the spiking build
can **drop the recurrent lateral** and use *subtractive-inhibition centering + a homeostatically-bounded
Hebbian feedforward* — simpler and lower-risk. (Honest process note: the first end-to-end run showed +0.18
with the learner *below* its saturating control — a single-sample-readout artifact; root-causing it to a
faithful windowed readout + correct normalization + the random-projection control produced the result above.)

## Comprehensive de-risk summary — all five axes GO

| axis | result | runner |
|---|---|---|
| **Rule** (does a brain-based online learner reach the ceiling?) | GO — Oja +0.481, exact Pehlevan +0.515 = ~98% of offline +0.523, beats host +0.323, learning load-bearing | `learned_graded_cortex_fair_test`, `_l1_oja_validated`, `_l1_simmatch_converges_check` |
| **Input-spiking** (does the structure survive Poisson-spike input?) | GO — 78–89% at ~2–6 spikes/hub/concept | `_l1_spiking_oja_smoke` |
| **Learning-non-negativity + spiking** (rectified firing + spike-driven Hebbian?) | GO — 90% of signed | `_l1_nonneg_simmatch_check` |
| **Scale-capacity** (64→256 concepts?) | GO — fraction holds 88–100% | `_l1_scale_capacity_check` |
| **End-to-end spiking composition** (do the axes compose?) | GO — +0.545 (106% of rate) in full spikes; learning load-bearing; **the recurrent lateral is unnecessary** | `_l1_phaseA_end_to_end_spiking` |

Enabling operation across all: **common-mode removal (centering = subtractive-inhibition EMA)** — the
project's recurring whitening theme, here the single fix that converged the local online rule, and (Phase-A)
the spike-robust way to do it explicitly so the recurrent lateral can be dropped.

## Recommendation (decision point — the next step is owner-gated)

The cheap-first de-risk the owner asked for is **resolved: GO**, and comprehensively so — every numpy/smoke
axis is positive. The only remaining unknowns are **the bridge assembly** (a spiking similarity-matching
network on the `SimulationBridge` — real LIF/rate neurons, lateral-inhibition synapses, PPMI-shaped spiking
input via dendritic-log + the Phase-1 divisive gain + threshold) and **real-data-noise at the full 2,048
scale**. These *are* the **weeks-scale owner-gated build** — not cheaply de-riskable.

**Recommendation:** the evidence supports committing to the spiking similarity-matching learned-cortex
build as the artificial-life / biology-translatable frontier; a concrete build proposal accompanies this
finding (`docs/plans/2026-06-14-spiking-similarity-matching-cortex-build-proposal.md`). The flat
2,048-concept curated cortex (Option A, delivered) remains the shipped conversational product in parallel.
**The build itself is owner-gated** — this finding + the proposal are for the owner's steer; the build is
not started autonomously.
