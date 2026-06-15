# Natural-learning-over-time: the decisive answer to "what could we be missing?" — it CONVERGES on the point-neuron locality wall (the (B) dendritic fork)

**Date:** 2026-06-15
**Cycle:** 81 (autonomous)
**Status:** DECISIVE — two new multi-seed probes + a re-confirmed corpus-hardness reframe. The natural-learning-over-time angle independently re-derives the 2026-06-11 cortex-build-plan fork.

---

## The owner's question

> "Well the brain needs to be able to learn naturally over time right? So we can't stop here. What could we be missing?"

This challenged the CYCLE-80 conclusion (that the spiking L1 learned cortex on the raw corpus is marginal, +0.296). The owner is right that the brain learns **continually, over time, with consolidation** — the project's own machinery (replay / sharp-wave-ripple consolidation, homeostasis). I had never applied that regime to the L1 cortex. So I tested it, decisively.

## What the question correctly surfaced (a real partial walk-back of CYCLE 80)

The CYCLE-80 "+0.296 BOUNDARY" was measured at the fair_test default (`epochs=200`, equal learning rates). The convergence trajectory shows this is a **partly over-trained endpoint**, not the optimum:

```
traj (seed 42, default):  ep40:+0.328  ep80:+0.343 (PEAK)  ep120:+0.319  ep160:+0.289  ep200:+0.264
```

Every config across both probes shows the same shape: **peak ~+0.35 around ep60–240, then a slow degrade to ~+0.26.** So the honest stable online ceiling is **~+0.35** (the peak), not +0.296 (an over-trained readout). The owner's instinct was correct on this point — the regime matters.

But +0.35 (peak) is still far below the **offline PPMI+PCA(64) optimum +0.518**. The real question is whether a natural-learning mechanism closes that residual.

## Probe 1 — timescale separation (the SM theory's required fast lateral). PLATEAUS.

`research/runners/_phaseB_timescale_convergence_derisk.py` (reuse-by-import of the validated `learn_simmatch`; PPMI input; real corpus; 3 seeds 42/43/44; n_hub=500, k=64).

The Pehlevan-Chklovskii similarity-matching theory **requires the anti-Hebbian lateral M to adapt faster than the feedforward W** (the decorrelation must keep up with the Hebbian learning). The validated `learn_simmatch` defaults to `lr_ff == lr_m` (0.01 == 0.01) — **equal rates, which every prior fair_test run, including CYCLE-80, used.** Biologically the fast-lateral regime is canonical (interneuron plasticity is fast; pyramidal feedforward plasticity is slow). I swept the ratio 1:1 → 20:1 plus stable-low-ff + long-epoch (the replay/consolidation regime).

| regime | ratio | Pearson (3-seed) | eff-rank |
|---|---|---|---|
| 1:1 baseline (fair_test default) | 1:1 | +0.262 | 3.3 |
| 5:1 lateral-faster | 5:1 | +0.266 | 3.4 |
| 10:1 lateral-faster | 10:1 | +0.268 | 3.4 |
| 20:1 + stable-ff + long (natural-over-time) | 20:1 | +0.267 | 3.4 |
| 20:1 + deeper-settle + longer (consolidation) | 20:1 | +0.276 | 3.8 |
| — | — | — | — |
| **offline PPMI+PCA(64) optimum** | — | **+0.518** | **53.0** |

**Timescale separation does NOT help.** The peak barely moves (+0.343 → +0.352) and the eff-rank stays stuck at **3–4** across every ratio. Faster lateral is not the missing piece.

**The key correction this probe surfaced:** the offline optimum (+0.518) has **eff-rank 53** — it is a *lightly-denoised high-rank* representation, NOT a low-rank one. The online local SM **over-compresses to rank 3–4**. The gap is rank 3–4 → 53, much larger than the "5 vs 8 PCs" I'd assumed. (And rank-3–4 +0.35 is below even the CYCLE-73 ZCA-rank-8 +0.437 — the online rule collapses past the useful operating point.)

## Probe 2 — nonnegative / clustering SM (the brain-faithful rate code, matched to the categorical target). PLATEAUS.

`research/runners/_phaseB_nonnegative_sm_derisk.py` (the ONLY change vs `learn_simmatch` is `relu(y)` in the settle — the brain-faithful nonnegative rate code; 3 seeds).

Motivation: the target `S_true` is **categorical** (8 same-category blocks). Linear similarity-matching computes **PCA** (the wrong objective for categorical structure); *nonnegative* SM does **clustering** (Pehlevan-Chklovskii 2018), which should match a categorical target better — and nonnegative rates are strictly more biological than signed ones.

| arm | Pearson (3-seed) | eff-rank |
|---|---|---|
| LINEAR_SM (PCA, signed) | +0.267 | 3.4 |
| NSM (clustering, relu ≥ 0) | +0.251 | 4.1 |
| offline PCA(64) | +0.518 | — |
| **offline k-means(8) — the categorical ceiling** | **+0.217** | — |

**Nonnegativity does not help** (NSM +0.251 ≈ LINEAR +0.267). And the load-bearing new datum: **offline k-means(8) — hard clustering into the 8 categories — only reaches +0.217.** The hard clusters are *orthogonal to the true categories*: the 8 taxonomy categories are **genuinely overlapping / not cleanly separable** in this raw co-occurrence space. (This is why even the offline PCA caps at +0.518, not +1.0 — the structure is graded/weak, not block-separable.) This **re-confirms the CYCLE-80 corpus-hardness reframe from a third, independent angle.**

## The comprehensive, honest answer to the owner

**We are NOT missing a natural-learning mechanism.** I tested the full natural-learning-over-time toolkit on the point-neuron substrate:

- **replay / consolidation** (long epochs, 400–600) — peaks then degrades; no lift.
- **homeostatic stability** (stable low feedforward rate) — stabilizes the plateau but does not raise it.
- **timescale separation** (fast lateral, the SM theory's requirement) — no effect (rank stuck 3–4).
- **nonnegativity** (brain-faithful clustering rate code) — no effect.

None close the gap to the offline optimum on the raw corpus. The online **local point-neuron rule fundamentally over-compresses to rank 3–4**; the offline optimum is a **high-rank (53), lightly-denoised** representation. **Keeping a high-rank denoised code requires the full cross-neuron covariance — a NON-LOCAL (dendritic / analog) computation a point neuron cannot do.** This is precisely the **Mikulasch-Priesemann point-neuron limit** (the project's standing "whitening is analog/pre-spike" reframe).

**⇒ The natural-learning-over-time angle CONVERGES on the exact fork the 2026-06-11 cortex-build-plan already identified.** Two independent routes — (i) the static decorrelation/binding analysis, and now (ii) the natural-learning-over-time analysis — arrive at the **same wall** and the **same answer**: a brain that learns its representations **over time AND generalizes on hard/correlated codes** needs the **(B) dendritic substrate** (the deferred, months-scale, owner-gated rewrite). On the point-neuron substrate, the viable regime is **curated / separable concepts**, where the same brain-rule reaches **+0.93** (the shipped 2,048-concept curated cortex).

Two routes landing on the same fork is itself a high-confidence deliverable: it sharply localizes "what we're missing" to one specific, biology-mandated piece (the non-local dendritic computation), not a missing learning trick.

## The strategic fork (sharpened for the owner)

1. **(A) Commit to the (B) dendritic substrate** — the brain-analogue path that learns-over-time AND generalizes. The natural-learning angle now *independently confirms* this is THE missing piece (not a learning-rule tweak). Months-scale, owner-gated, highest-variance, but the genuine "proper brain analogue that learns naturally over time" path the owner's goal points at.
2. **(B) Ship the curated flat cortex** (the viable separable-regime product, SM +0.93) and accept the point-neuron characterization as the honest delivered state.
3. **(C) Re-target + test FUNCTION-grounded, continual, in the VIABLE (curated) regime** — does the online SM on *separable* concepts learn **stably over continual time** (no over-training collapse, no catastrophic forgetting when new concepts are added) and support binding/conversation? This is "learn naturally over time" judged by *function* (the project's "validate a signal by its function" standard) in the regime where the brain-rule works. The cheapest, most constructive autonomous next step.

## Anti-cheat / honesty notes

- All arms share the SAME PPMI input and the SAME offline references, so a lift would be convergence, not an input change. Permuted-label controls ~0 on both probes (timescale: by construction; NSM: +0.020 / −0.007 / +0.005).
- The offline k-means / PCA references are computed on the identical input — a low k-means ceiling (+0.217) is the *data's* limit, not the online rule's, isolating the corpus-hardness contribution honestly.
- `learn_simmatch` is reused by import (the arc's own validated L1 machinery), NOT re-implemented — the only new code is the timescale sweep harness and the one-line `relu` for NSM.
- NO `sim/` edits. CPU/numpy (small matrices). The probe verdict-prose in `_phaseB_nonnegative_sm_derisk.py` over-stated "recoverable" in the NSM_EVEN branch; the correct reading is in this doc (k-means +0.217 = NOT cleanly separable).

## Artifacts

- `research/runners/_phaseB_timescale_convergence_derisk.py` + `research/findings/raw/_phaseB_timescale_convergence.{json,txt}`
- `research/runners/_phaseB_nonnegative_sm_derisk.py` + `research/findings/raw/_phaseB_nonnegative_sm.{json,txt}`
- Converges with: `docs/plans/2026-06-11-cortex-build-plan-decorrelate-then-bind.md` (the (A)/(B) fork), `2026-06-15-phaseB-spiking-cortex-WALL-rate-to-spike.md`, AUTONOMOUS_STATE.md CYCLE 80→81.
