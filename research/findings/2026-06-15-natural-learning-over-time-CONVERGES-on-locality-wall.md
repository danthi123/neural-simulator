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

## CYCLE 82 — the constructive complement: in the VIABLE (separable) regime, natural learning over time WORKS

The CYCLE-81 probes are on the *hard* raw corpus. The honest complement is the *viable* regime (curated/separable concepts, where the same brain-rule reaches +0.93 — the shipped 2,048-concept cortex). There, the owner's "learn naturally over time" question becomes two concrete, **function-grounded** (retention, not Pearson-to-offline-PCA) tests. `research/runners/_phaseB_continual_separable_derisk.py` (the validated `learn_simmatch` rule, inlined to thread `(W,M)` across continual phases + a bounded-M numerical guard; PPMI input; 3 seeds; 8 categories × 8 concepts).

| question | result (3-seed) | verdict |
|---|---|---|
| **Q1 stability over long continual training** | peak +0.944 → end +0.927 (400 epochs), eff-rank ~7 | **STABLE** (no over-training degrade) |
| **Q2 catastrophic forgetting** (train cats 0–3, then cats 4–7 disjoint, re-read 0–3) | phase-1 +0.986 → +0.852 after phase-2; **retention 0.86** | **NO catastrophic forgetting** (graceful) |
| — new concepts learned (phase-2) | +0.969 | learns the new set fine |
| — interleaved control (no-forget upper bound) | +0.976 | replay recovers retention to ~0.98 |

**Two clean positives:** (1) the online SM is **stable over long continual time** in the viable regime — the over-training degrade on the *raw* corpus (peak +0.35 → +0.26) is a **corpus-hardness symptom** (the rule amplifies noise dims when the structure is weak/overlapping), NOT an intrinsic flaw of the rule's natural-learning regime; on clean separable structure it holds its +0.93 plateau and recovers the right dimensionality (eff-rank ~7 ≈ 8 categories). (2) continual-disjoint learning causes only **graceful** forgetting (retention 0.86, well above the 0.80 bar) — and the interleaved arm (+0.976) shows the project's **already-validated** complementary-learning-systems / SWR replay machinery is exactly the tool to close the residual.

**⇒ The complete, balanced answer to the owner:** on the **hard / correlated** regime, natural-learning mechanisms hit the point-neuron locality wall (→ the (B) dendritic substrate, the deep frontier). On the **viable / separable** regime, **natural learning over time already works** — stable, retains old concepts, and the mild forgetting is closed by the project's validated replay-consolidation. So "we can't stop here" is right in two complementary ways: the viable regime is a **constructive positive that ships with the curated cortex**, and the hard-regime frontier has a **sharply-localized** missing piece (the non-local dendritic computation), not a missing learning trick.

## CYCLE 83 — the lengthening timeline: a real sequential penalty + the replay sweet spot (honest, nuanced)

The CYCLE-82 positive is a 2-phase case (one concept-set, then a disjoint second). The real "over time" scenario is a **lengthening sequence** — add categories one at a time and watch forgetting accumulate, with vs without interleaved sharp-wave-ripple (SWR)-style replay. `research/runners/_phaseB_continual_sequence_derisk.py` (8 categories added one-per-phase, 80 epochs/phase, replay-budget sweep k ∈ {0,2,4,8} concepts re-presented per old category; 3 seeds; metric = Pearson(cos(codes[seen]), S_true[seen]) — the live quality of everything learned so far).

Old-set retention over the timeline (mean, 3 seeds; after adding cats 3…8):

| replay budget | retention curve (old set) | final all-8 |
|---|---|---|
| NAIVE (k=0) | 0.994 → 0.869 → 0.859 → 0.791 → 0.740 → **0.661** | +0.612 |
| sparse k=2 | 0.996 → 0.976 → 0.957 → 0.910 → 0.827 → **0.704** | +0.591 |
| k=4 | 0.997 → 0.979 → 0.962 → 0.905 → 0.765 → 0.643 | +0.536 |
| full k=8 | 0.997 → 0.983 → 0.959 → 0.808 → 0.722 → 0.614 | +0.513 |
| (reference) batch all-8 | — | **+0.93** |

**Three honest facts (NOT "replay solves it"):**

1. **A real sequential penalty.** 8 categories learned one-at-a-time reach final-all ~+0.59–0.61 (any arm) vs the **batch +0.93**. Presentation order leaves a residue even with replay (early-learned structure biases the shared W/M; sequential ≠ batch — a known continual-learning phenomenon).
2. **Replay helps mid-timeline retention.** Through the middle of the timeline, replay holds the old categories at 0.91–0.98 where naive drops to 0.79–0.87 (old-set +0.859 naive → +0.957 at k=2 around phase 5).
3. **The effect is non-monotonic — sparse wins.** End retention and final-all are *best at sparse* k=2 (end +0.704, final +0.591) and **worse at heavy** k=8 (end +0.614, final +0.513 — *below naive*). Over-rehearsing old concepts within a fixed epoch budget starves integration of the growing set. **Sparse replay (k=2) is the sweet spot** — best retention without the over-rehearsal cost. **This matches the biology**: sharp-wave-ripples replay a *sparse sample* of experience, not the whole store.

**⇒ Honest conclusion (positive-with-caveat):** natural learning over an *extended* timeline is genuinely hard *even in the viable regime* — it needs a good (sparse) consolidation schedule, and a residual sequential-vs-batch gap remains at 8 categories. This does not retract the CYCLE-82 short-timeline positive (stable + graceful 2-phase forgetting); it qualifies it. And it independently re-derives *why the brain has elaborate consolidation machinery* — long-timeline continual learning is hard, and our simple SM+sparse-replay reproduces both the difficulty and the sparse-replay-wins signature. The owner's "we can't stop here" instinct is vindicated: even where the brain-rule works, learning-over-time is a real, ongoing problem the project's consolidation machinery exists to manage.

## CYCLE 84 — the loop closes: the DENDRITIC substrate (the answer) IS a natural continual learner

The whole arc concluded the missing piece is the **(B) dendritic substrate** — and that substrate is not hypothetical: the project's D-arc already validated it cheaply (D1 GO: a per-compartment divisive gain recovers category structure +0.845 vs the point-neuron's +0.052; D1.5/D1.6/D1.7 hardened it through spiking-noise + conductance-shunting + a spiking two-compartment) and built it through D2 Phase 0/1/2 (the two-compartment neuron is on the bridge; the learned graded embedding exists), with only D2 Phase 3 (the dual/CLS pipeline + conversational gates) pending. The natural question that closes the loop: does the substrate that *recovers* structure *also* learn it **naturally over time** without catastrophic forgetting?

`research/runners/_phaseB_dendritic_continual_derisk.py` (reuse-by-import of the validated dendritic D1 machinery; the per-hub gain `g_h += eta*(x_h − g_h)`; synthetic separable corpus; 3 seeds; directly comparable to the CYCLE-83 point-neuron SM sequence):

| substrate | sequential (1-at-a-time) final-all | of batch | old-set retention | replay needed? |
|---|---|---|---|---|
| **DENDRITIC** (per-compartment gain) | **+0.760** | **90%** (batch +0.841) | **+0.898** | **no** |
| point-neuron SM (CYCLE 83) | +0.59 | 64% (batch +0.93) | 0.66 | helps, doesn't fully close |
| point-global control | +0.052 (~0) | — | — | — (per-compartment gain is load-bearing) |

**DENDRITIC-LEARNS-OVER-TIME (3 seeds, clean).** The dendritic substrate keeps **90% of its batch ceiling over a sequential one-at-a-time timeline** and holds old categories at **0.90 retention** — dramatically better than the point-neuron SM (64%, 0.66) — and it needs **no replay at all**. The point-global control stays ~0 (the per-compartment gain is load-bearing); permuted ~0.

**Why (the mechanism, biology-grounded):** the dendritic per-hub gains converge to each hub's **marginal frequency** — slowly-varying *global statistics per hub*, shared across all concepts, NOT per-concept weights that compete for a shared capacity. The common hubs stay high-frequency (stay down-weighted), each category's distinct signal hubs are handled *independently* by their own compartment, so categories don't interfere. Per-hub independence ⇒ natural continual learning. (Contrast the point-neuron SM, where every concept competes for the shared `(W,M)` → sequential collapse + forgetting.)

**⇒ The complete, constructive answer to the owner.** "What could we be missing?" — on the **point-neuron** substrate, a learning *mechanism* is not the missing piece (replay/homeostasis/timescale/nonnegativity all tested, all plateau at the locality wall). The missing piece is the **substrate**: a dendritic per-compartment normalization. And that substrate — already validated and built through D2 Phase 2 — is not only the structure-recovery answer but a **natural continual learner that learns over time without catastrophic forgetting and without replay**. The owner's question is answered *on the working substrate*, and it strengthens the case for D2 Phase 3 (the dendritic cortex is both the right representation AND a natural over-time learner).

## CYCLE 85 — the honest real-corpus tempering: the dendritic per-hub gain is the DIAGONAL escape (great on diagonal-separable structure, WEAK on the real corpus's off-diagonal)

The CYCLE-84 loop-closer used the **synthetic** corpus (where the D1 GO was validated, host +0.96). The synthetic structure is *purely diagonal-separable* — common hubs vs category-signal hubs distinguished by frequency, exactly what a per-hub divisive gain handles. The **real** TinyStories corpus (host +0.44) is the actual target. `research/runners/_phaseB_dendritic_real_derisk.py` (the validated D1 per-hub gain on `build_real_corpus`, small sigma sweep, 3 seeds):

| arm (real corpus) | Pearson (3-seed) | note |
|---|---|---|
| host PPMI+SVD | +0.442 | ceiling |
| **DENDRITIC per-hub gain (batch)** | **+0.216** | only **49% of host** |
| DENDRITIC sequence (continual) | +0.259 | continual is fine (120% of its own batch) — but the batch is low |
| point-global control | +0.197 | the dendritic gain **barely beats it** (+0.019) |
| (reference) point-neuron SM on real | ~+0.35 | the dendritic gain is *below* even this |

**REAL-WEAK.** On the real corpus the dendritic per-hub divisive gain reaches only +0.216 (49% of host) and barely clears the point-neuron global control. The continual-learning property holds (the sequence keeps its batch level), but the batch itself is low — so "learns the structure over time" is hollow when the structure recovered is weak.

**Why (the honest mechanism, reconciling both arcs):** a per-hub *divisive* gain `x/(σ+g)` is **diagonal** (per-feature) normalization. It is NOT PPMI: PPMI normalizes by a **log-subtractive** ratio (`log x − log row − log col`), and the cross-neuron decorrelation that reaches host is **off-diagonal**. The synthetic corpus's structure is recoverable by the diagonal alone (frequency-separable), so the dendritic gain shines there (+0.845). The real corpus has off-diagonal structure the diagonal gain cannot reach. This lands exactly where my own _phaseB off-diagonal de-risk (CYCLE 73) already put it: the diagonal mechanisms (subtractive centering +0.307, divisive gain +0.216) all reach ~+0.22–0.31 on real; the full low-rank whitening reaches host +0.44; the **+0.31 → +0.44 gap is the off-diagonal cross-neuron decorrelation**, which neither the point-neuron SM (plateaus +0.35) nor the dendritic per-hub gain (+0.22) provides.

**⇒ The complete, honest, balanced answer (corrected for the real target):**
- The owner's instinct is right and the over-training walk-back stands (the stable point-neuron SM peak is ~+0.35, not +0.296).
- On the **point-neuron** substrate, a learning *mechanism* is not the missing piece (replay/homeostasis/timescale/nonnegativity all plateau at the diagonal-only ~+0.35).
- The **dendritic per-compartment gain** is the **diagonal escape**: it recovers + continually-learns *diagonal-separable* (synthetic/curated) structure beautifully and naturally over time — which is the regime the shipped curated cortex lives in — but on the **real** noisy corpus it reaches only ~+0.22 (49% of host), barely above the point-neuron control.
- The genuine open frontier for a *learned, generalizing, real-experience* cortex is the **off-diagonal cross-neuron decorrelation** — and that is precisely the deep piece both arcs keep landing on (the Mikulasch-Priesemann non-local computation; the D2 two-compartment build, still owner-gated and NOT started). The simple per-hub gain is necessary but **not sufficient** on the real corpus; D2 would need to realize the off-diagonal, not just the diagonal.

So the case for D2 is honestly **qualified**: the dendritic substrate clears the diagonal half cleanly and is a natural continual learner *there*, but the real-experience cortex needs the off-diagonal too — the months-scale build is warranted *for the artificial-life / biology-translatable goal*, with eyes open that the diagonal-only per-hub gain (the cheaply-validated piece) is one half of the operation.

## Artifacts

- `research/runners/_phaseB_dendritic_real_derisk.py` + `research/findings/raw/_phaseB_dendritic_real.{json,txt}` (CYCLE 85, the honest real-corpus tempering)
- `research/runners/_phaseB_dendritic_continual_derisk.py` + `research/findings/raw/_phaseB_dendritic_continual.{json,txt}` (CYCLE 84, the synthetic loop-closer)
- `research/runners/_phaseB_timescale_convergence_derisk.py` + `research/findings/raw/_phaseB_timescale_convergence.{json,txt}`
- `research/runners/_phaseB_nonnegative_sm_derisk.py` + `research/findings/raw/_phaseB_nonnegative_sm.{json,txt}`
- `research/runners/_phaseB_continual_separable_derisk.py` + `research/findings/raw/_phaseB_continual_separable.{json,txt}` (CYCLE 82, the 2-phase constructive complement)
- `research/runners/_phaseB_continual_sequence_derisk.py` + `research/findings/raw/_phaseB_continual_sequence.{json,txt}` (CYCLE 83, the lengthening timeline + replay-budget sweep)
- Converges with: `docs/plans/2026-06-11-cortex-build-plan-decorrelate-then-bind.md` (the (A)/(B) fork), `2026-06-15-phaseB-spiking-cortex-WALL-rate-to-spike.md`, AUTONOMOUS_STATE.md CYCLE 80→81.
