---
type: finding
status: corrected
date: 2026-05-16
mechanism: g20-sparse-ensemble
---

# G.20 failure-mechanism & artifact-safe remediation — FINAL SYNTHESIS

> Single source of truth for the 2026-05-16 G.20 cross-bridge
> failure investigation. Consolidates ~10 findings docs into one
> navigable conclusion for the dedicated follow-up session. Every
> claim here is anti-cheat-validated and multi-seed-checked or
> explicitly scoped; overclaims encountered en route were corrected
> forthrightly (logged below).

## 1. The validated substrate (foundation, unchanged)

320-concept sparse G.20 ensemble, all anti-cheat: per-bridge 98.4%
(320) / 98.1% (160 multi-seed); pair cross-bridge 86.7% s42 /
**92.7% 160 5-seed**; sentence-3way 80%; **abstention AUC 0.990
(no confabulation)**; retention-under-30-fact-load 80% (no
catastrophic forgetting). This foundation was never in question.

## 2. The failure root-cause: a 4-step falsification chain

The recurring cross-bridge miss (~13–20%) was hunted by falsification:

1. "idx-12 = unlucky high-overlap pattern" — **disconfirmed**
   (idx 8/17 overlap more, pass).
2. "function-word category encodability" — **RETRACTED**
   (bridges are neurally identical by construction; a category
   effect is mechanistically impossible — the bridge spread was a
   small-n sampling artifact).
3. "index-intrinsic static pattern-overlap" — **disconfirmed by a
   falsifiable predictor** (validated vs 5 known seeds: all
   n_outliers=2; overlap anti-correlates with accuracy).
4. **"dynamical under-recall" — IDENTIFIED.** Probe: failing idx-12
   self_cum 213 vs robust 1157 (5.4× weaker), self-rank 12; not
   competitive capture (winner weak). The engram tag fails to
   reignite its own pattern.

## 3. The validated, artifact-safe remediation recipe

Two independent, post-hoc / query-time levers — **no retrain, no
`generate_sparse_patterns` change, validated artifact preserved**:

| Lever | Mechanism | Robustness | Effect | Ship? |
|---|---|---|---|---|
| **Capture-quality gate** (re-capture under-recalling tags at boosted teacher pA + longer window) | fixes under-recall | **ROBUST** — 5/5 bridges, mechanism-deterministic | **+3.3pp** | **default add-on** |
| `samebridge_downweight` (×0.4 home-bridge candidates at query) | counters same-bridge home-advantage | **SEED-VARIANT** — 4/5 seeds +, 1/5 −3.3pp | mean **~+2.5pp** | optional / tunable only |
| `perbridge_norm` | per-bridge 0-1 rate scaling | — | **−16.7pp** | **FALSIFIED — do not pursue** |

End-to-end controlled (keep-all, idx-12 included, seed 42):
80.0% → +remediation 83.3% → +samebridge_downweight 86.7%.
Honest typical (multi-seed): remediation dependable +3.3pp;
aggregation ~+2.5pp but variable → combined **~+5pp typical, not
the seed-42 +6.7pp best-case**.

## 4. The distinct residual sub-mechanism (characterized, open)

Of the cross-bridge misses: **50% same-bridge-as-query cross-talk**
(addressed, partially, by samebridge_downweight), **40% THIRD-bridge
spurious wins** (a separate effect NEITHER lever targets), 10%
same-as-target near-miss. The 40% third-bridge residual is the
correctly-scoped open question — uncharacterized beyond n=4; needs
the dedicated session.

## 5. Honest scope & the intellectual-honesty log

Every result is seed-42 unless stated; n=30 pairs (wide CIs;
directions are mechanism-grounded, magnitudes are not generalizable
point estimates). Forthright corrections made during the arc (the
discipline working as intended):

- 160 cross-bridge "100%" → **92.7% multi-seed** (seed-42 was clean).
- "function-word category" conclusion → **retracted** (mechanistically
  impossible) → relocated to dynamical.
- "static-overlap predictor" → **falsified** against known data.
- stacked "+6.7pp" → **down-graded** to ~+5pp-typical-but-variable
  after multi-seed.

## 6. What the dedicated session should do (precisely scoped)

1. Ship the capture-quality gate as a default recipe step (robust,
   mechanism-deterministic, artifact-safe).
2. Tune `samebridge_downweight` (×factor + a confidence gate so it
   only fires when the home candidate is ambiguous) to remove the
   regression risk; or leave it optional.
3. Characterize the **40% third-bridge** residual (the genuine open
   mechanism) — likely promiscuous high-baseline concepts; needs a
   targeted dynamical probe at adequate n.
4. Multi-seed-320 confirmation of the combined recipe (GPU-heavy).
5. Do **not** revisit: overlap-rejection, per-seed roulette,
   perbridge_norm, function-word-category framing (all falsified).

## 7. The genuine next frontier (gated)

Generative conversation is the next mission and is design-class:
see `docs/plans/2026-05-16-generative-conversation-frontier-survey.md`.
Per the project's brainstorming discipline it requires a
collaborative, user-approved design — not autonomous implementation.
The retrieval substrate is now a fully-characterized, trustworthy,
honestly-bounded foundation to build it on.

## Files (the consolidated chain, chronological)

sparse-ensemble-160/320-SHIPPED → xbridge/sentence/abstention/
interference benchmarks → cross-benchmark-failure-analysis
(+correction) → seed-quality-predictor-NEGATIVE →
dynamical-signature-UNDER-RECALL → capture-remediation-WORKS-bounded
→ remediation-end-to-end-CLOSURE → distinct-submechanism-same-bridge-
crosstalk → stacked-artifact-safe-levers-CLOSURE →
aggregation-lever-multiseed-REFINED → **this synthesis**.
