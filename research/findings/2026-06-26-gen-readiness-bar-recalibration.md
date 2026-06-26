# Generalization readiness-bar re-calibration (Pearson-based)

**Date:** 2026-06-26
**Status:** FRAMEWORK (result placeholder pending the n_per24 @ 150K decisive run, blb5vcsjd)
**Owner-facing purpose:** define the *right* "does the curriculum-trained brain generalize?" bar for the first-chat-ready gate, replacing the miscalibrated `generalization >= 0.80`.

## The problem with the old bar (`gen-coherent >= 0.80`)

The runner's frozen bar is `generalization >= 0.80`, where `generalization` =
mean cosine of a held-out concept's code to its a-priori category-mates
(the `measure_generalization` "coherent" path). Two reasons that number is
the wrong yardstick:

1. **It is scale-dependent.** Mean within-category cosine falls as the number
   of categories rises (more categories -> finer partition -> lower absolute
   within-cat cosine even when structure is perfect). The `0.80` was set
   against an ~8-category toy; at the real ~40-51 coherent categories the same
   *quality* of structure scores far lower in absolute terms. Comparing a
   40-cat run to an 8-cat bar is apples-to-oranges.

2. **It conflates absolute cosine with structure quality.** What "generalizes"
   actually means is: *codes that are more similar are in more-related
   categories* — a monotone relationship, not an absolute magnitude.

## The C0 finding this rests on

The C0 numpy substrate-vs-scale harness (`_curriculum_gen_C0_substrate_vs_scale.py`)
split the gen miss into two roughly-equal halves:
- **(a) scale / metric granularity** — the 8-cat -> 40-cat partition change
  (the absolute-cosine deflation above), and
- **(b) spiking read-out noise** — numpy-exact `corr(M,C)=1.0` gave gen 0.45
  (Pearson **+0.215**), while the spiking population read-out at corr 0.756
  dropped it; generalization needs the *fine off-diagonal* similarity that is
  the first casualty of read-out noise (recall/moat need only
  distinguishability, so they survive).

The chance-independent tell that survives both (a) and (b) is the **Pearson
correlation `r(measured_cosine, S_true)`** — how well the measured pairwise
code similarity tracks the true a-priori category similarity. It is invariant
to the absolute-cosine deflation (a) and directly measures the structure (b)
preserves or loses.

## Proposed re-calibrated bar

Use **Pearson `r(cos, S_true)`** as the primary generalization metric, with the
two anchored reference points:
- **Ceiling (numpy-exact ideal, full training):** Pearson **+0.215** — the
  best the *representation* can do; the read-out cannot exceed this.
- **Floor (no structure):** Pearson **~0.0** (the category-derangement control
  must collapse to ~0).

Candidate readiness bar (to be confirmed against the 150K result + a
sample-conversation quality check, NOT set in stone here):
- **Pearson `r >= ~0.12`** on the spiking read-out at full training
  (~>=55% of the numpy ceiling) AND derangement collapses — i.e. the spiking
  substrate retains a majority of the recoverable structure.
- The absolute `gen-coherent` cosine is *reported* (for continuity) but is no
  longer a pass/fail gate.

This bar is deliberately provisional: the final first-chat-ready gate should
also include a **sample-conversation quality check** (does the brain relate
queried concepts to plausibly-related ones?), because Pearson is necessary but
may not be sufficient for the *felt* richness of a first chat.

## Data points (filling in)

| Run | n_per | windows | corr(M,C) | gen-coherent | Pearson r | note |
|---|---|---|---|---|---|---|
| numpy ideal (C0) | exact | full | 1.000 | 0.45 | **+0.215** | representation ceiling |
| n_per-16 base | 16 | 150K | 0.885 | 0.125 | (TBD from log) | prior real-corpus base |
| 10K smoke | 24 | 10K | 0.902 | 0.056 | +0.017 | under-trained (15x); inconclusive |
| **n_per24 @ 150K** | 24 | 150K | (pending) | (pending) | **(pending)** | THE decisive fidelity test |

## The decision this run resolves

Does lifting read-out fidelity (corr 0.885 -> ~0.95 via the n_per-24 population
code) lift the **Pearson r** at full training — toward the +0.215 ceiling — or
does gen stay near the n_per-16 base (Pearson ~?) regardless?
- **If Pearson lifts materially** -> fidelity (more neurons/concept) is a real
  gen lever; tune it up + set the bar against the lifted value.
- **If Pearson stays flat** -> fidelity is a minor lever; the dominant gen
  levers are corpus richness (TinyStories' simple co-occurrence gives weak
  category structure) and scale, which is the foundational-curriculum
  direction. Conclude the fidelity micro-question and pivot.

(Result + conclusion to be appended when blb5vcsjd completes.)
