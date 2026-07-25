# Consolidation two-sided-read GENERALIZATION de-risk → NO-GO: the "fact-1 own/other 3.67" LEAD is a WINNER-SLOT metric artifact (permuted-core control REFUTES it); the write produces NO earned per-fact selectivity (2026-07-25)

**Supersedes the "⚠️ CORRECTION + LIVE LEAD" section of** `2026-07-25-consolidation-boundary-REATTRIBUTED-dense-CA1-code-not-the-write.md`.
That correction reported the unsaturated-graded-write + core-gated-recall reaching **own/other 3.67 for fact 1** (seed 42)
as "the FIRST non-flat write result … the two-sided read IS the realizable surpass." **This de-risk adds the permuted-core
control the lead never ran, and it REFUTES the lead:** the 3.67 is a **winner-slot metric artifact**, not per-fact
selectivity. The boundary is SHARPENED, not surpassed.

## The task
Decide whether the two-sided read (unsaturated graded BTSP write + core-gated recall) GENERALIZES across all 3 facts (and
seeds), or whether it is fact-1-luck. GO-in-principle = core-gated own/other ≥ 2.5 AND own-is-max on ≥2/3 facts, ≥3 seeds,
**AND the selectivity survives a permuted-core control** (weight the recall by a random fact's core → must collapse to ~1.0).

## Reproduction (seed 42, the exact lead config)
`--commit-top-k 15 --hippo-izh-type IZH2007_STRIATAL_MSN --hippo-izh-regions dg,ca3,ca1 --elig-tau 30 --elig-hard-thresh
0.4 --cycles 3 --btsp-wmax 2000 --btsp-lr 0.000003` on the shipped `_consol_decoupled_plateau_probe.py` reproduces the lead
**exactly**: CORE-GATED RECALL `[0.331, 3.687, 0.441]`, core_sizes `[1, 14, 22]`, dw=124 (unsaturated, ≪ w_max=2000).
Sparse ceiling for fact 2 = **14.84** (its core code is highly separable) yet its written weights read **0.44** — i.e. the
write is failing to localize onto fact 2's *separable* core. That mismatch was the entry point.

## THE DECISIVE DIAGNOSTIC — it is a WINNER-SLOT artifact (the permuted-core control the lead never ran)
New probe `research/runners/_consol_twosided_generalize_probe.py` (reuse-by-import, NO sim/ edit) reproduces the core-gated
recall and adds three controls the shipped probe lacked: **(1) a PERMUTED-CORE control** (read slot_i's recall with a
DIFFERENT fact's core), **(2) a RANDOM-CA1 control** (read slot_i with a random set of CA1 cells), **(3) the per-slot mean
`ca1→slot` weight**. Seed 42, baseline (interleaved) lead config:

| metric | fact 0 | fact 1 | fact 2 |
|---|---|---|---|
| CORE-GATED own/other (the lead's GO metric) | 0.44 | **3.63** | 0.45 |
| **PERMUTED-CORE control** (must collapse ~1.0) | 0.44 | **3.46** | 0.46 |
| **RANDOM-CA1 control** (must collapse ~1.0) | 0.49 | **3.20** | 0.47 |
| PER-SLOT mean `ca1→slot` weight | 24.47 | **81.70** | 24.00 |

**Slot 1 carries ~3.4× the weight of slots 0 and 2** (81.70 vs ~24). The "fact-1 own/other 3.63" is *exactly* that global
imbalance (81.70/24.2 = 3.37). Reading slot 1 with fact-1's core, a **different** fact's core (3.46), or **random** CA1
cells (3.20) all give ~3.4 — **the ratio is independent of which cells you read.**

**The refutation stands on the LEAD's OWN numbers, not my probe's substrate state:** the lead reported core-gated recall
`[0.33, 3.69, 0.44]` (facts 0/1/2). Core-gated recall for fact i = (fact-i core-weighted sum of `w[·→slot_i]`) /
(mean over j≠i of the same sum to `slot_j`). For fact 1 to read 3.69 while facts 0 and 2 read 0.33/0.44 on the *same*
weight matrix, `slot_1` must carry ~3.4× the `ca1→slot` weight of the others — i.e. the lead's own three numbers already
encode the winner-slot imbalance (facts 0 and 2 are low precisely because their denominator contains the heavy slot 1).
My probe's separate per-slot measurement [24.5, 81.7, 24.0] and the non-collapsing controls confirm it directly; the
arithmetic makes it inescapable regardless of the 210-step pre-write fire read (dw 128 vs the lead's 124; core-timing
changes only which cells are *read*, never the written weights the per-slot mean reports). The permuted/random controls do NOT
collapse. ⇒ the "own/other ≥ 2.5" is a **winner-slot metric artifact**: one attractor slot accumulates ~3.4× more total
`ca1→slot` weight than the others, and the core-gated ratio reports that imbalance for whichever fact maps to the heavy
slot. The tiny excess of the matched read (3.63) over random (3.20) is ~13% — swamped by the 3.4× artifact, i.e. no
meaningful genuine fact-specificity.

## Multi-seed (3 seeds) — the artifact is robust; the "winning" slot is SEED-dependent (a schedule/init artifact, not a fact property)

| seed | per-slot weights | winner slot | winner fact own/other (real · permuted · random) |
|---|---|---|---|
| 42 | [24.47, **81.70**, 24.00] | 1 | 3.63 · 3.46 · 3.20 |
| 43 | [**74.23**, 23.26, 22.90] | 0 | 3.18 · 3.42 · 3.33 |
| 44 | [22.54, **76.03**, 22.67] | 1 | 3.74 · 3.50 · 3.35 |

Every seed: **exactly one slot wins** (~74–82) vs two losers (~22–24), ratio ~3.3–3.4 ≈ **n_facts = 3**. The winning slot
changes with the seed (1, 0, 1). n_pass = **1/3 every seed** (only the winner "passes"), and even that pass fails the
permuted + random controls. **GO-in-principle is FALSE**: never ≥2 facts, and the one "passing" fact is an artifact.

## Fact-2 leakage diagnosis (the load-bearing question) — it is the OTHER SIDE of the winner-slot coin
Fact 2 reads 0.44 not because of a fact-2-specific write idiosyncrasy but because **fact 2 is not the winner slot at seed
42** (slot 1 is): its own/other = w[core2→slot2] / mean(w[core2→slot0], w[core2→slot1]), and the denominator contains the
heavy slot 1 → deflated to 0.44. **Decisive proof:** switching the write schedule to BLOCKED (all cycles of a fact before
the next) makes **slot 2 the winner** (per-slot [24.4, 24.1, **80.4**]) and now **fact 2 "passes" (3.70) while facts 0 and 1
fail (0.43, 0.44)** — the identity of the "passing" fact flips with the *schedule*, confirming it is not a fact property.

**Why the write can't localize (cross-fact core-firing, the leak driver):** the probe captures each fact's core-cell firing
under its own tag AND during the ACTUAL write windows.
- **Fire-under-tag (isolated):** each core fires ~2× more under its own tag — core_K diag ≈ 12.3 vs off-diag ≈ 6–7. The
  cores ARE fact-specific in isolation (this is the sparse ceiling ~5.6–7).
- **During the actual multi-fact write:** every core fires ~30 spikes in EVERY fact's window — nearly FLAT (e.g. core_1 =
  [32.5, 28.7, 32.8]; its OWN window is the *lowest*). The isolated 2:1 specificity is **destroyed** during the back-to-back
  write because CA1 never returns to baseline between facts (unlike `_fire_under_tag`, which runs 30 no-drive warmup steps
  before each tag). So the eligibility that drives the write is not fact-specific → the write can't localize per-fact, and
  which slot happens to accumulate most (the winner) is set by a seed/schedule eligibility-accumulation asymmetry.

Hypotheses tested: **(a) cores fire in other facts' windows during the write — CONFIRMED** (flat ~30 everywhere).
**(b) eligibility temporal bleed — reset-elig alone does NOT help** (own/other unchanged; resetting the trace at window
boundaries doesn't change the flat within-window firing that rebuilds it). **(c) apical clamp not exclusive via the
self-regen latch — REJECTED** (`--self-regen 0.0` still gives slot 1 winner [23.97, **79.74**, 23.49]; the winner is not
the plateau latch). **(d) fact-0's 1-cell core — a Hebbian after-write sharpening artifact** (the write-independent
before-write cores are robust ~17–24 for all facts; the own/other pattern is identical either way).

## Tuning for uniform selectivity — every reachable config either keeps the artifact or flattens to no selectivity
- **Removing the artifact removes ALL apparent selectivity.** Full isolation (`--blocked --settle-steps 30 --reset-elig`)
  equalizes the slots: per-slot weights **[5.88, 5.80, 5.80]**, CORE-GATED own/other **[1.05, 1.02, 1.01]**, all controls
  collapse to ~1.0, n_pass 0/3. So when inter-fact settling removes the winner, **no genuine per-fact selectivity remains** —
  the write does not localize.
- **Sharpening levers don't rescue it.** Settle alone flattens to ~1.0 (and, mysteriously, the settled write-window firing
  is still not diagonal-dominant — settling reduces firing but does not restore per-fact specificity). Higher
  `elig_hard_thresh` (0.6), shorter `elig_tau` (10), supralinear `elig_exp` (2) — none produce a control-collapsing ≥2-fact
  GO (consistent with the prior finding's ~1.1 marginal results and the dense-code overlap ceiling 1.45).

## VERDICT — NO-GO, and the boundary is SHARPENED, not surpassed
**The two-sided read does NOT generalize.** The lead's "fact-1 own/other 3.67 > 2.5 gate" was a **winner-slot metric
artifact** the original core-gated-recall metric could not detect because it never ran a permuted/random-core control. On
this substrate the multi-fact consolidation write produces **one globally-heavy attractor slot** (~n_facts× the weight of
the others, seed/schedule-determined), and no earned per-fact `ca1→slot` selectivity: the matched-core read is ~13% above a
random-core read of the same slot — noise, not a signal. Removing the artifact (settle/isolation) yields flat own/other
~1.0 with the controls collapsed. **Not "fact-1 works," but "one SLOT artifactually wins per run."**

This is fully consistent with the parent finding's own repeatedly-confirmed root cause — the point-neuron REPLAY-FLOODING /
CA1 code-density wall: during the multi-fact write CA1 does not return to a fact-specific baseline between facts, so the
write's eligibility is flat and cannot localize. The sparse fact-specific core EXISTS (fire-under-tag 2:1, sparse ceiling
~5.6–15) but is **not operative** for a graded point-neuron write, exactly as the parent finding concluded before the (now
refuted) lead. **Per THE LAW the capability stays OPEN**; the honest next method is unchanged from the parent finding's
deep verdict: a mechanism that reinstates each fact ISOLATED and reads/writes on the sustained-firing core only — the
dendritic per-branch spike-count-threshold read (the substantial D2 substrate arc), NOT a cheaper write/schedule tweak. The
cheap two-sided-read space is now exhausted **with the controls in place** (the piece the lead was missing).

## Anti-cheat / rigor notes
- **Selectivity must be EARNED:** the permuted-core AND random-CA1 controls are the load-bearing test; both fail to collapse
  for the winner fact (they equal the winner ratio) → the metric was measuring a global slot imbalance, not fact-specificity.
- **Not a degenerate-core artifact:** the winner fact's core is 12–24 cells (not 1–2); the write-independent before-write
  cores are robust ~17–27 for all facts, and the own/other pattern is identical to the after-write cores.
- **Seeded substrate verified:** 3 distinct `cp_neuron_firing_thresholds` hashes (ee7fcf106bea / 90ff32ec773a /
  7cb65bc009d2) — the seed-never-controlled-substrate trap is not present.
- **NO sim/ edit** (`git diff --ignore-cr-at-eol --stat -- sim/` empty); reuse-by-import + one new runner-only probe.

## Shipped infra
`research/runners/_consol_twosided_generalize_probe.py` — the two-sided generalization probe: per-fact core-gated recall +
**permuted-core + random-CA1 + per-slot-weight controls** + cross-fact core-firing (fire-under-tag AND during-write) +
`--blocked / --settle-steps / --reset-elig / --fixed-order` isolation knobs. Raw: `research/findings/raw/consol_opsweep_gpu/twosided_*.json`.

## Provenance
Seed 42/43/44 baseline + `--blocked`, `--reset-elig`, `--settle-steps 30`, `--self-regen 0.0`,
`--blocked --settle-steps 30 --reset-elig` (all seed 42), GPU (`SIM_BACKEND=cupy`). Reuse-by-import, NO sim/ edit.
