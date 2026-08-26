---
type: finding
status: contributing
date: 2026-08-26
mechanism: inductive-coverage (premise-diversity) — category-based induction on a two-region SimulationBridge; the
  coverage effect (diverse-2 > within-2 at matched premise count) emerges from soft-bound Hebbian concavity over
  shared category/subcategory population cores (Osherson et al. 1990 similarity-coverage; Kandel Ch 17 normalization
  + Ch 30 population-coverage)
lane: semantics / reasoning
seeds: [42, 43, 44, 100, 101, 102]
verdict: >-
  PARTIAL (2026-08-26, 6-seed, SIM_BACKEND=numpy) — the runner's OWN computed verdict. The load-bearing COVERAGE
  contrast holds robustly: diverse-2 minus within-2 = +0.05 median, POSITIVE in all 6/6 seeds, and
  diverse > within > 1-premise on every seed. Anti-cheat 2 (premise-lesion: empty premise set → no plasticity)
  collapses to 0 on every seed → PASS. Anti-cheat 1 (permuted concept codes) fires on ORDERING only — the
  diverse>within ordering vanishes cleanly (permuted diverse approximately equals permuted within; per-seed gap
  tiny) — but NOT on MAGNITUDE: a residual depolarization survives full code-scrambling at ~13% (median) of the
  diverse strength (per-seed 7.9% to 23.1%), above the runner's strict 10%-of-diverse floor, so ac1_ok is False.
  With coverage_ok True, ac2_ok True, ac1_ok False the runner returns PARTIAL (not GO). NOT overclaimed as GO per
  docs/TERMS.md (GO = the gate's own verdict is positive).
instrument: research/runners/_inductive_coverage_derisk.py — one SimulationBridge, two regions (concept +
  property-with-foil), rate-Hebbian concept→property learning, graded-population-depolarization readout above the
  untrained floor over held-out category members, dynamic-state reset before every read. Verdict on the MEDIAN over
  6 seeds. Deterministic per seed (re-run of seed 42 byte-reproduced; substrate seeded via cfg.seed).
artifacts:
  - research/findings/raw/_inductive_coverage_6seed.json
runner: research/runners/_inductive_coverage_derisk.py
biology: research/biology/inductive-coverage-premise-diversity.md
external: Osherson, Smith, Wilkie, Lopez & Shafir 1990, "Category-Based Induction," Psychological Review
  97(2):185-200 (similarity-COVERAGE model; confirmed via WebSearch, not in local corpus). Kandel Ch 17 (normalization
  companion process) + Ch 30 (population-coverage principle), both anchors resolve via tools/biology_check.py.
supersedes: none — first multi-seed validation of the 1-seed smoke committed on
  research/catalog-derisk-inductive-coverage (836d639). The commit's reported 1-seed smoke-GO does NOT reproduce
  with the committed defaults; see "What did not reproduce" below.
---

# Premise-diversity coverage is REAL and seed-robust on the substrate, but it is not yet fully SELECTIVE — the permuted-code control collapses in ordering but keeps a ~13% magnitude residual, so the 6-seed verdict is PARTIAL

Artifact: `research/findings/raw/_inductive_coverage_6seed.json`
Biology binding: `research/biology/inductive-coverage-premise-diversity.md`

## One line

Category-based induction's **coverage effect** — a generalization to a superordinate is stronger when the premises
are *diverse* (spread across subcategories) than when they are *concentrated*, at the SAME premise count (Osherson
et al. 1990) — **emerges cleanly and seed-robustly** on a two-region spiking bridge from soft-bound Hebbian concavity
alone, **but the runner's strongest anti-cheat (permuted concept codes) does not fully collapse**, so the runner's
own verdict is **PARTIAL, not GO**.

## Verified medians (traced to the cited artifact `research/findings/raw/_inductive_coverage_6seed.json`)

The runner's median-over-6-seeds summary, at full precision (these are the load-bearing values the verdict rests on;
each is `summary.<k>` in the cited JSON):

- 1-premise strength = `0.094376`
- within-subcategory(2) strength = `0.20535`
- diverse(2) strength = `0.258326`
- **COVERAGE effect  diverse(2) − within(2) = `0.052976`**  (matched premise count; runner threshold 0.02) → coverage_ok True
- diverse(2) − 1-premise = `0.16395`
- permuted diverse = `0.035119`, permuted within = `0.033319` (anti-cheat 1; ordering collapses, magnitude does not)

Strength = taught-property-block graded depolarization INCREMENT above the untrained floor, averaged over held-out
category members (one per subcategory, spanning the superordinate, never a premise).

## Per-seed detail (rounded restatement of the cited JSON's per-seed rows)

<!--derived-->
diverse > within > 1-premise is **unanimous across all 6 seeds**, and the matched-count coverage margin is positive
every seed:

| seed | 1-prem | within(2) | diverse(2) | coverage margin (div−within) | permuted-diverse residual (÷ diverse) |
|---|---|---|---|---|---|
| 42  | +0.0951 | +0.2131 | +0.2605 | +0.0474 | 16.6% |
| 43  | +0.1056 | +0.2012 | +0.2514 | +0.0502 | 23.1% |
| 44  | +0.0790 | +0.1887 | +0.2417 | +0.0530 | 13.9% |
| 100 | +0.0936 | +0.2186 | +0.2947 | +0.0761 | 7.9% |
| 101 | +0.1106 | +0.2095 | +0.2770 | +0.0674 | 13.3% |
| 102 | +0.0870 | +0.1771 | +0.2561 | +0.0791 | 10.5% |

Premise-monotonicity (within(2) − 1-premise, median ≈ +0.11) is a SEPARATE, co-existing Osherson effect (2 premises
beat 1 via the same concavity on the category core); it is not the diversity claim.

## The anti-cheats — the instrument, verified

<!--derived-->
- **Anti-cheat 2 — premise-lesion (empty premise set → no co-activation training → no plasticity applied):**
  strength **+0.0000 on every seed** → **PASS**. Without the learning step, the coverage effect is exactly zero.
  This is a genuine lesion (no plasticity is run, so nothing can regrow — the manipulation holds by construction at
  the moment of measurement, per docs/TERMS.md `lesion`).

- **Anti-cheat 1 — permuted concept codes (category/subcategory sharing destroyed; each concept gets its own
  disjoint block of matched cardinality, so held-out members share NO neurons with premises):** the **ordering
  collapses** — permuted diverse ≈ permuted within (per-seed |diverse−within| ≤ 0.004, vs the un-permuted +0.053),
  so the diversity ordering is destroyed exactly as designed. **But the magnitude does NOT collapse to floor**: a
  residual of **median ~13% of the diverse strength** (per-seed 7.9% to 23.1%) survives full code-scrambling, above
  the runner's strict floor (10% of the diverse strength), so the runner scores **ac1_ok = False**.

With coverage_ok True, ac2_ok True, ac1_ok False, the runner's verdict function returns **PARTIAL**
(`coverage_ok and (ac1_ok or ac2_ok)`).

## Why PARTIAL, honestly, and what it localizes

The coverage *effect* is not in doubt — it is unanimous across 6 seeds and vanishes entirely under premise-lesion,
driven by the soft-bound Hebbian concavity the biology binding predicts (delta_w = rate·(w_max − w) gives w2 < 2·w1,
so spreading two premises across two subcategory cores beats concentrating them on one). What the PARTIAL flags is a
**selectivity residual**: even when concept codes are fully permuted so a held-out cue overlaps *zero* premise
neurons, the taught-property block still reads out ~13% of the trained response. That residual is a non-cue-specific
component — most plausibly the taught block's own afferents being globally shifted during training (the taught
property assembly is co-active on every premise, so its incoming synapses are potentiated in a way a disjoint
held-out cue can still partially drive), which the dynamic-state reset does not remove because it is stored in the
WEIGHTS, not the dynamic state. This is exactly the kind of residual the permuted-code control exists to expose, and
the runner correctly refuses a GO while it is present above floor.

This is a genuine, reportable **PARTIAL** under the project's honesty standard: the capability (premise-diversity
coverage induction) demonstrably emerges on the substrate, but the instrument shows it is not yet cleanly SELECTIVE.
The residual is the load-bearing next lever — not a re-derivation of the effect.

## What did not reproduce

<!--derived-->
The source commit (836d639, research/catalog-derisk-inductive-coverage) reported a 1-seed numpy smoke as **GO** with
diverse(2) +0.388 and both anti-cheats PASS (permuted diverse +0.034). Running the committed file unchanged at seed
42 reproduces **deterministically** to diverse(2) +0.2605, permuted-diverse +0.0432, and **AC1 FAIL → PARTIAL** —
re-run byte-identical, so this is not RNG noise (the substrate is seeded via cfg.seed, verified). The committed
defaults therefore give PARTIAL at 1 seed and at 6 seeds; the commit's reported smoke-GO numbers do not reproduce
with the committed defaults (likely produced under different CLI params than were committed). Reported as-measured.

## Bottom line

- **Runner verdict (6-seed, its own computation): PARTIAL.**
- Coverage effect: **robust on its own** (unanimous 6/6 seeds, +0.05 median margin, premise-lesion clean).
- Blocker to a clean GO: the **permuted-code selectivity residual (~13% median)** above the runner's 10% floor.
- Not overclaimed: this is NOT a GO (docs/TERMS.md — GO requires the gate's own positive verdict), and "selective"
  is reported WITH its permuted control and raw per-seed magnitudes, not asserted.
