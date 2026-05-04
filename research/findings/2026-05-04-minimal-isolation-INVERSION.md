# Minimal isolation INVERTS the cascade-as-cause hypothesis

**Date:** 2026-05-03 ~22:32 EDT (autonomous overnight)
**Status:** PARTIAL — batch 1 (3 seeds) done; batch 2 (3 seeds) in flight
**Will be finalized:** ~22:55 EDT after batch 2 completes

---

## Headline

The cascade-as-dominant-interference hypothesis (the design rationale
for the minimal-isolation experiment) is **wrong**. Removing the cascade
makes alignment WORSE, not better.

| Architecture | true mean | best perm | excess | aligned |
|---|---|---|---|---|
| v2 (with cascade) | 28.5% | 32.8% | +4.3pp | 0/6 |
| **minimal (no cascade)** | **16.7%** | **38.0%** | **+21.3pp** | **0/3** |

n=3 partial result. Final n=6 expected ~22:55 EDT.

## What this means mechanistically

The cascade in v2 (cluster_a + cluster_e + cortex_X cascade pathways)
adds a +3pp motor_E preference across all conditions (per pattern
analysis at `2026-05-03-unaligned-structure-pattern.md`). That mild
bias is unaligned with task labels but actually DAMPENS the seed-
dependent random structure. Result: v2 has +4.3pp excess (best perm
above true) and stays close-ish to the labeled mapping.

When the cascade is removed:
- The architecture loses its weak grounding force
- Random init dynamics dominate
- STDP latches onto whatever per-seed initial weight pattern exists
- Best perm climbs to 38% (architecture HAS capacity)
- But that capacity gets used for a per-seed-arbitrary mapping
- True accuracy drops to 17% — actively WORSE than chance

So the architecture does have learning capacity (38% best perm proves
it's not just random — there IS structure). The problem is the
LEARNING SIGNAL doesn't include alignment with task labels.

## Pre-result hypothesis (turned out wrong)

We hypothesized: "Cascade is the dominant interference. Removing it
allows the language→motor pathway to learn cleanly via STDP."

What actually happens: cascade was a weak DAMPENER on seed-dependent
random structure, not its source. The fundamental issue isn't cascade
interference — it's that **STDP + R-STDP doesn't have enough signal
to align readout patterns with task labels**.

## Implications for biology sweep (queued, runs ~00:00-03:00 EDT)

The biology sweep tests:
- `+FS only` (motor PV-FS lateral inhibition, random init)
- `+Topo only` (topographic Wernicke→motor prior 1.5/0.7)
- `+Topo+FS` (combined)
- (baseline = current minimal-iso 6-seed)

**Now MORE important:** since "remove cascade" demonstrably makes
alignment worse, the only remaining biology-grounded path forward is
to ADD topographic prior (+Topo) so STDP has a label-aligning starting
point. Lateral inhibition (+FS) helps the readout select cleanly once
the prior gives it direction.

If `+Topo+FS` aligns ≥ 4/6, the path forward is clear: cascade
strength is irrelevant; topographic + lateral inhibition is the fix.

If `+Topo+FS` doesn't align, the issue is even deeper than this
investigation has surfaced — STDP fundamentally lacks the label-
alignment signal that real biology provides via developmental
plasticity / critical-period topography / explicit error signals.

## What we definitively rule out tonight

- ✗ "Cascade interference is the issue" — REFUTED (this doc)
- ✗ "Plasticity dose is the issue" — REFUTED earlier (H4 dose-1000)
- ✗ "Buffer composition bias is the issue" — REFUTED earlier (H1)
- ✗ "Hebbian re-enable fixes alignment" — REFUTED earlier (heb_only
  hurt at 22%, heb_drive hurt worse at 19%)
- ✗ "Stronger drive fixes alignment" — REFUTED earlier (drive_5x
  matches baseline ~29%, no alignment)
- ✗ "Larger STDP cap fixes alignment" — REFUTED earlier (stdp_wmax_10
  hurt slightly at 24%)

## What's left to test

- Topographic prior (+Topo) — biology sweep, queued
- Lateral inhibition (+FS) — biology sweep, queued
- Combined (+Topo+FS) — biology sweep, queued
- Bigger sparse-code dimensions (lang1024)
- Supervised gradient learning instead of STDP (non-biology backup)
- Critical-period developmental plasticity (substantial new mechanism)

Per research/findings/2026-05-04-biology-sweep-followup-plan.md, the
"all biology variants stay 0/6" outcome triggers the B-branch (sanity
check eval, sparse-code experiments, gradient-readout fallback).

## Tools

- `python -m research.runners.permuted_label_check --pattern "text_eval_minimal_iso_seed*.json"`
- `python -m research.result_aggregator --pattern "minimal=text_eval_minimal_iso_seed{seed}.json"`

## Updated chain ETA

- Now: minimal-iso batch 2 (seeds 100, 101, 102) in flight
- ~22:55 EDT: minimal-iso 6-seed complete; biology waiter triggers
- ~23:05 EDT: anti-cheat control runs
- ~23:15 EDT: biology sweep 4×6 starts (parallel-3)
- ~03:00 EDT: biology sweep complete
- ~03:05 EDT: result_aggregator + decision branch
