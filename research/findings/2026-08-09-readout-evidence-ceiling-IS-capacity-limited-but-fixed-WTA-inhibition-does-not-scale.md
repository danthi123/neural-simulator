---
title: "The world-model read-out's evidence ceiling IS capacity-limited (rises 0.77→0.96 with reservoir size) — but the FIXED WTA inhibition doesn't scale, so the neural read-out collapses"
date: 2026-08-09
type: finding
status: contributing
lane: world-model-readout
seeds: [42, 43, 44]
---

# The read-out evidence ceiling is capacity-limited; the fixed WTA inhibition is what fails to exploit it

## Claim (a cross-thread connection to the breadth capacity finding)

The last read-out negative (`1cfbb3b0f`) concluded the residual is "the STRUCTURAL rate→spike-count margin, which
neither competition nor integration lifts" — implying a hard code limit. **A reservoir-SIZE sweep (never run
before) shows that is only half true:** the host-decodable **evidence ceiling RISES MONOTONICALLY with reservoir
capacity** — the same lever that resolved the breadth crux. But the **neural spiking read-out does NOT exploit it**
because its lateral-inhibition WTA, tuned for the small reservoir, does not scale to more units. So the bottleneck
is not "the code is unfixably tied" — it is a **fixed (non-population-scaled) inhibition**.

## Data (`_fm_neural_wta_readout_derisk`, G=5, 3-seed, de-clamped substrate)

<!--derived-->

| n_pool | RIDGE / two-pathway heldout (evidence ceiling) | neural-WTA heldout (spiking read-out) |
|---|---|---|
| 250 (baseline) | 0.773 | 0.400 |
| 500 | 0.853 | 0.173 |
| 1000 | **0.960** | 0.173 |

Per-seed ridge: 250 [0.84,0.76,0.72] · 500 [0.68,0.92,0.96] · 1000 [0.88,1.0,1.0]. Per-seed neural-WTA:
250 [0.6,0.56,0.04] · 500 [0.08,0.24,0.2] · 1000 [0.04,0.32,0.16]. chance = 0.04.

## Read

<!--derived-->

- **Evidence ceiling ↑ with capacity: 0.773 → 0.853 → 0.960** (monotonic, 3-seed). The reservoir's rate-code
  separability IS capacity-limited — the "~8% structural margin" narrows as the reservoir grows. **This is
  breadth-consistent: reservoir capacity improves code separation in BOTH the teacher-loop DG (retention) AND the
  forward-model read-out (evidence).** Capacity is a real, cross-cutting code-separation lever.
- **Neural read-out ↓ / flat: 0.400 → 0.173 → 0.173.** The fixed lateral-inhibition WTA (`WTA_IE`, `ens_p`
  calibrated at n_pool=250) cannot resolve among the larger ensemble population at n_pool=500/1000 — it collapses
  toward chance even as the underlying evidence improves. The winner op is not the bottleneck (prior finding);
  the **fixed, non-population-scaled inhibition** is.
- **All arms remain NO-GO** on the strict neural-read-out bar at every n_pool — capacity alone (bigger reservoir)
  does NOT close the neural read-out, because the mechanism that reads it doesn't scale with it.

## The precisely-scoped next lever (was: "plateau"; now: two coupled fixes)

1. **Reservoir capacity** (bigger `n_pool`) to raise the evidence ceiling toward ~0.96 (demonstrated here), AND
2. **Population-SCALED divisive normalization** in the WTA (Carandini-Heeger normalization is population-size-
   invariant by construction: inhibition ∝ Σ population activity), so the neural read-out tracks the improved
   evidence instead of collapsing. The fixed `WTA_IE` must become a normalization that scales with the ensemble
   count. (The subtractive-inhibition + slow-NMDA-reverberatory read, Wang 2002, DR-recorded, is the complementary
   temporal-integration lever if scaled-normalization alone is insufficient.)

## Artifacts + rigor

Raws (3-seed each): `research/findings/raw/fm_readout_capacity_np250.json`,
`research/findings/raw/fm_readout_capacity_np500.json`, `research/findings/raw/fm_readout_capacity_np1000.json`.
Substrate seeded byte-identical (`seeded=True` all runs, verified in logs); content-path clean +
matched-sham/lesion anti-cheats are the runner's own (unchanged). 3-seed (a scoping sweep, not a 6-seed headline).

NO-EXTERNAL-NEEDED: read-out lane external round DR-recorded (subtractive-inhibition/decorrelation/Wang-2002);
Carandini-Heeger divisive normalization is the population-scaled form named there.
