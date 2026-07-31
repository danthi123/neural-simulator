---
type: finding
status: contributing
date: 2026-07-17
mechanism: ca3-completion
---

# Gap #5 CA3 completion — ROOT-CAUSED by a 6-agent adversarially-verified workflow: the blocker is a TRAINING Hebbian-collapse confound, NOT a substrate floor and NOT a transmission bug (both prior diagnoses refuted). The substrate CAN complete; the close is a learned pattern-specific attractor + dendritic-dAP completion.

**2026-07-17.** Gap #5 (owner chose "whichever closes quicker"). Two prior diagnoses of the CA3-completion failure
were **both refuted** this session: (1) the 2026-07-08 "silent recurrents / ~1000× too weak transmission bug" — a
direct g_e probe showed the recurrents transmit + scale with weight; (2) the "point-neuron completion boundary" — a
strong hand-installed symmetric attractor completes. A weight×density completion sweep then returned BYTE-IDENTICAL
results across a 5× weight / 2× density range (a silent-failure signature), so a 6-agent Workflow (4 parallel
hypotheses → synthesis → adversarial verify) diagnosed it.

## The verified diagnosis (unanimous, cross-consistent, adversarially confirmed)

| hypothesis | verdict | decisive evidence |
|---|---|---|
| **H1** build/train params not applied | **REFUTED → root cause** | build params DO apply (density 0.30→6654 syn / 0.60→13312; weight 120→build mean 120.07 / 600→599.64). BUT after training BOTH collapse to an IDENTICAL ca3→ca3 mean=min=max=**0.846**. Hebbian-OFF control: 120.07→120.07 (unchanged); Hebbian-ON: 120 *and* 600 → 0.846. |
| **H2** held-out never reaches threshold at recall | **SUPPORTED** | 0/8 held-out fire at both w=120 and w=600; max Vm −52.9 (10.6 mV short of −42.3 threshold); g_e 0.014 vs 0.015 (weight-insensitive — because the effective weight is the collapsed 0.846). |
| **H3** the completion metric is broken | **REFUTED** | positive control: force held-out firing (800 pA) → heldout_completion 0.000→**6.023**, own_cos 0.541→0.745. The metric faithfully reports "not completed" as a REAL result. |
| **H4** genuine point-neuron floor (can't complete at any weight) | **REFUTED** | a hand-installed strong symmetric attractor DOES complete: held-out ignites to match the cue, non-ensemble silent, recurrence-dependent (w=0 control silent). |

**Root cause:** `cfg.enable_hebbian_learning=True` (`_riii_..._derisk.py:35`) drives every ca3→ca3 recurrent to a
uniform ~0.846 fixed point during training (the documented CLAUDE.md Hebbian-collapse gotcha) — **erasing the swept
weight AND stdp_w_max before recall.** 0.846 × propagation 0.05 ≈ 0.04 mV/spike, sub-threshold at any density. That
is why the sweep was byte-identical (a dead weight axis) and why held-out stays at the pure drive artifact
(heldout≈0.027, own_cos≈√0.5=0.506). The prompt's direct g_e probe saw weight-scaling only because it OVERRODE
`cp_connections`, bypassing the Hebbian collapse.

## The adversarial verifier's crucial correction (holds=True, but two corrections)
1. **QUANTITATIVE (flips the mechanism ordering):** H4's completion threshold is **NON-REPRODUCIBLE** as stated.
   Independently hand-installing the attractor with frozen plasticity: completion ~0.000 at raw w=1600 AND w=3000
   (w=3000 actually SUPPRESSES the cue); real completion appears only at raw w≈**6000** (held-out 0.172 ≈ cue 0.163,
   non-ensemble 0.001, w=0 silent). So the point-neuron all-to-all volley needs ~4× the weight H4 reported, and it is
   finicky (intermediate weights suppress). The built/swept weights are only ~120–600, so even a de-confounded sweep
   (Hebbian-OFF preserving 120–600) will likely make the weight axis LIVE but STILL fall short of the >0.30 gate.
   ⇒ the **dendritic-dAP completion (already 6-seed GO, completes at far lower weight) should be the PRIMARY closer**,
   with symmetric recurrent-LTP as the trained-attractor formation feeding it; the point-neuron volley is a secondary
   path only at extreme weight.
2. **SECONDARY:** the diagnostic's no-train arm builds ca3→ca3 at a hardcoded `weight_mean=1.5` (`_build(train=False)`,
   line ~29) while the trained arm builds at the swept weight — a real weight asymmetry to fix so trained-vs-notrain
   differ ONLY in training.

## The close (verified plan, cheapest-first)
- **STEP 1 (decisive, near-free):** de-confound in a scratchpad copy — freeze the ca3→ca3 Hebbian
  (`enable_hebbian_learning=False`, or the ca3→ca3 `RegionPathway plastic=False` / hold `ca3_swr_burst` at 0 during
  training) + fix the `train=False` weight=1.5 bug → re-run the sweep. The weight axis goes LIVE (completion moves with
  weight) — empirically confirms the confound. (Do NOT treat STEP-1 passing as closure; built weights 120–600 are
  below the ignition regime.)
- **STEP 2 (the closer):** a LEARNED, pattern-specific attractor via **symmetric recurrent-LTP** (the guarded
  `hebbian_symmetric` config exists but is capped at `hebbian_max_weight=30` and forms only a WEAK +0.87 attractor —
  raise the cap + a RATE-WINDOW co-activity rule / many events to reach the ignition weight) + **dendritic-dAP
  completion** (`2026-07-08-riii-onsubstrate-dendritic-dAP-completion-SURPASS-6seed.md`, completes at far lower weight
  than the ~6000 point-neuron volley). GO gate: trained held-out completion > 0.30 & recurrence-gain > 0.15 &
  specificity (non-ensemble silent) & no-train collapse, 6-seed.
- Then wire completion → the SWR generative-replay loop → a queryable console.

⇒ **gap #5 completion is achievable** — it was a training-rule confound, not a substrate wall. The substrate
completes; the work is to LEARN a strong pattern-specific attractor and read it out with the (already-GO) dendritic
completion. Both prior "boundary" diagnoses are formally refuted with reproductions.
