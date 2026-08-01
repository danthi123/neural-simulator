---
type: finding
status: live
date: 2026-07-31
mechanism: affect-state-region
runner: research/runners/_affect_eviction_derisk.py
artifacts:
  - research/findings/raw/affect/AGG_gabab_kill_4seed.json
  - research/findings/raw/affect/_affect_eviction_SMOKE_sweep_smoke.json
  - research/findings/raw/affect/_affect_eviction_sweep_3seed.json
---

# Affect eviction: slow GABA_B feedback is KILLED as a method — on a sweep that is actually valid

**Verdict: NO-GO for the method, capability stays OPEN.**

**CONFIRMED AT 4 SEEDS (2026-07-31 20:15).** The original kill rested on seed 42 alone, which is below this
project's own bar and should not have been written up as a method-kill without saying so. Re-run on seeds
43/44/100 through the retrofitted runner, so each artifact now carries its preconditions block:

    80 sweep points across 4 seeds · 80 of 80 with a VALID arm · 3 of 80 passing G1 alone
    0 of 80 satisfying G1-G4 together · kill_criterion_met on 4 of 4 seeds

Aggregate: `research/findings/raw/affect/AGG_gabab_kill_4seed.json`, built through `tools.verdict.Verdict`.
**Honest bar note: 4 seeds, not 6.** The kill is unanimous and every arm was valid, but the standing bar is
six and this is four — stated rather than rounded up.

## The defect being attacked

The on-brain affect state is a measured **ratchet**: driven mood goes HIGH → LOW → LOW → silence and stays at
100–102% of its high value. It rises and never comes back down. This is a textbook instance of the project's
own wall-reframe — an attractor implemented *without the companion process that evicts it* — so the question
is not "what makes mood rise" but "what does the real system run alongside that brings it down".

Slow GABA_B/GIRK feedback was the first candidate: each affect pool drives its own slow-feedback interneuron.

## Why the FIRST run of this is VOID, and must not be cited

An earlier invocation today reported `NO-GO / BOUNDARY (3-seed, core 0/3)`. **That verdict is void**, for two
independent reasons, both recorded in `research/FAILURE_LOG.md`:

1. **The arm was crushed.** The runner pre-registers A5 — *"during(evict ON) ≥ 0.5 × during(baseline); else
   UNDEFINED, not a pass"* — evaluated it correctly, got `arm_valid=False` on 3 of 3 seeds, and printed NO-GO
   regardless. Every downstream ratio in that artifact is literally `null` because no signal survived.
2. **The sweep never executed.** `--sweep-weights` is consumed only inside `if a.smoke:`, so the full path
   silently ran at the `--gabab-weight` default of **1.5** — 10–30× the swept values, and precisely the
   over-strong regime the record predicted (2026-06-09-N9-SNc-rV: at 33–53 Hz GABA_B clamps to 0 rather than
   subtracting). argparse accepted the flags, so every existing check passed them.

That artifact is retained as the counter-example, not as a result.

## The valid sweep

Artifact: `research/findings/raw/affect/_affect_eviction_SMOKE_sweep_smoke.json`. The void run is retained
alongside it as `research/findings/raw/affect/_affect_eviction_sweep_3seed.json`.

Re-run through `--smoke` so the sweep machinery actually executes, at weights **below** the default:

- **20 points** over weight × τ × GIRK-cap. **All 20 have `arm_valid=True`** — the crush was entirely the
  weight-1.5 default.
- `lever_moved=True`: the manipulation engaged.
- **G2, G3, G4, G5, G6 all pass** at the closest point — drive-dependence, reignition, persistence, the
  NMDA-off control and the lesion-restores-ratchet power control. The instrument is verified.
- **G1 — the one that matters — fails at `evict_ratio_low = 0.975`.** The mood sits at 97.5% of its peak.
- **1 of 20** points passes G1 in isolation; **0 of 20** satisfy G1–G4 together.

`kill_criterion_met: True`, exactly as pre-registered: *no (weight × τ) point satisfies G1–G4 together ⇒ slow
GABA_B feedback is killed as the method.*

The runner's own attribution line is the honest summary: **"ATTRIBUTION UNDEFINED — the slow-GABA_B arm did
not evict either, so there is no effect to attribute to slowness."** The comparison against the fast control
cannot be made, because neither arm evicted.

## Consequence

Under THE LAW a killed method does not close a capability. Mood eviction stays **open**; slow GABA_B feedback
is retired as the route to it. The runner already names the successor — **intrinsic spike-frequency
adaptation** (`--sfa IZH_A IZH_D`, a slow after-hyperpolarization) — and that arm has **never run**:
`sfa_arm` is null in every artifact on record, because the flag was omitted from each invocation. Three
settings are queued.

The wider lesson is the one this session keeps re-learning: the first run produced a plausible negative from
a crushed arm, and the difference between a void verdict and a real one was entirely in whether the
instrument was checked before the result was read.
