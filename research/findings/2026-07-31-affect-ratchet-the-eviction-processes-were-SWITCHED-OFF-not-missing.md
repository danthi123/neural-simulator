---
type: finding
status: live
date: 2026-07-31
mechanism: affect-state-region
artifacts:
  - research/findings/raw/affect/_affect_eviction_SMOKE_sweep_smoke.json
  - research/findings/raw/affect/_affect_sfa_0p005_60_smoke.json
---

# The affect ratchet: the processes that would evict it were SWITCHED OFF, not missing

**This is wall-cause 1 in its purest form, and stronger than the usual version.** The standing diagnosis is
"biology runs interacting processes, we implement one and substitute a static proxy for the rest". Here the
companion processes were not proxied by a constant — they were **disabled outright**, and the resulting
behaviour was then treated as a property of the mechanism that needed a new mechanism to fix.

## The sequence

The on-brain affect state is a measured ratchet: mood rises and never returns to baseline (3/3 seeds,
100–102% of peak). Two levers were built against it, both aimed at *braking* the mood:

1. **Slow GABA_B/GIRK feedback** — KILLED as a method on a valid 20-point sweep (all arms valid, G2–G6
   passing, G1 failing at `evict_ratio_low` 0.975). Artifact:
   `research/findings/raw/affect/_affect_eviction_SMOKE_sweep_smoke.json`.
2. **Intrinsic spike-frequency adaptation (sAHP)** — UNCONTROLLED, not negative: its power control gates a
   synaptic pathway and cannot reach an intrinsic mechanism. Artifact:
   `research/findings/raw/affect/_affect_sfa_0p005_60_smoke.json`.

Two levers without resolution fired the research gate. The corpus check returned the mechanism twice, and
reading the source settled it.

## What the record already said

`2026-07-24-P0.3-affect-state-region-6seed-GO.md` states it directly: persistence **is** the slow-NMDA
reverberatory attractor — mood retains 0.62 of peak with NMDA on versus 0.00 with NMDA off, per-seed
0.60–0.65 across 6 seeds. And critically: *"The point-neuron slow-NMDA opponent attractor **IGNITES at a low
threshold and SATURATES**."*

Today's valid sweep reproduces exactly that: `nmda_off_retention` 0.0, `persistence_retention` 0.5667.

So both levers were adding an outward brake against a **saturated** recurrent loop. That is why neither
moved it, and it was predictable from the record before either was built.

## The actual cause

`research/runners/_affect_eviction_derisk.py:224` and `research/runners/_affect_state_region_derisk.py:149`
both contain:

```python
cfg.enable_short_term_plasticity = False
```

inside a block that also sets `enable_stdp`, `enable_reward_modulation`, `enable_hebbian_learning`,
`enable_homeostasis` and `enable_structural_plasticity` all to `False`.

**The recurrent NMDA loop runs with no synaptic depression, no homeostasis, and no adaptation of any kind.**
A reverberatory attractor whose own drive can never weaken has no way to terminate, so it saturates and
latches — which is precisely the observed ratchet.

The isolation is defensible on its own terms: it was chosen so the attractor is the only live dynamic. What
is not defensible is reading the resulting latch as a property of the mechanism and building brakes for it.
Short-term depression is present in the engine and ON by default (`sim/config.py:577`), with
`stp_tau_d = 200.0` ms — itself far too fast to accumulate against a seconds-long reverberation, so even
re-enabling it needs a slower recovery constant, not just a flag flip.

## Next lever, and an honest note on why it is not yet run

Let the loop depress: enable short-term plasticity on the affect region's recurrent synapses with a
depression-recovery τ on the seconds scale, keeping the rest of the isolation intact so it stays a
one-variable change.

I started wiring a `--stp-tau-d` flag and **reverted it deliberately**. The flag reached the constructor but
`run_point`, `run_smoke` and `main` did not thread it through roughly eight call sites, so it would have been
accepted by argparse and silently inert — the exact defect logged earlier today, where
`--sweep-weights` was accepted and ignored and produced a void NO-GO. A half-wired flag in the tree is a trap
for the next reader. The wiring is mechanical and should be done in one pass, with the lever's effect
asserted before any arm is read.

Also still owed: the sAHP power control. `set_sfa_lesion` now exists (it restores the pre-`--sfa`
`cp_izh_a` / `cp_izh_d_increment` on the same substrate and asserts the write landed), but it is not yet
wired into G6's evaluation for the sfa arm. Until it is, sAHP has no verdict.
