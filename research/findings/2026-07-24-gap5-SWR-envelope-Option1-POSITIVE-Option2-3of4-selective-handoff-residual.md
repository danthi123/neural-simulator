# gap#5 SWR readout — Option 1 DECISIVE POSITIVE (not a substrate wall) + Option 2 SWR-envelope 3/4 ingredients confirmed, 4th (attractor-SELECTIVE hand-off) precisely localized (2026-07-24)

Result of the SWR research gate's build (`_gap5_swr_envelope_replay_derisk.py`, reuse-by-import, all 9 anti-cheats,
**NO `sim/` edit**; GPU seed 42 n_ca3=2000). The 5-method boundary is NOT a substrate wall.

## STEP 1 — Option 1 (cued completion op-point): DECISIVE POSITIVE
Sustained `recall_drive=700`/150 steps + `self_regen=0.15` + `k_thresh=110` onto assembly-0 → `per_asm_frac=[0.244,
0.0, 0.013]`. **Assembly-0 IGNITES (24.4% mean firing)**; 1&2 silent → no forward hand-off yet. ⇒ **ignition IS
achievable at the op-point; the 5-method boundary was operating-point / brain-STATE, not a substrate wall.** Residual
localized to "ignition compatible with forward hand-off + self-termination" (the research gate's [1,0,0]-type prediction).
**Correction to the boundary finding's framing:** the earlier `per_asm_active [0,0,0]` "no-ignition" was PARTLY a
detection artifact — sustained cued firing is not a discrete "event" for the windower, so `per_asm_active` read 0 while
`per_asm_frac` shows the assembly WAS igniting. The genuine residual is HAND-OFF, not ignition-from-scratch.

## STEP 2 — Option 2 (SWR envelope): HONEST PARTIAL — 3/4 SWR ingredients confirmed, 4th localized
Full 9-anti-cheat run (env_exc=550, noise_pa=4000, self_regen=0.1, period=250/env_dur=110, rest=2500):
- GO arm: ev=3 multi=3 `per_asm_active=[3,3,3]` FWD=0.000 REV=0.333 chance=0.167 **duty=0.006** (net RESTS silent),
  frozen=True, dendrite_reset=True.
- **CONFIRMED (3/4):** (1) discrete ignition; (2) self-termination (duty 0.006, discrete events not a continuous ON);
  (3) noise-seeded (NO-NOISE acid → [0,0,0] = noise IS the seed).
- **MISSING (4th):** attractor-SELECTIVE forward sequential hand-off — lands in co-fire `[3,3,3]`, forward not robust.

## The decisive diagnosis (from a NO-ENCODE ignition-SELECTIVITY gate added to the sweeps)
- **Strong drive is a DETONATOR, not a seed:** at `env_exc≥380` OR `noise_pa=4000`, the NO-ENCODE store (weights=0.5,
  no attractor) ALSO ignites [2..10] → ignition is NOT attractor-selective → co-fire `[3,3,3]`. Forward order DOES appear
  when it ignites (FWD up to 0.667 at exc=520; a knife-edge [2,2,2] FWD=0.5 at exc=550/npa=2500) but NEVER
  attractor-selectively.
- **Disinhibition variant** (sel_inhib_spare=20, env_exc≈0, basket-drop, weak noise) is ANTI-selective: the encoded
  attractor self-inhibits via its own ca3→basket feedback (silent everywhere) while incoherent cells ignite. Dead end.
- **Tuning band mapped:** env_exc=300 silent → 550 co-fire → 650 over-drives to REVERSE. **SFA constraint:** inter-envelope
  rest must exceed ~1/a_abs≈125 steps or the chain stays fatigue-locked ([0,0,0]).
- ⇒ **the residual is the knife-edge:** WEAK-enough noise to be a SEED (not detonate) + a recurrent attractor that
  SELECTIVELY amplifies it to ignition + still hands off forward. Not found in the FLAT-envelope regimes tested; NOT a
  substrate wall (Option 1 proves ignition is achievable).

## Next (ranked NEW mechanisms — all additive, no `sim/` edit; NOT 6-seed yet, no GO point)
1. **TOP — TIME-VARYING self_regen WITHIN the envelope:** LATCH (`self_regen≈0.15`) to SELECTIVELY ignite from WEAK
   noise (RANK-1 already proved weak-noise+latch = selective single-assembly ignition), then RELEASE (→0) mid-envelope so
   the latched bump de-latches and hands off via the forward links + SFA. Directly realizes "bistable to ignite, transient
   to hand off." Small additive edit to `_rest_swr_envelope` (self_regen as a per-step schedule).
2. Depolarizing RAMP within the envelope (env_exc ramps up) so the most-excitable assembly crosses threshold FIRST.
3. Weak per-envelope TARGETED seed to ONE random assembly (biologically the last-active place cell / DG mossy seed).
- **Runner fix:** add the NO-ENCODE ignition-selectivity gate to the GO CONDITION (require NO-ENCODE to NOT ignite) —
  the current forward-frac-only noencode check let a non-selective [10,10,10] pass.

Raw: `research/findings/raw/gap5_r4/swr_envelope_seed42.json`. Capability OPEN + advancing; mechanism #1 is the next build.
