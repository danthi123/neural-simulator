# Nav Stage-B neural-critic + GABA_B — 2-seed smoke NEGATIVE (SNc silenced; honest negative + diagnosis)

**Date:** 2026-06-08
**Type:** Honest NEGATIVE de-risk (the cheap-first smoke gate failed → the 6-seed A/B was NOT run, per the pre-registered gate).
**Predecessors:** `2026-06-08-gabab-girk-stageB-derisk-GO.md` (the GABA_B edit + CPU de-risk that PASSED); the nav integration build (commit pending, runner-side only, `sim/` byte-empty).

## What was tested

The BRAIN-BASED-ONLY nav completion: `--enable-neural-critic` adds a dedicated GABAergic `striosome_value`
critic (driven by the perceived ventral object code `cortex_it`, plastic afferent trained by the SNc δ),
routes `striosome_value → snc` `receptor="gaba_b"` + `cfg.enable_gabab=True`, and DROPS the host
`−snc_value_gain·_V_scaffold` term in `_I_snc` so the value subtraction `r − V` happens at the SNc membrane
via the neural GABA_B/GIRK inhibition. Smoke = flagship A+E+G v2.5 + `--spiking-snc --enable-neural-critic`,
seed 42 (a complete 1800-step run; seed 43 not needed once seed 42 failed decisively).

## Result — NEGATIVE (the critic config catastrophically fails)

| Metric | Stage A (`--spiking-snc`, host scaffold) | Neural-critic (`--enable-neural-critic`) |
|---|---|---|
| SNc firing (`snc_rate_log` mean) | **7.16 Hz** (max 13) | **0.0 Hz** (all zero) |
| Critic `striosome_value` firing (reward windows) | n/a | **0.0** (all zero in the logged windows) |
| Critic weight (`cortex_it→striosome_value`) | n/a | **2.990 → 2.990 (no learning)** |
| Nav: summed final-quarter distance | **2.0** | **133.0** |
| Nav: steps at goal | (good) | **0** (never reached the goal) |

The neural-critic path **silenced the SNc** (0 Hz vs Stage A's 7 Hz). With the SNc silent, the dopamine signal
(`from_region_firing_signed` on the SNc pool) is dead → no three-factor learning anywhere → the critic weight
stays frozen at init AND the actor cannot learn → navigation collapses (distance ~32 on a 32×32 grid, never
reaching the goal). This is an honest negative: it maps a real integration failure, **not** a property of the
GABA_B conductance itself (which passed its CPU de-risk 3/3 and is byte-identical-when-off).

## Diagnosis so far (the paradox + the prime suspect)

The puzzle: the neural-critic drive `_I_snc = snc_tonic(220) + snc_reward_gain·max(0,reward)` is **larger**
than Stage A's `220 + reward − snc_value_gain·_V_scaffold` (Stage A *subtracts* the host value; the neural
path drops that subtraction). A *larger* depolarizing drive producing *zero* firing means **something is
adding strong hyperpolarization** in the neural-critic path. Candidates, in order of suspicion:

1. **GABA_B over-inhibition from a critic that fires outside the logged windows (PRIME SUSPECT).**
   `striov_rate_log` only counts critic spikes during the brief reward-hold windows (runner ~line 5268), so
   `striov_rate_log == 0` does **not** prove the critic never fires — it may fire during the regular nav
   steps (driven by `cortex_it`), pumping the **slow** GABA_B conductance (τ = 150 ms) on the SNc. The slow
   decay would sustain a hyperpolarizing K⁺ current (E_K = −90 mV) across many steps → SNc silenced. The CPU
   de-risk did NOT expose this because it was a clean train-then-test Pavlovian schedule, not a continuous
   free-running nav loop with persistent cortical drive to the critic.
2. **SNc readout index** (`_snc_idx_host`) reading the wrong neurons after the `striosome_value` region was
   added — but this is unlikely the whole story, because nav itself collapses (the RPE really is dead, not
   just mis-measured).
3. **A per-action DA / synapse-mask misalignment** from the added region/synapses shifting the synapse count.

## Next — instrumented root-cause (NOT brute-force)

A focused diagnostic must localize the SNc silence (the load-bearing question, since the fix depends on the
cause): a short (~100–200 step) neural-critic build that logs `cortex_it` / `striosome_value` / `snc` firing
**every step** (not just reward windows), the GABA_B conductance magnitude on the SNc, and a NaN check —
compared against the same with `cfg.enable_gabab` forced off (critic on) and against Stage A. Likely fixes if
the prime suspect holds: a much faster GABA_B τ for the nav loop, a weaker `gabab_propagation_strength`, gating
the critic→SNc GABA_B to the reward window only, or rebalancing the SNc tonic. An honest "the slow GABA_B
subtraction is incompatible with the continuous nav loop without further mechanism" would itself be a valid
deliverable (it maps a limit of porting the Pavlovian-validated mechanism into free-running navigation).

## Status

GABA_B conductance (the protected `sim/` edit): VALIDATED + shipped (commit a7370d49), unaffected — its CPU
de-risk PASSED and it is byte-identical-when-off. What failed is the **nav-runner integration** of it (a
runner-side calibration/interaction issue, `sim/` byte-empty). The 6-seed A/B gate is NOT run (the smoke gate
failed). Next: instrumented diagnosis → fix-or-flag.
