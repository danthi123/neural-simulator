---
type: research-gate
status: selected
date: 2026-08-08
mechanism: affect-state-region-graded-ladder
lane: A (affect / emotion keystone)
artifacts: []
---

# Graded affect persistence — research gate: the surpass is a BISTABLE-LADDER robust discrete integrator (Koulakov-2002 / Goldman-2003), NOT another continuous line attractor

**One-line decision:** two methods have now failed to hold a *graded* affect value over time in the same way —
a single point-neuron slow-NMDA opponent pool (P0.3, 2026-07-24) that **ignites and saturates** into a bistable
good/bad LATCH, and a continuous **line/bump attractor** (Wave 1) that **collapsed to a point attractor**
(held range 0.003 vs 0.07 input). <!--derived: quoted from Wave-1 task brief--> Both failures share one root: a *marginally-stable continuum* has no robust
graded middle — it either saturates to a latch or drifts/relaxes to a point. The genuinely-new, different-in-kind
mechanism is to **stop trying to hold a continuum** and instead hold graded value as the **integer count of
independently-latched bistable sub-pools recruited at staggered thresholds** — the robust discrete integrator of
Koulakov 2002 and Goldman/Seung 2003, whose entire design purpose is drift-free graded persistence. **It is
buildable NOW with NO `sim/` edit** (every primitive is proven in the P0.3 runner), with one honest build risk
(intra-ladder collapse) the smoke must resolve. The single-cell gold standard (Egorov CAN intrinsic graded
persistence) is a second, stronger mechanism that **needs a `sim/` change** (no I_CAN in Izhikevich) — reported
below as a design note, not the first build.

## Research gate: what the record + literature say (READ, not skimmed)

**Lever count against ONE defect (graded affect persistence) = 2 distinct methods, both failed ⇒ the gate FIRES**
(per CLAUDE.md, ≥2 levers on one defect ⇒ research before the next lever):

1. **P0.3 opponent slow-NMDA point-neuron attractor** (`2026-07-24-P0.3-affect-state-region-6seed-GO.md`,
   verdict QUALIFIED-GO/BOUNDARY): the three core faculties (persistence / causal-bias / value⟂plausibility)
   held 6/6, but the 4th gate — GRADED history-integration — is a characterized open boundary: the pool
   *"IGNITES at a low threshold and SATURATES"*, sustained mood flat ~0.09–0.11 across appraisal drive 0.15→1.0
   (magnitude r=0.33 unreliable, sign-crossing 1/6 seeds). **A bistable good/bad LATCH, not a graded circumplex.**
   That doc itself names the surpass: *"a graded Russell circumplex needs graded persistent activity — a
   line/bump attractor … / the dendritic substrate."*
2. **Line/bump attractor** (Wave 1): collapsed to a point attractor, held range **0.003 vs 0.07 input** <!--derived: quoted from Wave-1 task brief--> — the
   graded value is destroyed in persistence. This is the classic **line-attractor drift** failure (Seung 1996):
   a continuous attractor is a knife-edge (recurrent gain must be tuned to α≈1 to a fraction of a percent), and
   any detuning relaxes the bump to the nearest stable point.

**The corpus already characterized the *class* boundary that unifies both** — the STP-annihilation finding
(`2026-08-01-affect-ratchet-STP-annihilates-…`): *"There is no graded middle because a bistable attractor has no
graded middle."* And the eviction gate concluded the next mechanism *"is a NON-brake, and it needs a research
gate not another sweep."* This gate is that research.

**The WKV state-fidelity gate is the load-bearing pointer** (`2026-07-19-onbridge-wkv-state-fidelity-research-gate.md`):
its five line-attractor de-risks **all read the population MEAN FIRING RATE and all capped ≤0.55**, and it
concluded — *reading the source, not the rerank hit* — that **"biology holds integrator state in graded slow
conductances, not spike rates: NMDA plateaus, Wong–Wang 2006 recurrent-NMDA attractors, Seung-1996 /
Goldman-2003 line-attractor persistent activity."** It also records that the substrate **already exposes** two
graded slow-conductance channels — `cp_ssm_state` (per-neuron leaky SSM integrator, byte-equal to numpy) and
`cp_conductance_g_graded_plateau` (per-neuron dendritic Ca-plateau conductance, 0.98 for a clean input,
`graded_plateau_tau_decay_ms≈80`). **Both confirmed per-neuron and additive/default-off** (`sim/bridge.py:324,
360, 1544, 1649, 6731`): byte-identical when the flag is off.

**External literature (the different-in-kind mechanisms):**
- **Koulakov, Raghavachari, Kepecs, Lisman 2002, "Model for a robust neural integrator" (Nat Neurosci 5:775):**
  a line attractor built from **bistable units** is *drift-robust* precisely because it is discrete — between
  the stable levels, perturbations relax back to the nearest level instead of wandering along a continuum. The
  robustness↔resolution trade-off is fundamental: you buy drift-immunity with quantization.
- **Goldman, Levine, Major, Tank, Seung 2003 (Cereb Cortex 13:1185), "Robust persistent neural activity in a
  model integrator with multiple hysteretic dendrites per neuron":** the *same* robustness obtained with
  **multiple bistable dendritic plateaus per neuron** — directly relevant to the substrate's existing per-neuron
  graded dendritic plateau conductance.
- **Egorov, Hamam, Fransén, Hasselmo, Alonso 2002, "Graded persistent activity in entorhinal cortex neurons"
  (Nature 420:173)** + **Fransén et al. 2006 (Neuron 49:735):** the single-cell gold standard — a
  calcium-activated non-specific cation current (I_CAN, TRPM4/5) that lets ONE cell hold many stable graded
  firing levels, stepped up/down by brief inputs, *self-sustaining across silence with no network tuning*. This
  is the mechanism affect most likely uses in vivo; it needs an intrinsic current the Izhikevich substrate lacks.

## The candidate mechanism (build-ready) — a staggered bistable LADDER read as a population conductance

**Different in kind from both failures.** Not one population on a continuous manifold (drifts → Wave 1), not one
saturating latch (→ P0.3). Instead **N independent self-recurrent bistable sub-pools per valence sign**
(`affect_vplus_L1..LN`, `affect_vminus_L1..LN`), each latched by its OWN within-pool slow-NMDA recurrence, with
**staggered recruitment thresholds** so appraisal magnitude *m* turns ON sub-pools 1..k(m). Held value = the
**number of latched sub-pools = graded population firing rate** ∝ k. Robust to drift because there is no
continuum to drift along: each sub-pool sits in its own ON/OFF basin (Koulakov). Holds across silence because
each latch is self-sustaining NMDA (P0.3 already proved this holds 0.62 of peak with NMDA-on vs 0.00 off).

**Every primitive is already proven in `research/runners/_affect_state_region_derisk.py` (reuse-by-import, no
`sim/` edit):**
- Self-recurrent NMDA bistable pool = `BrainRegion(internal_density=RECUR_DENSITY, exc_weight_mean=recur_weight,
  enable_nmda=True)` — the existing `aff()` factory, instantiated N times per sign.
- **Staggered recruitment thresholds** = a per-sub-pool tonic offset written to
  `cp_external_input_current[idx[pool_k]]` (a constant negative bias that raises pool *k*'s effective threshold),
  OR a per-pool `excitability_drive` sensitivity in the appraisal `NeuromodulatorConfig` (already scoped
  `group:<pool>`). Higher appraisal concentration crosses more pools' thresholds.
- **Appraisal → ladder** = the existing diffuse neuromodulator bus (volume transmission, `excitability_drive`),
  broadcast to all sub-pools of a sign; magnitude is the concentration.
- **NO intra-ladder lateral inhibition** (the critical design rule — see risk 1). The opponent V+/V-
  cross-inhibition (Namburi-Tye) is applied only at the **aggregate** level (a V+ summary interneuron pool ⊣ the
  V- ladder and vice-versa), never between sub-pools of the same sign.
- **STATE read = population rate differential** `rate(Σ V+ ladder) − rate(Σ V- ladder)`, delivered downstream as
  synaptic current through the existing `affect_out` transmission gate to `recall_pos/recall_neg/speak_acc`.
  **This is a neural read (population firing → synaptic drive), NEVER a host count/argmax** — the honesty line
  Wave 1 B/D crossed with host argmax / index-multiply proxies.
- **Optional conductance-read upgrade** (stronger, still no `sim/` edit): route the ladder's summary spikes into
  `cp_conductance_g_graded_plateau` and read the graded plateau — the WKV gate's untested "attractor + graded
  conductance read" combination. Deferred to the second iteration; the rate read is the first, cheapest test.

## Anti-cheats WITH TEETH (each comparator can flip in the failing direction)

1. **GRADED-HOLD (falsifies the P0.3 saturation):** drive to magnitude *m* ∈ {0.2, 0.4, 0.6, 0.8, 1.0}, then
   drive-OFF ≥1 s; held level must **monotonically track *m*** — Spearman ρ(m, held) ≥ 0.8 **and** held-range
   ≥ 0.05 across the *m* sweep. **Flips:** saturation → ρ≈0 / range≈0 (the P0.3 read); a 2-level latch → range
   collapses. A held range of 0.003 (Wave 1) fails outright. <!--derived: quoted from Wave-1 task brief-->
2. **DRIFT-ROBUSTNESS (falsifies the Wave-1 collapse; the like-for-like that proves different-in-kind):** hold a
   mid level, no drive, T=2 s; measure |slope| and variance of the held level. Run a **matched same-N single-pool
   continuous line-attractor CONTROL** in the same harness. Teeth: the **control must drift** (|slope| large /
   relaxes toward a point, reproducing Wave 1's 0.003 <!--derived: quoted from Wave-1 task brief-->) while the **ladder stays flat**. If the ladder drifts too,
   the mechanism is refuted — it flips.
3. **LADDER-LESION + matched SHAM (falsifies "it's just tonic bias, not the latch"):** real lesion = zero the
   ladder's within-pool NMDA recurrence (or `enable_nmda=False` on the ladder) → held must **collapse to ≈0**.
   Matched SHAM = zero an equal-neuron-count *unrelated* recurrence (`speak_acc` internal recurrence, same neuron
   count) → held must **survive**. Teeth: real flips, sham does not; a tautological lesion (zeroing the read
   itself) is explicitly avoided — the lesion targets the *recurrence*, the read stays the population rate.
4. **NMDA-OFF dissociation (mechanism attribution, from P0.3):** whole-ladder NMDA-off must decay to <0.1 of
   peak — proves persistence is the slow-NMDA latch, not the tonic recruitment bias.

`cfg.seed` set per-seed (verify: two builds at one seed hash-identical on `cp_neuron_firing_thresholds`,
cross-seed differ — the P0.3 runner already does this). 6-seed gate (42 43 44 100 101 102), SIM_BACKEND=numpy
(CPU lane).

## Honest feasibility — `buildable_now = YES` for the ladder, with one real build risk; the CAN variant NEEDS a `sim/` change

**Buildable now (ladder):** YES — no `sim/` edit; a runner-level extension of the P0.3 `AffectStateBrain`
(N sub-pools per sign + staggered tonic offsets + aggregate-only opponent inhibition + population-rate read).
No new bridge attribute is required.

**Honest risks (build risks, not buildability blockers):**
1. **Intra-ladder collapse (the load-bearing risk).** If sub-pools of one sign share *any* lateral inhibition or
   cross-recurrence, the incumbent suppresses the challengers → WTA hysteresis → the P0.3 latch returns (2 levels,
   not N). The design forbids intra-sign inhibition, but the pools still share the diffuse appraisal broadcast and
   the aggregate opponent interneuron; the smoke must confirm the staircase does not collapse to all-or-none.
   This is the single thing most likely to sink it — it is a *different* failure mode from the two banked ones,
   so testing it is genuine forward progress either way.
2. **Quantization is a feature, reported as one, not a smooth continuum.** N=6–8 sub-pools ⇒ a 6–8-level monotone
   staircase, not a Russell continuum. Per Koulakov this is *fundamental*: drift-robustness is bought with
   resolution. The honest claim is "a robust *quantized* graded persistent code," and the gate bar (ρ≥0.8,
   range≥0.05) is written for a staircase, not a continuum. Selling it as a smooth circumplex would be an
   overclaim.
3. **Recruitment-vs-latch confound.** The tonic staggering bias must not itself sustain a pool after drive-OFF
   (that would be host-injected persistence, not the NMDA latch). Anti-cheat 4 (NMDA-off decay) is the teeth.
4. **The stronger mechanism (Egorov I_CAN single-cell graded persistence) NEEDS a `sim/` change** — the
   Izhikevich substrate has no calcium-activated cation current; adding I_CAN (a Ca-gated depolarizing
   conductance with the Fransén-2006 dynamics) is an additive intrinsic current on the neuron model. **Reported
   as a design note, deferred:** the ladder buys the *capability* now on the existing substrate; the CAN current
   is the faithful single-cell upgrade to schedule if/when the ladder's quantization proves insufficient for the
   circumplex. The existing per-neuron `cp_conductance_g_graded_plateau` is the closest current analogue and is
   the basis of the optional conductance-read upgrade above (Goldman-2003 multiple-hysteretic-plateau flavor),
   but it is a leaky integrator (decays across silence) — it does NOT self-sustain a graded level alone, which is
   exactly why the *bistable* ladder (not the bare plateau/SSM state) is the mechanism.

## Why this is not a retry of the failed method

The line/bump attractor and this ladder are different *kinds* of dynamical object: a continuous attractor is one
population with a marginally-stable 1-D manifold of fixed points (fine-tuned gain, drifts — Seung/Wave 1); the
ladder is N *disjoint* bistable systems each with a robust 2-point basin, coupled only by staggered feedforward
recruitment (Koulakov's explicit robust *alternative* to the drift-prone continuum). It is not the P0.3 latch
either: that was ONE bistable pool (2 states); this is N (N+1 levels). No parameter of the failed methods is
being re-tuned — the topology is different.
```
<!--derived-->
FAILED METHOD 1 (P0.3):  sustained mood 0.09–0.11 flat across drive 0.15→1.0 ; magnitude r=0.33 (unreliable)
FAILED METHOD 2 (Wave1): held range 0.003 vs 0.07 input (collapsed to point attractor)
GATE BAR (ladder):       Spearman ρ(m,held) ≥ 0.8 AND held-range ≥ 0.05 ; drift |slope| < control ; real-lesion collapses, sham survives
```
Numbers for FAILED METHOD 1/2 are quoted from `2026-07-24-P0.3-…` and the Wave-1 task brief; the GATE BAR row
is the pre-registered target for the un-built ladder, not a measured result.
