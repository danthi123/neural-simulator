---
type: finding
status: qualified
date: 2026-08-08
mechanism: affect-state-region-graded-ladder
lane: A (affect / emotion keystone)
artifacts:
  - research/findings/raw/_affect_graded_ladder_6seed.json
  - research/findings/raw/_affect_graded_ladder_6seed_smoke.json
---

# Graded affect via a STAGGERED BISTABLE LADDER — 6-seed GO (a drift-robust QUANTIZED value code, not a continuum)

**Verdict: GO (6/6 seeds).** The Wave-2a mechanism surpasses the two banked failures for holding a *graded* affect
value over time. Appraisal magnitude is held as the **integer count of independently-latched slow-NMDA sub-pools**
recruited at staggered thresholds (Koulakov et al. 2002 robust discrete integrator), read NEURALLY as a downstream
population firing rate. It is **different in kind** from both failures and does not re-tune either. **Honest scope:
this is a QUANTIZED N+1-level staircase, NOT a smooth Russell continuum** — Koulakov's robustness is bought with
resolution; the smooth-continuum / single-cell-I_CAN surpass remains a future upgrade (see "what this is not").

Artifact: `research/findings/raw/_affect_graded_ladder_6seed.json`. Runner (NO `sim/` edit, reuse-by-import):
`research/runners/_affect_graded_ladder_derisk.py`.

## What was banked, and why this is not a retry (from the research gate)

The research gate `2026-08-08-graded-affect-persistence-research-gate-bistable-ladder-robust-integrator.md`
recorded two methods that both failed to hold a graded value:
- **P0.3 single point-neuron slow-NMDA opponent pool** — ignites and SATURATES to a good/bad LATCH (sustained mood
  flat ~0.09-0.11 across appraisal 0.15->1.0). A bistable latch, not a graded code.
- **Wave-1 line/bump attractor** — collapsed to a point attractor (held range 0.003 vs 0.07 input, <!--derived--> quoted from the research gate); classic marginally-stable-continuum drift.

The ladder is N *disjoint* bistable systems, each with a robust 2-point basin, coupled only by staggered
feedforward recruitment — a different dynamical object from a continuum (drifts) or one saturating latch (2 states).

## Mechanism (built on ONE numpy spiking Izhikevich bridge; NO `sim/` edit)

- N=6 self-recurrent slow-NMDA sub-pools per valence sign (`affect_vplus_L1..L6`, `affect_vminus_L1..L6`), each
  latched by its OWN within-pool NMDA recurrence (the P0.3 `aff()` bistable pool, instantiated 6x per sign).
- **Staggered recruitment** = a per-rung cell-autonomous intrinsic excitability offset (descending, via
  `BrainRegion.intrinsic_current_pA`). Recruitment is linear in appraisal m: rung *i* latches when
  `off_i + gain*m` crosses ignition. **Calibration lesson (measured): the offset range is bounded below by a
  HOLDING FLOOR** — during silence a rung sits at its bare offset, so a too-negative rung (< ~-180 pA here) is
  monostable-OFF and cannot persist; the count then caps below N. `off_hi=40 / off_step=42` keeps the deepest rung
  at -170 pA (holdable) while spreading recruitment across m.
- **Appraisal** = a UNIFORM diffuse neuromodulator broadcast (`excitability_drive` volume transmission), delivered
  as a graded RAMP. A synchronous step-onset over-ignites every ready rung at once (the intra-ladder-collapse
  failure mode — confirmed, see AC below); the ramp recruits rungs sequentially per Koulakov.
- **NO intra-sign lateral inhibition / cross-recurrence** (the critical design rule). Sub-pools of one sign share
  only the appraisal broadcast and a feedforward readout.
- **Namburi-Tye opponent cross-inhibition ONLY at the AGGREGATE** (`agg_plus`/`agg_minus` interneuron pools: the
  V+ summary ⊣ the whole V- ladder and vice-versa), never between same-sign sub-pools.
- **State read = population-rate differential** `rate(pos_readout) - rate(neg_readout)`, where `pos_readout`
  receives a fixed feedforward projection from every V+ rung through the `affect_out` transmission gate. This is a
  NEURAL read (ladder spikes -> synaptic drive -> a downstream population's rate), never a host count/argmax. The
  host-side latched-count is a diagnostic only.

## Anti-cheats with teeth (6/6 seeds; each comparator can flip in the failing direction)

The rows below are computed (min-max / rounded across seeds) from `research/findings/raw/_affect_graded_ladder_6seed.json`.

<!--derived-->

| anti-cheat | result (6-seed) | teeth |
|---|---|---|
| AC1 MONOTONIC STAIRCASE (neural readout rate vs m) | Spearman rho = **1.00** all seeds; held-range 0.055-0.069 (bar 0.05); held counts `[0,1,2,3,4,4/5]` (5-6 distinct levels) | vs single lumped pool (P0.3 latch) and vs UNSTAGGERED ladder — **both collapse to 2 plateaus** (neural readout too: single-pool flat after ignition, unstaggered all-or-none 0->0.085) |
| AC2 PERSISTENCE / DRIFT | retention **1.04** NMDA-on (held after 800 ms silence) vs **0.00** NMDA-off; drift 0.04 over 1 s | NMDA-off decays to ~0 — the latch is the slow-NMDA recurrence, not the tonic bias |
| AC3-B READ-PATH lesion + matched SHAM | real `affect_out`=0 -> staircase range **0.000**; sham `decoy_out`=0 (equal-size unrelated projection) -> range **intact** (0.055-0.069) | real flips, sham does not; the lesion is on the projection GATE, not the read pool (non-tautological) |
| AC3-A RECURRENCE lesion + matched SHAM | real ladder NMDA-off -> persistence 0.00; sham unrelated recurrence off (`speak_acc`/`silence_acc` NMDA-off, equal neuron count) -> held **survives** (ret ~1.0) | tests specificity: the held value is the ladder's latch, not any recurrence |
| AC4 NEURAL READ | the graded quantity is `pos_readout` population rate; proven by the AC3-B `affect_out` lesion driving it to 0 | not a host argmax/index |
| AC5 HONESTY FLOOR (FM4) | abstain (no evidence) -> silence wins at EVERY arousal level incl. max, all seeds; assert (evidence) -> speak wins and speak-rate rises with arousal | graded arousal never flips abstain->assert |

Determinism: two builds at one `cfg.seed` are hash-identical on `cp_neuron_firing_thresholds`, differ across seeds
(`substrate_seeded=True`). Verdict carries a `tools.verdict.Verdict` preconditions block (all preconditions met).

## What this is NOT (honest boundary, not a caveat to hide)

- **A smooth continuum.** It is a QUANTIZED 6+1-level code. Selling it as a smooth Russell circumplex would be an
  overclaim. Koulakov: drift-robustness is bought with resolution.
- **A learned engram.** The ladder is a designed circuit; the dependence tested here is on the STAGGERED STRUCTURE
  (the unstaggered ladder — same neurons, same wiring, stagger removed — collapses to all-or-none, the structure
  control that must fail), not on training.
- **The single-cell gold standard.** Egorov I_CAN graded persistence (one cell, many stable levels) needs a `sim/`
  change (no Ca-activated cation current in Izhikevich); it is the deferred stronger surpass if the quantization
  proves insufficient for the circumplex.

## Reproduce

```
SIM_BACKEND=numpy python -u -m research.runners._affect_graded_ladder_derisk --smoke        # single-seed
SIM_BACKEND=numpy python -u -m research.runners._affect_graded_ladder_derisk --seeds 42 43 44 100 101 102   # 6-seed
```
