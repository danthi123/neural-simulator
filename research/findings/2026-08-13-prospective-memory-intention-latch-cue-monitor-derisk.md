---
type: finding
status: qualified
date: 2026-08-13
mechanism: pmem-intention-latch-cue-monitor
runner: research/runners/_pmem_intention_latch_derisk.py
artifacts:
  - research/findings/raw/_pmem_intention_latch.json
  - research/findings/raw/_pmem_intention_latch.json.prov.json
---

# Prospective memory (Tier-2) — a spiking INTENTION LATCH + BA10 CUE-MONITOR: mechanism de-risked, release-amplitude is the residual

**Verdict: QUALIFIED / BOUNDARY at the pre-registered 6-seed gate (3/6).** The scientifically hard, novel part is
bulletproof: the spiking latch + cue-monitor holds a deferred intention across intervening turns and releases it
ONLY on the right cue, and the release is destroyed by a latch-lesion — every specificity/persistence/lesion clause
passes 6/6. The single failing clause is the release AMPLITUDE against a FIXED absolute threshold (`fire_on_cue`,
3/6): under a CONSTANT tonic bias the release rate has ~4x per-seed spread and one seed's readout pool is too
hypo-excitable to fire. Named residual: per-readout-pool homeostatic gain/threshold normalization (the companion
process the constant bias proxies). CPU, 19 s, reuse-by-import, NO `sim/` edit.

Artifact: `research/findings/raw/_pmem_intention_latch.json` (provenance sidecar `.prov.json` beside it).

NO-EXTERNAL-NEEDED: this BOUNDARY is a verdict on a METHOD (a CONSTANT tonic-bias readout), not a capability wall
or a fundamental-limit claim — the mechanism is validated (every specificity/persistence/lesion clause 6/6) and the
residual is a NAMED, buildable biological surpass (per-`rel`-pool homeostatic excitability regulation). The
prospective-memory biology it realizes is already in-corpus: Burgess & Gilbert rostral-PFC/BA10 prospective-memory
cue-monitoring; Gollwitzer (1999) implementation intentions ("when X, I will do Y"); the reused persistent-attractor
WM is the project's own 6-seed-GO `SpikingLoopContextBuffer`. No fundamental limit is asserted, so no external
overturn risk of the class this gate guards.

## The missing rung (why this is not within-turn WM)

A genuine conversant remembers to do something LATER — "remind me to X when Y comes up." That is an intention held
ACROSS intervening turns that fires when its cue appears, and NOT before / NOT on a wrong cue. Distinct from
within-turn working memory (the item being manipulated now): the prospective intention must survive a stretch of
UNRELATED distractor turns, then be RELEASED by a specific external cue. Nobody in this repo owned it. Mechanism
(Burgess/Gilbert rostral-PFC/BA10 prospective memory; Gollwitzer implementation intentions "when X, I will do Y"):
(1) a PFC INTENTION LATCH — sustained recurrent attractor activity holding the deferred intention; (2) a BA10
CUE-MONITOR — a coincidence detector that releases the intention only when the current cue matches.

## Mechanism (all spiking; every read is `cp_firing_states`)

Reuse the VALIDATED persistent-attractor WM (`content_selection_spiking.SpikingLoopContextBuffer` /
`biased_competition_buffer`, 6-seed GO): a `cortex_ctx<->dlpfc_wm` loop with per-concept outer-product attractors
that self-sustain on Izhikevich spikes and hold a SET of >=3 concepts. The intention = a self-sustaining assembly
(the LATCH); distractors = other self-sustaining assemblies (real competing WM load). ADDED per action X a small
NMDA-recurrent accumulator region `rel_X` (the sel-pool idiom used as a coincidence detector) that receives
FEEDFORWARD from BOTH act_X.cortex (the held intention) AND cue_X.cortex (the cue). A tonic hyperpolarizing bias
holds `rel_X` sub-rheobase; the held intention supplies a subthreshold priming depolarization; only when the CUE
arrives on the primed pool does `rel_X` cross threshold and fire = the intention is RELEASED. This is a spiking AND,
computed by neurons/synapses.

## Pre-registered gate + results (6 seeds 42/43/44/100/101/102, N=5 intervening turns, 4 distractors)

Per-clause pass counts (the runner's own frozen thresholds: FIRE_THR=0.20, SILENT_MAX=0.06, HOLD_FLOOR=0.05,
LESION_HELD_MAX=0.02, SEP_RATIO=2.5, need 5/6 seeds):

Per-clause counts and the rounded/max per-seed reads below are all computed from the cited raw JSON.

<!--derived-->
| clause | pass | meaning |
|---|---|---|
| persistence | **6/6** | held assembly stays 0.32–0.35 through ALL 5 distractor turns (the latch survives) |
| no_fire_before | **6/6** | release stays <=0.048 on every intervening turn (silent until the cue) |
| no_fire_wrongcue | **6/6** | present the WRONG cue -> release <=0.047 (the monitor is cue-specific) |
| no_intention_silent | **6/6** | NO intention latched -> the cue ALONE cannot fire (<=0.031) — the coincidence is real, not a cue passthrough |
| lesion_holds | **6/6** | zero the latch -> held collapses 0.32–0.35 -> **0.000** at measurement (the lesion HOLDS) |
| lesion_forgets | **6/6** | after the lesion the cue does NOTHING (<=0.038) — the intention is FORGOTTEN |
| separation | 5/6 | fire beats max-silent by >=2.5x AND clears the silent ceiling (seed 100 fails: release ~0) |
| **fire_on_cue** | **3/6** | correct-cue release >= 0.20 absolute — seeds 44 (0.164), 100 (0.054), 102 (0.198) miss |

<!--derived-->
Per-seed correct-cue release (min of act_A/act_B) vs max-silent: 42 `0.267/0.038`, 43 `0.211/0.018`,
44 `0.164/0.048`, 100 `0.001/0.000`, 101 `0.217/0.019`, 102 `0.198/0.018`. **5/6 seeds show the intention firing
clearly above every silent condition** (3.4x–7x); seed 100 is a genuine release-amplitude failure (its `rel` pools
barely spike on the joint drive). The fixed FIRE_THR=0.20 additionally rejects seeds 44 and 102, whose separation is
actually fine — so the absolute threshold is stricter than the demonstrated separation warrants.

## The wall is a CONSTANT proxying a companion process (the project's own reframe)

The release readout is gated by a CONSTANT tonic bias. Biology runs a companion process alongside — per-pool
homeostatic excitability regulation (tonic-inhibition set-point / intrinsic plasticity to a firing-rate target) that
normalizes each readout pool's operating point. Replaced by a constant, the release amplitude drifts ~4x across
seeds: seed 42's pool is hyper-excitable (would fire on singles at a looser bias), seed 100's is hypo-excitable
(barely fires on the joint at the fixed bias). Two levers on this residual (bias sweep; rel recurrence) did not
close it — recurrence cannot amplify seed 100's near-absent coincidence spikes (a subtractive/threshold deficit,
not a gain deficit). **The single-variable surpass is a per-`rel`-pool homeostatic bias/threshold (intrinsic
plasticity to a rate set-point), so a fixed FIRE_THR is cleared on every seed** — not a re-tuned constant.

## Honest scope

- **Brain-based:** the latch (attractor persistence), the cue-monitoring (coincidence integration), and the release
  (accumulator crossing threshold) are all spiking; every read is `cp_firing_states`. The tonic bias is a constant
  stand-in for tonic inhibition (the flagged residual above).
- **HOST-SCAFFOLD, FLAGGED:** the cue->action CONTENT binding (WHICH cue releases WHICH action) is INSTALLED
  synaptically (outer-product edges), exactly as every `SpikingLoopContextBuffer` attractor is. The mechanism
  (hold-across-turns + cue-gated release) is brain-based; the content binding's LEARNED version — one-shot Hebbian
  potentiation of cue->action at intention-formation (Gollwitzer implementation-intention) — is the named follow-on.
- **Instrument verified:** the lesion HOLDS (held -> 0.000 at measurement, per `docs/TERMS.md` lesion condition);
  the release is gated by the held intention (no_intention + lesion_forgets both 6/6 -> not a cue passthrough); the
  latch persistence is genuine (act_X is NOT re-driven on intervening turns — only distractors are written). An
  `attributable_to` call (tools.lab) quantifies it: **95.7%** <!--derived--> of the correct-cue release is owned by
  the intact latch (only 4.3% leaks through the cue path after the lesion). The verdict travels with a `Verdict`
  preconditions block (persistence / silence-controls / lesion-reaches / release-separation all measured and hold).

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._pmem_intention_latch_derisk --smoke    # 1 seed, N=3, ~4 s
SIM_BACKEND=numpy python -m research.runners._pmem_intention_latch_derisk --derisk   # 6 seeds, N=5, ~19 s
```

## Next single-variable de-risk

Add a homeostatic set-point to each `rel` pool (a self-inhibitory FS partner giving divisive normalization, or
intrinsic-plasticity threshold adaptation to a target baseline rate) and re-run the frozen gate; expected to lift
seeds 44/100/102 over FIRE_THR without touching the 6/6 silence, converting BOUNDARY -> GO. Separately, neuralize
the cue->action binding via one-shot Hebbian encoding at intention-formation (retire the installed edges).
