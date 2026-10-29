---
type: finding
status: qualified
lane: gap#4
date: 2026-10-29
---
# C2 step 0b/0c/0d: local inhibition among the patient word pools (dev seed 7) -- feedback and flagged-pool feedforward NO-GO; feedforward with an external-only divisive stage selects one correct pool in 9/9 blocks

Status: DRAFT, dev seed 7 only (never a gate seed), research-only. Nothing here is a GO; Claude rules (the 9/9 is one seed, one
run, no six-seed ablation).
Artifacts: `research/findings/raw/_c2_step0b/s7_off.json`, `research/findings/raw/_c2_step0b/s7_all_v2.json` (0b, with the
conductance leak, superseded), `research/findings/raw/_c2_step0b/s7_all_v3.json` (0b, leak fixed),
`research/findings/raw/_c2_step0b/s7_ff.json` (0c, leak), `research/findings/raw/_c2_step0b/s7_ff_v2.json` (0c, leak fixed),
`research/findings/raw/_c2_step0b/s7_ffx.json` (0d).
Runner: `research/runners/c2_hops_step0.py` (`--smoke-0b OUT --patient-inhib {off,on,lesion,all,ff,ff_lesion,ffx,ffx_lesion}`),
tests `tests/test_c2_step0b_patient_inhib.py`, `tests/test_divnorm_external_only.py`. One `sim/` edit (0d), default OFF.

## Defect and lever count

Step 0 read (c) (STOP S2_patient_pool): relaying each block's patient cleanup row through `onbridge_divnorm_drive` on the
composer's score bridge lights 3-5 patient pools per block (seed 7, 9 blocks), though the decoded patient leads ~2.5x.
Divisive normalization divides; nothing makes the pools compete. Levers tried against this one defect: **4** -- stable
integration (cells), feedback E->I->E (0b), feedforward FS inhibition (0c), and the external-only divisive stage that lets the
feedforward inhibition act (0d). The 0c diagnosis, not a new mechanism, is what turned lever 3 into lever 4.

## Instrument bug found during 0d (applies to 0b and 0c; fixed)

`_reset_score_bridge` resets v/u/firing only. The inhibition arms reuse one bridge across the 9 blocks, so inhibitory
conductance leaked from block to block (g_i of hundreds of nS carried over). `onbridge_inhib_drive` now zeroes every
`cp_conductance_*` array and `cp_prev_firing_states` per query. The 0b and 0c shards were re-run after the fix (`_v3`, `_v2`).
The leak explains the zero-pool blocks and part of the "inverted gain" in the first 0b table (below); the 0b/0c conclusions
survive the fix.

## 0b: feedback inhibition (E->I->E), NO-GO

One shared FS population `pinh` (30 cells, FS preset re-applied after the global heterogeneity overwrote it with RS-centred
jitter, FS cells started at rest), word pools -> `pinh` (60 pA), `pinh` -> word pools (inhibitory by `exc_fraction=0`).
Lesion arm = same population, I->E zeroed. With the leak (first run, `_v2`): on 0/9 correct (4 multi, 4 zero, 1 wrong).
Leak fixed (`_v3`): lesion 0/9 (9 multi); on 0/9 (I->E 320: 8 multi, 1 block fires 122 pools; I->E 40 and 100: identical to
lesion). So at physiological-ish gains the feedback inhibition does nothing, at 320 nS-scale it explodes in one block.
Reasons: late recruitment (pooled FS cells first fire at step ~8, the leading pool at step 3), and the divisive stage (below).

## 0c: feedforward inhibition (Swadlow, 2002; Mallet et al., 2005)

Record: thalamocortical FS cells take the same input as the principal cells, pool it broadly, fire ~4.6 ms earlier at lower
current (Swadlow, 2002, Cereb Cortex; Mallet et al., 2005, J Neurosci), and the synchronous volley leaves a brief excitability
window in which only the strongest-driven cell fires. Our precedent: `2026-06-10-N9-spiking-fs-critic-inhibition-GO.md`.

Build: `pinh` gets ff_gain (3e-4) x the summed patient-slice drive as external current (fixed per-bridge gain, no
normalization/argmax/per-query rescale), keeps E->`pinh` and `pinh` -> word synapses (I->E 40), on a stable-integration
bridge (`enable_bounded_conductance_integration=True`, `izh_membrane_substeps=4`, `"model_reset"`) for all arms
(`stable_off`, `ff`, `ff_lesion`).

Result (leak fixed, `_v2`): `pinh` fires first in 9/9 blocks (step 0 vs the earliest pool at step 3-4); one-correct-pool
blocks `ff` 0/9, `ff_lesion` 0/9, `stable_off` 0/9, FF-gain sweep (1e-4, 1e-3, 3e-3 first run; 1e-3 re-run) 0/9: the
inhibition is engaged and not load-bearing on the flagged pools (fired-pool drop attributable to it: -0.026
<!--derived-->, i.e. null). Max inhibitory conductance 601 nS at I->E 40, max |v| 80 mV (the spike apex).

### Root cause (accepted in the c2 ruling): the divisive block divides the inhibition too

`sim/bridge.py` divides `synaptic + external` current of every flagged neuron by `sigma + gain*mean(total input over the
flagged set)`. Here D = 3515: winner ~685 pA, runner-up ~270 after division, and an inhibitory synaptic current is divided by
the same D (~1/3500 of nominal). Diagnostic with a host divisor (not a deliverable) suppressed strongly, which is what 0d
does on-substrate.

### Relay variant (inhibition after the divisive stage, runner-side), partial

Unflagged relay pool `r{w}` per word (300 pA from the word pool), `pinh` (same FF drive) inhibits the relay (I->E 5); the
relay is read. Leak fixed (`_v2`): `ffr` 7/9 blocks exactly one pool, 2 multi (blocks 0 and 6); `ffr_lesion` 0/9; mean fired
pools drop 65.6% attributable to the inhibition <!--derived-->. Without the fix it was 0/9 (2-3 pools), so the leak had masked it.
It adds a layer; 0d achieves the same without one.

## 0d: feedforward inhibition on an external-only divisive stage

`sim/` edit, additive, default OFF: `CoreSimConfig.input_divisive_norm_external_only` (False). When True the divisive stage
divides only `cp_external_input_current` (the divisor's mean is over the flagged set's external drive) and synaptic current
is added undivided (numpy path; no kernel/cupy path uses the mask, the fused path is already disabled when the mask exists).
`tests/test_divnorm_external_only.py`: flag default False; flag-off run (divnorm ON, with inhibitory synaptic input engaged,
g_i up to 77 nS) hashes to the golden computed from the PRE-edit `sim/` (a `git archive HEAD sim` copy run before the edit);
flag on changes the trajectory and an undivided inhibitory conductance hyperpolarizes the flagged pools.

Arms (all stable integration, flag ON, word pools read directly): `ffx_off` (population built, no FF drive, I->E 0), `ffx`
(FF gain 3e-4, I->E 2.0 nS-scale weight), `ffx_lesion` (FF drive on, I->E 0), plus an I->E sweep 1/2/3/5.

| arm | one correct pool (of 9) | multi | zero | max g_inh (nS) | pinh first in 9/9 |
|---|---|---|---|---|---|
| ffx_off | 0 | 9 | 0 | 0 | no (step 7-9, feedback only) |
| ffx_lesion | 0 | 9 | 0 | 0 | yes |
| ffx (I->E 2) | 9 | 0 | 0 | 30 | yes |
| ffx@1 | 8 | 1 (block 6: worm, gnat) | 0 | 15 | yes |
| ffx@3 | 9 | 0 | 0 | 45 | yes |
| ffx@5 | 7 | 0 | 2 (blocks 0, 2) | 75 (over 50) | yes |

At every one-correct block the fired pool is the decoded patient. Max |v| 79-80 mV (spike apex). The earliest pool first
spike is at step 3-6 (step 7-10 for the survivor at I->E 5) against `pinh` at step 0. Fired-pool drop attributable to the
inhibition 0.757 <!--derived-->. Lesion != on, on != off; engagement 300 inhibitory spikes per block.

### Cliff / margin (sandbox, synthetic block-0 shape, I->E 2, FF 3e-4; seed 7 only)

Selection is graded, then all-or-none. Two driven pools, winner fixed: the runner-up is silenced up to a runner/winner drive
ratio of 0.30 and fires from 0.39 (the block-0 ratio is 0.39: 2.5x), reaching parity at 0.9. The winner is lost (zero pools)
at I->E >= 8 on the block-0 shape (0.2 spikes at 5, 0 at 8), and on the real 9 blocks at I->E 5 two blocks lose the winner
(g_inh 75 nS, also outside the physiological band). So the working window is I->E 1-3 (15-45 nS) with a runner-up/winner
drive ratio below ~0.4; below I->E 1 the lesion regime returns.

## STOP lines

- S0D lesion == on: not fired. S0D not-exactly-one-correct-pool: not fired for `ffx` (I->E 2) and `ffx@3`; fires for `ffx@1`
  (one block with two pools) and `ffx@5` (two zero-pool blocks, g_inh over 50 nS).
- S0B (feedback) not exactly one correct: fires. S0C (flagged-pool FF): fires.

## What this retires and what is open

Retires: feedback-only E->I->E onto flagged pools; FF onto flagged pools while the divisive block divides synaptic input.
Open for ruling: six-seed/dev-seed replication (1-6) of 0d and its sensitivity to the divisive gain and FF gain; the 0.4 drive
ratio bound against the real runner-up distribution; whether the external-only flag is wired into the composer's real score
bridge (the 0d bridge is a runner-built twin, word-pool parameters copied from the composer's).
