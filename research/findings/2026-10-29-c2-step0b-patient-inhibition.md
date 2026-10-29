---
type: finding
status: qualified
lane: gap#4
date: 2026-10-29
---
# C2 step 0b/0c: local inhibition among the patient word pools (dev seed 7) -- feedback NO-GO, feedforward NO-GO on the flagged pools, relay variant partial

Status: DRAFT, dev seed 7 only (never a gate seed), research-only, default OFF. No GO is claimed; Claude rules.
Artifacts: `research/findings/raw/_c2_step0b/s7_ff.json`, `research/findings/raw/_c2_step0b/s7_all_v2.json`, `research/findings/raw/_c2_step0b/s7_off.json`.
Runner: `research/runners/c2_hops_step0.py` (`--smoke-0b OUT --patient-inhib {off,on,lesion,all,ff,ff_lesion}`), tests
`tests/test_c2_step0b_patient_inhib.py`, shards `research/findings/raw/_c2_step0b/`. No `sim/` edit.

## Defect and lever count

Step 0 read (c) (STOP S2_patient_pool): relaying each block's patient cleanup row through `onbridge_divnorm_drive` on the
composer's score bridge lights 3-5 patient pools per block (seed 7, 9 blocks), though the decoded patient leads ~2.5x.
Divisive normalization divides; nothing makes the pools compete. Levers tried against this one defect: **3** (stable cells,
feedback E->I->E, feedforward FS inhibition), plus one variant of the third (a relay layer, below). The research gate
applies at two unresolved levers; the third was taken on the record's named companion process.

## 0b: feedback inhibition (E->I->E), NO-GO

One shared FS population `pinh` (30 cells, IZH2007_FS preset re-applied after the global heterogeneity overwrote it with
RS-centred jitter), word pools -> `pinh` (60 pA), `pinh` -> word pools (inhibitory by `exc_fraction=0`), word-pool
parameters copied from the composer's own score bridge. Lesion arm = same population, I->E zeroed. Blocks with exactly one
pool firing that equals the decoded patient: off 0/9, on 0/9 (4 multi, 4 zero, 1 wrong), lesion 0/9.

| block | decoded | off | on (I->E 320) | lesion |
|---|---|---|---|---|
| 0 | spikes | 5 | 5 | 5 |
| 1 | words | 3 | 0 | 3 |
| 2 | memory | 5 | 144 | 5 |
| 3 | cat | 4 | 0 | 4 |
| 4 | fish | 3 | 0 | 3 |
| 5 | frog | 4 | 6 | 4 |
| 6 | gnat | 3 | 1 (wrong) | 3 |
| 7 | wren | 5 | 8 | 5 |
| 8 | moth | 4 | 0 | 4 |

Gain sweep (I->E 10/40/100/160): 0/9 correct at every gain; 10 = lesion; 40 and 100 add pools; 160 fires all 144 pools in two
blocks. Inverted gain: more inhibition, more firing. Two reasons were measured: (i) late recruitment, the pooled FS cells
first fire at step ~8 while the leading pool fires at step 3; (ii) Euler overshoot/rebound, inhibitory conductance 100-500 nS
on ~100 pF cells. A third reason was found in 0c (below), and it is the main one.

## 0c: feedforward inhibition (Swadlow, 2002; Mallet et al., 2005)

Record: thalamocortical FS cells take the same input as the principal cells, pool it broadly, fire ~4.6 ms earlier at lower
current (Swadlow, 2002, Cereb Cortex; Mallet et al., 2005, J Neurosci), and the synchronous volley leaves a brief excitability
window in which only the strongest-driven cell fires. Our precedent: `2026-06-10-N9-spiking-fs-critic-inhibition-GO.md`
(`place_fs` -> striosome feedforward inhibition).

Build: the FS pool gets ff_gain x the summed patient-slice drive as external current (fixed per-bridge gain, default 3e-4,
no normalization/argmax/per-query rescale), keeps the E->`pinh` feedback synapses and the `pinh` -> word synapses (I->E 40).
All 0c arms are built with the stable-integration set (`enable_bounded_conductance_integration=True`,
`izh_membrane_substeps=4`, `izh_substep_refractory_rule="model_reset"`) so they differ only in the FF inhibition:
`stable_off` (population built, no FF drive, I->E 0), `ff`, `ff_lesion` (FF drive on, I->E 0). One sweep of three FF gains.

Results, seed 7, 9 blocks:
- ENGAGEMENT and FS-first hold. `pinh` first spikes at step 0 (step 1 at ff_gain 1e-4) in all 9 blocks; the earliest word
  pool first spikes at step 3-4 (`ff_pinh_first_in_all_blocks` True). 2700 inhibitory spikes per arm.
- `ff` is not load-bearing on the flagged pools. Blocks with one correct pool: stable_off 0/9, ff 0/9, ff_lesion 0/9, and
  every swept gain (1e-4, 1e-3, 3e-3) 0/9, all multi-pool. `ff` vs `ff_lesion` differ only by +1 pool in block 1
  (fired-pool drop attributable to the inhibition: -0.026, i.e. null; <!--derived-->).
- Physical validity: no overshoot in the stable arms, but max |v| 80 mV is the spike apex (vpeak) and max inhibitory
  conductance is 695 nS at I->E 40 (87 nS in the relay arm at I->E 5), so the I->E gain is still unphysiological on 100 pF.

### Why the flagged pools do not feel the inhibition (a substrate fact, measured)

`sim/bridge.py` divisive-norm block (around line 9915) divides `total_input_current_pA = synaptic + external` for every
flagged neuron by `sigma + gain*mean(total input over the flagged set)`. Here that divisor is D = 3515 (sigma 1, gain 0.11,
144 pools): the winner's post-division current is ~685 pA, the runner-up's ~270. An inhibitory synaptic current onto a flagged
pool is divided by the same D, so it is ~1/3500 of its nominal size; making it matter needs conductances 3500x physical,
which then overshoot (the 0b inverted gain; a negative pool mean can even flip the sign). Diagnostic only (a host-side
divisor, NOT a deliverable): with the divnorm block disabled and the drive pre-divided by D, the same FF inhibition
suppresses strongly (I->E 10: 1.05/0.40/0.20 vs 2.15/1.05/0.95 spikes per cell; FF gain 3e-4: all pools silent), so the
FS volley itself is sound; the interaction with divnorm is what blocks it.

### Relay variant (inhibition delivered after the divnorm stage), partial

Runner-side, no `sim/` edit: each flagged word pool also projects (300 pA) onto an UNFLAGGED relay pool `r{w}`; `pinh`
(same FF drive) inhibits the relay pools (I->E 5); the relay is read. Arms `ffr`, `ffr_lesion` (stable integration).
- The decoded patient fires in the relay in all 9 blocks (ffr fired pools: 2,2,2,3,3,3,2,3,2 vs ffr_lesion 5,7,7,7,6,6,5,5,8;
  mean fired pools drop 60.7% attributable to the inhibition). `ffr` != `ffr_lesion` in all blocks.
- Still 0/9 blocks with exactly one pool: every block keeps the decoded patient plus 1-2 extras. Sandbox (synthetic top-6 row)
  shows inhibition scales all relay rates together (rates 1.25/0.85/0.20 at I->E 5; 0.75/0.55/0.10 at 7), then everything
  vanishes by 10-40: a rate-shifting subtraction with a cliff, not winner-take-all, under the any-spike readout.

## STOP lines (registered in the runner)

- S0B/S0C lesion == on: not fired for `on`, `ff`, `ffr` (they differ from their lesions), but the `ff` difference is null.
- S0B/S0C/S0C_relay on_not_one_correct_pool: FIRED for `on`, `ff` (all swept gains) and `ffr` (0/9 exactly-one-correct).
- Zero-pool block: `on` 4 blocks (over-suppression, 0b); none in `ff`/`ffr`.

## What this retires and what is open

Retires: feedback-only E->I->E onto flagged word pools; FF inhibition onto flagged word pools while the divnorm block
divides synaptic input. Open (for Claude's ruling, none built): (a) a sim-level option that applies the divisive stage to the
excitatory/external drive only and leaves synaptic inhibition undivided (additive-default-OFF flag with a byte-identity
test), which is the diagnostic's configuration done on-substrate; (b) read the relay with a rate-sensitive (not any-spike)
criterion, e.g. a second FS stage or a thresholded relay, so the 2-3 pools that survive are separated by rate.
