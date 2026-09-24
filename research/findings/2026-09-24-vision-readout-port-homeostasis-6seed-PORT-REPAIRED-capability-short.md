---
type: finding
status: live
claim_check: measured
date: 2026-09-24
lane: D · Perception
mechanism: readout-port homeostasis (Triesch intrinsic plasticity + multiplicative synaptic scaling on each class-population
  LIF unit of the attention-gated soft feedback-gain read; learned on train trials only, frozen for test; default OFF)
seeds: [42, 43, 44, 100, 101, 102]
prereg: research/findings/2026-09-24-vision-readout-port-homeostasis-intrinsic-plasticity-PREREGISTERED.md
artifacts:
  - research/findings/raw/lanes/perception/conjbind_fbgain_gainonly_PORTIP_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s42.json
  - research/findings/raw/lanes/perception/conjbind_fbgain_gainonly_PORTIPlesion_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s42.json
  - research/findings/raw/lanes/perception/_readout_port_homeostasis_run_commands_12threads.txt
  - research/findings/raw/lanes/perception/conjbind_fbgain_gainonly_PORTIP_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s43.json
  - research/findings/raw/lanes/perception/conjbind_fbgain_gainonly_PORTIP_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s44.json
  - research/findings/raw/lanes/perception/conjbind_fbgain_gainonly_PORTIP_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s100.json
  - research/findings/raw/lanes/perception/conjbind_fbgain_gainonly_PORTIP_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s101.json
  - research/findings/raw/lanes/perception/conjbind_fbgain_gainonly_PORTIP_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s102.json
  - research/findings/raw/lanes/perception/conjbind_fbgain_gainonly_PORTIPlesion_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s43.json
  - research/findings/raw/lanes/perception/conjbind_fbgain_gainonly_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s43.json
  - research/findings/raw/lanes/perception/_superseded_1thread/conjbind_fbgain_gainonly_PORTIPlesion_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s43.json
  - research/findings/raw/_scratch_threadcheck/s43_lesion_th12.json
  - research/findings/raw/_scratch_threadcheck/s43_lesion_th4.json
verdict: Band 5, PORT-REPAIRED / CAPABILITY SHORT, the band predicted before any evaluation seed. G0-G5 all pass on the
  pre-registered counts (G0 and G1 6/6, G2 and G3 paired t = 8.55, G4 6/6, G5 6/6). capability_go 3/6 (below the 5/6 GO bar).
---

# Vision readout-port homeostasis: the collapsed port is repaired on 6/6 seeds; the capability stays short (3/6)

## Result

Per-seed artifacts: `research/findings/raw/lanes/perception/conjbind_fbgain_gainonly_PORTIP_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s42.json`
(HOMEO) and `..._PORTIPlesion_...` (LESION), and the same names for seeds 43, 44, 100, 101, 102, at revision `fde276684`.
The NEUTRAL arm is the pinned artifact set at `9f4b35138`, as registered. Every arm ran with 12 math-library threads (exact
commands: `research/findings/raw/lanes/perception/_readout_port_homeostasis_run_commands_12threads.txt`).

(Values from `by_code.count.per_seed[0]`, rounded to 4 decimals.)

<!--derived-->
| seed | HOMEO held | NEUTRAL held | lesion_permuted held | RANDOM held | label-shuffle null | capability_go |
|---|---|---|---|---|---|---|
| 42 | 0.5625 | 0.25 | 0.25 | 0.3229 | 0.3542 | yes |
| 43 | 0.625 | 0.25 | 0.25 | 0.2917 | 0.2604 | yes |
| 44 | 0.5729 | 0.25 | 0.25 | 0.2292 | 0.3333 | no |
| 100 | 0.4375 | 0.25 | 0.25 | 0.2708 | 0.2917 | no |
| 101 | 0.5729 | 0.25 | 0.25 | 0.3125 | 0.1979 | yes |
| 102 | 0.4271 | 0.25 | 0.25 | 0.2292 | 0.2292 | no |

- G0 integrity (LESION == NEUTRAL on decode/reframe/dissociation/verdicts, learned state exactly 0): 6/6.
- G1 train output not constant (2.0 bits of prediction entropy, trial variance > 0): 6/6.
- G2 held-out above NEUTRAL and G3 specific learned state (the permuted-state lesion returns to 0.25 everywhere): both paired
  t = 8.55 <!--derived-->, all six lesion_permuted reads <= 0.40.
- G4 scramble null <= 0.40: 6/6. G5 fair nulls (RANDOM and label-shuffle, each regulated by its own homeostasis) <= 0.40: 6/6.

## Capability residual (named by band 5)

capability_go fails on seeds 44, 100 and 102. On seed 44 object position is still decodable from the regulated class code
(`position_decode_heldsplit` 0.5833, above the 0.40 bar). On seeds 100 and 102 held-out accuracy (0.4375, 0.4271) sits under
the no-go floor, so `beats_config_c_nogo` is false. The next lever is position invariance of the regulated port code, likely a
competitive or lateral process across the class populations.

## The thread-count confound G0 caught

The first run of all 12 arms used 1 math thread and failed G0 on seed 43: its LESION arm read a linear decode of 0.625 where the
pinned NEUTRAL artifact reads 0.6458. Re-running that arm with 12 threads (the pool's default when NEUTRAL was produced) reproduced
0.6458 exactly; 1 and 4 threads both gave 0.625 (`research/findings/raw/_scratch_threadcheck/`). The 1-thread set is kept under
`research/findings/raw/lanes/perception/_superseded_1thread/` and is not scored. Provenance now records thread counts (FAILURE_LOG).
The registered compute plan put the runs on the pool; they ran locally at the same revision and backend.

## Honesty

Functional read-outs only.
