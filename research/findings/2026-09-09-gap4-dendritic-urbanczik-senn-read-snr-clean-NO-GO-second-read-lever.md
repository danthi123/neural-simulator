---
type: finding
status: boundary
claim_check: measured
date: 2026-09-09
mechanism: gap#4 few-spike READ regime — dendritic (Urbanczik-Senn) two-compartment read-SNR lever
lane: gap4
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_wkv_mouth_readout_snr_ensemble_dendritic_derisk.py
artifacts:
  - research/findings/raw/_wkv_mouth_readout_snr_ensemble/dendritic/aggregate_dendritic_decisive.json
  - research/findings/raw/_wkv_mouth_readout_snr_ensemble/dendritic/dendritic_sub_pop1_decisive.json
builds_on:
  - research/findings/2026-08-11-gap4-ALLIN-ARC-SUMMARY-a-spiking-deep-credit-WALL-was-a-hyperparameter-READ-BEFORE-RE-ATTACKING.md
---

# gap#4 few-spike read regime: the dendritic (Urbanczik-Senn) two-compartment read is a CLEAN NO-GO — teacher-load-bearing + provenance-clean, but recovers only 0.40 of the copied-weight oracle (bar 0.55/0.83); the SECOND read-SNR lever to fail, triggering the deep-research gate

## The test
The gap#4 arc summary root-caused the genuine residual to the PRODUCTION Izhikevich FEW-SPIKE READ REGIME (deep
credit is lr-invariant AND a perfect Wᵀ oracle also fails — the limit is read SNR, not the feedback type). The
first read-SNR lever (the ensemble / `--sub-pop`) was a proven STRUCTURAL NO-GO (common-mode noise cancels √P by
construction). Its pre-registered contingency — a DENDRITIC Urbanczik-Senn two-compartment read (basal prediction
+ apical spiking teacher, per-unit local error, no weight transport) — was built + committed on main (3c75d58c0);
this is its decisive 6-seed eval (`research/runners/_wkv_mouth_readout_snr_ensemble_dendritic_derisk.py`, cupy,
`--lever dendritic --coverage decisive`).

## Result — clean NO-GO (artifact `research/findings/raw/_wkv_mouth_readout_snr_ensemble/dendritic/aggregate_dendritic_decisive.json`)
- **VERDICT NO-GO**, `go_count = 0` of 6 seeds. The dendritic read's learned recovery `sub_learned_recov_mean =
  0.4011` sits far below the pre-registered bar (per-seed `sub_learned recov >= 0.85 × sub_copied OR >= 0.55` <!--derived-->;
  0.85 × 0.9754 = 0.829 <!--derived-->). The copied-weight ORACLE recovers `sub_copied_recov_mean = 0.9754` — so the
  read CAN carry the signal; the LEARNED dendritic rule captures only ~41% of it (`sub_recov_ratio_mean = 0.4113`),
  barely off the ~0.37 plateau. <!--derived-->
- **The mechanism is genuinely CLEAN, not a cheat** (this is what makes it an honest negative, not an instrument
  failure): freeze-apical collapses recovery to `0.0003` and shuffle-apical to `0.0011` (the effect REQUIRES the
  apical spiking teacher — `dendritic_anticheats_ok_count = 6/6`); `apical_reads_match_all = True` (the apical read
  ran every gradient step); `host_matmul_on_forward_max = 0` + `forward_is_substrate_all = True` (no host-linear
  shortcut); `verify_first_all_ok = True`. So the dendritic Urbanczik-Senn read is a real, transport-free,
  teacher-load-bearing mechanism — it simply does not lift the few-spike read SNR to the bar.

## What it means, and the no-defer NEXT
This is a METHOD verdict, not a capability closure. The `deep_research_at_wall` round (2026-09-09) surfaced the
external burst-multiplexing family (Payeur et al. 2020 Nat Neurosci; Greedy et al. 2022 BurstCCN; Stuck et al.
2024 Burstprop) — but reconciling against THIS PROJECT'S OWN record corrects the naive read of it, and the
correction is the important part:

**⛔ Burst-multiplexed / dendritic / BDSP deep-credit is ALREADY TESTED-NEGATIVE here — do NOT re-attack it.** The
project's own deep-research finding `2026-07-22-gap4-real-issue-NOT-dendrites` (which already cited Payeur 2021)
concluded: (1) the two-compartment dendrite is TOPOLOGICALLY FAITHFUL (Payeur/Sacramento/Urbanczik-Senn all use
one apical compartment) — so the read/topology is NOT the crux; (2) burst-multiplexing fidelity / graded credit /
population size are RANK 3-4 SMALL enhancements, not the fix; (3) BDSP credit-training already lost to a random
reservoir (clean negative, see also `2026-05-17-dendritic-credit-assignment-NEGATIVE`). THIS dendritic read NO-GO
(learned recovery 0.40 <!--derived--> while the COPIED-weight oracle recovers 0.975 <!--derived-->) is a fourth confirmation of the same thing:
the read CAN carry the signal, but the LEARNED rule cannot — because the apical carries a FROZEN fixed-random
projection of the raw error that never zeroes when the network is already correct (feedback-alignment signal),
not a true learned error.

**The genuinely-untested next levers are on the FEEDBACK-SIGNAL side (per 2026-07-22), NOT another read/burst
mechanism:** (1) RANK 1 — a **learned interneuron self-predicting microcircuit (Sacramento 2018)**: plastic
SST/PV `W^IP`/`W^PI` LEARN to cancel top-down feedback so the apical is SILENT when correct and carries a true
prediction error otherwise (the `enable_bdsp_microcircuit` stub exists but its cancellation is runner-supplied,
not learned in-engine); (2) RANK 2 — **learned feedback weights (PAL / weight-mirror)**, untested on a task with
informative credit. Both are no-weight-transport and target the CAUSE (the frozen credit signal) the read-side
levers cannot. Run `bash tools/before_you_build.sh` before building either. gap#4 is NOT the load-bearing blocker
on the live conversation (the working faculties use zero deep credit — arc summary Q_C). Not a phenomenal claim.

Sources / negatives cited: `2026-07-22-gap4-real-issue-NOT-dendrites` + `2026-05-17-dendritic-credit-assignment-NEGATIVE`
(dendritic/burst read-side deep-credit is tested-negative here; the crux is the feedback SIGNAL); external:
Payeur et al. 2020 (Nat Neurosci), Greedy et al. 2022 (BurstCCN), Stuck et al. 2024 (Burstprop) — burst-multiplexing
is RANK 3-4 small per 2026-07-22, not the crux; Sacramento 2018 (learned interneuron microcircuit, the RANK-1 fix).
