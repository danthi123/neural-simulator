---
type: finding
status: contributing
date: 2026-09-24
mechanism: gap4-transport-ceiling-readout-instrument (research/runners/_gap4_transport_ceiling_readout_derisk.py,
  Gap4ReadoutNet over Gap4InEngineNet / OnBridgeBDSPNet)
lane: F · gap#4 deep credit (plan step S19)
seeds: [7]
seed-waiver: dev-seed instrument calibration under the pre-registration; no evaluation seed was run and no
  generalisation is claimed. The prereg forbids queueing seed 42 until a dev config qualifies, and none did.
verdict: UNDEFINED at dev (the pre-registered fallback). No config in C1-C24 qualified under the selection rule, so
  the transport ceiling is still not interpretable and seed 42 was NOT queued. The dev run at the best
  non-qualifying config (C21, 4 arms x 3 replicates) cleared chance on 0 of 3 replicates, with n_fa_wall 1 of 3. The
  calibration found a likely explanation for the 2026-09-15 ceiling at chance: at the legacy config the feedforward
  pathway carried no detectable input, and it took two changes together (feedforward STP bypassed AND a 10x
  feedforward gain) to make it transmit. Four read-regime defects are measured, each with a flag-level fix in the
  runner. One residual is measured but not yet fixed. Revised after an independent review (prereg AMENDMENT 5).
runner: research/runners/_gap4_transport_ceiling_readout_derisk.py
prereg: research/findings/2026-09-24-gap4-transport-ceiling-readout-lever-PREREGISTRATION.md
artifacts:
  - research/findings/raw/gap4/transport_ceiling_readout/round1_rev9654a99/diag_eread_monotonic_s7.json
  - research/findings/raw/gap4/transport_ceiling_readout/round2_rev5c3a865/diag_transmit_scan_stp_ts0.0_s7.json
  - research/findings/raw/gap4/transport_ceiling_readout/round2_rev5c3a865/diag_transmit_scan_stp_ts0.5_s7.json
  - research/findings/raw/gap4/transport_ceiling_readout/round2_rev5c3a865/diag_transmit_scan_stp_ts1.0_s7.json
  - research/findings/raw/gap4/transport_ceiling_readout/round2_rev5c3a865/identity_selftest_s7.json
  - research/findings/raw/gap4/transport_ceiling_readout/round2_rev5c3a865/calib_C9_s7_ckpt/s7_r0_frozen.json
  - research/findings/raw/gap4/transport_ceiling_readout/round4_rev8f16994/calib_C15_s7_ckpt/s7_r0_frozen.json
  - research/findings/raw/gap4/transport_ceiling_readout/round4_rev8f16994/calib_C15_s7_ckpt/s7_r0_transport_ceiling.json
  - research/findings/raw/gap4/transport_ceiling_readout/round4_rev8f16994/diag_oracle_online_budget_s7.json
  - research/findings/raw/gap4/transport_ceiling_readout/round4_rev8f16994/diag_fullsize_ff_transmission_s7.json
  - research/findings/raw/gap4/transport_ceiling_readout/round5_rev1ec6635/calib_C21_s7.json
  - research/findings/raw/gap4/transport_ceiling_readout/round5_rev1ec6635/calib_C24_s7.json
  - research/findings/raw/gap4/transport_ceiling_readout/calib_selection_all_rounds_s7.json
  - research/findings/raw/gap4/transport_ceiling_readout/dev_rev1ec6635/dev_s7.json
  - research/findings/raw/gap4/transport_ceiling_readout/dev_rev1ec6635/fullsize_timing_s7.json
  - research/findings/raw/gap4/transport_ceiling_readout/fix2_rev23d98e7/diag_tx_noise_full_legacy_ts1.0_s7.json
  - research/findings/raw/gap4/transport_ceiling_readout/fix2_rev23d98e7/diag_tx_noise_full_stp_bypass_ts1.0_s7.json
  - research/findings/raw/gap4/transport_ceiling_readout/fix2_rev23d98e7/diag_tx_noise_full_gain_ts1.0_s7.json
  - research/findings/raw/gap4/transport_ceiling_readout/fix2_rev23d98e7/diag_tx_noise_full_stp_bypass_gain_ts1.0_s7.json
  - research/findings/raw/gap4/transport_ceiling_readout/fix2_rev23d98e7/diag_tx_noise_dev_stp_bypass_gain_ts1.0_s7.json
  - research/findings/raw/gap4/transport_ceiling_readout/fix2_rev23d98e7/lesion_selftest_s7.json
  - research/findings/raw/gap4/transport_ceiling_readout/fix2_rev23d98e7/guard_selftest_s7.json
  - research/findings/raw/gap4/transport_ceiling_readout/fix2_rev23d98e7/calib_selection_all_rounds_tiebreak_fixed_s7.json
external: Bellec et al. (2020) Nat Commun 11:3625; Mazurek et al. (2026) Front Neurosci; Carandini & Heeger (2012)
  Nat Rev Neurosci 13:51 (cited in the pre-registration, recorded with tools/record_external_search.sh).
builds_on:
  - research/findings/2026-09-15-gap4-inengine-selfpredict-interneuron-UNDEFINED-transport-ceiling-foreclosed.md
---

# gap#4 transport ceiling: instrument calibration on dev seed 7 — UNDEFINED, four read-regime defects found

**Headline.** The plan asked for a longer or stronger readout so the copied-weight transport ceiling could clear chance.
No config did, on dev seed 7, so the pre-registered fallback applies: UNDEFINED at dev, and seed 42 was not queued. The
calibration found the most likely reason the 2026-09-15 ceiling sat at chance: at the legacy config (the 2026-09-15
net), cutting the input weights changed the first hidden layer's reads by no more than run-to-run noise, so the class
could not reach the output read in any arm. Neither of two changes restores transmission alone. Together they do on
the dev net, and at full size by one of the two pre-registered statistics (item 1).

## What was measured (dev seed 7; small net H32/pool 4 unless stated)

<!--derived-->
1. **At the legacy config the feedforward pathway carried no detectable input, and the cause has two factors.**
   Statistic (AMENDMENT 5 G, `_gap4_tc_transmission_noise_ref_diag.py`): H1 effect/noise = mean per-unit |intact -
   input-cut| over mean per-unit |intact - intact|, plus the across-item reliability of each unit's read (about 0 when
   the item-to-item variation is noise). Full size (H64/pool 16) at the 2026-09-15 operating point (tonic 1.0), spike
   read / legacy event read
   (`research/findings/raw/gap4/transport_ceiling_readout/fix2_rev23d98e7/diag_tx_noise_full_legacy_ts1.0_s7.json`,
   `research/findings/raw/gap4/transport_ceiling_readout/fix2_rev23d98e7/diag_tx_noise_full_stp_bypass_ts1.0_s7.json`,
   `research/findings/raw/gap4/transport_ceiling_readout/fix2_rev23d98e7/diag_tx_noise_full_gain_ts1.0_s7.json`,
   `research/findings/raw/gap4/transport_ceiling_readout/fix2_rev23d98e7/diag_tx_noise_full_stp_bypass_gain_ts1.0_s7.json`):

   | variant | H1 effect/noise | H1 item reliability |
   |---|---|---|
   | legacy (STP on, ff 4, propagation 0.05) | 0.91 / 0.95 | -0.02 / -0.03 |
   | STP bypassed on the feedforward synapses, legacy gain | 0.94 / 0.98 | 0.01 / -0.04 |
   | gain only (STP on, ff 40, propagation 0.5) | 0.94 / 1.00 | -0.02 / -0.04 |
   | STP bypassed + gain | 1.99 / 2.41 | 0.77 / 0.34 |

   So neither factor alone gives any detectable transmission. The pre-registered two-factor criterion (spike read:
   effect/noise above 2 AND reliability above 0.6) holds at dev size (H32/pool 4: 2.29 and 0.82 at tonic 1.0; 2.55 and
   0.83 at tonic 0.5). At full size the combined variant clears the reliability bar (0.77) and misses the effect/noise
   bar by 0.01 (1.99). The full-size two-factor record is therefore supported but not claimed under that rule. The
   mean H1 read barely moves even where the item reliability shows transmission (full size, both factors: reliability
   0.77, mean 0.0917 to 0.0900), which is why the earlier probe's mean statistic could not discriminate. That probe
   (`research/findings/raw/gap4/transport_ceiling_readout/round4_rev8f16994/diag_fullsize_ff_transmission_s7.json`)
   read the same in both of its variants: legacy 0.0548 to
   0.0547 (per-unit change 0.0093) and STP bypassed at the legacy gain 0.0547 to 0.0547 (0.0091). The round-2 dev scan
   (`diag_transmit_scan_stp_ts*_s7.json`) agrees: it switched STP off for ALL synapses, not only the feedforward ones,
   and at tonic 1.0 and legacy gain that left H1 input dependence at 0.159 against 0.160. Only STP off plus ff 40 / ps
   0.5 moved it (0.429). The test `tests/test_gap4_tc_feedforward_transmission.py` pins all four directions at tonic 0
   and at tonic 1.0.
2. **The event read is non-monotonic in drive** (`diag_eread_monotonic_s7.json`). Extra output current from 0 to
   +1600 pA raises the total spike rate from 86 to 404 Hz but lowers the BDSP event rate `E` from 0.050 to 0.002. The
   default output layer sits at the peak, so strengthening an output neuron lowers its read. Round 1 (event read)
   trained every arm to BELOW-chance training accuracy. Fix: `--read-quantity spikes`.
3. **The burst-probability baseline cancels each teaching transient.** Training chance is the majority-class rate of
   the 400 training items (class 8 has none): 0.1825 on replicate 0, which every calibration config used (AMENDMENT 5
   A; the earlier "1/9" reference was wrong). With the EMA baseline (alpha 0.05/step) the frozen readout's training
   accuracy is BELOW that chance at the transmitting operating point (C12, lr 5: 0.115, binomial p 1.5e-4 for below;
   C18, 30 epochs: 0.142, p 0.02). With the preset baseline (`--pbar-alpha 0`) it fits the training set to 0.265 (C15,
   p 2.9e-5 for above chance).
4. **Synapse elimination, not learning, made up most of the reported weight movement.** With elimination off (C9),
   the frozen arm's change in feedforward L1 norm (`ff_weight_moved`, |L1 end - L1 start|) is 2.1, against about 1270
   with it on. Default-on elimination makes every
   weight below 0.05 eligible for zeroing (5e-7 per step), and that includes every negative signed feedforward weight.
5. **Residual, not fixed: the transport ceiling's hidden learning collapses the output.** At the best configs the
   ceiling predicts one class for 360-395 of 400 training items (C15, C17). At every preset-baseline config (C15,
   C17, C19-C24) its training accuracy (0.150-0.190) is at or below the training chance of 0.1825 and below the frozen
   readout's (0.242-0.273). (At the two EMA-baseline configs the frozen readout itself stays low: C16 0.180, C18
   0.142.) The collapse is not specific to the ceiling: in the dev run fixed_fa (286 and 324 of 400 on class 0) and
   micro_inengine (237 and 244 of 400 on one class) collapse too, on replicates r0 and r1. The best held-out is C21 (hidden
   step x0.2, 30 epochs): 0.222,
   12 of 54 items, binomial p 0.18 against chance 0.167, with headroom 0.093 over the frozen control. Neither 30
   epochs (C20-C23) nor a smaller hidden step (C21, C22, C24) reached p < 0.05.
   `calib_selection_all_rounds_s7.json` applies the pre-registered rule to C0-C24: none qualifies.
6. **The dev budget sits at the exact gradient's threshold** (`diag_oracle_online_budget_s7.json`). The rate
   backprop oracle, trained online at lr 0.05, reads 0.81 held-out at H32 after 4000 updates (10 epochs x 400) and
   0.94 after 8000. At lr 0.3 it stays at or below 0.33 held-out at every checkpoint, for H32 and H64 and for 400
   and 1260 training items.

## Dev run at the best non-qualifying config (C21), 4 arms x 3 replicates

No config qualified, so the pre-registration's dev run had no chosen config. It was run at C21, the largest
held-out ceiling of C0-C24, to get the plan's two measurements: FA-wall coverage and wall time. It carries no
pre-registered weight. Artifact: `research/findings/raw/gap4/transport_ceiling_readout/dev_rev1ec6635/dev_s7.json` (rev 1ec6635, pool2, numpy, H32/pool 4, 30 epochs).

<!--derived-->
| replicate (task seed) | frozen | fixed_fa | micro_inengine | transport_ceiling | ceiling p | headroom | fa_wall |
|---|---|---|---|---|---|---|---|
| r0 (7) | 0.130 | 0.204 | 0.167 | 0.222 | 0.18 | 0.093 | no |
| r1 (10014) | 0.185 | 0.130 | 0.074 | 0.148 | 0.70 | -0.037 | yes |
| r2 (20021) | 0.130 | 0.185 | 0.167 | 0.148 | 0.70 | 0.019 | no |
| mean | 0.148 | 0.173 | 0.136 | 0.173 | | | |

Held-out inheritance accuracy, 54 items per replicate, chance 0.16667 (the majority-class rate, 9 of 54). The ceiling clears chance on 0 of 3 replicates,
so seed 7 is UNDEFINED under the pre-registered rule, and n_fa_wall is 1 of 3. Replicate 0 is the same task, config
and cfg.seed as C21, and it reproduced C21's reads exactly (ceiling 0.222, frozen 0.130, ff-moved 162621.2 and 9063.4
in both). C21's 0.222 did not recur on the other two task replicates. Training accuracy against each replicate's
training chance (0.1825, 0.170, 0.1625): the ceiling is at or below chance on all three (0.190, 0.1525, 0.140; binomial
p for above 0.37, 0.84, 0.90). The frozen readout is above chance on r0 and r2 (0.2725, p 6e-6; 0.265, p 1.4e-7) but not
on r1 (0.200, p 0.065). That is the same residual as item 5. The plan's success check (n_fa_wall at least 3 and a
defined ceiling) is not met. Logs: `dev_rev1ec6635/dev_agg.log`, `dev_rev1ec6635/fullsize_timing.log` (the per-shard
stdout of the dev run was not kept).

**Wall time.** The dev run took 2.8-3.8 ms per step per process on pool2 (numpy, 12 dev processes plus one full-size
process at once). A full-size timing burst (H64/pool 16, C21 flags, 16 training items, 1 epoch, same machine and
load) took 68 ms per step for frozen and 119 for the ceiling (`research/findings/raw/gap4/transport_ceiling_readout/dev_rev1ec6635/fullsize_timing_s7.json`). At that
rate a 40-epoch shard (1.04M training steps) takes 20-34 hours on the CPU, so the full size belongs on the GPU. The
only cupy timing of this net is the 2026-09-14 legacy run (research/queue/gpu_queue.log): 4026-4473 s per arm for
about 1.06M steps, roughly 4 ms per step. For a full seed (4 arms x 3 replicates x 40 epochs, 12 shards) that is
about 14 GPU-hours if the shards ran one after another. The GPU launcher runs the 4 arms as parallel processes, so
the wall time lies between about 3.5 hours (perfect overlap) and 14 hours (none), before any per-step cost the C21
flags add. On the small net the C21 flags cost about 1.4x the legacy per-step time on numpy.

The byte-identity selftest passed (`identity_selftest_s7.json`): with every new knob at its legacy value,
`Gap4ReadoutNet` reproduces `Gap4InEngineNet`'s weights and reads byte-for-byte in all four arms, and cfg.seed gives
identical thresholds at build.

## What this changes

On the 2026-09-15 net at full size, cutting the input weights changed the H1 reads by no more than run-to-run noise
(item 1: effect/noise 0.91-0.95, item reliability about 0). That is the most likely explanation of the 2026-09-15 UNDEFINED, with two causes acting together: default
feedforward STP and the legacy feedforward gain. Other on-bridge BDSP results built on `OnBridgeBDSPNet` at the legacy
gain with default STP may have the same problem. The per-unit, noise-referenced check
(`_gap4_tc_transmission_noise_ref_diag.py`, pattern in `tests/test_gap4_tc_feedforward_transmission.py`) is the way to
re-read them. This is not a verdict on the credit rules those runs tested. The instrument is now closer to askable,
since transmission, the read and the baseline are each fixed behind a flag. It is not yet interpretable, because
hidden learning collapses the output (every hidden-learning arm, the ceiling included).

## The next lever (no-defer)

Ask what the real circuit runs alongside this that the net replaced with a constant. The first candidate is the
static weight bound itself. At C21 the hidden-learning arms end with a feedforward L1 norm of about 225k-259k over
25,600 synapses, a mean |w| of 8.7-10.1 against the +-12 bound, while the frozen arm ends near 3.7. Many hidden weights
may therefore be pinned at the clamp, and this finding had not checked it (the CLAUDE.md question: 97% of a gap#5
weight change was the clamp). The roadmap's gap#4 ledger row already names the BDSP weight clamp as a lever
(`fused_bdsp_update` clips even at lr 0, commit 6a9a44c3). Second: the output population has no competition, no lateral inhibition. Third: the
engine's threshold homeostasis targets 0.02/ms, below the operating rates
<!--derived-->
here (0.02-0.09/ms), with thresholds bounded at -30 mV. Whether it sits pinned at that bound was not measured. Next
rungs, in this order:

0. **The bound census (AMENDMENT 5 H, queued on the pool).** Per arm and pathway, the fraction of feedforward synapses
   at +-w_max after C21 training, 4 arms x 3 replicates on dev seed 7. If at least 10% of a hidden pathway's synapses
   sit at the bound in the ceiling arm on 2 of 3 replicates, the bound is the next lever (the companion process the
   clamp replaced), before 1 and 2.
1. Spiking lateral inhibition or divisive normalization in the output population, as neurons, not host code.
2. Output-unit homeostasis, meaning intrinsic plasticity toward the target rate inside its bounds.
3. The full-size, full-budget run (H64/pool 16, 40 epochs) on dev seed 7 on the GPU, started 15:55 EDT
   (`research/queue/_a9_gap4_tc_gpu.sh`, registered after it started by AMENDMENT 5 F). It answers whether C21
   transfers to the 2026-09-15 net size, read under the amended rule 1, and measures the cupy wall time. An
   evaluation seed still needs its own EVALUATION CONFIG amendment, which the runner now enforces (AMENDMENT 5 D).

## Review fix round (prereg AMENDMENT 5, committed before the diagnostics it governs)

- The apical-lesion anti-cheat is now a matched cut: the lesion arm also zeroes the interneuron rate, so the hidden
  apical stays at rest (tiny-net selftest: 0.0 mV, against 49.2 mV for the old unmatched lesion). Each lesion shard
  records the measured deviation, and the lesion counts as held only at 0 (docs/TERMS.md "lesion").
- The evaluation-seed guard can fail: a committed EVALUATION CONFIG amendment must register the run's config
  fingerprint and seeds; the prereg itself is refused (`guard_selftest_s7.json`).
- Rule 1 now needs headroom of at least 0.05 per interpretable replicate, so a NO-GO cannot be read with no headroom.
- The calibration tie-break uses epochs x steps per example as registered; re-run over C0-C24: still NONE QUALIFIES.
- Every shard now records its git SHA, the training chance, and the bound census.

Seed 42 stays unqueued until a dev config qualifies under the pre-registration. Functional read-outs only.
