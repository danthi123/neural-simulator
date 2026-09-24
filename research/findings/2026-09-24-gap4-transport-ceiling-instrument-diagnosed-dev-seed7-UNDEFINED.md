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
  calibration did find why the 2026-09-15 ceiling sat at chance. Four read-regime defects are measured, and each has
  a flag-level fix in the runner. One residual is measured but not yet fixed.
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
external: Bellec et al. (2020) Nat Commun 11:3625; Mazurek et al. (2026) Front Neurosci; Carandini & Heeger (2012)
  Nat Rev Neurosci 13:51 (cited in the pre-registration, recorded with tools/record_external_search.sh).
builds_on:
  - research/findings/2026-09-15-gap4-inengine-selfpredict-interneuron-UNDEFINED-transport-ceiling-foreclosed.md
---

# gap#4 transport ceiling: instrument calibration on dev seed 7 — UNDEFINED, four read-regime defects found

**Headline.** The plan asked for a longer or stronger readout so the copied-weight transport ceiling could clear chance.
No config did, on dev seed 7, so the pre-registered fallback applies: UNDEFINED at dev, and seed 42 was not queued. The
calibration did explain why the 2026-09-15 ceiling sat at chance. On this on-bridge net the class never reached the
output read, in any arm.

## What was measured (dev seed 7; small net H32/pool 4 unless stated)

<!--derived-->
1. **The feedforward pathway transmitted almost nothing** (`diag_transmit_scan_stp_ts*_s7.json`). With the default
   Tsodyks-Markram short-term depression on, the H1, H2 and output rates did not change at any tonic level when
   `ff_w_init` went from 4 to 40 or `propagation_strength` from 0.05 to 0.5. At tonic 0 the input layer fires at
   0.096/ms and H1 stays at 0.005/ms. With depression bypassed on the explicit feedforward synapses only
   (`--no-ff-stp`), ff 40 / ps 0.5 transmits (tonic 0: H1 0.035/ms). The engine's own STP block documents the same
   effect for sustained stimulus-driven firing. On the full-size 2026-09-15 net (H64/pool 16, legacy config), zeroing
   the input->H1 weights moved the mean H1 read only from 0.0548 to 0.0547 (`diag_fullsize_ff_transmission_s7.json`).
   Per unit, the H1 read changed by 0.009 on average (about 17% of the mean). That probe took no intact-vs-intact
   noise reference, so the per-unit change cannot yet be told apart from run-to-run noise.
2. **The event read is non-monotonic in drive** (`diag_eread_monotonic_s7.json`). Extra output current from 0 to
   +1600 pA raises the total spike rate from 86 to 404 Hz but lowers the BDSP event rate `E` from 0.050 to 0.002. The
   default output layer sits at the peak, so strengthening an output neuron lowers its read. Round 1 (event read)
   trained every arm to BELOW-chance training accuracy. Fix: `--read-quantity spikes`.
3. **The burst-probability baseline cancels each teaching transient.** With the EMA baseline (alpha 0.05/step) the
   frozen readout's training accuracy stays near chance at the transmitting operating point (C12, lr 5: 0.115; C18,
   30 epochs: 0.142). With the preset baseline (`--pbar-alpha 0`) it fits the training set to 0.265 (C15; guessing among nine training classes gives 0.111).
4. **Synapse elimination, not learning, made up most of the reported weight movement.** With elimination off (C9),
   the frozen arm's total weight change is 2.1, against about 1270 with it on. Default-on elimination makes every
   weight below 0.05 eligible for zeroing (5e-7 per step), and that includes every negative signed feedforward weight.
5. **Residual, not fixed: the transport ceiling's hidden learning collapses the output.** At the best configs the
   ceiling predicts one class for 360-395 of 400 training items (C15, C17). At every preset-baseline config (C15,
   C17, C19-C24) its training accuracy (0.150-0.190) stays BELOW the frozen readout's (0.242-0.273). (At the two
   EMA-baseline configs the frozen readout itself stays low: C16 0.180, C18 0.142.) The best held-out is C21 (hidden
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
in both). C21's 0.222 did not recur on the other two task replicates. On all three, the ceiling's training accuracy
(0.140-0.190) stays below the frozen readout's (0.200-0.273), the same residual as item 5. The plan's success check
(n_fa_wall at least 3 and a defined ceiling) is not met.

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

On the dev net the forward pass barely carried the input. On the 2026-09-15 net at full size the mean H1 read did not
move when the input weights were zeroed; the per-unit change is unresolved without a noise reference. That is the
most likely explanation of the 2026-09-15 UNDEFINED. Other on-bridge BDSP results built on `OnBridgeBDSPNet` with the
default STP on should be re-read with the transmission check (`tests/test_gap4_tc_feedforward_transmission.py` shows
the pattern). This is not a verdict on the credit rules those runs tested. The instrument is now closer to askable, since transmission, the read and the baseline are
each fixed behind a flag. It is not yet interpretable, because the ceiling's own hidden learning collapses the output.

## The next lever (no-defer)

Ask what the real circuit runs alongside this that the net replaced with a constant. Here the output population has
no competition: no lateral inhibition. The engine's threshold homeostasis targets 0.02/ms, below the operating rates
<!--derived-->
here (0.02-0.09/ms), with thresholds bounded at -30 mV. Whether it sits pinned at that bound was not measured. A
hidden layer driven by copied-weight credit then drives one output unit up without check. Next rungs:

1. Spiking lateral inhibition or divisive normalization in the output population, as neurons, not host code.
2. Output-unit homeostasis, meaning intrinsic plasticity toward the target rate inside its bounds.
3. The full-size, full-budget calibration (H64/pool 16, 30-40 epochs) on dev seed 7 on the GPU. This answers whether
   the small-net result transfers before any evaluation seed is spent, and measures the cupy wall time directly.
   The launcher is `research/queue/_a9_gap4_tc_gpu.sh` (C21 flags, 40 epochs, 4 arms x 3 replicates, dev seed 7 by
   default; the runner refuses an evaluation seed without a committed amendment).

Seed 42 stays unqueued until a dev config qualifies under the pre-registration. Functional read-outs only.
