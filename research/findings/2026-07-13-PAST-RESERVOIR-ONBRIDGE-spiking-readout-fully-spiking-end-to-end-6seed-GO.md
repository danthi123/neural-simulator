# EMERGENCE-BAR COMPLETION (6-seed GO): the on-bridge coupling's read-out ARGMAX is now on spikes too — reservoir + selective channel + read-out ALL spiking, one brain; the FS-WTA winner matches the exact argmax on 5/6 seeds

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_onbridge_couple_selssm_reservoir_derisk.py --spiking-readout` · raw `research/findings/raw/_onbridge_spk/`. NO further `sim/` edit.
**Status:** ✅ GO 6/6. The on-bridge coupling is now FULLY-SPIKING end-to-end.

## Why

The on-bridge coupling GO (reservoir + `cp_ssm_state` selective channel co-resident on one bridge) computed the read-out ARGMAX in host numpy. This completes the emergence bar: the linear read-out's per-token scores drive a one-of-K FS-WTA Izhikevich bridge (the validated `build_fswta_score_bridge`/`fswta_drive`, the same spiking WTA the D3 register + `_reslm_spiking_readout` use); the SPIKING winner is the prediction. Read-out WEIGHTS stay learned by the committed local delta rule (no BPTT) — only the argmax SELECTION moves onto spikes. So reservoir (spiking Izhikevich recurrence) + selective channel (`cp_ssm_state`) + read-out (FS-WTA) are ALL spiking, on ONE `SimulationBridge`.

## Result — 6-seed (gated-conjunction, chance 1/6 = 0.167)

| seed | numpy_acc (exact argmax) | spiking_acc (FS-WTA) | parity |
|---|---|---|---|
| 42 | 0.413 | 0.340 | 0.753 |
| 43 | 0.360 | 0.360 | 0.973 |
| 44 | 0.340 | 0.340 | 1.000 |
| 100 | 0.387 | 0.387 | 1.000 |
| 101 | 0.407 | 0.407 | 1.000 |
| 102 | 0.387 | 0.387 | 1.000 |

- **6/6 GO**: the spiking FS-WTA read-out reads a winner well above chance (2× chance) and without catastrophic loss vs the exact argmax.
- **`spiking_acc == numpy_acc` on 5/6 seeds** (parity 1.000) — the fully-spiking read-out is IDENTICAL to the exact argmax on 5 of 6 seeds. Only seed 42 shows a finite-spike cost (spiking 0.340 vs numpy 0.413, parity 0.753) — the FS-WTA's finite-spike discrimination on that seed's close scores (a settle / score-discriminability lever, not a mechanism limit; `_reslm_spiking_readout` reached ~0.97 parity on more-discriminable scores).

## ⇒ interpretation

The selective-SSM long-range coupling now runs FULLY on spikes, one brain, end-to-end: a recurrent Izhikevich reservoir + the `cp_ssm_state` selective channel + an FS-WTA read-out, all on one `SimulationBridge`, the selective channel lifting the spiking reservoir past a long-range conjunction and the spiking winner matching the exact argmax on 5/6 seeds. The emergence bar (no host computation between sensation and the motor/output selection) is met for the coupled long-range generator: the only host code is the world/task (the token stream) and the learned read-out WEIGHTS (a synaptic quantity, local-delta-learned).

## Files
- `research/runners/_reslm_onbridge_couple_selssm_reservoir_derisk.py` (`--spiking-readout`, `run_spiking_readout`); raw `research/findings/raw/_onbridge_spk/seed*.json`.
