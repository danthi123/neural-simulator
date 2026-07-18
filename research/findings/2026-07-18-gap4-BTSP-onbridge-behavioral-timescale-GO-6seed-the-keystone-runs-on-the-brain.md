# 🎉 Gap #4 — on-bridge BTSP is BEHAVIORAL-TIMESCALE via the REAL bistable plateau (6-seed GO). The local plateau-gated one-shot credit rule now runs ON THE SPIKING SUBSTRATE, and the gap#5 bistable dendritic plateau is its enabler on-brain. NO new `sim/` edit.

**2026-07-18.** The gap#4 BTSP mechanism (6-seed GO, rate/analytic) is now realized and validated ON a real
`SimulationBridge`, at behavioral timescale, driven by the ACTUAL on-bridge bistable dynamics — closing the on-bridge
rung of the local-credit keystone.

## What runs on the brain
- `fused_btsp_update` (`sim/kernels.py`): `dw = eta * Etilde_pre * IS_post * (w_max - w)` (saturating one-shot
  potentiation), IS_post = the dendritic plateau above v_hold. Local; no weight transport; no global loss.
- A guarded default-off `enable_btsp` block in `bridge._run_one_simulation_step`: a SECONDS-long per-neuron
  pre-eligibility `cp_btsp_pre_elig` (low-pass of firing, `coo.row`) × the plateau instructive signal `cp_v_apical`
  above v_hold (`coo.col`), gated by the plastic-mask + `cp_plasticity_rate_gain` exactly like BDSP. Byte-identical
  when off (determinism 9pass; `test_onbridge_btsp` byte-identical assert).

## The behavioral-timescale demonstration (6-seed GO)
Reuses BOTH of this session's committed edits — NO new `sim/` edit: the plateau is evolved by the REAL **bistable BDSP
apical** (`enable_bdsp` + `bdsp_apical_bistable`, self-regen SUSTAIN + KIR) from a BRIEF `cp_bdsp_apical_drive` pulse, and
`enable_btsp` reads that held plateau. BDSP learning is OFF (lr=0) so BTSP is the sole weight-mover.
`_gap4_btsp_onbridge_behavioral_timescale_derisk.py`:

| seed | held_dw (v_apical_end) | transient_dw (v_apical_end) | moat_dw | off_dw |
|---|---|---|---|---|
| 42  | 103.2 (−24.2) | 11.6 (−65.0) | 0.0000 | 0.0000 |
| 43  | 120.3 (−24.2) | 15.0 (−65.0) | 0.0000 | 0.0000 |
| 44  | 120.2 (−24.2) | 14.5 (−65.0) | 0.0000 | 0.0000 |
| 100 | 96.3 (−24.2)  | 10.9 (−65.0) | 0.0000 | 0.0000 |
| 101 | 122.7 (−24.2) | 15.1 (−65.0) | 0.0000 | 0.0000 |
| 102 | 95.3 (−24.2)  | 11.0 (−65.0) | 0.0000 | 0.0000 |

- **The bistable plateau is LOAD-BEARING on the substrate:** a HELD plateau (v_apical latched at −24.2, above v_hold
  −35) potentiates the co-active pre→post synapse one-shot over a seconds-long window (held_dw ~110); a TRANSIENT plateau
  (v_apical decayed to −65) gives only a brief window (transient_dw ~13, **8.4× less**). All 6 seeds.
- **Moat clean:** a silent apical (no pulse) → dw 0.0000 (no plateau → no instructive signal → no potentiation).
- **Byte-identical off:** `enable_btsp=False` → dw 0.0000 all 6 seeds.
- CI-pinned: `test_onbridge_btsp` (potentiation / moat / byte-identical / behavioral-timescale). `cfg.seed` set
  (substrate genuinely seeded); a fresh bridge per condition (dendritic state reset by construction).

## Gap #4 status — the local-credit keystone is a WORKING, on-bridge, biological rule
The full ladder is closed for BTSP: rate mechanism GO (6-seed) → on-bridge rule VALIDATED (potentiation/moat/byte-id) →
on-bridge BEHAVIORAL-TIMESCALE GO (6-seed, the real bistable plateau, 8.4× vs transient). The gap#5 bistable dendritic
plateau (the `sim/` keystone) is the enabler on-brain — it converts ms spike-timing plasticity into a seconds-long
one-shot credit window. Deep supervised backprop credit stays a confirmed boundary (banked method, not the capability).
NEXT: (b) a one-shot TASK (association/place-field) the substrate LEARNS via BTSP; (c) gap#5 UNIFICATION (BTSP stores
the CA3 assembly the bistable CA3 completes — the two gaps share the keystone). Honest scope: local one-shot credit,
NOT multi-layer/deep credit (confirmed-hard, not claimed).
