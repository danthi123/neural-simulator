# Render-LEARNING co-resides with the composer on ONE bridge — the WRITE side of "everything on one substrate" (GO)

**Date:** 2026-07-20 · **Status:** GO (6-seed 42/43/44/100/101/102) — the render-LEARNING (a read-out over
`cp_ssm_state`, updated by the pure exact DELTA rule) LEARNS on the shared bridge WHILE the composer (RF bind/query on
its own region) coexists and its recall + no-confab moat stay correct. Closes the WRITE side of the single-shared-
substrate consolidation (the capstone proved the READ side). NO `sim/` edit.

## Why this matters (owner steer: "load everything onto the one shared substrate")

The capstone put the composer's RF ops + the WKV read-out on ONE bridge, and the crux proved `cp_ssm_readout_w @
cp_ssm_state` (the read) co-resides byte-clean with the composer. This closes the remaining piece: the LEARNING (the
delta-rule WRITE to `cp_ssm_readout_w`, reading `cp_ssm_state` as the presynaptic eligibility) runs on the SAME bridge
as the composer, learning while the composer binds/queries — so the render-learning is ON the shared substrate too,
not just the read.

## Result (`_gap_onebridge_learning_coresident_derisk.py`, 6-seed)

On the capstone bridge (chan + encoder + composer regions), a read-out over the chan region's `cp_ssm_state` learns a
supervised target by `dw = -lr·err·state` (no BPTT, no weight transport), with a composer query interleaved every 40
steps:
- **on-bridge delta-rule learning: loss → 0.000 (≥7×10⁸ drop, learned) — all 6 seeds.**
- **composer recall (interleaved into the learning loop): `['cat','mouse']` correct — all 6 seeds.**
- **no-confab moat: `None` (abstains on the unstored cue) — all 6 seeds.**
- **ANTI-CHEAT (frozen read-out): loss DRIFTS (1.69 → 2.01) but NEVER learns (no 5× drop)** — with `W` frozen the loss
  still moves because the ssm integrator settles `s` toward `inject` (so `out = W@s` changes), but it does not descend
  toward 0; the delta update is load-bearing. (First-pass gate mis-specified this as "flat"; verify-first caught it —
  the state evolution means frozen drifts, so the correct control is "does not LEARN," not "is flat.")

CI: `tests/test_onebridge_learning_coresident.py` (2 tests, GPU-only).

## Read-out

- **⇒ the render-LEARNING (exact delta rule over `cp_ssm_state`) + the composer + the WKV read-out are ALL on ONE
  substrate, learning while the composer works.** Combined with the capstone (composer + WKV forward on one bridge,
  byte-identical) this puts the whole grounded loop — comprehend/store/recall/abstain + spiking render + the
  render-LEARNING — on a single `SimulationBridge`.
- **Remaining "not on the shared substrate" item (honest):** the composer's FACT-STORE (the numpy-kb idealization —
  the composer's documented "principled idealization"; its spiking bind/query resonate ops ARE on the shared bridge).
  Consolidating the fact-store onto the shared bridge is a real arc (the substrate store uses `rf_set_complex_weights`
  which REPLACES `cp_rf_w_*`, conflicting with the per-op bind — needs the persistent-store-on-slice machinery the
  `CoResidentOneBrainComposer` uses), not a quick win.

Runner: `_gap_onebridge_learning_coresident_derisk.py` (`--seed`, `--n-steps`, `--lr`, `--frozen`).
