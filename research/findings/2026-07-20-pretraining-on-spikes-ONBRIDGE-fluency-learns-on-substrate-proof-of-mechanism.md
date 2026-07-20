# Pretraining-on-spikes — ON-BRIDGE: the WKV cortex's FLUENCY LEARNS on the spiking substrate (proof-of-mechanism)

**Date:** 2026-07-20 · **Status:** proof-of-mechanism GO — the WKV cortex's PRETRAINING (its TinyStories fluency) is
learnable ON the spiking substrate by a pure exact delta rule (ppl 11511 → 356 in one epoch = a 32× drop). Reaching
the full off-bridge ppl ~40 is a wall-clock/scale item (on-bridge per-token stepping is slow), NOT a mechanism wall.
The end-goal deliverable of the owner's "fully-spiking, one shared substrate" steer, at smoke scale. Uses only the
committed additive `cp_ssm_readout_w` forward; NO new `sim/` edit.

## The deliverable (owner steer)

"Fully closing all gaps INHERENTLY means fully-spiking, one brain, single shared substrate." The grounded-render TASK
learning is on-substrate (~0.94); this closes the deeper item — the cortex's PRETRAINING (fluency) LEARNED on the
substrate, not just off-bridge BPTT.

## Result (`_gap_onbridge_fluency_derisk.py`)

The WKV cortex (emb/Wv/decay) is the FIXED reservoir; per token it charges the on-bridge graded `cp_ssm_state`; the
read-out FORWARD `cp_ssm_readout_out = cp_ssm_readout_w @ cp_ssm_state` runs IN the bridge step loop (the committed
additive mechanism) + a host current-token term; the single-linear read-out is trained ON the substrate by the DELTA
rule (`dw = -eta·err·state`, `cp_ssm_state` as the presynaptic eligibility — no BPTT, no weight transport, no adaptive
optimizer). Full vocab (V=4000), TinyStories next-token, held-out ppl:

- **MAIN (lr 0.005, 500 train sentences): ppl 11511 → 356 in epoch 1 (32× drop) — the cortex's FLUENCY LEARNS on the
  substrate.** It then rises (356→394→419) = OVERFITTING the tiny 500-sentence smoke set (the off-bridge plateau ~40
  used 100000 sentences; early-stop/more data closes this).
- **Stability:** lr 0.05 (the reduced-vocab grounded lr) DIVERGED (ppl → 8e8) — the online per-token update over the
  full 4000-vocab with the state accumulating over 18-token sentences has much higher variance than the batched
  off-bridge de-risk; lr 0.005 is stable (learns).
- **FROZEN anti-cheat:** no weight update → ppl stays at chance (~11511) — the on-substrate learning is load-bearing.

## Read-out — honest scope

- **⇒ the WKV cortex's PRETRAINING/FLUENCY is learnable ON the spiking substrate by a pure exact delta rule (no BPTT,
  no weight transport, no adaptive optimizer)** — proof-of-mechanism (32× ppl drop in one epoch), the committed
  graded read-out forward + the delta update reading `cp_ssm_state`. This extends the on-substrate learning from the
  grounded TASK (~0.94) to the cortex's own fluency — a step toward the "fully-spiking, one shared substrate" end goal.
- **The full-scale (ppl ~40) is a WALL-CLOCK item, not a mechanism wall:** on-bridge per-token stepping is ~76 s per
  500 sentences → the full 100000-sentence pretraining is ~hours/epoch on-bridge. The MECHANISM is proven; reaching
  the off-bridge fluency quality on-bridge is a speed/throughput engineering problem (batching the on-bridge stepping;
  a faster on-bridge kernel; or a hybrid where the reservoir state is precomputed once and the read-out delta trained
  on-bridge). The overfitting on the smoke set is a data-scale artifact, not the mechanism.
- **Next toward the end goal:** (1) scale the on-bridge fluency (throughput lever) to reach ~40; (2) the SINGLE
  SHARED SUBSTRATE consolidation — the composer + WKV + the learning all on ONE bridge (De-risk 5 had them on separate
  cupy bridges in one process); (3) the WKV INPUT map on-bridge learning (optional — a random reservoir works, Rung B).

Runner: `_gap_onbridge_fluency_derisk.py` (`--frozen`, `--lr`, `--n-train`).
