# Single-shared-substrate CAPSTONE — the composer + the WKV cortex run a grounded turn on ONE bridge (GO)

**Date:** 2026-07-20 · **Status:** GO (seeds 42/43/100) — the composer (RF bind/unbind/cleanup) AND the WKV cortex
(cp_ssm_state read-out + RF spike-encoder) run a full grounded turn on ONE `SimulationBridge` (three regions), with
the composer recall + the no-confab moat + the WKV render ALL byte-identical to isolated, and the composer op
byte-isolated from the WKV state even when INTERLEAVED between WKV tokens. The owner's "fully-spiking, one brain,
single shared substrate" end goal, realized for the grounded conversational turn. NO `sim/` edit.

## What this closes (owner end-goal)

De-risk 5 ran the grounded turn with the composer (RFPhasorComposer, per-op RF bridges) + the WKV renderer (two
bridges) as SEPARATE bridges co-executing in one PROCESS. This CAPSTONE consolidates them onto ONE bridge — the
literal "single shared substrate."

## The structure — ONE bridge, three regions

- **`chan`** (`2*D_wkv` = 512) — `cp_ssm_state`, the WKV leaky read-out state.
- **`encoder`** (`2*D_wkv` = 512) — the WKV's RF spike-encoder (masked `rf_resonate_steps`).
- **`composer`** (`7*D_cmp` = 448) — the composer's RF bind/unbind/cleanup ops, via the `MergedRFComposer` index-shift
  port: `SharedBridgeComposer(RFPhasorComposer)` overrides `_resonate` to rebase the bind conns by `rf_base`, kick with
  `neuron_mask=composer-slice`, `rf_resonate_steps`, read the slice.

A whole grounded turn on ONE substrate: composer **STORE** facts (RF bind) → composer **QUERY** (RF unbind + cleanup on
the shared bridge) → the WKV **RENDER**s the retrieved answer (ssm forward on the same bridge).

## Result (`_gap_onebridge_capstone_derisk.py`, seeds 42/43/100)

Facts `[dog→chase→cat, owl→eat→mouse, wolf→hunt→deer]`; shared-bridge composer + WKV vs isolated references:
- **composer recall shared vs isolated: `['cat','mouse','deer']` == `['cat','mouse','deer']` → True** (all seeds) —
  the real composer store/query (RF bind/unbind + cleanup) is byte-faithful on the shared slice.
- **no-confab moat shared vs isolated: `None` == `None` → True** (all seeds) — an unstored cue (`lion roar`) abstains
  on the shared bridge exactly as isolated.
- **WKV generation shared vs isolated: True** (all seeds) — both emit `you help me find my way home and`.
- **composer op INTERLEAVED between WKV tokens → WKV logits `max|err| = 0.000e+00`** (all seeds) — the STRONGEST
  result: a composer query run BETWEEN two WKV `_charge` calls on the SAME bridge does not perturb the WKV state.
  Genuine byte-isolation under interleaved use (not merely sequential), because the composer touches `v`/`u` + `cp_rf_*`
  (re-kicked per op) while the WKV state lives in `cp_ssm_state` (touched only by the ssm block).

CI: `tests/test_onebridge_capstone.py` (2 tests, GPU + ckpt, else skip).

## Read-out — the consolidation is realized

- **⇒ the composer + the WKV cortex run a full grounded turn (STORE → QUERY → RENDER) on ONE shared spiking substrate**,
  every output identical to the separate-bridge De-risk-5 pipeline, the no-confab moat intact, byte-isolated even
  interleaved. Combined with the on-bridge learning (the delta rule over `cp_ssm_state`, already on this bridge type),
  the whole grounded conversational loop — comprehend/store/recall/abstain/render + the render-learning — is realizable
  on a SINGLE `SimulationBridge`. The end-goal "single shared substrate" is met for the grounded turn.
- **De-risk chain that got here (all byte-clean, this session):** co-residence crux (ssm read-out + RF phasor, 6-seed
  0.0) → encoder-equivalence (`rf_resonate_steps` == step-loop, 0.0) → WKV physical merge (two bridges → one,
  byte-exact) → this capstone (composer + WKV, one bridge, byte-identical + interleave-isolated).
- **Honest scope:** the composer's FACT STORE is the numpy-kb idealization (the composer's documented "principled
  idealization"; its on-substrate resonate ops ARE on the shared bridge — the spiking part is consolidated). The
  on-bridge fluency THROUGHPUT (batched stepping to reach the off-bridge ppl ~40 LIVE) remains the wall-clock lever,
  not a mechanism/consolidation gap. Different D per faculty (WKV D=256, composer D=64) coexist as separate regions.
- **Next:** (1) the throughput lever (batched on-bridge stepping) to run the LIVE full-scale fluency on the single
  substrate; (2) optionally consolidate the composer's substrate store (`enable_substrate_store`) onto the same bridge.

Runner: `_gap_onebridge_capstone_derisk.py` (`--seed`, `--ckpt`, `--D-cmp`).
