# EMERGE-80 — the reservoir form→role mechanism PORTS to the SPIKING substrate: a recurrent Izhikevich liquid-state machine — **GO** (6-seed)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge80_spiking_lsm_port_derisk.py`
**Test:** `tests/test_emerge80_spiking_lsm_port.py`
**Raw:** `research/findings/raw/_emerge80_spiking_lsm_port.json`
**Scoping:** `research/findings/2026-07-03-rank2-production-reservoir-residual-scoping.md` (RANK-2 = whack-a-mole → do this port instead)

## Why (the RANK-1.5 directive-aligned move)

EMERGE-78/79 established the fronto-striatal reservoir as a LEARNED, genuinely non-local replacement for the hand
form→role labeler — but at RATE level (a numpy `tanh` echo-state pool). The RANK-2 scoping found the production side
already self-organized (a "production reservoir" would be whack-a-mole) and named the highest-leverage next move: PORT the
already-GO rate reservoir onto the SPIKING substrate — the non-negotiable fully-spiking-one-brain end state, pre-registered
by the EMERGE-6b "reservoir + trained read-out" gate. EMERGE-80 is that port: a genuine **liquid-state machine** (Maass
2002) on the project's own **Izhikevich 2007 RS** neuron model.

## The mechanism

`SpikingLSM` mirrors the EMERGE-78 `Reservoir` API (a `final_state(U)`), so it **drops into the entire EMERGE-78 harness**
(construction generators, final-state slot read-out, the governing-cue + symmetric-window baselines, hand-labeler control,
anti-cheats) with only the pool swapped. The pool: a fixed-random recurrent Izhikevich pool (C=100/k=0.7/vr=−60/vt=−40/
a=0.03/b=−2/c=−50/d=100, v_peak=35 — the project's RS cortical operating point; dt=0.5, 30 sub-steps/token, tonic bias to
keep it fluctuation-driven) driven token-by-token by the EMERGE-62 discovered closed-class configuration. The read-out
feature = per-neuron **spike-count** over the whole sequence (the population rate vector); a ridge read-out maps it → per
content slot's thematic role.

## The de-risk — **GO** (6 seeds 42/43/44/100/101/102; rate-level CPU/numpy; NO `sim/` edit)

| gate | value (6-seed) | bar |
|---|---|---|
| pool genuinely spiking (mean spikes/neuron over a sentence) | **1.19–1.32** | > 0.5 |
| **(A) CONSOLIDATION** — the spiking LSM learns the form→role map (train role acc) | **1.000** | ≥ 0.90 |
| **(B) NON-LOCAL** — spiking LSM on the relative-clause HEAD | **1.000** | ≥ 0.85 |
| — left-context governing-cue baseline | **0.500** (chance) | ≤ 0.65 |
| — symmetric ±2 window baseline | **0.500** (chance) | ≤ 0.65 |
| rel-head word-order scramble | ~0.32 (≈ chance) | collapse |
| **POOL-SILENCE lesion** (zero all drive) — the read collapses | **0.500** (drop 0.500) | genuinely from spikes |
| hand labeler on the multi-arg shapes | 0.000 | — |

Every gate clears on every seed. The spiking Izhikevich pool **learns the full form→role map** (train 1.000) via a ridge
read-out over its whole-sequence population spike-counts, and **resolves the non-local relative-clause head** (1.000) where
both a left-context governing-cue rule and a symmetric ±2 window are at chance — the pool's spiking recurrence integrates
the whole sequence. The **pool-silence lesion** (zeroing all input + recurrent drive) collapses the read to chance,
proving the read-out is genuinely from POOL SPIKES, not a static bias. The reservoir form→role mechanism is therefore not a
rate artifact — it works on the project's spiking neuron model.

## Honest scope

- **RUNG 1** (this): a DIRECT recurrent Izhikevich pool (the project's spiking neuron model) — a faithful spiking LSM,
  CPU-runnable, cheap-first. GO = the reservoir mechanism survives on spikes (learns the map + non-local, pool genuinely
  active + silence-lesion collapses).
- **RUNG 2** (follow-on): put the pool on a full `SimulationBridge` region (the on-substrate realization, so it composes
  with the rest of the one brain).
- The EMERGE-79 distal-cue **memory-DEPTH** (how far a spiking pool holds a 1-bit cue across distance — a spiking pool's
  fading memory is typically SHORTER than a near-critical `tanh` reservoir) is a separate characterization follow-on.
- Reuse-by-import (the EMERGE-78 harness + the project's Izhikevich params); NO `sim/` edit.

## Together with EMERGE-78/79

EMERGE-78 (learned form→role map, no hand branch) + EMERGE-79 (uncontingent non-local graded memory) + EMERGE-80 (the
mechanism realized on the spiking Izhikevich substrate) advance the fronto-striatal reservoir from a rate abstraction to a
**spiking liquid-state machine** replacing the hand form→role labeler — the anti-whack-a-mole RANK-1 mechanism, now on the
project's own neuron model, toward the fully-spiking one brain.

## Files
- `research/runners/_emerge80_spiking_lsm_port_derisk.py` — `SpikingLSM` (recurrent Izhikevich pool, EMERGE-78 API) + the
  de-risk (reuses the EMERGE-78 harness; the pool-silence lesion; the spiking-activity check).
- `tests/test_emerge80_spiking_lsm_port.py` — 5 CPU tests (active pool, silence-collapse, determinism, project Izhikevich
  operating point, the seed-42 spiking GO gates).
- `research/findings/raw/_emerge80_spiking_lsm_port.json` — the 6-seed de-risk.
