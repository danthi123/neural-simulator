# EMERGE-81 — the spiking Izhikevich LSM's graded-memory advantage SURVIVES the port: holds a distal 1-bit cue ≥ 16 fillers — CHARACTERIZATION (3-seed)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge81_spiking_memory_depth_derisk.py`
**Test:** `tests/test_emerge81_spiking_memory_depth.py`
**Raw:** `research/findings/raw/_emerge81_spiking_memory_depth.json`

## Why

EMERGE-80 ported the reservoir form→role mechanism to a spiking Izhikevich liquid-state machine (learns the map + resolves
the single-embedding non-local rel-head). EMERGE-79 showed the RATE `tanh` reservoir holds a real-discovered distal 1-bit
cue across ≥ 28 fillers. A spiking pool's fading memory is *typically* shorter — so the honest follow-on measures the
spiking pool's memory DEPTH on the SAME uncontingent variable-distance task, to see whether the graded-memory advantage
survives the port and where the RANK-3 rung becomes necessary on the spiking substrate.

## The characterization (3 seeds 42/43/44; reuse EMERGE-79 task + EMERGE-80 SpikingLSM; NO `sim/` edit)

A REAL discovered voice marker (act/pas, `marks_discovered = True`) flips a far word's role across a variable number of
fillers; the spiking LSM's accuracy vs distance:

| fillers | spiking LSM | ±2 window |
|---|---|---|
| 0 | **1.00** | 0.53 |
| 4 | **1.00** | 0.51 |
| 8 | **1.00** | 0.54 |
| 12 | **1.00** | 0.49 |
| 16 | **1.00** | 0.53 |

*(3 seeds; the spiking LSM 1.000 at every distance; the ±2 window at chance throughout; mark-lesion collapses to ~0.47.)*

**Result:** the spiking Izhikevich pool holds the distal 1-bit cue at **1.000 across the entire reduced sweep (≥ 16
fillers, ~21 tokens)**, beating every fixed window at every distance (the window is blind), with the mark-lesion collapsing
the role to chance (genuinely mark-determined). So the **graded-memory advantage SURVIVES the port to spikes** — the
spiking realization is not merely a rate abstraction that loses the distal memory; the recurrent Izhikevich pool retains a
1-bit distal feature comparably to the near-critical `tanh` reservoir in this range (a strong distal perturbation persists
in the recurrent + tonic-bias dynamics).

## Honest scope

- A **characterization**, not a new capability. The sweep is REDUCED (max 16 fillers, 3 seeds) because the spiking sim is
  ~30× heavier per token than the `tanh` reservoir; the depth limit was not reached in range (a longer/GPU sweep would
  find it). The rate reservoir held ≥ 28; the spiking pool held ≥ 16 here (comparable within the tested range).
- Beyond the pool's depth, and for multi-feature / deeper-recursion dependencies, the RANK-3 rung (theta-gamma WM buffer /
  assembly-calculus stack) is the named next mechanism for the spiking substrate.
- Reuse-by-import (EMERGE-79 task + EMERGE-80 SpikingLSM); NO `sim/` edit.

## Together with the reservoir arc

EMERGE-78 (learned map) → EMERGE-79 (uncontingent non-local, rate) → EMERGE-80 (realized on spiking Izhikevich) → EMERGE-81
(the graded-memory advantage survives on spikes): the fronto-striatal reservoir is a **spiking** liquid-state machine
replacing the hand form→role labeler, with genuine, uncontingent, distance-spanning graded memory — the anti-whack-a-mole
RANK-1 mechanism on the project's own neuron model.

## Files
- `research/runners/_emerge81_spiking_memory_depth_derisk.py` — composes the EMERGE-79 distal task with the EMERGE-80
  SpikingLSM; distance sweep + memory-depth + local-baseline/mark-lesion controls.
- `tests/test_emerge81_spiking_memory_depth.py` — 2 CPU tests (composition, the seed-42 distal-hold smoke).
- `research/findings/raw/_emerge81_spiking_memory_depth.json` — the 3-seed characterization.
