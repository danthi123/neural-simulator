# EMERGE-86 — RANK-3 SURPASS, ON SPIKES: the theta-gamma WM buffer + stack-match realized on the project's spiking RF ordered-WM — **GO** (6-seed)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge86_spiking_wm_buffer_recursion_derisk.py`
**Test:** `tests/test_emerge86_spiking_wm_buffer_recursion.py`
**Raw:** `research/findings/raw/_emerge86_spiking_wm_buffer_recursion.json`
**Research gate:** `research/findings/2026-07-03-spiking-theta-gamma-wm-buffer-research-gate.md`

## Why (the gate's decisive finding: ~95% already built)

EMERGE-85 surpassed the reservoir's recursion boundary with a RATE-level functional theta-gamma WM buffer (an ordinal
array + a host mirror-pair `==`). The spiking-theta-gamma research gate found the spiking realization is **~95% already
built**: `research/runners/ordered_position_wm.py` `OrderedPositionWM` is a PRODUCTION (6-seed GO) spiking Lisman-Idiart
ordered-WM whose encode/read run on real **resonate-and-fire** spikes (`_unbind_phases` = spiking conj-diagonal
complex-synapse unbind; Im zero-crossing = gamma-slot phase). The only residual was the mirror-pair `==`, for which the
substrate already has its spiking primitive (a phase-coincidence / familiarity op).

## The mechanism (fully spiking)

`SpikingWMBuffer` wraps `OrderedPositionWM(vocab=['sng','plu'], n_slots=8)`. `feature(toks)` extracts the number-marker
sequence, ENCODES it into the ordered gamma-slots on RF spikes (`encode_sequence`; items past 8 dropped = the bounded
stack), then for each MIRROR pair (slot *k* vs slot *N−1−k* = the LIFO stack pop) it reads BOTH slots on spikes
(`read_slot` = spiking unbind **then cleanup**, which removes the bundle crosstalk) and takes the **phase-coincidence
between the two cleaned concept phasors** — the spiking familiarity op that replaces the host `==`. The per-pair coincidence
vector → a ridge read-out (grammatical iff all pairs cohere). Storage, recall, and match are all on the RF substrate.

## The result (6 seeds; reuse the EMERGE-84 task + EMERGE-85 depths; NO `sim/` edit)

| depth | fully-spiking WM | slot-scramble | unbind-lesion | count baseline |
|---|---|---|---|---|
| 1 (2 pairs) | **1.00** | 0.48 | 0.50 | 0.50 |
| 2 (3 pairs) | **1.00** | 0.49 | 0.50 | 0.54 |
| 3 (4 pairs) | **0.95** | 0.51 | 0.50 | 0.54 |
| 4 (5 pairs, 10 numbers > 8 slots) | **0.43** (overflow) | 0.49 | 0.50 | 0.56 |

*(6-seed means; per-seed d3 ranges 0.91–1.00. wm_stack_depth_star = 3.)*

**The spiking surpass:** the WM buffer realized on real RF spikes reaches stack-depth **d\*=3** — past the plain reservoir's
**d\*=2** (EMERGE-84) — via the spiking encode + unbind + cleanup + coincidence, then **boundaries at the buffer capacity**
(depth 4 = 10 numbers overflow the 8 slots) — the biologically-faithful BOUNDED recursion limit (the human ~2–3-embedding
bound). Controls all collapse: **slot-scramble** → chance (the ordered gamma-slots = the LIFO stack are load-bearing);
**unbind-lesion** → chance (the match is genuinely from the spiking slot recall, not a static read); **count-multiset**
shortcut → chance (defeated by the multiset-preserving swap).

**Honest process note:** the first pass compared the RAW recovered phasors and failed at chance — each unbind read carries
different position-crosstalk that dominates the raw coincidence. The fix is the substrate's own **cleanup** (read_slot's
spiking unbind→cleanup removes the crosstalk, recovering the clean concept), then the coincidence between the two cleaned
concepts — still fully spiking.

## Honest scope

- **RUNG 1** — the buffer + stack-match on the validated spiking RF ordered-WM (multiplex + storage + recall + coincidence
  all spiking). **RUNG 2** — a literal time-domain theta/gamma OSCILLATOR nesting the slots (catalog N.15; a thin additive
  default-off `sim/` oscillator driver: theta-phase-modulated gamma `excitability_drive`, reusing `StimulusManager`
  SINUSOIDAL + `CORTEX_GAMMA_FS_NETWORK` + the per-region NMDA mask) is the separable fuller realization, NOT on the RANK-3
  critical path.
- The recursion is BOUNDED by the buffer capacity (the human ~2–3-embedding limit), not unbounded — the biologically
  faithful bound.
- Reuse-by-import (OrderedPositionWM + EMERGE-84 task); NO `sim/` edit.

## The RANK-3 arc, complete on spikes

EMERGE-83 (retention, not recursion) → EMERGE-84 (the genuine stack-recursion BOUNDARY: reservoir d\*=2) → EMERGE-85 (the
RANK-3 theta-gamma WM buffer SURPASSES it, rate) → **EMERGE-86 (the surpass realized ON the project's spiking RF substrate)**.
The boundary was an undiscovered mechanism; the mechanism was found, and now runs on spikes — bounded stack-recursion
resolved on the one brain.

## Files
- `research/runners/_emerge86_spiking_wm_buffer_recursion_derisk.py` — `SpikingWMBuffer` (OrderedPositionWM + spiking
  cleaned-concept mirror-pair coincidence) + the depth-scaling de-risk + slot-scramble / unbind-lesion controls.
- `tests/test_emerge86_spiking_wm_buffer_recursion.py` — 4 CPU tests.
- `research/findings/raw/_emerge86_spiking_wm_buffer_recursion.json` — the 6-seed surpass.
