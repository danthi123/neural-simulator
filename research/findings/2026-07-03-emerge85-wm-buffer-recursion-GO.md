# EMERGE-85 — RANK-3 SURPASS: a theta-gamma WM buffer + stack-match pushes recursion depth PAST the reservoir's limit, to the biologically-faithful bounded bound — **GO** (6-seed)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge85_wm_buffer_recursion_derisk.py`
**Test:** `tests/test_emerge85_wm_buffer_recursion.py`
**Raw:** `research/findings/raw/_emerge85_wm_buffer_recursion.json`

## Why (the EMERGE-84 boundary launches this mechanism)

EMERGE-84 found the genuine recursion boundary: a plain reservoir does depth-1 nested pair-matching perfectly, then
degrades (d\*=2) — it has fading memory, not a push/pop stack. The research gate named the RANK-3 mechanism: a **theta-gamma
multiplexed WM buffer** (catalog N.15; Lisman-Idiart 1995 — a theta cycle nesting ~7 gamma-locked slots = a
capacity-bounded stack). This de-risk ADDS that mechanism and re-tests the same task.

## The mechanism (a bounded buffer + a stack match)

`WMBuffer(capacity=8)` multiplexes each number-marker token into the next ordered gamma-slot (unfading, up to capacity).
The **stack match** is the theta-gamma coincidence that pairs each verb to its top-of-stack subject: in center-embedding
the subjects fill the first slots and the verbs fill the rest in REVERSE, so verb *j* sits at the MIRROR slot *N−1−j* of
subject *j* (LIFO pop). The mechanism's output = the per-mirror-pair AGREEMENT (slot *k* vs slot *N−1−k*) — a bounded set of
gamma-coincidence comparisons, unfading within capacity.

## The result (6 seeds; rate level; NO `sim/` edit)

| depth | plain reservoir (EMERGE-84) | **WM buffer + stack-match** | buffer-slot-scramble |
|---|---|---|---|
| 1 (2 pairs) | 1.00 | **1.00** | 0.50 |
| 2 (3 pairs) | ~0.91 | **1.00** | 0.51 |
| 3 (4 pairs) | ~0.81 | **~0.97** | 0.50 |
| 4 (5 pairs, 10 numbers > 8 slots) | ~0.70 | **~0.46** (overflow) | 0.49 |

*(6 seeds; markers discovered = True; the count-multiset shortcut stays at chance — inherited from EMERGE-84's
construction.)*

**The surpass:** the WM-buffer-augmented read-out reaches stack-depth **d\*=3** vs the plain reservoir's **d\*=2** — strictly
deeper. The buffer holds the whole nested number sequence in ORDERED slots (no fading), so the stack-match resolves every
pair **within capacity** (8 slots = depth-3, 8 numbers) — then **boundaries at the capacity** (depth-4 = 10 numbers overflow
the 8 slots → the verbs are dropped → matching fails). This is the biologically-faithful **BOUNDED** recursion limit — the
theta-gamma buffer holds ~7±2 items, matching the human ~2–3-center-embedding bound — NOT unbounded recursion. The
**buffer-slot-scramble** collapses the read to chance, proving the ORDERED slots are the STACK structure (LIFO mirror
pairing), not a bag.

## Honest scope

- The rate-level **functional** theta-gamma buffer (a running-ordinal slot multiplex + the mirror-pair stack match). The
  **spiking** theta-gamma realization — a theta oscillation nesting gamma-locked assembly slots on the substrate (catalog
  N.15), with the pop/match as a gamma-phase coincidence — is the pre-registered follow-on rung.
- The recursion is BOUNDED by the buffer capacity (the human ~2–3-embedding limit), not unbounded — the biologically
  faithful bound, not a wall.
- Reuse-by-import (EMERGE-84 task + EMERGE-78 Encoder); NO `sim/` edit.

## The RANK-3 arc

EMERGE-83 (the reservoir RESISTS agreement-attraction to depth ≥4 — *retention*, not recursion) → EMERGE-84 (the genuine
stack-recursion BOUNDARY: reservoir d\*=2, fading memory not a stack) → **EMERGE-85 (the RANK-3 theta-gamma WM buffer +
stack-match SURPASSES it to d\*=3, the bounded human limit)**. The boundary was an undiscovered mechanism; the mechanism was
found and surpasses it. Next: the spiking theta-gamma realization of the buffer on the substrate.

## Files
- `research/runners/_emerge85_wm_buffer_recursion_derisk.py` — `WMBuffer` (ordered multiplex + mirror-pair stack match) +
  the reservoir-vs-buffer depth-scaling de-risk + the slot-scramble control.
- `tests/test_emerge85_wm_buffer_recursion.py` — 5 CPU tests.
- `research/findings/raw/_emerge85_wm_buffer_recursion.json` — the 6-seed surpass.
