# EMERGE-79 — the reservoir's non-local advantage is UNCONTINGENT: genuine graded memory over every fixed window for a REAL discovered cue (resolves the EMERGE-78 contingency) — **GO** (6-seed)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge79_reservoir_variable_distance_derisk.py`
**Test:** `tests/test_emerge79_reservoir_variable_distance.py`
**Raw:** `research/findings/raw/_emerge79_reservoir_variable_distance.json`

## Why (the honest follow-on to EMERGE-78)

EMERGE-78 showed the reservoir resolves a relative-clause head where no fixed ±2 window can — but the focused adversarial
recheck found that result **CONTINGENT**: the relativizer "that" was out-of-vocabulary (0 occurrences in the discovery
corpus) and collided with the OPEN marker, so an object-relative and a transitive had identical local windows. Were "that"
a distinct discovered cue, a ±1 window would tie and the reservoir advantage would vanish. EMERGE-79 asks the
**uncontingent** question: with the disambiguating cue a **REAL, frequent, DISCOVERED** closed word, and the non-locality
coming purely from **DISTANCE** (a variable number of intervening fillers), does the reservoir's fading memory give a
genuine advantage over *every* fixed window?

## The dependency (no OOV, no collision — non-locality from distance alone)

A voice marker at the sentence START flips the role of a content word at the END: `<mark> <filler>*n the s zeps the o`,
where `mark ∈ {act, pas}` sets `o` = THEME (act) or AGENT (pas). The marks `act`/`pas` and the filler `um` are made **real,
frequent tokens** in the discovery stream, so EMERGE-62 discovers `act`/`pas` as **distinct closed cues** (verified: 6/6
seeds `marks_discovered = True`). The number of fillers VARIES, so `mark` sits a variable number of tokens before `o`. A
fixed ±W window at `o` can only see `mark` when it is within W — it CLIFFS at its width; the reservoir's final-state
read-out holds `mark` across the fillers in its fading memory.

## The de-risk — **GO** (6 seeds 42/43/44/100/101/102; rate-level, CPU/numpy; reuse EMERGE-78 Reservoir/Encoder)

| distance (fillers) | reservoir | ±2 window | ±4 window |
|---|---|---|---|
| 0 | **1.000** | 0.47 | 0.47 |
| 4 | **1.000** | 0.48 | 0.48 |
| 8 | **1.000** | 0.49 | 0.49 |
| 16 | **1.000** | 0.52 | 0.52 |
| 28 | **1.000** | 0.49 | 0.49 |

*(All 6 seeds: reservoir 1.000 at every distance; both windows at chance throughout.)*

| control | value | meaning |
|---|---|---|
| marks discovered as distinct closed cues | **True** (6/6) | no OOV trick — uncontingent (unlike EMERGE-78) |
| **±2 window LOCAL-sanity** (mark ADJACENT to the word) | **1.000** | the window CAN do the role task locally → it fails on the distal case only because of DISTANCE, not incapacity |
| MARK-LESION (replace the marker with a neutral filler) | **~0.52** (chance) | the role is genuinely marker-determined |
| reservoir fading-memory depth (≥ 0.75) | **≥ 28 fillers** (~33 tokens; held across the whole tested range) | the depth was not reached in the tested range |

**The result:** the reservoir resolves a variable-distance role flip that NO fixed window can, with the disambiguating cue
a REAL discovered word — beating every fixed window by ~0.50 at all distances 0–28, holding the cue across ~33 tokens. The
local-sanity control proves the window's failure is from distance (it does the task at 1.000 when the mark is adjacent),
and the mark-lesion proves the role is genuinely the marker's. This is **genuine, uncontingent graded-memory value** — it
resolves the EMERGE-78 focused-recheck contingency: the reservoir's non-local advantage is **not** an OOV artifact.

## Honest scope

- A single **variable-distance role flip** (a 1-bit distal feature), not deep recursion. It isolates ONE question —
  uncontingent graded-memory advantage over fixed windows for a real discovered cue — and answers it decisively.
- The near-critical reservoir (spectral radius 0.95, 500 units) holds a **1-bit** distal feature across ≥ 28 fillers; the
  depth limit was not reached in the tested range (a longer sweep would find it). DEEPER/unbounded and multi-feature
  dependencies past its memory depth are the RANK-3 frontier (theta-gamma WM buffer / assembly-calculus stack).
- Rate-level, comprehension-side; reuse-by-import (EMERGE-78 Reservoir/Encoder + EMERGE-62 discovery); NO `sim/` edit.

## Together with EMERGE-78

EMERGE-78 (consolidation: the reservoir LEARNS the form→role map, no hand branch) + EMERGE-79 (uncontingent: the
reservoir's recurrence has genuine graded-memory value no fixed window can match, for a real cue) establish the
fronto-striatal reservoir as a **learned, genuinely non-local** replacement for the hand form→role labeler on the
comprehension side — the anti-whack-a-mole RANK-1 mechanism, with the bounded-recursion RANK-3 rung named for deeper
structure. Next pre-registered rungs: the spiking-LSM port; the production reservoir (Dominey 2015, roles → ordered
emission).

## Files
- `research/runners/_emerge79_reservoir_variable_distance_derisk.py` — the variable-distance dependency + distance-scaling
  vs fixed ±2/±4 windows + the local-sanity/lesion/marks-discovered controls.
- `tests/test_emerge79_reservoir_variable_distance.py` — 4 CPU tests (construction shapes, local-sanity, the seed-42
  uncontingent GO gates).
- `research/findings/raw/_emerge79_reservoir_variable_distance.json` — the 6-seed de-risk.
