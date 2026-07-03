# EMERGE-83 — the reservoir RESISTS agreement-attraction across center-embedding to depth ≥4 (a retention capability) — but this is NOT stack-recursion — CHARACTERIZATION (6-seed)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge83_reservoir_recursion_depth_derisk.py`
**Test:** `tests/test_emerge83_reservoir_recursion_depth.py`
**Raw:** `research/findings/raw/_emerge83_reservoir_recursion_depth.json`

## Why (the RANK-3 cheap-first probe)

The research gate ranked RANK-3 (bounded recursion) as the deeper frontier past the reservoir's single-embedding non-local
capability. This de-risk probes it on the canonical psycholinguistic test — **subject-verb number agreement across
center-embedding** (agreement attraction): the MATRIX verb must agree with the MATRIX subject's number, resisting the
lure of the nearer EMBEDDED subjects. Number markers `sng`/`plu` and the relativizer `that` are real, frequent, discovered
words (no OOV).

## The result (6 seeds; rate reservoir; NO `sim/` edit)

| depth (# embeddings) | reservoir | nearest-number (attraction) baseline |
|---|---|---|
| 0 | 1.00 | 1.00 |
| 1 | **1.00** | 0.51 (chance — predicts the innermost subject) |
| 2 | **1.00** | 0.50 |
| 3 | **1.00** | 0.50 |
| 4 | **1.00** | 0.50 |

*(6 seeds; markers discovered = True; matrix-number-lesion collapses the answer to ~0.53 = genuinely the matrix subject's
number.)*

The reservoir tracks the **matrix** subject's number to depth **≥4** (the max tested, held at 1.000), decisively beating
the agreement-**attraction** baseline (which predicts the nearest/innermost number and is at chance for depth ≥1). It does
NOT fall for the attraction humans/models are lured by, across up to 4 embedded distractors.

## Honest scope — this is RETENTION, not STACK-recursion (the load-bearing distinction)

This task scores the **matrix (first) subject's number**, which sits at the START of the sequence — so the reservoir
succeeds by **retaining the first cue amid interference** (a memory capability; EMERGE-79/81 already showed the reservoir's
first-cue memory is long, ≥16–28 fillers). It is **not** the stack-requiring core of recursion: it does not require
matching each embedded subject to *its own* verb across a reversed verb sequence (`… V3 V2 V1`), which needs a **push/pop
stack** a reservoir provably cannot maintain past shallow depth. So the depth-≥4 result is a genuine (and notable —
attraction-resistant) **retention** capability, NOT evidence that a plain reservoir does unbounded recursion.

⇒ The genuine recursion boundary — where the theta-gamma multiplexed WM buffer (catalog N.15) / assembly-calculus stack
(Mitropolsky) becomes necessary — requires a **nested pair-matching** task (each subject ↔ its own verb across center-
embedding), which is the direct follow-on (EMERGE-84). This finding correctly AVOIDS over-claiming a recursion boundary the
data does not show, and names the task that would expose it.

## Files
- `research/runners/_emerge83_reservoir_recursion_depth_derisk.py` — the center-embedding agreement task + depth-scaling +
  the attraction baseline + matrix-number-lesion.
- `tests/test_emerge83_reservoir_recursion_depth.py` — 4 CPU tests.
- `research/findings/raw/_emerge83_reservoir_recursion_depth.json` — the 6-seed characterization.
