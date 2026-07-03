# EMERGE-84 — the genuine STACK-recursion boundary: a plain reservoir does depth-1 nested pair-matching perfectly, then DEGRADES with depth — names where RANK-3 (WM-buffer/stack) is necessary — BOUNDARY (6-seed)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge84_reservoir_stack_recursion_derisk.py`
**Test:** `tests/test_emerge84_reservoir_stack_recursion.py`
**Raw:** `research/findings/raw/_emerge84_reservoir_stack_recursion.json`

## Why (the genuine recursion test, vs EMERGE-83's retention)

EMERGE-83 showed the reservoir tracks the MATRIX subject's number across center-embedding to depth ≥4 — but that is
*retention* (the first cue is at the START; the reservoir just holds it), NOT the stack-requiring core of recursion. This
de-risk requires genuine nested **pair-matching**: a center-embedded structure where each subject must agree with its OWN
verb, and the verbs appear in **reversed** pairing order (`… v3 v2 v1`) — so judging grammaticality needs a **push/pop
stack** (push subjects; pop for verbs in order). A reservoir has fading memory, not a stack.

**The count shortcut is defeated:** the ungrammatical case SWAPS two verbs' numbers (both classes carry ≥2 distinct subject
numbers), so grammatical and ungrammatical share the identical number multiset — a "count the sng/plu" baseline is at
chance, forcing genuine per-pair matching. Markers (`sng`/`plu`) are real discovered cues.

## The result (6 seeds; rate reservoir; NO `sim/` edit)

| depth (# nested pairs) | reservoir | count baseline | position-shuffle |
|---|---|---|---|
| 1 (2 pairs) | **1.00** | 0.50 (chance) | 0.50 (chance) |
| 2 (3 pairs) | **~0.91** | 0.55 | 0.50 |
| 3 (4 pairs) | **~0.81** | 0.54 | 0.51 |

*(6 seeds; markers discovered = True; the count-multiset shortcut and position-shuffle are both at chance throughout.)*

**The boundary:** the reservoir judges grammaticality of nested pair-matching **perfectly at depth 1 (1.00)** — it CAN do
shallow nested matching — then **degrades with nesting depth** (~0.91 at depth 2, ~0.81 at depth 3, heading toward chance).
Its stack-recursion depth **d\* = 2** (last depth ≥ 0.90); it boundaries at depth 3. This is the honest, expected signature:
a plain reservoir has fading memory, **not a push/pop stack**, so it handles bounded nesting and strains as depth grows —
exactly where humans also fail (~2 center-embeddings). The count-shortcut being at chance + position-shuffle collapsing
confirm the reservoir is doing genuine structural matching (not a count or a lexical artifact), and that the boundary is a
*recursion-depth* limit, not a task-invalidity.

## The mechanism this names (the boundary = an undiscovered mechanism)

Per the master directive, this boundary LAUNCHES the search for the next mechanism. Genuine stack-recursion past the
reservoir's fading-memory depth needs an explicit **working-memory stack**: the RANK-3 candidates named by the research
gate — a **theta-gamma multiplexed WM buffer** (catalog N.15; Lisman-Idiart 1995 — a theta cycle nesting ~7 gamma-locked
slots is a time-multiplexed stack) or **assembly-calculus disinhibition-controlled recursion** (Mitropolsky et al.,
arXiv:2206.13217). The direct follow-on (EMERGE-85) ADDS a minimal WM-buffer/stack alongside the reservoir and re-tests
whether it pushes the recursion depth past d\*=2 — the surpass step.

## Honest scope

The canonical stack-recursion test (center-embedded agreement matching) on a bounded corpus, rate level. The boundary is a
*depth* limit of a plain reservoir (a known property — reservoirs are not stack machines), precisely measured (d\*=2) and
named, NOT a wall: the RANK-3 stack mechanism is the next de-risk. Reuse-by-import (EMERGE-78 Reservoir/Encoder); NO `sim/`
edit.

## Files
- `research/runners/_emerge84_reservoir_stack_recursion_derisk.py` — the nested pair-matching grammaticality task
  (reversed verbs, multiset-preserving-swap ungrammatical) + depth-scaling + count/shuffle controls.
- `tests/test_emerge84_reservoir_stack_recursion.py` — 4 CPU tests.
- `research/findings/raw/_emerge84_reservoir_stack_recursion.json` — the 6-seed boundary.
