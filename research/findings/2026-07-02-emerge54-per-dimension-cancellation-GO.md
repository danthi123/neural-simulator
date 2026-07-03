# EMERGE-54 — PER-DIMENSION Collins-Quillian cancellation over pooler-DISCOVERED conversational codes — GO (3-seed)

**Date:** 2026-07-02
**Runner:** `research/runners/_emerge54_per_dimension_cancellation_derisk.py`
**Test:** `tests/test_emerge54_per_dimension_cancellation.py` (3 tests, CPU/numpy, offline)
**Raw:** `research/findings/raw/_emerge54_per_dimension_cancellation.json`
**Verdict:** **GO** — 3-seed (42/43/44), NO `sim/` edit, reuse-by-import.

## The bug this fixes (disclosed in EMERGE-52)

EMERGE-52's conversational console let a member-specific EXCEPTION **dominate the whole read across ALL
dimensions**. Because `_best(member)` returned the strongest OVR regardless of which property was asked,
`ask_can` answered the exception for any query once a member had one:

```
you> a penguin walks               (a LOCOMOTION exception)
you> can a penguin fly?     brain> No, a penguin walks.   (correct — locomotion overridden)
you> can a penguin breathe? brain> No, a penguin walks.   (WRONG — the exception leaked into RESPIRATION;
                                                            the penguin should still INHERIT breathes)
```

EMERGE-52's own script documented this as a known wrinkle: *"the member's exception dominates the read
(honest Collins-Quillian: the strongest specific fact wins; per-property override is a follow-on)."* This is
a real reasoning-correctness bug: a LOCOMOTION exception must override ONLY locomotion, not block unrelated
RESPIRATION inheritance (Collins-Quillian: an exception overrides only its own dimension). EMERGE-27 already
validated this per-DIMENSION cancellation over hand-assigned codes; EMERGE-54 applies that pattern to the
pooler-DISCOVERED conversational codes.

## The fix (reuse-by-import; the only change is the READ)

`PerDimensionConsole` subclasses EMERGE-52's `MultiLevelConversationalConsole`. The stacked competitive
pooler (discover sub-category → genus → order), the committed `sim/` three-term-kernel teaching (class
properties on the members' discovered L2/L3 codons; a member exception on its identity ensemble), and the
graded apical read are all **unchanged**. Two small changes:

1. **`learn_exception` records the exception's DIMENSION** (from a small property→dimension lexicon —
   locomotion: fly/walk/swim/lurk; respiration: breathe — exactly EMERGE-27's host-side `DIMS`, the
   keyboard/language interface, not a brain computation).
2. **`ask_can(member, prop)` reads only the asked property's dimension:** an exception cancels the class
   default ONLY if that exception is in the SAME dimension as the asked property; otherwise the member
   inherits the class default for P purely from the discovered-codon graded drive. One predicate
   (`_exception_in_dim`) gates the whole fix.

## Result — the demo (seed 42), the fix visible

```
you> a penguin walks               (member EXCEPTION on LOCOMOTION only)
you> can a penguin fly?     brain> No, a penguin walks.        (LOCOMOTION overridden)
you> can a penguin breathe? brain> Yes, a penguin can breathe. (RESPIRATION INHERITED — THE FIX)
you> can a pike swim?       brain> No, a pike lurks.           (LOCOMOTION overridden)
you> can a pike breathe?    brain> Yes, a pike can breathe.    (RESPIRATION INHERITED — THE FIX)
you> can an owl fly?        brain> Yes, an owl can fly.        (non-overridden inherits locomotion)
you> can an owl breathe?    brain> Yes, an owl can breathe.    (non-overridden inherits respiration)
you> can an owl swim?       brain> I don't know...             (sibling-discrimination — owl is a bird)
you> can a zzz breathe?     brain> I don't know what a zzz is. (no-confab MOAT)
```

## De-risk gates (3-seed 42/43/44), all met

| gate | value | requirement |
|---|---|---|
| **PER-DIMENSION cancellation** (override on overridden dim AND inherit on other dim, both hold) | **1.00** | ≥ 0.99 |
| — override-locomotion (penguin flies → No, walks) | 1.00 | |
| — inherit-other-dimension (penguin breathes → Yes) | 1.00 | |
| non-overridden inherit (owl/minnow on all dimensions) | 1.00 | ≥ 0.99 |
| sibling-confusion (owl inherits fish 'swim'?) | 0.00 | ≤ 0.05 |
| moat abstains on unknown token | True (all seeds) | all |
| moat false-accepts | 0 | 0 |
| **dAP-LESION collapses inheritance** (primary control) | 0.00 | ≤ nonoverride − 0.30 |
| permute-co-occurrence sibling-confusion (secondary) | max +0.50 over real | seed-variable, reported |

**PER-DIM cancel 1.00, override-loco 1.00 + inherit-other-dim 1.00, non-override inherit 1.00,
sibling-confusion 0.00, moat 0 FA, dAP-lesion 0.00** — identical across all three seeds. GO is stable
across repeated runs (verified 3×).

## Controls — the honest read

- **PRIMARY (load-bearing): dAP-LESION** removes the coincidence / two-compartment substrate the graded
  apical read flows through → the per-dimension inheritance read collapses to abstain (0.00) every seed,
  deterministic. The substrate the fix reads is load-bearing.
- **SECONDARY (seed-variable, reported): PERMUTE-CO-OCCURRENCE** (EMERGE-52's control) breaks the
  codon-driven sibling-discrimination on at-least-one seed (raising sibling-confusion). Per EMERGE-45's
  honest scope this is seed-variable (the co-occurrence stream is keyed by the spoken taxonomy), so it is a
  secondary diagnostic, not a hard mean gate — mirroring EMERGE-52's own test.
- **NOT used: permute-FEATURES.** It does NOT collapse here — the co-occurrence stream is keyed by the
  spoken taxonomy, so it still groups same-branch members even with random features (exactly EMERGE-52's
  documented finding). Using it as a collapse gate would be dishonest; it is deliberately excluded.

## Honest scope

- Composes EMERGE-52 (discover multi-level taxonomy + NL console) + EMERGE-27 (per-dimension cancellation).
  The property→dimension map is a small host-side lexicon (the keyboard/language interface, EMERGE-27's
  `DIMS`), not a brain computation — it tells the reader which dimension a query/exception lives in. The
  teaching, discovery, and graded read are unchanged (reuse-by-import).
- **This fixes the EMERGE-52 reasoning-correctness wrinkle** (an exception leaking across dimensions).
  **Follow-on:** DISCOVERING which properties belong to the same dimension from statistics (rather than the
  host lexicon) — the same emergent-structure-from-experience direction as EMERGE-30/32.
- Curated bird/fish taxonomy (the EMERGE-52 demo vocabulary); corpus-scale is a follow-on.

## Bottom line

The conversational reasoning now does correct **per-dimension Collins-Quillian cancellation** over
pooler-discovered codes: **penguin flies = No (walks), penguin breathes = Yes (inherited)** — both hold,
3-seed, non-overridden members inherit on all dimensions, sibling-discrimination and the no-confab moat
intact, dAP-lesion collapses the read. On one spiking `SimulationBridge`, transformer-free, NO `sim/` edit.
