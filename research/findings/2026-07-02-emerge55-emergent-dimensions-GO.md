# EMERGE-55 — EMERGENT DIMENSIONS: the per-dimension cancellation grouping is LEARNED from experience statistics, not host-listed (GO, 3-seed)

**Date:** 2026-07-02
**Runner:** `research/runners/_emerge55_emergent_dimensions_derisk.py`
**Test:** `tests/test_emerge55_emergent_dimensions.py` (3 CPU/numpy tests, offline)
**Raw:** `research/findings/raw/_emerge55_emergent_dimensions.json`
**Verdict:** **GO** (3-seed 42/43/44). NO `sim/` edit. Reuse-by-import (EMERGE-54 console + its EMERGE-52 machinery).

## The shortcut burned down

EMERGE-54's per-dimension (Collins-Quillian) cancellation — an exception overrides only its own property
DIMENSION, so a penguin's `walks` cancels only LOCOMOTION and still lets it INHERIT respiration (`breathes`) —
relied on a **host-side lexicon** `PROP_DIM = {"fly":"LOCOMOTION", "walk":"LOCOMOTION", "swim":..., "breathe":"RESPIRATION"}`
that HAND-LISTS which properties share a dimension (the same host scaffold as EMERGE-27's `DIMS`). Per the
no-host-scaffolding standard, EMERGE-55 makes the **DIMENSION STRUCTURE EMERGENT** — LEARNED from the statistics of
experience, not hand-listed.

## The mechanism (learn dimensions from statistics)

Properties in the SAME dimension are **alternatives**: they are MUTUALLY EXCLUSIVE across members (a member has
exactly ONE locomotion — a robin flies XOR a penguin walks XOR a trout swims XOR a pike lurks), and an exception
REPLACES the class default on that dimension. Properties in DIFFERENT dimensions **co-occur** freely (a member both
flies AND breathes). So the grouping is discoverable from a property×member co-occurrence matrix:

1. Reconstruct `{member → set(property TRUE for it)}` from the taught facts: each member inherits the class defaults
   of its ancestor classes, and its own exception REPLACES the class default on the same slot (the deepest/most-specific
   class default — a taxonomy-depth rule, so `penguin walks` replaces `fly`, not `breathe`).
2. Co-occurrence count `C[p][q]` = members for which both are true. `C > 0` ⇒ p, q are in **different** dimensions
   (a member cannot have two values of one dimension). This is the CONFLICT signal.
3. Two properties are **alternates** (same-dimension candidate) iff they NEVER co-occur (`C == 0`) and both are present.
4. Connected components of the alternates graph = the discovered dimensions — each a "which one" WTA group of
   mutually-exclusive alternatives (competitive/lateral-inhibition grouping; the same co-occurrence family the
   project uses for stream-cortex learning; Numenta semantic-folding of alternatives; Bates-MacWhinney competition).

`breathe` co-occurs with EVERY locomotion (every member breathes AND has one locomotion) ⇒ isolated in its own
dimension; `fly/walk/swim/lurk` never co-occur with each other ⇒ one connected component = the locomotion dimension.

That LEARNED grouping REPLACES `PROP_DIM` in EMERGE-54's read (`EmergentDimensionConsole._dim_of` now reads the
learned grouping). An exception cancels only its LEARNED dimension; other LEARNED dimensions inherit untouched.
**There is deliberately NO host `PROP_DIM` fallback**: an unlearned/wrong dimension makes the exception's dimension
not match the asked property's, so the read fails/leaks rather than silently falling back.

## Gate numbers (3-seed 42/43/44, all identical)

| gate | value | note |
|---|---|---|
| **(1) DIMENSION-DISCOVERY** | **1.00** | learned `{breathe}` + `{fly, lurk, swim, walk}` == the true partition, every seed |
| **(2) PER-DIM cancellation (learned dims)** | **1.00** | `penguin flies`=No/`breathes`=Yes AND `pike swims`=No/`breathes`=Yes (override-loco 1.00 + inherit-other 1.00) |
| non-override inheritance | 1.00 | owl/minnow inherit on both learned dimensions |
| sibling-discrimination | 0.00 | owl (a bird) does NOT inherit fish `swim` |
| **(3) MOAT** | abstains, **0 false-accepts** | unknown token `zzz` → "I don't know what a zzz is" |
| **(4) DESTROYED-EXCLUSIVITY control** | per-dim → **0.00** (discovery 0.40) | **load-bearing: the read BREAKS every seed** |
| dAP-lesion (secondary) | inherit → 0.00 | the graded-apical substrate is load-bearing |

## The load-bearing control (an honest correction during the build)

The first control I wrote was a **global label bijection** (permute the property words). It does NOT break the read —
a bijection merely RENAMES properties and preserves the partition, so the learned dimensions stay correct (the signal
lives in the ASSIGNMENT, which member holds which property, not the names). A **per-member random re-draw** broke the
assignment but was **seed-fragile** (a lucky draw re-created a valid RESP/LOCO structure on 1–2 of 3 seeds — the mean
passed but individual seeds did not break, which would have been a dishonest GO).

The final control is **truth-blind and deterministic**: give every member the FULL property vocabulary, so every
property pair co-occurs → the learner forces every property into its OWN singleton dimension → the exception's
learned dimension (`walk`) no longer matches the asked property's (`fly`) → `_exception_in_dim` returns False → the
read falls through to inheritance → `penguin flies` wrongly answers **Yes** (cancellation failed). This destroys the
mutual-exclusivity statistics the grouping is learned from, and breaks per-dim cancellation to **0.00 on every seed**.
It confirms the LEARNED grouping is doing the work, with no host fallback masking it.

## Honest scope

- The **member→property statistics** are reconstructed from the taught facts (class defaults + per-member exception
  substitution). The one remaining host-side statistics heuristic is *which class default an exception substitutes for*
  — the deepest/most-specific class default (a taxonomy-depth rule, NOT a dimension lexicon). Corpus-scale dimension
  discovery (many dimensions, learned substitution) is a follow-on.
- `TRUE_DIM` is used ONLY to SCORE discovery accuracy, never in the read.
- The pooler, teaching, graded read, and bridge are unchanged from EMERGE-54 (reuse-by-import); the only change is that
  `_dim_of` reads the LEARNED grouping.

## Bottom line

The per-dimension Collins-Quillian cancellation structure is now **EMERGENT** — the brain LEARNS which properties are
alternatives-on-one-dimension from the co-occurrence statistics of experience (mutually-exclusive alternates), and
that learned grouping drives correct per-dimension cancellation, with the destroyed-exclusivity control breaking it
every seed. The last host-scaffolding shortcut disclosed in EMERGE-54 (`PROP_DIM`) is burned down. One spiking brain,
transformer-free, NO `sim/` edit.
