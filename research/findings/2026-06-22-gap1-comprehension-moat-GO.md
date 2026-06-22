# GAP-1 — the comprehension-time no-confab moat for the multi-cue parser (2026-06-22)

**Scope:** the conversational-PRIMARY (robust multi-cue Competition-Model parser) deep-research's ranked #1 gap
(`2026-06-22-robust-multicue-parser-scoping.md`). **Reuse-by-import, NO `sim/` edit.** On `main`.

## The hole (verified in code)
`brain_conversational_agent.py:342` (multicue) / `:319` (case): `roles = parse(words, voice)` → `composer.store(...)`
— the production `hear_multicue`/`hear_case` call `parse()` (which ALWAYS commits a role), NOT `parse_decisive()`.
`parse_decisive` (the content gate that reports `decisive=False` on a content-ambiguous sentence — two animate nouns
+ a symmetric verb, no decisive content cue) exists on BOTH parsers but was **never called on the production hear
path**. So an ambiguous degraded sentence **CONFABULATED + STORED a wrong fact** at comprehension; the composer's
query-time moat only abstains on *unstored* facts — it cannot un-store a *wrong stored* one. The R1 imperfect-English
demo even sidestepped this by using inanimate patients (always content-decisive).

## The fix
Route `hear_multicue`/`hear_case` through `parse_decisive`; on `decisive=False`, **ABSTAIN at comprehension** (store
nothing, return `None`) rather than confabulate. Both `MultiCueRoleParser.parse_decisive` (`multicue_role_parser.py:122`)
and `CaseAwareRoleParser.parse_decisive` (`case_aware_role_parser.py:205`) already exist + are validated. Flag-OFF
byte-identical (the fix only runs when `enable_multicue_competition`/`enable_case_competition` is ON).

## De-risk (cheap-first, numpy) — GO
`research/runners/_phaseB_multicue_comprehension_moat_derisk.py` drives the proposed fix EXTERNALLY (no agent edit) on
fresh per-sentence agents, multi-seed:

| metric | seed 42 | 6-seed (42,43,44,100,101,102) |
|---|---|---|
| FIX ambiguous-degraded → ABSTAIN (0 confabulated stores) | 1.00 / 0 | **6/6** (1.00 / 0 every seed) |
| the bug genuinely exists (CURRENT `parse` path confabulates) | 1.00 | **6/6** (1.00 every seed) |
| **MARGIN-LESION** (`abstain_margin=0` → gate can't fire → confabulates) | 0.00 | **6/6** (0.00 every seed) |
| FIX decisive-degraded (object-fronted) → resolved | 1.00 | **6/6** (1.00 every seed) |
| FIX canonical → unregressed | 1.00 | **6/6** (1.00 every seed) |
| query-time moat intact | True | **6/6** (True every seed) |

**6/6 GO** (`b8sqhxacf`, seeds 42,43,44,100,101,102) — every metric identical across all 6 seeds; the content gate's
decision is deterministic given the parser's competition read, so there is no seed variance.

The **margin-lesion** is the decisive control: with the gate disabled it reproduces today's confabulation, proving
the abstention is *caused by the gate* (not the parser silently failing on ambiguous input). The CURRENT-confabulates
arm proves the hole is real, not hypothetical.

## Gate (the WIRED behavior) — PASS
`tests/test_multicue_competition_agent.py` (+ the new `test_multicue_comprehension_moat_abstains_on_ambiguous`) +
`tests/test_case_cue_crosslanguage_agent.py` + `tests/test_brain_conversational_agent.py` (byte-identity):
**24 passed, 5 skipped (GPU-gated)**. The agent now abstains (`hear()` returns `None`, stores nothing) on a
content-ambiguous degraded sentence; decisive + canonical unregressed; the no-confab moat is **strengthened at the
comprehension layer, never weakened**.

## Next (the deep-research's remaining gaps, cheapest-first)
GAP-2 (R3 coverage: attributed adj+noun + multi-clause — the *moved* bottleneck; robustness is solved, coverage is
the gap), GAP-3 (the production-default flip, gated at V=320), GAP-4 (the WTA object-front operating-point
calibration, deep-research-gated). Or Tier-2 (one-brain integration). The moat fix (GAP-1) came first because the
moat is sacrosanct.
