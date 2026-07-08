# Taxonomic is-a from DEFINITIONAL syntax — the mechanism has genuine signal on an encyclopedic corpus but is WEAK on WikiText-2 (noisy mining + small corpus): "X is a Y" statements DO carry real taxonomy (fungus←species-names, video←game-titles) that children's-story co-occurrence lacks, but the cheap extraction on 1.67M tokens gives only a borderline inheritance (0.300 vs chance 0.125, deranged 0.172 doesn't cleanly collapse). The taxonomic path needs a cleaner/bigger encyclopedic corpus + POS-based extraction. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_isa_definitional_inheritance_derisk.py` (mine "X is a/an [adj]* HEAD-NOUN" → is-a groups; reuse-by-import: stream-cortex codes + rung-1 inheritance). NO `sim/` edit.
**Verdict:** WEAK/borderline — a new mechanism for the multi-level frontier, with genuine signal but data+extraction-quality-limited on WikiText-2.

## Why this ran (the multi-level negative's next mechanism, driven per owner "no deferrals")
The multi-level negative (2026-07-08) showed co-occurrence CLUSTERING gives FLAT categories, not a taxonomic is-a hierarchy — because children's-story co-occurrence lacks the is-a signal. Confirmed directly: TinyStories "X is a Y" is DESCRIPTIVE, not taxonomic (top superordinates are adjectives big/good/nice from "it is a big surprise"; subjects are pronouns it/that/he). An ENCYCLOPEDIC corpus should carry explicit taxonomy. So this mines is-a categories from definitional syntax (not clustering) and does inheritance over them.

## The signal exists on WikiText (unlike children's stories)
WikiText "X is a/an ... HEAD-NOUN" yields REAL taxonomic superordinates: species (17), member, song (13), character, **fungus** (10, ← bohemica/carbonaria/earthstar/hygrometricus — real fungus species), **video** (← blackwyche/crush/destiny/revolution — game titles), novel, church, game. The is-a hierarchy signal children's stories lack IS present in encyclopedic text.

## The result — WEAK/borderline (WikiText-2, 3-seed)
| min-subjects/super | is-a inherit | deranged | chance |
|---|---|---|---|
| 4 (40 noisy supers) | 0.120 | 0.060 | 0.025 |
| 6 | 0.150 | 0.121 | 0.059 |
| **8 (cleaner supers)** | **0.300** | 0.172 | 0.125 |

With cleaner supers (min-subjects=8) the inheritance is above chance (0.300 vs 0.125) but WEAK — the deranged control (0.172) does NOT cleanly collapse (margin 0.128, borderline; NEGATIVE by the ≥0.15 bar). Two causes: (1) NOISY mining — the head-noun heuristic grabs adjectives/participles ("located"←"X is located...", "american"←"X is an american...") so many supers are junk; (2) the is-a categories are only WEAKLY separated in CO-OCCURRENCE space (encyclopedic entities across different is-a groups have somewhat similar codes → random groupings retain partial inheritance → deranged stays elevated). WikiText-2 is also small (1.67M tokens → only 1683 is-a patterns → 40 noisy supers).

## Extraction improved (noun-filter) — clean categories, bottleneck moves to code strength
Adding a structural NOUN-filter (a real superordinate noun also appears as a SUBJECT of "is a"; adjectives/participles like located/american do not) + a boundary-aware head-noun cleanly extracts TAXONOMIC categories: **game←chess/crush/destiny (games), species←banksia/capensis/carbonaria (species), song←…, version←…** (the noise supers removed). But the inheritance stays borderline (min-subjects=6: inherit 0.225 vs chance 0.077, deranged 0.096 — aggregate margins pass ~0.13-0.15 but per-seed borderline). The bottleneck MOVED from extraction-quality to CODE strength: the specific entities within a clean is-a group (banksia/capensis/carbonaria) do NOT co-occur enough in the small WikiText-2 (1.67M tokens) to give SIMILAR co-occurrence codes → weak inheritance even over clean categories. ⇒ a bigger encyclopedic corpus (more co-occurrence per entity → stronger, more-similar codes within an is-a group) is the lever for a clean taxonomic-reasoning GO.

## What this establishes (the taxonomic path is a data+extraction-quality lever)
The is-a-from-definitional-syntax MECHANISM is sound in principle and has genuine signal on an encyclopedic corpus (real taxonomy: fungus←species, video←games) — the multi-level negative's data-gate is REAL (children's stories lack is-a; encyclopedic text has it). But the cheap extraction on the small WikiText-2 is too noisy for a clean taxonomic-reasoning GO. The path: (a) a bigger/cleaner encyclopedic corpus (full Wikipedia / a dictionary with definitional glosses); (b) POS-based head-noun extraction (skip adjectives/participles cleanly, keep noun supers only); (c) possibly a code space that separates is-a categories better than raw co-occurrence. This confirms the arc's standing conclusion extends to the hierarchy frontier: taxonomic reasoning is a DATA (encyclopedic corpus) + extraction-quality lever, not a mechanism wall.

## Files
`research/runners/_realcorpus_isa_definitional_inheritance_derisk.py`; `research/findings/raw/_rc_isa_wiki.json`. Prior: the multi-level negative `2026-07-08-multilevel-taxonomy-generalization-over-real-corpus-clustering-NEGATIVE.md`; the single-level inheritance rungs (GO over co-occurrence categories).
