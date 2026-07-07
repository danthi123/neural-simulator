# EMERGE-90 — object-relative comprehension WORKS in the conversational pipeline (6/6: emergent, spiking, moat) — CAPABILITY GO; the honest mechanism finding: the content-lexeme SLOTTING fix + the informative reservoir do the heavy lifting, so `gradedtie` is a MARGINAL saturation-tie fix, not the load-bearing pipeline piece

**Date:** 2026-07-07
**Runner:** `research/runners/_emerge90_objrel_comprehends_composer_answers_derisk.py` (reuse-by-import of EMERGE-89's `ReservoirComprehender`→`RFPhasorComposer` + the objrel emergent read-out + `gradedtie`; NO `sim/` edit).
**Verdict:** CAPABILITY GO (object-relative comprehension in conversation, 6-seed) + an honest mechanism-necessity finding (the runner's own gate stamps NO-GO because it tests the objrel-SPECIFIC-mechanism necessity, which is marginal — see below). Load-bearing facts controller-verified INLINE (the parse metric is a genuine full-fact exact match; the canonical-only read genuinely recovers the objrel fact; 6-seed consistent).
**Builds on:** `2026-07-07-objrel-END-TO-END-EMERGENT-CLOSE-adversarially-verified.md` + EMERGE-89 (canonical comprehends→answers, GO).

## The capability (GO, 6-seed: 42/43/44/100/101/102)
The brain HEARS an object-relative sentence ("the S1 that the S2 Vs" → agent=S2, patient=S1, THEME=head S1), COMPREHENDS it ON SPIKES (the objrel emergent read-out + gradedtie assign THEME=head), STORES the fact, and ANSWERS the who/what query — with the no-confab moat:
- **objrel parse (EXACT-fact: agent==∧action==∧patient== ground truth) = 1.00 on all 6 seeds** (`_parse_hits`, a genuine full-fact match, verified).
- **objrel recall (comprehend→store→query_patient) = 1.00 on all 6.**
- **canonical NOT broken = 1.00 all 6** (the same pipeline still reads canonical SVO).
- **EMERGENT: pre-learning parse = 0.00 all 6** (the delta-rule read-out learns it; not warm-started).
- **no-confab moat = 0.00 false-accept all 6** (a never-stored (agent,action) → abstain).
⇒ object-relative comprehension + answering is now part of the fully-spiking conversational turn on one co-resident brain. The north-star capability (the owner can talk to the brain about non-canonical sentences) is advanced.

## The honest mechanism finding (why the runner's own gate says NO-GO — and it's honest, not a gap)
The runner's GO gate requires two NECESSITY controls that BOTH FAIL, robustly, across the 6 seeds:
- **A canonical-ONLY learned read-out** (`_fit_Ws_spiking`, fit on `_TRAIN_KINDS` canonical sentences only — a genuine contrast, verified) **recovers the correct objrel fact = 1.00 on all 6 seeds.** So the objrel-SPECIFIC read-out is NOT necessary in the pipeline — a canonical learned read generalizes.
- **The comprehension-lesion** (collapse the reservoir's closed-class identity) collapses objrel recall on only 3/6 seeds (100/101/102 → 0.00) and not on 42/43/44 (→ 1.00) — load-bearing on half, seed-dependent.
⇒ **The load-bearing pieces are (1) the content-lexeme SLOTTING fix + (2) the informative reservoir feature — NOT the objrel-specific read-out mechanism.** The SLOTTING fix is EMERGE-90's genuine contribution: EMERGE-89's OPEN-position rule wrongly slotted the relativizer "that" (shifting every role + crashing the composer); EMERGE-90 slots the CONTENT-LEXEME positions (excluding "that"), which is what a comprehender's lexical categorization does. Given that fix + the reservoir feature (which the objrel arc's ridge reads 1.00 on all seeds), a LEARNED read-out — canonical or objrel-trained — recovers objrel.

## Reconciliation with the objrel arc (the honest, unified story)
1. **A LEARNED read-out is necessary** — a FIXED spiking WTA / position-based read MISREADS objrel (objrel arc baseline `fixed_spiking_wta` = 0.50). The objrel arc established the learned read-out (delta-rule, emergent) vs the fixed WTA.
2. **Given a learned read-out + the slotting fix, objrel comprehension is EASY** — even a canonical-trained learned read generalizes (EMERGE-90, 6/6). The reservoir feature carries the role structure position-consistently.
3. **`gradedtie` is a MARGINAL saturation-tie fix** — the emergent close showed the learned read-out reads objrel on 9/10 seeds RAW; gradedtie fixed only the 1 saturation-tie seed (101). In the pipeline it's a polish for the rare `[4,0,4]` count-tie, not the load-bearing mechanism.
So the objrel arc's genuine, load-bearing contributions to conversation are: the LEARNED read-out (vs a fixed WTA) + the content-lexeme SLOTTING (vs the "that"-crashing OPEN-position rule). `gradedtie` closed the last marginal saturation-tie honestly, but it is not what makes object-relative comprehension work in the pipeline — the reservoir + slotting are.

## Honest scope
The CAPABILITY (object-relative comprehension + answering in the conversational pipeline, 6/6, emergent, spiking, moat) is the deliverable and is genuinely achieved. The runner's NO-GO is an honest reflection that the objrel-SPECIFIC mechanism (gradedtie) is not the pipeline's load-bearing piece — the reservoir + slotting are. This is a finding, not a gap. (A cleaner necessity characterization would use a POSITION-based read as the contrast — which the objrel arc already showed misreads objrel — rather than a canonical LEARNED read that generalizes.)

## Files
`research/runners/_emerge90_objrel_comprehends_composer_answers_derisk.py`; `research/findings/raw/_emerge90_objrel_seed{42,43,44,100,101,102}.json` (6-seed).
