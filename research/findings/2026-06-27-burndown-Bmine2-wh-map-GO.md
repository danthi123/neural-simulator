# Burndown Bucket-B B-mine-2 — the wh→role MAP as the INVERSE INDEX of the corpus-mined verb-frames — **GO** (structure ACQUIRED, not given)

**Date:** 2026-06-27
**Type:** cheap-first DE-RISK (gated by `2026-06-27-burndown-bucketB-structure-learning-research-gate.md`, the B-mine-2 corollary — "the wh→role map falls out nearly free as the INVERSE INDEX of the mined frames"). Reuse-by-import + an ADDITIVE `wh_question_parser.py` kwarg; **NO `sim/` edit, NO composer edit.**
**Verdict:** **GO — 6 seeds (42–47).** The second hand-authored Bucket-B relational table (the wh→role map `WH_ROLE_CANDIDATES`) is **DERIVED as the INVERSE INDEX of the B-mine-1 corpus-mined verb-frames** over the brain's OWN learned verbs and **MATCHES the hand `WH_ROLE_CANDIDATES` on the validated wh-words**; the wh-parser's parse/answer on the MINED map == on the hand map (parse-parity **1.000**, 6/6 seeds); and the decisive **PERMUTED-MINING control collapses (mean 0.250, 6 seeds)** — the mined frames, not the apparatus, carry the wh-map. This is the literal **inverse index** of B-mine-1 (`2026-06-27-burndown-Bmine1-corpus-mined-frames-GO.md`).

> **Why this is the near-free corollary.** B-mine-1 mined the verb-frame LEXICON (which typed roles each verb licenses) from corpus argument co-occurrence. A wh-question is a filler-gap dependency: `where` gaps a SPATIAL role, `what` gaps an ENTITY/THEME role, `who` gaps an ANIMATE-PARTICIPANT role. So the wh→role map is the **inverse index of the frame lexicon** — collect, per wh-word, which roles fall in its role-class, ordered by corpus attestation. The gate's find held exactly: the only NEW input is a SMALL CLOSED wh-word→role-CLASS affinity (the same kind of closed grammatical lexicon as B-mine-1's preposition→role table — NOT a per-verb hand list); the concrete candidates + their order come entirely from the mined frames.

---

## 1. What was built (the inverse-index converter + one additive wh-parser kwarg)

`research/runners/_bucketB_corpus_mined_wh_map_derisk.py` + CI guard `tests/test_bucketB_corpus_mined_wh_map.py` (11 tests, CPU). The deliverable is a MINED `WH_ROLE_CANDIDATES` (+ `WH_MULTIWORD`); the wh-parser uses it **behind a flag** (the hand map is retained as the parity oracle / default).

- **half 1 — the verb-frame role inventory (== B-mine-1).** The B-mine-1 mined `FRAME_LEXICON` gives, per verb, the set of typed roles the verb LICENSES (each corpus-attested), plus the per-role corpus attestation total (the sum of the role's slot counts across verbs, read from B-mine-1's provenance). This is the brain's ACQUIRED structure — the INPUT.
- **half 2 — INVERT it.** `derive_wh_role_map(mined_frames, attest_count)` builds the wh-map:
  - `WH_ROLE_CLASS` — the ONLY hand input: a small CLOSED lexicon mapping each wh-word to a thematic ROLE-CLASS (`who`→PARTICIPANT, `what`→ENTITY, `where`→SPATIAL, `when`→TEMPORAL, `whom`→RECIPIENT-class, `with`→INSTRUMENT-class). `ROLE_CLASS` is the dual (a role's class membership; RECIPIENT belongs to BOTH PARTICIPANT — a `who` candidate — and RECIPIENT-class — a `whom` candidate). This is a closed grammatical fact (a wh-word's selectional class), the exact analogue of B-mine-1's `PREP_ROLE`.
  - For each wh-word, gather the mined roles whose class intersects the wh-word's class, and ORDER them by the **inverse index**: CORE roles (agent/patient — the obligatory subject/object gap) first, then by **descending corpus attestation** (the most-attested role is the dominant candidate). SOURCE is reserved for the where-from multiword (not a bare-`where` candidate).
  - The MULTIWORD cues (`where from`→SOURCE, `with what`→INSTRUMENT, `to whom`→RECIPIENT) are derived from `PREP_ROLE` (the content preposition's role), with `to whom` resolved to the dative RECIPIENT (the same `to` GOAL↔RECIPIENT disambiguation B-mine-1 uses).
- **the additive wh-parser kwarg** — `wh_question_parser.py` gained ONE additive parameter, `frame_roles=` (threaded through `_resolve_wh_role` / `parse_wh_question` / `answer_wh`; default `None` → the module hand `FRAME_ROLES`, **byte-identical** — every existing caller + the production path is unchanged, verified by the existing Tier-0.3 test passing verbatim). The wh-parser already threaded `role_map=` (the existing permuted-mapping seam); `frame_roles=` lets the resolution intersect the wh-candidates against the MINED (or scrambled) per-verb licensing inventory. This is what makes the wh-map resolve against the ACQUIRED frames AND what makes the permuted-mining control bite.

Corpus: **TinyStories** (child-directed-speech-like). Brain: `brainALL_w7000.npz_seed42` (the brain's learned vocab gates which verbs are mineable). The same artifacts as B-mine-1.

---

## 2. The MINED wh-map vs the hand one (the match-or-justify result)

The mined frames (152 verbs cleared attestation) invert to the wh-map. On the **validated wh-words** (the hand `WH_ROLE_CANDIDATES` single-word keys):

| wh-word | MINED candidates | HAND candidates | status | corpus evidence (inverse index) |
|---|---|---|---|---|
| **who** | agent, RECIPIENT | agent, RECIPIENT | ✅ MATCH | agent CORE (152 verbs); RECIPIENT oblique (1230 attest, 5 verbs: give/bring/carry/tie) |
| **what** | patient, THEME | patient, THEME | ✅ MATCH | patient CORE (14181 attest, 62 verbs); THEME (810, 5 verbs) |
| **where** | GOAL, LOCATION | GOAL, LOCATION | ✅ MATCH | **GOAL (3437) out-attests LOCATION (2819)** → GOAL first, matching the hand order |
| **whom** | RECIPIENT | RECIPIENT | ✅ MATCH | RECIPIENT (1230, give/bring/carry/tie) |
| **with** | INSTRUMENT | INSTRUMENT | ✅ MATCH | INSTRUMENT (1078, fill/play) |
| **when** | — (empty) | TIME | ⚠️ DIFFER — **corpus-JUSTIFIED** | TinyStories attests NO temporal-PP argument in the mined frames → no TIME slot to license `when` |
| **MULTIWORD** | where-from→SOURCE, to-whom→RECIPIENT, with-what→INSTRUMENT | (same) | ✅ MATCH | from→SOURCE (702 attest, 83 verbs); to-dative→RECIPIENT; with→INSTRUMENT |

**5/5 validated single-word wh-words MATCH** the hand map exactly, AND the multiword table matches. The one DIFFER (`when`) is **corpus-justified** (the corpus simply does not attest temporal arguments — a property of the input, not a wrong derivation; the wh-parser already abstains on an unlicensed `when` via the moat) → **0 unjustified differences.**

**The decisive ordering recovery: `where`'s GOAL-first.** GOAL is licensed by FEWER verbs (8) than LOCATION (11), so a naïve verb-COUNT ranking would order LOCATION first (and an early iteration of this de-risk did exactly that — an UNJUSTIFIED difference). The fix is the faithful inverse-index weight: the **corpus ATTESTATION** (GOAL 3437 ≫ LOCATION 2819 — motion verbs fire GOAL far more often than the few locative verbs fire LOCATION) reproduces the hand order. The order encodes which gap is the *default reading* of `where`, and that is an attestation-frequency fact.

---

## 3. The decisive evidence — parse parity + the PERMUTED-MINING control

### 3a. PARSE PARITY — the gate's headline questions answer byte-identically on the MINED map

`answer_wh(comp, q, role_map=MINED, frame_roles=MINED_FRAME_ROLES)` == the hand-map answer for the validated questions:

```
"where does the boy go?"     -> role=GOAL      filler=park    (hand: GOAL/park,      match True)
"what does the mom give?"    -> role=THEME     filler=ball    (hand: THEME/ball,     match True)
"who does the girl give to?" -> role=RECIPIENT filler=cat     (hand: RECIPIENT/cat,  match True)
```

parse-parity **1.000** (every validated question's gapped role + filler matches the hand-map answer), 6/6 seeds.

### 3b. ⭐ PERMUTED-MINING (the decisive control, mirror B-mine-1) — collapses to 0.25

SCRAMBLE the mined-frame INPUT (assign each verb a RANDOM other-verb role inventory), then RE-DERIVE the wh-map AND the per-verb `FRAME_ROLES`. The wh-parser resolves the gapped role by **intersecting the wh-candidates with the verb's licensed roles**, so when `go`'s frame is scrambled to (say) `give`'s `[THEME, RECIPIENT]`, then `"where does the boy go?"` → `where`=[GOAL,LOCATION] ∩ scrambled-go={THEME,RECIPIENT} = ∅ → **abstain**. Result: **permuted-mining acc mean 0.250 over 6 seeds** (per-seed 0.17–0.33; ≤ 0.5 required, AND ≥ 0.4 below the mined-acc of 1.00). ⇒ **the corpus-mined frame inventory, NOT the apparatus (the composer, the parser, the affinity lexicon), carries the wh-map** — exactly B-mine-1's permuted-mining logic. (The composer still holds the CORRECT facts, so the collapse is the wh-MAP failing to resolve, not the facts breaking.)

### 3c. The full anti-cheat bar — every control passes

| control | result | required | verdict |
|---|---|---|---|
| MINED wh-map **match-or-justify** the hand map (validated wh-words) | 5 match, 1 corpus-justified (`when`), 0 unjustified; multiword match | 0 unjustified | ✅ |
| **PARSE PARITY** (parse + answer on mined == hand) | parse-parity **1.00** (6/6 seeds) | answer-identical | ✅ |
| ⭐ **PERMUTED-MINING** (scrambled frame inventory) | **mean 0.250** (6 seeds) | ≤ 0.5 AND ≥ 0.4 below mined | ✅ |
| **no-confab MOAT** (unstored agent / unstored pair / unlicensed wh → None) | abstains | 0-FA | ✅ |
| **PROVENANCE** (every mined wh-candidate backed by a corpus-attested licensing frame) | asserted | attested | ✅ |
| **byte-identity** (the additive `frame_roles=` default == hand; existing Tier-0.3 test passes verbatim) | holds | unchanged default | ✅ |

---

## 4. Honest scope, caveats, residuals

- **What this is:** the wh→role map — the second hand-authored Bucket-B relational table — **derived as the INVERSE INDEX of the B-mine-1 corpus-mined verb-frames** over the brain's OWN learned verbs, verified to match the hand `WH_ROLE_CANDIDATES` on the validated wh-words, with the wh-parser resolving through the MINED map at parity, gated on the **permuted-mining** collapse. **Structure ACQUIRED, not given** — the near-free corollary of B-mine-1. Reuse-by-import + ONE additive wh-parser kwarg (default-off byte-identical); **NO `sim/` edit, NO composer edit.**
- **The ONE new hand input is a closed grammatical lexicon, not a per-verb list.** `WH_ROLE_CLASS` maps each wh-word to a thematic ROLE-CLASS — a closed fact about the language's wh-words (a wh-word's selectional class), the exact analogue of B-mine-1's `PREP_ROLE` (a preposition's dominant role). The concrete candidates AND their ordering are entirely DERIVED from the mined frames; the affinity lexicon does not encode which roles any verb takes.
- **The mining is corpus-budget- and vocab-dependent (the same honest scope as B-mine-1).** `when→[]` is corpus-justified-differ because TinyStories does not attest temporal-PP arguments (a property of the input); the believability is the **signature** (match-or-justify on the attested wh-words + the permuted-mining collapse + provenance), NOT 100% coverage of role-classes the corpus barely states. The lever is the same as B-mine-1: a richer corpus that attests temporal/instrumental arguments.
- **Frame USAGE in production:** the mined wh-map is a validated drop-in — a caller passes `role_map=MINED_WH, frame_roles=MINED_FRAME_ROLES` to `answer_wh`/`parse_wh_question` (the production drop-in answers `where`/`what`/`who...to` correctly and abstains on `when`, moat intact). Wiring it as the default in the console/agent (vs. keeping the hand map as the oracle) is a separate deployment pass, deliberately not flipped here (the hand map stays the parity oracle, exactly as B-mine-1 kept its hand frames).
- **What B-mine-2 does NOT touch (the gate's deferred items):** the closed-class / morphology lexicons (a finite function-word class — the hard part is the recursive grammar, not the list); the tag-VALUE inference frontiers (common-ground / tense); and the genuine months-frontier (a learned RECURSIVE generative grammar + the developmental self-organization of the binding connectivity). Those remain correctly deferred.

---

## 5. Reproduce

```bash
# the de-risk (CPU/numpy; spaCy parses TinyStories once, then 6 seeds of parse parity + permuted-mining)
SIM_BACKEND=numpy python -m research.runners._bucketB_corpus_mined_wh_map_derisk --seeds 42 43 44 45 46 47

# the CI regression guard (CPU; synthetic-frame inverse-index logic is instant, + a corpus-gated end-to-end test)
SIM_BACKEND=numpy python -m pytest tests/test_bucketB_corpus_mined_wh_map.py -v
```

Raw: `research/findings/raw/_bucketB_corpus_mined_wh_map.json` (the mined wh-map, multiword, provenance, per-seed parity + permuted-mining).

---

## 6. Bottom line

The gate's corollary held: **the wh→role map falls out nearly free as the INVERSE INDEX of the corpus-mined verb-frames.** `WH_ROLE_CANDIDATES` is **DERIVED** by inverting the B-mine-1 mined frames (a wh-word gaps a role-CLASS via a small closed affinity lexicon; the mined frames say WHICH roles + in what attestation order); it **MATCHES the hand map on the validated wh-words** (5/5 + the multiword, the one differ — `when` — corpus-justified, 0 unjustified); the wh-parser's parse/answer on the MINED map == on the hand map (`"where does the boy go?"→GOAL=park`; `"what does the mom give?"→THEME=ball`); and the **PERMUTED-MINING control collapses (mean 0.250, 6 seeds)**, proving the mined frames — not the apparatus — carry the wh-map. **Structure ACQUIRED, not given**, the no-confab moat 0-FA, NO `sim/` edit.
