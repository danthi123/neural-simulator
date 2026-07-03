# EMERGE-64 — MINE the per-construction slot INVENTORY from the corpus (S1a) — GO (6-seed)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge64_mine_slot_inventory_derisk.py`
**CI:** `tests/test_emerge64_mine_slot_inventory.py` (9 tests, CPU/numpy, offline)
**Raw:** `research/findings/raw/_emerge64_mine_slot_inventory.json`
**Gate:** `research/findings/2026-07-03-self-organizing-grammatical-structure-research-gate.md` (RANK 3 / Move 3, S1a residual)
**Verdict:** **GO** — the last residual of the self-organizing-grammar chain (S1a, the per-frame slot INVENTORY) SELF-ORGANIZES from corpus experience. NO `sim/` edit; reuse-by-import; the gate-first no-confab moat is untouched.

---

## What this closes (S1a)

EMERGE-59's `FRAMES` dict (`_emerge59_spiking_broca_frame_slots_derisk.py:98-105`) is HOST-WRITTEN — it names, per frame, WHICH ordered typed slots the construction contains:
- `F_MODAL  = [(DET,the),(SUBJ,None),(FUNC,can),(VERB,bare)]`   → "the owl can fly"
- `F_INTR   = [(DET,the),(SUBJ,None),(VERB,3sg)]`               → "the penguin walks"
- `F_NEGMOD = [(DET,the),(SUBJ,None),(FUNC,does),(FUNC,not),(VERB,bare)]` → "the penguin does not fly"

EMERGE-62 discovered WHICH tokens are function words (the closed class, S2); EMERGE-63 learned the slot ORDER from corpus word-order precedence (S1b) — but **both still took the slot INVENTORY (which typed slots per frame) from this host dict.** EMERGE-64 MINES the inventory itself: for each construction TYPE, it LABELS each token's role from already-discovered signals (NO host FRAMES dict as input) and reconstructs the ordered `(role-type[, function-word][, inflection])` list. The `FRAMES` dict becomes the VALIDATION ground-truth only.

This is the `_bucketB` "mine the structure from corpus co-occurrence, render/recall through the composer" pattern (`_bucketB_corpus_mined_frames_derisk.py`, GO; the mined verb-frame lexicon `research/findings/raw/_bucketB_corpus_mined_frames.json`) applied to the SLOT INVENTORY, with the `_bucketB` **permuted-mining decisive control**. Grounding: Dominey-Hinaut (thematic roles read from the ORDER/POSITION of the CLOSED class; open vs closed separated on input); catalog **G.12** Broca open/closed dissociation; usage-based construction grammar (Tomasello/Goldberg — constructions abstracted from repeated exemplars).

---

## The mechanism (reuse-by-import; NO `sim/` edit)

**half 1 — LABEL each token's role from discovered signals** (`label_sentence`, per corpus sentence):
- FUNCTION-word slot ⇐ the token is in **EMERGE-62's DISCOVERED closed-class set** (`discover_closed_class`, reused). WHICH function word = its identity. DET vs FUNC is itself distributional (the determiner opens the NP and immediately precedes a content word — the/a; `can`/`does`/`not` are FUNC).
- CONTENT slot ⇐ the token is NOT in the discovered closed class (the open class, EMERGE-62's complement). **SUBJECT = the NP head right after the determiner** (first content word); **VERB = the clause-final content word** (the one the function words govern). Inflection tag (bare | 3sg) is read from the verb SURFACE (a trailing -s over the discovered content-verb lexeme = 3sg — the same morphology EMERGE-59's `emerge_v3` renders).
- A sentence with a content word neither the NP-head nor the clause-final verb is skipped (keeps the mine to clean single-clause constructions, like `_bucketB`).

**half 2 — GROUP exemplars into construction TYPES + reconstruct the inventory** (`mine_inventory`). The key design decision that makes the anti-cheat genuinely collapse: a construction is keyed by its **shuffle-invariant BAG of slots** (the sorted multiset of role-labels + function-word payloads); the construction's inventory is the **DOMINANT ordering** of that bag, kept only if attested ≥ `min_count` **AND** its dominant-order fraction ≥ `min_dominance` (0.80).
- Under the TRUE corpus, each construction's dominant-order fraction is **1.000** ("the owl can fly" is ALWAYS in that order).
- Under PERMUTED-MINING (each exemplar's word order shuffled first), the same bag's orderings scatter across permutations (dominant fraction **0.18–0.50**), so the multi-slot constructions fall below the dominance threshold and are NOT confidently mined → render collapses. This is the `_bucketB` "the corpus's word order, not the apparatus, carries the inventory."

**Feed into the producer.** The mined ordered slot lists REPLACE the host FRAMES dict via `MinedInventoryFrameSlotCQ` (subclass of EMERGE-63's `CorpusOrderFrameSlotCQ` over EMERGE-61's inter-utterance wash-out); the producer renders the frames ON SPIKES from the fully-mined structure. ADDITIVE / default-preserving: `mined_slots=None` is byte-behavior-identical to the template producer (EMERGE-59/61/63 untouched).

---

## Results (6 seeds 42/43/44/100/101/102, CPU/numpy)

| metric | value | gate |
|---|---|---|
| **mined-inventory accuracy** (exact slot recovery vs FRAMES) | **1.000** all seeds | ≥ 0.999 ✔ |
| **producer render exact-surface on spikes** | **1.000** all seeds | ≥ 0.999 ✔ |
| **PERMUTED-MINING / SHUFFLED-CORPUS** (must collapse) | **0.333** all seeds | ≤ main − 0.30 ✔ (margin 0.667) |
| **NO-CORPUS** (must be empty/chance) | **0.000**, empty=True all seeds | ✔ |
| **HELD-OUT-FRAME role-type backbone** (det<subj<verb generalizes) | **1.000** all seeds (F_MODAL 1.00 / F_INTR 1.00 / F_NEGMOD 1.00) | ≥ 0.999 ✔ |
| **gate-first MOAT** (producer calls on abstains) | **0** all seeds | == 0 ✔ |

**All three canonical frames render EXACT on spikes from the MINED (not host) inventory:**
```
you> can an owl fly?        broca> the owl can fly            [producer INVOKED]
you> can a penguin fly?     broca> the penguin walks          [producer INVOKED]
you> ... [deny]             broca> the penguin does not fly   [producer INVOKED]
you> can a zzz fly?         broca> I don't know.              [producer NOT invoked — moat]
```

**Permuted-mining collapse is genuine, not a metric artifact.** Under shuffle: F_MODAL (4 slots) and F_NEGMOD (5 slots) collapse to `found=False` (their dominant ordering falls below the 0.80 threshold); only F_INTR (3 slots) sometimes survives by chance (a 3-token shuffle has a higher chance of a dominant surviving order) → 0.333 = 1/3. The `_bucketB` decisive-control shape holds: the multi-slot constructions the producer depends on are destroyed.

---

## Honest boundary — the named residual (NOT a wall)

The MAIN arm (all frames' exemplars) mines every inventory EXACTLY. The HELD-OUT-FRAME arm generalizes the **shared det+subj+verb ROLE-TYPE backbone** to a fully-held-out frame (gated, 1.000). The **DISTINCTIVE** parts are the precisely-named residual — the SAME category as EMERGE-63's does<not residual, NOT forced into the GO:
- a held-out frame's **distinctive function-word slots** (F_MODAL's `can`; F_NEGMOD's `does`/`not`) are not recoverable if that frame is held out AND no other frame attests those function words in that position;
- **F_INTR's `3sg` verb inflection** is not recoverable when F_INTR is held out — only F_INTR attests 3sg (the other two frames are VERB:bare). Reported: `heldout_frame_inflection_recovered(train, "F_INTR") == False` all seeds (expected False).

**The next single signal** for the residual is one attestation of the held-out frame's own function word / inflection (or Yang-Getz's phrase-boundary-alignment cue, the gate's Move-2 3rd distributional property). Still not a wall.

**Subject-vs-verb disambiguation is honest, not a positional coincidence.** The SUBJ/VERB roles are assigned by position (first-content-word = NP head; clause-final content word = verb), and the permuted-mining control collapses precisely because that positional structure is destroyed under shuffle. The dominance-threshold mining (not signature-existence) is what makes the collapse load-bearing — an earlier signature-existence metric did NOT collapse under shuffle (the correct signature still appeared among the shuffle noise above `min_count`), and was corrected before claiming GO. The held-out tie-breaks are honest (role-type backbone, no template smuggling); no alphabetical/positional-coincidence artifact (EMERGE-63's flag) is relied on.

---

## Scope + composition

- HONEST SCOPE: mines the per-construction slot INVENTORY for the **BOUNDED** EMERGE frame domain (ability-affirm / intransitive-exception / negated-modal). It does NOT make the domain open-ended (open arbitrary generation, R4, is the separate deferred wall — the from-scratch spiking LM ~4 orders too small, `2026-05-07-Phase-2.3a-NEGATIVE`).
- BRAIN-BASED-ONLY compliant: the corpus mining is offline syllabus prep (like rendering a retinal image the neural retina reads); the inventory is rendered on REAL spikes (EMERGE-61 wash-out for position-independence). The gate-first moat is untouched (0 productions on abstains, by construction).
- **⇒ S1a self-organized.** With **S2** (EMERGE-62, function-word inventory + open/closed distinction) + **S1b** (EMERGE-63, slot ORDER) + **S1a** (EMERGE-64, slot INVENTORY), the WHOLE producer structure is now discovered from experience — the host FRAMES dict is fully removed as an INPUT. **EMERGE-65** composes S2+S1b+S1a end-to-end into a producer whose grammatical structure is emergent from raw experience.

## No-regression

- EMERGE-59 de-risk 6-seed **GO** (default de-risk byte-preserved); EMERGE-63 de-risk 6-seed **GO**; EMERGE-60/61/62/62b/63 CI **39/39 pass**; EMERGE-64 CI **9/9 pass**. NO `sim/` edit (verified `git status -- sim/` empty). Additive: `MinedInventoryFrameSlotCQ(mined_slots=None)` is behavior-identical to the base.
