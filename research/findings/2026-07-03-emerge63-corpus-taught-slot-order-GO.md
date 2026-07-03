# EMERGE-63 — corpus-taught slot ORDER for the spiking-Broca producer: **GO** (S1b self-organized), 6-seed

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge63_corpus_taught_slot_order_derisk.py`
**Test:** `tests/test_emerge63_corpus_taught_slot_order.py` (9 tests, CPU/numpy, offline)
**Raw:** `research/findings/raw/_emerge63_corpus_taught_slot_order.json`
**Verdict:** **GO** — the spiking-Broca producer's per-frame slot ORDER (S1b) SELF-ORGANIZES from the corpus's actual word-order statistics; the host template order-teacher is removed. NO `sim/` edit; the gate-first no-confab moat is intact.

---

## What this closes

The self-organizing-grammar research gate (`2026-07-03-self-organizing-grammatical-structure-research-gate.md`, RANK 2 / Move 3) named residual **S1b** — the per-frame slot ORDER — with the exact prescription: *"swap FrameCQ's order-teacher to corpus n-gram statistics — near-free."* Dominey-Hinaut: grammar (the thematic-role order) is learned from corpus sequence statistics, without explicit rules; catalog G.12 (Broca); usage-based construction grammar (Tomasello).

**The host residual removed.** EMERGE-59's `FrameSlotCQ._teach_order` (`_emerge59_spiking_broca_frame_slots_derisk.py:239-260`) wrote each frame's per-slot primacy from the FRAMES dict's **host-written slot ORDER**: the teacher is `LR * (n-1-pool)` over the template pool index — i.e. *"the template says slot i is i-th."* EMERGE-63 replaces that teacher with one derived **purely from the observed word order** in corpus example sentences. **The FRAMES dict's slot ORDER is now only the VALIDATION ground-truth, not the teacher.**

**What stays (S1a, EMERGE-64's separate residual — stated clearly).** EMERGE-63 learns only the ORDER of the (given) slots. The slot TYPES per frame (`det / subj / func:can / func:does / func:not / verb` — which typed slots a construction licenses) are still taken from the FRAMES template. The type labels are also used to *locate* each role's token in a corpus sentence (DET→`the`/`a`, FUNC:x→token `x`, SUBJ→content noun via the `_SUBJECTS` lexicon, VERB→content verb via `_VERBS`) — this uses S1a (which roles + their lexical class), **not the order**; the order is read only from the observed token positions. Discovering the slot INVENTORY from the corpus is EMERGE-64.

## The mechanism

For each frame TYPE, collect its example sentences from the corpus stream (reuse EMERGE-62's `build_stream` + sentence segmentation on `SENT_PERIOD`), locate each role's token position, and compute a **pairwise precedence / bigram-order statistic over the slot ROLES**: `prec[A][B]` = fraction of examples where role A's token precedes role B's (det<subj, subj<func, func<verb, …). A role that precedes many others gets **high primacy → emitted first**. That precedence-derived order **reorders the EMERGE-59 frame slots**; a `CorpusOrderFrameSlotCQ` (subclass of the EMERGE-61 `ResetFrameSlotCQ` wash-out) teaches a plain descending primacy over the reordered slots and renders **ON SPIKES** on a real `SimulationBridge` — the learned primacy gradient = graded external current → the per-pool spiking-RATE ranking = the emission order (EMERGE-61 inter-utterance wash-out for position-independence). No host template order enters the teacher.

Role keys are canonical: shared roles by TYPE (`DET`/`SUBJ`/`VERB`) so they transfer across frames (held-out); FUNC keyed by its payload lemma (`FUNC:can`/`FUNC:does`/`FUNC:not`) since function words are frame-specific.

## Results (6-seed 42/43/44/100/101/102, CPU)

| metric | value | reading |
|---|---|---|
| **main order** | **1.000** | corpus-taught order == template ground-truth, all frames |
| **main exact-surface** (on spikes) | **1.000** | renders "the owl can fly" / "the penguin walks" / "the penguin does not fly" exactly |
| SHUFFLED-CORPUS order | 0.282 | ✅ collapses (margin ≥ 0.30) |
| NO-CORPUS order | 0.228 | ✅ chance |
| HELD-OUT shared (F_MODAL, F_INTR) | **1.000** | ✅ shared type-level order generalizes to a fully-held-out frame |
| HELD-OUT F_NEGMOD | 0.775 | honest residual (below — the named `does<not` gap) |
| moat calls on abstain | **0** | ✅ gate-first, producer never invoked on abstains |

Per-seed main order/exact were 1.000/1.000 on every seed; held-out F_NEGMOD ranged 0.65–0.85 across seeds. Elapsed ~9 s.

**Main includes the negated-modal's `does<not`.** When all frames' exemplars are available, `does<not` is DIRECTLY attested in F_NEGMOD's own sentences → precedence resolves it, so main F_NEGMOD is 1.000 **even with an honest random tie-break** (not the template). The main GO is not smuggling the template order.

## The honest residual (named, NOT a wall)

The **`does`-vs-`not` INTERNAL order of a HELD-OUT multi-function-word frame** is genuinely NOT learnable from the OTHER two frames alone — only F_NEGMOD attests two adjacent function words, so nothing in F_MODAL/F_INTR constrains their relative order. With an honest (non-template) tie-break the held-out F_NEGMOD sits at ~0.775 (the shared `det<subj<…<verb` roles order correctly; only the two-func internal order is chance). This is exactly the ambiguity the research gate flagged. **It is NOT a wall — the next single signal is one attestation of the `does<not` bigram** (any held-out slice that keeps even a few F_NEGMOD exemplars closes it), or Yang-Getz's phrase-boundary-alignment cue. We report held-out generalization on the SHARED precedences (the gated claim) and name this within-frame residual explicitly rather than forcing a GO on it.

A methodological note worth keeping: the honest tie-break was load-bearing. With a `template` tie-break the held-out F_NEGMOD reads 1.00, and even an `alpha` tie-break reads 1.00 (because `does`<`not` alphabetically coincides with the truth) — both would have falsely claimed full generalization. Only the `random` tie-break exposes the true residual (F_MODAL/F_INTR 1.00, F_NEGMOD 0.68). EMERGE-63 uses the honest random tie-break for the held-out arm and for the main arm.

## Anti-cheats (all collapse as required)

- **SHUFFLED-CORPUS** (scramble each example sentence's word order): precedence destroyed → 0.282 (main 1.000). **The load-bearing control** — the order genuinely comes from the corpus WORD ORDER, not elsewhere.
- **NO-CORPUS** (no example sentences): no precedence → 0.228 (chance).
- **HELD-OUT-FRAME**: shared type-level order generalizes (F_MODAL, F_INTR = 1.000); the does<not residual is honestly present.
- **MOAT**: 0 producer invocations on abstains (gate-first, by construction).

## Sample transcript (corpus-taught order, on spikes)

```
you> can an owl fly?              broca> the owl can fly            [producer INVOKED]
you> can a penguin fly?           broca> the penguin walks         [producer INVOKED]
you> can a penguin fly? [deny]    broca> the penguin does not fly  [producer INVOKED]
you> can a zzz fly?               broca> I don't know.             [producer NOT invoked]  ← moat
```

## Implementation notes / scope

- **Reuse-by-import; NO `sim/` edit.** `CorpusOrderFrameSlotCQ` subclasses EMERGE-61's `ResetFrameSlotCQ`; EMERGE-59/60/61 are untouched. With `corpus_order=None` the subclass is behavior-identical to the base (template order) — the additive/default-preserving property (test `test_corpus_order_none_preserves_template_behavior`).
- **One runner-side, behavior-neutral adjustment (not a `sim/` edit).** The slot bridge's structural plasticity (`enable_structural_plasticity`, default on) grows spurious synapses on the inert `_anchor` region over the ~24 renders this de-risk runs, resizing the STP arrays and breaking EMERGE-61's fixed-shape wash-out snapshot on some seeds. `CorpusOrderFrameSlotCQ.__init__` sets `self.bridge.core_config.enable_structural_plasticity = False` on its OWN bridge (then re-snapshots). Verified behavior-neutral: the slot pools have `internal_density=0.0` (no incoming synapses; driven purely by external current), so slot-pool rates are **bit-identical** with/without structural plasticity — the growth is on the inert anchor and cannot reach the read-out. EMERGE-61's own de-risk (short 5-emit sequence) never triggered the growth, so it did not need this; it stays untouched and still GO 6-seed.
- **Honest scope.** S1b for the BOUNDED EMERGE frame domain. Not open-ended generation (R4, the separate deferred wall). S1a (which slots a frame licenses) stays template-supplied — EMERGE-64. The order is produced on REAL spikes; the corpus stat is offline syllabus prep (BRAIN-BASED-ONLY compliant, like rendering a retinal image the neural retina reads).

## Regression

EMERGE-59/60/61/62/62b all still pass (defaults preserved): `test_emerge59` (…), `test_emerge61` 6-seed GO, `test_emerge62`/`test_emerge62b` GO, `test_emerge60` 6 pass, `test_emerge63` 9 pass. EMERGE-62 runner re-verified GO; EMERGE-61 runner re-verified GO 6-seed.

## Composition

EMERGE-63 (corpus-taught ORDER, S1b) composes with EMERGE-62 (discovered function-word SET, S2) and the pending EMERGE-64 (mined slot INVENTORY, S1a) into EMERGE-65 (the fully-self-organized spiking-Broca producer). Two of the three producer residuals (S2, S1b) are now self-organized from distributional experience.

## Sources

- Dominey & Hinaut, "Real-Time Parallel Processing of Grammatical Structure in the Fronto-Striatal System" (PLoS ONE 2013); Hinaut & Dominey, "Self-Organized Artificial Grammar Learning in Spiking Neural Networks" — thematic roles read from the ORDER/POSITION of elements; grammar learned from corpus, no explicit rules; generalizes.
- Tomasello usage-based / construction grammar; Bybee frequency effects — constructions abstracted from repeated exemplars (the slot-order is the statistics of high-frequency exemplar sequences).
- Catalog G.12 (feature-catalog.md:2774-2784; Kandel 6e Ch 55 pp 1382-1384) — Broca agrammatism.
- Research gate: `research/findings/2026-07-03-self-organizing-grammatical-structure-research-gate.md` (RANK 2 / Move 3).
- Project precedents: `_emerge59_spiking_broca_frame_slots_derisk.py` (FRAMES, FrameSlotCQ, `_teach_order:239-260`), `_emerge61_spiking_broca_order_robustness_derisk.py` (ResetFrameSlotCQ wash-out), `_emerge62_discover_function_words_derisk.py` (build_stream + segmentation + lexicons), `song_g1_core.score_order`.
