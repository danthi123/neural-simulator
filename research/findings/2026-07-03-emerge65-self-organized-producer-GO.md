# EMERGE-65 (THE CAPSTONE) — the FULLY-SELF-ORGANIZED spiking-Broca producer: the WHOLE grammatical structure discovered from the corpus END-TO-END — GO (6-seed)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge65_self_organized_producer_derisk.py`
**CI:** `tests/test_emerge65_self_organized_producer.py` (8 tests, CPU/numpy, offline, ~4s)
**Raw:** `research/findings/raw/_emerge65_self_organized_producer.json`
**Verdict:** **GO, 6-seed unanimous (42/43/44/100/101/102).** Reuse-by-import; **NO `sim/` edit**; the gate-first no-confab moat is untouched.

---

## Headline

**The spiking-Broca producer's ENTIRE grammatical structure is now self-organized from the corpus, end-to-end.** ONE pipeline (`SelfOrganizedProducer`) takes ONLY the corpus token stream and composes the three GO self-organized pieces — S2 (function words, EMERGE-62), S1a (slot inventory, EMERGE-64), S1b (slot order, EMERGE-63) — into a producer whose FRAMES-equivalent is BUILT from statistics, then renders the EMERGE answers ("the owl can fly" / "the penguin walks" / "the penguin does not fly") EXACT on spikes behind the gate-first moat. The host `FRAMES` dict + `FUNCTION_WORDS` list + template slot order are NONE of them inputs — validation ground-truth ONLY.

This closes RANK-4 (the integration) of the self-organizing-grammatical-structure research gate (`2026-07-03-self-organizing-grammatical-structure-research-gate.md`, MOVE 3 / EMERGE-65). It is a **composition of GO pieces (not a new mechanism)**, so the load-bearing proof is the COMPOSED anti-cheat: the permuted-corpus control collapses the two MULTI-SLOT constructions (F_MODAL, F_NEGMOD → 0), proving their structure is corpus-order-derived, not host-smuggled (the shortest F_INTR is a deterministically-reconstructed named residual — see the audit-corrected reading below + the EMERGE-64b follow-on). *This finding was corrected by the 2026-07-03 EMERGE-62..66 adversarial audit, which downgraded two overstated claims — the "whole-pipeline collapse" framing here and a tautological held-out-backbone GO-gate metric — without overturning the GO; see "Audit remediation" below.*

---

## The composed pipeline (`SelfOrganizedProducer.build_from_corpus`, from the corpus stream alone)

1. **(a) S2 — discover the FUNCTION-WORD inventory.** EMERGE-62 frequency + context-coverage Goldilocks discovery (freq-pct ≥ TF_PCT AND cover-pct ≥ TC_PCT, FIXED/pre-registered). Discovers `{a, and, big, can, cat, does, fast, in, is, it, not, on, the, to}` (seed 42) — all four frame function words `{the, can, does, not}` recovered.
2. **(b) S1a — mine each construction's ordered slot INVENTORY**, using (a)'s discovered function words to split closed vs open (EMERGE-64 `label_sentence` + `mine_inventory`, NO host FRAMES dict). `label_sentence` preserves token order → the mined inventory is already in corpus order. 12 construction signatures mined; the three EMERGE frames recovered exactly.
3. **(c) S1b — learn the slot ORDER** from pairwise role precedence over (b)'s mined constructions (EMERGE-63). Belt-and-suspenders: the order is (re)derived from precedence over the corpus word order — the SHUFFLED-CORPUS control breaks BOTH mining AND order, so neither is host-smuggled.
4. **(d) ASSEMBLE the per-frame structure** — the discovered FRAMES-equivalent (`mined_slots` + `corpus_order` + `discovered_function_words`), matched to the EMERGE frame ids ONLY by the frame-selection routing (`decision_from_emerge`'s polarity/negated-modal), NOT by reading the host FRAMES.
5. **(e) FEED the EMERGE-59/61 spiking producer** (`MinedInventoryFrameSlotCQ` over the EMERGE-61 inter-utterance wash-out) + the gate-first `BrocaProducer` moat → render ON SPIKES (the learned primacy gradient = graded current → the per-pool spiking-RATE ranking = the emission order). GATE-FIRST: abstain → the producer is NEVER invoked.

---

## 6-seed results (CPU/numpy)

| metric | value (all 6 seeds identical unless noted) | meaning |
|---|---|---|
| **end-to-end render (a)** | **1.000** | assembled-from-corpus structure renders the 3 EMERGE surfaces EXACT on spikes |
| **assembled-structure match (b)** | **1.000** | mined inventory + learned order == host FRAMES (slot set + function words + order) |
| inventory accuracy | 1.000 | S1a recovers all 3 frames' ordered typed-slot lists exactly |
| all frame function words discovered | True | S2 recovers `{the, can, does, not}` |
| **PERMUTED-CORPUS render (c)** | **0.333** | composed anti-cheat: scrambling word order at BOTH mining + order stages collapses the two MULTI-SLOT frames (F_MODAL, F_NEGMOD → 0); the 0.333 floor is F_INTR alone, a named residual (margin 0.667 ≥ 0.30) |
| **PERMUTED-CORPUS struct-match (c)** | **0.333** | the two multi-slot frames' structure collapses — corpus-order-derived, not host-smuggled (F_INTR residual per the audit note) |
| NO-CORPUS render | 0.000 (empty inventory) | no data → no structure → nothing |
| held-out shared order (c) — **GATED, genuine** | 1.000 | a held-out frame's shared type-level order (FUNC position) is recovered from the OTHER frames (tie-break-invariant over 200 draws) |
| held-out shared backbone (c) — **REPORTED, not gated** | 1.000 | det+subj+verb backbone; a LANGUAGE-UNIVERSAL constant all 3 frames share → a "control that cannot fail", so dropped from the GO gate per the audit (reported for transparency) |
| **moat (d)** | **0** producer-calls on abstains | gate-first no-confab moat holds by construction |

**Sample transcript (seed 42, on spikes, from the fully-self-organized structure):**
```
you> can an owl fly?          broca> the owl can fly           [INHERIT; producer INVOKED]
you> can a penguin fly?       broca> the penguin walks         [CANCEL;  producer INVOKED]
you> can a penguin fly? [deny] broca> the penguin does not fly [DENY;    producer INVOKED]
you> can a zzz fly?           broca> I don't know.             [MOAT;    producer NOT invoked]
```

The permuted-corpus floor of **exactly 0.333** is F_INTR alone (**NOT a chance floor** — corrected per the EMERGE-62..66 adversarial audit, 2026-07-03): under a scrambled corpus the two MULTI-SLOT constructions (F_MODAL, F_NEGMOD) **genuinely collapse to 0** (their orderings scatter below the dominance threshold → they fail to mine/render → their structure IS proven corpus-ORDER-derived), while the shortest construction (F_INTR "the penguin walks", det+subj+verb) is **deterministically reconstructed at dominance 1.0 on every seed and every shuffle** — its self-identifying determiner (`the` at NP-onset) + the shuffle-*variant* bag-keying (a non-onset `the` is re-labelled FUNC → a different bag, so the "wrong" orderings never dilute the F_INTR bag) mean the ~1/3 of shuffles that keep `the` at position 0 reconstruct F_INTR's inventory in a clean bag **without needing word order**. So this control genuinely proves the ORDER-derivation of the two multi-slot frames, but does NOT separately prove F_INTR's order — a **named residual** (F_INTR's inventory is corpus-derived; its order is not proven by *this* control). The **EMERGE-64b** control-strengthening follow-on (key bags by a shuffle-*invariant* token multiset) makes F_INTR collapse too, driving the floor toward 0.0. The GO bar (main − perm ≥ 0.30 margin) is met with headroom (0.667) on the genuine two-frame collapse.

---

## The COMPOSED anti-cheat is the load-bearing proof

Because EMERGE-65 is a composition of already-GO pieces, the decisive question is: **is the whole structure genuinely corpus-derived, or is something host-smuggled through the composition seams?** The answer is the permuted-corpus control, which scrambles each exemplar's word order at BOTH the inventory-mining (S1a) and order-learning (S1b) stages simultaneously. It collapses end-to-end render AND assembled-structure match to **0.333** on every seed — and, per the audit-corrected reading above, that 0.333 is the two MULTI-SLOT constructions (F_MODAL, F_NEGMOD) **genuinely collapsing to 0** while F_INTR is a deterministically-reconstructed named residual (its inventory recoverable without order; the EMERGE-64b follow-on closes it to 0.0). So the control **proves the corpus-ORDER-derivation of the two multi-slot constructions** (the load-bearing claim, not "the whole pipeline"). Note the S2 discovery (frequency + coverage) survives the shuffle (token IDENTITY multiset is unchanged) — which is correct and honest: the ORDER-derived structure (inventory + slot order) is what the shuffle destroys, and that is exactly the corpus-derived part being proven. NO-CORPUS (empty stream) yields an empty inventory. Neither the FRAMES dict, the FUNCTION_WORDS list, nor the template order is an input — verified independently by grepping the pipeline path (the audit's host-smuggling dimension found NO leak: `build_from_corpus` takes only the corpus stream).

---

## Carried-forward residuals (named, NOT hidden, NOT walls)

Each component's honestly-named residual is carried forward and surfaced in the JSON, not buried:
- **EMERGE-64 residual:** a HELD-OUT frame's DISTINCTIVE function-word slots (F_MODAL's `can`, F_NEGMOD's `does`/`not`) + F_INTR's `3sg` inflection are NOT recoverable from the OTHER two frames alone (only that frame attests them). The held-out arm generalizes only the SHARED det+subj+verb backbone + type-level order (the gated claim). Measured: `heldout_intr_inflection_recovered = False` (expected False — only F_INTR attests 3sg).
- **EMERGE-63 residual:** a HELD-OUT multi-function-word frame's `does<not` INTERNAL order is not learnable from the other frames (only F_NEGMOD attests two adjacent function words). Same category.

The next single signal for both is ONE attestation of the held-out frame's own function word / inflection / bigram (or Yang-Getz's phrase-boundary cue). **These are precisely-named residuals, NOT walls.**

**Honest scope:** this renders the BOUNDED EMERGE frame inventory (ability-affirm / intransitive-exception / negated-modal) on spikes from the corpus-derived structure — NOT open prose (R4, the separate deferred wall; the from-scratch spiking LM is ~4 orders too small, `2026-05-07-Phase-2.3a-NEGATIVE`). The corpus discovery/mining is offline syllabus prep (BRAIN-BASED-ONLY compliant — like rendering a retinal image the neural retina reads); the structure is rendered on REAL spikes; the gate-first moat is untouched.

---

## No `sim/` edit; defaults preserved

Everything is reuse-by-import. `SelfOrganizedProducer` composes the existing `discover_closed_class` (S2), `mine_inventory`/`label_sentence`/`MinedInventoryFrameSlotCQ` (S1a), `learn_corpus_order`/`order_heldout_frame` (S1b), and `BrocaProducer`/`decision_from_emerge` (the producer + moat). EMERGE-59/60/61/62/62b/63/64 are all untouched — **56/56 CI tests across `test_emerge60..65` pass** (49 from EMERGE-60..64 + the 8 new EMERGE-65 tests + emerge62b).

---

## Follow-on (named, NOT built here) — EMERGE-66

Wire `SelfOrganizedProducer` into EMERGE-60's `SpikingBrocaConsole` (an additive default-off flag `self_organized=True` that builds `SelfOrganizedProducer(seed).build_from_corpus(build_stream(seed))` and sets `self.broca = that.producer(spell=spell)`) so the flagship console renders from the fully-self-organized producer. It is a genuinely clean ~6-line additive change, but it touches the committed console file and warrants its own focused de-risk (confirming no regression on the fluid-path gate + the heavier torch/GPU console build), matching how EMERGE-60/61 each got their own. Named as **EMERGE-66** rather than bolted in unverified, to keep this capstone's additive / NO-`sim/`-edit discipline clean.

---

## Audit remediation (2026-07-03 EMERGE-62..66 adversarial audit)

A 6-dimension sequential adversarial audit of the EMERGE-62..66 chain confirmed **2 non-GO-changing defects** (the other 4 dimensions — host-smuggling, struct-match circularity, real-corpus-precision framing, moat + byte-identity — came back clean):
- **[major] a "control that cannot fail":** the GO gate included `heldout_shared_backbone ≥ 0.999`, but all three EMERGE frames share the identical (DET,SUBJ,VERB) role backbone (FUNC + inflection stripped), so that metric is a language-universal constant structurally incapable of returning < 1.0 (the audit proved it returns 1.0 even trained on unrelated garbage). **Fix:** dropped from the GO gate (now `heldout_generalizes = heldout_shared_order ≥ 0.999`, the GENUINE held-out evidence — the FUNC position learned from another frame's `does/not`, verified tie-break-invariant over 200 draws); backbone is REPORTED only. GO stands.
- **[minor] permuted-corpus framing overstated:** the 0.333 floor is F_INTR *deterministically* reconstructed (dominance 1.0), not a chance floor, and not "the whole pipeline collapsing" — only the two multi-slot frames genuinely collapse. **Fix:** the framing above + the verdict/docstring corrected; the **EMERGE-64b** follow-on (shuffle-invariant bag-keying) will make F_INTR collapse too, driving the floor to 0.0 and making the claim literally true.

Both fixes are honest-propagation only; the GO (render 1.0, struct-match 1.0, two-frame permuted-collapse with 0.667 margin, genuine held-out order, moat 0) is unchanged. This is the project's adversarial-audit discipline catching the same *class* of metric/framing defect it caught in EMERGE-58 and EMERGE-38..45.

## ⇒ Bottom line

**From the corpus alone, the brain now discovers the function-word inventory, mines the construction slot inventory, learns the slot order, assembles the grammatical structure, and speaks its grounded answers ON SPIKES — transformer-free, moat intact, with NO host-designed grammatical structure anywhere in the producer.** The spiking-Broca producer's entire structure is self-organized from experience, end-to-end. The composed permuted-corpus control is the load-bearing proof: scramble the corpus word order and the two multi-slot constructions' structure genuinely collapses to 0 (F_INTR being a deterministically-reconstructed named residual the EMERGE-64b follow-on closes).
