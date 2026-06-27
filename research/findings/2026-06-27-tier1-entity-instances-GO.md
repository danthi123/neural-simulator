# Tier 1.1 — entity-instance / discourse-referent layer: cheap-first de-risk GO (6/6 seeds)

**Date:** 2026-06-27
**Scope:** the KEYSTONE of the conversation-depth roadmap (`2026-06-27-conversation-thinking-ROADMAP.md`, Tier 1.1;
fronts 1+2 BOTH ranked it #1). Turn the brain's TYPE-keyed knowledge ("the concept boy") into INSTANCE tracking
("this boy vs that boy") so **"which boy?" is genuinely answerable** — upgrading the console's honest-generic Tier-0.4
clarification ("I track the idea of boy but not specific ones yet") into a real disambiguation.
**De-risk runner:** `research/runners/_tier1_entity_instances_derisk.py` (tiny vocab, numpy CPU, ~2.5 s for 6 seeds).
**Verdict: GO — 6/6 seeds, all three gates + all anti-cheats clean.** Step 2 (the production build + console wire)
proceeded; see `entity_instance_layer.py` + the console wiring below.

---

## The mechanism (reuse-by-import; NO `sim/` edit, NO production-composer edit)

An entity INSTANCE token is a **phasor code minted as the TYPE code blended with a per-instance sparse "barcode"** —
the hippocampal **episodic-index** / DG-sparsified token (the SHIPPED D.14 engram API is functionally this barcode;
Quian-Quiroga concept cells = the TYPE, the barcode = the individuating index). In the complex domain:

    z(boy#i) = normalize( (1-alpha) * z_type[boy] + alpha * z_barcode_i ),   instance_phases = angle(z)/2pi

- **alpha = 0** → the pure type code → ALL `boy#i` are IDENTICAL == the **merge lesion** (DG separation OFF).
- **alpha = 0.7** (the DG operating point, default) → instances near the random-floor decorrelation **yet still
  carrying the type** (inst-inst phase-cos ≈ 0.06, inst-type ≈ 0.22, random floor ≈ 0).
- **Overlap-rejection** on the barcode draw (the project's own recovery path from the 320-concept sparse-codes work)
  realizes DG pattern-separation + adult-neurogenesis "fine pattern separation": redraw until the new instance is
  decorrelated (< 0.12) from every same-type sibling. This made the two previously-pathological seeds (42, 102 —
  unlucky barcode collisions) separate cleanly, lifting 4/6 → 6/6.

Instance codes are **injected into the deployed `RFPhasorComposer.concepts` dict** (the composer is concept-AGNOSTIC
for binding, `rf_phasor_composer.py:262`), so a fact attaches to the INSTANCE via the SAME spiking RF bind/unbind the
production composer uses (`agent = boy#1`, not the bare type "boy"). A **DRT-style file-card** (`_tokens` type metadata
+ `_held` referent registry) maps surface refs → tokens. "which X?" is a **biased-competition WTA** over the type's
candidate instances, scored by which one's distinguishing fact matches the cue (the de-risked
`biased_competition_buffer.py` pattern: a clean winner or an abstain on a tie).

**Biology:** hippocampal episodic-index barcode (eLife 2024 PMC11429605) binds co-active concept TYPES into an
individuated TOKEN; DG pattern separation (catalog D.12) keeps same-type instances decorrelated; CA3 pattern
completion (D.13) recovers the right one from a partial cue; Tonegawa engram (D.14, SHIPPED) = the barcode;
Eichenbaum-Cohen items-in-context (D.02) = the discourse-referent store; Desimone-Duncan / Wong-Wang biased
competition picks among several matching referents; DRT/file-card (Kamp 1981) = the surface-ref → token map.

---

## The three gates + anti-cheats (all 6/6 seeds; the controls are load-bearing)

### Gate (a) — PATTERN SEPARATION (instances stay separable, don't collapse to one "boy")
- **Diagnostic:** inst-inst phase-cos sits BELOW the principled midpoint between the random floor and the type-overlap
  the instances necessarily share (the `cleanup_separated` placement rule, not a tuned bound). All 6 seeds:
  inst-inst ∈ [−0.03, +0.11] vs inst-type ∈ [+0.17, +0.28], random floor ≈ 0. The instances are near-floor
  decorrelated yet type-linked.
- **MERGE-LESION control (load-bearing):** at alpha=0 the two instances become byte-IDENTICAL codes
  (`merged_inst_inst_cos=1.0`), so the agent-binding can no longer individuate — `unbind(fact,'agent')` cleans up to
  the SAME token for both facts. The system then **cannot recover the distinct (boy#1, boy#2) pair**
  (`merged_distinct_correct=false`; one query resolves, the other → `null`). PASS = the merge BREAKS disambiguation,
  proving separation is what lets the binding individuate.

### Gate (b) — RIGHT REFERENT ("which boy went to the park?" → boy#1)
- **which-park → boy#1**, **which-apple → boy#2** on ALL 6 seeds, with a decisive WTA margin (scores park: boy#1=2,
  boy#2=0; symmetric for apple). The console disambiguation text the layer produces:
  *"the one that went to the park, or the one that ate the apple?"* and the answer *"the boy that went to the park."*
- **BINDING-LESION control (load-bearing):** sever the instance→fact binding (every fact a candidate for every
  instance) → both instances match the cue equally → **abstain (`null`)**. Proves the agent-binding does the
  disambiguation, not a code coincidence.
- **PRONOUN / definite:** with boy#2 the held discourse referent, "the boy"/"he" pattern-completes to boy#2 ✓.
  With an EMPTY file-card a pronoun has NO antecedent → **abstain (None)** (the reset is a load-bearing control).

### Gate (c) — MOAT, 0 FALSE-ACCEPTS (never fabricate an instance)
- An UNSTORED instance query ("which boy chased the cat?" — no boy did) → `None`; a NEVER-ALLOCATED type
  ("which girl…") → `None`; a fact-query for an unstored predicate (`query_patient(boy#1,'chase')`) → `None`.
  **0 false-accepts, all 6 seeds.**

### Robustness beyond 2 instances (pre-validates the WTA at scale)
- **3 same-type instances, genuine tie** (two boys both went to the park, scores 2/0/2) → `None` (abstain) — the
  no-confab moat IS the "which X?" clarification trigger when several match.
- **3 same-type instances, unique match** → the correct instance; **3-way distinct** → each query → its own instance.

---

## Honest scope / boundaries

- **The barcode "structure" is a developmental-random wiring rule** (a per-instance code drawn from a disjoint rng
  stream + overlap-rejection), the genome-style self-organization the project already accepts (`sim/dendritic_neuron.py`
  style; CLAUDE.md FHRR-B note). The bind/unbind that attaches facts is the validated spiking RF FHRR primitive.
- **Capacity is biology-bounded** (Lisman-Idiart ~7 active referents in the file-card) — a *biology-faithful* limit,
  not a defect; report it, don't brute-force it (the de-risk's overlap-rejection scales separation cleanly at the
  small instance counts a discourse holds).
- **The which-X scoring loop is host-side** (iterate candidates, count matched cue roles). The WIN — the instance
  CODES, the fact BINDING, the abstain — is brain-based (spiking RF). The candidate-scoring is the same host scaffold
  flagged in `biased_competition_buffer.py` (`content_bias_target`); the neuralized version (the biased-competition
  buffer's spiking WTA over the candidate assemblies) is the obvious follow-on, exactly as that module documents.
- **Multi-REFERENT bare-pronoun disambiguation** (which of several held referents a bare "it" binds) still needs the
  biased-competition WTA + finer agreement cues (the documented `2026-06-17-multireferent-disambiguation-NEGATIVE.md`
  mechanism) — here the file-card resolves a single salient referent and abstains on a tie.

---

## Verdict

**GO.** The keystone mechanism — allocate same-type instances as separable barcode tokens, attach facts to the
instance, resolve "which X?" by distinguishing facts (biased-competition WTA), pattern-complete a pronoun to the held
referent, abstain on the unknown — is validated end-to-end, multi-seed, with every anti-cheat control load-bearing
(merge-lesion + binding-lesion both break disambiguation; the moat holds 0-FA). NO `sim/` edit; reuse-by-import on the
deployed `RFPhasorComposer` + the SHIPPED D.14 barcode primitive. Step 2 (production `EntityInstanceLayer` +
console "which boy?" upgrade) followed.
