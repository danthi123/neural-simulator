# EMERGE-64b — SHUFFLE-INVARIANT bag-keying closes the EMERGE-64/65 permuted-corpus F_INTR residual — GO (6-seed)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge64b_shuffle_invariant_bagkey_derisk.py`
**Miner edit (additive, default-off):** `research/runners/_emerge64_mine_slot_inventory_derisk.py` (`_bag_key_invariant` + `mine_inventory(..., shuffle_invariant_bag=False)`)
**EMERGE-65 opt-in (additive, default-off):** `research/runners/_emerge65_self_organized_producer_derisk.py` (`SelfOrganizedProducer(seed, shuffle_invariant_bag=False)` + `permuted_corpus_collapse(..., shuffle_invariant_bag=False)` + `--shuffle-invariant-bag`)
**CI:** `tests/test_emerge64b_shuffle_invariant_bagkey.py` (12 tests, CPU/numpy, offline, ~4s)
**Raw:** `research/findings/raw/_emerge64b_shuffle_invariant_bagkey.json`
**Verdict:** **GO, 6-seed unanimous (42/43/44/100/101/102).** Reuse-by-import; **NO `sim/` edit**; the gate-first no-confab moat is untouched; the EMERGE-64/65/66 committed defaults are **byte-identical**.

---

## Headline

**The EMERGE-64 slot-inventory mining's permuted-corpus anti-cheat now GENUINELY collapses ALL constructions — including the shortest, F_INTR — closing the residual the EMERGE-62..66 adversarial audit surfaced.** With a shuffle-invariant bag-keying, the permuted-corpus render drops from **0.333 (F_INTR alone, deterministically reconstructed) → 0.000** while the MAIN (unshuffled) mining stays exact (inventory-accuracy 1.0, spiking render 1.0). So the EMERGE-64/65 claim "permuted-corpus collapses the whole pipeline" becomes **literally true** — every construction is proven corpus-ORDER-derived, not just honestly-reframed.

This closes the `[minor]` audit remediation item flagged in `2026-07-03-emerge65-self-organized-producer-GO.md` ("Audit remediation") + AUTONOMOUS_STATE CYCLE 876.

---

## The defect (the audit's precise diagnosis)

Under the permuted-corpus control, F_INTR ("the penguin walks", det+subj+verb) was **DETERMINISTICALLY reconstructed at dominance 1.0** — so the perm floor `0.333` was **F_INTR alone, not a chance floor**. The cause is entirely in the bag-keying:

- `label_sentence` (`_emerge64:189-191`) decides `is_det` by **POSITION**: a closed-class token is `DET` iff it opens the clause AND immediately precedes a content word; otherwise it is `FUNC`.
- `_bag_key(sig)` (`_emerge64:223-226`) sorts the **signature**, which embeds that `det:`/`func:` position label.

For the F_INTR token multiset `{the, penguin, walks}`, the labellable orderings scatter into **two bags** (verified in CI, `test_default_keying_scatters_fintr_orderings_into_two_bags`):

| ordering | signature | DEFAULT bag |
|---|---|---|
| `the penguin walks` | `(det:the, subj, verb:3sg)` | `(det:the, subj, verb:3sg)` |
| `penguin the walks` | `(subj, func:the, verb:3sg)` | `(func:the, subj, verb:3sg)` |
| `penguin walks the` | `(subj, verb:3sg, func:the)` | `(func:the, subj, verb:3sg)` |

So the ~1/3 of shuffles that keep `the` at NP-onset re-label it `det:the` → the **exact F_INTR bag** → reconstructed at dominance 1.0, while the "wrong" orderings (`func:the`) go to a **different bag** and never dilute it. **Word order is thus NOT actually needed to mine F_INTR** — the control only proved the ORDER-derivation of the two multi-slot frames (F_MODAL, F_NEGMOD, which genuinely collapse to 0), leaving F_INTR a named residual.

---

## The fix (the audit's named remediation — additive, default-off)

Key the mining bags by a **SHUFFLE-INVARIANT token multiset** that does NOT embed the DET/FUNC POSITION label: closed-vs-open is decided by **EMERGE-62's DISCOVERED function-word SET** (token IDENTITY, position-independent), not by `is_det(the at pos 0)`. `_bag_key_invariant(slots)` maps:

- a **DET or FUNC** slot (both closed-class by set membership) → `closed:<lexeme>` — so a `the` at NP-onset (DET) and a `the` elsewhere (FUNC) map to the SAME `closed:the`;
- a **VERB** slot → `verb:<inflection>` — the inflection is read from the surface **morphology** (a trailing `-s` over a content-verb lexeme), itself position-independent, so it stays in the key;
- a **SUBJ** (open-class NP head) → `open`.

Now **every labellable ordering of a frame's tokens shares ONE bag** (CI `test_invariant_keying_merges_fintr_orderings_into_one_bag`), so under shuffle the F_INTR orderings **dilute the dominant fraction below `min_dominance` (0.80)** → F_INTR fails to mine confidently → it **collapses too**.

Crucially, the MAIN corpus still **separates** the three frames because their **closed-token multiset + verb inflection differ**: F_MODAL `{the,can}`+bare / F_INTR `{the}`+3sg / F_NEGMOD `{the,does,not}`+bare → distinct bags. The shuffle-invariant key merges only orderings *within a frame*, never the three frames.

The change is an **additive default-off flag** on EMERGE-64's `mine_inventory` (`shuffle_invariant_bag=False` == byte-identical) + the `_bag_key_invariant` helper. NO `sim/` edit.

---

## 6-seed results (CPU/numpy)

| metric | value (all 6 seeds identical) | meaning |
|---|---|---|
| **MAIN inventory-accuracy (invariant keying)** | **1.000** | still recovers all 3 EMERGE frames exactly — the multiset distinguishes them |
| **MAIN spiking render-exact (invariant keying)** | **1.000** | producer renders "the owl can fly" / "the penguin walks" / "the penguin does not fly" EXACT on spikes |
| **PERMUTED-CORPUS render — DEFAULT keying (BEFORE)** | **0.333** | F_INTR alone reconstructed (the audit-named residual) |
| **PERMUTED-CORPUS render — INVARIANT keying (AFTER)** | **0.000** | F_INTR collapses too → the whole pipeline genuinely collapses |
| PERMUTED-CORPUS inventory-accuracy (before → after) | 0.333 → 0.000 | same, at the inventory level |
| NO-CORPUS | empty inventory | no data → no structure |
| held-out shared det+subj+verb backbone (invariant keying) | 1.000 | generalization preserved |
| held-out F_INTR distinctive 3sg inflection recovered | False (expected) | the carried-forward named residual (same category as EMERGE-63's `does<not`) |
| **gate-first moat** | **0** producer-calls on abstains | untouched, by construction |

Per-seed line (all six identical):
```
MAIN inv-acc 1.000 render 1.000 | PERM render default 0.333 -> invariant 0.000
(inv-acc 0.333 -> 0.000) | no-corpus empty True | held-out backbone 1.000 | moat 0
```

**GO bar met:** MAIN unregressed (inv-acc 1.0 AND render 1.0) AND perm_render materially lower than the 0.333 baseline (→ 0.000), held-out backbone generalizes, moat 0, 6-seed. No BOUNDARY tension: the invariant multiset collapses F_INTR under shuffle **without** degrading the MAIN mining.

---

## Wired into EMERGE-65 (additive, default-off)

EMERGE-65's `SelfOrganizedProducer` gains an additive default-off `shuffle_invariant_bag` (threaded through `_mine_inventory`, `permuted_corpus_collapse`, `_derisk_one`/`_derisk`, and a `--shuffle-invariant-bag` CLI flag). This makes EMERGE-65's composed permuted-corpus claim literally true when opted in, while the committed default de-risk is byte-preserved:

- `--derisk` (default): **GO**, PERMUTED-CORPUS render **0.333** (committed, unchanged) — verdict text preserves the audit-corrected F_INTR-residual scope wording.
- `--derisk --shuffle-invariant-bag`: **GO**, PERMUTED-CORPUS render **0.000** — verdict text switches to "ALL THREE constructions genuinely collapse … the whole-pipeline claim is LITERALLY TRUE".

The verdict scope sentence is now conditional on the flag (honest in both modes). EMERGE-66 consumes EMERGE-65 on the DEFAULT keying (unchanged), so its committed console de-risk is byte-identical.

---

## Verification — the committed defaults are byte-identical

- **EMERGE-64 de-risk** (`--derisk`, 6-seed): **GO**, perm floor stays **0.333** (default keying unchanged).
- **EMERGE-65 de-risk** (`--derisk`, 6-seed): **GO**, perm floor stays **0.333** (default keying unchanged); `--shuffle-invariant-bag` → **GO** perm **0.000**.
- **EMERGE-66 de-risk** (`--derisk`, 6-seed): **GO** (consumes EMERGE-65 default keying).
- **CI:** `test_emerge59..66` + `test_emerge64b` — **84 passed** (0 regressions). The `shuffle_invariant_bag=False` path is proven byte-identical (`test_default_off_is_byte_identical`, `test_emerge65_default_producer_byte_identical`).

---

## Honest scope + carried-forward residual

- This strengthens the **anti-cheat control** for the BOUNDED EMERGE frame domain — it does not extend the domain (open prose R4 is the separate deferred wall). It makes the "permuted-corpus proves ALL constructions are corpus-order-derived" claim literally true rather than honestly-reframed.
- The one carried-forward residual is unchanged and honestly named: a **held-out frame's DISTINCTIVE verb inflection** (F_INTR's `3sg`) is not recoverable from the OTHER two frames (only F_INTR attests 3sg) — the same category as EMERGE-63's `does<not` internal-order residual. Reported, not gated.
- The corpus mining is offline syllabus prep (the closed/open split reads EMERGE-62's *discovered* set, itself self-organized from distributional statistics — not a host label); the inventory is rendered on REAL spikes; the gate-first moat is untouched (0 producer invocations on abstains). BRAIN-BASED-ONLY compliant.

---

## ⇒ Bottom line

**The permuted-corpus anti-cheat now genuinely collapses the whole self-organized spiking-Broca pipeline — including the shortest F_INTR construction — proving ALL of the mined grammatical structure is corpus-ORDER-derived, not host-smuggled.** The fix is a single additive default-off bag-keying flag (`shuffle_invariant_bag`) that decides closed-vs-open by the discovered function-word SET identity instead of a position-derived DET/FUNC label; MAIN mining is unregressed; the EMERGE-64/65/66 defaults are byte-identical; NO `sim/` edit; the moat is untouched.
