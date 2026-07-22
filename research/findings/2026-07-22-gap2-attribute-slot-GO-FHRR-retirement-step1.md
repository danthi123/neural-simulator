# gap#2 — ATTRIBUTE SLOT: single-attribute patients on the learned spiking slot-binder, 6-seed GO (FHRR-retirement step 1/2)

**2026-07-22, CPU/numpy, coexisting with the fluency training.** First of the two de-risks that fully retire the FHRR
exact-inverse algebra (per `2026-07-22-recursive-slotbinder-research-gate.md`). The FHRR's single-attribute path is one
extra flat role, NOT recursion; the slot-binder gets the same via one more competitive slot.

## The change (`research/runners/slotbinder_composer.py` ONLY; additive, default-preserving; NO `sim/` edit)
`_ROLES` 4→5 (agent, verb, patient, polarity, **attribute**); a `NOATTR="__NOATTR__"` filler pool appended like
`AFFIRM/NEGATE` (a fact with no attribute writes NOATTR → reading NOATTR returns None = no confabulated adjective = the
moat by construction); `_resolve_patient` splits a `(adjs,noun)` tuple (single-attribute = first adjective); `store`
gains `attribute=` / a tuple patient + binds slot `_ROLES*i+4`; new `query_attribute`; `render_fact` prepends the
adjective when present. The bare-string flat path is byte-identical (attribute→NOATTR).

## Result — 6-seed GO (independently controller-reproduced)
- **MAIN (attributed facts big-apple / small-fish / hot-river + flat facts):** patient AND attribute JOINT recovery
  **1.000** all 6 seeds (bar ≥0.90); flat SVO un-regressed (patient 1.000, attr=None 1.000).
- **AC1 permuted-attribute (derangement):** attr-vs-TRUE **0.000** (≤ chance 0.33 → the attribute slot is fact-specific,
  not a fixed/derived default); attr-vs-PERMUTED **1.000** (faithful storage of what was taught).
- **AC2 moat:** un-attributed fact `query_attribute`→None (no confabulated adjective); never-stored cues abstain on
  BOTH patient and attribute → None. All 6 seeds.
- **Existing CI `tests/test_slotbinder_composer.py`: 6 passed, 0 regression** (controller-run).
⇒ single-attribute attributed patients close on the learned spiking slot-binder, moat-safe. Do NOT claim 2-attribute /
per-noun-attribution (the FHRR's own boundary; the latter is a mini-pointer problem, out of single-attribute scope).

## Remaining for full FHRR retirement
Step 2/2 = the depth-1 embedded-clause POINTER de-risk (MOVE 3 #1 of the research gate: the inner clause is its own
slot-group, the matrix patient binds a pointer to it, recall follows the pointer — indirection, not copy). On both GO,
the slot-binder covers the COMPLETE deployed FHRR set (flat SVO + polarity + multi-hop + single-attr + depth-1 clause)
and the FHRR exact-inverse algebra is retired. De-risk: `research/runners/_gap2_attribute_slot_derisk.py`.
