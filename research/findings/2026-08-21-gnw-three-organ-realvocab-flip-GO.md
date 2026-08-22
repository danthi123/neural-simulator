---
type: finding
status: contributing
date: 2026-08-21
mechanism: gnw-three-organ-d4-realvocab-comprehension-flip
lane: integration
integration_faculty: gnw-three-organ-bus
---

# GNW three-organ ignition bus flipped default-ON — the D4 over-veto is fixed by a real-vocab comprehension read (GO)

**Board #126.** The GNW three-organ bus (recall ∧ ¬surprise ∧ COMPREHENDED — organ C adds a comprehension veto the
2-organ bus can't make) was wired but HELD default-OFF: the composed no-regression check had CONFIRMED its D4
comprehension monitor over-vetoed legitimately-recalled common facts ("what does dog chase?", "what does cat eat?" →
abstain), because organ C scored comprehension over a TOY cue-lexicon so a perfectly-comprehended fact read low-margin.

## The fix (webapp/gnw_three_organ_bus.py, real-vocab `_comprehension_vote`; NO sim/ edit)

Organ C's veto authority now reads **real-vocab entity/role competence**: if the recalled fact's agent, action and
patient-head are all in the brain's OWN learned vocabulary (the engram-derived inventory the recall composer binds), it
CORROBORATES — a known recall is comprehended, no veto. Only when a content entity/role is OUTSIDE the learned vocab
(genuine non-comprehension) does it consult the spiking D4 `SpikingRoleCompetition` sel-pool WTA — its correct
"do this proposition's roles resolve?" instrument — and veto if they can't. The `organc_lesion` lever still severs the
veto (load-bearing). A recalled fact's roles are already resolved by its stored engram, so the cue-competition margin
was the wrong instrument for it (it is right for a NOVEL incoming assertion) — that mismatch was the whole bug.

## Verification — composed re-verify GO

`research/runners/_wave4_composed_flip_noregression.py` isolating `BRAIN_GNW_3ORGAN` over the 8-turn out-of-scope panel
(`research/findings/raw/_gnw_realvocab_reverify/composed_noregression.json`): **VERDICT GO, n_turns 8, n_diverged 0** —
the dog-chase / cat-eat recalls no longer abstain with the veto ON, i.e. the over-veto is gone. Genuine out-of-vocab
non-comprehension still vetoes and the lesion still reverts (the de-risk agent's deterministic logic validation:
`wizard chase cat` / `dragon eat apple` veto; lesion → commit).

seed-waiver: the composed re-verify is a DETERMINISTIC behavioural check — the real-vocab comprehension read is a
membership/logic change, so the over-veto either fires or it does not, reproducibly; n_diverged=0 is not a statistical
outcome needing seed replication. The underlying spiking organs are already multi-seed GO in their own de-risks (D4
comprehension AUC and the N-organ ignition bus 6/6 <!--derived: values quoted from finding
2026-08-12-comprehension-production-monitor-wired-into-gate-b and 2026-08-13-gnw-norgan-ignition-bus, not measured
here-->). The flip gate here is the REMOVAL of a deterministic regression, not a seeded generalization claim.

## The flip

`BRAIN_GNW_3ORGAN` default flipped `""`→`"1"` in `webapp/server.py` (=0 is the byte-identical escape → delegates to the
2-organ bus). Ledger `gnw-three-organ-bus` on_by_default → YES. The install block is fail-SAFE (a wiring failure
degrades to the 2-organ path, never crashes a turn) and now **logs on failure** rather than a bare `except: pass` — a
default-on faculty that silently no-installs is the silent-failure class, so the catch is made observable.

## Honest scope

The real-vocab membership test is host code, but the vocabulary is the brain's OWN learned inventory; the in-shard
reads and the spiking D4 WTA veto stay genuine. FUNCTIONAL comprehension-gated deliberation correlate, no phenomenal
claim. `BRAIN_GNW_3ORGAN_ORGANC_LESION=1` remains the load-bearing severance lever.
