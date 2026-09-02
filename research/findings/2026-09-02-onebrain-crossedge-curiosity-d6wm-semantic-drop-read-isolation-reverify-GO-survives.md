---
type: finding
status: live
date: 2026-09-02
mechanism: onebrain-xedge-curiosity-d6-semantic-drop-read-isolation-reverify
board: one-brain integration / measurement-integrity (C2 bug class, record follow-through)
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_onebrain_xedge_curiosity_d6_semantic_drop_readfix_6seed.json
runner: research/runners/onebrain_xedge_curiosity_d6_production.py
builds_on:
  - research/findings/2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md
  - research/findings/2026-09-02-onebrain-crossedge-curiosity-to-d6wm-read-isolation-fix-corrects-GO-to-NOGO-3-6.md
  - research/findings/2026-09-01-onebrain-crossedge-curiosity-d6wm-semantic-drop-GO.md
---

# Semantic-drop rung re-verified against the fixed `AskToW0Pool`: the 6/6 GO SURVIVES read-isolation — the mechanism's own `clip(cross_weight, 0, 1)` saturates regardless of the leak — but the rung is currently INERT in production (parent gate flipped OFF by a separate, already-landed fix) and a NEW, unrelated confound was found

**One-line:** the 2026-09-02 curiosity→d6wm read-isolation fix's own honest scope note flagged
`2026-09-01-onebrain-crossedge-curiosity-d6wm-semantic-drop-GO.md` as "almost certainly inherits an analogous
correction... but this finding did not re-run it." Re-running it against the FIXED `AskToW0Pool` (same code,
same config the original finding tested) reproduces **6/6 GO, unchanged** — the semantic-drop mechanism only
consumes `scale = clip(pool.cross_weight, 0, 1)`, and the trained cross-edge weight stays **> 1 on every seed
both before AND after the fix** (before: 1.74-2.12; after: 1.59-2.89), so `scale` saturates at exactly `1.0`
either way and the injected erase current (`-1500 pA`/200 steps, `MultiSlotHold`'s own fixed clear-strength
constants) is bit-for-bit identical regardless of the leak. **The standalone claim was NOT inflated.** Separately:
the base cross-edge this rung depends on was already flipped `_XEDGE_CD6_DEFAULT_ON=False` by commit `afcb3ba7b`
(a prior, independent correction), so `get_xedge_curiosity_d6_pool()` returns `None` on every live request and
`semantic_drop_current()`'s own guard clause (`if pool is None ... return None`) means the rung **never fires in
production today**, regardless of its own flag reading `True` — distinguishing "inflated as a standalone claim"
(NO) from "inert in production" (YES, but for an unrelated, already-fixed reason). A NEW, genuinely separate
confound was also found and is NOT fixed here (§4): a LATER, unrelated production default flip
(`BRAIN_MULTIREF_COMPETITIVE=1`, commit `96ebbffc8`, landed AFTER this finding) breaks the drop at seed 44 when
tested against CURRENT defaults instead of the original finding's own tested configuration.

## 1. The re-verification — apples-to-apples against the FIXED pool

Re-ran the finding's OWN command exactly (`SIM_BACKEND=numpy python -m research.runners.
onebrain_xedge_curiosity_d6_production --grow --semantic-drop --seeds 42,43,44,100,101,102`), on a worktree
branched from post-fix `main` (so `AskToW0Pool._hard_reset` already restores the full
`_PER_NEURON_STATE`+`_SEQ_EXTRA_STATE` set — confirmed via `--selftest`, bitwise-identical repeat-read, PASS).
`BRAIN_MULTIREF_COMPETITIVE=0` was set explicitly to match the register-allocation configuration that existed
when the original finding was authored (see §4 for why this matters).

The "BEFORE fix" column below is quoted verbatim from `2026-09-01-onebrain-crossedge-curiosity-d6wm-semantic-drop-GO.md`'s own §4 table, not from an artifact cited by this finding; the "AFTER fix" column and every other cell come from this finding's own cited artifact:

<!--derived-->

| seed | cross_weight (intact, AFTER fix) | cross_weight (intact, BEFORE fix, from the original finding) | crave+intact recovered | crave+lesioned recovered | flag-off recovered | GO |
|---|---|---|---|---|---|---|
| 42 | 2.5781 | 2.0202 | **cat** (dog dropped) | dog, cat | dog, cat | GO |
| 43 | 1.5891 | 1.8899 | **cat** (dog dropped) | dog, cat | dog, cat | GO |
| 44 | 2.8880 | 1.9818 | **cat** (dog dropped) | dog, cat | dog, cat | GO |
| 100 | 2.1866 | 1.9708 | **cat** (dog dropped) | dog, cat | dog, cat | GO |
| 101 | 1.6224 | 2.1176 | **cat** (dog dropped) | dog, cat | dog, cat | GO |
| 102 | 1.6614 | 1.7391 | **cat** (dog dropped) | dog, cat | dog, cat | GO |

## 1a. Scope note

The block above is marked `<!--derived-->` only because its BEFORE column re-quotes the original finding's own
numbers for side-by-side comparison; the AFTER column and every claim elsewhere in this document that is NOT so
marked is checked against this finding's own cited artifact.

`n_go: 6/6` (`research/findings/raw/_onebrain_xedge_curiosity_d6_semantic_drop_readfix_6seed.json`,
`semantic_drop.n_go=6`). Every seed: `dog_dropped_intact=True`, `dog_recovered_lesioned=True`,
`byte_identical_flagoff=True` — all four honest-baseline/lesion/flag-off checks reproduce exactly as originally
reported. The trained cross-edge weight visibly moved (the SAME training-leak effect the parent finding
documents — grown weights differ substantially before vs after because `train()` calls `_hard_reset()` every
episode), but this has ZERO effect on the semantic-drop readout: `scale = max(0.0, min(1.0, cross_weight))`
clips every one of these 12 values (6 before, 6 after) to `1.0`, so `erase_pa = -abs(clear_gain)*scale =
-1500.0` is bit-identical in all 12 cases. **Verified, not assumed**: a standalone probe that bypasses
`AskToW0Pool` entirely (constructs `MultiReferentWMOrgan` directly and injects the hand-computed
`(-1500.0, 200)` current) reproduces the identical 6-seed pattern (5/6 or 6/6 depending on the register-
allocation config — see §4), confirming the semantic-drop readout has NO other hidden dependency on
`AskToW0Pool`'s internal state beyond this one saturated scalar.

## 2. Why this rung is immune where the parent rung was not

The parent finding's own `read_w0()` measures a SIGNED, continuously-varying rate shift (`delta_intact`) against
a fixed `INTACT_FLOOR=0.008` — a leak that shifts the measured rate by even a few thousandths can cross that
floor, which is exactly what happened (3 of 6 seeds flipped). This rung instead reads `pool.cross_weight`
through a `clip(..., 0, 1)` gate whose only two functional outputs are "some hyperpolarizing pull" (weight > 0)
or "none" (weight == 0, the lesion case) — the SPECIFIC continuous value above 1.0 is thrown away by design (the
module's own docstring: "the frozen edge grows to ~1.7-2.1 (clamps to the FULL scale=1.0)"). A leak that changes
the trained weight's MAGNITUDE (while keeping its SIGN and its >1 range) is invisible to a binary/saturated
consumer — this is the same "is the read binary-saturating or continuously threshold-sensitive" distinction that
explains why the parent's own `delta_intact` (continuous, threshold-sensitive) flipped while this rung's
`clip(w,0,1)` (saturating) did not.

## 3. Production status — inert, for an ALREADY-FIXED, unrelated reason

`docs/PRODUCTION_INTEGRATION_LEDGER.yaml`'s `onebrain-xedge-curiosity-d6` row and `_CD6_SEMANTIC_DROP_DEFAULT_ON`
(`research/runners/onebrain_xedge_curiosity_d6_production.py:130`) still read `True`, but
`_XEDGE_CD6_DEFAULT_ON` (the BASE rung's own flag, line 115) is `False` as of commit `afcb3ba7b`
("fix(integrity): curiosity->d6-WM default-ON -> OFF (banked GO inflated; true verdict NO-GO 3/6)") — a prior,
separate correction this finding did not need to make. Since `get_xedge_curiosity_d6_pool()` returns `None`
whenever `xedge_curiosity_d6_enabled()` is `False`, and `semantic_drop_current(pool, d6org)`'s first line is
`if pool is None ... return None`, the semantic-drop rung's own code is unreachable on a live `/api/brain-chat`
turn today — **not because this rung is broken, but because its prerequisite is currently off pending the base
rung's own re-tuning** (per `afcb3ba7b`'s commit message, "residual 3/6 is a real mechanism weakness (NO-DEFER
next-lever)"). `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` is corrected in this same landing (§5) to state this
explicitly, since its current text ("BOTH flags flipped ON 2026-09-01") is now stale.

## 4. A NEW, unrelated confound found while re-verifying (NOT fixed here — flagged, logged)

Testing this rung against **current** production defaults (leaving `BRAIN_MULTIREF_COMPETITIVE` unset, i.e. its
own current default of `1` per `_MULTIREF_COMPETITIVE_DEFAULT_ON=True`, flipped by commit `96ebbffc8`
2026-09-02 — a LATER, unrelated commit to `d6_multiref_wm_production_organ.py` that landed AFTER this
semantic-drop finding was authored) instead of the register-allocation configuration the original finding
actually tested (`BRAIN_MULTIREF_COMPETITIVE=0`, "role-by-position") reproduces only **5/6 GO**: seed 44 fails
to drop 'dog' (`recovered=['dog','cat']`) while all other 5 seeds still succeed. This is COMPLETELY INDEPENDENT
of `AskToW0Pool`/the read-isolation fix — confirmed by a standalone probe that hand-injects the identical fixed
`(-1500.0, 200)` current directly into a freshly-built `MultiReferentWMOrgan`, bypassing the cross-edge pool
entirely: under `BRAIN_MULTIREF_COMPETITIVE=0` all 6 seeds drop 'dog' cleanly (matching the original finding
exactly); under the CURRENT default (`=1`, competitive slot allocation) seed 44 alone fails, identically. This is
a genuinely NEW integrity gap — the semantic-drop mechanism was validated only against the register-assignment
mechanism that existed when it was built, and was never re-tested against the later competitive-allocation
default. It does not corrupt anything reported in §1 (which correctly reproduces the ORIGINAL test's own
configuration), but it means the rung's claim of "6/6 GO, wired" would NOT currently hold at 6/6 if both this
rung's base flag AND `BRAIN_MULTIREF_COMPETITIVE`'s current default were live simultaneously. Logged in
`research/FAILURE_LOG.md` (2026-09-02 row); a follow-up task is flagged separately rather than chased here — it
is a distinct bug class (an untested interaction between two independently-shipped default flips), not a
read-isolation defect, and is out of this lane's scope.

## 5. Ledger correction

`docs/PRODUCTION_INTEGRATION_LEDGER.yaml`'s `onebrain-xedge-curiosity-d6` row's `on_by_default` field said "BOTH
flags flipped ON 2026-09-01" — stale since `afcb3ba7b` flipped the base flag OFF. Corrected in this same commit
to note the base rung is OFF (pending re-tuning per `afcb3ba7b`) and the semantic-drop rung, while its own
constant remains `True`, is therefore currently unreachable.

## 6. What survives, what is corrected, what is newly flagged

**Survives, unretracted:** the semantic-drop mechanism's own 6/6 GO, its lesion-attributability, its
byte-identical-off property, and its "closes the base rung's declared residual" characterization — all
reproduce to the same qualitative result under the fixed pool and the finding's own originally-tested
configuration.

**Corrected (this finding, not a retraction of the GO):** the specific `cross_weight` values printed in the
original finding's table <!--derived--> (§4, e.g. "2.0202" at seed 42 — quoted from that finding's own text, not
from an artifact this finding cites) reflect the PRE-FIX, leak-inflated/deflated training trajectory; the true
(isolated) values are listed in §1 above. Since the mechanism's own consumer saturates the metric, this does not
change the GO verdict, but a reader citing the ORIGINAL table's specific weight numbers as the substrate's true
state would be citing stale figures.

**Newly flagged, not fixed:** §4's competitive-slot-allocation interaction (logged to `research/FAILURE_LOG.md`).

## 7. Files

Unmodified (this is a re-verification, not a code fix): `research/runners/onebrain_xedge_curiosity_d6_production.py`,
`research/runners/_onebrain_crossedge_curiosity_to_d6wm.py` (already fixed by `ffa229876`),
`research/runners/d6_multiref_wm_production_organ.py`, `research/runners/_multi_slot_binding_derisk.py`.
New: `research/findings/raw/_onebrain_xedge_curiosity_d6_semantic_drop_readfix_6seed.json`. Modified:
`docs/PRODUCTION_INTEGRATION_LEDGER.yaml` (`onebrain-xedge-curiosity-d6` row, `on_by_default` field), `research/FAILURE_LOG.md`
(new row, §4's confound). No `sim/` file touched.

Functional read-outs only; no phenomenal-experience claim.
