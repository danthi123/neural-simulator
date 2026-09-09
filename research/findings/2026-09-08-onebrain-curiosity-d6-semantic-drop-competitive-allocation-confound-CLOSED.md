---
type: finding
status: live
date: 2026-09-08
mechanism: onebrain-xedge-curiosity-d6-semantic-drop
lane: curiosity (roadmap #197, one-brain integration / measurement-integrity follow-up)
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_onebrain_xedge_curiosity_d6_semantic_drop_competitive_fix_6seed.json
  - research/findings/raw/_onebrain_xedge_curiosity_d6_semantic_drop_rolebyposition_parity_6seed.json
runner: research/runners/onebrain_xedge_curiosity_d6_production.py
builds_on:
  - research/findings/2026-09-02-onebrain-crossedge-curiosity-d6wm-semantic-drop-read-isolation-reverify-GO-survives.md
  - research/FAILURE_LOG.md (2026-09-02 row, §4 confound)
---

# Curiosity->D6-WM SEMANTIC-DROP: the competitive-allocation confound (seed 44, 5/6) is CLOSED — the erase's fixed duration, not magnitude, needed a margin; the rung's own 6/6 GO now reproduces under the CURRENT production default

**One-line:** the 2026-09-02 re-verify found the semantic-drop rung's 6/6 GO drops to 5/6 (seed 44 fails) when
tested against the CURRENT production default (`BRAIN_MULTIREF_COMPETITIVE=1`, competitive slot allocation)
instead of the config it was originally calibrated against (`=0`, role-by-position), and flagged the fix as
"not attempted here." Read the substrate directly rather than theorizing: register assignment is NOT the
culprit (dog always lands in register 0 regardless of seed or config — ties at a genuinely zero-occupancy fresh
buffer resolve deterministically). The actual cause is that competitive allocation's extra `probe_occupancy()`
read adds idle/held time before the erase pulse fires, and on seed 44 alone the original fixed `-1500pA/200-step`
erase (calibrated with less elapsed hold time) no longer clears the bump. A duration bisection found the true
margin is thin (220 steps already clears it) and that MAGNITUDE is the wrong lever (`-3000pA` at 200 steps still
fails while `-2000pA` succeeds — non-monotonic). A new, dedicated `_SEMANTIC_DROP_CLEAR_STEPS=300` constant
(50% margin over the empirical threshold), used only by this rung's own erase call and never touching the
shared `MultiSlotHold.clear_steps`/`clear_gain` `write()` itself relies on, restores 6/6 GO under the current
default and reproduces 6/6 under the originally-tested config — 12/12 total. **The rung's own parent gate
(`_XEDGE_CD6_DEFAULT_ON`) remains OFF for its own, separate, unrelated reason (the base rung's read-isolation-
corrected NO-GO 3/6) — this fix does not make the semantic-drop rung live; it removes the ONE remaining reason
it would fail if the parent gate were ever re-enabled.**

## 1. Read the substrate first, not a theory

The natural first hypothesis — competitive (occupancy-based) allocation routes 'dog' into a DIFFERENT register
than 0 on seed 44, so the erase (hardwired to register 0) misses it — was checked directly rather than assumed
(per this project's own standing instruction to read the substrate before theorizing). `MultiSlotHold` has no
background OU noise (`ou_std_current_pA=0`, the class's own docstring), so a freshly-reset register's occupancy
probe reads EXACTLY 0.0; `argmin` over an all-zero vector resolves ties to index 0 deterministically. Directly
instrumenting `MultiReferentWMOrgan.load(['dog','cat'], competitive=True)` across all 6 seeds confirms this:
`registers=[0, 1]` on every single seed, both for the initial load and the hold-query's re-load. **Register
assignment is not the cause.**

## 2. The actual cause — extra elapsed hold time, and a non-monotonic magnitude response

Competitive allocation calls `buf.probe_occupancy()` (an 18-step zero-input hold read) once per referent BEFORE
each write — role-by-position does not. By the time the erase fires, register 0's bump under competitive
allocation has survived strictly more idle/held simulation steps than it did under the config the `-1500pA/200-
step` erase was originally calibrated against. On seed 44 alone this extra entrenchment is enough to push the
bump just past what the original duration reliably clears.

Direct instrumentation (`MultiReferentWMOrgan.load(..., xedge_drop_current=(pa, steps))`, real production path,
not a toy probe) at seed 44, competitive=True:

<!--derived-->

| steps @ -1500pA | 200 | 220 | 240 | 300 | 400 |
|---|---|---|---|---|---|
| dog cleared? | NO | yes | yes | yes | yes |

| pa @ 200 steps | -1500 | -2000 | -3000 |
|---|---|---|---|
| dog cleared? | NO | yes | NO |

The magnitude sweep is the important negative result: DOUBLING the pull (`-3000pA`) at the original duration
does **not** fix seed 44, while an intermediate value (`-2000pA`) does — a non-monotonic response to erase
magnitude. This is an adjacent instance of a property this module's own docstring already names (forward/
excitatory drive was found "non-monotonic/seed-inconsistent" when this rung was first calibrated); here the
same substrate property shows up for inhibitory magnitude on this specific bump/timing combination. **Duration,
not magnitude, is the reliable lever** — the margin is thin (220 clears it) but monotonic in the swept range.

## 3. The fix

`research/runners/onebrain_xedge_curiosity_d6_production.py`, `semantic_drop_current()`: added a dedicated
module constant `_SEMANTIC_DROP_CLEAR_STEPS = 300` (50% margin over the empirical 220-step threshold) and
changed the function's returned duration from `int(buf.clear_steps)` (the SHARED `MultiSlotHold` constant,
200) to this new, rung-owned constant. `buf.clear_gain` (the magnitude) is untouched — per §2, magnitude was
not the lever, and this preserves the already-verified `-1500pA` calibration. Critically, `MultiSlotHold.
clear_steps`/`clear_gain` themselves are UNCHANGED — `write()`'s own overwrite-clear protocol (a different
consumer of the same class) still runs at its own already-verified 200 steps; only this rung's own erase call
now asks for a longer pull.

## 4. Verification — 12/12 (both configs), via the real production path and the module's own official CLI

Direct instrumentation (`MultiReferentWMOrgan.load(xedge_drop_current=...)`, bypassing nothing):

<!--derived-->

| seed | competitive=True (current default) | competitive=False (originally-tested config) |
|---|---|---|
| 42 | OK (dog dropped, cat retained) | OK |
| 43 | OK | OK |
| 44 | OK | OK |
| 100 | OK | OK |
| 101 | OK | OK |
| 102 | OK | OK |

Reproduced via the module's own official self-test/CLI (`SIM_BACKEND=numpy python -m research.runners.
onebrain_xedge_curiosity_d6_production --grow --semantic-drop --seeds 42,43,44,100,101,102`, provenance-
stamped by `research/runners/__init__.py`):

- `BRAIN_MULTIREF_COMPETITIVE` unset (current default, `=1`): `research/findings/raw/
  _onebrain_xedge_curiosity_d6_semantic_drop_competitive_fix_6seed.json` — **SEMANTIC-DROP 6/6 GO** (every
  seed: `dog_dropped_intact=True`, `dog_recovered_lesioned=True`, `byte_identical_flagoff=True`,
  `no_crave_unchanged=True`). The BASE rung's own self-test in the SAME run is UNCHANGED at **3/6** (seeds
  43/101/102 still fail `clears_registered_floor`) — confirming this fix touches only the semantic-drop
  consumer and does not perturb (or paper over) the base rung's own, separate, unrelated NO-GO.
- `BRAIN_MULTIREF_COMPETITIVE=0` (the config `2026-09-01-onebrain-crossedge-curiosity-d6wm-semantic-drop-GO.md`
  originally tested): `research/findings/raw/_onebrain_xedge_curiosity_d6_semantic_drop_rolebyposition_parity_
  6seed.json` — **SEMANTIC-DROP 6/6 GO**, reproducing that original finding's own result with no regression.

**Existing regression tests** (`tests/test_webapp_server.py -k xedge_curiosity_d6`, 8 tests covering no-
regression-on-ordinary-turns, explicit-disable byte-identity, ambient-default, qualifier+lesion, the semantic-
drop referent-drop test itself, and session isolation) pass unchanged against this fix.

## 5. What survives, what changes, what remains open

**Survives:** the semantic-drop mechanism's own architecture (§ of `onebrain_xedge_curiosity_d6_production.py`'s
module docstring) — the erase still rides the frozen cross-edge's OWN measured weight (`scale = clip(cross_
weight, 0, 1)`), still targets register 0 exclusively, still vanishes under lesion (`cross_weight~0`), and is
still byte-identical when the rung's own flag is off. The base rung's own NO-GO 3/6 (a distinct, already-
documented, unrelated mechanism weakness in `crossedge_w0_shift`'s continuous-shift read) is untouched by this
fix — this finding does not claim to close it, and does not retune it.

**Changes:** the semantic-drop rung's own erase call now uses a dedicated 300-step duration instead of the
shared `MultiSlotHold` class default (200); `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`'s `onebrain-xedge-
curiosity-d6` row and `research/FAILURE_LOG.md`'s 2026-09-02 row are updated to record the closure (§7).

**Remains open, honestly:** (1) the base rung's own 3/6 — this faculty's PARENT gate — is a genuinely separate,
harder mechanism problem (a continuous, threshold-sensitive read, not this rung's saturating one) and is NOT
addressed here; re-enabling the semantic-drop rung in live production still requires either accepting the
parent gate's own current NO-GO status or a separate fix to `crossedge_w0_shift`. (2) The general CLASS this
confound belongs to (no gate cross-checks a faculty's own 6-seed GO artifact against a LATER, independently-
shipped default flip in a shared dependency) remains NOT-GATEABLE, per the original 2026-09-02 row — this
finding closes the ONE instance, not the class. (3) The 220-step empirical threshold was found on ONE seed
(44) under ONE task config (2 referents, 'dog'/'cat'); the 300-step margin is not re-derived for a k>2 or
different-filler configuration — an honest scope limit, not assumed to generalize further than tested.

## 6. Production status — unchanged; no default flipped

`_XEDGE_CD6_DEFAULT_ON` (base rung) stays `False`. `_CD6_SEMANTIC_DROP_DEFAULT_ON` (semantic-drop rung) stays
`True` in source, exactly as before this fix — it was already functionally inert in production (the parent gate
being off means `get_xedge_curiosity_d6_pool()` returns `None` on every live request, so `semantic_drop_current`
never fires) and remains functionally inert after this fix, for the identical, unrelated reason. **This finding
changes zero bytes of live `/api/brain-chat` behavior** — it only changes what a re-verify or a future re-
enablement of the parent gate would find. Flipping either flag is an owner/caller UX call, not made here.

## 7. Files

Modified: `research/runners/onebrain_xedge_curiosity_d6_production.py` (`_SEMANTIC_DROP_CLEAR_STEPS` constant +
`semantic_drop_current`'s returned duration), `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` (`onebrain-xedge-
curiosity-d6` row), `research/FAILURE_LOG.md` (2026-09-02 row, closure appended). New: this finding,
`research/findings/raw/_onebrain_xedge_curiosity_d6_semantic_drop_competitive_fix_6seed.json` (+ `.prov.json`),
`research/findings/raw/_onebrain_xedge_curiosity_d6_semantic_drop_rolebyposition_parity_6seed.json` (+
`.prov.json`). Unmodified: `research/runners/_multi_slot_binding_derisk.py` (`MultiSlotHold`, including its
shared `clear_steps`/`clear_gain`), `research/runners/d6_multiref_wm_production_organ.py`, `webapp/server.py`.
No `sim/` file touched; no production default changed.

Functional read-outs only; no phenomenal-experience claim.
