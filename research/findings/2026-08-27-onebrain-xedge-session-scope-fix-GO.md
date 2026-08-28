---
type: finding
status: go
date: 2026-08-27
mechanism: onebrain-xedge-session-scope-fix
lane: one-brain/integration/production
artifacts:
  - research/findings/raw/_onebrain_xedge_session_leak_verify/summary_postfix.json
  - research/findings/raw/_onebrain_xedge_session_leak_verify/verify_session_leak_fixed.py
  - research/findings/raw/_onebrain_xedge_session_leak_verify/verify_session_leak_fixed_output.log
  - research/findings/raw/_onebrain_xedge_session_leak_verify/selftest_part1_seed42_output.log
  - research/findings/raw/_onebrain_xedge_session_leak_verify/selftest_part2_seed42_output.log
  - research/findings/raw/_onebrain_xedge_session_leak_verify/selftest_part3_seed42_output.log
builds_on: research/findings/2026-08-27-production-default-flips-session-verification-no-flips-landed.md
---

# The one-brain cross-edge's cross-session `xedge_focus` leak is CLOSED — a session-scoped calling convention,
verified through the real production organs: a fresh session reads NO focus, the holding session's own
PART 1/2/3 behaviour is unchanged, teardown clears it, the lesion still severs it, GO

Artifact: `research/findings/raw/_onebrain_xedge_session_leak_verify/summary_postfix.json`.

seed-waiver: this is a CODE-LOGIC / calling-convention correctness fix, not a seed-dependent biological effect
size — the claim is "a shared mutable attribute is no longer read/written across sessions", checked by direct
equality of the resolved focus value and presence/absence of a `wm_resolved` key, both deterministic given a
fixed seed and independent of which specific network realization that seed produces. Confirmed reproducible
(identical output) across two separate process invocations at seed 42 before finalizing this script. The PART
1/2/3 regression re-runs are explicitly labelled single-seed sanity checks of an ALREADY-established (6-seed GO,
`[[2026-08-27-onebrain-xedge-production-frozen-GO]]` / `[[2026-08-27-onebrain-xedge-production-live-learning-GO]]`)
mechanism, not a new generalisation claim.

**One-line:** `research/FAILURE_LOG.md`'s 2026-08-27 row documented a process-global, never-cleared
`XedgeProductionPool.pool.xedge_focus` latch that let one session's held WM referent silently color another,
unrelated session's comprehension-repair wording. This session re-scoped the calling convention so a session's
xedge focus lives on that session's OWN `MultiReferentWMOrgan` instance and is threaded EXPLICITLY into every
`comprehension_production_organ` call, never read off the shared process pool — closing the leak while
preserving the PART 1/2/3 mechanisms `[[2026-08-27-onebrain-xedge-production-frozen-GO]]` and
`[[2026-08-27-onebrain-xedge-production-live-learning-GO]]` established.

## The defect (recap)

`MultiReferentWMOrgan.load()` wrote `self._shared.xedge_focus = CAND_POOLS[0]` directly onto the ONE
process-shared `MergedPool` whenever ANY session's turn held >=2 discourse referents, and nothing ever cleared
it. `comprehension_production_organ.ComprehensionProductionOrgan` is ALSO a process-shared singleton
(`_ORGAN`), so its `judge`/`repair_target` read that SAME attribute for EVERY session's turn — a session with an
empty referent codebook read whatever the last session to hold a referent, anywhere in the process's lifetime,
had written.

## The fix

Three files, additive, no `sim/` edit:

* `research/runners/d6_multiref_wm_production_organ.py`: `MultiReferentWMOrgan` now stores its own xedge focus on
  `self._own_focus` (an instance attribute) instead of writing `self._shared.xedge_focus`. `current_focus()` /
  `clear_focus()` expose it. Since `webapp/server.py` already keeps exactly one `MultiReferentWMOrgan` per
  `cache_key` (`_SESSION_MULTIREF`, the SAME per-session-isolation pattern the task pointed at), an instance
  attribute is already correctly session-scoped — no new dict/keying scheme was needed.
* `research/runners/comprehension_production_organ.py`: `judge`, `repair_target`, `read_margin`, and the internal
  `_read`/`_read_per_noun`/`_xedge_codrive`/`_wm_resolved_role` now take an explicit `wm_focus` argument. A
  `_WM_FOCUS_UNSET` sentinel keeps the OLD ambient-`self._shared.xedge_focus` read as a fallback ONLY for callers
  that omit the argument (the offline self-tests below, which still directly assign `sh.xedge_focus` for
  controlled, single-session testing) — production always passes an explicit value now, so it never consults the
  shared attribute.
* `research/runners/onebrain_xedge_production.py`: `XedgeProductionPool.credit_live_turn` and
  `credit_live_turn_from_comprehension` (PART 3's per-turn live-plasticity hook) take the same explicit
  `focus`/`wm_focus` argument with the identical `_FOCUS_UNSET` sentinel fallback.
* `webapp/server.py`: `/api/brain-chat` hoists `d6org` (this turn's own `_get_multiref_organ(cache_key)`) to
  function scope, resolves `_wm_focus = d6org.current_focus() if d6org is not None else None` once, and threads
  it into `corg.judge(...)`, `corg.repair_target(...)`, and `credit_live_turn_from_comprehension(...)`.

`XedgeProductionPool.pool.xedge_focus` (and `set_focus`/`clear_focus`) remain on the pool as a LEGACY scratch
attribute so the offline self-tests (`_selftest_livelearn`, `_selftest_perturn`) — single-session, sequential,
never exercising cross-session isolation — keep running unmodified; production code no longer reads or writes
it (verified below: it stays `None` throughout a real-production-style run).

## Verification (numpy, seed 42; full data in the cited artifacts)

Built the real `XedgeProductionPool` via `BRAIN_ONEBRAIN_XEDGE=1` (PART 1, frozen host-trained edge — chosen over
PART 2/3's per-turn default because that build leaves the cross-edge uniform at W0=0.05 until real credited
turns accrue, giving a near-zero, borderline `wm_resolved_role` margin unrelated to this fix; PART 1 shares the
identical `d6_multiref_wm_production_organ.py` / `comprehension_production_organ.py` coupling code the leak was
found in) and the SAME process-shared `comp_organ` `webapp/server.py`'s `_get_comprehension_organ()` returns.

1. **Own-session resolution preserved.** Session A held 2 referents through the real `.judge()` call path;
   `orgA.current_focus() == 'w0'`. Passing that explicitly, `comp_organ.repair_target(..., wm_focus='w0')` read
   `role='agent', wm_resolved=True` (content_role was `'patient'` — the WM state still flips it), matching the
   PART 1/2 mechanism unchanged.
2. **Leak closed.** A brand-new `MultiReferentWMOrgan` (session C) sharing the SAME process pool, confirmed
   `_slot_of_ref == {}` (never held anything), read `orgC.current_focus() is None`, and
   `repair_target(..., wm_focus=None)` carried NO `wm_resolved` key at all — indistinguishable from xedge being
   off, regardless of session A's earlier hold in the same process. (Pre-fix, this exact scenario read
   `wm_resolved=True/role='agent'`, per the sibling `summary.json`.)
3. **Teardown clears it.** Dropping session A's organ (mirroring `webapp/server.py`'s
   `_SESSION_MULTIREF.pop(cache_key)` on a conversation reset) and building a fresh organ for the same slot
   starts `current_focus() is None` again; re-holding re-establishes `wm_resolved=True`.
4. **Lesion still severs it.** `pool_holder.lesion_cross()` reverted session A's own held-focus read to
   `role='patient'` (== `content_role`) with no `wm_resolved` key — the mechanism's lesion-attributability,
   established by PART 1's own `_selftest_loadbearing`, is unaffected by the refactor.
5. **Flag-off path unchanged.** With `BRAIN_ONEBRAIN_XEDGE=0`, an omitted `wm_focus` (every pre-existing caller's
   style) and an explicit `wm_focus=None` (what production now always passes) resolve to the identical value
   (`None`), verified by direct equality of the resolved value — a hash/exact compare of the decision variable,
   not an inference from reading the code — and both reads carry no `wm_resolved` key. Full numeric output
   equality across two separate `repair_target` calls on one organ was NOT used as the instrument: back-to-back
   reads on the SAME organ are not bit-reproducible even before this fix (`_hard_reset` restores membrane
   potentials but not an ongoing background-noise process), so exact-output comparison would have been the wrong
   test; the resolved-focus-value equality is the precise claim this fix makes.

**No regression in PART 1/2/3.** Re-ran the module's own offline self-tests at seed 42 after the refactor:
`--seeds 42` (PART 1) `lesion_attributable=True`; `--verify-live --seeds 42` (PART 2) `GO=True,
caveat_closed=True, flips(intact=5/5, lesioned=0/5)`; `--verify-per-turn --seeds 42` (PART 3) `GO=True,
taught_role_signs_later_read=True, lesion_no_accumulation=True`. These are single-seed sanity re-runs of the
existing mechanism (not a new 6-seed claim) confirming the calling-convention refactor did not disturb it.

Also confirmed `webapp/server.py` still imports cleanly and the pre-existing
`tests/test_comprehension_learned_animacy_cue.py::test_flag_off_byte_identical_to_pre_existing_scope` (which
calls `organ.judge(TEXT)` with no `wm_focus` kwarg at all) still passes unmodified — the new parameter's default
does not break any existing caller.

## Residual (noted, not fixed here)

`webapp/gnw_three_organ_bus.py`'s `_comprehension_vote` also calls `corg.read_margin(...)` on the SAME
process-shared comprehension organ, without a `wm_focus` argument — it was ALSO exposed to the same leak class
whenever `BRAIN_ONEBRAIN_XEDGE` and the (default-ON) `BRAIN_GNW_3ORGAN` gate were both live, since that call
path has no session-identity concept to plumb `wm_focus` from today. Removing the write side closes this
exposure too (it now permanently resolves `foc=None` there, since `pool.xedge_focus` is never written in
production anymore), which is strictly safer than before, but a principled fix (threading a real per-session
focus into `three_organ_combine`) would need `gnw_three_organ_bus.py`'s calling convention extended with a
session identity it does not carry today — out of scope for this session (`BRAIN_ONEBRAIN_XEDGE` is default OFF,
so this is not currently reachable in production).

## Scope

Additive only. No `sim/` edit. `BRAIN_ONEBRAIN_XEDGE`/`BRAIN_ONEBRAIN_XEDGE_LEARN` remain default OFF — this
finding closes the specific isolation defect that blocked the candidate-1 flip in
`[[2026-08-27-production-default-flips-session-verification-no-flips-landed]]`; it does not itself flip the
default (a separate flip-soak pass, per that finding's own instruction, is the next step).
