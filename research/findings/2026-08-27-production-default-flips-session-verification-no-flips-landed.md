---
type: finding
status: no-go
date: 2026-08-27
mechanism: production-default-flip-verification
lane: production-integration
artifacts:
  - research/findings/raw/_onebrain_xedge_session_leak_verify/summary.json
  - research/findings/raw/_onebrain_xedge_session_leak_verify/verify_session_leak.py
  - research/findings/raw/_onebrain_xedge_session_leak_verify/verify_session_leak_output.log
builds_on: research/findings/2026-08-27-onebrain-xedge-production-live-learning-GO.md
---

# Verify-then-flip pass over 3 default-off candidates — ZERO flips landed: xedge PART 2 blocked by a NEWLY-FOUND
cross-session state leak in the real production wiring, generative-wander's production-scale 6-seed verify is
still not landed, and the remaining ledger candidates are excluded by the task's own scope (superseded / matches
the excluded generation-time-honesty and confidence-forthcomingness categories)

**One-line:** this session verified three candidates for a default-off -> default-on flip
(`BRAIN_ONEBRAIN_XEDGE`+`BRAIN_ONEBRAIN_XEDGE_LEARN`, `BRAIN_CONTINUOUS_IDEATE_SPIKING`, and a scan of
`docs/PRODUCTION_INTEGRATION_LEDGER.yaml` for other de-risked+wired-but-off rows) and flipped NONE of them — each
for a verified, specific reason, not a blanket punt. The most consequential result is a NEWLY DISCOVERED defect in
xedge's real production wiring (a process-global cross-session state leak) that the underlying 6-seed GO finding's
own self-test never exercised, found by testing through the REAL production organs rather than trusting the
self-test's own controlled instrument.

## Candidate 1 — onebrain cross-edge PART 2 (`BRAIN_ONEBRAIN_XEDGE` + `BRAIN_ONEBRAIN_XEDGE_LEARN`) — NOT FLIPPED

[[2026-08-27-onebrain-xedge-production-live-learning-GO]] reports a 6-seed GO: the cross-edge grows from an
in-brain self-supervised credit signal and its live-learned weights flip the real production `repair_target` role
5/5 on ambiguous items, lesion-attributable 0/5. That GO stands as a characterization of the MECHANISM — this
session reproduces it. What this session adds is new: **verification through the real production call path**
(`d6_multiref_wm_production_organ.MultiReferentWMOrgan.judge`/`.load` and
`comprehension_production_organ.ComprehensionProductionOrgan.repair_target`, exactly as `webapp/server.py`'s
`/api/brain-chat` handler calls them at lines ~4926-4940 and ~5150-5176) rather than the finding's own
`_selftest_livelearn`, which manually sets `sh.xedge_focus = pa` / `pp` directly and never drives the write
through the real d6-organ code path, and never involves a second session.

### The defect: a process-global, never-cleared, cross-session WM-focus latch

`XedgeProductionPool.pool.xedge_focus` (`research/runners/onebrain_xedge_production.py`) is a single mutable
attribute on the ONE process-shared `MergedPool` singleton. It is written by
`d6_multiref_wm_production_organ.py:224` — `self._shared.xedge_focus = CAND_POOLS[0]` — inside
`MultiReferentWMOrgan.load()`, which fires whenever ANY session's turn introduces >=2 discourse referents. The
value written is a POSITIONAL CONSTANT (always `CAND_POOLS[0]`, regardless of which referents or which session)
— a declared residual the finding already named ("the live focus is a POSITIONAL proxy"). What was NOT
previously verified: **`xedge_focus` is never cleared by any live code path.** `XedgeProductionPool.clear_focus()`
exists but has zero callers in `webapp/server.py` or either organ. So once ANY conversation, anywhere in the
process's lifetime, holds 2+ referents even once, `xedge_focus` latches to `CAND_POOLS[0]` and stays there
forever — read by every SUBSEQUENT session's `comprehension_production_organ.repair_target()` call as if it
reflected THAT session's own held WM state.

### Reproduction (numpy, seed 42; full data in `research/findings/raw/_onebrain_xedge_session_leak_verify/summary.json`)

Built the real `XedgeProductionPool` (`get_xedge_pool(42)`, `BRAIN_ONEBRAIN_XEDGE=1` +
`BRAIN_ONEBRAIN_XEDGE_LEARN=1`) and the process-shared `comp_organ` it installs (confirmed `comp_organ is
pool_holder.comp_organ`, i.e. the SAME object `webapp/server.py`'s `_get_comprehension_organ()` would return).

1. **Session A** (a fresh `MultiReferentWMOrgan(seed=42, shared=pool.pool)`) called `.judge("The fox and the wolf
   walked in.")` — the real production hold path. `pool.xedge_focus` became `'w0'`. Session A's own ambiguous
   turn, `comp_organ.repair_target("The fox chased the wolf.")`, then read `role='agent'`, `wm_resolved=True`,
   `wm_margin=0.020000` (`content_role='patient'` — the WM state genuinely flipped it, matching the finding's claim).
2. **Session C**: a BRAND NEW `MultiReferentWMOrgan(seed=42, shared=pool.pool)` sharing the SAME process pool,
   whose own `_slot_of_ref` codebook was confirmed EMPTY (`{}`) — it never called `.load()`, never mentioned any
   referent. Given the IDENTICAL ambiguous sentence, `comp_organ.repair_target(...)` read `role='agent'`,
   `wm_resolved=True`, `wm_margin=0.017778` — indistinguishable from session A's own result, purely because session
   A's unrelated hold happened earlier in the SAME process.
3. **Lesion** (`pool.lesion_cross()`, zeroing the cross-edge weights in place): re-measuring the same sentence
   reverted `role` to `'patient'` (== `content_role`) with the `wm_resolved` key absent — the mechanism's own
   lesion-attributability is confirmed to still hold on the real call path.

This is the exact failure class the "vary the driver -> output differs; lesion -> vanishes" verification bar
exists to catch: the driver that actually varies in production is NOT "this conversation's own held WM referent"
(session C never held one) — it is "has ANY conversation, ever, held one" — a process-wide latch, not a
per-conversation signal. `PART 1` (`BRAIN_ONEBRAIN_XEDGE` without `_LEARN`,
[[2026-08-27-onebrain-xedge-production-frozen-GO]]) shares this SAME `d6_multiref_wm_production_organ.py` /
`comprehension_production_organ.py` coupling code (only the cross-edge's growth mechanism differs between PART 1
and PART 2), so this defect blocks flipping either part of `BRAIN_ONEBRAIN_XEDGE`, not just the `_LEARN` half.

### Why this blocks the flip

A multi-conversation production server (the stated end-state — `/api/brain-chat` serving real chat sessions) with
this wiring on would mean: user B's ambiguous-repair clarification wording is silently colored by whatever user A
(a totally unrelated conversation) happened to mention earlier, for as long as the process has been running. This
is not "hollow" (the mechanism is real, causally load-bearing, and lesion-attributable, as re-confirmed above) —
it fails the **moat-safe / no-cross-session-bleed** bar specifically, a DIFFERENT property than what the 6-seed
self-test measured. `research/FAILURE_LOG.md` carries a new row for this (below); a fix requires threading the
CALLING session's own held-referent state explicitly into `repair_target` (or scoping `xedge_focus` per-session)
rather than reading a shared mutable pool attribute — a real redesign of the coupling's calling convention, not a
one-line change, and out of scope for a same-session verify-then-flip pass. `BRAIN_ONEBRAIN_XEDGE` and
`BRAIN_ONEBRAIN_XEDGE_LEARN` stay at their existing default OFF; no ledger row exists for this faculty yet (it
was never added), so none was touched.

### A second, unrelated, honestly-noted observation (not xedge-caused)

The well-formed control sentence ("the dog chases the ball") read `comprehended=True` on the shared xedge pool's
`comp_organ` (margin 0.397222 vs threshold 0.328472) but `comprehended=False` on a STANDALONE `ComprehensionProductionOrgan`
built in the SAME process (margin 0.036806 vs threshold 0.248611). To rule out same-process RNG-order contamination
(many other builds happened first in that process) as the explanation, a truly fresh process was run where the
xedge module was never even imported: it ALSO read `comprehended=False` on this sentence (margin 0.037500 vs
threshold 0.248611; `well_formed_judge_OFF_fresh_process_baseline` in the cited artifact) — identical to the
same-process standalone result. This confirms the False reading is a **pre-existing property of
`ComprehensionProductionOrgan(seed=42)`'s own calibration on this specific sentence**, not something xedge
introduces or worsens (the shared-pool topology differs from the standalone one — 47 regions/1752 neurons vs 12
regions/1032 neurons — so its own numbers differ too, but that is a separate, already-existing fact about how this
organ's calibration depends on which pool it is built into, not a xedge regression). Left open for whoever next
does a full production verification pass on the comprehension organ generally; not a blocker for this candidate's
(already NO-GO) verdict.

## Candidate 2 — generative mind-wander (`BRAIN_CONTINUOUS_IDEATE_SPIKING`) — NOT FLIPPED, per the finding's own condition

[[2026-08-27-generative-wander-production-scale-PARTIAL]] explicitly conditions any flip on its own staged 6-seed
production-scale (n_ca3=2000, emergent DG-selected) GPU verify landing first, naming its own expected output path
(the directory `research/findings/raw/_generative_attractor_wander_onsubstrate/` plus the filename
`production_n_ca3_2000_6seed.json`, deliberately not written as one contiguous path here — same reason the source
finding itself split it across two lines — so this NOT-YET-EXISTING path is not misread as an already-cited
artifact) as NOT-YET-EXISTING at write time. Checked this session: that file still does not exist
(the directory above contains only `batch1.json`/`batch2.json`, the REDUCED-scale n_ca3=400 GO's own artifacts).
Checked the `research/generative-wander-production` branch
(`origin/research/generative-wander-production`, `gitea/research/generative-wander-production`): its HEAD commit
(`a4e2a015e`) is exactly the PARTIAL finding's own landing commit — no follow-up commit with the 6-seed result
exists on that branch either. Per the task's own instruction ("flip ONLY if that verdict is GO, else leave off +
say why") and the finding's own explicit `seed-waiver`, this candidate stays OFF. `BRAIN_CONTINUOUS_IDEATE_SPIKING`
is unchanged (default OFF); no ledger row was touched.

## Candidate 3 — scan of `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` for other de-risked+wired-but-off rows — NONE FLIPPED

Scanned all 55 ledger rows programmatically for `de_risked: YES` AND `wired: YES` AND `on_by_default: NO`. Three
matched, and each is excluded by the task's own scope or the row's own content, not a fresh judgment call:

1. **`gnw-thought-swap`** — the row's own text says it is `⚠️ SUPERSEDED by swap-drives-response` (already
   `on_by_default: YES`), and that the observe-only fallback this row's flag used to gate was **REMOVED from
   `webapp/server.py brain_chat` on 2026-08-20** ("the fallback ... is REMOVED"). Flipping `_GNW_SWAP_DEFAULT_ON`
   would toggle a flag with zero live callers left — not a genuine flip, a no-op on dead code. Left as-is; the
   row is explicitly "KEPT (ratchet count) as the historical #77 observer entry."
2. **`source-provenance-honesty`** — matches the task's own exclusion ("DO NOT flip: generation-time honesty
   (PARTIAL)") directly: the row's own `on_by_default: NO` note names two un-flipped sub-rungs, the second of
   which reads verbatim "this landing is wire-in only, not a flip proposal — do not read this note as flipping
   it." Left OFF per the task's explicit instruction, not re-litigated.
3. **`confidence-forthcomingness`** — matches the task's own exclusion verbatim ("a separate agent is re-testing
   it"). Left OFF, untouched, so as not to race a concurrent verification.

## Summary

| candidate | flipped? | reason |
|---|---|---|
| onebrain-xedge PART 1 + PART 2 (`BRAIN_ONEBRAIN_XEDGE`[`_LEARN`]) | **NO** | newly-discovered cross-session `xedge_focus` latch (this session) — mechanism is real + lesion-attributable, but the real wiring is not session-isolated |
| generative-wander (`BRAIN_CONTINUOUS_IDEATE_SPIKING`) | **NO** | production-scale 6-seed GPU verdict still not landed (file + branch both checked, absent) |
| gnw-thought-swap | **NO** | superseded / dead flag, no live callers left to flip |
| source-provenance-honesty | **NO** | matches the task's own excluded "generation-time honesty (PARTIAL)" category |
| confidence-forthcomingness | **NO** | explicitly excluded by the task (concurrent re-test) |

No `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` row was edited this session (no `on_by_default` value changed from NO
to YES) because no flip landed. No `sim/` file was touched. No production code was edited.

## Memory discipline

`free -m` checked before the one brain-load verification run in this session (available ~25-29GB throughout,
comfortably above the ~13GB floor); no faculty verification was deferred for memory reasons this session.
