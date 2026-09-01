---
type: finding
status: go
date: 2026-09-01
mechanism: EMERGENT free-slot-wins register allocation for the D6 multi-referent working-memory buffer — before
  each write, a genuine zero-input `cp_firing_states` probe over every register's current band-max activity
  (`MultiSlotHold.probe_occupancy()`) replaces the role-by-position host MARKER; the new referent is routed to
  `argmin(occupancy)`, the register the substrate itself currently shows as free, not the referent's position in
  the sentence/call
lane: working memory / conversation (D6 — which slot binds which referent)
verdict: GO — 6/6 seeds. >=2 referents introduced together land in DISTINCT registers via the occupancy-read
  argmin, invariant to mention order (both AB and BA separate and both fully recover); a referent introduced
  AFTER an already-held ANCHOR referent (no intervening reset) correctly avoids the anchor's occupied register
  too — the non-trivial case a pure position marker cannot handle, because it never reads occupancy at all. The
  selection-only LESION (register 0 forced regardless of the probe, HOLD recurrence untouched) collapses the
  separation into the already-validated SUPERPOSED-collide regime on every seed (collision count 6/6 lesioned
  vs 0/6 intact — load-bearing). `competitive=False` (the untouched default) reproduces the exact pre-existing
  role-by-position registers on every seed (byte-identical-off, additive).
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_d6_wm_competitive_slot_binding_verify.py
organ: research/runners/d6_multiref_wm_production_organ.py
artifacts:
  - research/findings/raw/_d6_wm_competitive_slot_binding/verify_6seed.json
depends_on:
  - 2026-08-11-multi-slot-variable-binding-working-memory-holds-k-bindings-no-crosstalk-ceiling-k5-6seed-GO.md
  - 2026-08-12-D6-multiref-WM-production-organ-holds-two-plus-referents-lesion-load-bearing.md
builds_on:
  - 2026-08-28-onebrain-xedge-position-invariant-indirection-closes-positional-residual.md
---

# D6 multi-referent WM: register ALLOCATION is now an emergent substrate read, not a host position marker — the free/least-active register wins a new referent, 6-seed GO

## The frontier (verify-first)

The D6 multi-referent WM organ ([`2026-08-12-D6-...`](2026-08-12-D6-multiref-WM-production-organ-holds-two-plus-referents-lesion-load-bearing.md)) declares an open residual: "the register assignment is today a role-by-position host MARKER (referent 0 -> reg0, ...)". The 2026-08-28 onebrain-xedge indirection finding closes a DIFFERENT residual (the comprehension organ's *read* of an already-held referent's grown role, position-invariantly) and explicitly leaves this one open: "the SEMANTIC referent->pool binding... remains a DECLARED residual". RAG (`finding`/`kandel` corpora) + a grep of `research/runners/d6_multiref_wm_production_organ.py` confirmed the residual: `load()` wrote `for r, loc in enumerate(locals_): buf.write(r, loc)` — referent `r` (its position in the input list) drove register `r` directly, with **zero** substrate read in the decision. No in-flight branch touches this: `research/xedge-strengthen-wm-resolve-margin` (a live worktree, `agent-aa0d75869b2bc5ad2`) fixes a cupy backend/pool-build bug in the comprehension cross-edge's *margin read*, and `research/132-binder-integrate` is a closed, unwired gap#2 delta-rule binder assessment for the moat verifier — neither is the register-allocation residual. This session picks it up.

## The mechanism (additive; NO sim/ edit; reuse-by-import)

`MultiSlotHold.probe_occupancy()` (new method, `research/runners/_multi_slot_binding_derisk.py`) is a genuine zero-input read of every register's current band-max firing rate in one pass — the exact instrument class `read()` already uses (`cp_firing_states`, external input asserted zero), generalised across all `R` registers instead of one. The D6 organ's `load()` (`research/runners/d6_multiref_wm_production_organ.py`) gained a `competitive` mode (`BRAIN_MULTIREF_COMPETITIVE`, default-OFF): instead of `referent i -> register i`, each write is routed to `argmin(probe_occupancy())` — the register the brain's own current state shows as most free. Because a just-written register's post-hold activity is measurably above baseline (the D6 GO's own `hold_alive_min` ~0.06), the NEXT referent's probe correctly avoids it without any host "claimed" bookkeeping — the exclusion is read, not remembered. `BRAIN_MULTIREF_COMPETITION_LESION` ablates ONLY this selection (forces register 0 every write, ignoring the probe) — distinct from the pre-existing `BRAIN_MULTIREF_LESION` (`recur=0`, kills the HOLD's own recurrence). This substrate has no background OU noise (`ou_std_current_pA=0` in `build_persistent_slot`), so a probe over an all-baseline bank ties exactly and breaks to the lowest free index — a real, measured tie (not a formula), but deterministic absent prior occupancy. The ANCHOR scenario below is the non-trivial case where prior occupancy genuinely differs and the read demonstrably steers the allocation.

## Verify (6 seeds — `research/findings/raw/_d6_wm_competitive_slot_binding/verify_6seed.json`)

Four scenarios per seed, each run twice (both referent-mention orders) where relevant:

| test | intact (competitive=True) | lesioned (competition_lesion=True) |
|---|---|---|
| k=2 TOGETHER, distinct registers, both orders | 6/6 seeds | — |
| k=2 TOGETHER, both fully recovered, both orders | 6/6 seeds | 0/6 seeds (collide) |
| k=3 TOGETHER, distinct + fully recovered | 6/6 seeds | — |
| ANCHOR (pre-occupied register + 2 new refs, both orders): avoids occupied register | 6/6 seeds | — |
| ANCHOR: all 3 referents recovered | 6/6 seeds | 0/6 seeds (collide) |
| `competitive=False`: byte-identical to the pre-existing `[0,1,2,...]` registers | 6/6 seeds | n/a |

Collision count (>=1 referent not recovered OR two referents sharing a register): **6/6 seeds lesioned vs 0/6 seeds intact** — the load-bearing lever the runner reports as `MOVED`. The lesion reproduces the already-validated SUPERPOSED-single-slot collide regime (`_multi_slot_binding_derisk.eval_superposed_single`) as a genuine collision here (both referents forced into register 0's local competition; the read-back keeps only the local-WTA winner), not merely assumed. All 6 seeds pass every sub-check (`seed_pass=True` for 42/43/44/100/101/102).

## Honest residuals (unchanged by this session; declared, not closed)

What moved: **which register** a referent lands in is now decided by a substrate read, not a host loop index. What did NOT move (same as the parent D6 finding): referent EXTRACTION (which tokens count as referents) stays a host lexicon parse; the referent<->LOCAL-slot BIND stays the host-numpy RUNG6c `HebbianBinder`; the register READ stays a host argmax over firing rate (the same read-out-instrument class used throughout this codebase's honest functional read-outs — affect/comprehension/metacog). This is an allocation-POLICY change on an already-grounded substrate (the D3 slow-NMDA hold + shared-FS competitive registers), not a new biological mechanism — the "no host code supplies the answer" bar for `self-organized` (`docs/TERMS.md`) is **not** claimed here: `argmin` over a genuine spike read is the same read-out-instrument class as every other D6 register read, not a self-organizing learning rule. The tie-break for a FIRST referent into an all-baseline bank is deterministic (lowest free index) absent prior occupancy — an honest property of a substrate with no background noise, not a flaw; the ANCHOR scenario shows the mechanism is genuinely occupancy-driven once occupancy differs. Additive, default-OFF (`BRAIN_MULTIREF_COMPETITIVE`); **not wired, not flipped default-on** — this session's scope was the allocation mechanism + its verification, not production integration.
