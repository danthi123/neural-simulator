---
type: finding
status: contributing
date: 2026-08-20
mechanism: dendritic-plateau-coincidence-burst
lane: integration
integration_faculty: d5-live-consolidation
seeds: [42, 43, 44, 100, 101, 102]
seed-waiver: The 6/6 is on the robustness-reserve teeth; the load-bearing DECISION here is a production-integration
  verification (default-off byte-identity + ON-path crash-rollback + the handler-signal-flat boundary), not a stochastic
  effect size. The mechanism is the arc-1 6-seed GO already banked ([[2026-08-20-d5-learn-through-use-recall-driven-plateau-gated-BTSP-strengthens-a-memory-6seed-GO-arc1-closed]]).
instrument: research/runners/_d5_live_consolidation_integration_derisk.py — wires the arc-1 recall→window→BTSP loop
  under webapp/continuous_engine.py's idle tick (default-off behind BRAIN_D5_CONSOLIDATE) and measures a used memory's
  robustness before/after consolidation with a same-store flag-OFF lesion control.
runner: research/runners/_d5_live_consolidation_integration_derisk.py
external: NO-EXTERNAL-NEEDED — a production-integration wiring of the in-repo arc-1 GO; no literature question.
artifacts:
  - research/findings/raw/_d5_live_consolidation/summary_6seed.json
  - research/findings/raw/_d5_live_consolidation/seed42.json
  - research/findings/raw/_d5_live_consolidation/seed43.json
  - research/findings/raw/_d5_live_consolidation/seed44.json
  - research/findings/raw/_d5_live_consolidation/seed100.json
  - research/findings/raw/_d5_live_consolidation/seed101.json
  - research/findings/raw/_d5_live_consolidation/seed102.json
---
# D5 live-consolidation is WIRED (default-off, ON-path-safe) and strengthens a used memory's ROBUSTNESS RESERVE — but NOT yet the conversation OUTPUT (the step-5 residual)

Artifact: research/findings/raw/_d5_live_consolidation/summary_6seed.json + the per-seed
research/findings/raw/_d5_live_consolidation/seed42.json … seed102.json.

**One line.** Step-4 (the production-integration rung) wires the arc-1 recall→self-terminating-window→BTSP loop under
`webapp/continuous_engine.py`'s idle tick, default-off. Adversarial verification (4 lenses) confirms the mechanism is
real and safe-when-off, and **corrects an overclaim**: the strengthening lives in the memory's ROBUSTNESS RESERVE, NOT
in the production recall OUTPUT — the handler-visible signal is FLAT pre/post on 5/6 seeds — so it is **not yet
load-bearing on the conversation** (the owner's "metadata moves, conversation doesn't" bar). It also found and fixed an
**ON-path crash-corruption defect**. Banked default-off with the honest scope; the production-default flip is deferred.

## What is wired (additive, default-off, brain-based; NO `sim/` edit)
`consolidate_used_memory(cache_key, organ)` runs the arc-1 step-3 loop (re-activate → step-2 apical-plateau window →
the substrate's OWN plateau-gated BTSP `fused_btsp_update`) on the organ's real store (`mem.R.C.data` by object
identity), invoked from `tick_idle_sessions` for each idle session that recalled a topic; `mark_recall` records what a
turn recalled. Flag `BRAIN_D5_CONSOLIDATE` (default `"0"` → off; unset = off), budget `BRAIN_D5_CONSOLIDATE_BUDGET`
(default 1). The tick NEVER builds an organ (uses the already-built one or skips). `webapp/server.py` calls
`mark_recall` guarded by the flag + `in_memory`. Diffs additive (continuous_engine +~175, server +23); `sim/` untouched.

## What is PROVEN (6/6, adversarially verified — no confound on these)
<!--derived-->
- **Robustness-reserve strengthens through use:** a memory recalled in turn T, consolidated between turns, survives a
  bigger within-recurrence lesion (seed42 max-lesion-survived 0.5→0.9) and completes from a sparser cue (160→120 pA);
  robustness reserve seed42 340→780, weight-attributable (w_dog within-assembly +13→+22 via the substrate BTSP).
- **Vanishes exactly under lesion:** flag `"0"` → `consolidate_used_memory` returns None → store hash byte-identical
  (`hash_off == hash_before` all 6), the later read identical, robustness-gain 0.0. So the gain is DRIVEN by the loop.
- **Default-off byte-identity:** the whole D5 path is triple-gated (server `mark_recall` guarded; tick short-circuits on
  `_d5_on`; `consolidate_used_memory` returns None before any state read) → off = HEAD byte-for-byte.
- **Brain-based:** the weight change is the substrate's own `fused_btsp_update` (no host `dw` formula), writing the real
  store by object identity; `sim/` git-diff empty. Specific (never-recalled 'cat' within-weight ~unchanged).

## The CORRECTION (the honest boundary — do not overstate)
<!--derived-->
The GO-deciding teeth (max-lesion-survived, min-cue-current) are read by the arc's `reactivate(strengthen=False)`
completion-MARGIN instrument on the store's robustness reserve — **NOT** by the production `EpisodicRecallOrgan.recall()`.
The handler-visible signals (`apical_cue` / `in_memory` → the reply text) are **FLAT pre/post on 5/6 seeds** (apical_cue
unchanged: seed42 0.7143, seed43 0.9231, seed100/101/102 1.0; only seed44 moves 0.833→1.0). Because the production
handler drives a FULL cue through the UNDEGRADED store, the reply is byte-identical before/after consolidation on 5/6.
⇒ the mechanism is load-bearing on the store's robustness RESERVE, **not yet on the conversation OUTPUT** — the exact
"metadata moves, conversation doesn't = hollow integration" pattern the owner's bar
([[feedback_faculties_must_drive_not_observe]]) forbids. The earlier "measured through the live handler / load-bearing
on the conversation" framing was an OVERCLAIM, corrected here and in the runner docstring.

## The ON-path safety fix (found + fixed + verified)
The reactivate loop mutates the PERSISTENT store (`bridge.cp_connections.data`) in place across episodes; the original
`finally` restored only cfg/BTSP, so a crash mid-loop (the RTX 3090's documented "falls off the bus mid-load") would
corrupt the store with no rollback, leave the topic un-drained (next tick retries from corrupted weights), and be
swallowed by a bare `except: pass`. FIXED: capture `W_pre` before the try; on any failure restore
`bridge.cp_connections.data[:] = W_pre` + drain the topic + re-raise; the tick logs it. **Verified on cupy:** a
simulated mid-loop crash now rolls the store back byte-identically (hash restored) + drains the topic, with the happy
path intact. This was a hard precondition for any flag-ON soak.

## Honest scope + NEXT (step-5, before any production-default flip)
Corrected caveat: production `note_topic` forms memories at full one-shot strength (train_events=40) vs this de-risk's
borderline 15 — but an independent train=40 reproduction gave 4/6 GO with headroom (NOT the "1/3, no headroom" the
build report stated), so "load-bearing only for under-consolidated stores" is TOO STRONG; at full strength the reserve
still strengthens on most seeds, but the handler-visible reply is unchanged. The real reasons to hold default-off:
(1) the conversation-visible signal does not move yet (the boundary above); (2) a multi-turn soak + the chat
no-regression suite are still required; (3) the organ's non-idempotent recall read (repeated reads drift — the step-2
residual) is handled by snapshot isolation in the instrument but is a live-flow consideration. **Step-5:** find a regime
where the PRODUCTION recall (`in_memory` / `apical_cue` / the reply) actually moves — e.g. a weaker initial encode so a
freshly-spoken memory starts under-consolidated AND a later partial/degraded cue exposes the reserve — then soak +
no-regression, THEN the owner-UX default-on flip. (Agent-wired + agent-verified GO; parent adversarially re-verified,
CORRECTED the load-bearing overclaim to the robustness reserve, FIXED + tested the ON-path rollback, and banked
default-off.)
