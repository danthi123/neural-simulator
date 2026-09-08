---
type: finding
status: go
claim_check: measured-result
date: 2026-09-08
mechanism: thread onebrain_k_max through load_developed_brain + MultiTurnAgent + webapp/server.py's loader, auto-sized from the bundle's own fact count (additive, default-preserving)
lane: scaffold-retirement (backlog rank-1, load-path blocker closure)
seeds: [42]
artifacts:
  - research/findings/raw/_onebrain_kmax_loadpath_thread/verify_smoke.json
  - research/findings/raw/_onebrain_kmax_loadpath_thread/rf_byteident_before.json
  - research/findings/raw/_onebrain_kmax_loadpath_thread/rf_byteident_after.json
  - research/runners/_onebrain_kmax_loadpath_verify.py
  - research/runners/_onebrain_kmax_rf_byteident_check.py
---

# `onebrain_k_max` load-path thread — the rank-1 GO finding's named blocker is CLOSED (additive, `'rf'` path proven byte-identical by hash)

<!--derived-->
<!-- Every number below is a read-out of the cited artifacts (verify_smoke.json, rf_byteident_before.json,
     rf_byteident_after.json) -- a human-readable summary of machine artifacts -> derived, per the claim_check
     gate's own guidance. -->

**Board/lane: scaffold-retirement backlog RANK-1 (the owner's #1 arc).**
`research/findings/2026-09-08-rank1-composer-rebuild-rf-to-onebrain-real-bundle-parity-GO.md` established GO
parity/recall/moat for the spiking `onebrain` composer against the deployed `rf` bundle, but named its own
production-flip blocker verbatim: *"`load_developed_brain` hardcodes `onebrain_k_max=None -> 32`, so a >32-fact
onebrain bundle would reload truncated."* This finding closes that blocker: `onebrain_k_max` is now threaded
through the WHOLE developed-brain load path, additively, with the `'rf'` production default path proven
byte-identical by hash (not just "expected unchanged" — docs/TERMS.md's own bar for the word).

## Verify-first: the blocker was WORSE than "truncated" — it CRASHES

Re-running `load_developed_brain` against the rank-1 finding's own rebuilt 404-fact onebrain bundle
(`bridges/developed/rebuilt_scale787_onebrain`, untracked/regenerable, `composer_kind='onebrain'`, no
`kb_composites.npz` by design since onebrain composites live on-substrate) on the pre-fix code raised:

```
RuntimeError: OneBrainComposer store full: k_max=32 reached (shard or raise k_max)
```

at fact #33 of 404 — a hard crash, not a silent truncation, because the composite fast-path (`_restore_facts`'s
direct-set) never applies to a from-scratch onebrain rebuild (no persisted composites to direct-set), so every
fact re-stores via `comp.store()`, which raises the moment `k_max` is reached. The deeper root cause: NEITHER
`load_developed_brain` NOR `MultiTurnAgent` (the only class `load_developed_brain(use_multiturn=True)` — the
ONLY path `webapp/server.py`'s `_build_chat_brain` ever calls — constructs) had an `onebrain_k_max` parameter
at all. `BrainConversationalAgent(onebrain_k_max=...)`'s seam existed but was UNREACHABLE from the production
load path; only its own hardcoded `None -> 32` default ever arrived at the composer.

## The fix (additive, default-preserving)

- `research/runners/multi_turn_agent.py` — `MultiTurnAgent.__init__` gained `onebrain_k_max=None`, forwarded
  to the inner `BrainConversationalAgent` (previously dropped on the floor entirely).
- `research/runners/developed_brain_io.py::load_developed_brain` — gained `onebrain_k_max=None`. When `None`
  AND the resolved `composer_kind == 'onebrain'`, it is auto-sized to `len(facts) + 16` (this bundle's OWN fact
  count, matching `_rank1_composer_rebuild_onebrain_verify.py`'s own `k_max=n_facts+16` convention) instead of
  falling through to the composer's hardcoded 32. Passed into both the `MultiTurnAgent` and bare
  `BrainConversationalAgent` construction branches.
- `webapp/server.py` — new `_onebrain_k_max_override()` (env `BRAIN_ONEBRAIN_K_MAX`, unset -> `None` -> the
  auto-sizing above) threaded into `_build_chat_brain`'s `load_developed_brain(...)` call, mirroring the
  file's existing `BRAIN_INTEGRATED_LOOP` / `BRAIN_LTM_CODEBOOK_CACHE` override pattern.

`onebrain_k_max` is read ONLY inside `BrainConversationalAgent.__init__`'s `composer_kind == 'onebrain'`
branch (by construction, unconditionally on every other branch) — every other `composer_kind`
(`'rf'`/`'rate'`/`'slotbinder'`, including a bundle's own saved `'rf'` default) never touches the value, so
this is a no-op there regardless of what is passed. `composer_kind` itself is untouched — the production
default stays `'rf'`.

## Verification (`research/runners/_onebrain_kmax_loadpath_verify.py`, `SIM_BACKEND=numpy`, seed 42)

| check | result |
|---|---|
| bundle fact count (`facts.json`, ground truth) | **404** |
| AFTER-FIX reload: `k_max` (auto-sized) | **420** (404+16) |
| AFTER-FIX reload: facts recalled (`composer.kb` length) | **404 / 404** — full recall, no truncation |
| BEFORE-FIX repro (`onebrain_k_max=32` forced, the prior hardcoded value) on the SAME bundle | **crashes**: `OneBrainComposer store full: k_max=32 reached` — proves the fix is load-bearing, not a no-op |
| `'rf'` production bundle (`scale787/day_33`) reload: composer class | `RFPhasorComposer` (this runner sets `BRAIN_COMPOSER_MERGE=0`, matching `_rank1_composer_rebuild_onebrain_verify.py`'s own convention of testing the bare composer; see the separate hash check below for the `BRAIN_COMPOSER_MERGE` default-on production config) |
| `'rf'` production bundle reload: facts recalled | **404 / 404** (== `manifest_n_facts`) |
| `'rf'` production bundle: has an `onebrain_k_max`-relevant `k_max` attribute | **No** (`RFPhasorComposer` carries no `k_max` concept) |

Full artifact: `research/findings/raw/_onebrain_kmax_loadpath_thread/verify_smoke.json` (`verdict: "GO"`, all
three `go_flags` True).

## `'rf'` byte-identical, proven by hash (not inferred from reading the code)

docs/TERMS.md requires `byte-identical` be asserted IN THE DATA. `research/runners/_onebrain_kmax_rf_byteident_check.py`
loads the `'rf'` production bundle via `load_developed_brain` and hashes the resulting composer's entire stored
state (every fact dict + its composite array bytes, `sha256`). Run once with the code at `HEAD~1` (before this
change; file-level `git checkout HEAD~1 --` inside this task's own worktree) and once at `HEAD` (after), same
env, same bundle:

| tag | composer class | facts recalled | `kb_sha256` |
|---|---|---|---|
| before_fix (HEAD~1) | `Pool1BoundComposer` | 404 | `53ceff752c294ff39ebdbbd41e434e5c978062091d9a0e939793e77418e0713b` |
| after_fix (HEAD) | `Pool1BoundComposer` | 404 | `53ceff752c294ff39ebdbbd41e434e5c978062091d9a0e939793e77418e0713b` |

**Identical hash.** `research/findings/raw/_onebrain_kmax_loadpath_thread/rf_byteident_before.json` and
`research/findings/raw/_onebrain_kmax_loadpath_thread/rf_byteident_after.json` agree on every field except the
`tag` label.

## Scope / honesty

- Single deployed seed (42, the bundle's own developmental seed) and a single bundle (`scale787/day_33` /
  its onebrain rebuild) — same scope as the rank-1 GO finding this closes; the onebrain composer's own
  correctness generalization is already 6-seed (prior de-risks).
- This closes the HARD/correctness flip-gate only. The rank-1 finding's second, UX-scoped gate — the
  ambiguous-`ask_yes_no` under-recall on same-(agent, action) multi-patient cues (moat-safe, `ob_no_on_stored
  = 0`) — is UNTOUCHED by this change and remains the named next rung if ambiguous yes/no recall is wanted.
- **`composer_kind` production default STAYS `'rf'`.** This is additive load-path plumbing, not a flip; the
  production flip to `'onebrain'` remains the owner's call.
- Not verified: an ACTUAL webapp `_build_chat_brain('onebrain-bundle-path', ...)` end-to-end HTTP round-trip
  (the fix is verified at the `load_developed_brain` layer `_build_chat_brain` calls into, with
  `MultiTurnAgent`'s own thread also unit-covered by the existing `pytest` suite — `tests/test_multi_turn_agent.py`,
  `tests/test_one_brain_composer_agent.py`, `tests/test_communicable_turn_wirein.py`,
  `tests/test_rank2_integrated_loop_thread.py`, `tests/test_developed_brain_io_codes_roundtrip.py`,
  `tests/test_cross_session_persistence.py` all pass unmodified against the new code, 35 passed / 19 skipped).
