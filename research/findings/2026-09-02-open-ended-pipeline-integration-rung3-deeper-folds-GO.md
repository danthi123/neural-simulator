---
type: finding
status: partial
date: 2026-09-02
mechanism: R1 (open-ended pipeline-integration audit residual) rung 3 -- an open-ended (BRAIN_OPEN_ENDED) turn in `webapp/server.py::brain_reply` now ALSO runs the DEEPER query-branch per-turn SESSION-STATE FOLDS a normal turn runs inside its query-answer branches further down the pipeline (E2 worldview `_SESSION_WORLDVIEW`, D6 multiref `_SESSION_MULTIREF`, Gate-B prospective-memory `_SESSION_PMEM`, Mongillo activity-silent-WM `_SESSION_SILENT_WM`), for their STATE-WRITE side effect ONLY (surprise-notice / reminder-prefix / read-out leads discarded, surface byte-identical). This closes the rung-2 disclosed residual (the four deeper folds bypassed by the open-ended early return ~server.py 4880-5253). The specialist-query ROUTING + the FORM-override/default-ON flip remain for rungs 4-5.
lane: architecture (production-integration -- the open-ended branch now runs the shared pipeline's deeper session-state writes, not a bypass)
verdict: GO on this rung's own scope (byte-identical OFF, open-ended response-surface byte-identical ON, and each of the four deeper session-state stores now genuinely moves on an open-ended turn, each with its original-code control showing NO write -- the key stays ABSENT) -- see the artifact. BRAIN_OPEN_ENDED is UNCHANGED (still default-OFF); this is NOT the default-ON flip and does NOT close R1. Do not read this as "open-ended integrated."
seed-waiver: this is a DETERMINISTIC integration proof, not a stochastic metric. Each phase is a deterministic `brain_reply` run (numpy backend, fixed seed 42) compared PATCHED-vs-ORIGINAL by exact-string identity (byte-identical) and by the presence/content of each session-state write. The verdict is a code-structure / wiring fact (does the additive block run, and is the OFF path unperturbed), which is seed-independent; replicating across 6 seeds re-runs identical deterministic comparisons. No stochastic performance number is claimed.
artifacts:
  - research/findings/raw/2026-09-02-open-ended-pipeline-state-r3-rung3-verify.json
  - research/runners/_open_ended_pipeline_state_r3_verify.py
---

# R1 rung 3: an open-ended turn now runs the DEEPER query-branch per-turn session-state folds

## Context: the rung-2 residual (verbatim)

Rung 2 (just merged to main) moved the NAMED per-turn session-state writers a normal turn runs into the
open-ended early return (affect-drives #84, affective-ToM, DA-mode #79, common-ground ledger, the D3 discourse
register fold), for their STATE-WRITE side effect only. It named this rung's residual precisely:

> "Deeper query-branch state folds still skipped: E2 worldview (`_SESSION_WORLDVIEW`), D6 multiref
> (`_SESSION_MULTIREF`), prospective-memory (`_SESSION_PMEM`), activity-silent-WM (`_SESSION_SILENT_WM`) -- these
> folds live inside query-answer branches further down (~server.py 4880-5173) and are still bypassed by the
> open-ended early return."

Rung 3 closes exactly that residual, following rung-2's exact pattern.

## What this rung closes

Inside the existing `if BRAIN_OPEN_ENDED truthy` block, immediately after rung-2's writers and before the
pre-existing `return _safe_json_response(_oe_resp, ...)`, the open-ended path now ADDITIONALLY runs the WRITE side
of the four deeper folds a normal turn runs below the block:

| fold | store | normal-path site | the WRITE lifted (surface discarded) |
|---|---|---|---|
| E2 worldview affective forward-model | `_SESSION_WORLDVIEW[cache_key]` | ~5038-5094 | appraise this turn -> if the affect sign is non-zero, UPDATE `context_sign`/`expected_sign` (persistence prior); the surprise-notice prefix is discarded |
| D6 multi-referent WM | `_SESSION_MULTIREF[cache_key]` | ~5098-5211 | on a non-hold-query turn, `d6org.judge(msg)` MAINTAIN load+hold (>=2 referents into the spiking buffer) |
| Gate-B prospective memory | `_SESSION_PMEM[cache_key]` | ~4907-4964 | on a formation ("remind me to X when Y"), `form_intention` LATCHES the deferred intention; else if one is held, `read_turn` ADVANCES the hold; the ack/reminder surface is discarded |
| Mongillo activity-silent WM | `_SESSION_SILENT_WM[cache_key]` | ~5213-5252 | on a non-recall-query turn, MAINTAIN write-only: a named referent -> `write_referent` (silent focus), else `note_distractor` (grow the silent delay) |

Each call reuses the SAME faculty function, the SAME flag-gate, and the SAME `cache_key` the normal pipeline
uses. Each fold lifts ONLY the NON-query WRITE branch -- the specialist QUERY short-circuits (the worldview
expectation-query, the D6 hold-query read-out, the pmem formation ACKNOWLEDGEMENT surface, the silent-WM
temporal-recall read-out) are ROUTING, and are deferred to rung-4. The faculties' returned surprise-notice /
reminder-prefix / read-out LEADS are intentionally DISCARDED (state-write side effect ONLY): rung-3 moves the
deeper STATE writes, not the generation FORM or the routing, so `_oe_resp` / the free-talk surface stay
byte-identical. Every call is independently `try/except`ed (the standing "never let a faculty crash a turn"
convention), so with every faculty off this is byte-identical, and additive otherwise. All new lines sit inside
the already-existing `BRAIN_OPEN_ENDED`-truthy guard, so the default production path (flag unset) imports and
executes NONE of it.

All four folds were liftable as pure side effects -- none needed the query-answer an open-ended turn does not
produce. The pmem case is the subtle one: a "remind me..." turn on the normal path returns a specialist
acknowledgement, but the ACKNOWLEDGEMENT is a SURFACE/routing concern; the intention LATCH is a pure state write,
so it lifts here (the open-ended turn free-generates its reply while the intention is now genuinely held), and
the acknowledgement routing is left for rung-4. Consistent with the existing design, a monitor FIRE whose prefix
is dropped leaves the intention held (it fires again on the next main-path cue), so discarding it here is safe.

## Evidence (artifact `research/findings/raw/2026-09-02-open-ended-pipeline-state-r3-rung3-verify.json`, verdict GO)

Each check is PATCHED-vs-ORIGINAL, comparing a `brain_reply` run against the changed `webapp/server.py` vs the
pre-change file (recovered by `git stash push -- webapp/server.py`), on the numpy backend with the warm-Qwen
loader and `open_ended_chat.answer_turn` monkeypatched (no real model). "byte-identical" is an EXACT string
compare of the JSON response (asserted in the data), not a code read.

| check | patched | original | verdict |
|---|---|---|---|
| OFF (flag unset): single-fact JSON response | full response through the real single-fact pipeline | identical string | **byte-identical** |
| ON: open-ended JSON response surface | `_oe_resp` (answer "Dogs chase cats around here.") | identical string | **byte-identical** |
| ON: worldview `_SESSION_WORLDVIEW[cache_key]` after an affective open-ended turn | key present, `context_sign=1`, `expected_sign=1` (written) | key ABSENT (branch never ran) | write NOW fires |
| ON: multiref `_SESSION_MULTIREF[cache_key]` after a 2-referent open-ended turn | key present, `n_held=2` | key ABSENT (branch never ran) | write NOW fires |
| ON: prospective-memory `_SESSION_PMEM[cache_key]` after a formation turn | key present, `held=True`, `action_text="run"` | key ABSENT (branch never ran) | latch NOW fires |
| ON: silent-WM `_SESSION_SILENT_WM[cache_key]` after the open-ended turns | key present, `_focus="fish"` | key ABSENT (branch never ran) | write NOW fires |

Four `control` rows (worldview key-present 1 vs 0; multiref n_held 2 vs 0; pmem key-present 1 vs 0; silent-WM
key-present 1 vs 0) each clear their min-separation, so every write is attributable to THIS change (the ORIGINAL
server.py holds everything else fixed and varies ONLY the rung-3 block). `tools.lab.attributable_to` reports
100% of each effect attributable to the manipulation.

## Tests / notes

The two open-ended turns exercise all four folds: turn A ("I'm thrilled about the dog and the cat!") is
affective (worldview UPDATE) with two referents (D6 MAINTAIN) and a named referent (silent-WM focus); turn B
("remind me to run when the fish appears") is an intention formation (pmem LATCH). A benign, pre-existing
`ONEBRAIN XEDGE build FAILED` degradation (a missing `data/corpus/tinystories.txt` in the worktree; xedge is
opt-in / default-off) is caught inside the D6 organ getter and does not affect any result -- the multiref hold
still loads two referents, and it occurs only on the patched path (the original never reaches the D6 branch), so
it cannot perturb the byte-identical compares.

## What is NOT closed (rungs 4-5 -- disclosed residual)

1. **Specialist-query ROUTING still bypassed.** An open-ended turn that IS a worldview expectation-query, a D6
   hold-query, a pmem formation (wanting the acknowledgement), or a silent-WM temporal-recall query still
   free-generates rather than routing to the specialist branch that would ANSWER it. Rung-3 moved the WRITE side
   of each fold; the READ-OUT / short-circuit surfaces those branches own are the rung-4 restructure ("run the
   shared pipeline, override only the final surface generation").
2. **Duplicated call sites, not one shared pipeline.** These are additive CALL SITES mirroring the normal
   pipeline, not yet a single helper both paths invoke; a future rung can unify them (touching the normal path,
   which needs its own byte-identical-off proof).
3. **NOT the default-ON flip.** `BRAIN_OPEN_ENDED` stays default-OFF. R1 is not closed; the open-ended path is
   not `integrated` / `production-default`.
