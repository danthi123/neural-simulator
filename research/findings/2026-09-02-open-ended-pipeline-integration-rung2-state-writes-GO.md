---
type: finding
status: partial
date: 2026-09-02
mechanism: R1 (open-ended pipeline-integration audit residual) rung 2 -- an open-ended (BRAIN_OPEN_ENDED) turn in `webapp/server.py::brain_reply` now runs the shared pipeline's PER-TURN SESSION-STATE WRITERS before returning (affect-drives #84 felt body-state EMA, affective-ToM, DA-mode #79, common-ground per-referent ledger, and the D3 discourse register's per-turn fold), for their STATE-WRITE side effect only (tone/reference leads discarded). This closes the rest of the NAMED session-state-write gaps the completeness audit found in the open-ended EARLY RETURN (board #199); the deeper query-branch folds (worldview/multiref/prospective-memory/silent-WM) + the specialist-query ROUTING + the FORM-override/default-ON flip remain for rungs 3-5.
lane: architecture (production-integration -- the open-ended branch runs the shared pipeline's session-state writes, not a bypass)
verdict: GO on this rung's own scope (byte-identical OFF, open-ended response-surface byte-identical ON, and the affect-drives / common-ground / discourse session-state writes now genuinely fire on an open-ended turn, each with its original-code control showing NO write) -- see the artifact. BRAIN_OPEN_ENDED is UNCHANGED (still default-OFF); this is NOT the default-ON flip and does NOT close R1. Do not read this as "open-ended integrated."
seed-waiver: this is a DETERMINISTIC integration proof, not a stochastic metric. Each phase is a single deterministic `brain_reply` run (numpy backend, fixed seed 42) compared PATCHED-vs-ORIGINAL by exact-string identity (byte-identical) and by the presence/absence of a session-state write. The verdict is a code-structure / wiring fact (does the additive block run, and is the OFF path unperturbed), which is seed-independent; replicating across 6 seeds would re-run identical deterministic comparisons. No stochastic performance number is claimed.
artifacts:
  - research/findings/raw/2026-09-02-open-ended-pipeline-state-r2-rung2-verify.json
  - research/runners/_open_ended_pipeline_state_r2_verify.py
---

# R1 rung 2: an open-ended turn now runs the shared pipeline's per-turn session-state writers

## Context: the R1 residual (rung 1 recap)

A completeness audit found that `brain_reply`'s open-ended branch (`webapp/server.py`, the
`if os.environ.get("BRAIN_OPEN_ENDED", ...)` guard, default-OFF) **returns early**, before the shared pipeline's
session-state-writing faculties and the rich/single-fact composer. Rung 1 wired the ONE D5 episodic write
(`d5_episodic_production_organ...note_topic`) so a topic discussed in open-ended mode is recallable later. It
left the rest of the skip-list staged. This rung (rung 2) closes the remaining NAMED per-turn session-state
writers a normal turn runs.

Before this rung, exactly ONE session-state store already moved on an open-ended turn: the Gate-B affect block
(`_update_session_mood` → `_SESSION_MOOD`) runs UNCONDITIONALLY earlier in the function (before the open-ended
guard), so mood was never bypassed. Everything else the audit named — the #84 felt body-state, the common-ground
ledger, the discourse register — was frozen by the early return.

## What this rung closes

Inside the existing `if BRAIN_OPEN_ENDED truthy` block, immediately after rung 1's D5 episodic write and before
the pre-existing `return _safe_json_response(_oe_resp, ...)`, the open-ended path now ADDITIONALLY runs the SAME
per-turn faculty state-writers the normal pipeline runs below the block (affect-drives ~4687, affective-ToM
~4716, DA-mode ~4740, common-ground ~4764, D3 discourse fold ~5217):

```python
try:
    if _affect_drives_on():                        # #84 felt body-state EMA -> chat._affect_drives_workspace
        from webapp import affect_drives_chat as _OE_ADC
        _OE_ADC.observe_turn(chat, msg)
except Exception:
    pass
# ... affective-ToM (_affective_tom_on), DA-mode (_da_drives_on), common-ground (_common_ground_drives_on) ...
try:                                               # D3 discourse register per-turn fold (part i ONLY)
    import research.runners.d3_discourse_event_register_production_organ as _OE_DR
    if _OE_DR.discourse_register_enabled() and getattr(getattr(chat, "agent", None), "_event_register", None) is not None:
        _oe_dstate = _SESSION_DISCOURSE.setdefault(cache_key, _OE_DR.new_state())
        _OE_DR.note_turn(msg, chat.agent, _oe_dstate, actions=getattr(chat, "actions_set", None))
except Exception:
    pass
```

Each call reuses the SAME faculty function, the SAME flag-gate, and the SAME `cache_key` the normal pipeline
uses. The faculties' returned tone / reference LEADS are intentionally DISCARDED here (state-write side effect
ONLY): rung 2 moves the STATE writes, not the generation FORM, so `_oe_resp` / the free-talk surface stay
byte-identical. The discourse fold runs part (i) only (the register fold); the who-was-before/now query
short-circuit (part ii) stays on the normal path. Every call is independently `try/except`ed (the standing
"never let a faculty crash a turn" convention), so with every faculty off this is byte-identical, and additive
otherwise. Because all new lines sit inside the already-existing `BRAIN_OPEN_ENDED`-truthy guard, the default
production path (flag unset) imports and executes NONE of it.

## Evidence (artifact `research/findings/raw/2026-09-02-open-ended-pipeline-state-r2-rung2-verify.json`, verdict GO)

Each check is PATCHED-vs-ORIGINAL, comparing `brain_reply` run against the changed `webapp/server.py` vs the
pre-change file (recovered by `git stash push -- webapp/server.py`), on the numpy backend with the warm-Qwen
loader and `open_ended_chat.answer_turn` monkeypatched (no real model). "byte-identical" = an EXACT string
compare of the JSON response, not a code read.

| check | patched | original | verdict |
|---|---|---|---|
| OFF (flag unset): single-fact JSON response | full response through the real curiosity/da-encoding/non-contradiction pipeline | identical string | **byte-identical** |
| ON: open-ended JSON response surface | `_oe_resp` (answer "Dogs chase cats around here.") | identical string | **byte-identical** |
| ON: #84 affect body-state (`chat._affect_drives_workspace`) after 2 open-ended affective turns | workspace exists, body-state body_h=0.8254999999999999 body_a=0.9449999999999998 (moved off the neutral set-point h=0.5 <!--derived: AffectDrivesWorkspace set-point, not an artifact measurement--> / a=0.0 <!--derived-->) | NO workspace (faculty never ran) | write NOW fires |
| ON: common-ground ledger organ for the cache_key | n_turns=2, 1 grounded slot | n_turns=0 (never observed) | write NOW fires |
| ON: discourse `_SESSION_DISCOURSE[cache_key]` after 1 open-ended clause turn ("dog chase cat") | key present, `heard_any=True` | key absent, `heard_any=False` | fold NOW fires |

The two `control` rows (affect body-state moved-off-neutral 1 vs 0; common-ground n_turns 2 vs 0) each clear
their min-separation, so the writes are attributable to THIS change, not to something already running. (The
discourse register's `who_agent()` reads "dog" in BOTH arms — it is the fresh-build default, NOT the
discriminating field; `heard_any` / key-present are.)

## Tests

`SIM_BACKEND=numpy .venv/bin/python -m pytest tests/test_webapp_server.py tests/test_open_ended_generation_fluent.py`
stays green (the suite builds real tiny-demo brains and is slow; it is unaffected by this additive,
guard-scoped change).

## What is NOT closed (rungs 3-5 — disclosed residual)

1. **Deeper query-branch state folds still skipped.** The E2 worldview note (`_SESSION_WORLDVIEW`), D6
   multi-referent WM (`_SESSION_MULTIREF`), prospective-memory (`_SESSION_PMEM`) and activity-silent-WM
   (`_SESSION_SILENT_WM`) per-turn folds live INSIDE their query-answer branches further down the pipeline
   (~4880–5173) and are still bypassed by the open-ended early return. They were not in the audit's named
   set (mood / felt-arousal / episodic / discourse / common-ground) but are the honest next targets.
2. **Specialist-query ROUTING still bypassed.** An open-ended turn that IS a referential-recall / worldview /
   multiref / discourse-query still free-generates rather than routing to the specialist branch that would
   answer it. Moving the open-ended dispatch to REPLACE ONLY the final rich/single-fact surface (so the
   specialist branches keep precedence) is the fuller "run the shared pipeline, override only surface
   generation" restructure — a later rung.
3. **Duplicated call sites, not one shared pipeline.** These are additive CALL SITES mirroring the normal
   pipeline, not yet a single helper both paths invoke; a future rung can unify them (touching the normal
   path, which needs its own byte-identical-off proof).
4. **NOT the default-ON flip.** `BRAIN_OPEN_ENDED` stays default-OFF. R1 is not closed.
