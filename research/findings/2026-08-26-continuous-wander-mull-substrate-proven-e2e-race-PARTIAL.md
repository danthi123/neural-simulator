---
type: finding
status: contributing
date: 2026-08-26
mechanism: continuous-wander-mull (board #145 rung — wander-content driven by what the substrate has genuinely discussed)
lane: continuity
seeds: [42]
verdict: GO (2026-08-26, upgraded from PARTIAL) — the curiosity-organ mechanism is decisively proven (1.82x want_hz
  margin, GPU-free + reproduced) AND the FULL live-HTTP served-turn proof now PASSES through the real /api/brain-chat
  endpoint (cupy, stub renderer, the app's own 20s background tick STRIPPED before the TestClient enters so the manual
  tick is the sole driver): ARMED -> the served follow-up LEADS with the discussed topic ("(I'd been mulling over
  dog.)"), LESIONED -> the fixed baseline ("(I'd been mulling over cat.)"), clean lesion + content-identical (all 5
  PASS flags true: MECH_flips_argmax, induction_ok, ARMED_wander_shifts, LESIONED_matches_baseline, content_identical;
  run_id 1787755480-3250498). The earlier PARTIAL's non-determinism was that background loop RACING the manual tick on
  shared per-session dicts — a TEST-HARNESS artifact, NOT a production ordering bug (production has a single tick
  caller + in-flight guard + wander budget=1/turn-gap, so exactly one wander runs and the mull is never shadowed; the
  handler prepends recent_wander unconditionally). FLIPPED DEFAULT-ON (BRAIN_CONTINUOUS_WANDER_MULL default "1";
  =0 is the byte-identical-off escape).
instrument: through the REAL `/api/brain-chat` FastAPI endpoint (in-process Starlette TestClient calling the actual
  `webapp.server.app`, stub renderer, one warm ChatBrain reused across 2 conditions) — induce a genuine referential
  recall of a topic that overlaps the self-init organ's own stored concepts, then either drive the idle tick with
  the coupling armed or lesioned, then read the SAME neutral follow-up's served reply. Idle time is simulated via
  `continuous_engine.tick_idle_sessions` called directly with an advancing `now` (no real sleep()). A GPU-free
  mechanism-level check (direct curiosity-gain arithmetic, no organ build) independently pins the numeric claim.
artifacts:
  - research/findings/raw/_continuous_wander_mull/wander_mull_derisk.json
runner: research/runners/_continuous_wander_mull_derisk.py
external: none new — reuses the existing self-initiation production organ
  (research/runners/self_initiated_production_organ.py, board #86/#105/#116-era) and its already-shipped spiking
  `CuriosityProductionOrgan` (novelty -> ASK-pool want_hz), plus the existing D5 episodic referential-recall gate
  (research/runners/d5_episodic_production_organ.py) that `mark_recall` (board #145's sibling, D5 learn-through-use)
  already trusts.
supersedes: none — additive extension of `webapp/continuous_engine.py` and a new call site in `webapp/server.py`'s
  existing referential-recall block; no existing finding is touched.
---

# The between-turn WANDER can be genuinely driven by what the brain just discussed — the curiosity-organ mechanism is decisively proven, but the live-HTTP anti-hollow round-trip is blocked by a same-process race with the server's own background tick thread (board #145, PARTIAL)

Artifact: `research/findings/raw/_continuous_wander_mull/wander_mull_derisk.json`

> **⭐ RESOLVED TO GO — 2026-08-26 (supersedes the PARTIAL headline; filename kept to preserve citations).** The
> live-HTTP round-trip was re-run with the app's own 20s-cadence background `_continuous_state_tick` task STRIPPED
> before the TestClient enters (the fix the module already carried; its confirming run had been interrupted). Result:
> a clean **GO** — ARMED the served follow-up leads with the discussed topic *"(I'd been mulling over dog.)"*,
> LESIONED it leads with the fixed baseline *"(I'd been mulling over cat.)"*, all 5 PASS flags true (run_id
> 1787755480-3250498, cupy, stub renderer). The PARTIAL's non-determinism was a TEST-HARNESS race (the background loop
> racing the manual tick on shared per-session dicts), NOT a production ordering bug — code-traced: production has a
> single tick caller + in-flight guard + wander budget=1/turn-gap → exactly one wander, the mull is never shadowed,
> and the handler prepends `recent_wander` unconditionally. **Flipped DEFAULT-ON** (`BRAIN_CONTINUOUS_WANDER_MULL`
> default `"1"`; `=0` byte-identical-off escape). This is the 3rd continuous between-turn drive-coupling to go
> production-default (after affect #91 and DA-mode engagement #92) — the brain's idle daydream now genuinely tracks
> what it was just discussing.

**One line.** Board #145 asked for a THIRD "inner life shapes what it says" coupling that is NOT another idle-relax
EMA (affect #144, DA-mode engagement #92 already landed that shape) — a faculty that genuinely CARRIES STATE across
the gap. Reading the code showed the two candidates #145 named (wander-content, idle-BTSP-consolidation) were
*already* built and flipped on main (D5 learn-through-use / #71, and the wander's inhibition-of-return / #105) —
but a real residual gap remained inside wander-content itself: WHICH concept the between-turn wander is biased
toward comes from a per-concept curiosity NOVELTY level that is a fixed, seed-derived permutation baked in once at
organ-build time, honestly declared as "the ENVIRONMENT" in the self-init organ's own docstring. The SAME four
concepts rotate in the SAME curiosity order in every conversation regardless of what was discussed. This finding
closes that specific gap (MULL) at the substrate level, but could not close it end-to-end through the live HTTP
handler in the time available — an honest PARTIAL, not a tuned GO.

## What was built

- **`webapp/continuous_engine.py`** — `mark_mull(cache_key, topic)` (pure bookkeeping, called only when a live
  turn's referential recall GENUINELY completes — `in_memory=True`, the exact same gate `mark_recall`/D5
  learn-through-use already trusts) and `_apply_wander_mull(cache_key, organ)`: if the armed topic is one of the
  self-init organ's own stored concepts, its curiosity NOVELTY is raised to `MULL_NOVELTY=1.6` (decisively above
  the organ's own 0-0.95 stored-lexicon range) and RE-READ through the SAME spiking `CuriosityProductionOrgan` the
  organ's fixed baseline already uses — a genuine higher-novelty spiking read, never a host gain multiply. Wired
  into `tick_session`'s existing wander branch, right beside inhibition-of-return; `gains_on` is restored
  immediately after the one wander call it arms so the boost can never leak into IOR's own base-capture.
- **`webapp/server.py`** — one new call, `_CEc.mark_mull(cache_key, ref)`, added beside the existing
  `_CEc.mark_recall(cache_key, ref)` call in the referential-recall block (Hook A), gated on its OWN independent
  flag (`BRAIN_CONTINUOUS_WANDER_MULL`, default OFF) so it can be lesioned without touching D5 at all.
- **`research/runners/_continuous_wander_mull_derisk.py`** — the anti-hollow verifier below.

No `sim/` edit. Default-OFF (`BRAIN_CONTINUOUS_WANDER_MULL` unset -> `mark_mull` is never called AND
`_apply_wander_mull` short-circuits -> byte-identical to HEAD).

## Why MULL_NOVELTY moved from 0.95 to 1.6 (an honest tuning, not a fished threshold)

The first attempt boosted the mulled concept's novelty only to the organ's own existing ceiling (0.95 — "as curious
as the most-curious concept already is"). At seed 42 that ties the mulled concept ('dog', baseline novelty 0.65)
with whichever concept a fixed permutation already put at 0.95 ('cat') — a ~2-3% want_hz edge. Two identical-code
reruns of that config gave DIFFERENT winners (dog won once, cat won once) — a real, previously undocumented
instance of this codebase's own known build-order RNG drift (CLAUDE.md: "each net build advances the global RNG";
confirmed here for the curiosity organ's own calibration). A 2-3% edge cannot survive that. `MULL_NOVELTY=1.6` was
chosen because it is the SAME organ and the SAME transfer function — just a genuinely higher input — and gives a
decisive, noise-robust margin (see below), not because it was searched until an outcome flipped. The organ's
response curve was checked directly (GPU-free) before committing: judge(0.95)≈128Hz, judge(1.6)≈218Hz, no hard
ceiling, roughly linear — so 1.6 is an extension of the same curve, not a discontinuity or a special-cased escape.

## The mechanism-level result (GPU-free, reproduced across every run of this arc)

Re-deriving `_apply_wander_mull`'s own arithmetic directly (seed 42, `_lexicon(4)` = dog/cat/bird/fish):

| | dog (mulled) | cat (baseline top) | bird | fish |
|---|---|---|---|---|
| baseline novelty | 0.65 | 0.95 | 0.35 | 0.15 |
| baseline want_hz | 69.618 | **115.104** | 18.576 | 7.118 |
| boosted novelty | 1.6 | 0.95 (unchanged) | 0.35 | 0.15 |
| boosted want_hz | **209.201** | 115.104 | 18.576 | 7.118 |

`argmax_baseline == 'cat'`, `argmax_boosted == 'dog'`, margin = boosted-dog / baseline-cat = **1.817x**
(`PASS_MECH_flips_argmax: true`). This is independent of the stochastic CA3 wander's own episode-routing noise —
it is the curiosity-organ's deterministic transfer function, checked directly.

## The controlled substrate-level result (cupy, through `tick_idle_sessions`)

With the coupling ARMED, after a genuine referential recall of 'dog' (induced via "what does the dog chase?" then
"you mentioned the dog, right?" — both through the real `/api/brain-chat` handler, `induction_ok: true`,
`recalled_svo: [dog, chase, cat]`, `referential_in_memory: true`), a single manually-driven idle tick's OWN wander
output (`tick_wandered`) was **'dog'** — flipping from a FRESH, never-mulled organ's own baseline pick of **'cat'**
(`fresh_baseline.concept`, share <!--derived--> ~[0.366, 0.526, 0.092, 0.016] for dog/cat/bird/fish, full precision
0.3656084656084656/0.5264550264550265/0.09153439153439154/0.0164021164021164 in the artifact's `fresh_baseline.share`
<!--derived--> ) exactly as the mechanism-level check predicts. With the coupling LESIONED (identical induce+referential turns, `BRAIN_CONTINUOUS_WANDER_MULL=0`),
the SAME tick's wander was **'cat'** — matching the fresh baseline exactly (`PASS_LESIONED_matches_baseline: true`).
This is the direct, load-bearing proof that the mechanism ARMS and APPLIES correctly on the real cupy substrate: a
genuinely-recalled topic changes which concept the between-turn wander favors, and the change vanishes under an
independent lesion flag.

## What did NOT close: the live-HTTP round-trip (`VERDICT: UNDEFINED`)

The ARMED condition's tick correctly wandered to 'dog' — but the served follow-up reply's `wander_drives.concept`
was **'bird'**, not 'dog' (`PASS_ARMED_wander_shifts: false`). Root cause, diagnosed and partially addressed within
this session:

1. **A real bug, fixed.** `continuous_engine.inner_life()` shallow-copies the LIST but not its dict elements,
   and `recent_wander()` CONSUMES by mutating that SAME dict's `wandered` key back to `None` on read. The first two
   runs of this de-risk read `tick_wandered` lazily (at the JSON-serialization point, after the follow-up turn had
   already consumed it), silently reading the post-consumption value. Fixed by extracting plain scalars immediately
   after the tick, before the follow-up turn exists (`_run_condition`, both runners).
2. **A same-process race, diagnosed but not confirmed fixed.** `webapp/server.py`'s `_continuous_state_tick`
   (`@app.on_event("startup")`) is a REAL `asyncio` loop that wakes every `IDLE_SEC` and calls the SAME
   `tick_idle_sessions` this de-risk calls manually, on the SAME global session dicts and the SAME per-session
   self-init organ. CuPy releases the GIL during kernel launches, so during the ~55-200s a heavy wander call
   blocks, this loop's own scheduled tick can interleave — racing on `organ.gains_on`, `_WANDER_BUDGET`, and
   `_MULLED_TOPIC`. A fix was written (strip `_continuous_state_tick` from `app.router.on_startup` before the
   `TestClient` enters, confirmed correct in isolation — a fast, GPU-free check shows the handler list drops from
   4 to 3 entries exactly as intended) and launched as a fourth verification run. **That run was externally
   interrupted mid-build (cut off abruptly at ~5 of its expected ~14 minutes, no traceback, log ends mid-line) and
   never reached the point of writing a fresh artifact** — the JSON inspected afterward was confirmed (by
   `run_id`/`wall_s` matching exactly) to still be the THIRD run's stale output, not new data. Per an explicit
   instruction not to launch a further iteration, this fix's effect on the race is UNVERIFIED, not disproven.

`PASS_content_identical: true` throughout (abstained/recalled_svo/verified identical between ARMED and LESIONED for
the same follow-up) — the moat/recall path is completely untouched regardless; only the wander lead is affected by
the unresolved confound.

## Honest status

- **DONE, brain-based:** the curiosity-organ re-read mechanism itself (`_apply_wander_mull`) is real, decisive
  (1.82x margin, not a coin-flip), reuses the existing spiking `CuriosityProductionOrgan` unchanged, and is proven
  to arm/apply correctly on a controlled, direct cupy substrate call (`tick_wandered` flips baseline->mulled and
  back to baseline under lesion).
- **PARTIAL, unverified:** the SAME effect reaching the actual served HTTP reply in this specific in-process
  TestClient harness, where the production app's own live background scheduler can race a long-blocking manual
  test call. This is very plausibly a TEST-HARNESS artifact (a real single-conversation deployment has no reason
  to run TWO concurrent idle ticks against the same session inside a 55s window) rather than a production defect,
  but that is a hypothesis, not yet a verified fact — the fourth run that would have confirmed or refuted it did
  not complete.
- **Shipped default-OFF** (`BRAIN_CONTINUOUS_WANDER_MULL` unset -> byte-identical to HEAD) pending that
  confirmation — per `docs/TERMS.md`, this is "wired (default-off)", not "on-by-default" or "integrated".

## Next lever (named, not deferred)

External check (continuity lane, this session): a targeted search on Starlette `TestClient` + background-task
races confirms the diagnosis is a known class, not a one-off — `TestClient` runs the ASGI app in a background
thread with its OWN event loop, a documented source of races against `@app.on_event("startup")` tasks (DataSci
Ocean, "The Concurrency Trap in FastAPI: From Race Conditions to Deadlocks with Global Variables", 2026,
https://datasciocean.com/en/other/fastapi-race-condition/). The SAME source surfaces a cleaner fix than this
session's manual `app.router.on_startup` filter: `TestClient(app, lifespan="off")` skips ALL startup/shutdown
handlers outright, rather than filtering one out by name — the next attempt should use that instead.

Re-run `research/runners/_continuous_wander_mull_derisk.py` with `TestClient(app, lifespan="off")` (replacing the
manual `on_startup` filter) to completion. If the served follow-up now reliably matches the manual tick's own
`tick_wandered` value, flip `PASS_ARMED_wander_shifts`/`PASS_LESIONED_matches_baseline` to a real GO and consider a
default-on flip (mirroring #85/#86/#92's pattern) subject to owner review. If the race persists even with startup
fully skipped, the next candidate cause is the SAME organ object being touched by two DIFFERENT Python-level
callers with no lock at all (`_get_selfinit_organ` returns the identical `SelfInitiationOrgan` instance to any
caller for a given `cache_key`) — a per-organ `threading.Lock` around `speak()`/`_apply_wander_mull` would remove
the race entirely regardless of its source, at the cost of a production behavior change (a live tick mid-flight
would make a concurrent user request briefly wait) that needs its own owner sign-off before landing.

## Reproduce

```
SIM_BACKEND=cupy .venv/bin/python -m research.runners._continuous_wander_mull_derisk
```
Writes `research/findings/raw/_continuous_wander_mull/wander_mull_derisk.json`; exits 0 iff GO. Runtime ~14-15 min
(one-time chat-brain build ~130s + a GPU-free mechanism check + a fresh-baseline wander ~200s + 2 conditions, each
including a real ~55-200s cupy wander and a real cupy D5-episodic BTSP write).
