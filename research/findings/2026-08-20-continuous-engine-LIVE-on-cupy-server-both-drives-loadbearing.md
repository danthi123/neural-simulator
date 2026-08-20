---
type: finding
status: live
date: 2026-08-20
mechanism: continuous-state-engine
lane: continuous-substrate
seeds: [42]
seed-waiver: This is a DETERMINISTIC integration + load-bearing verification (does the continuous engine RUN on the cupy server, and do the two between-turn couplings CHANGE the reply and VANISH under lesion), not a stochastic effect size across a population. The evidence is functional presence/absence with explicit lesion controls (BRAIN_CONTINUOUS on->concept vs off->None; wander present->lead vs absent->no lead) — a 6-seed distribution measures nothing here; the single fixed seed is the substrate build seed.
instrument: live cupy chat server (webapp/server.py) + in-process load-bearing runner
runner: research/runners/_continuous_drive_loadbearing_cupy.py
artifacts:
  - research/findings/raw/_continuous_live_cupy/live_verify.json
  - research/findings/raw/_continuous_live_cupy/loadbearing_cupy.json
  - research/findings/raw/_continuous_live_cupy/two_turn_shortgap.txt
---
# The continuous-state engine runs LIVE on the cupy production chat server, and BOTH between-turn drives are load-bearing

Artifacts: research/findings/raw/_continuous_live_cupy/live_verify.json · research/findings/raw/_continuous_live_cupy/loadbearing_cupy.json · research/findings/raw/_continuous_live_cupy/two_turn_shortgap.txt

**One line.** The "make the brain CONTINUOUS" arc (2026-08-19 reframe: the LLM-surpassing differentiator is a brain
that keeps feeling/thinking BETWEEN questions) is now verified END-TO-END on the PRODUCTION cupy chat server, not
just in isolation — a real turn returns a fluent reply + honest internal monologue, an idle tick evolves the session
while no one is asking, and the next turn LEADS with the thought that wandered in during the gap. Both between-turn
couplings are load-bearing under lesion (drive-not-observe), and three production-robustness fixes were needed to get
there. Default-OFF still (`BRAIN_CONTINUOUS`); this is the integration-to-production step before flipping the default.

## Live end-to-end verification on the cupy server (artifact live_verify.json)
Server: `SIM_BACKEND=cupy BRAIN_CONTINUOUS=1 BRAIN_CONTINUOUS_DRIVES=1 uvicorn webapp.server:app`. Drivers: two curl
scripts (short-gap two-turn via the OpenAI shim; long-gap two-turn via /api/brain-chat on a fresh session).
1. **A real turn returns fluent + honest (HTTP 200).** Turn 1 "What is a cat?" -> reply "The dog chased the cat. The
   cat eats fish." + monologue "mood reads neutral (valence +0.00, arousal +0.00) [spiking differential +0.000];
   recalled from my store: dog chase cat". 0 errors after the fixes; 0 non-finite breadcrumbs once warm.
2. **The tick evolves idle sessions on the live server.** The server logged `continuous tick: evolved N idle
   session(s)` repeatedly — the always-on background loop runs the spiking affect re-read + CA3 wander on cupy while
   the session sits idle, without freezing the request path.
3. **The idle WANDER drives the next reply.** With a >~75s idle gap (long enough for the ~55s cupy CA3 wander to
   complete + record), turn 2 "Tell me about dogs." returned `(I'd been mulling over cat.) I tell you about dogs.`
   with `wander_drives={"on":true,"concept":"cat"}`. The 25s short-gap run is the NATURAL LESION: the gap is shorter
   than the wander, no concept is recorded, and turn 2 has NO lead — the drive vanishes exactly when the wander does.

## Both between-turn drives are load-bearing under lesion (artifact loadbearing_cupy.json, VERDICT GO)
An in-process runner on real cupy organs, with explicit lesion controls (drive-not-observe: vary the state, the
downstream must change, and the change must vanish when the coupling is cut):
- **WANDER-DRIVE (rung 2.5).** Armed: an idle tick surfaces a concept ("cat"); `recent_wander` returns it then
  CONSUMES it (second read = None, so it surfaces exactly once). Lesion (`BRAIN_CONTINUOUS=0`): `recent_wander`
  returns None — no lead. `PASS_wander_drive_loadbearing = true`.
- **FEELING-DRIVE (v1).** The idle tick relaxes the felt mood 0.80 -> 0.68 (toward neutral) and RE-READS the spiking
  affect ladder; the differential differs at the relaxed mood (0.050) vs the original (0.0597 <!--derived-->). Because
  `_update_session_mood` (server.py:3210) HOLDS the prior mood on a neutral next message and uses it as the EMA prior
  on an affective one, that relaxed mood is the baseline the next turn's tone is built on — so a message sent after
  idling is answered from a measurably cooler felt state than the same message sent immediately.
  `PASS_feeling_drive_loadbearing = true`.

## Three production-robustness fixes this integration required (all committed)
1. **Non-blocking tick** (server.py). The tick's self-init CA3 wander is ~55s on cupy; it had been called
   synchronously inside the asyncio loop -> a 55s freeze of every chat request per idle session. Now run in a thread
   executor with an in-flight guard (at most one heavy tick at a time; no per-20s pile-up).
2. **JSON-safe response** (server.py). Starlette's JSONResponse serializes with allow_nan=False and 500s on a NaN/Inf
   float; a faculty metadata read can go non-finite on the cupy path (notably when a turn races the startup warm).
   Added `_json_safe`: recursively nulls non-finite floats + coerces numpy/cupy 0-d scalars, LOGGING the offending
   key-path (null-with-a-breadcrumb, never a silent 500). Unit-tested vs NaN/Inf/-Inf/numpy-nan/cupy-scalar.
3. **Wander-budget throttle** (continuous_engine.py). Without a bound the tick fires a ~55s wander on every idle
   session every 20s FOREVER — an abandoned server would peg the GPU indefinitely, and N idle sessions serialize into
   N*55s batches. Added a per-session heavy-wander budget (`BRAIN_WANDER_BUDGET`, default 1) refilled on each real
   turn: the wander fires once per idle period (so a returning user still sees the wandered thought) then the mind
   SETTLES to cheap mood-relaxation only. Unit-tested: wander fires on tick 1, skips tick 2 (drained), fires again on
   tick 3 after a turn refills it — exactly 2 heavy wanders across 3 idle ticks.

## Honest scope / residuals
- **Still default-OFF** (`BRAIN_CONTINUOUS`). This is the integration + load-bearing proof; flipping the production
  default-on is the next step (wants a broader multi-turn soak first). A faculty is DONE only at production-default.
- **Speed:** one cupy wander is ~55s (the full 4000-step CA3 operating point) — speed-secondary, but it means the
  wandered-thought lead only appears after ~1 min of idle; the FEELING relaxation is cheap and evolves every tick.
- **Multi-session serialization:** `tick_idle_sessions` wanders idle sessions serially; the budget now bounds each,
  but a many-user server would still want the wander moved to its own worker/queue (a scaling residual, flagged).
- **`STDP IS INERT` log line** during organ builds is a pre-existing informational notice (`_run_one_simulation_step`
  does not advance the clock), unrelated to the affect/wander reads verified here.
