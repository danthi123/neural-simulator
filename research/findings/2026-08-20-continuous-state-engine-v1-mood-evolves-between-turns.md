---
type: finding
status: live
date: 2026-08-20
mechanism: continuous-state-engine
lane: continuity
seeds: [42]
instrument: unit test of the idle tick through the real spiking affect ladder + a byte-identical-off + no-regression check
artifacts:
  - research/findings/raw/_continuous_engine/evidence.json
---
# Continuous-state engine v1: the brain's felt mood keeps evolving BETWEEN turns (the first "alive between questions" property)

Artifact: research/findings/raw/_continuous_engine/evidence.json

**One line.** From the 2026-08-19 reframe (the LLM-surpassing differentiator is the substrate's CONTINUOUS LIFE, not
the store or the mouth): the sim brain now has an always-on background TICK so its felt state keeps evolving while a
session is idle — "unplug the conversation and it's still changing." v1 is the MOOD seed (1 of the 4 continuous
properties); the tick relaxes the appraisal toward baseline and RE-READS the spiking affect ladder, so the mood the
brain feels drifts between turns and is logged to a per-session inner-life the next turn's monologue surfaces.

## Why this, and why it's less blocked than fluent speech
The four LLM-surpassing properties (learn-through-use, feeling, trains-of-thought, novelty) live in a substrate that
keeps RUNNING, not one that gets queried. They need recurrent activity + local plasticity + neuromodulation — all
on-substrate today, NOT deep supervised credit. A scout (wf_2e3921c7, 4 seeds) found all four converge on ONE
missing primitive: no HTTP-independent clock. This builds that clock as the first rung.

## What it does (v1, the mood seed)
An `asyncio` background loop (`@app.on_event("startup")` in webapp/server.py, mirroring the existing periodic-scan
loop), default-OFF behind `BRAIN_CONTINUOUS`. Each `IDLE_SEC` it runs `continuous_engine.tick_idle_sessions` over
every session with no request for >= IDLE_SEC: the appraisal EMA RELAXES toward the neutral setpoint (the felt state
decays with no new input — a homeostat), then the spiking affect ladder is RE-READ at the relaxed appraisal. The
result is recorded to a per-session inner-life log surfaced first in the OpenAI-shim monologue ("while you were away
(N idle ticks): felt state relaxing toward neutral, was +0.80 now +0.35").

## Verification (see evidence.json)
- **Continual state:** starting a session at valence 0.8 and ticking 5× with NO request, the mood evolves
  0.68 -> 0.58 -> 0.49 -> 0.42 -> 0.35, with the spiking differential re-read each tick; inner-life logs 5 entries.
- **Byte-identical off:** `BRAIN_CONTINUOUS` unset -> `tick_idle_sessions` returns 0 (loop inert).
- **No regression:** the server imports cleanly with the new startup handler + wiring; `tests/test_determinism.py`
  9/9.

## Brain-based scope + honest residual
The mood VALUE is a genuine spiking affect-ladder read; the host code is only the CLOCK, the relaxation formula, and
the log — legitimate world/body-timer infrastructure (it computes no cognition). The relaxation-toward-baseline is a
declared host homeostat (the appraisal drive is the body/appraisal boundary). This is 1 of 4 seeds: the next rungs
(own tasks) are the self-initiated WANDER on the tick (a surfaced concept), idle BTSP CONSOLIDATION (learn/consolidate
while idle), and generative attractor-wandering. Default-off until the wander/consolidation rungs make it richer.

## Rung 2 (2026-08-20) — a THOUGHT now wanders between turns too

Artifact: research/findings/raw/_continuous_engine/wander_evidence.json

The idle tick now also runs the already-GO'd self-initiation organ`speak()` (curiosity-biased spiking selection): a
CONCEPT surfaces while idle ("a thought wandered to ‘cat’"), recorded in the inner-life alongside the mood drift and
surfaced in the monologue. So 2 of the 4 continuous properties now run between turns (FEELING + TRAINS-OF-THOUGHT),
both reusing existing spiking mechanisms, additive + default-off. Verified: over 3 idle ticks the mood evolves
0.60→0.37 AND a concept surfaces each tick. Honest scope: the numpy light path surfaces the curiosity-top concept
(stable); the stochastic multibasin CA3 wander (varied concepts) is the cupy path. Next rungs: idle BTSP
CONSOLIDATION (learn while idle) + generative attractor-wandering.

## Rung 2.5 (2026-08-20, board #86) — the idle-wandered thought now DRIVES the reply (not just observed)

Artifact: research/findings/raw/_continuous_engine/wander_drives_evidence.json

The between-turn wander was OBSERVE-only (shown in the monologue). Now it is LOAD-BEARING (drive-not-observe): on the
next live turn, if an idle tick surfaced a concept, the brain brings it up — a lead `(I'd been mulling over X.)`
prepended to the reply, mirroring the #84 affect-lead / #85 swap-lead pattern. `continuous_engine.recent_wander(key)`
returns AND CONSUMES the most recent wandered concept (surfaces exactly once, on the next turn); `server.py` brain_chat
prepends the lead when `BRAIN_CONTINUOUS_DRIVES` is on. Default-off + inert when the continuous engine is off.
Adversarially verified through the REAL brain_chat: byte-identical-off (no key/lead, reply unchanged), moat-safe
(abstained/recalled_svo/verified unchanged with the lead on), load-bearing (consumed -> no repeat; remove the wander
-> the lead vanishes). So the continuous state is now the brain's FOURTH drive-coupling (after #84/#85/#79) — the
first one carried by the always-on engine rather than the current turn.
