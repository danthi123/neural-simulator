"""Continuous-state engine (2026-08-19 reframe: "make the brain continuous") — v1: the mood keeps evolving BETWEEN turns.

THE POINT (the LLM-surpassing differentiator): a brain is ALIVE between questions — it keeps feeling / thinking /
changing when nothing is asked; an LLM (and a plastic RAG) is only alive while being queried. Today the sim brain
mostly wakes per-turn. This engine adds an always-on background TICK so the substrate runs between requests. It is
LESS blocked than fluent speech: it needs recurrent activity + a spiking read + a homeostatic relaxation — all
on-substrate today, no deep credit.

v1 scope (one seed of the four; the smallest genuinely-continuous property): the FELT MOOD keeps evolving while a
session is idle. Each tick, for an idle session: (a) the appraisal EMA RELAXES toward baseline (the felt state
decays with no new input — a homeostatic process; the appraisal drive is the body/appraisal boundary), then
(b) the spiking affect ladder is RE-READ at the relaxed appraisal — so the mood the brain *feels* now is a genuine
spiking read, evolving between turns. Each tick is recorded to a per-session INNER-LIFE log the next turn's
monologue surfaces ("while you were away, my mood drifted from X toward neutral"). The mood read IS spiking; the
relaxation is a host homeostat (declared). Next rungs (own tasks): self-initiated WANDER on the tick (surface a
concept), idle BTSP CONSOLIDATION, generative attractor-wandering.

Default-OFF behind `BRAIN_CONTINUOUS`; when off the tick loop is inert (byte-identical to today). Host code here is
the CLOCK + the relaxation formula + the log — legitimate world/body-timer infrastructure (it computes no
cognition; every mood read reuses the existing spiking affect ladder). No sim/ edit.
"""
from __future__ import annotations

import os
import time

IDLE_SEC = 20.0            # a session with no request for this long is "idle" -> the tick may run on it
RELAX = 0.85              # per-tick appraisal relaxation toward the neutral setpoint (felt state decays with no input)
NEUTRAL = 0.0            # the appraisal setpoint the mood relaxes toward
_INNER_LIFE_MAX = 24     # keep the last N tick records per session (a short autobiographical trace)

# per-session inner-life: cache_key -> list of {t, valence, arousal, differential, note}
_INNER_LIFE: dict = {}
# per-session last-request wall-clock (set by the handler); "idle" = now - last >= IDLE_SEC
_LAST_REQUEST: dict = {}


def continuous_enabled() -> bool:
    """Default-OFF anchor. `BRAIN_CONTINUOUS` in {1,true,on,yes} arms the background tick."""
    return os.environ.get("BRAIN_CONTINUOUS", "0").strip().lower() in ("1", "true", "on", "yes")


def mark_request(cache_key) -> None:
    """The handler calls this each turn so the tick knows a session is active (and not to tick mid-conversation)."""
    _LAST_REQUEST[cache_key] = time.time()


def forget_session(cache_key) -> None:
    """On a session reset, drop its continuous state (mirrors _SESSION_MOOD/_SESSION_SELFINIT cleanup)."""
    _LAST_REQUEST.pop(cache_key, None)
    _INNER_LIFE.pop(cache_key, None)


def inner_life(cache_key) -> list:
    """The recent between-turn tick records for a session (for the monologue read-out)."""
    return list(_INNER_LIFE.get(cache_key, []))


def tick_session(cache_key, session_mood: dict, affect_organ, now: float | None = None,
                 selfinit_organ=None) -> dict | None:
    """One idle tick for ONE session: (a) FEELING keeps evolving — relax the appraisal + RE-READ the spiking affect
    ladder; (b) a THOUGHT wanders — if a self-initiation organ is given, its curiosity-biased spiking selection
    surfaces a concept (the mind drifting to something while idle). Both are recorded to the inner-life.

    Returns the tick record (or None if the session has no mood yet). Pure w.r.t. the reply path — it only mutates
    this session's mood EMA + the inner-life log; a live turn re-appraises from the message anyway."""
    now = time.time() if now is None else now
    m = session_mood.get(cache_key)
    if m is None:
        return None
    v0, a0 = float(m.get("valence", 0.0)), float(m.get("arousal", 0.0))
    # (a) FEELING: homeostatic relaxation of the felt state toward baseline, then the spiking read at the new point
    v1 = NEUTRAL + (v0 - NEUTRAL) * RELAX
    a1 = a0 * RELAX
    m["valence"], m["arousal"] = v1, a1
    session_mood[cache_key] = m
    try:
        diff = float(affect_organ.read_differential(v1, lesion=False)["differential"])
    except Exception:
        diff = None
    trend = "toward neutral" if abs(v1) < abs(v0) else "steady"
    note = "idle: felt state relaxing %s (was %+.2f, now %+.2f)" % (trend, v0, v1)
    # (b) THOUGHT: a curiosity-biased spiking selection surfaces a wandered concept (a thought drifting while idle)
    wandered = None
    if selfinit_organ is not None:
        try:
            out = selfinit_organ.speak(lesion=False)
            wandered = out.get("concept")
            if wandered:
                note += "; a thought wandered to ‘%s’" % wandered
        except Exception:
            wandered = None
    rec = {"t": now, "valence": v1, "arousal": a1, "differential": diff, "wandered": wandered, "note": note}
    lst = _INNER_LIFE.setdefault(cache_key, [])
    lst.append(rec)
    if len(lst) > _INNER_LIFE_MAX:
        del lst[:-_INNER_LIFE_MAX]
    return rec


def tick_idle_sessions(session_mood: dict, affect_organ_getter, now: float | None = None,
                       selfinit_getter=None) -> int:
    """Run one tick over every session that is IDLE (no request for >= IDLE_SEC). Returns #sessions ticked.

    Called by the server's background loop. Skips sessions mid-conversation (raced writes) and any with no mood yet.
    `selfinit_getter(cache_key)` (optional) supplies that session's self-initiation organ for the thought-wander."""
    if not continuous_enabled():
        return 0
    now = time.time() if now is None else now
    n = 0
    for cache_key in list(session_mood.keys()):
        last = _LAST_REQUEST.get(cache_key)
        if last is not None and (now - last) < IDLE_SEC:
            continue  # still active -> don't tick mid-turn
        try:
            organ = affect_organ_getter()
            siorg = None
            if selfinit_getter is not None:
                try:
                    siorg = selfinit_getter(cache_key)
                except Exception:
                    siorg = None
            if tick_session(cache_key, session_mood, organ, now, selfinit_organ=siorg) is not None:
                n += 1
        except Exception:
            continue
    return n
