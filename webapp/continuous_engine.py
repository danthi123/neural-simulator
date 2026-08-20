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


def _wander_budget_per_turn() -> int:
    """How many heavy self-init WANDERs a session may run per idle period (refilled on each real turn).

    PRODUCTION-SAFETY (2026-08-20): the wander is a ~55s CA3 run on cupy. Without a bound, an idle session
    would fire one every IDLE_SEC forever -> an abandoned server pegs the GPU indefinitely, and N idle sessions
    serialize into N*55s tick batches. Bounding it to a small budget per idle period keeps the ALIVE-between-turns
    behaviour a returning user sees (the wander surfaces on the next turn) while the mind SETTLES when a
    conversation is truly abandoned. The cheap mood relaxation still runs every tick regardless. Biologically:
    mind-wandering during rest is ongoing but not maximal-intensity forever; an abandoned rest state settles."""
    try:
        return max(0, int(os.environ.get("BRAIN_WANDER_BUDGET", "1")))
    except Exception:
        return 1


# per-session inner-life: cache_key -> list of {t, valence, arousal, differential, note}
_INNER_LIFE: dict = {}
# per-session last-request wall-clock (set by the handler); "idle" = now - last >= IDLE_SEC
_LAST_REQUEST: dict = {}
# per-session remaining heavy-wander budget for this idle period (refilled to _wander_budget_per_turn() each turn)
_WANDER_BUDGET: dict = {}


def continuous_enabled() -> bool:
    """Default-OFF anchor. `BRAIN_CONTINUOUS` in {1,true,on,yes} arms the background tick."""
    return os.environ.get("BRAIN_CONTINUOUS", "0").strip().lower() in ("1", "true", "on", "yes")


def mark_request(cache_key) -> None:
    """The handler calls this each turn so the tick knows a session is active (and not to tick mid-conversation).
    Also REFILLS this session's heavy-wander budget: a real turn re-opens the mind to wander again during the next
    idle period (so a returning user sees a fresh wandered thought), then the budget drains as it wanders."""
    _LAST_REQUEST[cache_key] = time.time()
    _WANDER_BUDGET[cache_key] = _wander_budget_per_turn()


def forget_session(cache_key) -> None:
    """On a session reset, drop its continuous state (mirrors _SESSION_MOOD/_SESSION_SELFINIT cleanup)."""
    _LAST_REQUEST.pop(cache_key, None)
    _INNER_LIFE.pop(cache_key, None)
    _WANDER_BUDGET.pop(cache_key, None)
    _WANDER_ADAPT.pop(cache_key, None)


def inner_life(cache_key) -> list:
    """The recent between-turn tick records for a session (for the monologue read-out)."""
    return list(_INNER_LIFE.get(cache_key, []))


def recent_wander(cache_key) -> str | None:
    """Rung 2 (board #86, 2026-08-20): make the idle-wandered THOUGHT load-bearing on the NEXT real turn (drive, not
    just observe -- mirrors the #84 affect-lead / #85 swap-lead pattern). Returns the most recent concept an idle
    tick's self-initiation organ wandered to for this session, or None if continuous is off, no idle tick has run
    yet, or no tick surfaced a concept. CONSUMES the record on read (sets it back to None) so the same wandered
    concept is brought up exactly once -- on the next live turn after the tick that produced it -- not repeated on
    every subsequent turn. Pure bookkeeping over the existing inner-life log; no new spiking read (the concept was
    already produced by the selfinit organ's spiking selection at tick time, see tick_session)."""
    if not continuous_enabled():
        return None
    lst = _INNER_LIFE.get(cache_key)
    if not lst:
        return None
    for rec in reversed(lst):
        w = rec.get("wandered")
        if w:
            rec["wandered"] = None  # consume -> surfaces on exactly the next turn, not every turn after
            return w
    return None


IOR_STRENGTH = 0.15      # multiply the just-wandered basin's curiosity gain by this (inhibition-of-return fatigue)
IOR_RECOVERY = 0.5       # fraction of each basin's adaptation deficit recovered per tick (the fatigue wears off)
# per-session wander inhibition-of-return: cache_key -> {"base": [gains], "adapt": [multipliers]}
_WANDER_ADAPT: dict = {}


def _wander_ior_enabled() -> bool:
    """Default-ON anti-fixation for the between-turn wander. Without it the wander is content-DEGENERATE (6/6 'cat',
    finding 2026-08-20-continuous-wander-content-degenerate) — a load-bearing coupling to a constant. IOR (fatigue the
    just-ignited basin so the next wander explores elsewhere; GO in 2026-08-20-inhibition-of-return-breaks-the-
    degenerate-wander) makes trains-of-thought actually move. `BRAIN_WANDER_IOR=0` restores the pre-IOR behaviour."""
    return os.environ.get("BRAIN_WANDER_IOR", "1").strip().lower() in ("1", "true", "on", "yes")


def _pre_wander_ior(cache_key, organ) -> None:
    """Before the wander: bias this session's curiosity gains AWAY from recently-visited basins (apply the adaptation).
    SCAFFOLD NOTE: this modulates the neuromod curiosity DRIVE host-side — the de-risked stand-in for a per-neuron CA3
    spike-frequency-adaptation current (the faithful burn-down, reusing the 2026-08-14 SFA machinery). No-op on the
    first tick (no adaptation captured yet) and whenever gains are unavailable."""
    st = _WANDER_ADAPT.get(cache_key)
    g = getattr(organ, "gains_on", None)
    if st is not None and g:
        organ.gains_on = [float(st["base"][j] * st["adapt"][j]) for j in range(len(st["base"]))]


def _post_wander_ior(cache_key, organ, concept) -> None:
    """After the wander: FATIGUE the basin that just won, then RECOVER all basins toward rest. Captures the pristine
    base gains on the first call (before any pre-adaptation has run)."""
    g = getattr(organ, "gains_on", None)
    agents = getattr(organ, "agents", None)
    if not g or not agents or concept not in agents:
        return
    st = _WANDER_ADAPT.get(cache_key)
    if st is None:  # first post: gains_on is the pristine base (this tick's pre was a no-op)
        st = {"base": list(g), "adapt": [1.0] * len(g)}
        _WANDER_ADAPT[cache_key] = st
    i = list(agents).index(concept)
    st["adapt"][i] *= IOR_STRENGTH                                              # fatigue the just-won basin
    st["adapt"] = [1.0 - (1.0 - a) * (1.0 - IOR_RECOVERY) for a in st["adapt"]]  # all basins recover toward rest


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
            ior = _wander_ior_enabled()
            if ior:
                _pre_wander_ior(cache_key, selfinit_organ)   # bias away from recently-visited basins
            out = selfinit_organ.speak(lesion=False)
            wandered = out.get("concept")
            if ior and wandered:
                _post_wander_ior(cache_key, selfinit_organ, wandered)  # fatigue the just-won basin
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
            # The cheap mood-relax runs every tick; the EXPENSIVE self-init wander only while this session has
            # budget left for this idle period (refilled by mark_request each turn). Once drained, the mind keeps
            # relaxing but stops wandering -> a truly idle server does not peg the GPU with endless ~55s wanders.
            siorg = None
            if selfinit_getter is not None and _WANDER_BUDGET.get(cache_key, 0) > 0:
                try:
                    siorg = selfinit_getter(cache_key)
                except Exception:
                    siorg = None
            rec = tick_session(cache_key, session_mood, organ, now, selfinit_organ=siorg)
            if rec is not None:
                n += 1
                if rec.get("wandered"):  # a heavy wander actually fired -> spend one unit of budget
                    _WANDER_BUDGET[cache_key] = max(0, _WANDER_BUDGET.get(cache_key, 0) - 1)
        except Exception:
            continue
    return n
