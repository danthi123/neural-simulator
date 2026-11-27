# Capability: Memory strength and forgetting
# Record: brain/research/2026-11-27-fable-review-dg-port.md
# Expected-fail: partly. Replay is drawn in proportion to effective strength, so a replay-driven boost is rich-get-richer
#   in expectation (hubs are replayed most); the homeostat keeps the store's mean but preserves ratios, so it cannot undo
#   that. Lifting the whole store toward the set-point may raise false fires. Strength logic is outside the LM (not
#   scale-sensitive, PREREQUISITES.md "Memory strength and forgetting").
"""Reachable strength: nightly replay strengthens the replayed episode's OWN trace, then a homeostat scales every
non-permanent strength by one factor toward a set-point (Brain(replay_strengthen=..., strength_setpoint=...), both
default OFF).

Biology: research/biology/repetition-retrieval-strengthen-same-trace.md (reactivation strengthens the existing trace
instead of adding a copy; restabilization renews it) and research/biology/homeostatic-scaling-relative-strength.md (one
multiplicative factor over all inputs, proportional to each one's strength, so the relative order is kept; it does not
equalise memories). The per-episode store has no shared neurons, so the homeostat here is ONE factor for the whole
non-permanent store (the store-level analogue of the per-neuron rule): ratios between episodes are preserved exactly.

What a night does with the switches on (after the replay into the slow synapses, which is unchanged):
  1. counts: how many of the night's replay draws went to each episode. This is the brain's own replay schedule: the
     draw sequence of brain/sleep.py (random.Random(seed); each step one rng.choices over the episodes with weights
     max(w, 1e-3), then one rng.randrange over the generic texts), recomputed here without touching sleep.py.
  2. strengthen (replay_strengthen = b): each non-permanent replayed episode gets strength += b * count, and its decay
     clock restarts at the night (last_used = night time: the reactivated trace is restabilised, as a use is).
  3. scale (strength_setpoint = S): g = S / mean(effective strength of the non-permanent episodes at the night), clamped
     to [SCALE_MIN, SCALE_MAX]; every non-permanent strength *= g. Permanent episodes are untouched (their effective
     strength is pinned at >= 3 by Hippocampus.effective_strength).
Either switch may be on alone (scaling only: S without b). No RNG of the caller is used (an own random.Random).
"""
from __future__ import annotations

import random

REPLAY_BOOST = 0.125    # registered per-draw boost: USE_BOOST / 4 (a draw is a weaker reactivation than a cued fetch)
SET_POINT = 1.0         # registered set-point: the birth strength of a told-once episode (salience 1.0; = routing SC_CAP)
SCALE_MIN, SCALE_MAX = 0.25, 4.0   # the homeostat's range per night (declared host constants, not fit)


def replay_counts(weights, n_generic: int, steps: int, seed: int) -> list:
    """Per-episode draw counts of brain/sleep.py's replay loop for these weights (same RNG, same call order)."""
    if n_generic < 1:
        raise ValueError("replay_counts: sleep draws a generic text every step; n_generic must be >= 1")
    rng = random.Random(seed)
    w = [max(s, 1e-3) for s in weights]
    idx = range(len(w))
    counts = [0] * len(w)
    for _ in range(int(steps)):
        counts[rng.choices(idx, weights=w)[0]] += 1
        rng.randrange(n_generic)
    return counts


def strengthen(episodes, counts, boost: float, now: float, reset_clock: bool = True) -> list:
    """strength += boost * count on each non-permanent episode replayed at least once; its decay clock restarts at now
    (reset_clock). Returns the per-episode strength added."""
    added = []
    for e, c in zip(episodes, counts):
        if e.permanent or c <= 0:
            added.append(0.0)
            continue
        e.strength += boost * c
        if reset_clock:
            e.last_used = now
        added.append(boost * c)
    return added


def homeostatic_scale(episodes, eff_fn, setpoint: float, now: float, lo: float = SCALE_MIN, hi: float = SCALE_MAX) -> dict:
    """One factor g over every non-permanent strength so their mean effective strength moves to setpoint (clamped)."""
    live = [e for e in episodes if not e.permanent]
    if not live:
        return {"g": 1.0, "mean_eff_before": None, "n": 0}
    m = sum(eff_fn(e, now) for e in live) / len(live)
    g = 1.0 if m <= 0 else min(max(setpoint / m, lo), hi)
    for e in live:
        e.strength *= g
    return {"g": g, "mean_eff_before": m, "n": len(live)}


def night_update(episodes, weights, n_generic: int, steps: int, seed: int, now: float, eff_fn, boost=None,
                 setpoint=None, reset_clock: bool = True) -> dict:
    """The whole night rule. episodes: the night's replayed episodes (weights[:len(episodes)] are theirs; weights may
    hold more entries, e.g. sleep_extra, which take draws but are not episodes). Returns the night's record."""
    rec = {"boost": boost, "setpoint": setpoint, "reset_clock": bool(reset_clock)}
    if boost is not None:
        counts = replay_counts(weights, n_generic, steps, seed)[:len(episodes)]
        rec["counts"] = counts
        rec["added"] = strengthen(episodes, counts, float(boost), now, reset_clock)
    if setpoint is not None:
        rec.update(homeostatic_scale(episodes, eff_fn, float(setpoint), now))
    return rec
