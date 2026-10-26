"""C3 per-topic ask stop (L) and topic-locked phasic DA (P) on the production curiosity follow-up. Default OFF.

Flags (each default OFF; unset or anything but 1/true/on/yes is off):
  BRAIN_CURIOSITY_TOPIC_STOP         L: the follow-up decision is read off the conversation's topic-stop organ
                                     (research/runners/curiosity_topic_stop_organ.py, method 2's per-topic
                                     depressing slices onto an ASK pool) instead of the shared-pool ASK read.
  BRAIN_CURIOSITY_TOPIC_STOP_LESION  lesion_L: the organ's slice synapses recover between turns (tau_d 200 ms, method
                                     2's lesion_habituation): the per-topic memory is removed, the drive is kept.
  BRAIN_CURIOSITY_PHASIC_DA          P: the host crave-gain formula (webapp/da_curiosity_drives_chat.py) is NOT
                                     applied; instead the organ's SNc gets a tonic afferent from the session DA level
                                     and a phasic afferent from the topic's own salience, and the resulting bus
                                     dopamine acts on the ASK pool's excitability. Requires L (the organ hosts it); with
                                     L off this flag does nothing (the handler never reaches this module).
  BRAIN_CURIOSITY_PHASIC_DA_LESION   lesion_P: the phasic afferent pinned to 0 (the tonic afferent intact).

With every flag off nothing here runs: `topic_stop_enabled()` is False, the handler skips this module entirely, no
Context attribute is created and no trace key is attached, so the turn is byte-identical to main.

STATE AND ISOLATION. The organ and the turn clock live on THIS conversation's `Context` (webapp/one_brain.py: one per
conversation key, dropped by `ContextRegistry.remove` on reset/eviction), as the attribute CTX_ATTR. The production
shared-pool ASK read restores a clean state per read; the topic-stop organ deliberately does NOT restore its slice
synapses between reads (that STP state IS the per-topic trace), so it must not be process-shared: one organ per
conversation, built lazily on the first abstain (about 0.6 s, 450 neurons). A reset is a new conversation with fresh
synapses. This mirrors the per-conversation organs `_SESSION_MULTIREF` / `_SESSION_PMEM` / `_SESSION_SELFINIT`.
"""
from __future__ import annotations

import os

CTX_ATTR = "curiosity_topic_stop"
_TRUE = ("1", "true", "on", "yes")


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in _TRUE


def topic_stop_enabled() -> bool:
    return _flag("BRAIN_CURIOSITY_TOPIC_STOP")


def topic_stop_lesioned() -> bool:
    return _flag("BRAIN_CURIOSITY_TOPIC_STOP_LESION")


def phasic_da_enabled() -> bool:
    return _flag("BRAIN_CURIOSITY_PHASIC_DA")


def phasic_da_lesioned() -> bool:
    return _flag("BRAIN_CURIOSITY_PHASIC_DA_LESION")


class ConvState:
    """One conversation's topic-stop state: the turn clock and the lazily built organ."""

    def __init__(self):
        self.turns_seen = 0
        self.organ = None


def state_of(ctx) -> ConvState:
    st = getattr(ctx, CTX_ATTR, None)
    if st is None:
        st = ConvState()
        setattr(ctx, CTX_ATTR, st)
    return st


def note_turn(ctx) -> None:
    """Count one conversation turn (called at the start of every `brain_reply` while L is on)."""
    state_of(ctx).turns_seen += 1


def _organ(st: ConvState, seed: int, curiosity_lesioned: bool):
    if st.organ is None:
        from research.runners.curiosity_topic_stop_organ import TopicStopOrgan
        st.organ = TopicStopOrgan(seed, lesion_stop=topic_stop_lesioned(), drive_lesion=bool(curiosity_lesioned),
                                  phasic=phasic_da_enabled(), phasic_lesion=phasic_da_lesioned())
        st.organ.ensure_built()
    return st.organ


def salience_from_judge(j: dict) -> float:
    """The topic's normalized salience on the shared ASK-pool read the production organ already made this turn:
    `CuriosityProductionOrgan.salience_of`'s own formula, applied to `judge()`'s want (the same read)."""
    cal = j.get("calib") or {}
    span = float(cal.get("want_novel_hz", 0.0)) - float(cal.get("want_familiar_hz", 0.0))
    if abs(span) <= 1e-9:
        return 0.0
    return (float(j.get("want_hz", 0.0)) - float(cal.get("want_familiar_hz", 0.0))) / span


def decide(ctx, chat, topic, novelty: float, j: dict, seed: int, curiosity_lesioned: bool) -> dict:
    """The topic-stop decision on an abstain. Returns the trace; `trace["applied"]` False means the caller keeps the
    production decision (no extractable topic). Raises on an organ failure (the caller records it and falls back)."""
    st = state_of(ctx)
    p_on = phasic_da_enabled()
    trace = {"on": True, "applied": False, "lesioned": topic_stop_lesioned(),
             "phasic_da_flag": p_on, "phasic_da_lesion_flag": phasic_da_lesioned(),
             "turns_seen": int(st.turns_seen),
             "want_hz_shared": float(j.get("want_hz", 0.0)), "threshold_shared": float(j.get("threshold", 0.0))}
    if not topic:
        trace["reason"] = "no topic extracted: the production decision stands"
        return trace
    org = _organ(st, seed, curiosity_lesioned)
    da_level = sal = None
    if p_on:
        from webapp import da_curiosity_drives_chat as _DAC
        da_level = _DAC.da_level_of(chat)
        sal = salience_from_judge(j)
    rec = org.read(topic, novelty, turns_seen=st.turns_seen, da_level=da_level, salience=sal)
    trace.update(rec)
    trace["applied"] = True
    trace["phasic_da_applied"] = bool(p_on)
    trace["calib"] = dict(org.calib or {})
    return trace
