"""A5 THEORY OF MIND: false-belief wired for the PRODUCTION conversational turn (2026-09-24).

Turns a live conversation's narration into events on the W3 false-belief register
(`research/runners/tom_false_belief_chat_organ.FalseBeliefChatOrgan`, itself reuse-by-import of the 6/6-seed
GO'd `research/runners/_false_belief_register_derisk.py`). A change-of-location (Sally-Anne) scenario is
narrated across one or more chat turns:

    "Sally puts the marble in the basket."     -> PLACE (always witnessed by the tracked agent)
    "Sally leaves the room."                   -> LEAVE (the tracked agent stops witnessing)
    "Sally returns."                           -> RETURN (the tracked agent witnesses again)
    "Anne moves the marble to the box."        -> MOVE  (witnessed iff the tracked agent is currently present)
    "Where will Sally look for the marble?"    -> QUERY -> a belief-store read-out (short-circuits the reply)

A message may narrate several of these sentences at once (split on `.`/`!`/`?`) or one per turn; state
accumulates in a per-`cache_key` (conversation-scoped) dict exactly like the other Gate-B session organs
(`_SESSION_DISCOURSE`, `_SESSION_WORLDVIEW`, ...).

HONEST SCOPE (declared, not claimed closed — see the PRE-REGISTRATION,
research/findings/2026-09-24-tom-false-belief-chat-wire-PREREGISTRATION.md):
  * WITNESSING/PRESENCE is a HOST comprehension boundary (the regexes below): whether the tracked agent is "in
    the room" when an event happens comes from parsing "leaves"/"returns" sentences, not from a spiking organ.
    No existing production organ computes third-party physical presence from text. Credited as "belief-store
    read-out given host-parsed witnessing", per the plan's own framing.
  * The belief-location -> reply-string mapping is a host template (the sanctioned articulation-crutch
    pattern every other Gate-B organ here uses, e.g. `affective_tom_production_organ.empathic_lead`); the
    NEURAL work is the belief-store read (the organ's `query()`, load-bearing under lesion).
  * ONE concurrent (tracked agent, object) scenario per conversation; a PLACE sentence naming a NEW tracked
    agent starts a fresh scenario (the old organ instance is discarded). Up to `K_LOC=4` distinct locations
    (the organ's own geometry); a fifth aliases onto the last slot.
  * The open-ended chat path is not wired (only the normal `brain_reply` turn path) -- named as an explicit
    next rung, not a held-back shortcut.

CONTRACT (additive, reversible, byte-identical-off), mirroring every sibling Gate-B organ:
  * `false_belief_chat_enabled()` gates the whole block. DISABLED (`BRAIN_FALSE_BELIEF_CHAT` unset or falsy)
    -> the server-side hook never imports this module -> byte-identical, including on a turn whose text would
    otherwise match one of the regexes below.
  * An ORDINARY turn (matches none of PLACE/LEAVE/RETURN/MOVE/QUERY) returns `acted=False` WITHOUT touching the
    organ (no bridge build, no RNG perturbation) even with the flag ON.
  * A narration turn (PLACE/LEAVE/RETURN/MOVE, no query) is a WRITE-ONLY side effect: state updates, the reply
    is UNCHANGED (mirrors the D6/E2/silent-WM "maintain" folds already in `webapp/server.py`).
  * A QUERY turn short-circuits with the belief-store read-out IF a scenario is active; otherwise it falls
    through unchanged (nothing to report) -- never a fabricated answer.

LESION (`BRAIN_FALSE_BELIEF_LESION=1`): forwarded to `FalseBeliefChatOrgan.observe_event`/`.query` as
`lesion=True` on every call for the lifetime of the flag -- the witnessing gate is forced open at write AND
query time (mirrors the derisk's own `lesion_other`), so the belief store collapses onto reality: an
unwitnessed-move query answers with the CURRENT location instead of the stale pre-move one. Load-bearing proof:
the query's `belief_loc` rides the neural write/gate, not this module's host parse (which is identical on
both arms).

Backend: process backend (cupy in production, numpy in tests); NO global-backend flip; NO `sim/` edit.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Optional

_DEFAULT_SEED = 42


def false_belief_chat_enabled() -> bool:
    """`BRAIN_FALSE_BELIEF_CHAT` truthy (1/true/on/yes) enables. Default-OFF (unset -> disabled)."""
    return os.environ.get("BRAIN_FALSE_BELIEF_CHAT", "0").strip().lower() in ("1", "true", "on", "yes")


def false_belief_lesioned() -> bool:
    """`BRAIN_FALSE_BELIEF_LESION` truthy -> force the witnessing gate open at every write and through the
    query (the derisk's `lesion_other`): the belief store collapses onto reality (load-bearing proof)."""
    return os.environ.get("BRAIN_FALSE_BELIEF_LESION", "0").strip().lower() in ("1", "true", "on", "yes")


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# THE HOST COMPREHENSION BOUNDARY (declared residual; see the module docstring). A small closed Sally-Anne
# grammar: <Name> PLACE/MOVEs an object into a location; <Name> LEAVEs/RETURNs; a QUERY asks where <Name> will
# look. The agent-name style mirrors every other Gate-B organ's name-detection convention
# (`affective_tom_production_organ._NAME_SUBJ_RE`), matched case-insensitively (see the note below).
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
#
# `re.IGNORECASE` (not case-sensitivity of the NAME per se): splitting a multi-sentence message capitalizes
# whatever word happens to start each sub-sentence ("... . Where will Sally look ...") regardless of whether
# that word is a leading verb/wh-word or a proper name, so the fixed-case keywords below (leaves/where/...)
# must match case-insensitively; the captured name group still returns exactly the text matched, casing intact.
_PLACE_RE = re.compile(
    r"^([A-Za-z][a-z]+)\s+(?:puts?|places?|put|placed)\s+the\s+(\w+)\s+(?:in|into|on|inside)\s+the\s+(\w+)$",
    re.IGNORECASE)
_MOVE_RE = re.compile(
    r"^([A-Za-z][a-z]+)\s+(?:moves?|moved|hides?|hid|takes?|took)\s+the\s+(\w+)\s+"
    r"(?:to|into|in|inside)\s+the\s+(\w+)$", re.IGNORECASE)
_LEAVE_RE = re.compile(
    r"^([A-Za-z][a-z]+)\s+(?:leaves?|left|goes?\s+out|steps?\s+out|went\s+out|stepped\s+out)"
    r"(?:\s+(?:the|of\s+the)\s+room)?$", re.IGNORECASE)
_RETURN_RE = re.compile(
    r"^([A-Za-z][a-z]+)\s+(?:returns?|returned|comes?\s+back|came\s+back|is\s+back)$", re.IGNORECASE)
_QUERY_RE = re.compile(
    r"^where\s+will\s+([A-Za-z][a-z]+)\s+look\s+for\s+the\s+(\w+)$"
    r"|^where\s+does\s+([A-Za-z][a-z]+)\s+think\s+the\s+(\w+)\s+is$", re.IGNORECASE)


def _clean(sentence: str) -> str:
    return " ".join(sentence.strip().rstrip(".!?").split())


def _split_sentences(message: str):
    for part in re.split(r"(?<=[.!?])\s+|(?<=[.!?])$", (message or "").strip()):
        s = _clean(part)
        if s:
            yield s


def classify(sentence: str) -> Optional[dict]:
    """Classify ONE cleaned sentence into an event dict, or None (not a false-belief sentence -> ignored)."""
    s = _clean(sentence)
    if not s:
        return None
    m = _QUERY_RE.match(s)
    if m:
        agent = m.group(1) or m.group(3)
        obj = (m.group(2) or m.group(4)).lower()
        return {"kind": "query", "agent": agent, "object": obj}
    m = _PLACE_RE.match(s)
    if m:
        return {"kind": "place", "agent": m.group(1), "object": m.group(2).lower(), "location": m.group(3).lower()}
    m = _MOVE_RE.match(s)
    if m:
        return {"kind": "move", "agent": m.group(1), "object": m.group(2).lower(), "location": m.group(3).lower()}
    m = _LEAVE_RE.match(s)
    if m:
        return {"kind": "leave", "agent": m.group(1)}
    m = _RETURN_RE.match(s)
    if m:
        return {"kind": "return", "agent": m.group(1)}
    return None


def has_false_belief_content(message: str) -> bool:
    """Cheap pre-check: does ANY sentence in this message match the grammar? Lets the caller skip building the
    organ entirely on an ordinary turn (byte-identical + no RNG perturbation)."""
    return any(classify(s) is not None for s in _split_sentences(message or ""))


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Per-conversation scenario state. `_SESSION_FALSE_BELIEF[cache_key]` holds the organ instance + the host-side
# bookkeeping (which locations have been named -> ordinal slot, who is currently "present", which agent/object
# is tracked). Mirrors `_SESSION_DISCOURSE` / `_SESSION_MULTIREF` in webapp/server.py.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
class _Scenario:
    __slots__ = ("organ", "tracked_agent", "obj", "locations", "present", "lock")

    def __init__(self, organ):
        self.organ = organ
        self.tracked_agent: Optional[str] = None
        self.obj: Optional[str] = None
        self.locations: dict = {}   # location noun -> slot index (0..K_LOC-1)
        self.present: dict = {}     # agent name -> bool (default True until a LEAVE is narrated)
        self.lock = threading.Lock()

    def loc_index(self, name: str) -> int:
        from research.runners import _false_belief_register_derisk as _FB
        name = name.lower()
        if name not in self.locations:
            idx = len(self.locations) if len(self.locations) < _FB.K_LOC else _FB.K_LOC - 1
            self.locations[name] = idx
        return self.locations[name]

    def loc_name(self, idx: int) -> str:
        for name, i in self.locations.items():
            if i == idx:
                return name
        return "somewhere"

    def is_present(self, agent: str) -> bool:
        return self.present.get(agent, True)


_SESSIONS: dict = {}
_SESSIONS_LOCK = threading.Lock()


def reset_session(cache_key) -> None:
    """Drop this conversation's false-belief scenario (mirrors every other `_SESSION_*.pop` reset)."""
    _SESSIONS.pop(cache_key, None)


def _get_or_start_scenario(cache_key, agent: str, seed: int) -> _Scenario:
    from research.runners.tom_false_belief_chat_organ import FalseBeliefChatOrgan
    with _SESSIONS_LOCK:
        scen = _SESSIONS.get(cache_key)
        if scen is None or scen.tracked_agent != agent:
            # a fresh scenario: either none yet, or a NEW tracked agent supersedes the old one (declared scope
            # limit -- one concurrent (agent, object) scenario per conversation).
            scen = _Scenario(FalseBeliefChatOrgan(seed=seed))
            scen.tracked_agent = agent
            _SESSIONS[cache_key] = scen
        return scen


def _get_active_scenario(cache_key) -> Optional[_Scenario]:
    with _SESSIONS_LOCK:
        return _SESSIONS.get(cache_key)


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# THE PRODUCTION ENTRY POINT.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def observe_turn(cache_key, message: str, *, seed: int = _DEFAULT_SEED, lesion: Optional[bool] = None) -> dict:
    """Fold every recognized sentence in `message` into the false-belief scenario for `cache_key`; if the LAST
    recognized sentence is a QUERY, answer it from the organ's belief-store read-out. Returns
    `{"acted": bool, "answer": str|None, ...}`. `acted=True` ONLY on a successfully-answered query (the caller
    should short-circuit the reply then); a narration-only turn returns `acted=False` with the events it folded
    (a write-only side effect -- the caller's reply is unaffected). NEVER raises out (any error degrades to an
    inert `acted=False` so a turn can never crash on this faculty)."""
    les = false_belief_lesioned() if lesion is None else bool(lesion)
    info = {"acted": False, "answer": None, "events": [], "reason": "no_false_belief_content", "seed": int(seed),
            "lesioned": bool(les)}
    try:
        sentences = list(_split_sentences(message or ""))
        events = [(s, classify(s)) for s in sentences]
        events = [(s, e) for s, e in events if e is not None]
        if not events:
            return info
        info["reason"] = None
        last_query = None
        for _sent, ev in events:
            kind = ev["kind"]
            if kind == "query":
                last_query = ev
                continue   # a query does not itself mutate state; answered after folding every prior sentence
            agent = ev["agent"]
            if kind == "place":
                scen = _get_or_start_scenario(cache_key, agent, seed)
                scen.obj = ev["object"]
                scen.present.setdefault(agent, True)
                loc = scen.loc_index(ev["location"])
                scen.organ.observe_event(loc, witnessed=True, lesion=les)   # a placement is always witnessed
                info["events"].append({"kind": "place", "agent": agent, "location": ev["location"]})
            elif kind == "leave":
                scen = _get_active_scenario(cache_key)
                if scen is not None:
                    scen.present[agent] = False
                    info["events"].append({"kind": "leave", "agent": agent})
            elif kind == "return":
                scen = _get_active_scenario(cache_key)
                if scen is not None:
                    scen.present[agent] = True
                    info["events"].append({"kind": "return", "agent": agent})
            elif kind == "move":
                scen = _get_active_scenario(cache_key)
                if scen is None:
                    continue   # a move with no active scenario -> nothing to track; ignored (byte-identical)
                witnessed = scen.is_present(scen.tracked_agent) if scen.tracked_agent else True
                loc = scen.loc_index(ev["location"])
                scen.organ.observe_event(loc, witnessed=witnessed, lesion=les)
                info["events"].append({"kind": "move", "agent": agent, "location": ev["location"],
                                       "witnessed_by_tracked": bool(witnessed)})
        if last_query is not None:
            scen = _get_active_scenario(cache_key)
            if scen is None or scen.obj is None:
                info["reason"] = "no_active_scenario"
                return info
            read = scen.organ.query(lesion=les)
            belief_name = scen.loc_name(read["belief_loc"])
            info.update({
                "acted": True,
                "answer": f"{scen.tracked_agent} will look in the {belief_name} for the {scen.obj}.",
                "tracked_agent": scen.tracked_agent, "object": scen.obj,
                "belief_location": belief_name, "reality_location": scen.loc_name(read["world_loc"]),
                "belief_loc_idx": read["belief_loc"], "world_loc_idx": read["world_loc"],
                "n_events": read["n_events"], "reason": "queried",
            })
    except Exception as e:   # never let the false-belief read/write crash a turn
        info = {"acted": False, "answer": None, "events": [], "reason": f"error:{type(e).__name__}: {e}",
                "seed": int(seed), "lesioned": bool(les)}
    return info
