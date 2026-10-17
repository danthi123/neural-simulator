"""TRANSITIVE multi-hop reasoning over the shared spiking substrate (A6, midnight-plan 2026-09-24 S15(b)).
See research/findings/2026-09-24-reasoning-transitive-chat-PREREGISTRATION.md for the full design + gate.

WHAT THIS CLOSES. The declared-crutches register's "reasoning" row (item 22) credits multi-hop chat answers only
as "multi-hop recall (host-chained)" because `OneBrainComposer.query_chain` (one_brain_composer.py:1818) is a host
`for` loop over `actions` -- the LOOP decides the hop count, not the brain. This module answers a genuinely
TRANSITIVE yes/no question about a NON-ADJACENT pair ("does A precede D?", never directly taught) WITHOUT that
host loop: it reuses the ALREADY-PRODUCTION-WIRED, ALREADY-6/6-seed-GO'd keystone re-entrant chase
(`webapp.gnw_multistep_deliberation.multistep_chase` -> `research.runners.
_gnw_reentrant_metacog_gated_deliberation_derisk.confidence_gated_chase`), whose hop count is an EMERGENT read of
the GNW workspace's own spiking ignition (`n_ignited`), never a host counter. No new spiking mechanism is added
here -- the contribution is a new DISPATCH (a transitive yes/no question shape) onto that already-verified chase,
plus the yes/no semantics read off its trace.

HONESTY (do not relabel as biology). DETECT (the regex below) is host comprehension of raw text -- the SAME
declared scaffold boundary `research.runners.compositional_chain_route`'s `_POSSESSIVE_CHAIN_RE` and
`webapp.gnw_multistep_deliberation`'s `_CHASE_MARKERS` already occupy. What runs on the substrate is every
relational hop (`query_patient`, inside the reused chase) and the STOP decision (the workspace's own ignition
read) -- never the reachability verdict itself, which this module reads off the chase's already-substrate-decided
trace.

MOAT. Only a POSITIVE (reachable) answer is ever emitted, as a `ChainedSVO` (reused verbatim from
`compositional_chain_route`, so it gets that module's already-hardened honesty framing for free: `derived`/
`derived_from`, the "I derived this from: ..." lead, GENERATED-not-PERCEIVED provenance, exclusion from episodic/
discourse-WM writes). A chase that terminates without reaching the target, or that itself abstains, returns an
HONEST ABSTAIN ("I don't know about that.") -- never an asserted "no" (named v1 scope limitation, see the
prereg). A question matching the transitive shape NEVER falls through to the ordinary `chat.gate(msg)` on the raw
text: the generic parser truncates the 3rd content token (the same documented bug `compositional_chain_route.py`
names for its own possessive-clause shape) and would silently answer a DIFFERENT question.

LESION. `BRAIN_TRANSITIVE_LESION` reuses, unmodified, the keystone's own already-verified lesion
(`multistep_chase(..., lesion=True)` -> the GNW workspace built with its assembly self-recurrence ZEROED,
`webapp/gnw_deliberation.py::_get_bridge`, cached per `(seed, lesion)` so the intact and lesioned workspaces never
collide) -- the SAME lesion the shipped `gnw-multistep-deliberation` faculty already uses. No new lesion
mechanism.

DEFAULT-OFF / BYTE-IDENTICAL. `BRAIN_TRANSITIVE_CHAT` unset -> `resolve_transitive_query` returns `None` on its
very first line -> the caller's pre-existing branch (`compositional_chain_route` then `chat.gate`) is untouched.
NO `sim/` edit.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Tuple

from research.runners.compositional_chain_route import ChainedSVO
from research.runners.lexical_lemma import lemma_verb

# "does <agent> <relation> <target>" -- a narrow, high-confidence shape (precision over recall), the SAME kind of
# declared host regex scaffold `compositional_chain_route._POSSESSIVE_CHAIN_RE` already is. Case-insensitive;
# trailing "?" optional. Entity/relation tokens allow letters, digits and underscore (e.g. gate-runner entities
# "e0".."e4"), matching the alias-token convention used elsewhere in this codebase (e.g. "u_s_of_a").
_TRANSITIVE_RE = re.compile(
    r"^\s*does\s+([a-z][a-z0-9_]*)\s+([a-z][a-z0-9_]*)\s+([a-z][a-z0-9_]*)\s*\??\s*$"
)


# PRODUCTION-DEFAULT ANCHOR (integration wave 2, 2026-09-30; packet research/packets/integration-wave2/
# transitive-chat/, prereg research/findings/2026-09-30-iw2-transitive-chat-flip-PREREGISTRATION.md). While False
# the reader below is EXACTLY the pre-anchor expression (byte-identical for every env value; pinned by
# tests/test_integration_wave2_anchors.py). tools/integration_wave2_flip.py sets it True only after the packet's
# reply-level gate passes: then unset -> ON and only an explicit 0/false/no/off/"" turns it off. The one production
# reader of BRAIN_TRANSITIVE_CHAT; docs/PRODUCTION_INTEGRATION_LEDGER.yaml pins this literal once flipped.
_TRANSITIVE_CHAT_DEFAULT_ON = True
_ON = ("1", "true", "on", "yes")
_OFF = ("0", "false", "no", "off", "")


def transitive_chat_enabled() -> bool:
    """The master flag. While `_TRANSITIVE_CHAT_DEFAULT_ON` is False: `BRAIN_TRANSITIVE_CHAT` in {1,true,on,yes}
    enables the route; unset/anything else -> the route never dispatches (byte-identical to today). After the wave-2
    flip: unset -> enabled; an explicit 0/false/no/off/"" disables."""
    v = os.environ.get("BRAIN_TRANSITIVE_CHAT")
    if _TRANSITIVE_CHAT_DEFAULT_ON:
        return not (v is not None and v.strip().lower() in _OFF)
    return (v if v is not None else "").strip().lower() in _ON


def transitive_lesion_on() -> bool:
    """The load-bearing lesion lever (reuses the keystone's own assembly-self-recurrence-zeroing lesion via
    `multistep_chase(..., lesion=True)`). `BRAIN_TRANSITIVE_LESION` truthy -> the chase can no longer sustain
    ignition past hop 0 -> a genuinely-derivable non-adjacent "yes" collapses to an honest abstain."""
    return os.environ.get("BRAIN_TRANSITIVE_LESION", "").strip().lower() in ("1", "true", "on", "yes")


# SPIKING REACHABILITY (session-2 packet `transitive-spiking-reach`, 2026-10-01; research/runners/
# transitive_spiking_reach_organ.py). Default OFF and byte-identical when off: `resolve_transitive_query` runs the host
# chase above, untouched. ON: after the direct-fact short-circuit the "yes"/abstain decision is the spike count of a
# `detect` population fed by a wave that propagates along the stored edges (no host hop loop, no `target in path`).
# Same anchor convention as `_TRANSITIVE_CHAT_DEFAULT_ON` (flip is a separate, gated act).
_TRANSITIVE_SPIKING_REACH_DEFAULT_ON = False
_TRANSITIVE_REACH_DEFAULT_OP = 0   # the operating point used when BRAIN_TRANSITIVE_REACH_OP is unset (T21 sets it)


def transitive_spiking_reach_enabled() -> bool:
    """`BRAIN_TRANSITIVE_SPIKING_REACH` in {1,true,on,yes} -> the spiking reachability route (while the anchor is
    False). After a flip: unset -> enabled; an explicit 0/false/no/off/"" disables."""
    v = os.environ.get("BRAIN_TRANSITIVE_SPIKING_REACH")
    if _TRANSITIVE_SPIKING_REACH_DEFAULT_ON:
        return not (v is not None and v.strip().lower() in _OFF)
    return (v if v is not None else "").strip().lower() in _ON


def transitive_reach_lesion_on() -> bool:
    """`BRAIN_TRANSITIVE_REACH_LESION` truthy -> the organ's `reach_hop` transmission gate is closed (the wave cannot
    leave the start assembly), so detect is silent and a derivable pair abstains."""
    return os.environ.get("BRAIN_TRANSITIVE_REACH_LESION", "").strip().lower() in _ON


def transitive_reach_op() -> int:
    """`BRAIN_TRANSITIVE_REACH_OP`: the organ's frozen operating-point GRID index (default 0; bad value -> 0)."""
    try:
        return int(os.environ.get("BRAIN_TRANSITIVE_REACH_OP", str(_TRANSITIVE_REACH_DEFAULT_OP)).strip()
                   or _TRANSITIVE_REACH_DEFAULT_OP)
    except Exception:
        return _TRANSITIVE_REACH_DEFAULT_OP


# the last spiking-reach read (diagnostics only: detect count, the read-out path, whether the path held the target)
last_reach_meta: dict = {}


def _spiking_reach_answer(composer, agent: str, relation: str, target: str, seed: int):
    """Run the spiking reachability organ over the composer's stored facts. Returns (True, ChainedSVO | None); never
    raises (any failure is an honest abstain). The path is narration for `derived_from` only; the decision is
    `reached` (detect's spike count)."""
    global last_reach_meta
    last_reach_meta = {}
    try:
        if target == agent:                                # a host scope guard: a trivial self-reach is not a derivation
            return True, None
        from research.runners import transitive_spiking_reach_organ as organ
        organ.set_operating_point(transitive_reach_op())
        r = organ.reach_from_composer(composer, agent, relation, target, seed=int(seed),
                                      # the older BRAIN_TRANSITIVE_LESION cuts this path too, so iw2's row keeps
                                      # its meaning once T21 makes the spiking reach the default
                                      lesion=transitive_reach_lesion_on() or transitive_lesion_on())
        path = [str(x) for x in (r.get("path") or [])]
        last_reach_meta = {"reached": bool(r.get("reached")), "detect_spikes": r.get("detect_spikes"), "path": path,
                           "n_concepts": r.get("n_concepts"), "lesioned": bool(r.get("lesioned")),
                           "path_has_target": bool(target in path[1:])}
        if not r.get("reached"):
            return True, None
        if target in path[1:] and path and path[0] == agent:
            idx = path.index(target)
            derived_from = [[path[i], relation, path[i + 1]] for i in range(idx)]
        else:                                              # detect fired but the read-out path is incomplete: recorded
            # NOTE (2026-10-13): this derived_from names the never-told conclusion itself, not a stored hop. The
            # ChainedSVO keeps conclusion_is_hop=False, so consumers do not treat it as stored (default-off path).
            derived_from = [[agent, relation, target]]
        return True, ChainedSVO([agent, relation, target], derived_from=derived_from)
    except Exception as e:  # recorded (ENGAGEMENT, session 7): an exception must not read as a lesion effect
        last_reach_meta = dict(last_reach_meta, error="%s: %s" % (type(e).__name__, e))
        return True, None


# ENGAGEMENT RECORD (cloud Claude session 7, 2026-10-05; RULING iw2-merged (b)). Both routes catch every exception
# and abstain, so without a record an exception inside a lesioned build reads exactly like the lesion's effect.
# `last_transitive_meta` holds, for the last MATCHED question: the route taken ("direct" | "spiking_reach" |
# "chase"), whether a lesion flag was on, any error, and the substrate's own read (the reach's detect count, or the
# chase's per-hop action / n_ignited). webapp/server.py attaches it to the reply as `transitive_trace` (matched
# turns only; every other turn's body is unchanged). Scorers make a lesion cell without it, or with an error,
# UNDEFINED.
last_transitive_meta: dict = {}


def _chase_trace(meta) -> list:
    out = []
    for entry in (meta or {}).get("trace") or []:
        if isinstance(entry, dict):
            out.append({k: entry.get(k) for k in ("action", "committed", "n_ignited") if k in entry})
    return out


# SCOPE GUARD (integration wave 2, 2026-09-30, before any gate data; research/findings/2026-09-30-iw2-transitive-chat-
# flip-PREREGISTRATION.md). A matched question never falls through to the ordinary path, so default-on the bare regex
# would also capture ordinary yes/no questions whose first or last token is a closed-class word ("does the cat
# sleep?", "does it rain now?") and answer them with this route's abstain. Such a question is not the "does <entity>
# <relation> <entity>?" shape: decline it (None), and the caller's ordinary path runs unchanged. A declared host
# comprehension boundary, the same kind as the regex itself.
_TRANSITIVE_CLOSED_CLASS = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "some", "any", "every", "each", "no", "all", "my", "your",
    "his", "her", "its", "our", "their", "it", "he", "she", "they", "we", "you", "i", "me", "him", "them", "us",
    "what", "who", "whom", "which", "where", "when", "why", "how", "there", "here", "not", "anything", "something",
    "everything", "nothing", "anyone", "someone", "everyone",
})


def parse_transitive_question(question: str) -> Optional[Tuple[str, str, str]]:
    """Detect the "does X R Y?" shape. Returns (agent, relation, target) (all lowercased) or None. None also when the
    agent, relation or target is a closed-class word (the wave-2 scope guard above)."""
    if not isinstance(question, str):
        return None
    m = _TRANSITIVE_RE.match(question.strip().lower())
    if not m:
        return None
    if {m.group(1), m.group(2), m.group(3)} & _TRANSITIVE_CLOSED_CLASS:
        return None
    return m.group(1), m.group(2), m.group(3)


def _chase_path(chat, agent: str, relation: str, *, seed: int, lesion: bool):
    """Reconstruct the sequence of concepts the SUBSTRATE actually broadcast back (each `ADVANCE` cycle's
    `committed` concept, off the keystone chase's own trace -- never a host-assumed path), starting from `agent`.
    Returns (path, meta) where path[0] == agent and path[i+1] is the concept the workspace committed to on the
    i-th resolved hop. On any error / no queryable composer, returns ([agent], {})."""
    try:
        from webapp.gnw_multistep_deliberation import multistep_chase
    except Exception as e:  # recorded (ENGAGEMENT, session 7): an import failure must not read as a lesion effect
        return [agent], {"error": "%s: %s" % (type(e).__name__, e)}
    try:
        _terminal, meta = multistep_chase(chat, agent, relation, seed=seed, lesion=lesion)
    except Exception as e:  # recorded: a raising (e.g. lesioned) build abstains, but the arm file says why
        return [agent], {"error": "%s: %s" % (type(e).__name__, e)}
    trace = (meta or {}).get("trace") or []
    path = [agent]
    for entry in trace:
        if entry.get("action") == "ADVANCE" and entry.get("committed") is not None:
            path.append(entry["committed"])
        elif entry.get("action") == "COMMIT":
            break                                          # the substrate's own ignition collapsed -> terminal reached
    return path, (meta or {})


def resolve_transitive_query(chat, question: str, *, seed: Optional[int] = None
                             ) -> Optional[Tuple[bool, Optional[ChainedSVO]]]:
    """DETECT + EXECUTE a transitive yes/no question against `chat`'s composer.

    Returns `None` when the flag is off OR the question does not match the "does X R Y" shape -- the caller MUST
    fall through to its ordinary gate unchanged (byte-identical path).

    Returns `(True, svo_or_None)` when the question DID match the shape -- the caller must NOT also run the
    ordinary gate on this raw text (see module docstring: the generic parser truncates the 3rd content token and
    would silently answer a different question). `svo_or_None`:
      * a plain `[agent, relation, target]` list (NOT a ChainedSVO) when a direct single-hop fact already answers
        it -- an ordinary recall, not a multi-hop derivation.
      * a `ChainedSVO([agent, relation, target], derived_from=[...])` when `target` was reached only by chaining
        >=2 hops of the substrate's own committed path.
      * `None` -- an honest abstain: the chase terminated without reaching `target`, or the chase itself
        abstained, or `target` was reached with only 0 intervening hops it did not already directly confirm
        (defensive; should not occur given the direct-fact check runs first).
    """
    if not transitive_chat_enabled():
        return None
    parsed = parse_transitive_question(question)
    if parsed is None:
        return None
    agent, relation, target = parsed
    relation = lemma_verb(relation)
    global last_transitive_meta
    last_transitive_meta = {"route": None, "lesioned": bool(transitive_lesion_on()), "error": None}
    composer = getattr(getattr(chat, "inner", None), "composer", None)
    if composer is None or not hasattr(composer, "query_patient"):
        last_transitive_meta["route"] = "no_composer"
        return True, None
    # DIRECT-FACT SHORT-CIRCUIT: an adjacent pair is a plain recall, never mislabeled as "derived".
    try:
        direct = composer.query_patient(agent, relation)
    except Exception:
        direct = None
    if direct == target:
        last_transitive_meta["route"] = "direct"
        return True, [agent, relation, target]
    if transitive_spiking_reach_enabled():                 # default OFF: the host chase below runs untouched
        if seed is None:
            seed = int(os.environ.get("BRAIN_CHAT_SEED", "42"))
        out = _spiking_reach_answer(composer, agent, relation, target, seed)
        last_transitive_meta.update(route="spiking_reach",
                                    lesioned=bool(transitive_reach_lesion_on() or transitive_lesion_on()),
                                    error=last_reach_meta.get("error"),
                                    reach={k: last_reach_meta.get(k) for k in ("reached", "detect_spikes", "lesioned",
                                                                              "n_concepts", "path_has_target")})
        return out
    # SUBSTRATE CHASE (only reached when there is no direct fact): the keystone's own emergent-stop re-entrant
    # chase, UNCHANGED. `seed` threads through exactly like the rest of the tiny-demo brain (BRAIN_CHAT_SEED).
    if seed is None:
        seed = int(os.environ.get("BRAIN_CHAT_SEED", "42"))
    path, _meta = _chase_path(chat, agent, relation, seed=seed, lesion=transitive_lesion_on())
    last_transitive_meta.update(route="chase", error=(_meta or {}).get("error"), chase=_chase_trace(_meta),
                                path_len=len(path))
    if target not in path[1:]:                             # path[0] is the starting agent, not a derived hop
        return True, None                                  # chain resolved (or the chase abstained) -> honest abstain
    idx = path.index(target)
    derived_from = [[path[i], relation, path[i + 1]] for i in range(idx)]
    return True, ChainedSVO([agent, relation, target], derived_from=derived_from)
