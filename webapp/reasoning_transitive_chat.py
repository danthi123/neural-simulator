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


def transitive_chat_enabled() -> bool:
    """The master flag, DEFAULT-OFF. `BRAIN_TRANSITIVE_CHAT` in {1,true,on,yes} enables the route; unset/anything
    else -> the route never dispatches (byte-identical to today)."""
    return os.environ.get("BRAIN_TRANSITIVE_CHAT", "").strip().lower() in ("1", "true", "on", "yes")


def transitive_lesion_on() -> bool:
    """The load-bearing lesion lever (reuses the keystone's own assembly-self-recurrence-zeroing lesion via
    `multistep_chase(..., lesion=True)`). `BRAIN_TRANSITIVE_LESION` truthy -> the chase can no longer sustain
    ignition past hop 0 -> a genuinely-derivable non-adjacent "yes" collapses to an honest abstain."""
    return os.environ.get("BRAIN_TRANSITIVE_LESION", "").strip().lower() in ("1", "true", "on", "yes")


def parse_transitive_question(question: str) -> Optional[Tuple[str, str, str]]:
    """Detect the "does X R Y?" shape. Returns (agent, relation, target) (all lowercased) or None."""
    if not isinstance(question, str):
        return None
    m = _TRANSITIVE_RE.match(question.strip().lower())
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def _chase_path(chat, agent: str, relation: str, *, seed: int, lesion: bool):
    """Reconstruct the sequence of concepts the SUBSTRATE actually broadcast back (each `ADVANCE` cycle's
    `committed` concept, off the keystone chase's own trace -- never a host-assumed path), starting from `agent`.
    Returns (path, meta) where path[0] == agent and path[i+1] is the concept the workspace committed to on the
    i-th resolved hop. On any error / no queryable composer, returns ([agent], {})."""
    try:
        from webapp.gnw_multistep_deliberation import multistep_chase
    except Exception:
        return [agent], {}
    try:
        _terminal, meta = multistep_chase(chat, agent, relation, seed=seed, lesion=lesion)
    except Exception:
        return [agent], {}
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
    composer = getattr(getattr(chat, "inner", None), "composer", None)
    if composer is None or not hasattr(composer, "query_patient"):
        return True, None
    # DIRECT-FACT SHORT-CIRCUIT: an adjacent pair is a plain recall, never mislabeled as "derived".
    try:
        direct = composer.query_patient(agent, relation)
    except Exception:
        direct = None
    if direct == target:
        return True, [agent, relation, target]
    # SUBSTRATE CHASE (only reached when there is no direct fact): the keystone's own emergent-stop re-entrant
    # chase, UNCHANGED. `seed` threads through exactly like the rest of the tiny-demo brain (BRAIN_CHAT_SEED).
    if seed is None:
        seed = int(os.environ.get("BRAIN_CHAT_SEED", "42"))
    path, _meta = _chase_path(chat, agent, relation, seed=seed, lesion=transitive_lesion_on())
    if target not in path[1:]:                             # path[0] is the starting agent, not a derived hop
        return True, None                                  # chain resolved (or the chase abstained) -> honest abstain
    idx = path.index(target)
    derived_from = [[path[i], relation, path[i + 1]] for i in range(idx)]
    return True, ChainedSVO([agent, relation, target], derived_from=derived_from)
