"""Compositional (possessive-relative-clause) question -> a 2-hop chain over the brain's OWN fact store
(reasoning-frontier arc, 2026-08-25). See research/findings/2026-08-25-reasoning-frontier-chain-routing.md.

THE GAP THIS CLOSES. The 2026-08-25 integrated-conversational-state diagnostic ran the REAL production
`/api/brain-chat` handler for 19 turns and found the brain NEVER reasons to a new conclusion: every turn
collapses to one stored SVO recall or an honest abstain. After teaching `(wolf, hunts, deer)` and
`(deer, eats, grass)`, "what does the wolf's prey eat?" abstained -- even the fully-spelled-out chain
abstained -- because the comprehension front-end (`ChatBrain._extract_route` /
`_neural_question_parse`) is POSITION-ONLY over exactly 2 content tokens (agent, action); a 3-content-token
possessive question ("wolf's", "prey", "eat") gets its 3rd token silently discarded and is mis-parsed as a
plain (agent="wolf's", action="prey") query, which of course abstains. The INFERENCE machinery to answer this
already exists and is already de-risked (`ShardedPhasorStore.query_chain` / `.chain_of_thought`,
`RFPhasorComposer`/`OneBrainComposer.query_chain`, Tier-2.2 GO 2026-06-27) -- the gap was ENTIRELY in the
front-end never dispatching to it.

THE MECHANISM. (1) DETECT a possessive-relative-clause question shape ("what does X's ROLE V?") with a host
regex -- a DOCUMENTED SCAFFOLD (see "HONESTY" below), analogous to the codebase's existing
`ChatBrain._definitional_copula_route` (a host regex for "what is X?" that hands off to a substrate recall).
(2) RESOLVE the two hop relations: hop2 is the question's OWN verb (lemmatized); hop1 is drawn from a tiny,
extensible ROLE-NOUN hint table (a role noun like "prey" hints at a small set of candidate relations) PLUS
always the question's own verb as a generic fallback (many role nouns, "prey" included, are themselves
definitionally "the object of X's own instance of the asked verb" -- a wolf's prey IS what it eats). (3)
EXECUTE strictly on the brain's OWN composer via TWO genuinely-spiking `query_patient` reads (the SAME op
`query_chain` iterates) -- each hop is its own gate: a missing first hop tries the next candidate; a missing
second hop for a candidate rules that candidate out. NO bridging fact is ever invented; only a hop pair BOTH
independently confirmed by the composer is returned. If NO candidate resolves both hops, the caller (the
`RichAnswerComposer` / the single-fact endpoint) sees `None` and abstains exactly as it already does for any
unmatched question -- the no-confab moat is a strict superset of the direct-recall case (every returned
answer is ITSELF a literal stored fact, reached via two verified hops instead of one).

HONESTY (do not relabel as biology). The SHAPE DETECTION (the regex) and the ROLE-NOUN hint table are HOST
CODE -- a scaffold, exactly like `_definitional_copula_route`'s own regex for "what is X?". The task's own
framing is followed here: any host-side "is this compositional?" routing is a documented scaffold on the
ladder to a learned replacement (a spiking relation-extraction circuit is the named next rung), never
biology. What genuinely runs on the substrate is every DATA READ: both `query_patient` hops are the SAME
spiking recall (`OneBrainComposer._seq_block` on the production onebrain composer) the single-fact path
already uses and already counts as brain-based -- this module adds no new recall primitive, only a new
DISPATCH path to the existing one, run TWICE with the moat re-checked at each hop.

LESION / LOAD-BEARING. `BRAIN_CHAIN_ROUTE=0` (or false/off/no) disables the whole route -- a compositional
question then falls through to the pre-existing comprehension path (which mis-parses it, as documented above)
and abstains, exactly as it did before this arc. This is the load-bearing proof the route DRIVES the derived
answer rather than decorating an answer the old path already produced.
"""
from __future__ import annotations

import os
import re

from research.runners.lexical_lemma import lemma_verb

# "what does the wolf's prey eat?" / "what does wolf's prey eat" (article + apostrophe-s optional-ish, but the
# apostrophe itself is required -- precision over recall: a narrow, high-confidence shape beats a loose one that
# could hijack an ordinary question). Case-insensitive; trailing "?" optional.
_POSSESSIVE_CHAIN_RE = re.compile(
    r"^\s*what\s+does\s+(?:the\s+)?([a-z][a-z_]*)'s\s+([a-z][a-z_]*)\s+([a-z][a-z_]*)\s*\??\s*$"
)

# A tiny, EXTENSIBLE role-noun -> candidate-relation hint table (the declared host scaffold; see module
# docstring). Each entry is tried BEFORE the generic same-verb fallback (which is always appended). A role noun
# not in this table still resolves via the generic fallback alone (e.g. any "X's <unlisted-role> V" question
# where the role noun and V happen to name the same relation, as "prey"/"eat" do in the worked example).
_ROLE_NOUN_HINTS: dict[str, tuple[str, ...]] = {
    "prey": ("hunt", "eat"),
    "food": ("eat",),
    "meal": ("eat",),
    "victim": ("attack", "hunt", "eat"),
    "quarry": ("hunt", "eat"),
    "kill": ("hunt", "eat"),
}

_CHAIN_ROUTE_DEFAULT_ON = True


def chain_route_enabled() -> bool:
    """Default-ON (the route is production-wired, not an opt-in). `BRAIN_CHAIN_ROUTE` in
    {0,false,no,off} is the LESION/escape: it reverts a compositional question to the pre-route
    behavior (mis-parse -> abstain), which is how this route's load-bearing-ness is verified."""
    v = os.environ.get("BRAIN_CHAIN_ROUTE")
    if v is None:
        return _CHAIN_ROUTE_DEFAULT_ON
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def parse_possessive_chain_question(question: str):
    """Detect the "what does X's ROLE V?" shape. Returns (x, role, verb) (all lowercased) or None."""
    m = _POSSESSIVE_CHAIN_RE.match((question or "").strip().lower())
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def resolve_compositional_chain(composer, question: str):
    """DETECT + EXECUTE a 2-hop possessive-relative-clause question against `composer` (the brain's OWN fact
    store -- `chat.inner.composer`, which may itself be a `TieredFactStore` buffer+LTM; `query_patient`
    transparently checks both tiers). Returns a literal `[hop1_agent, hop2_relation, hop2_patient]` triple --
    itself a genuinely stored fact, reached via two independently brain-verified hops -- on success, or `None`
    when the question is not this shape, OR every candidate hop pair failed (an HONEST ABSTAIN: never invents
    a bridging fact). `None` is also returned immediately when `chain_route_enabled()` is False (the lesion)."""
    if not chain_route_enabled():
        return None
    parsed = parse_possessive_chain_question(question)
    if parsed is None:
        return None
    x, role, verb = parsed
    v2 = lemma_verb(verb)
    # candidate hop-1 relations: the role-noun hints first, then the generic same-relation fallback (deduped,
    # order-preserving) -- see the module docstring for why the same-verb fallback is a reasonable default.
    seen = set()
    candidates = []
    for cand in tuple(_ROLE_NOUN_HINTS.get(role, ())) + (v2,):
        cl = lemma_verb(cand)
        if cl not in seen:
            seen.add(cl)
            candidates.append(cl)
    for v1 in candidates:
        hop1 = composer.query_patient(x, v1)     # genuinely-spiking recall, hop 1 (abstains -> None on no fact)
        if hop1 is None:
            continue
        hop2 = composer.query_patient(hop1, v2)  # genuinely-spiking recall, hop 2 (abstains -> None on no fact)
        if hop2 is not None:
            return [hop1, v2, hop2]
    return None                                  # no candidate hop-pair confirmed -> honest abstain
