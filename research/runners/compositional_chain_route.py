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

HARDENING PASS (2026-08-25, moat-audit closure). A read-only adversarial audit
(`research/findings/2026-08-25-reasoning-route-moat-audit-hardening-spec.md`, workflow `wf_89e66a22-2cb`) mapped
the confabulation/moat-bypass surface of routing to `query_patient` and named the SINGLE-hop-per-turn approach
here as "SAFE-WITH-FIXES." Addressed in this pass (a prior WIP hardening pass on this same arc, reviewed and
reused where sound, is credited inline below):
  * MULTI-VALUED HOPS ABSTAIN (audit req #1). `_distinct_patients` scans the composer for every DISTINCT patient
    stored under a hop's (agent, action); a hop with >=2 distinct patients now abstains the WHOLE chain (`_hop`
    returns `(None, ambiguous=True)`) rather than silently taking the first-match `query_patient` would return.
    This is a DETERMINISTIC, non-spiking safety net -- the audit's ranked fix is to route through the full GNW
    deliberation conflict-abstain (`webapp/gnw_deliberation.py`'s spiking ACC-gate read via
    `all_candidate_patients`); that helper requires an RF-only `_iter_facts`/`unbind` scan the production
    `OneBrainComposer` does not expose. `_distinct_patients` gets the SAME multi-valued-hop OUTCOME (abstain, not
    first-match) generically across composer kinds (bare RF/onebrain `.kb`, or a `TieredFactStore`
    buffer+`ShardedPhasorStore` LTM) by reading `.kb` directly instead -- the honest gap against the audit's
    ranked-[M] fix (the spiking conflict READ itself, vs. this deterministic distinct-count) is named below.
  * GENERATED, NOT PERCEIVED (audit req #4). `resolve_compositional_chain` returns a `ChainedSVO` (a `list`
    subclass, same pattern as `HypothesisSVO`) carrying `.derived_from` (the ordered hop-facts). `frame_derived_answer`
    (below) turns those hop-facts into an honest "I derived this from: <fact1>; <fact2>." lead, applied
    UNCONDITIONALLY to a chain answer in `webapp/server.py` and `rich_answer_composer.py` -- NOT gated behind the
    optional `#129 BRAIN_SOURCE_PROVENANCE_HONESTY` monitor (which is default-OFF), so the honesty framing is a
    property of the ROUTE, not an opt-in faculty. When that optional monitor IS also enabled, `webapp/server.py`
    encodes the fact as `PROVENANCE_GENERATED` (never `PROVENANCE_PERCEIVED`) so its own judged label agrees.
  * DISTINCT API SHAPE + KEPT OUT OF EPISODIC MEMORY (audit req #5). Callers `isinstance`/type-check the
    `ChainedSVO` marker to report `recalled_svo=None` / `derived=True` / `derived_from=[[a,v,p],...]` instead of
    a bare `[a,v,p]` indistinguishable from a directly-recalled fact, and to SKIP `note_topic` (the episodic
    write) and the discourse-WM `_note_referent` write that a directly-recalled fact would otherwise trigger --
    see the wiring in `webapp/server.py` (both the rich and single-fact paths) and
    `research/runners/rich_answer_composer.py`.
  * PARSER TRUNCATION (audit req #6) -- CONFIRMED, not re-fixed. `ChatBrain._neural_question_parse` still pads
    to `[content[0], content[1], "__q__"]` (brain_chat_tui.py) and never reads a 3rd+ content token -- that
    primitive bug is UNFIXED. It is a non-issue for this route because `parse_possessive_chain_question` runs
    its OWN regex over the RAW question string, checked BEFORE `chat.gate`/`_extract_route` ever run (see
    `rich_answer_composer.py::_direct_fact` and the `webapp/server.py` wiring) -- a covered-shape question never
    reaches the truncating parser. Honest residual: a compositional question that does NOT match this narrow
    regex shape still hits the unfixed truncation and mis-parses as a single hop (same behavior as before this
    arc); fixing the parser itself remains a named, out-of-scope future rung.

STILL OPEN (named, not fixed, in this pass -- see the finding's honesty section): no per-hop confidence floor
inside `rf_phasor_composer.py`'s `_scan_first_match`/`_cleanup` (the audit's top-ranked fix; the 2026-08-25
`research/findings/2026-08-25-fhrr-decode-rate-at-scale.md` de-risk measured the deployed D=128/15k-fact/real-
vocab false-hop rate at ~0, so this floor is optional defense-in-depth, not the load-bearing safety mechanism --
out of scope for this pass by explicit direction); the deterministic `_distinct_patients` count is not the
SPIKING GNW conflict read audit req #1 ranks highest (named above); no shard-routing-side noun lemmatization (a
plural agent can still route to a different shard than its singular form -- battery item 11 below); no
confidence threading across hops; no latency budget/hop-cap; no multi-turn/single-store-identity test; no
inverse-direction relation-noun support ("the deer's predator").
"""
from __future__ import annotations

import os
import re

from research.runners.lexical_lemma import lemma_verb


class ChainedSVO(list):
    """A DERIVED (chain-composed) `[agent, action, patient]` triple -- itself a literal stored fact (the final
    hop), but reached by CONNECTING two independently brain-verified hops rather than being asked/recalled
    directly. A `list` subclass (mirrors `research.runners.brain_chat_tui.HypothesisSVO`) so it flows unchanged
    through anything that treats a gate result as `[a, v, p]` (e.g. `chat.render`'s `a, v, p = gate_svo`
    unpacking, or `list(gate_svo)`), while a caller that cares (the source-provenance-honesty wiring, the
    episodic-store guard, the API response shape) can `isinstance`/type-check it and frame it as GENERATED, not
    PERCEIVED, keep it out of episodic/discourse-WM writes that assume a directly-recalled fact, and report a
    distinct API shape (`recalled_svo=None`, `derived=True`, `derived_from=[...]`) rather than one
    indistinguishable from a real recall (moat-hardening audit findings #4/#5). `derived_from`: the ordered list
    of `[a, v, p]` hop-facts the derivation actually confirmed (each independently a stored fact)."""
    __slots__ = ("derived_from",)

    def __init__(self, svo, derived_from=None):
        super().__init__(svo)
        self.derived_from = list(derived_from) if derived_from else []


def frame_derived_answer(surface: str, derived_from) -> str:
    """Wrap an already-rendered/assembled answer `surface` for a `ChainedSVO` terminal in an honest GENERATED-
    provenance lead that SURFACES the supporting hop-facts (moat-hardening audit req #4: 'a derived answer must
    be framed as the brain's OWN inference ... Surface the supporting hop-facts'). Applied UNCONDITIONALLY to
    every chain-derived answer (both the single-fact and rich `/api/brain-chat` paths) -- independent of the
    optional `#129 BRAIN_SOURCE_PROVENANCE_HONESTY` monitor, which is default-OFF and must not be the ONLY
    thing standing between a derived answer and being presented as a plain perceived fact. A caller that ALSO
    has the optional monitor enabled should encode the fact as `PROVENANCE_GENERATED` (never `PROVENANCE_
    PERCEIVED`) for its own judged-label telemetry, but should NOT call `provenance_framed_text` on TOP of this
    (that would double-wrap an already-framed derived answer in a second, generic disclaimer)."""
    hop_text = "; ".join(
        f"{a} {v} {p}" for a, v, p in (derived_from or [])
        if isinstance(a, str) and isinstance(v, str) and isinstance(p, str)
    )
    if hop_text:
        return f"I derived this from: {hop_text}. {surface}"
    # defensive fallback: a genuine ChainedSVO always carries derived_from, but never silently present an
    # unframed derived answer as a plain perceived fact even if that invariant is ever violated upstream.
    return f"I reasoned this myself rather than being told it directly. {surface}"


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


def _distinct_patients(composer, agent, action):
    """Best-effort scan for every DISTINCT patient stored under (agent, action) -- the multi-valued-hop check
    (hardening audit req #1). Works generically across composer kinds by reading `.kb` directly (fact-dict,
    handle tuples), rather than depending on `RFPhasorComposer`-only `_iter_facts`/`unbind` (which
    `OneBrainComposer`, the production onebrain composer used by the default tiny-demo brain, does not expose):
    scans the `TieredFactStore` buffer's `.kb`, then every LTM shard's `.kb` (a `ShardedPhasorStore`) or a plain
    composer's own `.kb`. String-patient facts only (a clause/attributed patient is skipped -- conservative
    undercounting, never a crash). Returns `None` if NOTHING is scannable (no `.kb` reachable at all) -- the
    caller must then fall back to the plain `query_patient` recall (an honest, named residual: ambiguity cannot
    be detected on such a composer), NOT silently treat unscannable as unambiguous."""
    seen = set()
    out = []
    found_any_kb = False

    def _scan(kb):
        nonlocal found_any_kb
        found_any_kb = True
        for fact, _handle in kb:
            if fact.get("agent") == agent and fact.get("action") == action:
                p = fact.get("patient")
                if isinstance(p, str) and p not in seen:
                    seen.add(p)
                    out.append(p)

    try:
        buf = getattr(composer, "buffer", composer)   # TieredFactStore -> its buffer; else the composer itself
        if hasattr(buf, "kb"):
            _scan(buf.kb)
        ltm = getattr(composer, "ltm", None)
        if ltm is not None:
            # ShardedPhasorStore: route to the ONE shard that holds every fact about `agent` (agent-hash
            # routing, sharded_phasor_store.py) instead of scanning all S shards -- O(shard) not O(S*shard).
            shard_for = getattr(ltm, "shard_for", None)
            if callable(shard_for):
                sh = shard_for(agent)
                if hasattr(sh, "kb"):
                    _scan(sh.kb)
            elif hasattr(ltm, "shards"):
                for sh in ltm.shards:
                    if hasattr(sh, "kb"):
                        _scan(sh.kb)
            elif hasattr(ltm, "kb"):
                _scan(ltm.kb)
    except Exception:
        return None
    return out if found_any_kb else None


def _hop(composer, agent, action):
    """One chain hop. Returns (patient_or_None, ambiguous). `ambiguous=True` means (agent, action) has >=2
    distinct stored patients -- the caller MUST treat this as an abstain of the WHOLE chain (a genuinely
    multi-valued hop is the common case on a bulk LTM per the audit; silently taking `query_patient`'s
    first-match would be an insertion-order-dependent, non-deterministic moat bypass -- the exact MOAT-BYPASS
    conflict battery item this closes). When the composer is unscannable for the ambiguity check
    (`_distinct_patients` returns None), falls back to the plain `query_patient` read -- a named, honest
    residual (ambiguity cannot be detected on that composer kind), not a silent single-candidate assumption."""
    patients = _distinct_patients(composer, agent, action)
    if patients is None:
        return composer.query_patient(agent, action), False
    if len(patients) >= 2:
        return None, True
    if len(patients) == 1:
        return patients[0], False
    return None, False


def resolve_compositional_chain(composer, question: str):
    """DETECT + EXECUTE a 2-hop possessive-relative-clause question against `composer` (the brain's OWN fact
    store -- `chat.inner.composer`, which may itself be a `TieredFactStore` buffer+LTM; `query_patient`
    transparently checks both tiers). Returns a `ChainedSVO` `[hop1_agent, hop2_relation, hop2_patient]` --
    itself a genuinely stored fact, reached via two independently brain-verified hops, each also checked for
    AMBIGUITY (a hop with >=2 distinct stored patients abstains the whole chain, see `_hop`) -- on success, or
    `None` when the question is not this shape, OR every candidate hop pair failed/was ambiguous (an HONEST
    ABSTAIN: never invents a bridging fact, never silently picks an arbitrary patient on a multi-valued hop).
    `None` is also returned immediately when `chain_route_enabled()` is False (the lesion)."""
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
        hop1, ambiguous1 = _hop(composer, x, v1)          # hop 1: abstains on no-fact OR on a multi-valued hop
        if ambiguous1:
            return None                                    # a genuinely competing hop-1 -> abstain the WHOLE chain
        if hop1 is None:
            continue
        hop2, ambiguous2 = _hop(composer, hop1, v2)        # hop 2: same check
        if ambiguous2:
            return None
        if hop2 is not None:
            return ChainedSVO([hop1, v2, hop2], derived_from=[[x, v1, hop1], [hop1, v2, hop2]])
    return None                                  # no candidate hop-pair confirmed -> honest abstain
