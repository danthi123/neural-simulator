"""brain_chat_tui — an easy TUI to LOAD a developed/trained brain and hold a MULTI-TURN conversation with it.

The owner uses this to TALK to a developed brain (e.g. the self-knowledge brain). It LOADS the EXACT developed
brain (its grounded concept codes + its stored facts + its vocab) and runs a multi-turn chat loop:

    prompt -> parse the question (self-aliases resolved, anaphora resolved from the discourse buffer)
           -> RECALL from the brain (who/what/yes-no/describe/reason via the agent; the no-confab GATE)
           -> RENDER fluently (default: the OFF-BRIDGE Qwen grounded-language faculty, gate->constrain->verify,
              loaded ONCE + kept warm; --stub-renderer uses the template-stub, GPU-FREE, for the CPU smoke)
           -> print the answer, OR "I don't know about that." on abstention (the MOAT).

The render is GATED + VERIFIED: the brain supplies + verifies the CONTENT (the moat holds EVEN WITH a real
generative LLM in the loop); the faculty's only job is fluent surface form.

GENERATE channel (#3E) SURFACE -- brain-native SPIKING mouth (production default): when the brain VOLUNTEERS a
novel grounded HYPOTHESIS (an open-ended "what might a dog do" turn -> a moat-verified `HypothesisSVO`), its
surface is rendered grammatically ON FIRING NEURONS by the composed spiking BROCA ("perhaps the <S> <V-3sg> the
<O>": word order = the per-pool spiking-RATE ranking on a real Izhikevich SimulationBridge, EMERGE-59/61 x the
#3E draw, `_spiking_fluent_surface_derisk`, 6-seed GO), TRANSFORMER-FREE -- replacing the agrammatic host
f-string 'perhaps bear walk foot'. It is re-parse VERIFIED (the same moat the recall path uses) so it recovers
the drawn SVO; a verify miss falls back to the raw flagged template (NEVER a leak); the guess stays clearly
FLAGGED either way. Escape: `BRAIN_SPIKING_MOUTH=0` reverts to the pre-spiking mouth (Qwen / stub / template).
Open ARBITRARY prose the spiking Broca can't frame still falls back to the Qwen mouth -- the banked A1 residual.

LOAD SOURCES (auto-detected from --load):
  * a `developed_brain_io` BUNDLE directory (brain.json + grounded_codes.npz + facts.json + lineage/) -- the
    self-contained "developed brain" the develop loop / a save_developed_brain call writes. THE GENERIC PATH.
  * the SELF-KNOWLEDGE brain: a `_self_knowledge_grounded_codes.json` codes blob (+ the curriculum it was
    developed on) -- the brain reconstructs on the learned codes and re-teaches the curriculum facts. Pass the
    codes .json (or just `--self-knowledge` to use the default codes path).
  * NOTHING / a tiny fallback (the GPU-FREE smoke): build a tiny CPU brain from a handful of facts.

COMMANDS in the chat loop:
  /raw      toggle the brain's OWN neural renderer (no LLM) -- the unvarnished brain (raw recalled triple).
  /facts    list what the brain knows (its stored facts).
  /help     show the commands.
  /quit     exit (also: /exit, /q, Ctrl-D).

SELF-REFERENCE: 'you'/'your'/'I'/'me'/'it' map to the agent 'brain' so 'what are you?' / 'how do you learn?'
resolve against the brain's self-facts.

REUSE-BY-IMPORT, NO `sim/` edit. The OFF-BRIDGE Qwen faculty is the runtime fluent renderer (used when the owner
runs it for real with a free GPU); the GPU-FREE smoke validates the BRAIN side on CPU with the template-stub.

Usage:
    # talk to a saved developed brain (real, with the off-bridge Qwen renderer, free GPU):
    SIM_BACKEND=cupy python -m research.runners.brain_chat_tui --load <developed-brain-dir-or-codes.json>

    # the self-knowledge brain (after `_self_knowledge_demo` saved its codes):
    SIM_BACKEND=cupy python -m research.runners.brain_chat_tui --self-knowledge

    # GPU-FREE smoke (template-stub renderer, scripted stdin):
    SIM_BACKEND=numpy python -m research.runners.brain_chat_tui --stub-renderer --tiny-demo
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.developed_brain_io import (  # noqa: E402
    load_developed_brain, is_developed_brain_bundle,
)
# VERB LEMMATIZATION (reasoning-frontier, 2026-08-25): canonicalize an inflected action to one stable store/query
# key (hunts/hunt/hunted -> "hunt") -- see research/runners/lexical_lemma.py + the 2026-08-25 reasoning-frontier
# finding. Used in `ChatBrain._maybe_acquire` (store-write) and `ChatBrain._substrate_recall` (query fallback).
from research.runners.lexical_lemma import lemma_verb  # noqa: E402

# default self-knowledge artifacts (so `--self-knowledge` works with no path)
_SK_CODES = os.path.join(_REPO, "research", "findings", "raw", "_self_knowledge_grounded_codes.json")
_SK_CURRICULUM = os.path.join(_REPO, "research", "findings", "raw", "_curriculum_self_knowledge.json")


# ============================================================================================================
# KNOWLEDGE GROUNDING (reasoning-frontier arc, 2026-08-26 -- board #65/#66, frontier A). See
# research/findings/2026-08-26-knowledge-grounding-natural-language.md +
# research/runners/_knowledge_core_curate.py's `build_alias_facts`.
#
# THE GAP. The shipped `wikidata_core_15k` cortical LTM stores facts under Wikidata-derived CANONICAL tokens
# (e.g. agent='atlantic_jazz', action='instance_of') -- clean, but NOT the words a natural question uses
# ('atlantic jazz', 'is a'). `_extract_route` splits a question into single stopword-stripped words and hands
# (agent, action) as two of THOSE single words: a multi-word entity phrase never becomes the one underscore-
# joined token the store keys on, and `_definitional_copula_route` hardcodes the copula to the in-conversation-
# teaching convention 'isa', which the Wikidata core does not use (its instance-of relation is 'instance_of').
# Once the right tokens reach `query_patient`, recall is exact and safe (2026-08-25-fhrr-decode-rate-at-scale.md,
# 6-seed, false-hop=0.0 at the deployed D=128/15k scale) -- this closes a pure GROUNDING/COMPREHENSION gap, not
# a recall or capacity problem.
#
# THE MECHANISM. A host-side phrase-SEGMENTATION scaffold (deciding how many stopword-stripped words form one
# candidate span, longest-first) tries each candidate against a GENUINE spiking hop: `composer.query_patient
# (candidate, "alias_of")` -- the SAME `query_patient` primitive `compositional_chain_route.py`'s 2-hop
# reasoning already counts as brain-based, reading a NEW relation type ('alias_of') the curation script bakes
# into the SAME store ordinary facts live in. An unresolved surface form returns None -> the caller passes it
# through UNCHANGED (moat-safe: this can only ADD a successful grounding, never invent one) -- see
# `_ground_content_words`.
#
# HONESTY (do not relabel as biology). The PHRASE SEGMENTATION (how many words form one candidate span before
# any recall runs) is HOST CODE -- a scaffold, exactly like `_extract_route`'s pre-existing stopword-strip and
# `_definitional_copula_route`'s own regex. What genuinely runs on the substrate is the RESOLUTION READ itself:
# 'is this surface form known under this store's canonical name' is a real stored fact recalled via the SAME
# spiking op the rest of the system already counts as brain-based, with the SAME no-confab moat (an unresolved
# alias abstains exactly like an unknown agent) -- not a host dict consulted at answer time. The alias FACTS
# themselves are host-curated from Wikidata's own crowd-sourced alias lists at BUILD time (the identical
# honesty class as the already-shipped 15k curation itself). NOT built (the named next rung, out of scope
# here): a LEARNED entity-linking/synonymy mechanism that acquires alias<->canonical associations from
# co-occurrence in running text, so grounding emerges from exposure/use rather than a teacher-curated alias
# file -- this closes reachability of the ALREADY-SHIPPED core in natural language now; that rung is what would
# let the brain ground a genuinely novel entity's surface forms it was never given an alias list for.
#
# LESION / LOAD-BEARING. `BRAIN_KNOWLEDGE_GROUNDING` in {0,false,no,off} disables the whole pass -- a natural-
# language knowledge question then reverts to the pre-grounding behavior (a literal underscore-token miss ->
# abstain), exactly as it did before this arc. This is the load-bearing proof the pass DRIVES the answer.
# ============================================================================================================


def _knowledge_grounding_enabled() -> bool:
    """Default-ON (production-wired, not opt-in). `BRAIN_KNOWLEDGE_GROUNDING` in {0,false,no,off} is the
    LESION/escape."""
    v = os.environ.get("BRAIN_KNOWLEDGE_GROUNDING")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


# ============================================================================================================
# RELATION-FRONTED QUESTIONS (Vikunja #142, 2026-08-27 -- knowledge-in-live-chat wrongful-veto fix). See
# `ChatBrain._relation_fronted_route`'s docstring for the mechanism; summary: 'what country is chelsea fc
# from?' fronts the RELATION noun ('country') before the copula, unlike the in-conversation teaching shape
# 'what does <entity> <verb>?' the generic (agent,action)=(content[0],content[1]) positional parse in
# `_extract_route` already handles -- so that generic parse mis-assigns the relation word to the AGENT slot
# and the entity to the ACTION slot, producing a query the substrate correctly has no fact for (an honest
# abstain on a fact the store genuinely holds -- see the 2026-08-27 finding for the traced repro). Default-ON
# (production-wired, matching every other knowledge-grounding-arc flag); `BRAIN_RELATION_FRONTED_QUESTIONS` in
# {0,false,no,off} is the LESION/escape (byte-identical to pre-fix: the route never matches, `_extract_route`
# falls through to the unchanged generic parse for every question).
# ============================================================================================================

def _relation_fronted_enabled() -> bool:
    """Default-ON (production-wired, not opt-in). `BRAIN_RELATION_FRONTED_QUESTIONS` in {0,false,no,off} is the
    LESION/escape."""
    v = os.environ.get("BRAIN_RELATION_FRONTED_QUESTIONS")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


# 'what <relation> is/are/was/were <entity> [trailing prep]?' -- the relation is required to be a SINGLE bare
# word (the store's own relation names are single underscored tokens; a multi-word relation phrase is left to
# the generic parse, out of scope here). Deliberately excludes 'does/do/did' (those already route correctly
# through the existing 'what does <entity> <verb>?' shape) so this can never fire on an already-working question.
_REL_FRONTED_RE = re.compile(
    r"^what\s+(?P<relation>[a-z]+)\s+(?:is|are|was|were)\s+(?P<entity>.+?)\s*\??\s*$", re.IGNORECASE)

# A trailing preposition left dangling by the copula shape ('... is chelsea fc FROM?', '... is the tower IN?')
# is part of the English question frame, not the entity -- stripped before the entity is grounded/joined.
_REL_FRONTED_TRAILING_PREPS = ("from", "in", "of", "at", "on", "for", "with", "to")


# ============================================================================================================
# KB RELATION QUESTION TEMPLATES (2026-09-01, board #94 frontier -- the NL-parser-vs-real-KB residual named in
# 2026-09-01-confidence-forthcomingness-ltm-elaboration-load-bearing-GO.md). See
# `ChatBrain._kb_relation_question_route`'s docstring for the mechanism; summary here.
#
# THE GAP. The shipped `wikidata_core_15k` core's TOP relations (curation_report.json `relations_used`) include
# many whose canonical token is a Wikidata property label COLLAPSED TO ONE UNDERSCORED TOKEN at curate time --
# 'country_of_citizenship', 'member_of_political_party', 'place_of_birth', 'headquarters_location', ... --
# because the store keys on ONE atomic vocab word per concept (module docstring of
# `_knowledge_core_curate.py`). `_relation_fronted_route`'s regex deliberately requires the fronted relation to
# be a SINGLE bare word ("a multi-word relation phrase is left to the generic parse, out of scope here" -- its
# own docstring), and the generic `_extract_route` positional parse has no notion of a relation NOUN PHRASE at
# all. So a natural question about any of these relations -- "what is X's country of citizenship?", "where was
# X born?", "what political party is X a member of?" -- never reaches a routable (agent, action) pair and the
# turn honestly (but wrongly) abstains, even though `composer.query_patient`/`what_does` answers instantly once
# the right (entity_token, relation_token) pair reaches it (the SAME already-verified recall primitive every
# other knowledge-grounding route uses; this closes ROUTING/COMPREHENSION only, not a recall or capacity gap).
#
# THE MECHANISM. A curated table maps each underscored relation to a SHORT list of natural-English question
# shapes (regex, each capturing an `entity` group) -- two GENERIC shapes built from the relation's own
# underscore-replaced phrase ("what is <entity>'s <phrase>?" / "what is the <phrase> of <entity>?", covers
# every relation in the table with zero per-relation authoring) plus, for a subset that a real speaker phrases
# more idiomatically ("where was X born?" rather than "what is X's place of birth?"), one or two EXTRA curated
# shapes layered on top. This is COMPREHENSION-ONLY host code -- exactly the same honesty class as
# `_relation_fronted_route`'s own regex and `_definitional_copula_route`'s hardcoded 'isa' -- a phrase-shape
# scaffold; the entity side reuses `_ground_content_words`'s SAME genuine spiking alias-hop (or the naive
# underscore-join fallback) already established for `_relation_fronted_route`, and the actual RECALL is
# untouched (`what_does`/`query_patient` on the substrate). The no-confab moat is therefore intact: an unknown
# entity, or a relation the store genuinely has no fact for, still returns nothing -> honest abstain. This can
# only ADD a resolvable route, never invent an answer.
#
# LESION / LOAD-BEARING. `BRAIN_KB_RELATION_QUESTIONS` in {0,false,no,off} disables the whole pass -- byte-
# identical to before this arc (the route never matches, `_extract_route` falls straight through to the
# unchanged `_relation_fronted_route` / definitional-copula / generic parse for every question). Default-ON,
# matching every other knowledge-grounding-arc flag (production-wired, not opt-in).
# ============================================================================================================

def _kb_relation_questions_enabled() -> bool:
    """Default-ON (production-wired, not opt-in). `BRAIN_KB_RELATION_QUESTIONS` in {0,false,no,off} is the
    LESION/escape."""
    v = os.environ.get("BRAIN_KB_RELATION_QUESTIONS")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


# The shipped `wikidata_core_15k` core's TOP relations (curation_report.json `relations_used`) whose canonical
# token carries an underscore -- exactly the set `_relation_fronted_route` cannot reach (its own docstring: "a
# multi-word relation phrase is left to the generic parse"). 'instance_of' is deliberately EXCLUDED: it already
# has a dedicated route (`_definitional_copula_route`'s alias-hop to 'is_a'/'is_an' -> 'instance_of'); adding a
# second, differently-shaped route for the same relation risks two routes disagreeing on precedence for no gain.
# A relation NOT in this table because its own canonical token is a single bare word (e.g. 'country', 'sport',
# 'genre', 'follows') is already reachable via `_relation_fronted_route`'s "what <relation> is <entity>?" shape
# or the generic 'what does X V' verb shape -- out of scope here for those. TWO single-word exceptions ARE kept
# in this table: 'employer' and 'occupation' -- their natural phrasing ("who does X work for?", "what is X's
# occupation?") does NOT fit `_relation_fronted_route`'s fixed "what <relation> is/are/was/were <entity>"
# grammar (that route is never reached from these questions, so there is no double-routing risk); the generic
# possessive/of templates below cover 'occupation' with zero extra authoring, and 'employer' gets its own idiom.
_KB_UNDERSCORED_RELATIONS = (
    "located_in_time_zone", "located_in_the_administrative_territoria", "subclass_of",
    "headquarters_location", "shares_border_with", "language_of_work_or_name", "member_of", "part_of",
    "taxon_rank", "country_of_citizenship", "followed_by", "contains_administrative_territorial_enti",
    "country_of_origin", "languages_spoken_written_or_signed", "award_received", "participant_of",
    "given_name", "place_of_birth", "place_of_death", "educated_at", "record_label", "work_location",
    "original_language_of_film_or_tv_show", "member_of_political_party", "position_held", "parent_taxon",
    "family_name", "employer", "occupation",
)

# A per-relation readability override for the GENERIC phrase (the plain underscore->space replacement reads
# badly for these two -- both are curation-time TRUNCATIONS of a longer Wikidata property label, capped at
# `sanitize(maxlen=40)` in `_knowledge_core_curate.py`, so there is no "correct" un-truncated phrase to recover;
# this is a readable stand-in for the SAME literal canonical token, not a claim about the untruncated label).
_KB_RELATION_PHRASE_OVERRIDE = {
    "located_in_the_administrative_territoria": "administrative territorial entity",
    "contains_administrative_territorial_enti": "administrative territorial entities it contains",
}

# Idiomatic EXTRA shapes for relations a real speaker phrases more naturally than the generic possessive/of
# pair below (layered ON TOP of, never replacing, the generic pair -- both are always tried).
_KB_RELATION_IDIOMS = {
    "place_of_birth": [r"^where\s+was\s+(?P<entity>.+?)\s+born\??$"],
    "place_of_death": [r"^where\s+did\s+(?P<entity>.+?)\s+die\??$"],
    "educated_at": [r"^where\s+(?:was|is)\s+(?P<entity>.+?)\s+educated\??$",
                     r"^where\s+did\s+(?P<entity>.+?)\s+study\??$"],
    "work_location": [r"^where\s+does\s+(?P<entity>.+?)\s+work\??$"],
    "headquarters_location": [r"^where\s+(?:is|are)\s+(?P<entity>.+?)\s+headquartered\??$",
                               r"^where\s+are\s+(?P<entity>.+?)'s\s+headquarters\??$"],
    "employer": [r"^who\s+does\s+(?P<entity>.+?)\s+work\s+for\??$"],
    "award_received": [r"^what\s+award\s+did\s+(?P<entity>.+?)\s+receive\??$",
                        r"^what\s+awards?\s+has\s+(?P<entity>.+?)\s+(?:received|won)\??$"],
    "member_of_political_party": [r"^what\s+political\s+party\s+is\s+(?P<entity>.+?)\s+a\s+member\s+of\??$",
                                   r"^what\s+party\s+does\s+(?P<entity>.+?)\s+belong\s+to\??$"],
    "country_of_citizenship": [r"^what\s+country\s+is\s+(?P<entity>.+?)\s+a\s+citizen\s+of\??$",
                                r"^what\s+is\s+(?P<entity>.+?)'s\s+nationality\??$"],
    "followed_by": [r"^what\s+is\s+(?P<entity>.+?)\s+followed\s+by\??$"],
    "subclass_of": [r"^what\s+is\s+(?P<entity>.+?)\s+a\s+subclass\s+of\??$"],
    "part_of": [r"^what\s+is\s+(?P<entity>.+?)\s+part\s+of\??$"],
    "member_of": [r"^what\s+is\s+(?P<entity>.+?)\s+a\s+member\s+of\??$"],
    "participant_of": [r"^what\s+did\s+(?P<entity>.+?)\s+participate\s+in\??$"],
    "shares_border_with": [r"^what\s+does\s+(?P<entity>.+?)\s+share\s+a\s+border\s+with\??$",
                            r"^what\s+borders\s+(?P<entity>.+?)\??$"],
    "record_label": [r"^what\s+record\s+label\s+is\s+(?P<entity>.+?)\s+signed\s+to\??$"],
    "languages_spoken_written_or_signed": [r"^what\s+languages?\s+does\s+(?P<entity>.+?)\s+speak\??$"],
}


def _kb_relation_phrase(relation: str) -> str:
    """The NATURAL noun phrase a question uses for `relation`'s generic templates: the readability override if
    one exists, else the plain underscore->space replacement (reads fine for most of the table, e.g.
    'country_of_citizenship' -> 'country of citizenship', 'given_name' -> 'given name')."""
    return _KB_RELATION_PHRASE_OVERRIDE.get(relation, relation.replace("_", " "))


def _build_kb_relation_patterns():
    """Precompile every (regex, canonical_relation) pair ONCE at import time (not per-question) -- the idioms
    plus the two generic shapes for every relation in `_KB_UNDERSCORED_RELATIONS`."""
    out = []
    for relation in _KB_UNDERSCORED_RELATIONS:
        phrase = _kb_relation_phrase(relation)
        phrase_re = re.escape(phrase).replace(r"\ ", r"\s+")
        pats = list(_KB_RELATION_IDIOMS.get(relation, ()))
        # 'who' as well as 'what' -- a person-typed relation ('employer', 'given_name', ...) reads as naturally
        # with 'who'/'who's' as 'what'/"what's"; a non-person relation asked with 'who' simply finds no fact and
        # honestly abstains (the moat), so broadening the WH-word here costs nothing.
        pats.append(r"^(?:what|who)\s+is\s+(?P<entity>.+?)'s\s+%s\??$" % phrase_re)
        pats.append(r"^(?:what|who)\s+is\s+the\s+%s\s+of\s+(?P<entity>.+?)\??$" % phrase_re)
        for pat in pats:
            out.append((re.compile(pat, re.IGNORECASE), relation))
    return out


_KB_RELATION_PATTERNS = _build_kb_relation_patterns()


def _alias_hop(composer, candidate: str):
    """ONE genuinely-spiking hop: does the brain's OWN store know `candidate` as an ALIAS of some canonical
    concept? Reuses the EXACT `query_patient` primitive `compositional_chain_route.py`'s reasoning hops already
    count as brain-based (no new recall primitive) -- this only reads a new relation, 'alias_of', baked into
    the shipped knowledge core by `_knowledge_core_curate.build_alias_facts`. Returns the canonical token, or
    None (an unresolved surface form -- the moat's abstain, not a guess). Never raises: a composer that lacks
    `query_patient`/an `alias_of` fact simply returns None, so this degrades to a no-op on any composer built
    before this arc (a bundle with no alias facts at all)."""
    if composer is None or not candidate:
        return None
    try:
        return composer.query_patient(candidate, "alias_of")
    except Exception:
        return None


def _ground_content_words(composer, content, *, max_span=6, min_span=2, known_words=frozenset()):
    """Collapse multi-word ENTITY/RELATION phrases in `content` (a token list, already stopword-stripped for
    the `_extract_route` caller) to their canonical single-token form via `_alias_hop`, longest-span-first,
    non-overlapping, left to right. A span that resolves is replaced by its canonical token; anything that does
    NOT resolve (including every span below `min_span`) passes through UNCHANGED -- byte-identical for any
    question no alias covers, or for one already using the raw canonical token.

    `min_span=2` (the `_extract_route` default) deliberately EXCLUDES single-word candidates: ordinary
    conversational SVO teaching ('wolf hunts deer') uses single-word roles, and the shipped alias vocabulary is
    large (~38k tokens) enough that a coincidental collision with an everyday word is a real risk -- grounding
    a ROUTINE single word against a Wikidata alias could silently REWRITE an already-working in-session recall
    into an unrelated concept. Restricting the eager pass to multi-word spans (a much rarer, higher-precision
    surface shape) avoids that failure mode; single-word alias resolution is instead offered ONLY as a LAST-
    RESORT fallback in `_substrate_recall`, tried after the literal + lemma recall have both already failed
    (so it can only rescue an otherwise-abstaining turn, never override a working one). `_definitional_copula_
    route` calls this with `min_span=1` for its OWN subject, which is safe because that route only fires on an
    explicit 'what is X'/'who is X'/'define X' prefix -- a shape ordinary SVO teaching never matches.

    `known_words`: content words already recognized this conversation (agents_set|actions_set|patients_set) are
    never re-ground even when `min_span` would otherwise allow it -- an already-established in-session word's
    meaning is never overridden by a store-wide alias lookup."""
    if composer is None or not content:
        return list(content)
    out = []
    i = 0
    n = len(content)
    while i < n:
        matched = None
        span_len = min(max_span, n - i)
        for span in range(span_len, min_span - 1, -1):
            candidate = "_".join(content[i:i + span])
            if candidate in known_words:
                continue
            canon = _alias_hop(composer, candidate)
            if canon:
                matched = (span, canon)
                break
        if matched:
            span, canon = matched
            out.append(canon)
            i += span
        else:
            out.append(content[i])
            i += 1
    return out


# ============================================================================================================
# OPEN-ENDED GENERATION (production wire-in of the 6-seed-GO #3E "brain owns open-ended generation" faculty).
# On an EXPLICIT open-ended prompt ("what might X ...", "tell me something new about X", "what else about X",
# "guess ..."), the brain VOLUNTEERS a NOVEL grounded proposition via generative replay over its OWN learned
# association graph (the substrate-learned Hebbian co-occurrence on the onebrain path), gated by the validated
# #3E/b2 plausibility + non-contradiction gate, moat-verified (a proposal that contradicts a stored fact or
# passes known-fact retrieval is REJECTED -> abstain), and rendered as a FLAGGED hypothesis ("perhaps a v p").
# `_NOT_OPEN_ENDED` is the sentinel `_parse_open_ended` returns for EVERY non-matching turn, so gate() stays
# byte-identical on the recall / abstain / learn / anaphora paths. `HypothesisSVO` is a list subclass so it
# still behaves as an [a, v, p] triple everywhere a plain gate result flows (JSON, transcript), while render()
# can recognise it and mark it as a guess (never asserted as knowledge).
# ============================================================================================================

_NOT_OPEN_ENDED = object()


class HypothesisSVO(list):
    """A GENERATED, moat-verified HYPOTHESIS triple [a, v, p] (plausible + non-contradictory + NOT a known
    fact). A `list` subclass so it flows unchanged through everything that treats a gate result as [a, v, p]
    (the webapp JSON `recalled_svo`, the smoke transcript), while `render()` recognises it and renders an
    explicit guess rather than an asserted fact."""
    __slots__ = ()


# Each entry: (compiled regex on the lowercased/stripped question, has_topic). Named groups: `topic` (the
# subject to generate about) and, for "what might", an optional `action`. These fixed lead-ins are the WHOLE
# trigger surface — a normal recall ("what does dog chase"), teach ("dog eat bone"), yes/no, or anaphora turn
# matches NONE of them, so it never enters the generation branch (gate() stays byte-identical).
_OPEN_ENDED_PATTERNS = [
    (re.compile(r"^what might (?:a |an |the )?(?P<topic>[a-z]+)(?:\s+(?P<action>[a-z]+))?\b"), True),
    (re.compile(r"^tell me something (?:new |else |more )?about (?:a |an |the )?(?P<topic>[a-z]+)\b"), True),
    (re.compile(r"^what else (?:about|can you (?:tell me|say)(?: something)? about|do you know about) "
                r"(?:a |an |the )?(?P<topic>[a-z]+)\b"), True),
    (re.compile(r"^(?:make something up|imagine something|dream up something) about "
                r"(?:a |an |the )?(?P<topic>[a-z]+)\b"), True),
    (re.compile(r"^guess(?:\s+.*?\babout (?:a |an |the )?(?P<topic>[a-z]+)\b)?"), True),
]

# MASTER ON/OFF for the whole #3E open-ended GENERATE channel (the brain VOLUNTEERS a novel grounded HYPOTHESIS on
# an explicit open-ended prompt via generative replay over its own fact-association graph). Follows this codebase's
# universal faculty-switch convention ("default-ON; BRAIN_X=0 is the byte-identical-oracle escape" — cf.
# BRAIN_AFFECT / BRAIN_SURPRISE / BRAIN_METACOG / BRAIN_CURIOSITY / BRAIN_SPIKING_MOUTH): the channel is already the
# committed production default (ledger row open-ended-generation on_by_default:YES), so DEFAULT-ON preserves it.
# BRAIN_GENERATE_CHANNEL=0 (or false/off/no) disables the WHOLE channel: `_parse_open_ended` returns _NOT_OPEN_ENDED
# for EVERY turn, so gate()/gate_extract() fall through to the unchanged recall/abstain/learn/anaphora pipeline —
# byte-identical, and NO generative proposer / spiking-draw organ is ever built. This is the single clean master
# switch the channel previously lacked (the pre-existing BRAIN_SPIKING_DRAW/BRAIN_SPIKING_MOUTH flags only control
# HOW the channel draws/speaks, not WHETHER it fires).
_GENERATE_CHANNEL_DEFAULT_ON = True


def _generate_channel_enabled():
    """Whether the #3E open-ended GENERATE channel fires. Default = `_GENERATE_CHANNEL_DEFAULT_ON` (ON — the
    committed production default + the codebase's default-ON-with-=0-escape convention). Set BRAIN_GENERATE_CHANNEL
    to 0/false/off/no to disable the channel entirely (byte-identical to the pre-generate recall/abstain pipeline)."""
    v = os.environ.get("BRAIN_GENERATE_CHANNEL")
    if v is None:
        return _GENERATE_CHANNEL_DEFAULT_ON
    return v.strip().lower() not in ("0", "false", "off", "no", "")


# BRAIN-NATIVE PLAUSIBILITY (2026-09-01 burn-down): the #3E plausibility GATE — the decision "is `a` plausibly
# related to `ac`" that selects which recombinations pass — was a HOST float comparison `P[w1,w2] >= tau` over
# the brain's own co-occurrence matrix (the declared host residual of the generate-channel GO). This flag routes
# that gate through a SPIKING monosynaptic associative read (SpikingAssociativePlausibilityOrgan): the co-occurrence
# graph is installed as synapses and relatedness is decided by whether spikes propagate to the readout assembly.
# Default = `_SPIKING_PLAUSIBILITY_DEFAULT_ON`, now **ON** (2026-09-01 robustness rung landed). The QUALIFIED
# single-assembly read (research/runners/_brain_native_plausibility_derisk.py) matched host ON AVERAGE but had 2/6
# tiny-graph seeds underperform (parity 0.54/0.78) + suppressed generation, so it shipped OFF. The ENSEMBLE read
# (K=8 redundant readout populations averaged + density=0 internal recurrence + gain=12 in the non-saturating
# regime; PRODUCTION_READ_CONFIG in spiking_plausibility_organ.py) lifts agreement with host `P>=tau` to 1.0 on
# ALL 6 seeds -> the spiking gate reproduces the host relation EXACTLY: the 6-seed de-risk
# (research/runners/_plausibility_ensemble_graded_derisk.py) is provenance-clean (0 host P>=tau calls),
# lesion-load-bearing (shuffle/ablate collapse it), moat-safe, byte-identical-off, parity >= host AND generation >=
# host on EVERY seed. The host `P>=tau` shortcut is RETIRED to the =0 oracle. Because agreement is 1.0 the ON output
# equals the pure-host output (zero regression); the load-bearing proof that the BRAIN computes it is the synapse
# lesion + the 0 host-call provenance. =1/unset build the organ; BRAIN_SPIKING_PLAUSIBILITY=0 -> the host `_related`
# gate (byte-identical: the organ is never built).
_SPIKING_PLAUSIBILITY_DEFAULT_ON = True


def _spiking_plausibility_enabled():
    """Whether the #3E plausibility gate is computed by the brain (the spiking associative read) rather than the
    host `P>=tau` matrix comparison. Default ON (the ensemble read reaches host parity + generation on ALL 6 seeds,
    agreement 1.0; provenance-clean, lesion-load-bearing, moat-safe, byte-identical-off). BRAIN_SPIKING_PLAUSIBILITY=0
    -> the host `_related` gate (byte-identical — the plausibility organ is never built)."""
    v = os.environ.get("BRAIN_SPIKING_PLAUSIBILITY")
    if v is None:
        return _SPIKING_PLAUSIBILITY_DEFAULT_ON
    return v.strip().lower() not in ("0", "false", "off", "no", "")


# ============================================================================================================
# SELF/IDENTITY + ANAPHORA-MISS scaffold-retirement DE-RISK (scaffold-retirement-backlog rank-13, 2026-09-05).
# The 2026-08-12 CHOOSE-1 integration made the (agent, action) COMPREHENSION of a factual-SVO question NEURAL
# (`_neural_question_parse`, the on-brain `BridgeParser.role_of`) and AUTHORITATIVE — a comprehended parse feeds
# the substrate recall (+, when installed, the GNW ignition-bus combiner); a DECLINED parse honestly ABSTAINS
# instead of falling to `QuestionRouter.match_fact`'s role-blind keyword bag-of-words. That finding's own "Honest
# scope" named the residual verbatim: "the router... still owns self/identity + the anaphora-fallback" (also
# `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`'s content-selection row: "still the self/identity + noisy-anaphora
# fallback"). These two flags extend the SAME already-proven recipe to that residual — reuse, not a new
# mechanism — and default OFF (byte-identical) pending the de-risk in
# `research/runners/_selfid_anaphora_scaffold_derisk.py`:
#
#   BRAIN_NEURAL_SELFID  — self/identity comprehension:
#     (a) a self-referential FACTUAL SVO ('what do you eat?'): `_extract_route` today hard-blocks ANY self-alias-
#         bearing content from reaching `_neural_question_parse`, even with a genuine 2nd content word (the
#         action) present — so it ALWAYS falls to the host router, never the parser. Resolving the self-alias to
#         'brain' (mirroring `QuestionRouter._resolve_self`) BEFORE that block lets it flow through the IDENTICAL
#         on-brain-parser + substrate-recall path any other factual-SVO question already uses.
#     (b) a bare identity question ('what are you?' / 'who are you?'): no 2nd content word exists at all (the
#         predicate IS what's being asked), so there is no (agent, action) for the parser to comprehend — this is
#         NOT the neural-parser recipe, it is the SAME host regex/preference-list comprehension-HELPER convention
#         `_definitional_copula_route` already uses for 'what is X?' ('what/who is/are <subject>' -> [subject,
#         'isa']), extended to a self-alias subject (-> ['brain', 'isa']), plus a MISS-ONLY candidate-relation
#         retry in `_substrate_recall` mirroring the host router's OWN has/have/is/uses/use preference order —
#         the SAME shape as the existing alias-hop `v_candidates` fallback there, new candidate values only.
#         Recall stays entirely on the substrate (`what_does`) either way — this can only ADD a resolution, never
#         invent a fact (the moat is untouched). Class (a) reaches the genuinely-neural BridgeParser; class (b)
#         does not (be precise about which, in any report — see docs/TERMS.md "fully spiking").
#
#   BRAIN_NEURAL_ANAPHORA_ABSTAIN — the anaphora-miss: today, when the anaphora-resolved question's substrate
#     recall declines or finds no fact, `gate()` (and the installed GNW-bus `gate_via_bus`) fall to the host
#     router ("the WM referent may be noisy, so let the host router try"). This flag makes that ABSTAIN instead —
#     the SAME honesty already applied to the direct-query abstain (no host bag-of-words "rescue" of a possibly-
#     wrong referent, e.g. the "what does it fly?" -> wrong-referent -> keyword-confab shape).
#
# PRODUCTION FLIP (2026-09-05, Track-1 ship-the-validated-wins): the de-risk above earned GO 6/6 seeds, both
# combiners (research/findings/2026-09-05-rank13-selfid-anaphora-scaffold-derisk-GO-6of6.md). Both flags now
# default ON -- unset behaves as flag=1 (the neural self/identity comprehension + the honest anaphora-miss
# abstain). BRAIN_NEURAL_SELFID=0 / BRAIN_NEURAL_ANAPHORA_ABSTAIN=0 is the escape back to the pre-flip host-router
# behavior byte-identically, unchanged code path. See
# research/findings/2026-09-05-rank13-selfid-anaphora-PRODUCTION-FLIP-verify.md for the flip verification.
# ============================================================================================================

_NEURAL_SELFID_DEFAULT_ON = True
_NEURAL_ANAPHORA_ABSTAIN_DEFAULT_ON = True


def _neural_selfid_enabled():
    """BRAIN_NEURAL_SELFID=1 -> the self/identity comprehension extension above (a)+(b). Default OFF -> today's
    host-router-only self/identity path (byte-identical)."""
    v = os.environ.get("BRAIN_NEURAL_SELFID")
    if v is None:
        return _NEURAL_SELFID_DEFAULT_ON
    return v.strip().lower() not in ("0", "false", "off", "no", "")


def _neural_anaphora_abstain_enabled():
    """BRAIN_NEURAL_ANAPHORA_ABSTAIN=1 -> an anaphora-resolved question the substrate can't confirm ABSTAINS
    instead of falling to the host router. Default OFF -> today's host-router rescue (byte-identical)."""
    v = os.environ.get("BRAIN_NEURAL_ANAPHORA_ABSTAIN")
    if v is None:
        return _NEURAL_ANAPHORA_ABSTAIN_DEFAULT_ON
    return v.strip().lower() not in ("0", "false", "off", "no", "")


# ============================================================================================================
# Self-reference + a free-text question -> a (kind, cue) the brain answers against its stored SVO facts.
# (The keyword->fact matcher is faithful: it routes a question to the stored fact whose WORDS the question
# mentions, synonym-resolved; an unmatched question ABSTAINS -- the no-confab moat. Ported from the
# self-knowledge demo's router so a plain English question resolves, while carrying ZERO project knowledge.)
# ============================================================================================================

DEFAULT_SELF_ALIASES = {"you", "your", "yours", "i", "me", "my", "it", "its", "yourself", "itself"}

_STOP = {"what", "who", "does", "do", "the", "a", "an", "is", "are", "of", "to", "from", "that", "how",
         "did", "will", "can", "and", "with", "in", "on", "for", "by", "as", "be", "this", "these",
         "those", "there", "here", "prevent", "prevents", "tell", "about", "say", "know", "knows"}

_QUESTION_SYNONYMS = {
    "learn": {"learns", "learning"}, "learns": {"learns"},
    "forget": {"forgetting", "replays", "replay", "remembers"}, "forgetting": {"forgetting"},
    "remember": {"remembers", "memory"}, "memory": {"memory", "remembers", "consolidates"},
    "lie": {"moat", "confabulation", "abstains", "refuses", "honest"},
    "lying": {"moat", "confabulation", "abstains", "refuses", "honest"},
    "guess": {"moat", "confabulation", "refuses", "guessing"},
    "use": {"uses"}, "uses": {"uses"}, "using": {"uses"},
    "teach": {"teaches"}, "teaches": {"teaches"}, "taught": {"teaches"},
    "store": {"stores", "remembers", "composer"}, "speak": {"phrases", "faculty", "answers"},
    "answer": {"answers", "remembers"}, "think": {"uses", "neurons"}, "work": {"uses", "runs"},
    "consolidate": {"consolidates"}, "grow": {"grows", "develops", "tiers"},
    "develop": {"develops", "daily"}, "made": {"has", "uses", "neurons", "spikes"}, "make": {"has", "uses"},
}


class QuestionRouter:
    """Map a free-text question to a stored SVO fact (the GATE cue), resolving self-aliases. Decisive only when a
    CONTENT keyword of the question appears in some fact (a bare self-alias match is not enough -> abstain)."""

    def __init__(self, self_aliases=None):
        self.self_aliases = set(self_aliases) if self_aliases else set(DEFAULT_SELF_ALIASES)

    def _resolve_self(self, word):
        w = word.lower().strip(".,!?")
        return "brain" if w in self.self_aliases else w

    def keywords(self, question):
        toks = [self._resolve_self(t) for t in re.findall(r"[a-zA-Z]+", question.lower())]
        kws = set()
        for t in toks:
            if t in _STOP and t != "brain":
                continue
            kws.add(t)
            kws |= _QUESTION_SYNONYMS.get(t, set())
        return kws, toks

    def match_fact(self, question, stored_facts):
        """Return (gate_svo or None, score). The best stored fact by content-keyword overlap; an identity question
        ('what are you') routes to a defining 'brain has/is/uses ...' fact."""
        kws, toks = self.keywords(question)
        content_kws = kws - {"brain"}
        is_identity_q = ("brain" in kws and not content_kws
                         and any(w in {"be", "are", "is", "am"} for w in toks))
        if is_identity_q:
            # a defining fact about the brain, in preference order (covers base + 3rd-person inflected verbs)
            for want in ("has", "have", "is", "uses", "use"):
                for (a, v, p) in stored_facts:
                    if a == "brain" and v == want:
                        return [a, v, p], 1
            # fall back to ANY fact whose agent is 'brain' (the brain's own self-statement)
            for (a, v, p) in stored_facts:
                if a == "brain":
                    return [a, v, p], 1
        best, best_score = None, 0
        for (a, v, p) in stored_facts:
            ftoks = {a, v, p}
            content_hits = len(content_kws & ftoks)
            brain_hit = 1 if ("brain" in kws and "brain" in ftoks) else 0
            score = content_hits * 10 + brain_hit
            if content_hits >= 1 and score > best_score:
                best, best_score = (a, v, p), score
        return (list(best) if best is not None else None), best_score


# ============================================================================================================
# The fluent renderers (default = the off-bridge Qwen; --stub-renderer = the template-stub, GPU-free).
# Both expose `render_svo(a, v, p) -> (surface, asserted_svo_or_None)`; the TUI gate->constrain->verify wraps them.
# ============================================================================================================

class StubRenderer:
    """The GPU-FREE template-stub faculty (the P3 `TemplateStubFaculty`): renders a gated SVO into a fluent
    surface form CONSTRAINED to the fact's own words, and exposes the canonical content SVO it asserts (what
    VERIFY re-parses). Stands in for the real Qwen renderer in the CPU smoke -- NO model download, deterministic."""

    name = "template-stub (GPU-free)"

    def __init__(self):
        from research.runners._grounded_lang_p3_derisk import TemplateStubFaculty
        self._fac = TemplateStubFaculty()

    def render_svo(self, a, v, p):
        surface, asserted = self._fac.render_svo(a, v, p)
        return surface, asserted


class QwenRenderer:
    """The OFF-BRIDGE Qwen-0.5B grounded-language faculty (the spiking forward, reused-by-import from the
    integration de-risk). Loaded ONCE + kept warm. `render_svo` returns the generated prose + None for the
    asserted SVO (the TUI re-parses the PROSE to recover the asserted content -- the genuine VERIFY of a real
    generative model's output)."""

    name = "off-bridge Qwen-0.5B (spiking forward)"

    def __init__(self, T=16, max_new_tokens=24, seed=42):
        from research.runners._grounded_lang_integration_derisk import SpikingQwenFaculty
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("[tui] WARNING: CUDA not available -- the Qwen renderer will be slow on CPU.", flush=True)
        self._fac = SpikingQwenFaculty(T=T, max_new_tokens=max_new_tokens, seed=seed, device=device)
        self.load_seconds = self._fac.load_seconds

    def render_svo(self, a, v, p):
        surface, _surface_full, _gen_s = self._fac.render_svo(a, v, p)
        return surface, None      # asserted SVO recovered by the TUI's re-parse of the prose

    def render_svo_regen(self, a, v, p):
        surface, _surface_full, _gen_s = self._fac.render_svo_regen(a, v, p)
        return surface, None

    def render_svo_batch(self, triples):
        """BATCHED CONSTRAIN (2026-09-01, `research/batch-sentence-rendering`): render every (a, v, p) in
        `triples` via ONE `model.generate()` launch (`SpikingQwenFaculty.render_svo_batch`) instead of one
        launch per triple. Returns `[(surface, None), ...]` -- same per-item shape as `render_svo` (the asserted
        SVO is recovered by the caller's re-parse of the prose either way), same order as `triples`. Additive:
        `render_svo` / `render_svo_regen` above are untouched; a caller that never calls this (the default,
        `hasattr(renderer, "render_svo_batch")` gated) never triggers it."""
        results, secs = self._fac.render_svo_batch(list(triples))
        return [(surface, None) for (surface, _full) in results], secs

    def render_svo_regen_batch(self, triples):
        """BATCHED REGEN (2026-09-01): the batched analogue of `render_svo_regen`, so a caller can retry every
        first-pass VERIFY-reject together in ONE second launch instead of falling back to N separate
        single-item regen calls. Same shape/contract as `render_svo_batch`."""
        results, secs = self._fac.render_svo_regen_batch(list(triples))
        return [(surface, None) for (surface, _full) in results], secs


# ============================================================================================================
# The chat brain: wraps a loaded conversational agent + the router + the renderer + the gate/constrain/verify.
# ============================================================================================================

class ChatBrain:
    def __init__(self, agent, *, self_aliases=None, renderer=None, verbose_thinking=True):
        # agent is a MultiTurnAgent (preferred, for anaphora) or a BrainConversationalAgent
        self.agent = agent
        self.inner = getattr(agent, "agent", agent)             # the BrainConversationalAgent
        self.is_multiturn = hasattr(agent, "held_referent")     # MultiTurnAgent exposes this
        self.router = QuestionRouter(self_aliases=self_aliases)
        self.renderer = renderer
        self.verbose_thinking = verbose_thinking
        self.raw_mode = False                                   # /raw toggles the brain's own renderer (no LLM)
        # DISCOURSE EVENT TRACKING (2026-07-10): if the agent carries an event register, the console can HEAR a
        # multi-clause discourse and answer "who was doing it before?" across a connective (the D3 event-register arc,
        # deployed on MultiTurnAgent but previously unreachable in any console). Backward-compatible: no register -> off.
        self.has_event_register = self.is_multiturn and getattr(agent, "_event_register", None) is not None
        self._boundary_seen = False          # "who was doing it before?" only has meaning AFTER a discourse boundary
        self._heard_any_clause = False
        # OPEN-ENDED GENERATION (#3E wire-in): a lazily-built, fact-count-cached b2 generative-replay proposer over
        # the brain's OWN association structure. Fires ONLY on an explicit open-ended prompt (see gate()); every
        # other turn is untouched. Config below matches the #3E de-risk operating point (tau = 50th pctile of the
        # positive co-occurrence edges; see `_gen_spiking` below for the DRAW choice).
        self._gen_proposer = None
        self._gen_nfacts = None
        self._gen_tau_pct = 50.0
        self._gen_n_attempts = 400
        self._gen_min_facts = 3
        self._gen_seed = int(getattr(self.inner, "seed", 42)) * 7 + 1
        # the generative DRAW is the b2 HOST oracle (numpy weighted sampling). The b2 SPIKING soft-WTA sampler
        # (SpikingWTASampler) hardcodes the 8x8-taxonomy role pools and KeyErrors on an arbitrary conversational
        # vocab, so it cannot encode a runtime-grown lexicon; the host oracle is the b2-sanctioned numpy path
        # (`_genfrontier_b2` retains it for exactly the numpy-CPU/reproducibility case). The LOAD-BEARING part is
        # the plausibility SIGNAL — the brain's own learned fact-association graph — which is the brain's here.
        self._gen_spiking = False
        # VOCAB-AGNOSTIC SPIKING DRAW organ (B1 burn-down, 2026-08-13): converts the #3E generative DRAW above from
        # the host oracle to a genuinely-SPIKING soft-WTA read off cp_firing_states. The b2 taxonomy SpikingWTASampler
        # KeyErrors on a runtime lexicon (see comment above); this organ induces role pools from the brain's OWN
        # stored-fact concepts (no taxonomy) and pre-injects a taxonomy-free VocabAgnosticSpikingSampler. Built LAZILY
        # on the first open-ended generation turn (per-ChatBrain, avoiding process-singleton cache thrashing across
        # sessions/brains), so a session that never generates never imports it and its non-generation turns are
        # untouched. Default-ON; BRAIN_SPIKING_DRAW=0 leaves `_gen_spiking=False` -> the host oracle draw
        # (byte-identical). See research/runners/vocab_agnostic_spiking_generation_production_organ.py.
        self._spiking_draw_organ = None
        # BRAIN-NATIVE PLAUSIBILITY organ (2026-09-01 burn-down): converts the #3E plausibility GATE from the host
        # `_related(w1,w2) = P[w1,w2] >= tau` matrix comparison to a SPIKING monosynaptic associative read of the
        # co-occurrence graph (installed as synapses; relatedness = whether spikes reach the readout assembly).
        # Built LAZILY on the first open-ended generation turn (per-ChatBrain), so a non-generating session never
        # imports it. Default-ON (the ensemble read reaches host parity + generation on all 6 seeds — see the
        # 2026-09-01 ensemble finding); BRAIN_SPIKING_PLAUSIBILITY=0 -> the host `_related` gate (byte-identical).
        self._spiking_plausibility_organ = None
        # the brain's stored facts (string-only roles) + content-token sets for the VERIFY re-parse
        self._refresh_facts()

    def _refresh_facts(self):
        comp = self.inner.composer
        self.stored_facts = [(f.get("agent"), f.get("action"), f.get("patient")) for f, _ in comp.kb
                             if all(isinstance(f.get(r), str) for r in ("agent", "action", "patient"))]
        self.agents_set = {a for a, _, _ in self.stored_facts}
        self.actions_set = {v for _, v, _ in self.stored_facts}
        self.patients_set = {p for _, _, p in self.stored_facts}
        from research.runners._grounded_lang_integration_derisk import _build_inflection_map
        self.inflect = _build_inflection_map(sorted(self.actions_set))

    # --- the GATE: a free-text question -> a verified stored SVO fact, or None (abstain) ---
    def gate(self, question):
        """Resolve the question to a stored fact and VERIFY it against the spiking recall. Returns
        (gate_svo or None). An anaphor in the question is resolved from the discourse WM (multi-turn)."""
        # OPEN-ENDED GENERATION (#3E production wire-in) — fires ONLY on an EXPLICIT open-ended prompt pattern
        # ("what might X ...", "tell me something new about X", "what else about X", "guess ..."). On a match, the
        # brain VOLUNTEERS a novel grounded proposition via generative replay over its own learned association
        # graph, moat-verified + FLAGGED as a hypothesis (or abstains -> None; never confabulates). For EVERY
        # OTHER question `_parse_open_ended` returns the `_NOT_OPEN_ENDED` sentinel and gate() falls through to the
        # unchanged recall/abstain/learn/anaphora pipeline below — byte-identical.
        oe = self._parse_open_ended(question)
        if oe is not _NOT_OPEN_ENDED:
            return self._generate_hypothesis(*oe)
        # resolve anaphora in the question FIRST (multi-turn): replace a leading 'it'/'that'/'they' with the held
        # referent, so a follow-up 'what does it eat' uses the prior turn's referent.
        acq = self._maybe_acquire(question)      # IN-LOOP LEARNING (production path): an SVO ASSERTION is TAUGHT here in
        if acq is not None:                      # gate() so the /api/brain-chat endpoint (which calls gate(), NOT
            return acq                           # answer()) reaches it; gate returns the acquired SVO -> render confirms.
        q = self._resolve_anaphora(question)
        anaphora_used = (q != question)          # the extracted agent came from the (noisy) discourse WM, not the user
        # SUBSTRATE-FIRST recall (production-integration #2, in-loop learning). For a well-formed "what does AGENT
        # ACTION?" question where AGENT+ACTION are known, recall the patient FROM THE SPIKING SUBSTRATE
        # (`inner.what_does`) — which is ROLE-AWARE (it queries the specific (agent, action) binding, not the host
        # router's role-blind keyword overlap) AND sees a fact HEARD this conversation. `what_does` returns the stored
        # patient only if the binding is genuinely in the substrate, so this cannot confabulate (the no-confab moat
        # holds). The host QuestionRouter remains the fallback for self/identity questions and anything not in this form.
        sub = self._substrate_recall(q)
        if sub == "__ABSTAIN__" and not anaphora_used:
            return None                          # DIRECT well-formed query, substrate has no fact -> honest abstain
                                                 # (fixes the host-router keyword CONFAB, e.g. "what does fish fly?").
        if sub == "__ABSTAIN__" and anaphora_used and _neural_anaphora_abstain_enabled():
            return None                          # ANAPHORA-MISS EXTENSION (rank-13 de-risk, default OFF): the SAME
                                                 # honest abstain as the direct-query case above, applied to an
                                                 # anaphora-resolved query the substrate can't confirm -- retires the
                                                 # host router's "rescue" (its keyword match can confab off a
                                                 # possibly-wrong WM referent, e.g. "what does it fly?" -> wrong
                                                 # referent -> "cat eat fish"). Flag OFF: unchanged (falls through).
        if sub not in (None, "__ABSTAIN__"):     # anaphora abstain falls through: the WM referent may be noisy, so let
            return sub                           # the host router try (its keyword match masks a bad WM pick).
        # (host router fallback + spiking VERIFY combination) — factored into `_gate_router_combine`; reached ONLY for
        # the out-of-scope classes the substrate GNW ignition bus does NOT author (a self/identity turn routed by the
        # host QuestionRouter, and the anaphora-abstain fall-through). Keeping it inline here is byte-identical.
        return self._gate_router_combine(q)

    def _gate_router_combine(self, q):
        """The HOST QuestionRouter fallback + spiking VERIFY combination (`if recalled == p`), factored out of gate().
        Reached for the classes the substrate ignition bus does NOT author: a self/identity turn (routed by the host
        router) and the anaphora-abstain fall-through. Returns [a, v, p] on a VERIFIED match, else None. This is the
        residual host combination the scaffold-retirement KEEPS for the out-of-scope classes (the COVERED-class
        combination is retired in `gate_via_bus`, which never calls this on a routable factual recall)."""
        gate_svo, _score = self.router.match_fact(q, self.stored_facts)
        if gate_svo is None:
            return None
        a, v, p = gate_svo
        # VERIFY the matcher's pick against the brain's SPIKING recall (the answer must be the spiking memory's)
        recalled = self.inner.what_does(a, v)
        if recalled == p:
            # write the answer's salient referent (the PATIENT/object) into the discourse WM so a NEXT-turn pronoun
            # resolves to it -- exactly as MultiTurnAgent.hear() writes only the patient. We treat a CONCRETE entity
            # (one that is itself the AGENT of some fact -- i.e. something the brain can say more about) as the
            # discourse referent; this matches the validated single-referent anaphora pattern (a fresh referent
            # dominates the WM) and avoids polluting the WM with abstract patients (e.g. 'spikes'/'words') that are
            # not salient pronoun antecedents. The no-confab moat is unaffected.
            if isinstance(p, str) and p in self.agents_set:
                self._note_referent(p)
            return [a, v, p]
        return None

    def gate_extract(self, question):
        """EXTRACTION + SIDE-EFFECT phase of gate(), WITHOUT the covered-class recall-COMBINATION verdict — so a
        combiner (the host `if recalled == p`, OR the substrate GNW ignition bus) authors the verdict. Runs the SAME
        open-ended / acquisition / anaphora side effects gate() runs, then returns a discriminated tuple:
          ('done', svo)                      -- an OUT-OF-SCOPE class already produced its answer: an open-ended
                                                HypothesisSVO guess, or an in-loop ACQUISITION (`svo` may be None =
                                                honest abstain). The bus does not author these; the host mechanism did.
          ('route', q, a, v, anaphora_used)  -- a ROUTABLE factual query: the COMBINER recalls (a, v) and commits the
                                                ignited patient or abstains. THIS is the class the substrate bus authors
                                                (in gate_via_bus the host `if recalled == p` is never computed for it).
          ('decline', q, anaphora_used)      -- a factual-shaped question the on-brain parser DECLINED (comprehension
                                                abstain): abstain unless anaphora, then the host router may try.
          ('router', q)                      -- unroutable (self/identity/short): the HOST ROUTER owns it (out of scope).
        Extraction/comprehension + every side effect stay the host's (unchanged); ONLY the recall-COMBINATION is
        deferred to the combiner. gate() itself does not consume this (it stays byte-identical via `_substrate_recall`,
        which the production lesion probe patches); the webapp bus wrapper consumes it via `gate_via_bus` (the
        scaffold-retirement: the covered-class host combination is never computed)."""
        oe = self._parse_open_ended(question)
        if oe is not _NOT_OPEN_ENDED:
            return ('done', self._generate_hypothesis(*oe))
        acq = self._maybe_acquire(question)
        if acq is not None:
            return ('done', acq)
        q = self._resolve_anaphora(question)
        anaphora_used = (q != question)
        route = self._extract_route(q)                       # (agent, action) comprehension ONLY — no recall verdict
        if route == "__DECLINE__":
            return ('decline', q, anaphora_used)
        if isinstance(route, list):
            return ('route', q, route[0], route[1], anaphora_used)
        return ('router', q)

    # --- OPEN-ENDED GENERATION (#3E: the brain VOLUNTEERS novel grounded propositions via generative replay) ---
    def _parse_open_ended(self, question):
        """Detect an EXPLICIT open-ended generation prompt. Returns `(topic, action)` on a match (either may be
        None: a bare 'guess' -> free generation), else the `_NOT_OPEN_ENDED` sentinel. Deliberately conservative:
        only the fixed lead-ins in `_OPEN_ENDED_PATTERNS` match, so a normal recall/teach/yes-no/anaphora turn
        never enters the generation branch and gate() stays byte-identical."""
        # MASTER SWITCH (default-ON; BRAIN_GENERATE_CHANNEL=0 = byte-identical escape): when the GENERATE channel is
        # OFF, treat EVERY turn as not-open-ended, so gate()/gate_extract() never enter the generation branch and the
        # recall/abstain/learn/anaphora pipeline is byte-identical (no proposer / spiking-draw organ ever built).
        if not _generate_channel_enabled():
            return _NOT_OPEN_ENDED
        ql = question.lower().strip()
        for rx, _has_topic in _OPEN_ENDED_PATTERNS:
            m = rx.match(ql)
            if m is None:
                continue
            gd = m.groupdict()
            topic = (gd.get("topic") or "").strip(".,!? ") or None
            action = (gd.get("action") or "").strip(".,!? ") or None
            if topic in self.router.self_aliases:      # 'you'/'it' -> the brain's self-facts
                topic = "brain"
            return (topic, action)
        return _NOT_OPEN_ENDED

    def _build_generation_proposer(self):
        """Build (and fact-count cache) the #3E/b2 `GenerativeReplayProposer` over the brain's OWN association
        structure. The plausibility graph P is the brain's CLEAN concept co-occurrence over its stored facts (the
        agent's association structure — every fact's agent/action/patient co-occur), which is what the dlPFC
        `_assoc_graph` learned graph approximates but WITHOUT that graph's dense reserve-slot noise (which floods
        implausible recombinations) and WITH the runtime-taught facts the fixed-vocab `_learned_assoc` never sees.
        tau = the 50th percentile of the positive edges (the #3E operating point → 'related' = co-occurred). This
        is the same host-computed selectional-preference plausibility signal #3E used (there a corpus PPMI; here
        the brain's own heard facts), gating the reused b2 proposer's replay. Returns the proposer, or None when
        the brain knows too few facts (-> the caller abstains, never confabulates)."""
        facts = list(self.stored_facts)
        if len(facts) < self._gen_min_facts:
            return None
        if self._gen_proposer is not None and self._gen_nfacts == len(facts):
            return self._gen_proposer
        # clean concept co-occurrence over the brain's stored facts (agent/action/patient of each fact co-occur).
        graph = {}
        for a, v, p in facts:
            cs = [c for c in (a, v, p) if isinstance(c, str)]
            for x in cs:
                for y in cs:
                    if x != y:
                        graph.setdefault(x, {})[y] = graph.get(x, {}).get(y, 0.0) + 1.0
        vocab = sorted(graph.keys())
        if len(vocab) < 3:
            return None
        row = {w: i for i, w in enumerate(vocab)}
        P = np.zeros((len(vocab), len(vocab)), dtype=float)
        for a, nbrs in graph.items():
            for b, w in nbrs.items():
                P[row[a], row[b]] = float(w)          # symmetric by construction of the co-occurrence
        pos = P[P > 0]
        if pos.size == 0:
            return None
        tau = float(np.percentile(pos, self._gen_tau_pct))
        from research.runners._genfrontier_b2_generative_replay_derisk import GenerativeReplayProposer
        # the proposer reads the SAME composer the brain answers through (so a generated proposition must not
        # contradict a stored fact, and must never pass known-fact retrieval). negated=[] here: the composer's own
        # `ask_yes_no` (which the proposer's non-contradiction gate reads) still catches any stored negation.
        self._gen_proposer = GenerativeReplayProposer(
            self.inner.composer, facts, [], P, row, tau,
            np.random.default_rng(self._gen_seed), use_spiking_sampler=self._gen_spiking)
        self._gen_nfacts = len(facts)
        return self._gen_proposer

    def _generate_hypothesis(self, topic=None, action=None, n_attempts=None):
        """GENERATE a novel grounded proposition (the #3E faculty), optionally about `topic` (its agent) and/or
        `action`. Draws role-fillers with the reused b2 proposer's OWN weighted sampler (`_sample_weighted` /
        `_weight_partner`), gates each candidate with the reused b2 `_plausible` (selectional-preference over the
        brain's learned association graph) + `_contradicts` (non-contradiction vs the composer's store), then
        MOAT-VERIFIES it (not a degenerate self-loop; matches the requested topic/action; and — the no-confab
        guarantee — must NOT pass known-fact retrieval: `what_does` != patient AND `is_it_true` == 'unknown').
        EARLY-STOPS at the first passing proposal (so a turn runs only a few spiking moat queries, not a full
        exhaustive replay) and returns it as a FLAGGED `HypothesisSVO`; returns None (honest abstain) when no
        plausible grounded proposal exists. An unknown topic (not a known agent) ABSTAINS — the brain does not
        invent about what it has never heard of."""
        if topic is not None and topic not in self.agents_set:
            return None                                # unknown subject -> abstain (no confabulation)
        prop = self._build_generation_proposer()
        if prop is None:
            return None
        # BRAIN-NATIVE PLAUSIBILITY (opt-in, BRAIN_SPIKING_PLAUSIBILITY=1): route the #3E plausibility GATE through the
        # SPIKING monosynaptic associative read instead of the host `_related(w1,w2) = P[w1,w2] >= tau` comparison. install() embodies
        # the co-occurrence graph as synapses (weight ∝ co-occurrence count) and swaps prop._related for a spiking read
        # (drive w1's assembly; w2 is "related" iff its readout assembly fires above the brain's own threshold). The
        # UNCHANGED `_plausible` (= _related(a,ac) and _related(ac,p)) then decides via SPIKES. BRAIN_SPIKING_PLAUSIBILITY=0
        # -> the organ is never built and prop keeps the host `_related` (byte-identical). Built lazily + cached per proposer.
        if _spiking_plausibility_enabled():
            if (self._spiking_plausibility_organ is None
                    or getattr(prop, "_spiking_plausibility_organ", None) is not self._spiking_plausibility_organ):
                from research.runners.spiking_plausibility_organ import build_for_proposer
                self._spiking_plausibility_organ = build_for_proposer(prop, seed=self._gen_seed)
                self._spiking_plausibility_organ.install(prop)
        # Route the #3E generative DRAW through the VOCAB-AGNOSTIC spiking soft-WTA (default-ON, B1 burn-down): the b2
        # taxonomy sampler KeyErrors on runtime vocab, so install() induces role pools from the brain's OWN stored-fact
        # concepts and pre-injects a taxonomy-free VocabAgnosticSpikingSampler onto `prop` (flips use_spiking_sampler=True).
        # The UNCHANGED loop below then draws on FIRING NEURONS (prop._sample_weighted -> the injected sampler ->
        # draw_from_weights reads cp_firing_states). BRAIN_SPIKING_DRAW=0 -> install() is a no-op -> the host oracle draw
        # (byte-identical). BRAIN_SPIKING_DRAW_LESION=1 -> likelihood ablated (uniform drive) -> plausibility collapses.
        # Every downstream gate (_plausible / _contradicts) + the #3E moat verify are UNTOUCHED.
        if self._spiking_draw_organ is None:
            from research.runners.vocab_agnostic_spiking_generation_production_organ import (
                VocabAgnosticSpikingDrawOrgan,
            )
            self._spiking_draw_organ = VocabAgnosticSpikingDrawOrgan(seed=self._gen_seed)
        self._spiking_draw_organ.install(prop)
        if action is not None and action not in self.actions_set:
            action = None                              # a requested action the brain doesn't know -> don't hard-filter
        agents = [topic] if topic is not None else list(prop.agents)
        if not agents or not prop.actions or not prop.patients:
            return None
        n = int(self._gen_n_attempts if n_attempts is None else n_attempts)
        rng = prop.rng
        seen = set()
        for _ in range(n):
            a = agents[0] if len(agents) == 1 else agents[int(rng.integers(len(agents)))]
            ac = action if action is not None else prop._sample_weighted(
                prop.actions, prop._weight_partner((a,), prop.actions))
            p = prop._sample_weighted(prop.patients, prop._weight_partner((a, ac), prop.patients))
            triple = (a, ac, p)
            if a == p or triple in seen or triple in prop.all_stored:
                continue                               # degenerate / repeat / a stored fact (only NOVEL counts)
            seen.add(triple)
            if not prop._plausible(a, ac, p):          # b2 selectional-preference plausibility gate (reused)
                continue
            if prop._contradicts(a, ac, p):            # b2 non-contradiction gate (reads the composer's ask_yes_no)
                continue
            # MOAT VERIFY (the #3E hypothesis-not-known guarantee): a HYPOTHESIS never passes as a known fact.
            if self.inner.what_does(a, ac) == p or self.inner.is_it_true(a, ac, p) != "unknown":
                continue
            return HypothesisSVO([a, ac, p])
        return None

    def _resolve_anaphora(self, question):
        """If the question's first content token is a pronoun and the discourse WM holds a referent, substitute it
        (multi-turn anaphora). Only the MultiTurnAgent has a WM loop; otherwise pass the question through."""
        if not self.is_multiturn:
            return question
        anaphors = {"it", "that", "they", "them", "this"}
        toks = question.split()
        for i, t in enumerate(toks):
            tl = t.lower().strip(".,!?")
            if tl in anaphors:
                ref = self.agent.held_referent()[0]
                if ref is not None:
                    toks[i] = ref
                    return " ".join(toks)
        return question

    def _maybe_generate(self, question):
        """GENERATION: for an open-ended TOPIC prompt ('tell me about X' / 'describe X' / 'what about X'), VOLUNTEER what
        the brain knows about X by CHAINING ASSOCIATIONS on the substrate — describe(X) plus the dlPFC spiking
        `elaborate` (spreading-activation content-selection) to a related concept and describe THAT. This is generation
        from the brain's own knowledge, beyond single-fact recall. Returns (answer, abstained) or None. No confab:
        describe() returns None for an unknown topic (-> falls through to abstain)."""
        ql = question.lower().strip().rstrip("?. ")
        topic = None
        for pat in ("tell me about ", "describe ", "what about ", "what do you know about ", "say something about "):
            if ql.startswith(pat):
                topic = ql[len(pat):].strip().split()[-1] if ql[len(pat):].strip() else None
                break
        if not topic:
            return None
        topic = topic.strip(".,!?")
        if topic in self.router.self_aliases:
            topic = "brain"
        try:
            primary = self.inner.describe(topic)
        except Exception:
            primary = None
        if not primary:
            return None                          # unknown topic -> let the pipeline abstain (no confabulation)
        parts = [primary]
        try:                                     # ONE associative hop via the dlPFC spiking spreading-activation control
            assoc = self.inner.elaborate(topic)
            if assoc and assoc != topic:
                more = self.inner.describe(assoc)
                if more and more != primary:
                    parts.append(more)
        except Exception:
            pass
        return " ".join(p.rstrip(".") + "." for p in parts), False

    def _maybe_acquire(self, question):
        """IN-LOOP LEARNING acquisition: if the input is a declarative 3-word SVO ASSERTION (not a question), TEACH it to
        the spiking substrate (`inner.hear` -> composer.store with runtime code allocation for any new word) and refresh
        the recallable vocabulary, then acknowledge. Returns (answer, abstained) or None (not an assertion). This is what
        lets the owner grow the brain's knowledge by talking to it."""
        q = question.strip()
        ql = q.lower()
        if "?" in q or ql.split()[:1] and ql.split()[0] in (
                "what", "who", "whom", "where", "when", "why", "how", "is", "are", "was", "were", "does", "do", "did"):
            return None
        # ── NON-CONTRADICTION STORE-SIDE (Gate-B, B3, 2026-08-12) ──────────────────────────────────────────
        # So the non-contradiction gate has NEGATIONS to fire against (today the console stores ZERO negations — the
        # legacy path below hard-codes polarity="AFFIRM" and only acquires an EXACTLY-3-whitespace-token input),
        # acquire a heard assertion with its DETECTED polarity via the B3 organ's extractor: it strips negation cues +
        # function words to expose the 3-token SVO content and tags a heard negation ("the dog does not eat grass") as
        # NEGATE, using the SAME function-word-strip the gate's recall uses (so store + recall AGREE). Additive +
        # guarded: falls back to the EXACT legacy 3-token / AFFIRM path when B3 is unavailable OR disabled
        # (BRAIN_NONCONTRADICTION_GATE=0) -> byte-identical acquisition. (This edits the host conversational scaffold,
        # NOT sim/.)
        try:
            import research.runners.b3_noncontradiction_production_organ as _b3nc
            _b3nc_on = _b3nc.noncontradiction_enabled()
        except Exception:
            _b3nc = None
            _b3nc_on = False
        # VERB LEMMATIZATION AT STORE-WRITE (reasoning-frontier, 2026-08-25 -- see research/runners/lexical_lemma.py
        # + research/findings/2026-08-25-reasoning-frontier-chain-routing.md). The 2026-08-25 integrated-conversational
        # -state diagnostic found in-loop teaching FRAGILE to verb inflection: "the wolf hunts the deer" stored the
        # SURFACE token "hunts" as the action, so "what does the wolf hunt?" (extracting the base "hunt") missed on a
        # plain string-key mismatch -- an honest fact the brain was JUST told came back ABSTAIN. Canonicalizing the
        # action to ONE lemma key HERE (store-write) and symmetrically at query time (`_substrate_recall` below) makes
        # hunts/hunt/hunted collapse to the same stored key. HOST-SIDE SCAFFOLD (documented, not biology): a rule-based
        # suffix stemmer, no learned/spiking morphology segmentation exists yet (the named next rung). Byte-identical
        # for every already-base-form verb (lemma_verb is a no-op on a word with no matching inflectional suffix).
        if _b3nc_on:
            parsed = _b3nc.extract_polar_assertion(q)   # (agent, action, patient, polarity) or None (out of scope)
            if parsed is None:
                return None
            a, v, p, pol = parsed
            v = lemma_verb(v)
            try:
                self.inner.hear("%s %s %s" % (a, v, p), polarity=pol)   # a heard NEGATION stores as NEGATE
            except Exception:
                return None
            self._refresh_facts()
            return [a, v, p]
        # (B3 unavailable / disabled) — the EXACT legacy path (byte-identical acquisition) + the SAME lemma canonicalization
        toks = [t.strip(".,!?") for t in q.split() if t.strip(".,!?")]
        if len(toks) != 3:                       # the minimal SVO assertion the parser handles
            return None
        a, v, p = toks
        v = lemma_verb(v)
        try:
            self.inner.hear("%s %s %s" % (a, v, p), polarity="AFFIRM")
        except Exception:
            return None
        self._refresh_facts()                    # pick up the new fact -> agents_set/actions_set now include it
        return [a, v, p]                         # the acquired SVO; gate() returns it so the endpoint renders a confirm

    def _neural_question_parse(self, content):
        """CHOOSE (#1) — comprehend the question's (agent, action) NEURALLY. Present the stripped content words
        (position-padded to SVO, the queried patient a placeholder) to the ON-BRAIN BridgeParser, whose (position,
        voice)->role conjunction FIRES the role assignment on Izhikevich neurons — the SAME parser `hear()` uses to
        comprehend a stored sentence. Returns (agent, action) or None. This replaces the host first-known-token /
        positional heuristic so the question COMPREHENSION is on the substrate, not a Python vocabulary lookup. Requires
        the composer to carry a parser (the onebrain default does); returns None otherwise, so the caller falls back to
        the host heuristic (the rf escape path). Lesioning the parser -> role_of returns junk -> None -> the fact is not
        recalled (the load-bearing test)."""
        parser = getattr(getattr(self.inner, "composer", None), "parser", None)
        if parser is None or len(content) < 2:
            return None
        padded = [content[0], content[1], "__q__"]           # SVO with the queried patient a placeholder
        try:
            role_map = {}
            for pos in range(3):
                role_map[parser.role_of(pos)] = padded[pos]  # each position's role FIRES on the parser ensembles
        except Exception:
            return None
        a, v = role_map.get("agent"), role_map.get("action")
        if not (a and v) or a == v or a == "__q__" or v == "__q__":
            return None                                       # degenerate/lesioned parse -> let the caller fall back
        return a, v

    def _relation_fronted_route(self, question):
        """A RELATION-FRONTED question -- 'what country is chelsea fc from?', 'what sport is chelsea fc in?' --
        asks for ENTITY's RELATION value, but fronts the RELATION NOUN before the copula ('what <relation>
        is/are/was/were <entity> [prep]?'). This is a DIFFERENT surface shape from the in-conversation teaching
        pattern 'what does <entity> <verb>?' (the auxiliary 'does' immediately follows 'what', with no relation
        noun in between) that `_extract_route`'s generic (agent, action) = (content[0], content[1]) positional
        parse already handles correctly by ASSUMING SVO word order (entity first, relation/verb second).

        Without this route, a relation-fronted question hits that same generic positional parse and gets its
        roles SWAPPED: content[0] is the relation word, content[1] is (part of) the entity, so the parse
        extracts (agent=relation, action=entity) instead of (agent=entity, action=relation). The substrate then
        correctly has no fact under that backwards binding and honestly abstains -- reproducing Vikunja #142
        exactly: 'what country is chelsea fc from?' -> the generic parse extracts (agent='country',
        action='chelsea') -> `what_does('country', 'chelsea')` -> nothing stored -> 'I don't know about that.',
        even though the shipped Wikidata core holds (chelsea_fc, country, united_kingom) and
        `composer.query_patient('chelsea_fc', 'country')` answers it directly (see the 2026-08-27 finding's
        traced repro, which confirms this is the exact stage that drops the answer -- not affect/topic-tracking/
        the GNW consensus bus, which never see a routable (agent, action) pair to begin with here).

        COMPREHENSION only (mirrors `_definitional_copula_route`): tries the entity phrase against the
        knowledge-grounding alias hop (so a Wikidata alias surface form still resolves), falling back to the
        naive underscore-joined phrase (which already matches this store's own canonical-token convention, e.g.
        'chelsea fc' -> 'chelsea_fc', with zero alias facts required). Returns [agent, action] for the caller's
        UNCHANGED `what_does()` recall + no-confab moat (an unknown entity or relation still honestly abstains
        -- this can only ADD a resolution, never invent a fact), or None when the shape doesn't match / the
        lesion flag is set (falls straight through to the untouched generic parse, byte-identical for every
        other question)."""
        if not _relation_fronted_enabled():
            return None
        m = _REL_FRONTED_RE.match(question.strip())
        if not m:
            return None
        relation = m.group("relation").strip().lower()
        entity = m.group("entity").strip().strip(".,!?").lower()
        if not relation or not entity:
            return None
        for prep in _REL_FRONTED_TRAILING_PREPS:
            suffix = " " + prep
            if entity.endswith(suffix) and entity != prep:
                entity = entity[: -len(suffix)].strip()
                break
        if not entity:
            return None
        if relation in self.router.self_aliases or entity in self.router.self_aliases:
            return None                                       # self/identity turn -> the host router's job
        entity_toks = entity.split()
        entity_final = "_".join(entity_toks)                  # this store's own canonical-token convention
        if _knowledge_grounding_enabled():
            composer = getattr(self.inner, "composer", None)
            grounded = _ground_content_words(composer, entity_toks, min_span=1)
            if len(grounded) == 1:
                entity_final = grounded[0]
        return [entity_final, relation]

    def _kb_relation_question_route(self, question):
        """A KB RELATION question whose relation is an UNDERSCORED multi-word Wikidata property token the shipped
        `wikidata_core_15k` core actually uses -- 'what is chelsea fc's country of citizenship?', 'where was X
        born?', 'what political party is X a member of?'. `_relation_fronted_route` (above) deliberately excludes
        this shape (its relation group is a single bare word only); this route covers exactly the complementary
        multi-word-relation surface via the curated `_KB_RELATION_PATTERNS` table (see the module comment above
        it for the mechanism + honesty note). Tries every precompiled pattern in order (idioms first, then the
        two generic shapes, per relation); the first match wins. Returns [entity_final, relation] for a matched
        + resolvable entity, or None (no shape matched, the lesion flag is set, or the entity/relation collides
        with a self-alias) -- falls straight through to the unchanged definitional-copula / generic parse below,
        byte-identical for every other question. COMPREHENSION only, mirroring `_relation_fronted_route` exactly:
        recall stays on the caller's unchanged `what_does()`, so an unknown entity or a relation the store has no
        fact for still honestly abstains -- this can only ADD a resolution, never invent a fact."""
        if not _kb_relation_questions_enabled():
            return None
        q = question.strip()
        for rx, relation in _KB_RELATION_PATTERNS:
            m = rx.match(q)
            if not m:
                continue
            entity = m.group("entity").strip().strip(".,!?").strip().lower()
            if not entity:
                continue
            if relation in self.router.self_aliases or entity in self.router.self_aliases:
                continue
            entity_toks = entity.split()
            entity_final = "_".join(entity_toks)              # this store's own canonical-token convention
            if _knowledge_grounding_enabled():
                composer = getattr(self.inner, "composer", None)
                grounded = _ground_content_words(composer, entity_toks, min_span=1)
                if len(grounded) == 1:
                    entity_final = grounded[0]
            return [entity_final, relation]
        return None

    def _definitional_copula_route(self, question):
        """A DEFINITIONAL copula question -- 'what is X?', "what's a X?", 'who is X?', 'define X' -- asks for X's
        CATEGORY. English lexicalizes that relation as the copula 'is', which the stopword strip removes as a
        function word, leaving ONLY the subject X; so the normal (agent, action) extraction finds no action and the
        turn wrongly abstains. The in-conversation TEACHING convention lexicalizes the same relation as 'isa'
        (e.g. 'canada isa country'); the shipped Wikidata knowledge core lexicalizes it as 'instance_of' instead
        (curation_report.json: instance_of:2734, no 'isa' anywhere) -- KNOWLEDGE GROUNDING (2026-08-26, see the
        module docstring) tries an alias-hop-resolved 'is_a'/'is_an' FIRST (Wikidata's own P31 aliases literally
        list "is a"/"is an", sanitized identically to 'is_a'/'is_an'), falling back to the literal 'isa' so the
        pre-existing in-conversation teaching convention stays BYTE-IDENTICAL when grounding finds nothing (a
        checkout with no alias facts, or the lesion). Returns [subject, relation] or None. This is COMPREHENSION
        only (the same lexical-variant job as the inflection map + the router synonyms) -- the recall itself
        stays on the composer/substrate (`what_does`), so the no-confab MOAT is untouched: an unknown subject or
        relation -> `what_does` returns nothing -> honest abstain. Relational 'what is the capital of X' is NOT
        definitional (it carries a relation word + ' of '), so it is deliberately excluded and left to the
        normal parse."""
        import re as _re
        m = _re.match(r"^\s*(?:what(?:'s|s| is| are)|who(?:'s|s| is| are)|define)\s+"
                      r"(?:an?\s+|the\s+)?(.+?)\s*\??\s*$", question.lower())
        if not m:
            return None
        subj = m.group(1).strip().strip(".,!?").strip()
        # a SINGLE-ENTITY subject only: reject a relational question ('... of ...') or an empty/self subject. A
        # multi-word entity (e.g. 'north america', 'united states') is allowed -- the store keys on the phrase.
        if not subj or " of " in (" %s " % subj):
            return None
        if subj in self.router.self_aliases:
            # SELF/IDENTITY EXTENSION (rank-13 de-risk, BRAIN_NEURAL_SELFID, default OFF -- see the module-level
            # block above _neural_selfid_enabled): 'what are you?' / 'who are you?' fit this EXACT copula shape
            # ('what/who is/are <subject>') with subject = a self-alias. Reuse the SAME 'isa' comprehension-helper
            # recipe already proven for 'what is X?' below instead of ceding the turn outright to the host router
            # -- resolve the self-alias to 'brain' (mirroring `QuestionRouter._resolve_self`) and let
            # `_substrate_recall`'s candidate-relation retry (see there) try the SAME has/have/is/uses/use
            # preference order the host router uses, over the substrate. Flag OFF (default): unchanged, the host
            # router's job (byte-identical).
            if not _neural_selfid_enabled():
                return None
            return ["brain", "isa"]
        # KNOWLEDGE GROUNDING: collapse a multi-word Wikidata-style entity phrase ('chelsea fc') to its
        # canonical underscore token via the alias-hop -- `min_span=1` is safe HERE (unlike the generic
        # `_extract_route` pass) because this route only fires on an explicit 'what is X'/'who is X'/'define X'
        # prefix, a shape ordinary SVO teaching never matches. A PARTIAL grounding (not every word collapsed to
        # ONE token) falls back to the ORIGINAL space-joined subject unchanged -- preserves the pre-existing
        # 'the store keys on the phrase' convention for a conversationally-taught multi-word entity no alias
        # covers, exactly as before this arc. The RELATION itself stays the literal 'isa' HERE (unchanged,
        # PRECEDENCE-SAFE for the in-conversation teaching convention, which a taught 'isa' fact must keep
        # winning) -- the alias-hop-resolved 'instance_of'/'is_a' relation is tried ONLY as a FALLBACK, in
        # `_substrate_recall`, after the literal 'isa' recall has already failed (see there): trying it HERE
        # instead would have the resolved relation shadow an already-correct in-session 'isa' fact whenever the
        # composer also carries the shipped alias facts (the TieredFactStore buffer-then-LTM fall-through means
        # 'instance_of' could reach the LTM and silently outrank a real taught 'isa' answer on the SAME subject).
        subj_final = subj
        if _knowledge_grounding_enabled():
            composer = getattr(self.inner, "composer", None)
            grounded = _ground_content_words(composer, subj.split(), min_span=1)
            if len(grounded) == 1:
                subj_final = grounded[0]
        return [subj_final, "isa"]

    def _extract_route(self, question):
        """COMPREHENSION-ONLY phase of `_substrate_recall`: resolve the routable (agent, action) of a factual SVO query
        WITHOUT recalling the patient (no `what_does`). Returns [a, v] for a routable factual query (the caller — the
        host `if recalled == p`, OR the substrate GNW ignition bus — then authors the recall verdict), None for a
        self/identity/short/unextractable turn (the host router owns it), or the sentinel `"__DECLINE__"` when the
        on-brain parser DECLINES a factual-shaped question (a comprehension abstain). Mirrors `_substrate_recall`'s
        neural-parser-then-heuristic comprehension EXACTLY; it merely STOPS before the recall so the combination verdict
        can be authored by the substrate ignition bus instead of host Python (the scaffold-retirement)."""
        _STOP = {"what", "who", "whom", "does", "do", "did", "is", "are", "was", "were", "the", "a", "an",
                 "to", "it", "that", "this", "they", "them", "of", "about"}
        toks = [t.lower().strip(".,!?") for t in question.split()]
        content = [t for t in toks if t and t not in _STOP]
        # KNOWLEDGE GROUNDING (2026-08-26, see module docstring): collapse a multi-word Wikidata-style entity
        # phrase ('chelsea fc') into its canonical underscore token BEFORE the copula-length check + the (agent,
        # action) parse below, so e.g. 'what is chelsea fc' (2 content words purely from the entity's own name)
        # correctly reaches the copula branch instead of being mis-split into a bogus (agent='chelsea',
        # action='fc'). `min_span=2` + `known_words` guard an in-session word already established this
        # conversation is NEVER re-ground -- see `_ground_content_words`'s docstring for the collision-risk
        # reasoning. A grounding miss (no alias covers the phrase, or the composer carries none) leaves
        # `content` UNCHANGED -- byte-identical for every previously-routable question.
        if _knowledge_grounding_enabled():
            composer = getattr(self.inner, "composer", None)
            known = self.agents_set | self.actions_set | self.patients_set
            content = _ground_content_words(composer, content, known_words=known)
        # RELATION-FRONTED question ('what country is chelsea fc from?', Vikunja #142) -> fires BEFORE the generic
        # (agent, action) = (content[0], content[1]) positional parse below, which ASSUMES SVO word order (entity
        # first, relation/verb second, as in the in-conversation teaching shape 'what does the wolf hunt?') and
        # mis-assigns the RELATION noun to the agent slot when a Wikidata-style question instead fronts the
        # relation before the copula (see `_relation_fronted_route`'s docstring for the traced mechanism). Runs on
        # the RAW `question` (its own regex does its own tokenizing), independent of the `content` list above, so
        # it cannot be perturbed by -- or perturb -- the generic parse. A non-match / disabled flag returns None
        # -> falls straight through to the unchanged logic below, byte-identical for every other question.
        _relf = self._relation_fronted_route(question)
        if _relf is not None:
            return _relf
        # KB RELATION question ('what is X's country of citizenship?', 'where was X born?') -> a real
        # wikidata_core_15k UNDERSCORED multi-word relation `_relation_fronted_route` cannot reach (see
        # `_kb_relation_question_route`'s docstring). Runs on the RAW `question` (its own patterns do their own
        # tokenizing), same convention as the relation-fronted check above. A non-match / disabled flag returns
        # None -> falls straight through to the unchanged logic below, byte-identical for every other question.
        _kbrel = self._kb_relation_question_route(question)
        if _kbrel is not None:
            return _kbrel
        # DEFINITIONAL COPULA question ('what is X?') -> the instance-of relation 'isa'. Fires ONLY when the copula
        # strip left <=1 content word (i.e. the normal (agent, action) parse has NO verb to work with), so a question
        # that already carries two content words is untouched -> byte-identical for every previously-routable query.
        # The subject must not be a self-alias (identity questions stay the host router's job).
        if len(content) <= 1:
            _defo = self._definitional_copula_route(question)
            if _defo is not None:
                return _defo
        # CHOOSE (#1): the on-brain parser OWNS a factual-SVO-shaped question (>=2 content words, none a self-alias).
        # When it comprehends -> (agent, action) on FIRING neurons; when it DECLINES on such a question -> "__DECLINE__"
        # (do NOT fall to the host router's role-blind keyword confab). This makes the comprehension genuinely on the
        # substrate + LESION-LOAD-BEARING: lesion the parser -> role_of returns junk -> the factual CHOOSE abstains. A
        # self/identity/short question (or the rf escape — NO parser) keeps the host heuristic (prefer a KNOWN
        # agent/action, else STRUCTURAL position) + the router fallback in gate().
        # SELF-REFERENTIAL FACTUAL SVO EXTENSION (rank-13 de-risk, BRAIN_NEURAL_SELFID, default OFF): a question
        # like 'what do you eat?' carries a genuine 2nd content word (the action) alongside the self-reference --
        # the SAME shape as any other factual-SVO query, just with the agent self-named. Resolve a self-alias
        # token to 'brain' (mirroring `QuestionRouter._resolve_self`, the host router's OWN resolution) BEFORE the
        # has_self_alias gate below, so this reaches the SAME on-brain `_neural_question_parse` -- and, via the
        # installed GNW bus, the SAME substrate-ignition combiner -- every other factual-SVO question already
        # uses. Not a new mechanism: the identical recipe with 'brain' as a known agent. Flag OFF (default):
        # `content` is untouched -- byte-identical (a bare self-alias keeps blocking the parser, as today).
        if _neural_selfid_enabled():
            content = [self.router._resolve_self(t) for t in content]
        has_self_alias = any(t in self.router.self_aliases for t in content)
        parser_present = getattr(getattr(self.inner, "composer", None), "parser", None) is not None
        if parser_present and len(content) >= 2 and not has_self_alias:
            nq = self._neural_question_parse(content)
            if nq is None:
                return "__DECLINE__"            # factual-shaped question the on-brain parser could not comprehend
            a, v = nq
        else:
            a = next((t for t in content if t in self.agents_set), None) or (content[0] if content else None)
            v = next((t for t in content if t in self.actions_set), None) or (content[1] if len(content) > 1 else None)
        if not (a and v) or a == v:
            return None                          # could not extract a query -> let the host router try (self/identity)
        # a self/identity query (a or v is a self-alias) is the host router's job, not the substrate's.
        if a in self.router.self_aliases or v in self.router.self_aliases:
            return None
        return [a, v]

    def _substrate_recall(self, question):
        """IN-LOOP LEARNING recall: resolve (agent, action) from the question and recall the patient FROM THE SPIKING
        SUBSTRATE (`inner.what_does`), so a fact heard this conversation is answerable even though it is not in the
        build-time host snapshot. Returns [a, v, p], None, or the "__ABSTAIN__" sentinel. No confabulation: `what_does`
        returns nothing unless the binding is genuinely stored. The (agent, action) COMPREHENSION is factored into
        `_extract_route` (NEURAL BridgeParser on the onebrain default, host heuristic on the rf escape); this method
        adds the host recall verdict on top of it. BYTE-IDENTICAL to the pre-factor code (the production lesion probe
        patches this method, so gate() must keep calling it)."""
        route = self._extract_route(question)
        if route == "__DECLINE__":
            return "__ABSTAIN__"                # a factual-shaped question the on-brain parser could not comprehend
        if route is None:
            return None                          # could not extract a query -> let the host router try (self/identity)
        a, v = route
        try:
            p = self.inner.what_does(a, v)
            # VERB LEMMATIZATION AT QUERY (reasoning-frontier, 2026-08-25): SURFACE-FIRST, LEMMA-FALLBACK -- try the
            # question's literal action first (byte-identical whenever it already matches), and ONLY on a miss retry
            # with the canonicalized lemma (`research/runners/lexical_lemma.lemma_verb`), so "what does the wolf
            # hunt?" recalls a fact taught as "the wolf hunts the deer" (both canonicalize to "hunt" at store-write
            # in `_maybe_acquire`, above). Mirrors the SAME surface-first/lemma-fallback convention the B3 non-
            # contradiction organ already uses for its own recall (`_action_lemma_candidates`).
            if not p:
                v_lemma = lemma_verb(v)
                if v_lemma != v:
                    p_lemma = self.inner.what_does(a, v_lemma)
                    if p_lemma:
                        p, v = p_lemma, v_lemma      # report the canonical action that actually matched
            # KNOWLEDGE GROUNDING (2026-08-26): RELATION alias-hop, ONE MORE fallback candidate, tried ONLY
            # after BOTH the surface form and the lemma have already missed -- mirrors this loop's own surface-
            # first/lemma-fallback convention. Covers a single-word relation SYNONYM the copula/content-
            # grounding passes deliberately do not touch eagerly (e.g. 'nationality' -> the shipped core's
            # 'country', or the copula's literal 'isa' -> the Wikidata core's 'instance_of' -- see
            # `_definitional_copula_route`'s own comment on why that substitution is precedence-unsafe to make
            # THERE). Being a MISS-ONLY fallback (never tried while `p` is already truthy) makes it structurally
            # unable to override an already-correct recall -- it can only rescue an otherwise-abstaining turn.
            if not p and _knowledge_grounding_enabled():
                composer = getattr(self.inner, "composer", None)
                v_candidates = [_alias_hop(composer, v)]
                if v_lemma != v:
                    v_candidates.append(_alias_hop(composer, v_lemma))
                if v == "isa":
                    # The in-conversation TEACHING convention's copula token 'isa' is this codebase's OWN
                    # invented shorthand -- it is not itself a raw Wikidata alias, so alias-hopping the literal
                    # string 'isa' above never resolves. Bridge it explicitly via 'is_a'/'is_an', which ARE
                    # genuine raw Wikidata P31 aliases (confirmed in wikidata5m_relation.txt) and so DO resolve
                    # to 'instance_of' through the same alias-hop primitive -- closing the exact gap
                    # `_definitional_copula_route` names in its own comment (byte-identical for the checkout with
                    # no alias facts, since `_alias_hop` returns None there for every candidate).
                    v_candidates.append(_alias_hop(composer, "is_a"))
                    v_candidates.append(_alias_hop(composer, "is_an"))
                for v_cand in v_candidates:
                    if v_cand and v_cand not in (v, v_lemma):
                        p_alias = self.inner.what_does(a, v_cand)
                        if p_alias:
                            p, v = p_alias, v_cand
                            break
                # LAST-RESORT AGENT alias-hop: only reached when the relation-side fallbacks above ALSO missed.
                # Resolves a single-word entity alias (e.g. 'usa' -> 'u_s_of_a') that the `_extract_route`
                # content-grounding pass deliberately skips for single words (collision-risk, see that pass's
                # docstring) -- safe HERE because it only fires on an otherwise-already-abstaining query, so it
                # can only rescue, never shadow, a working recall.
                if not p:
                    a_alias = _alias_hop(composer, a)
                    if a_alias and a_alias != a:
                        p_alias = self.inner.what_does(a_alias, v)
                        if p_alias:
                            p, a = p_alias, a_alias
            # SELF/IDENTITY candidate-relation retry (rank-13 de-risk, BRAIN_NEURAL_SELFID, default OFF): reached
            # ONLY for a bare identity query on the self ('what are you?' -> `_definitional_copula_route`'s
            # self-branch above returns ['brain', 'isa']; a literal 'what is brain?'/'what is the brain?' reaches
            # the SAME ('brain', 'isa') pair through the ordinary copula path with no flag involved). Scoped
            # tightly to agent=='brain' so it can NEVER fire for any other entity's 'isa'/definitional miss.
            # MISS-ONLY (never runs while `p` is already truthy -- can only rescue, never override, exactly the
            # same shape as the alias-hop v_candidates loop above): try the HOST router's OWN defining-relation
            # preference order (`QuestionRouter.match_fact`'s is_identity_q branch: has/have/is/uses/use, first
            # match wins) against `what_does('brain', ·)` -- reproducing the host's answer through the substrate
            # recall instead of the router's bag-of-words scan. This is a comprehension-helper retry (matching the
            # existing v_candidates convention), not a claim of a neural BridgeParser parse for this bare-identity
            # shape -- see the module-level comment above `_neural_selfid_enabled`.
            if not p and a == "brain" and _neural_selfid_enabled():
                for v_cand in ("has", "have", "is", "uses", "use"):
                    if v_cand in (v, v_lemma):
                        continue
                    p_cand = self.inner.what_does(a, v_cand)
                    if p_cand:
                        p, v = p_cand, v_cand
                        break
        except Exception:
            return None
        if not p:
            # a WELL-FORMED question the substrate cannot answer -> ABSTAIN honestly. Do NOT fall through to the host
            # router's role-blind keyword guess (that is the confabulation the CHOOSE gap produced, e.g. "what does fish
            # fly?" -> "cat eat fish"). This retires the host router's CONFAB for well-formed queries.
            return "__ABSTAIN__"
        if isinstance(p, str) and p in self.agents_set:
            self._note_referent(p)
        return [a, v, p]

    def _note_referent(self, word):
        """Write a referent into the discourse WM (multi-turn), so a later pronoun resolves to it."""
        if self.is_multiturn and isinstance(word, str):
            try:
                self.agent._write_referent(word)
            except Exception:
                pass

    # --- the CONSTRAIN + VERIFY render of a gated fact into fluent prose ---
    def render(self, gate_svo):
        """Render the gated SVO into a fluent sentence (CONSTRAIN) and VERIFY the content re-parses to the gated
        fact. Returns the verified fluent string, or the brain's raw triple on a verify miss / raw mode / no
        renderer. NEVER emits unverified generative prose as the answer."""
        # OPEN-ENDED GENERATION (#3E): a moat-verified HYPOTHESIS is rendered as an EXPLICIT, clearly-FLAGGED guess
        # (the honesty boundary is a deliverable). We now speak it as FLUENT prose via the mouth — SVO-VERIFIED so
        # the mouth cannot swap the content — but framed 'Maybe ... -- that's a guess ...' so it can never be
        # mistaken for asserted knowledge; a mouth-verify miss / GPU-free host falls back to the raw flagged
        # template. The proposal was already gated (plausible + non-contradictory) + moat-verified in gate().
        if isinstance(gate_svo, HypothesisSVO):
            return self.render_hypothesis(gate_svo)
        a, v, p = gate_svo
        if self.raw_mode or self.renderer is None:
            return self._raw(gate_svo)
        # BRAIN-NATIVE SPIKING MOUTH (recall surface): a bounded transitive-SVO recall renders ON SPIKES (word order
        # = the per-pool spiking-RATE ranking), verify-gated. Flag BRAIN_SPIKING_MOUTH_RECALL (default OFF) / open
        # prose / a verify miss -> None -> falls straight through to the Qwen/template path below (BYTE-IDENTICAL).
        spk = self.spiking_recall_surface(a, v, p)
        if spk is not None:
            return spk
        surface, asserted = self.renderer.render_svo(a, v, p)
        if self._verify(surface, asserted, gate_svo):
            return surface
        # a generative renderer can DRIFT: try a tighter re-prompt once (if supported), else speak the raw fact
        if hasattr(self.renderer, "render_svo_regen"):
            surface2, asserted2 = self.renderer.render_svo_regen(a, v, p)
            if self._verify(surface2, asserted2, gate_svo):
                return surface2
        return self._raw(gate_svo) + "   [unverified render -> spoke the brain's raw fact]"

    def _verify(self, surface, asserted, gate_svo):
        """VERIFY: re-parse the rendered content back into an SVO and require it to MATCH the gated fact. For the
        stub, `asserted` is the canonical content SVO; for Qwen, `asserted` is None -> re-parse the PROSE."""
        if asserted is None:
            from research.runners._grounded_lang_integration_derisk import _extract_svo_from_prose
            asserted = _extract_svo_from_prose(surface, self.agents_set, self.actions_set,
                                               self.patients_set, self.inflect)
            if asserted is None:
                return False
        parsed = self.inner.parse(asserted, voice="active")
        rsvo = [parsed.get("agent"), parsed.get("action"), parsed.get("patient")]
        return rsvo == list(gate_svo)

    # --- the CLAIM-LEVEL moat generalization (de-risked ClaimEntailmentVerifier, wired for the multi-fact turn) ---
    @staticmethod
    def _claim_moat_enabled():
        """The escape hatch. `BRAIN_CLAIM_MOAT=0` reverts to the exact single-triple `_verify` per rendered
        sentence (the pre-generalization behaviour). Any other value (incl. unset) keeps the claim-level moat on
        -- the production default, so genuinely free-form MULTI-CLAUSE grounded prose survives the moat."""
        return os.environ.get("BRAIN_CLAIM_MOAT", "1") != "0"

    def _build_claim_verifier(self, gated_facts):
        """Build (and lazily cache on the gated SET) the de-risked ClaimEntailmentVerifier, REUSED BY IMPORT --
        NOT reimplemented. It decomposes multi-clause prose into its asserted proposition set, role-parses EACH
        clause on THIS brain's on-substrate parser (`self.inner.parse`, the same spiking role parser the
        single-triple `_verify` uses), and accepts IFF every asserted proposition is entailed by `gated_facts`
        (with the flagged-hypothesis carve-out + the coverage invariant). Returns None when the set is empty or
        has a role-permutation collision (the verifier's own well-formedness guard) -> the caller falls back to
        the single-triple `_verify` in the SAFE direction."""
        key = frozenset(tuple(f) for f in gated_facts
                        if isinstance(f, (list, tuple)) and len(f) == 3
                        and all(isinstance(x, str) for x in f))
        if not key:
            return None
        cache = getattr(self, "_claim_verifier_cache", None)
        if cache is None:
            cache = self._claim_verifier_cache = {}
        if key in cache:
            return cache[key]
        from research.runners._moat_claim_entailment_derisk import (
            ClaimEntailmentVerifier, VERB_SYNONYMS, _build_inflection_map)
        gated = [list(f) for f in key]
        nouns = {t for f in gated for t in (f[0], f[2])}
        verbs = {f[1] for f in gated}
        inflect = _build_inflection_map(sorted(verbs))
        try:
            ver = ClaimEntailmentVerifier(self.inner, gated, nouns, verbs, VERB_SYNONYMS, inflect)
        except AssertionError:
            ver = None                                   # role-permutation collision -> fall back (SAFE)
        cache[key] = ver
        return ver

    def _verify_claim_set(self, surface, gated_facts):
        """CLAIM-LEVEL VERIFY: does the rendered PROSE `surface` assert ONLY facts entailed by `gated_facts` (the
        set the turn gathered)? Returns (accepted: bool, result: dict) via the de-risked ClaimEntailmentVerifier,
        or (None, None) when the claim moat is DISABLED (escape flag) or the verifier is unbuildable -> the caller
        must fall back to the single-triple `_verify` (byte-identical old behaviour). This is a strict SUPERSET of
        `_verify`: a single grounded sentence still passes, AND multi-clause grounded prose passes, while any
        response carrying even one ungrounded/contradictory asserted clause is rejected (0 confab leaks)."""
        if not self._claim_moat_enabled():
            return None, None
        ver = self._build_claim_verifier(gated_facts)
        if ver is None:
            return None, None
        res = ver.verify(surface)
        return bool(res["accepted"]), res

    def _raw(self, gate_svo):
        """The brain's OWN renderer: the raw recalled triple as a plain sentence (no LLM)."""
        return " ".join(str(x) for x in gate_svo)

    # --- OPEN-ENDED GENERATION (#3E): render a generated HYPOTHESIS as a FLUENT, clearly-FLAGGED guess ---
    def render_hypothesis(self, hyp):
        """Render a GENERATED, moat-verified HYPOTHESIS (a #3E novel proposition) as a clearly-FLAGGED guess.
        Prefer FLUENT prose via the mouth, framed 'Maybe <fluent> -- that's a guess ...', VERIFYING that the
        fluent sentence re-parses to the SAME (a, v, p) the hypothesis asserts (so the mouth cannot swap the
        content). On a verify miss / no renderer / raw mode, fall back to the raw FLAGGED template 'perhaps a v
        p'. The guess is NEVER surfaced as an asserted fact -- the honesty framing is explicit in the surface text
        either way."""
        return self.render_hypothesis_verified(hyp)[0]

    def render_hypothesis_verified(self, hyp):
        """As `render_hypothesis`, but also report whether the FLUENT surface VERIFIED (True) or the raw flagged
        template FALLBACK was used (False). Returns (surface, fluent_verified). The VERIFY is the same re-parse the
        recall path uses: the fluent sentence must carry the hypothesis's exact (a, v, p).

        SURFACE ORDER OF PREFERENCE (production default): the BRAIN-NATIVE SPIKING BROCA mouth renders a supported
        structured hypothesis (a transitive SVO) grammatically ON FIRING NEURONS -- word order = the per-pool
        spiking-RATE ranking on a real Izhikevich SimulationBridge (EMERGE-59/61, composed with the #3E draw in
        `_spiking_fluent_surface_derisk`, 6-seed GO) -- transformer-FREE, replacing the agrammatic host f-string.
        It is re-parse VERIFIED (the moat) so it recovers the DRAWN SVO; a verify miss falls back to the raw flagged
        template (NEVER a leak). The escape flag `BRAIN_SPIKING_MOUTH=0`, OR content the spiking Broca can't frame
        (open/multi-word prose), falls through to the pre-spiking mouth (off-bridge Qwen / template-stub / raw
        flagged template) -- the documented A1 residual (open arbitrary prose = the banked deep-context wall)."""
        a, v, p = hyp
        template = self._hypothesis_template(a, v, p)
        # (1) BRAIN-NATIVE SPIKING MOUTH -- the production default for a structured (transitive SVO) hypothesis.
        if self._spiking_mouth_enabled() and self._hyp_frame_supported(hyp):
            spk = self._render_hypothesis_spiking(hyp)
            if spk is not None:
                return spk, True                      # grammatical, moat-verified, flagged -- produced on spikes
            return template, False                    # spiking verify miss -> honest flagged fallback (NO leak)
        # (2) the PRE-SPIKING mouth (escape flag BRAIN_SPIKING_MOUTH=0, or content the spiking Broca can't frame).
        if self.raw_mode or self.renderer is None:
            return template, False                    # GPU-free / --raw: the honest raw flagged guess
        surface, asserted = self.renderer.render_svo(a, v, p)
        if self._verify(surface, asserted, hyp):
            return self._frame_guess(surface), True
        # a generative mouth can DRIFT: one tighter re-prompt (if supported), else the raw flagged template
        if hasattr(self.renderer, "render_svo_regen"):
            surface2, asserted2 = self.renderer.render_svo_regen(a, v, p)
            if self._verify(surface2, asserted2, hyp):
                return self._frame_guess(surface2), True
        return template, False                        # mouth swapped/garbled the content -> honest flagged fallback

    @staticmethod
    def _hypothesis_template(a, v, p):
        """The raw FLAGGED-guess surface (GPU-free fallback / a mouth-verify miss). Byte-identical to the pre-fluent
        template so the moat framing is unchanged when the mouth is unavailable."""
        return f"perhaps {a} {v} {p}  [a guess from what I've learned -- not something I was taught]"

    @staticmethod
    def _frame_guess(surface):
        """Frame a VERIFIED fluent sentence as an EXPLICIT guess (the honesty boundary is a deliverable): the
        fluent content is kept verbatim, lower-cased into a 'Maybe ...' lead so it can never be read as asserted
        knowledge, with the not-taught disclaimer appended."""
        g = surface.strip().rstrip(".")
        if g[:1].isupper():
            g = g[0].lower() + g[1:]
        return f"Maybe {g} -- that's a guess from what I've learned, not something I was taught."

    # --- BRAIN-NATIVE SPIKING BROCA mouth for the GENERATE channel (#3E surface; REUSE-BY-IMPORT, NO sim/ edit) ---
    @staticmethod
    def _spiking_mouth_enabled():
        """Escape hatch. `BRAIN_SPIKING_MOUTH=0` reverts the GENERATE-channel hypothesis SURFACE to the pre-spiking
        mouth (off-bridge Qwen / template-stub / raw flagged template) -- byte-identical to the pre-wire behaviour.
        Any other value (incl. unset) keeps the brain-native SPIKING Broca render ON -- the production default."""
        return os.environ.get("BRAIN_SPIKING_MOUTH", "1") != "0"

    @staticmethod
    def _hyp_frame_supported(hyp):
        """True iff the hypothesis fits a structured frame the spiking BROCA supports (a transitive SVO with single-
        WORD alphabetic roles, subject != object). Open/arbitrary content (empty / multi-word roles) is NOT frameable
        here -> the caller falls back to the current mouth (the documented A1 residual = open arbitrary prose)."""
        if not isinstance(hyp, (list, tuple)) or len(hyp) != 3:
            return False
        a, v, p = hyp
        return all(isinstance(x, str) and x.isalpha() for x in (a, v, p)) and a != p

    def _spiking_broca_producer(self):
        """Lazily build + cache the reused spiking BROCA clause producer (EMERGE-59/61 order read-out on a real
        Izhikevich SimulationBridge). Built ONCE (bridge build + competitive-queuing learn of the 6-slot hedged-
        transitive order, ~0.35 s CPU); each hypothesis then emits in ~5 ms via the EMERGE-61 inter-utterance
        wash-out (the producer's `emit` restores the post-init substrate state before every clause). REUSE-BY-IMPORT
        from `_spiking_fluent_surface_derisk` -- NO reimplementation, NO sim/ edit."""
        prod = getattr(self, "_spk_producer", None)
        if prod is None:
            from research.runners._spiking_fluent_surface_derisk import SpikingClauseProducer, HEDGED_TRANSITIVE
            seed = int(getattr(self.inner, "seed", 42))
            prod = SpikingClauseProducer(seed)
            prod.learn(len(HEDGED_TRANSITIVE))         # competitive-queuing learn of the hedged-transitive slot order
            self._spk_producer = prod
        return prod

    def _render_hypothesis_spiking(self, hyp):
        """Render a GENERATED hypothesis SVO grammatically ON FIRING NEURONS (the composed spiking BROCA render:
        'perhaps the <S> <V-3sg> the <O>', word order = the per-pool spiking-RATE ranking on the SimulationBridge),
        then re-parse VERIFY (the SAME moat the recall path uses -> `_verify` re-parses the surface PROSE) that the
        rendered sentence recovers the DRAWN (a, v, p). Returns the framed FLAGGED guess on a verify PASS, or None on
        a verify miss (-> the caller uses the raw flagged template; NEVER a leak). Transformer-FREE: this path never
        touches the Qwen mouth."""
        from research.runners._spiking_fluent_surface_derisk import HEDGED_TRANSITIVE
        from research.runners._emerge57_ra_refinetune_emerge_frames_derisk import emerge_v3
        a, v, p = hyp
        dctx = {"subject": a, "verb_3sg": emerge_v3(v), "object": p}
        surface = " ".join(self._spiking_broca_producer().emit(HEDGED_TRANSITIVE, dctx))
        if self._verify(surface, None, hyp):           # the moat: the spiking sentence must recover THIS (a, v, p)
            return self._frame_guess_spiking(surface)
        return None                                    # verify miss -> caller falls back to the flagged template

    @staticmethod
    def _frame_guess_spiking(surface):
        """Frame the SPIKING-Broca hedged surface ('perhaps the S V-3sg the O') as an EXPLICIT flagged guess. The
        surface already leads with the epistemic hedge 'perhaps' (the spiking Broca's own CONN slot); we append the
        SAME not-taught disclaimer the raw template uses, so the honesty framing is identical whichever mouth spoke."""
        return f"{surface.strip()}  [a guess from what I've learned -- not something I was taught]"

    # --- BRAIN-NATIVE SPIKING BROCA mouth for the RECALL / RICH answer surface (ASSERTED SVO; REUSE-BY-IMPORT) ---
    def _spiking_recall_mouth(self):
        """Lazily build + cache the reused spiking BROCA clause producer for the ASSERTED recall surface (EMERGE-59/61
        order read-out on a real Izhikevich SimulationBridge, the 5-slot PLAIN_TRANSITIVE frame). Built ONCE (~0.35 s
        CPU); each recall SVO then emits in ~5 ms via the EMERGE-61 wash-out. REUSE-BY-IMPORT (spiking_mouth_recall_prod),
        NO reimplementation, NO sim/ edit."""
        mouth = getattr(self, "_spk_recall_mouth", None)
        if mouth is None:
            from research.runners.spiking_mouth_recall_prod import SpikingRecallMouth
            mouth = SpikingRecallMouth(seed=int(getattr(self.inner, "seed", 42)))
            self._spk_recall_mouth = mouth
        return mouth

    def spiking_recall_surface(self, a, v, p):
        """Render a GROUNDED recalled SVO grammatically ON FIRING NEURONS (the spiking BROCA render 'the <S> <V-3sg>
        the <O>', word order = the per-pool spiking-RATE ranking on the SimulationBridge), then re-parse VERIFY (the
        SAME moat the Qwen recall path uses -> `_verify` re-parses the surface PROSE) that the rendered sentence
        recovers the recalled (a, v, p). Returns the verified spiking surface on a PASS, or None on a verify miss /
        raw mode / no renderer / unsupported frame -> the caller falls back to the current mouth (BYTE-IDENTICAL).
        Transformer-FREE: this path never touches the Qwen mouth.

        Gated on `spiking_mouth_recall_prod.recall_mouth_enabled()` (flag BRAIN_SPIKING_MOUTH_RECALL, default OFF) +
        `frame_supported` (bounded transitive SVO). OFF or unsupported -> None -> the surface is byte-identical to the
        pre-wire Qwen/template path.

        MOOD-TONE COUPLING (2026-08-27 fix): once verified, `_apply_mouth_mood_tone` may append a spiking-read tone
        marker so the Gate-B mood signal stays load-bearing on THIS mouth too (see `spiking_mouth_recall_prod`'s
        module docstring). A neutral / absent mood is a no-op -> byte-identical to the pre-fix surface."""
        from research.runners.spiking_mouth_recall_prod import recall_mouth_enabled, frame_supported
        if self.raw_mode or self.renderer is None:
            return None                                   # GPU-free / --raw: keep the brain's raw triple path
        if not recall_mouth_enabled() or not frame_supported(a, v, p):
            return None                                   # flag OFF / open prose -> byte-identical fallback
        surface = self._spiking_recall_mouth().render(a, v, p)
        if self._verify(surface, None, [a, v, p]):        # the moat: the spiking sentence must recover THIS (a, v, p)
            return self._apply_mouth_mood_tone(surface)
        return None                                       # verify miss -> caller falls back (NEVER a leak)

    def _apply_mouth_mood_tone(self, surface):
        """LOAD-BEARING mood coupling on the spiking mouth's OWN surface (2026-08-27 fix). Closes a cross-faculty
        regression: the Gate-B affect faculty (`affect_production_organ.MoodConditionedRenderer`, default-ON since
        2026-08-12) colors PROSE only inside `self.renderer.render_svo` -- a call `spiking_recall_surface` pre-empts
        whenever it returns a verified surface, so once BRAIN_SPIKING_MOUTH_RECALL flipped default-ON (2026-08-26)
        the manner-coloring silently went dead for every bounded-transitive recall turn. `_mood_tone_level` is an
        additive per-turn attribute set ONLY by webapp.server's Gate-B affect block (the SAME `tone_level()` the
        Qwen path already colors with); every other caller (the TUI, unit tests, a disabled BRAIN_AFFECT) never
        sets it, so `getattr(..., 0)` defaults to neutral -> `surface` returned UNCHANGED (byte-identical). A
        nonzero level drives `spiking_mouth_recall_prod.mouth_tone_marker` -- a genuine spiking rate-vs-rate read,
        not a host if/else on the mood float -- which appends '!' (warm) / '.' (curt) / '' (undecided, safe no-op).
        Escape: `BRAIN_SPIKING_MOUTH_MOOD=0` reverts to mood-BLIND (today's regressed surface) even with the recall
        mouth itself ON."""
        level = int(getattr(self, "_mood_tone_level", 0) or 0)
        if level == 0:
            return surface                                # neutral / never set -> byte-identical, no spiking read
        from research.runners.spiking_mouth_recall_prod import mouth_mood_enabled, mouth_tone_marker
        if not mouth_mood_enabled():
            return surface                                # lesion escape -> mood-BLIND, byte-identical
        marker = mouth_tone_marker(level, seed=int(getattr(self.inner, "seed", 42)))
        return surface + marker if marker else surface

    # --- discourse event tracking (who is doing it now / who was doing it before) ---
    def _discourse_turn(self, line):
        """Route a discourse turn if the agent carries an event register: hear an SVO clause (updating the running
        event), or answer 'who was doing it before/now?'. Returns (answer, abstained) or None (not a discourse turn)."""
        if not self.has_event_register:
            return None
        reg = self.agent._event_register
        ql = line.lower().strip().rstrip(".!?").strip()
        toks = ql.split()
        if ql in ("who was doing it before", "who was before", "who did it before", "who was doing that before"):
            if not self._boundary_seen:      # no earlier event yet -> the no-confab moat abstains
                return ("I don't know who was doing it before -- no earlier event yet.", True)
            a = self.agent.who_agent_before()
            return (f"{a} was.", False) if a else ("I don't know who was doing it before.", True)
        if ql in ("who is doing it now", "who is doing it", "who is now", "who did it now"):
            if not self._heard_any_clause:
                return ("I don't know who is doing it now -- nothing said yet.", True)
            a = self.agent.who_agent_now()
            return (f"{a} is.", False) if a else ("I don't know who is doing it now.", True)
        # a discourse SVO clause (optionally with a leading connective): subject action object, where the action is
        # a known verb OR the subject is a pronoun the register tracks -> HEAR it (fold into the running event).
        w = list(toks)
        had_connective = bool(w) and w[0] in ("then", "but", "meanwhile", "and")
        if had_connective:
            w = w[1:]
        if len(w) == 3 and (w[1] in self.actions_set or w[1] == "chase"):
            self.agent.hear(line.rstrip(".!?"))
            self._heard_any_clause = True
            if had_connective:                       # a connective marks a discourse boundary (an earlier event now exists)
                self._boundary_seen = True
            now = self.agent.who_agent_now()
            return (f"ok -- now {now} is doing it." if now else "ok, i heard that.", False)
        return None

    # --- the full turn ---
    def answer(self, question):
        """One conversational turn: DISCOURSE (event tracking) -> GATE (recall + abstain) -> CONSTRAIN+VERIFY render.
        Returns (answer_string, abstained_bool)."""
        disc = self._discourse_turn(question)
        if disc is not None:
            return disc
        gen = self._maybe_generate(question)     # GENERATION (TUI/answer path): multi-fact associative topic reply
        if gen is not None:
            return gen
        gate_svo = self.gate(question)           # gate() now also handles ACQUISITION (assertions) -> reaches the webapp
        if gate_svo is None:
            return "I don't know about that.", True
        return self.render(gate_svo), False

    def list_facts(self):
        """The brain's stored facts (for /facts)."""
        self._refresh_facts()
        return list(self.stored_facts)


# ============================================================================================================
# Loading a developed brain from the various sources.
# ============================================================================================================

def _load_self_knowledge(codes_path, curriculum_path, seed, use_multiturn, enable_neural_render):
    """Reconstruct the self-knowledge brain: build a BrainConversationalAgent/MultiTurnAgent on the saved learned
    grounded codes + teach the curriculum facts. Returns (agent, self_aliases, n_facts)."""
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    with open(os.path.abspath(curriculum_path), "r", encoding="utf-8") as fh:
        cur = json.load(fh)
    # the full taught fact set as SVO (facts + attribute_facts as (noun, 'is', adj))
    facts = [tuple(f) for f in cur.get("facts", [])]
    facts += [(noun, "is", adj) for noun, adj in cur.get("attribute_facts", [])]
    # vocab: the concept set + general-knowledge + untaught fall-backs (so the moat abstains STRUCTURALLY)
    vocab = set(["is"])
    for a, v, p in facts:
        vocab.update([a, v, p])
    vocab |= {"france", "paris", "two", "plus", "four", "romeo", "juliet", "wrote", "shakespeare",
              "color", "blue", "legs", "has", "many"}
    for probe in cur.get("deliberately_untaught_project_facts", {}).get("probes", []):
        for w in probe:
            if isinstance(w, str) and w != "?":
                vocab.add(w)
    vocab = sorted(vocab)
    grounded = None
    if codes_path and os.path.exists(codes_path):
        with open(codes_path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        grounded = {w: np.asarray(v, dtype=float) for w, v in blob.get("grounded_codes", {}).items()}
        print(f"[tui] loaded {len(grounded)} developed grounded codes from "
              f"{os.path.relpath(codes_path, _REPO)}", flush=True)
    else:
        print("[tui] no developed codes file found -- the brain answers the taught facts on its own seed codes "
              "(run _self_knowledge_demo to develop + save the learned codes).", flush=True)
    concepts = {w: None for w in vocab}
    if use_multiturn:
        from research.runners.multi_turn_agent import MultiTurnAgent
        actions = {v for _a, v, _p in facts} | {"is"}
        referents = [w for w in vocab if w not in actions]
        # size the WM loop to hold every referent (2x headroom) so a large vocabulary does NOT overrun the
        # pattern budget (the SpikingLoopContextBuffer holds n/pattern_size patterns) -- same rule as
        # _longitudinal_develop_loop.build_agent.
        pattern_size = 40
        wm_n = max(600, 2 * pattern_size * max(1, len(referents)))
        # DISCOURSE EVENT REGISTER (2026-07-10): the running FACTORED (agent, patient) event so the real developed
        # brain can also answer "who was doing it before?" across a connective. Built on up to 6 of the brain's own
        # referents (the D3 arc's validated K=6 scale; a larger register is best-effort). numpy (spiking=False).
        ev_reg = None
        try:
            import research.runners.d3_discourse_event_register_production_organ as _DR
            reg_refs = referents[:6] if len(referents) >= 6 else (referents + ["dog", "cat", "fish", "bird", "worm", "ball"])[:6]
            # D3 discourse-event register (F2 wire, 2026-08-13): default-ON -> the validated genuinely-SPIKING twin
            # (who-was-before read off cp_firing_states). BRAIN_DISCOURSE_REGISTER=0 -> today's spiking=False rate
            # register (byte-identical). BRAIN_DISCOURSE_REGISTER_LESION=1 -> the prev-slot-silence spiking register
            # (load-bearing). ~76s one-time build at agent load (speed secondary).
            ev_reg = _DR.make_discourse_register(reg_refs, seed=seed)
        except Exception as _e:
            print(f"[tui] discourse event register unavailable ({_e!r}); who-was-before disabled.", flush=True)
        # defer_planner=True: the persistent discourse WM loop is built lazily on the first multi-turn referent
        # (the curriculum teach below uses BrainConversationalAgent.hear, which does NOT write WM referents, so a
        # loaded self-knowledge brain never pays the ~681s WM build at load -- only when a console turn actually
        # introduces a pronoun antecedent). Byte-identical otherwise.
        # --- SELECTIVE ATTENTION / biased-competition wire-in (env-gated, default OFF = byte-identical) ---------
        # BRAIN_BIASED_COMPETITION unset/"0" -> biased_competition_enabled() == False == the literal passed today.
        # ON -> a bare pronoun over >=2 held referents resolves to the content-favored referent (Wong-Wang WTA).
        from research.runners.biased_competition_prod import biased_competition_enabled as _bc_enabled
        agent = MultiTurnAgent(referent_concepts=referents, concepts=concepts,
                               grounded_codes=grounded if grounded else None, seed=seed,
                               wm_n=wm_n, wm_pattern_size=pattern_size,
                               enable_neural_render=enable_neural_render, composer_kind="rf",
                               enable_biased_competition=_bc_enabled(), defer_planner=True, event_register=ev_reg)
    else:
        agent = BrainConversationalAgent(seed=seed, concepts=concepts,
                                         grounded_codes=grounded if grounded else None,
                                         composer_kind="rf", enable_neural_render=enable_neural_render)
    inner = getattr(agent, "agent", agent)
    n = 0
    for a, v, p in facts:
        inner.hear(f"{a} {v} {p}", polarity="AFFIRM")
        n += 1
    aliases = set(cur.get("self_reference", {}).get("agent_aliases", [])) | DEFAULT_SELF_ALIASES
    return agent, aliases, n


def _build_tiny_demo(seed, use_multiturn, enable_neural_render, composer_kind="rf", integrated_loop=False):
    """A tiny CPU brain for the GPU-FREE smoke: a handful of self-facts + a couple of object facts. Mirrors the
    self-knowledge shape so the smoke exercises self-reference + the moat + multi-turn anaphora.

    `composer_kind` (default 'rf' = the numpy fast-path recall, byte-identical to before): pass 'onebrain' for the
    GENUINELY-SPIKING recall (resonate-and-fire per query + the on-substrate cleanup/store; runtime new-word LEARN
    works via the vocab_headroom recruit-an-assembly path). The onebrain build is much slower (~180s) but is the
    brain-based-only recall the mission requires; speed is secondary.

    `integrated_loop` (scaffold-retirement backlog rank-2, default False = byte-identical): pass-through to
    MultiTurnAgent/BrainConversationalAgent -> OneBrainComposer's spiking K-way cue-match SEQUENCER, a no-op unless
    `composer_kind='onebrain'`. Honest scope: this tiny fixture's vocab (~15 words) is BELOW the validated
    production margin -- the divnorm-WTA agent-line decode over-abstains at this scale (the SAFE direction, moat
    0-FA, never a false-accept -- `research/findings/raw/_burndown_1A_c2_smallvocab_derisk.json`), so turning this
    on for the tiny-demo brain specifically is expected to lose some who/what answers it would otherwise give. The
    validated GO regime is the production vocab tier (V=320, `2026-06-21-shortcut3-fold-integrated-loop-BUILD.md`);
    this parameter exists here so the SAME webapp/brain_chat_tui code path also serves a real (large-vocab)
    developed brain, not so the small tiny-demo fixture should be run with it on."""
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    # base-form verbs so the template-stub's 3rd-person inflection reads cleanly (use->uses, learn->learns).
    # 'cat' is the OBJECT of (dog chase cat) AND the SUBJECT of (cat eat fish) -- the validated chainable-referent
    # pattern so 'what does it eat' resolves 'it'->cat (the dog's chase-object) and answers 'fish'.
    facts = [
        ("brain", "use", "spikes"),
        ("brain", "learn", "words"),
        ("brain", "store", "memory"),
        ("dog", "chase", "cat"),
        ("cat", "eat", "fish"),
    ]
    actions = {v for _a, v, _p in facts}
    # include the discourse event register's animal referents (worm/ball/river/bird) so a multi-clause discourse
    # ("dog chase cat. then bird chase worm.") folds into BOTH the running event AND the composer without a code miss.
    vocab = sorted({w for f in facts for w in f} | {"river", "bird", "fish", "worm", "ball"})
    concepts = {w: None for w in vocab}
    if use_multiturn:
        from research.runners.multi_turn_agent import MultiTurnAgent
        referents = [w for w in vocab if w not in actions]
        # DISCOURSE EVENT REGISTER (2026-07-10): a running FACTORED (agent, patient) event so the console can answer
        # "who was doing it before?" across a connective ("dog chase cat. THEN bird chase worm. who was before?" -> dog).
        # The labelled PairEventRegister (0.928, its validated animal referents), numpy (spiking=False) for the CPU path.
        ev_reg = None
        try:
            import research.runners.d3_discourse_event_register_production_organ as _DR
            # D3 discourse-event register (F2 wire, 2026-08-13): default-ON -> the genuinely-SPIKING twin;
            # BRAIN_DISCOURSE_REGISTER=0 -> today's spiking=False (byte-identical); BRAIN_DISCOURSE_REGISTER_LESION=1
            # -> the prev-slot-silence spiking register (load-bearing).
            ev_reg = _DR.make_discourse_register(["dog", "cat", "fish", "bird", "worm", "ball"], seed=seed)
        except Exception as _e:
            print(f"[tui] discourse event register unavailable ({_e!r}); who-was-before disabled.", flush=True)
        # --- SELECTIVE ATTENTION / biased-competition wire-in (env-gated, default OFF = byte-identical) ---------
        from research.runners.biased_competition_prod import biased_competition_enabled as _bc_enabled
        agent = MultiTurnAgent(referent_concepts=referents, concepts=concepts, seed=seed,
                               enable_neural_render=enable_neural_render, composer_kind=composer_kind,
                               integrated_loop=integrated_loop,
                               enable_biased_competition=_bc_enabled(), defer_planner=True, event_register=ev_reg)
    else:
        agent = BrainConversationalAgent(seed=seed, concepts=concepts, composer_kind=composer_kind,
                                         integrated_loop=integrated_loop,
                                         enable_neural_render=enable_neural_render)
    inner = getattr(agent, "agent", agent)
    for a, v, p in facts:
        inner.hear(f"{a} {v} {p}", polarity="AFFIRM")
    return agent, DEFAULT_SELF_ALIASES, len(facts)


def _resolve_composer_kind(args):
    """The tiny-demo recall substrate for the TUI. Interactive default = 'onebrain' (the GENUINELY-SPIKING recall,
    resonate-and-fire per query, runtime new-word LEARN — the same production default the webapp uses), so the owner
    gets the full spiking brain in the TERMINAL too, not only the web UI. Resolution order: explicit --composer wins;
    then the automated --smoke path forces 'rf' (the GPU-free smoke must stay fast + byte-identical); then the
    BRAIN_COMPOSER_KIND env (shared with the webapp); else 'onebrain'. Pass --composer rf for the fast numpy path."""
    if getattr(args, "composer", None):
        return args.composer
    if getattr(args, "smoke", False):
        return "rf"
    return os.environ.get("BRAIN_COMPOSER_KIND", "onebrain")


def _resolve_integrated_loop(args):
    """The (agent, action) cue-match SELECTION substrate (scaffold-retirement backlog rank-2). Default OFF
    (byte-identical to before this flag existed -- the host first-match `_scan`): unlike `_resolve_composer_kind`,
    there is no interactive-default flip here, because the validated GO regime is the production vocab tier
    (V=320), not this TUI's small fixtures. Resolution order: explicit --integrated-loop/--no-integrated-loop wins;
    else the BRAIN_INTEGRATED_LOOP env (shared with the webapp's `_build_chat_brain`); else OFF."""
    if getattr(args, "integrated_loop", None) is not None:
        return bool(args.integrated_loop)
    return os.environ.get("BRAIN_INTEGRATED_LOOP", "").strip().lower() in ("1", "true", "yes", "on")


def load_brain(args):
    """Resolve --load / --self-knowledge / --tiny-demo into (agent, self_aliases, n_facts, source_desc)."""
    use_mt = not args.no_multiturn
    nr = args.neural_render
    il = _resolve_integrated_loop(args)
    # explicit developed-brain bundle directory
    if args.load and is_developed_brain_bundle(args.load):
        agent, manifest = load_developed_brain(args.load, use_multiturn=use_mt, enable_neural_render=nr,
                                               integrated_loop=il)
        aliases = set(manifest.get("self_aliases") or []) | DEFAULT_SELF_ALIASES
        n = manifest.get("n_facts", len(getattr(agent, "agent", agent).composer.kb))
        return agent, aliases, n, f"developed-brain bundle: {args.load}"
    # self-knowledge brain (explicit flag, or a --load pointing at a codes .json)
    if args.self_knowledge or (args.load and str(args.load).endswith(".json")):
        codes = args.load if (args.load and str(args.load).endswith(".json")) else _SK_CODES
        curriculum = args.curriculum or _SK_CURRICULUM
        # NOT threaded here: _load_self_knowledge hard-codes composer_kind="rf" (integrated_loop only reads on
        # 'onebrain'), so it would be a silent no-op -- out of this rank's scope (composer_kind selection, not the
        # cue-match SELECTION this flag targets).
        agent, aliases, n = _load_self_knowledge(codes, curriculum, args.seed, use_mt, nr)
        return agent, aliases, n, f"self-knowledge brain (codes={os.path.relpath(codes, _REPO) if os.path.exists(codes) else 'seed-codes'})"
    # tiny CPU demo — interactive default is the genuinely-SPIKING onebrain recall (the --smoke path stays 'rf' fast)
    if args.tiny_demo or not args.load:
        ck = _resolve_composer_kind(args)
        agent, aliases, n = _build_tiny_demo(args.seed, use_mt, nr, composer_kind=ck, integrated_loop=il)
        return agent, aliases, n, f"tiny CPU demo brain (composer={ck})"
    raise FileNotFoundError(f"--load {args.load!r} is neither a developed-brain bundle nor a codes .json")


# ============================================================================================================
# The renderer factory.
# ============================================================================================================

def build_renderer(args):
    """Build the fluent renderer: the off-bridge Qwen (default) or the template-stub (--stub-renderer / smoke)."""
    if args.stub_renderer:
        return StubRenderer()
    if args.no_renderer:
        return None
    return QwenRenderer(T=args.T, max_new_tokens=args.max_new_tokens, seed=args.seed)


# ============================================================================================================
# The interactive REPL.
# ============================================================================================================

_BANNER = """\
============================================================================
  BRAIN CHAT  --  talk to a developed brain about what it knows
============================================================================
  Source : {source}
  Knows  : {n_facts} facts   |   Renderer: {renderer}
  Self   : 'you'/'your'/'I'/'me'/'it' map to the brain (ask 'what are you?')
  Moat   : the brain ABSTAINS ('I don't know about that.') on anything it
           was not taught -- it never makes things up.
  Commands: /facts  /raw  /help  /quit
============================================================================
"""

_HELP = """\
  /facts   list the facts the brain knows
  /raw     toggle the brain's OWN renderer (no LLM) -- raw recalled triple
  /help    show this help
  /quit    exit  (also /exit, /q, Ctrl-D)
"""


def _print_facts(chat):
    facts = chat.list_facts()
    if not facts:
        print("  (the brain knows no facts.)", flush=True)
        return
    print(f"  the brain knows {len(facts)} facts:", flush=True)
    for a, v, p in facts:
        print(f"    - {a} {v} {p}", flush=True)


def run_repl(chat, source, n_facts, rich=None):
    """The interactive chat loop. When `rich` is a RichAnswerComposer, each turn produces a SUBSTANTIVE
    multi-sentence GROUNDED reply (direct recall + multi-hop chain + elaboration, each sentence verify-checked);
    'tell me more' / 'why?' elaborates the held topic further. Otherwise the default single-fact answer."""
    rname = chat.renderer.name if chat.renderer is not None else "(none -- raw brain triples)"
    mode = "RICH (multi-sentence grounded; 'tell me more' elaborates)" if rich is not None else "single-fact"
    print(_BANNER.format(source=source, n_facts=n_facts, renderer=f"{rname}   |   answers: {mode}"), flush=True)
    # THE SHARED FULL-FACULTY PIPELINE (2026-08-27): route every turn through webapp.brain_reply so the standalone
    # TUI fires the SAME faculty-DRIVE couplings the webapp does (affect/swap/metacog/curiosity/surprise/world-model/
    # prospective/... ), not just the recall+moat core `chat.answer` runs. Falls back to the local answer path if the
    # webapp package is unavailable (keeps the TUI runnable in a minimal env). See webapp/brain_reply.py.
    _shared = None
    try:
        from webapp import brain_reply as _shared
    except Exception as _imp_e:
        print(f"[tui] shared faculty pipeline unavailable ({type(_imp_e).__name__}: {_imp_e}); "
              f"using the local recall-core answer path (no faculty couplings).", flush=True)
        _shared = None
    _renderer_label = chat.renderer.name if chat.renderer is not None else "raw"
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[tui] bye.", flush=True)
            break
        if not line:
            continue
        low = line.lower()
        if low in ("/quit", "/exit", "/q", "quit", "exit"):
            print("[tui] bye.", flush=True)
            break
        if low in ("/help", "help", "?"):
            print(_HELP, flush=True)
            continue
        if low == "/facts":
            _print_facts(chat)
            continue
        if low == "/raw":
            chat.raw_mode = not chat.raw_mode
            print(f"  [raw mode {'ON -- the brain speaks its own raw triples (no LLM)' if chat.raw_mode else 'OFF -- fluent rendering'}]",
                  flush=True)
            continue
        if chat.verbose_thinking and chat.renderer is not None and not chat.raw_mode:
            print("  brain> thinking...", flush=True)
        # SHARED full-faculty pipeline (default): the couplings decorate/drive the reply exactly as they do on the
        # webapp, so the TUI now exercises the WHOLE brain, not the recall core alone.
        if _shared is not None:
            try:
                payload = _shared.reply_over_chat(
                    chat, line, source=source, brain=source, renderer=_renderer_label,
                    rich=(rich is not None), session="tui")
                ans = payload.get("answer", "")
                abstained = bool(payload.get("abstained"))
                if rich is not None and payload.get("rich"):
                    ns = int(payload.get("n_sentences", 0) or 0)
                    tag = "  (abstained -- the moat)" if abstained else f"  [{ns} grounded sentences]"
                else:
                    tag = "  (abstained -- the moat)" if abstained else ""
                print(f"brain> {ans}{tag}\n", flush=True)
                continue
            except Exception as _turn_e:
                print(f"[tui] shared pipeline turn failed ({type(_turn_e).__name__}: {_turn_e}); "
                      f"falling back to the local answer path.", flush=True)
        # LOCAL fallback (recall+moat core only — no faculty couplings): used when the webapp package is unavailable.
        if rich is not None:
            r = rich.answer(line)
            tag = "  (abstained -- the moat)" if r["abstained"] else f"  [{r['n_sentences']} grounded sentences]"
            print(f"brain> {r['answer']}{tag}\n", flush=True)
        else:
            ans, abstained = chat.answer(line)
            tag = "  (abstained -- the moat)" if abstained else ""
            print(f"brain> {ans}{tag}\n", flush=True)


# ============================================================================================================
# The GPU-FREE scripted SMOKE.
# ============================================================================================================

def run_smoke(chat, source, n_facts, out_path):
    """Scripted multi-turn turns (incl. anaphora + abstention + self-reference) on the tiny CPU brain with the
    template-stub renderer. Verifies the TUI loads + converses + the moat abstains + multi-turn anaphora works."""
    # the scripted multi-turn conversation. Each entry: (utterance, expectation-kind).
    # 'anaphora' uses the prior turn's referent; 'abstain' must hit the moat; 'self' is a self-reference question.
    script = [
        ("what are you", "answer"),              # self-reference: 'you' -> brain ('brain uses spikes')
        ("how do you learn", "answer"),          # self-reference synonym: learn -> learns ('brain learns words')
        ("what does the brain store", "answer"),  # direct self-fact ('brain store memory')
        ("what does the dog chase", "answer"),   # object fact -> the answer 'cat' is a chainable referent -> WM
        ("what does it eat", "anaphora"),        # anaphora: 'it' -> cat (the dog's chase-object) -> 'fish'
        ("what does the dragon do", "abstain"),  # untaught subject -> the moat abstains
        ("who wrote romeo and juliet", "abstain"),  # general knowledge never taught -> abstain (the firewall)
        ("what is the capital of france", "abstain"),  # Qwen knows this; the brain must NOT (firewall)
    ]
    transcript = []
    for utterance, kind in script:
        gate_svo = chat.gate(utterance)          # peek the gate so the transcript records what the brain recalled
        ans, abstained = (chat.answer(utterance) if gate_svo is None
                          else (chat.render(gate_svo), False))
        transcript.append({"you": utterance, "kind": kind, "gate_svo": gate_svo,
                           "brain": ans, "abstained": abstained})

    # DISCOURSE EVENT TRACKING (2026-07-10): hear a multi-clause discourse across a connective, then answer
    # "who was doing it before?" -- the deployed D3 event-register capability, now reachable in the console.
    disc_ok = None
    if getattr(chat, "has_event_register", False):
        fresh_before, fresh_abst = chat.answer("who was doing it before?")   # nothing said yet -> the moat abstains
        chat.answer("dog chase cat")
        chat.answer("then bird chase worm")                                  # the connective pushes dog's event
        before, _ = chat.answer("who was doing it before?")                  # -> dog
        now, _ = chat.answer("who is doing it now?")                         # -> bird
        disc_ok = bool(fresh_abst and ("dog" in before.lower()) and ("bird" in now.lower()))
        transcript.append({"you": "[discourse] dog chase cat / then bird chase worm / who was before? / now?",
                           "kind": "discourse", "gate_svo": None, "abstained": False,
                           "brain": f"before={before!r} now={now!r} (fresh-moat abstained={fresh_abst})",
                           "discourse_ok": disc_ok})

    # checks
    self_q = transcript[0]
    self_answered = (not self_q["abstained"]) and self_q["gate_svo"] is not None and self_q["gate_svo"][0] == "brain"
    learn_q = next((t for t in transcript if t["you"] == "how do you learn"), None)
    learn_answered = bool(learn_q and not learn_q["abstained"]
                          and learn_q["gate_svo"] is not None and learn_q["gate_svo"][1] == "learn")
    # anaphora (RIGOROUS): the 'what does it eat' turn must have RESOLVED 'it' to the EXACT prior referent ('cat',
    # the dog's chase-object) AND answered the cat-eat-fish fact. A resolution to anything but 'cat', or an
    # abstention, FAILS -- so a spurious WM read cannot pass.
    anaphora_turn = next(t for t in transcript if t["you"] == "what does it eat")
    resolved_to = chat._resolve_anaphora("what does it eat")
    anaphora_resolved = (("cat" in resolved_to.split()) and ("it" not in resolved_to.split())
                         and (not anaphora_turn["abstained"])
                         and anaphora_turn["gate_svo"] == ["cat", "eat", "fish"])
    # abstention turns must abstain (the moat)
    abstain_turns = [t for t in transcript if t["kind"] == "abstain"]
    moat_held = all(t["abstained"] for t in abstain_turns)
    # at least the self + object facts answered (the brain converses)
    answered = [t for t in transcript if t["kind"] == "answer" and not t["abstained"]]
    converses = len(answered) >= 3

    discourse_ok = (disc_ok is None) or bool(disc_ok)   # if a register is present it must track; else neutral
    go = bool(self_answered and learn_answered and anaphora_resolved and moat_held and converses and discourse_ok)

    verdict = (
        f"GO -- the TUI loads a saved/tiny brain + holds a multi-turn conversation: self-reference resolves "
        f"('what are you' -> {self_q['gate_svo']}), learn-synonym resolves to the 'brain learn words' fact, "
        f"multi-turn anaphora binds 'it' -> {resolved_to!r} (the dog's chase-object 'cat') and answers "
        f"['cat','eat','fish'], the no-confab moat abstains on all {len(abstain_turns)} untaught/general cues "
        f"(incl. 'capital of France' the LLM knows but the brain must not), and {len(answered)} fact turns "
        f"answered. Renderer={chat.renderer.name if chat.renderer else 'raw'}. READY for the owner to --load the "
        f"real developed brain (with the off-bridge Qwen renderer)."
        if go else
        f"PARTIAL/SNAG -- self_answered={self_answered} learn_answered={learn_answered} "
        f"anaphora_resolved={anaphora_resolved} (resolved={resolved_to!r}) moat_held={moat_held} "
        f"converses={converses} ({len(answered)} fact turns). See the transcript for the localize."
    )

    res = {
        "go": go,
        "verdict": verdict,
        "backend": os.environ.get("SIM_BACKEND"),
        "source": source,
        "renderer": (chat.renderer.name if chat.renderer is not None else "raw brain triples"),
        "n_facts": n_facts,
        "self_reference_answered": self_answered,
        "learn_synonym_answered": learn_answered,
        "multiturn_anaphora_resolved": anaphora_resolved,
        "anaphora_resolved_to": resolved_to,
        "moat_held": moat_held,
        "n_abstain_turns": len(abstain_turns),
        "n_answer_turns": len(answered),
        "converses": converses,
        "transcript": transcript,
        "tui_features": [
            "load a developed brain (codes + facts + vocab) from a developed_brain_io bundle, OR the self-knowledge "
            "codes+curriculum, OR a tiny CPU fallback",
            "multi-turn chat: GATE (recall + abstain) -> CONSTRAIN+VERIFY fluent render (off-bridge Qwen default; "
            "template-stub for the GPU-free smoke) -> answer or 'I don't know about that.'",
            "multi-turn anaphora (it/that/they -> the prior referent via the MultiTurnAgent discourse WM)",
            "self-reference (you/your/I/me/it -> the brain) so 'what are you' / 'how do you learn' resolve",
            "commands: /raw (brain's own renderer, no LLM), /facts (list knowledge), /help, /quit",
            "the no-confab moat: the brain abstains on anything it was not taught (verified at the recall layer)",
        ],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)

    # print the transcript
    print("\n" + "=" * 90, flush=True)
    print("[tui SMOKE] scripted multi-turn transcript:", flush=True)
    print("=" * 90, flush=True)
    for t in transcript:
        gate = "" if t["gate_svo"] is None else f"   (recalled: {t['gate_svo']})"
        atag = "  [ABSTAIN]" if t["abstained"] else ""
        print(f"  you>   {t['you']}", flush=True)
        print(f"  brain> {t['brain']}{atag}{gate}", flush=True)
    print("=" * 90, flush=True)
    print(f"[tui SMOKE] VERDICT: {verdict}", flush=True)
    print(f"[tui SMOKE] saved {os.path.relpath(out_path, _REPO)}", flush=True)
    return res


# ============================================================================================================
# main.
# ============================================================================================================

def main():
    ap = argparse.ArgumentParser(description="Talk to a developed/trained brain (multi-turn).")
    ap.add_argument("--load", default=None,
                    help="a developed-brain bundle DIR (brain.json+...) OR a grounded-codes .json (self-knowledge).")
    ap.add_argument("--self-knowledge", action="store_true",
                    help="load the self-knowledge brain (default codes + curriculum).")
    ap.add_argument("--curriculum", default=None,
                    help="curriculum .json for the self-knowledge brain (default: _curriculum_self_knowledge.json).")
    ap.add_argument("--tiny-demo", action="store_true",
                    help="build a tiny CPU brain from a handful of facts (GPU-free fallback / smoke).")
    ap.add_argument("--composer", choices=["rf", "onebrain"], default=None,
                    help="tiny-demo recall substrate: 'onebrain' (GENUINELY SPIKING, resonate-and-fire, the interactive "
                         "default) or 'rf' (numpy fast path). Default: onebrain interactively, rf under --smoke; the "
                         "BRAIN_COMPOSER_KIND env is honored when this is unset. The onebrain build is ~180s (speed secondary).")
    ap.add_argument("--integrated-loop", dest="integrated_loop", action="store_true", default=None,
                    help="route the (agent, action) cue-match SELECTION through the validated spiking K-way "
                         "sequencer instead of the host first-match scan (composer_kind='onebrain' only; a no-op "
                         "otherwise). Default OFF; the BRAIN_INTEGRATED_LOOP env is honored when neither flag is "
                         "passed. GO at production vocab (V=320); this TUI's tiny/self-knowledge fixtures are "
                         "smaller than the validated margin (see _build_tiny_demo's docstring).")
    ap.add_argument("--no-integrated-loop", dest="integrated_loop", action="store_false",
                    help="force the host first-match scan (the byte-identical default).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-multiturn", action="store_true",
                    help="use the bare BrainConversationalAgent (no discourse WM / anaphora).")
    ap.add_argument("--neural-render", action="store_true",
                    help="enable the brain's own spiking serial-order renderer (slow).")
    # renderer
    ap.add_argument("--stub-renderer", action="store_true",
                    help="use the GPU-FREE template-stub renderer (the CPU smoke); default is the off-bridge Qwen.")
    ap.add_argument("--no-renderer", action="store_true",
                    help="no fluent renderer (the brain speaks its own raw triples).")
    ap.add_argument("--T", type=int, default=16, help="off-bridge Qwen rate-code pool budget (16=GO).")
    ap.add_argument("--max-new-tokens", type=int, default=24, help="Qwen surface-form length cap.")
    # rich answers (opt-in)
    ap.add_argument("--rich", action="store_true",
                    help="SUBSTANTIVE multi-sentence GROUNDED replies (direct recall + multi-hop chain + "
                         "elaboration, each sentence verify-checked); 'tell me more'/'why?' elaborates further. "
                         "Default OFF = the single-fact oracle answer.")
    ap.add_argument("--rich-max-sentences", type=int, default=4, help="max sentences per rich reply.")
    ap.add_argument("--no-neural-planner", action="store_true",
                    help="(--rich only) DISABLE the spiking dlPFC discourse-planner; use the HOST gather/order/"
                         "stop heuristics instead. Default = neural-ON (the brain-based-purity version: the dlPFC "
                         "spreading-activation latency rank drives WHICH grounded facts to bring up, in WHAT order, "
                         "and WHEN to stop). The escape exists for the numpy-CPU / reproducibility / test-oracle "
                         "path (the host planner avoids building a per-topic SimulationBridge).")
    # smoke
    ap.add_argument("--smoke", action="store_true",
                    help="run the scripted GPU-FREE smoke (no interactive input) + write the JSON verdict.")
    ap.add_argument("--out", default="research/findings/raw/_brain_chat_tui_smoke.json",
                    help="smoke JSON output path.")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import logging
    logging.disable(logging.INFO)

    # load the brain
    agent, aliases, n_facts, source = load_brain(a)
    # build the renderer (the smoke forces the stub if neither flag set)
    if a.smoke and not a.stub_renderer and not a.no_renderer:
        a.stub_renderer = True   # the GPU-free smoke uses the template-stub by default
    renderer = build_renderer(a)
    chat = ChatBrain(agent, self_aliases=aliases, renderer=renderer)

    if a.smoke:
        res = run_smoke(chat, source, n_facts, os.path.join(_REPO, a.out) if not os.path.isabs(a.out) else a.out)
        return 0 if res["go"] else 1

    rich = None
    if a.rich:
        from research.runners.rich_answer_composer import RichAnswerComposer
        # DEFAULT = the NEURAL discourse-planner (brain-based-purity): the spiking dlPFC content-selection
        # (SpikingSpreadingController) drives WHICH grounded facts to bring up, in WHAT neural-relevance order,
        # and WHEN to stop -- the GO 3G replacement for the host gather/order/stop heuristics (quality-parity,
        # lesion-load-bearing, on-topic, moat 0-FA). `--no-neural-planner` is the host escape.
        # numpy-CPU nuance (mirrors the 1A sentinel): the planner builds + steps a per-topic SimulationBridge,
        # which is heavy on the CPU smoke path -- so on the numpy backend we keep the HOST default for
        # portability/speed (neural-on stays the GPU default). The explicit `--no-neural-planner` always forces
        # host regardless of backend.
        try:
            from sim.backend import is_gpu_backend
            _on_gpu = bool(is_gpu_backend())
        except Exception:
            _on_gpu = (os.environ.get("SIM_BACKEND", "").lower() == "cupy")
        neural_planner = (not a.no_neural_planner) and _on_gpu
        if a.no_neural_planner:
            print("[rich] neural discourse-planner DISABLED (--no-neural-planner): host gather/order/stop.",
                  flush=True)
        elif not _on_gpu:
            print("[rich] neural discourse-planner: HOST default on the numpy-CPU backend "
                  "(the spiking dlPFC planner needs a bridge; use SIM_BACKEND=cupy for neural-ON).", flush=True)
        else:
            print("[rich] neural discourse-planner ON (default): spiking dlPFC content-selection drives "
                  "gather/order/stop.", flush=True)
        rich = RichAnswerComposer(chat, max_sentences=a.rich_max_sentences,
                                  neural_planner=neural_planner, planner_seed=a.seed)
    run_repl(chat, source, n_facts, rich=rich)
    return 0


if __name__ == "__main__":
    sys.exit(main())
