"""NP-ENTAILMENT MOAT GATE (2026-09-01, Vikunja board frontier row) -- wires NPHeadBinder (spiking
NP-boundary binding, `research/runners/_spiking_np_boundary_extraction_derisk.py`) + entailment
classification (`FactStore`/`classify_claim`, `research/runners/_open_text_moat_verifier_derisk.py`)
into the LIVE open-text moat verifier: `webapp.open_ended_chat.post_filter`'s KNOWN-topic path.

THE GAP THIS CLOSES (not a hollow checkbox flip -- see the finding for the measured proof). The
live known-topic filter (`_clause_filter_sentence` -> `sentence_contradicts`,
`research/runners/_open_ended_known_supplement_filter_derisk.py`) is a HOST GAZETTEER: it only
recognizes THREE relation shapes -- borders / continent / capital -- plus a bare number/year regex.
A fabricated supplement on ANY OTHER relation has no matching branch, so `sentence_contradicts`
returns None and the sentence survives unedited: "Mercury discovered Neptune" (store only holds
mercury/orbits/sun) trips zero gazetteer branches and leaks. This gate closes that class
GENERICALLY: NPHeadBinder + BridgeParser (both spiking, vocabulary-agnostic position x voice role
assignment -- BOTH reused UNCHANGED, no `sim/` edit) extract an (agent, action, patient) triple for
any clause `segment_clause` can segment (the exact-3-content-word fast path is fully general over
the VERB -- "discovered" needs no lexicon entry, unlike the gazetteer's fixed relation set), then
`classify_claim`/`FactStore` (the SAME entailment semantics production's single-triple
`ask_yes_no` already uses, reused UNCHANGED) checks it against the SAME retrieved facts the live
turn already holds. Any relation the extractor can parse becomes checkable, not just the three
gazetteer branches.

MONOTONIC / ADDITIVE-ONLY SCOPE (bounds the false-reject risk deliberately). This gate can only
ever DROP a sentence the earlier stages already kept -- it never restores or edits text an earlier
stage removed, and it is a no-op (passes the sentence through unchanged) whenever:
  (a) no clause in the sentence reduces to a frame `segment_clause` can close (most compound /
      listy / long descriptive sentences fall here -- an honest coverage limit, not a defect: it
      means this gate can UNDER-catch, never OVER-reject content it cannot confidently parse);
  (b) the parsed clause's subject does not normalize to the retrieved topic (the fact store handed
      to this gate holds ONLY that topic's facts, so a claim about a different entity is out of
      its adjudicable scope -- left to whatever else in the pipeline handles it);
  (c) the clause is a hedge/opinion (`is_opinion`, reused unchanged -- entailment does not apply to
      an explicitly flagged guess, the same carve-out `_moat_claim_entailment_derisk.py` documents);
  (d) the parsed verb is a COPULA ("is"/"are"/"was"/"were"). A copula's object is typically a rich,
      elaborative predicate nominal ("a vast country located in North America") that a single
      store object cannot be expected to exact-match -- checking it with the SAME strict
      `classify_claim` used for a discrete relation produces FALSE REJECTS on true, merely
      elaborated content (measured directly against the real saved known-topic Qwen replies during
      this wiring's own verify -- see the finding). Excluding copula keeps the gate's value-add
      (catching SPECIFIC wrong-relation/wrong-entity confabulations on non-copula verbs, the
      observed hallucination shape) without regressing genuinely-grounded descriptive prose.
This is a scope decision, not a weakening: it is the SAME "when unsure, don't touch it" posture the
gazetteer filter and the base post-filter already take on constructions outside their own coverage.

FLAG (default OFF -- `BRAIN_OPEN_ENDED_NP_ENTAILMENT`, see `webapp.open_ended_chat.
np_entailment_enabled`). Additive: `webapp.open_ended_chat.post_filter` only imports THIS module
inside that flag's truthy branch (mirroring the existing `wkv_mouth_generator` / gen-time-honesty
lazy-import pattern in the same file), so with the flag off this module -- and the heavier
`research.runners._spiking_np_boundary_extraction_derisk` / `brain_conversational_agent` imports
it in turn pulls in, including their own `os.environ.setdefault("SIM_BACKEND", "numpy")` -- is
NEVER imported and `post_filter` is BYTE-IDENTICAL to before this file existed.

WIDENED COVERAGE (2026-09-02 follow-on, board frontier row, `BRAIN_OPEN_ENDED_NP_ENTAILMENT_
COPULA_COVERAGE`, default OFF). The 2026-09-01 real-traffic moat-safety soak (`research/findings/
2026-09-01-open-ended-bundle-moat-safety-soak-fabrication-delta.md`) measured that scope-(d)'s
copula exclusion is the dominant real-traffic miss: NP-entailment changed ZERO of 12 real known-
topic Qwen replies, because free prose is dominated by copula ("is a professional football club")
that scope (d) declines to touch. The concrete miss: `castleford_f_c` gets called a "professional
football club" when the store's only sport fact is `rugby_leauge` -- a confident wrong-type
fabrication that leaked through unedited. This flag adds a SEPARATE, NARROW, ADDITIVE-ONLY check
(`_copula_wide_extract` + `_copula_category_conflict` below) that fires ONLY when ALL of the
following hold, keeping the false-reject risk that motivated scope (d) bounded rather than
reopening the whole copula construction to the strict single-object `classify_claim` match that
would false-reject genuinely elaborated true content (the SAME risk scope (d) was written to
avoid):
  1. a copula auxiliary (is/are/was/were) is found anywhere in the sentence, with the SUBJECT
     read as the sentence's leading comma-segment (tolerates a between-subject-and-copula
     appositive -- "Castleford FC, commonly known as Castleford F.C., is ..." -- which
     `split_clauses`'s blanket comma-split (used by the ORIGINAL per-clause loop below) severs,
     losing the subject before the per-clause path ever sees it);
  2. that subject SEPARATOR-INVARIANTLY matches the retrieved topic (`_normalize_loose`: collapses
     underscores/hyphens/periods/whitespace and drops a leading article before comparing, so a
     Wikidata-slug topic like `castleford_f_c` matches the human-readable subject text extraction
     pulls from real prose, `castleford fc` -- still an EXACT-identity match after the collapse,
     not a fuzzy one, same precision the pre-existing item (b) already relies on);
  3. the predicate is not a negation, present-participle ("is playing ...", progressive aspect,
     not an identity predicate), or passive past-participle ("is built ...", a different
     construction already handled by `segment_clause`'s own passive pass) -- these are left
     untouched, an honest residual, not a guess;
  4. the predicate names a CATEGORY WORD from a small, explicit, extensible lexicon
     (`_CATEGORY_WORDS`, today: common sport names -- the concrete measured miss's own domain) that
     CONFLICTS with a DIFFERENT, also-recognized, category word in a store fact for the SAME
     topic (`_copula_category_conflict`) -- e.g. predicate "football" vs stored `sport=rugby_
     leauge` (substring-matched, so the store's own "leauge" typo does not block the "rugby" hit).
An unrecognized predicate, or a store with no matching-family fact, never trips this (silently
returns False) -- it cannot false-reject content this lexicon has no opinion about, the same
"when unsure, don't touch it" posture scope (a)-(d) already take. This check can only ever
ADDITIONALLY drop a sentence the rest of the gate already kept (monotonic, same contract as the
gate's existing scope) -- it never restores text an earlier stage removed, and every other
construction (participial, pronoun-referent, and any copula clause the new checks above do not
positively conflict) is untouched, a named residual, not attempted here (see the finding).
With the flag OFF, `copula_coverage_enabled()` short-circuits before any of the above runs and
`gate_sentence` is BYTE-IDENTICAL to before this addition (verified directly, not assumed -- see
`research/runners/_np_entailment_copula_coverage_verify.py`).

WIDENED COVERAGE #2 (2026-09-01 same-day follow-on, `BRAIN_OPEN_ENDED_NP_ENTAILMENT_PARTICIPIAL_
PRONOUN_COVERAGE`, default OFF). The copula-coverage finding above named two more real-traffic
constructions still uncaught (its "Honest limits" section, and the soak's own "Five concrete
before/after examples" #2 and #4): PARTICIPIAL phrases ("City, bordering Virginia, ..."; "the
club, founded in 1892, ...") and PRONOUN-REFERENT sentences ("It's often associated with ...")
whose subject is a pronoun standing in for the known topic, not the topic's own text. Both are
`split_clauses`-severed exactly like the copula appositive case (a participial phrase set off by
commas loses its subject to the blanket comma-split; a pronoun subject is never the retrieved
topic string so the pre-existing per-clause path's item (b) always treats it as off-topic and
no-ops), so both need their own whole-sentence extraction, mirroring `_copula_wide_extract`'s
fix exactly rather than reopening `split_clauses` itself.

  * **`_participial_wide_extract(sent)`** -- finds the first comma-segment (after the leading
    subject segment) whose FIRST word is a RECOGNIZED relational participle
    (`_PARTICIPIAL_ACTION_MAP`: border(ing), found(ed), built/build(ing), discovered/discover(ing),
    designed/design(ing), created/create(ing), constructed/construct(ing), established/
    establish(ing), located/locate(ing) -- a small, explicit, extensible lexicon, the SAME posture
    as `_CATEGORY_WORDS`), canonicalizes it to a relation-key action string, and takes the rest of
    that segment (minus one leading preposition, e.g. "in"/"by") as the object. Backs off on
    negation (same `_has_negation` guard) or an unrecognized first word -- an honest coverage
    limit, not a guess. `_participial_relation_conflict(...)` then fires ONLY when the SAME topic
    has a store fact for that SAME canonical relation whose patient, loosely normalized, has NO
    overlap (neither substring-contains the other) with the extracted object -- e.g. predicate
    "founded in 1892" vs stored `founded=1919` conflicts; predicate "founded in 1919" against the
    same fact does not. A topic with no fact for that relation at all never trips this (returns
    False) -- the identical "when unsure, don't touch it" posture `_copula_category_conflict`
    already takes for an unrecognized predicate.
  * **`_pronoun_wide_extract(sent)`** -- fires ONLY when the sentence's very first word is a
    third-person pronoun (it/he/she/they) immediately followed by a copula, contracted ("It's",
    "They're") or explicit ("It is"), with the SAME negation/participle/passive backoff guards as
    `_copula_wide_extract`. Because `gate_sentence` is called once per already-known topic (the
    single entity the whole reply is about), a leading pronoun subject is treated as standing for
    that topic directly -- no text-match needed, unlike the copula/participial paths, which is
    exactly why this needs its own function rather than a tweak to the topic-match step used
    elsewhere. The extracted predicate is checked with the SAME, UNCHANGED
    `_copula_category_conflict` the copula-coverage widening already built (no new conflict logic
    for this shape) -- e.g. "It's a well-known football team" after a rugby fact conflicts the
    same way the literal-subject copula case does.
  * **Category lexicon widened** (optional generalization the copula-coverage finding named as
    the natural next step): `_CATEGORY_WORDS` now also covers NATIONALITY, PROFESSION, and
    RELIGION families alongside the original SPORT family -- more mutually-exclusive-type
    conflicts become catchable through BOTH `_copula_category_conflict` call sites (the original
    copula path and the new pronoun-referent path), strictly additive (new word entries only add
    NEW possible conflicts, verified to introduce no regression on the copula-coverage widening's
    own saved battery -- see `research/runners/_np_entailment_copula_coverage_verify.py` re-run
    after this edit).

Both checks run as a second early-return block, AFTER the copula-coverage block and BEFORE the
per-clause loop (same reason: the per-clause loop never sees a comma-severed subject), and can
only ADDITIONALLY drop a sentence the rest of the gate already kept -- the same monotonic contract
every check in this module already holds. With the flag OFF, `participial_pronoun_coverage_
enabled()` short-circuits before any of the above runs and `gate_sentence` is BYTE-IDENTICAL to
before this addition (verified directly against the actual pre-widening file, not assumed -- see
`research/runners/_np_entailment_participial_pronoun_coverage_verify.py`).

Run the wiring verify: `python -m research.runners._np_entailment_moat_gate_wiring_verify`
Run the copula-coverage verify: `python -m research.runners._np_entailment_copula_coverage_verify`
Run the participial/pronoun-coverage verify: `python -m research.runners._np_entailment_participial_pronoun_coverage_verify`
"""
from __future__ import annotations

import os
import re
import threading

_COPULA_ACTIONS = {"is", "are", "was", "were"}

_LOCK = threading.Lock()
_PARSER = None
_NP_BINDER = None

# ---------------------------------------------------------------------------------------------
# WIDENED COVERAGE (flag-gated, additive-only) -- see module docstring for the full rationale.
# ---------------------------------------------------------------------------------------------

_FLAG_COPULA_COVERAGE = "BRAIN_OPEN_ENDED_NP_ENTAILMENT_COPULA_COVERAGE"


def copula_coverage_enabled() -> bool:
    """Truthy iff `BRAIN_OPEN_ENDED_NP_ENTAILMENT_COPULA_COVERAGE` is set to a non-empty,
    non-'0'/'false'/'no' value. Default OFF, matching the parent gate's own default-off
    convention (`webapp.open_ended_chat.np_entailment_enabled`)."""
    return os.environ.get(_FLAG_COPULA_COVERAGE, "").strip().lower() not in ("", "0", "false", "no")


_FLAG_PARTICIPIAL_PRONOUN_COVERAGE = "BRAIN_OPEN_ENDED_NP_ENTAILMENT_PARTICIPIAL_PRONOUN_COVERAGE"


def participial_pronoun_coverage_enabled() -> bool:
    """Truthy iff `BRAIN_OPEN_ENDED_NP_ENTAILMENT_PARTICIPIAL_PRONOUN_COVERAGE` is set to a
    non-empty, non-'0'/'false'/'no' value. Default OFF, matching both this module's own
    `copula_coverage_enabled()` and the parent gate's default-off convention. Independent of the
    copula-coverage flag -- either can be on/off without the other."""
    return os.environ.get(_FLAG_PARTICIPIAL_PRONOUN_COVERAGE, "").strip().lower() not in (
        "", "0", "false", "no")


_COPULA_RE = re.compile(r"\b(?:is|are|was|were)\b", re.IGNORECASE)
_CLAUSE_BOUNDARY_RE = re.compile(
    r",|\b(?:and|but|because|so|although|while|which)\b", re.IGNORECASE)
_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_NEGATION_WORDS = {"not", "never", "n't"}
# Reuses the SAME passive-participle lexicon `_spiking_np_boundary_extraction_derisk.segment_clause`
# already uses for its own passive pass -- a predicate opening on one of these is a passive
# ("is built ..."), a different construction already handled elsewhere, not a predicate nominal.
_PASSIVE_PARTICIPLES = {"built", "discovered", "designed", "founded", "created", "constructed"}


def _normalize_loose(s) -> str:
    """Separator-invariant normalization for the WIDENED-COVERAGE subject/topic comparison ONLY
    (never used by the pre-existing per-clause path's `_normalize`, which stays byte-identical).
    Strips a leading article, then collapses underscores/hyphens/periods/apostrophes/parens/
    whitespace, so a Wikidata-slug topic ('castleford_f_c') can match the human-readable subject
    text extraction pulls from real prose ('Castleford FC' -> 'castleford fc'). Still an
    EXACT-identity comparison after the collapse -- not a fuzzy/partial match, the same precision
    the gate's existing `_normalize` already relies on for item (b)."""
    s = str(s).strip()
    s = _LEADING_ARTICLE_RE.sub("", s)
    s = s.lower()
    s = re.sub(r"[\s_\-.'()]+", "", s)
    return s


def _has_negation(text: str) -> bool:
    words = re.findall(r"[a-zA-Z']+", str(text).lower())
    return any(w in _NEGATION_WORDS or w.endswith("n't") for w in words)


def _copula_wide_extract(sent: str):
    """Whole-sentence copula extraction that tolerates a leading comma-appositive between the
    subject and the copula ('X, known as Y, is Z') -- which `split_clauses`'s blanket comma-split
    (used by the pre-existing per-clause loop below) would otherwise sever, losing the subject
    before that loop ever sees the copula clause. Host lexical segmentation only -- never decides
    which word plays which grammatical role (same boundary the rest of this module already draws
    around `segment_clause`/NPHeadBinder/BridgeParser).

    Returns (subject_text, predicate_text) using the FIRST copula auxiliary found, or None when:
    no copula is present; the subject or predicate span is empty; or the predicate opens on a
    negation, a present participle (progressive aspect, e.g. 'is playing football' -- not an
    identity predicate), or a passive past participle (a different construction, already handled
    by `segment_clause`'s own passive pass)."""
    m = _COPULA_RE.search(sent)
    if not m:
        return None
    before, after = sent[:m.start()], sent[m.end():]
    subject_text = before.split(",", 1)[0].strip()
    predicate_text = _CLAUSE_BOUNDARY_RE.split(after, maxsplit=1)[0].strip()
    if not subject_text or not predicate_text:
        return None
    if _has_negation(predicate_text):
        return None
    first_word = re.findall(r"[a-zA-Z']+", predicate_text.lower())
    first_word = first_word[0] if first_word else ""
    if first_word.endswith("ing") or first_word in _PASSIVE_PARTICIPLES:
        return None
    return subject_text, predicate_text


# Canonical CATEGORY word -> family name. Deliberately narrow + extensible: started 2026-09-01 as
# only the concrete measured miss's own domain (sport); widened same-day (2026-09-01 follow-on) to
# three more mutually-exclusive-type families the copula-coverage finding named as the natural
# next step (nationality, profession, religion) -- purely ADDITIVE, no existing word removed or
# reassigned, so every existing sport-family conflict decision is unchanged. A conflict below fires
# ONLY when the copula predicate contains one of these words AND a store fact for the SAME topic
# contains a DIFFERENT word from the SAME family -- both recognized, both different -- never on an
# unrecognized word (an honest coverage limit, not a guess).
_CATEGORY_WORDS = {
    # sport (original, 2026-09-01)
    "football": "sport", "soccer": "sport", "rugby": "sport", "basketball": "sport",
    "baseball": "sport", "cricket": "sport", "hockey": "sport", "tennis": "sport",
    "golf": "sport", "boxing": "sport", "volleyball": "sport", "handball": "sport",
    "athletics": "sport", "swimming": "sport", "gymnastics": "sport", "netball": "sport",
    # nationality (2026-09-01 follow-on widening)
    "american": "nationality", "british": "nationality", "english": "nationality",
    "french": "nationality", "german": "nationality", "canadian": "nationality",
    "scottish": "nationality", "irish": "nationality", "welsh": "nationality",
    "italian": "nationality", "spanish": "nationality", "russian": "nationality",
    "chinese": "nationality", "japanese": "nationality", "indian": "nationality",
    "australian": "nationality", "mexican": "nationality", "brazilian": "nationality",
    "dutch": "nationality", "swedish": "nationality",
    # profession (2026-09-01 follow-on widening)
    "doctor": "profession", "lawyer": "profession", "engineer": "profession",
    "teacher": "profession", "scientist": "profession", "actor": "profession",
    "musician": "profession", "athlete": "profession", "politician": "profession",
    "journalist": "profession", "architect": "profession", "chef": "profession",
    "nurse": "profession",
    # religion (2026-09-01 follow-on widening)
    "christian": "religion", "muslim": "religion", "jewish": "religion",
    "hindu": "religion", "buddhist": "religion", "sikh": "religion",
    "catholic": "religion", "protestant": "religion", "atheist": "religion",
}


def _category_hits(text: str) -> set:
    """Every (family, word) `_CATEGORY_WORDS` recognizes in `text`, word-boundary matched against
    a plain letters-only token stream so e.g. 'rugby_leauge' (the real store typo the 2026-09-01
    soak's data has) still hits 'rugby' (the token 'leauge' just does not hit anything else)."""
    words = re.findall(r"[a-zA-Z]+", str(text).lower())
    return {(_CATEGORY_WORDS[w], w) for w in words if w in _CATEGORY_WORDS}


def _copula_category_conflict(predicate_text: str, facts: list, topic_norm_loose: str) -> bool:
    """True iff the copula predicate names a category (e.g. a sport) that CONFLICTS with a store
    fact for this topic naming a DIFFERENT word in the SAME family -- the concrete 2026-09-01 miss
    (Qwen: '... a professional football club'; store: sport=rugby_leauge). Conservative by
    construction: False unless BOTH sides hit a recognized, DIFFERENT word in the same family, so
    an unrecognized predicate or a store with no matching-family fact never trips this."""
    pred_hits = _category_hits(predicate_text)
    if not pred_hits:
        return False
    for (a, _v, p) in facts:
        if _normalize_loose(a) != topic_norm_loose:
            continue
        for fam, word in pred_hits:
            for fam2, word2 in _category_hits(p):
                if fam == fam2 and word != word2:
                    return True
    return False


# ---------------------------------------------------------------------------------------------
# WIDENED COVERAGE #2 -- participial + pronoun-referent (flag-gated, additive-only; see module
# docstring for the full rationale). Reuses `_normalize_loose`, `_has_negation`,
# `_CLAUSE_BOUNDARY_RE`, `_PASSIVE_PARTICIPLES`, and `_copula_category_conflict` above UNCHANGED.
# ---------------------------------------------------------------------------------------------

# Recognized relational-participle surface forms -> canonical action key, matched against the
# SAME string this project's fact stores already use for a "borders"/"founded"/... relation (see
# e.g. the TRUE_CASES `("canada", "borders", "united states")` fact in the copula-coverage
# verify). Deliberately small + extensible, the SAME posture as `_CATEGORY_WORDS`: an unrecognized
# participle is left untouched (a), never guessed at.
_PARTICIPIAL_ACTION_MAP = {
    "bordering": "borders", "border": "borders", "borders": "borders",
    "founded": "founded", "found": "founded", "founding": "founded",
    "built": "built", "build": "built", "building": "built",
    "discovered": "discovered", "discover": "discovered", "discovering": "discovered",
    "designed": "designed", "design": "designed", "designing": "designed",
    "created": "created", "create": "created", "creating": "created",
    "constructed": "constructed", "construct": "constructed", "constructing": "constructed",
    "established": "established", "establish": "established", "establishing": "established",
    "located": "located", "locate": "located", "locating": "located",
}
_LEADING_PREP_WORDS = {"in", "by", "to", "near", "from", "on", "at"}


def _participial_wide_extract(sent: str):
    """Whole-sentence participial extraction: takes the SENTENCE's leading comma-segment as the
    subject (same convention as `_copula_wide_extract`), then scans the LATER comma-segments for
    the first one whose opening word is a recognized relational participle
    (`_PARTICIPIAL_ACTION_MAP`) -- exactly the shape `split_clauses`'s blanket comma-split would
    otherwise sever the subject away from ('City, bordering Virginia, ...'; 'the club, founded in
    1892, ...'). Returns (subject_text, canonical_action, object_text), or None when: fewer than 2
    comma-segments; no segment opens on a recognized participle; that segment is negated ('not
    bordering ...' -- back off, do not assert the negation is itself the fabrication); or the
    object span (the rest of the segment, minus one leading preposition if present) is empty."""
    segments = [s.strip() for s in sent.split(",")]
    if len(segments) < 2:
        return None
    subject_text = segments[0].strip()
    if not subject_text:
        return None
    for seg in segments[1:]:
        if not seg:
            continue
        words = re.findall(r"[a-zA-Z']+", seg.lower())
        if not words:
            continue
        first = words[0]
        if first not in _PARTICIPIAL_ACTION_MAP:
            continue
        if _has_negation(seg):
            return None
        rest = seg.split(None, 1)
        object_text = rest[1].strip() if len(rest) > 1 else ""
        prep_split = object_text.split(None, 1)
        if prep_split and prep_split[0].lower().strip(".,") in _LEADING_PREP_WORDS:
            object_text = prep_split[1].strip() if len(prep_split) > 1 else ""
        if not object_text:
            return None
        return subject_text, _PARTICIPIAL_ACTION_MAP[first], object_text
    return None


def _participial_relation_conflict(canonical_action: str, object_text: str, facts: list,
                                    topic_norm_loose: str) -> bool:
    """True iff the SAME topic has a store fact for the SAME canonical relation whose patient,
    loosely normalized, has NO overlap (neither substring-contains the other) with the extracted
    object -- e.g. canonical_action='founded', object_text='1892' vs a stored founded=1919 fact
    conflicts; the same object against a stored founded=1892 fact (or any overlapping text) does
    not. Conservative by construction, the SAME posture as `_copula_category_conflict`: a topic
    with NO fact for this relation at all never trips this (returns False) -- it cannot
    false-reject content this relation has no store opinion about."""
    obj_norm = _normalize_loose(object_text)
    if not obj_norm:
        return False
    matching_patients = []
    for (a, v, p) in facts:
        if _normalize_loose(a) != topic_norm_loose:
            continue
        if _normalize(v) != canonical_action:
            continue
        matching_patients.append(_normalize_loose(p))
    if not matching_patients:
        return False   # no matching-relation fact for this topic -- never trips (same posture)
    for patient_norm in matching_patients:
        if patient_norm and (patient_norm in obj_norm or obj_norm in patient_norm):
            return False   # consistent with at least one matching fact
    return True   # every matching-relation fact disagrees with the extracted object -- conflict


_PRONOUN_COPULA_LEAD_RE = re.compile(
    r"^\s*(?:it|he|she|they)(?:'s|'re|\s+(?:is|are|was|were))\b", re.IGNORECASE)


def _pronoun_wide_extract(sent: str):
    """Whole-sentence pronoun-referent extraction: fires ONLY when the sentence's very FIRST word
    is a third-person pronoun (it/he/she/they) immediately followed by a copula, contracted
    ('It's', 'They're') or explicit ('It is', 'He was'). Because `gate_sentence` is called once
    per already-KNOWN topic (the single entity the whole reply is about -- see `webapp.
    open_ended_chat.post_filter`'s per-sentence loop, same `topic` passed to every sentence), a
    leading pronoun subject is treated as standing for that topic directly: no text-match step is
    needed here, unlike the copula/participial paths, which is exactly why this needs its own
    function. Returns predicate_text, or None on the SAME negation/present-participle/passive
    backoff guards `_copula_wide_extract` already uses (an honest coverage limit on those shapes,
    not a guess) -- or when the pronoun is not the sentence's first word (mid-sentence pronouns are
    left untouched: their antecedent is not necessarily the topic, out of scope by construction)."""
    m = _PRONOUN_COPULA_LEAD_RE.match(sent)
    if not m:
        return None
    after = sent[m.end():]
    predicate_text = _CLAUSE_BOUNDARY_RE.split(after, maxsplit=1)[0].strip()
    if not predicate_text:
        return None
    if _has_negation(predicate_text):
        return None
    first_word = re.findall(r"[a-zA-Z']+", predicate_text.lower())
    first_word = first_word[0] if first_word else ""
    if first_word.endswith("ing") or first_word in _PASSIVE_PARTICIPLES:
        return None
    return predicate_text


def _get_spiking_pair(seed: int = 42):
    """Process-shared (BridgeParser, NPHeadBinder), built ONCE under a lock -- mirrors
    `webapp.open_ended_chat.get_generator`'s one-shared-object pattern. Both classes are reused
    UNCHANGED by import; nothing here re-implements or retrains either spiking mechanism."""
    global _PARSER, _NP_BINDER
    if _PARSER is not None and _NP_BINDER is not None:
        return _PARSER, _NP_BINDER
    with _LOCK:
        if _PARSER is None or _NP_BINDER is None:
            from research.runners.brain_conversational_agent import BridgeParser
            from research.runners._spiking_np_boundary_extraction_derisk import NPHeadBinder
            _PARSER = BridgeParser(seed=seed)
            _NP_BINDER = NPHeadBinder(seed=seed)
        return _PARSER, _NP_BINDER


def reset_spiking_pair_for_test():
    """Test-only: force the next `_get_spiking_pair()` call to rebuild (used by the wiring verify
    to exercise the lazy-build path deterministically; never called from the live server)."""
    global _PARSER, _NP_BINDER
    with _LOCK:
        _PARSER = None
        _NP_BINDER = None


def _normalize(s) -> str:
    return " ".join(str(s).strip().lower().split())


def gate_sentence(sent: str, topic: str, facts: list, parser=None, np_binder=None):
    """Screen `sent` (a sentence the existing known-topic string filter already KEPT) with the
    spiking NPHeadBinder-extraction + FactStore-entailment gate.

    Returns `sent` unchanged when the gate does not apply to (any clause of) it -- see the module
    docstring's scope list (a)-(d) -- or every adjudicable clause is grounded. Returns None (drop
    the WHOLE sentence) the first time an adjudicable clause is classified ungrounded/contradicted
    by the SAME `classify_claim` semantics production's single-triple moat already uses.

    `facts` is the SAME (agent, action, patient) triple list `webapp.open_ended_chat.answer_turn`
    already retrieved for this turn (one shared shape end to end -- no new adapter). `parser` /
    `np_binder` default to the process-shared pair (built lazily, once); passed explicitly by the
    wiring verify so it can reuse ONE built pair across many cases without rebuilding."""
    from research.runners._open_text_moat_verifier_derisk import (
        FactStore, Claim, classify_claim, split_clauses, is_opinion,
    )
    from research.runners._spiking_np_boundary_extraction_derisk import extract_svo_npbind

    if parser is None or np_binder is None:
        parser, np_binder = _get_spiking_pair()

    store = FactStore()
    for (a, v, p) in facts:
        store.store(_normalize(a), _normalize(v), _normalize(p))
    topic_norm = _normalize(topic)

    # ---- WIDENED COVERAGE (flag-gated, additive-only; byte-identical when off -- see module
    # docstring). Runs BEFORE the per-clause loop below because `split_clauses`'s blanket
    # comma-split (which that loop depends on) severs the subject from a copula predicate whenever
    # an appositive sits between them ("X, known as Y, is Z") -- this whole-sentence extraction is
    # the only path that ever sees that subject. Can only ADDITIONALLY drop; never restores.
    if copula_coverage_enabled():
        wide = _copula_wide_extract(sent)
        if wide is not None:
            subject_text, predicate_text = wide
            if _normalize_loose(subject_text) == _normalize_loose(topic):
                if _copula_category_conflict(predicate_text, facts, _normalize_loose(topic)):
                    return None                                   # widened-coverage: copula category conflict

    # ---- WIDENED COVERAGE #2 (flag-gated, additive-only; byte-identical when off -- see module
    # docstring). Same reason as the copula block above: `split_clauses`'s blanket comma-split
    # severs a participial phrase from its subject, and a pronoun subject never text-matches the
    # retrieved topic -- both need a whole-sentence extraction, never the per-clause loop below.
    if participial_pronoun_coverage_enabled():
        wide_p = _participial_wide_extract(sent)
        if wide_p is not None:
            subject_text, canonical_action, object_text = wide_p
            if _normalize_loose(subject_text) == _normalize_loose(topic):
                if _participial_relation_conflict(canonical_action, object_text, facts,
                                                   _normalize_loose(topic)):
                    return None                               # widened-coverage: participial relation conflict
        wide_pr = _pronoun_wide_extract(sent)
        if wide_pr is not None:
            predicate_text = wide_pr
            if _copula_category_conflict(predicate_text, facts, _normalize_loose(topic)):
                return None                                   # widened-coverage: pronoun-referent category conflict

    for clause in split_clauses(sent):
        if is_opinion(clause.lower()):
            continue                                         # (c) hedge/opinion -- out of entailment scope
        parsed, _meta = extract_svo_npbind(clause, parser, np_binder)
        if parsed is None:
            continue                                         # (a) not confidently parseable -- no-op, safe
        agent, action, patient, negated = parsed
        if _normalize(agent) != topic_norm:
            continue                                         # (b) not a claim about the retrieved topic
        if _normalize(action) in _COPULA_ACTIONS:
            continue                                         # (d) copula predicate nominal -- out of scope
        claim = Claim(text=clause, kind="assertion", agent=_normalize(agent), action=_normalize(action),
                      patient=_normalize(patient), negated=negated)
        verdict = classify_claim(claim, store)
        if verdict != "grounded":
            return None                                       # NPHeadBinder-extracted, entailment-ungrounded
    return sent
