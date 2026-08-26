"""Minimal RULE-BASED lemma canonicalizer for the live conversational recall path (reasoning-frontier arc,
2026-08-25 -- see research/findings/2026-08-25-reasoning-frontier-chain-routing.md).

WHY. The 2026-08-25 integrated-conversational-state diagnostic found recall FRAGILE to verb inflection:
teaching "the wolf hunts the deer" (in-loop, `ChatBrain._maybe_acquire`) stores the SURFACE token "hunts" as
the action; asking "what does the wolf hunt?" extracts the SURFACE token "hunt"; `hunts != hunt` as plain
string keys, so the composer's exact-match recall (`query_patient`) abstains on a fact the brain was JUST
told. This module gives store-write and query ONE canonical verb/noun key so an inflected surface form
recalls regardless of which inflection was used to teach or to ask.

NO LEMMATIZER LIBRARY IS AVAILABLE (checked 2026-08-25): no spacy/nltk/lemminflect import or install exists
anywhere in this repo or its `.venv`. Two ad hoc suffix-strippers already exist for OTHER narrow jobs --
`comprehension_production_organ._lemma_verb` (only normalizes a form when the BASE is already in the toy
`VERB_SELECTS` whitelist, so it cannot canonicalize a fresh open-vocabulary verb like "hunt") and
`b3_noncontradiction_production_organ._action_lemma_candidates` (a recall-time FALLBACK candidate list, tried
only after the surface form misses, so it never establishes a single canonical STORE key). Neither is a
general bidirectional canonicalizer usable at store time, before any "known verb" whitelist exists. This
module generalizes their suffix-stripping STYLE into ONE canonical function, per the task's explicit
allowance: "a minimal rule-based stemmer is acceptable AS A DOCUMENTED SCAFFOLD" when no lemmatizer library
is available.

SCOPE (honest, deliberately narrow). Regular English inflection only (3rd-person -s/-es/-ies, past -ed,
progressive -ing, a small irregular-verb table) plus a CONSERVATIVE plural-noun stripper, guarded against
common false positives (short words, double-letter endings like "grass"/"class"/"bus"). This is a HOST-SIDE
scaffold on the ladder to a learned morphological decomposition on the substrate itself (a spiking
morphology-segmentation circuit is the named next rung -- see the finding); it changes NO spiking
computation, only which STRING KEY a fact's action/noun token is stored/queried under. It is intentionally
scoped to the VERB axis for production wiring (see the finding's "scope note" -- broad NOUN lemmatization was
assessed and NOT wired into the live store, because it would silently rewrite the canonical identity of
already-established patients/agents such as "spikes"/"words"/"memory" used verbatim across dozens of existing
de-risks and tests); `lemma_noun` below is exposed for the compositional-chain-route's head-noun extraction
only, not for blanket noun-store canonicalization.
"""
from __future__ import annotations

# A small closed table of common irregular verbs whose past/other forms do not follow the regular suffix
# rules below. Extend as real conversational teaching surfaces a miss (each addition is a one-line, reviewable
# change -- the honest alternative to a silent wrong guess).
_IRREGULAR_VERBS = {
    "ate": "eat", "ran": "run", "went": "go", "saw": "see", "gave": "give", "made": "make",
    "took": "take", "came": "come", "did": "do", "had": "have", "was": "be", "were": "be", "is": "be",
    "are": "be", "am": "be", "said": "say", "got": "get", "knew": "know", "thought": "think",
}

_VOWELS = set("aeiou")


def _restore_silent_e(stem: str) -> str:
    """Porter-stemmer-style SHORT-WORD rule: a stem ending in Consonant-Vowel-Consonant (len>=3, final
    consonant not w/x/y) or bare Vowel-Consonant (len==2) most likely lost a silent 'e' to the -ed/-ing
    suffix that was just stripped ('chas'+e -> 'chase', 'stor'+e -> 'store', 'us'+e -> 'use'), while a stem
    ending in two consonants ('hunt', 'learn') did not need one. This is the SAME disambiguation the classic
    Porter stemmer uses for its analogous step (its own worked example is 'conflat(ed)' -> 'conflate'); it is
    not a dictionary check, so a genuine miss degrades to a stem that is still CONSISTENT between store and
    query (which is what recall needs), just not a real English word."""
    if len(stem) >= 3:
        c1, v, c2 = stem[-3], stem[-2], stem[-1]
        if c1 not in _VOWELS and v in _VOWELS and c2 not in _VOWELS and c2 not in "wxy":
            return stem + "e"
    elif len(stem) == 2:
        v, c2 = stem[0], stem[1]
        if v in _VOWELS and c2 not in _VOWELS and c2 not in "wxy":
            return stem + "e"
    return stem


def lemma_verb(v: str) -> str:
    """Canonicalize an inflected surface VERB to one stable key: hunts/hunt/hunted -> 'hunt'. Conservative
    suffix-stripping (regular English morphology) plus the small irregular table above; a word not matching
    any rule (including an already-base form) is returned unchanged, so a base-form verb is a no-op (byte-
    identical for every fact taught in base form, e.g. the tiny-demo's build-time seed facts)."""
    v = (v or "").lower().strip()
    if not v:
        return v
    if v in _IRREGULAR_VERBS:
        return _IRREGULAR_VERBS[v]
    if v.endswith("ies") and len(v) > 4:
        return v[:-3] + "y"                                   # carries -> carry, flies -> fly
    if v.endswith(("ches", "shes", "xes", "zes")) and len(v) > 5:
        return v[:-2]                                          # catches -> catch, washes -> wash, buzzes -> buzz
    if v.endswith("ing") and len(v) >= 5:
        stem = v[:-3]
        if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in _VOWELS:
            return stem[:-1]                                   # running -> run
        return _restore_silent_e(stem)                         # hunting -> hunt, chasing -> chase, using -> use
    if v.endswith("ed") and len(v) >= 4:
        stem = v[:-2]
        if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in _VOWELS:
            return stem[:-1]                                   # stopped -> stop
        return _restore_silent_e(stem)                         # hunted -> hunt, chased -> chase, used -> use
    if v.endswith("s") and not v.endswith("ss") and len(v) > 3:
        return v[:-1]                                          # hunts -> hunt, eats -> eat, chases -> chase
    return v


def lemma_noun(n: str) -> str:
    """Canonicalize an inflected surface NOUN's HEAD to one stable key (plural -s stripped), guarded against
    the common false positives a bare '-s' strip would corrupt ('grass'/'class'/'bus'/'this'). Used by the
    compositional-chain-route's head-noun extraction (task 2); NOT wired into the general store/query path
    (see the module docstring's scope note) -- calling it on an already-canonical / invariant noun is a no-op."""
    n = (n or "").lower().strip()
    if not n:
        return n
    if n.endswith("s") and not n.endswith(("ss", "us", "is", "os")) and len(n) > 3:
        return n[:-1]
    return n
