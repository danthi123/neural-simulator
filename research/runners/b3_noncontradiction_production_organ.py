"""NON-CONTRADICTION ASSERTION-GATE wired into the PRODUCTION conversational turn (Gate-B, B3, 2026-08-12).

The owner's "the brain won't accept a claim that contradicts what it already holds": when a user ASSERTS a
transitive fact with a POLARITY ("the dog eats grass" = AFFIRM) that CONTRADICTS the brain's stored polarity for
that EXACT SAME SVO ("a dog does NOT eat grass" = NEGATE), the brain REJECTS the assertion instead of silently
overwriting a belief it holds. The load-bearing recall of the stored polarity is a genuinely-SPIKING read on the
PRODUCTION one-brain composer (`OneBrainComposer.ask_yes_no` -> `_read_blocks`/`_decode_batched_mem` -> `_select`
-> `_spiking_select`, a WTA read of `cp_firing_states` over the resonator cleanup membrane `cp_membrane_potential_v`;
`enable_spiking_cleanup=True` is the production composer default). The gate proper is the ONE host boolean
(`stored_polarity != asserted_polarity`) the project already accepts as the no-confab moat (the brain-GENERATION
path's `_contradicts == (ask_yes_no == "no")`). This adds that same gate on the USER-ASSERTION path, which had none.

It REUSES (does not reinvent) the adversarially-verified B3 de-risk
(`research/runners/_burndown_B3_onebrain_negation_moat_derisk.py`, 6-seed GO 42/43/44/100/101/102 D=256 numpy-CPU;
verdict status GO, 10/10 preconditions via tools.verdict.Verdict). That de-risk proved, on the onebrain substrate:
  * INTACT: recall neg/aff = 1.0/1.0; 6/6 contradictions REJECTED (0 false-accepts); 0 over-blocks (consistent
    restatements + novel assertions ACCEPTED); the canonical "dog !eat grass" recalls "no" on the substrate.
  * LESION (disable negation storage -> store all AFFIRM): 18 false-accepts total (0 -> 18); the canonical
    negation genuinely reads "yes" on every seed -> the negation is really gone -> LOAD-BEARING, not a bug.
  * ANTI-CHEAT no-store: rejections collapse to 0 (rejection is store-driven, not a fixed template).
  * ANTI-CHEAT shuffle: the reject set tracks the permuted store (not a memorized answer).
This organ imports that runner's gate logic (`_assert_gate`, `FLIP`) directly -- NO gate reimplementation.

BRAIN-BASED: the load-bearing element is `composer.ask_yes_no`, whose polarity winner is a `cp_firing_states`
read (`_spiking_select`, enable_spiking_cleanup=True by default on the production composer), NOT a host formula.
The host boundary is exactly the thin glue the project already treats as legitimate: (1) the ONE moat boolean
`stored != asserted`; (2) negation DETECTION on the input surface (parsing "not"/"n't"); (3) a minimal verb-LEMMA
fallback so a natural inflected assertion ("eats") recalls the stored lemma ("eat"). (2)+(3) are the declared,
mapped upstream residuals (see HONEST RESIDUALS) -- the same class of host front-end the composer already assumes
(it takes the polarity tag as an argument, and the sibling surprise organ defers inflection to "D4 lemmatization").

MOAT-SAFE + NON-REGRESSIVE by CONSTRUCTION:
  * ABSTAIN-PRESERVING: `ask_yes_no` returns "unknown" for any SVO the brain does not hold (no matching stored
    fact, or a different stored patient) -> the gate ACCEPTS (it never fabricates a belief to reject against).
    The no-confab moat is inverted, not weakened: it refuses a rejection it cannot justify from a stored belief.
  * SCOPE: fires ONLY on a 3-content-token transitive ASSERTION whose polarity the detector resolved. Questions
    (WH / "?"), non-assertions, self/identity turns, anaphora and open-ended prompts are OUT OF SCOPE -> the turn
    is byte-identical, unchanged.
  * COMPOSES with D2 SURPRISE (patient-mismatch) with ZERO overlap: `ask_yes_no` returns "unknown" unless the
    asserted PATIENT matches the stored one, so B3 fires ONLY on same-SVO / opposite-polarity; a different-patient
    assertion is "unknown" here (accepted) and handled by surprise. B3 rejects -> the turn returns before the
    acquire/store overwrites the held belief.

LESION-LOAD-BEARING (`BRAIN_NONCONTRADICTION_LESION=1`): the organ bypasses the spiking polarity recall (treats
every recall as "unknown"), so contradicting assertions slip through -> the gate goes INERT (rejections -> 0).
This isolates the SPIKING recall as the cause of the rejection. The verify harness ALSO reconfirms the de-risk's
STORAGE lesion (store all AFFIRM -> the substrate reads "yes" -> 0->N false-accepts) to tie the organ to the
6-seed GO's own load-bearing lesion.

HONEST RESIDUALS (declared):
  * NEGATION DETECTION is host, upstream (`detect_polarity` parses "not"/"n't"/"never"/"no"). A learned spiking
    polarity classifier on the input is the named next rung; the composer already RECALLS polarity on the
    substrate -- only the INPUT tagging is host.
  * VERB MORPHOLOGY is host, upstream + MINIMAL. `_action_lemma_candidates` tries the surface action first, then a
    single trailing-"s"/"es"/"ies" strip (a fallback used ONLY when the surface form does not recall), so a natural
    "the dog eats grass" hits the stored lemma "dog eat grass". Irregulars ("goes", "does") need the shared D4
    lemmatizer (the same residual the surprise organ declares). Surface-first means a base form ending in "s"
    (stored as-is) still recalls on its surface form -> the fallback never corrupts a hit.
  * STORE-SIDE: for the gate to fire, heard negations must be STORED with polarity=NEGATE. The production acquire
    path today hard-codes polarity="AFFIRM" (brain_chat_tui `_maybe_acquire`); the wiring must pass
    `detect_polarity(text)` there too (and the SAME lemma normalization, so stored + recalled agree). This organ
    EXPOSES `detect_polarity` + `extract_polar_assertion` for exactly that use. (Wiring spec.)
  * CO-RESIDENT: none added -- B3 reads the ONE production recall composer directly (no separate bridge), so it
    needs no one-brain merge. This is strictly simpler than the affect/surprise/comprehension co-resident organs.
  * The gate boolean (`stored != asserted`) is host, the accepted moat pattern (identical to `_contradicts`).

Additive, default-ON, `BRAIN_NONCONTRADICTION_GATE=0` -> the byte-identical oracle (fully skipped). NO `sim/`
edit; reuse-by-import; uses the process backend (cupy in production, numpy in tests).
"""
from __future__ import annotations

import os
import re

# Reuse the de-risked gate logic + fixtures directly (NO reimplementation).
from research.runners._burndown_B3_onebrain_negation_moat_derisk import (
    _assert_gate,   # (comp, agent, action, patient, asserted_polarity) -> ("accept"|"reject", recalled_yn)
    FLIP,           # {"AFFIRM": "NEGATE", "NEGATE": "AFFIRM"}
)

# Negation cues on the INPUT surface (the declared host upstream residual). A token match OR an apostrophe-n't
# suffix flips the asserted polarity to NEGATE. Deliberately small + explicit (a learned classifier is the next
# rung); double negation is out of scope (a single cue -> NEGATE).
_NEG_TOKENS = {"not", "never", "no", "cannot", "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent"}
_NT_SUFFIX = re.compile(r"n't\b")

# Function words stripped to expose the transitive content (agent verb patient). "not"/negation cues are removed
# by the polarity detector FIRST, so a negated declarative still reduces to 3 content tokens.
_FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "and", "or", "that",
    "this", "these", "those", "it", "its", "they", "them", "he", "she", "his", "her", "their",
    "my", "your", "our", "i", "you", "we", "me", "us", "him", "on", "in", "at", "by", "with",
    "for", "as", "so", "then", "now", "just", "please", "does", "do", "did",
}
_WH = {"what", "who", "whom", "whose", "where", "when", "why", "how", "which"}
_WORD_RE = re.compile(r"[a-zA-Z']+")


def noncontradiction_enabled() -> bool:
    """Default-ON. `BRAIN_NONCONTRADICTION_GATE` in {0,false,no,off} -> the byte-identical oracle (fully disabled)."""
    v = os.environ.get("BRAIN_NONCONTRADICTION_GATE")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def noncontradiction_lesioned() -> bool:
    """`BRAIN_NONCONTRADICTION_LESION` in {1,true,yes,on} -> bypass the spiking polarity recall (load-bearing lesion:
    the gate can no longer see the stored polarity, so contradictions slip through -> the gate goes inert)."""
    v = os.environ.get("BRAIN_NONCONTRADICTION_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def detect_polarity(text: str) -> str:
    """Host negation DETECTOR (the declared upstream residual): 'NEGATE' if a negation cue is present, else
    'AFFIRM'. A learned spiking polarity classifier is the next rung; the composer already RECALLS polarity on the
    substrate -- only this INPUT tagging is host. Also used STORE-side so heard negations store as NEGATE."""
    t = (text or "").lower()
    if _NT_SUFFIX.search(t):
        return "NEGATE"
    toks = [w for w in _WORD_RE.findall(t)]
    return "NEGATE" if any(w in _NEG_TOKENS for w in toks) else "AFFIRM"


def _action_lemma_candidates(action: str):
    """Minimal DECLARED verb-morphology fallback (host upstream residual). Yields the SURFACE action first (so a
    base form stored as-is always recalls on itself), then a single de-inflected candidate for the common English
    3rd-person-singular present ("eats"->"eat", "chases"->"chase", "flies"->"fly"). Irregulars ("goes","does")
    need the shared D4 lemmatizer -- named, not smuggled. Never yields a candidate shorter than 2 chars."""
    a = (action or "").lower()
    out = [a]
    cand = None
    if a.endswith("ies") and len(a) > 4:
        cand = a[:-3] + "y"                         # "flies" -> "fly"
    elif a.endswith("s") and not a.endswith("ss") and len(a) > 2:
        cand = a[:-1]                               # "eats"->"eat", "chases"->"chase", "sees"->"see"
    if cand and len(cand) >= 2 and cand not in out:
        out.append(cand)
    return out


def extract_polar_assertion(text: str):
    """Return (agent, action, patient, polarity) when `text` is a 3-content-token transitive ASSERTION (after
    stripping negation cues + function words), else None. A WH-question / non-assertion (patient is the query ->
    <3 content tokens, or a '?') is OUT OF SCOPE -> None. The polarity is `detect_polarity(text)`. The action is
    the SURFACE token (the lemma fallback is applied at recall time in `check`)."""
    raw = text or ""
    if "?" in raw:
        return None
    toks = [w.lower() for w in _WORD_RE.findall(raw)]
    if any(t in _WH for t in toks):
        return None
    polarity = detect_polarity(raw)
    # strip negation cues AND function words to expose the SVO content
    content = [t for t in toks if t not in _FUNCTION_WORDS and t not in _NEG_TOKENS]
    content = [t for t in content if not _NT_SUFFIX.search(t + " ")]  # drop any residual n't token
    if len(content) != 3:
        return None
    a, v, p = content
    if a == v or v == p:                                   # degenerate
        return None
    return a, v, p, polarity


class _RecallShim:
    """Adapts a plain recall callable `(agent, action, patient) -> 'yes'|'no'|'unknown'` into the `.ask_yes_no`
    interface `_assert_gate` expects, so the de-risked gate logic is reused verbatim against the PRODUCTION recall
    (`chat.inner.is_it_true` == `composer.ask_yes_no`) or a bare `OneBrainComposer`. A lesion forces every recall
    to 'unknown' (the load-bearing spiking recall is gone -> the gate goes inert)."""

    def __init__(self, recall, lesion=False):
        self._recall = recall
        self._lesion = bool(lesion)

    def ask_yes_no(self, agent, action, patient):
        if self._lesion:
            return "unknown"
        return self._recall(agent, action, patient)


class NonContradictionProductionOrgan:
    """The non-contradiction assertion-gate. Stateless: it holds no substrate of its own -- the load-bearing
    spiking polarity recall lives on the PRODUCTION composer, read through the caller-supplied `recall` callable
    (`chat.inner.is_it_true` == `composer.ask_yes_no`). This is the one-brain design: B3 reads the REAL stored
    beliefs on the REAL substrate, adding no co-resident bridge."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)

    def check(self, recall, text: str, lesion: bool = False) -> dict | None:
        """Gate an incoming assertion `text` against the brain's stored polarity via `recall`. Returns None when
        the input is OUT OF SCOPE (not a competent 3-token transitive assertion) -> the caller leaves the turn
        byte-identical. Otherwise a dict: the decoded SVO + asserted/recalled polarity + `reject` (True iff the
        assertion contradicts a stored belief). `recall(agent, action, patient) -> 'yes'|'no'|'unknown'`.

        SURFACE-FIRST, LEMMA-FALLBACK recall: try the surface action; only if the substrate returns 'unknown' fall
        back to the minimal de-inflected action candidate(s). The fallback never corrupts a genuine surface hit."""
        parsed = extract_polar_assertion(text)
        if parsed is None:
            return None
        a, v, p, asserted = parsed
        shim = _RecallShim(recall, lesion=lesion)

        decision, recalled_yn, used_action = "accept", "unknown", v
        for cand in _action_lemma_candidates(v):
            d, yn = _assert_gate(shim, a, cand, p, asserted)   # REUSED de-risk gate
            if yn != "unknown":
                decision, recalled_yn, used_action = d, yn, cand
                break
        stored = None if recalled_yn == "unknown" else ("AFFIRM" if recalled_yn == "yes" else "NEGATE")
        return {
            "on": True, "lesioned": bool(lesion), "in_scope": True,
            "svo": [a, v, p], "recall_action": used_action,
            "asserted_polarity": asserted, "recalled_yn": recalled_yn,
            "stored_polarity": stored, "reject": bool(decision == "reject"),
        }


_ORGAN: NonContradictionProductionOrgan | None = None


def get_organ(seed: int = 42) -> NonContradictionProductionOrgan:
    """The process-shared non-contradiction organ (built once on first use; stateless)."""
    global _ORGAN
    if _ORGAN is None:
        _ORGAN = NonContradictionProductionOrgan(seed=seed)
    return _ORGAN


def rejection_message(svo=None, stored_polarity=None) -> str:
    """The honest functional NOTICE surfaced when the gate REJECTS a contradicting assertion. A FUNCTIONAL read of
    the brain's stored polarity -- never a phenomenal claim."""
    if svo and len(svo) == 3:
        a, v, p = svo
        held = f"{a} does not {v} {p}" if stored_polarity == "NEGATE" else f"{a} {v} {p}"
        return (f"That contradicts what I hold: my polarity recall says {held}. "
                f"I won't accept the opposite without more.")
    return "That contradicts a belief I already hold, so I won't accept it as stated."
