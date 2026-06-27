"""Tier 0.3 (wh-questions as a filler-gap dependency) PRODUCTION module.

Promoted from the GO de-risk (research/findings/2026-06-27-tier0.3-wh-questions-GO.md;
research/runners/_tier0_wh_questions_derisk.py, 6/6 seeds). Lets the console understand NATURAL wh-questions
("where does the boy go?", "who does the girl give to?", "what does the dog chase?") instead of the rigid
"what does X Y" probe syntax, by parsing the fronted wh-word as a FILLER and the verb's Tier-0.1 frame as which
argument slot is GAPPED.

A wh-question is mechanically a FILLER-GAP dependency (front-1 research C1,
`2026-06-27-conv-thinking-research-comprehension-representation.md`): the fronted wh-word ("where/what/who") is the
filler (held in WM -- the dlPFC NMDA latch / SAN-LAN syntactic working memory; Hagoort MUC Unification + on-line
memory), the verb's stored FRAME (Tier 0.1, `argstructure_composer.FRAME_ROLES`) says which ROLE is the gap, and
the wh-word selects which role to query. "where does the boy go?" -> wh=where -> the GOAL gap (go's frame licenses
GOAL) -> query_role(agent=boy, action=go, role=GOAL) -> "park".

The wh->role MAPPING (the filler-gap lexicon, `WH_ROLE_CANDIDATES`) maps each wh-word to an ORDERED list of
candidate roles; the verb's frame CONSTRAINS which one is gapped (where->GOAL for `go`, LOCATION for `put`). This
is the host-side lexical scaffold (the dictionary, like the parser's morphology) -- the COGNITION is the composer's
spiking query on the resolved role.

COMPOSER-AGNOSTIC by design (reuse-by-import, NO sim/ edit, NO composer edit):
  * On a Tier-0.1 `ArgStructureComposer` (has `query_role` + the typed-role frame lexicon): the FULL filler-gap
    path -- typed roles (GOAL/RECIPIENT/THEME/LOCATION/SOURCE/INSTRUMENT/TIME), verb-frame-constrained.
  * On a plain `RFPhasorComposer` (the deployed first-chat console; only agent/action/patient): GRACEFUL FALLBACK
    -- "what does X V" -> query_patient, "who V P" / "who does X V" (subject) -> query_agent. The natural wh
    surface form still works; only the typed obliques (GOAL etc.) are unavailable (they need the Tier-0.1 roles).

The no-confab moat is the composer's: an unanswerable wh (no matching stored fact) OR a frame-unlicensed wh (e.g.
"when does X go" when go's frame has no TIME slot) returns None -> the console abstains, never fabricates.
"""
from __future__ import annotations

import re

# Reuse the Tier-0.1 verb-frame lexicon (FRAME_ROLES: which roles each verb licenses) + the tense de-inflection.
from research.runners.argstructure_composer import FRAME_ROLES, TENSE_3SG


# ===================================================================================================================
# THE WH->ROLE MAPPING (the filler-gap lexicon). A fronted wh-word questions ONE thematic role; the verb's frame
# CONSTRAINS the ambiguous ones (where -> GOAL for `go`, LOCATION for `put`). Each wh maps to an ORDERED list of
# candidate roles; the resolver picks the FIRST one the verb's frame licenses.
# ===================================================================================================================
WH_ROLE_CANDIDATES = {
    "who": ["agent", "RECIPIENT"],          # default subject question -> the agent gap
    "who_to": ["RECIPIENT", "agent"],       # "who does X give TO" (a trailing to-PP) -> the recipient gap
    "what": ["patient", "THEME"],
    "where": ["GOAL", "LOCATION"],
    "when": ["TIME"],
    "whom": ["RECIPIENT"],
    "with": ["INSTRUMENT"],                 # "with what" -> INSTRUMENT
}
# multiword wh-cues that fix the role unambiguously (checked before the single-word map).
WH_MULTIWORD = {
    ("where", "from"): "SOURCE", ("from", "where"): "SOURCE",
    ("to", "whom"): "RECIPIENT", ("with", "what"): "INSTRUMENT",
}

# trailing prepositions that mark the gap SITE ("who does the girl give TO" -> RECIPIENT gap). Kept out of the
# optional-object `\w+` so a bare trailing prep is captured as `trailprep`, not swallowed as an object word.
_TRAIL_PREPS = "to|on|in|with|from|at|by"

# FORM 1 -- "WH [does|do|did] AGENT VERB (OBJ)? (PREP)?" -- the auxiliary wh-question (the common surface form).
_WH_AUX_RE = re.compile(
    r"^\s*(?P<wh>where\s+from|from\s+where|to\s+whom|with\s+what|who|whom|what|where|when)\b"
    r"\s+(?:does|do|did)\s+(?:the\s+|a\s+|an\s+)?(?P<agent>\w+)\s+(?P<verb>\w+)"
    r"(?:\s+(?:the\s+|a\s+|an\s+)?(?!(?:" + _TRAIL_PREPS + r")\b)\w+)?"
    r"(?:\s+(?P<trailprep>" + _TRAIL_PREPS + r"))?\s*\??\s*$",
    re.IGNORECASE,
)
# FORM 2 -- the BARE SUBJECT wh-question "WHO/WHAT VERB (the) OBJECT?" (no auxiliary; the wh-word IS the subject).
_WH_SUBJ_RE = re.compile(
    r"^\s*(?P<wh>who|what)\s+(?P<verb>\w+)\s+(?:the\s+|a\s+|an\s+)?(?P<patient>\w+)\s*\??\s*$",
    re.IGNORECASE,
)

_INV_TENSE = {v: k for k, v in TENSE_3SG.items()}     # de-inflect a surface 3sg verb (goes->go)

# the short natural-answer scaffold per role ('to the park' for GOAL, 'a ball' for patient/THEME).
_BARE_LEAD = {"GOAL": "to the", "RECIPIENT": "to the", "LOCATION": "on the", "SOURCE": "from the",
              "INSTRUMENT": "with the", "THEME": "the", "patient": "the", "agent": "the", "TIME": "at"}


def is_wh_question(text):
    """True iff `text` is a recognizable natural wh-question (either surface form)."""
    return bool(_WH_AUX_RE.search(text) or _WH_SUBJ_RE.search(text))


def _resolve_wh_role(wh_cue, verb, trailprep=None, role_map=None, frame_roles=None):
    """Map a parsed wh-cue + verb-frame to the GAPPED typed role. `role_map` (default WH_ROLE_CANDIDATES) is the
    wh->candidate-roles table; the PERMUTED-MAPPING anti-cheat passes a WRONG one. Returns the role string, or None
    if the verb's frame licenses none of the candidates (e.g. when->TIME but `go`'s frame has no TIME -> abstain).

    `frame_roles` (default None -> the module-level hand-authored FRAME_ROLES, byte-identical to the prior
    behaviour) is the per-verb {verb: [roles]} licensing map the resolver intersects the wh-candidates against.
    Pass a same-shaped dict (e.g. the CORPUS-MINED frame roles, B-mine-2) so the wh resolution consumes ACQUIRED
    frames -- and so the B-mine-2 permuted-mining control (a SCRAMBLED frame inventory -> a verb licenses the WRONG
    roles -> the wh-gap can't resolve -> abstain) actually bites."""
    fr = FRAME_ROLES if frame_roles is None else frame_roles
    role_map = WH_ROLE_CANDIDATES if role_map is None else role_map
    licensed = set(fr.get(verb, fr.get("_default", FRAME_ROLES["_default"])))
    wh_tokens = tuple(wh_cue.lower().strip().split())
    if wh_tokens in WH_MULTIWORD:                       # multiword cues (where-from, to-whom, with-what)
        r = WH_MULTIWORD[wh_tokens]
        return r if r in licensed else None
    head = wh_tokens[0]
    key = "who_to" if (head == "who" and trailprep) else head      # both go THROUGH role_map (anti-cheat-derangeable)
    for r in role_map.get(key, []):
        if r in licensed:
            return r
    return None


def parse_wh_question(text, role_map=None, frame_roles=None):
    """Parse a natural wh-question into a parse dict, or None if it isn't a wh-question.

    Returns {role, cue, agent, verb} where `cue` is the {role: filler} the query matches on (the KNOWN arguments)
    and `role` is the GAPPED role to read back. Two surface forms:
      * auxiliary ("where does the boy go?"): cue = {agent, action}, role = the wh-gapped oblique/patient;
      * bare subject ("who chase river?"): cue = {action, patient}, role = agent.
    `role` is "__UNLICENSED__" when the verb frame licenses none of the wh-word's candidates (-> abstain).
    `role_map` lets the anti-cheat inject a wrong table; `frame_roles` selects the per-verb licensing map (default
    None -> the hand FRAME_ROLES; pass the MINED frame roles for B-mine-2)."""
    m = _WH_AUX_RE.search(text)
    if m:
        wh = m.group("wh").lower()
        agent = m.group("agent").lower()
        verb = _INV_TENSE.get(m.group("verb").lower(), m.group("verb").lower())
        trailprep = (m.group("trailprep") or "").lower() or None
        role = _resolve_wh_role(wh, verb, trailprep=trailprep, role_map=role_map, frame_roles=frame_roles)
        cue = {"agent": agent, "action": verb}
        return {"role": role if role is not None else "__UNLICENSED__", "cue": cue, "agent": agent, "verb": verb,
                "form": "aux", "wh": wh}
    ms = _WH_SUBJ_RE.search(text)
    if ms:
        wh = ms.group("wh").lower()
        verb = _INV_TENSE.get(ms.group("verb").lower(), ms.group("verb").lower())
        patient = ms.group("patient").lower()
        candidates = (role_map or WH_ROLE_CANDIDATES).get(wh, [])
        role = next((r for r in candidates if r == "agent"), None)   # the subject gap is the agent
        return {"role": "agent" if role else "__UNLICENSED__", "cue": {"action": verb, "patient": patient},
                "agent": None, "verb": verb, "form": "subj", "wh": wh}
    return None


def _query_composer(comp, role, cue):
    """Query `role` from a composer given the cue {role: filler}. Uses the Tier-0.1 `query_role` when available
    (typed roles); else falls back to the plain RFPhasorComposer who/what API (agent/action/patient only).

    Returns the filler or None (abstain). Falls back gracefully so the SAME wh route works on the deployed
    first-chat `RFPhasorComposer` (no typed roles) AND a Tier-0.1 `ArgStructureComposer`."""
    if hasattr(comp, "query_role"):                     # Tier-0.1 ArgStructureComposer -> full typed-role path
        return comp.query_role(role, **cue)
    # plain RFPhasorComposer fallback: only agent/action/patient roles exist.
    if role == "patient" and "agent" in cue and "action" in cue:
        return comp.query_patient(cue["agent"], cue["action"])
    if role == "agent" and "action" in cue and "patient" in cue:
        return comp.query_agent(cue["action"], cue["patient"])
    if role == "agent" and "action" in cue and "patient" not in cue:
        # "who does X V?" auxiliary subject question with no object cue -- not answerable on a 2-arg composer.
        return None
    return None                                         # a typed oblique on a plain composer -> abstain (no role)


def answer_wh(comp, text, role_map=None, frame_roles=None):
    """The full wh-question route: parse -> resolve the gapped role -> query the composer -> (filler, role, parse).
    `filler` is None on abstain (the no-confab moat: an unanswerable/unstored/unlicensed wh returns None). `parse`
    is the parse dict (for downstream render), or None if `text` is not a wh-question. `role_map` / `frame_roles`
    select the (mined-or-hand) wh-candidate table + per-verb licensing map (default None = the hand scaffold)."""
    parse = parse_wh_question(text, role_map=role_map, frame_roles=frame_roles)
    if parse is None:
        return None, None, None
    if parse["role"] == "__UNLICENSED__":
        return None, None, parse                        # the verb frame licenses no such role -> abstain (moat)
    filler = _query_composer(comp, parse["role"], parse["cue"])
    return filler, parse["role"], parse


def bare_answer(role, filler):
    """The short natural answer to a wh-question ('to the park' for GOAL, 'the ball' for patient/THEME)."""
    if filler is None:
        return None
    return f"{_BARE_LEAD.get(role, 'the')} {filler}".strip()
