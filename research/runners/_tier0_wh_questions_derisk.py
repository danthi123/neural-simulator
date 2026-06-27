#!/usr/bin/env python
"""STEP 1 cheap-first DE-RISK for Tier 0.3 (wh-questions as a filler-gap dependency).

The HARD GATE before the full build (research/findings/2026-06-27-conversation-thinking-ROADMAP.md, Tier 0.3;
front-1 research C1, `2026-06-27-conv-thinking-research-comprehension-representation.md`):

  Can a NATURAL wh-question -- "where does the boy go?" -- be parsed (the fronted wh-word = the FILLER, the verb's
  frame = which role is GAPPED) into a typed-role QUERY against the Tier-0.1 ArgStructureComposer, rendered via the
  0.1 verb-frame render, WITHOUT breaking the no-confab moat?

Concretely (per the prompt):
  * store (boy, go, GOAL=park) via the ArgStructureComposer (Tier 0.1);
  * ask "where does the boy go?" -> parse: wh=where -> the GOAL role (the verb go's frame licenses GOAL, not
    LOCATION/patient); verb=go; agent=boy -> query_role("GOAL", agent="boy", action="go") -> "park";
  * render via the 0.1 frame render -> "the boy goes to the park" (and the bare-answer "to the park");
  * MOAT: an unanswerable wh -- e.g. "where does the boy give?" (no GOAL stored for boy+give), an unstored agent
    ("where does the cat go?") -- ABSTAINS (returns None; 0 false-accepts);
  * the rendered answer re-parses (VERIFY) to the stored typed fact;
  * the LOAD-BEARING PERMUTED-MAPPING anti-cheat: a WRONG wh->role table (where->patient instead of GOAL) must
    give a wrong/abstaining answer -- proving the wh->role MAPPING carries the meaning, not coincidence.

This is a PROBE: it composes the PRODUCTION ArgStructureComposer (research/runners/argstructure_composer.py, the
Tier-0.1 build) by reuse-by-import, adding ONLY a wh-question parser (host parse of the surface form -> the
filler-gap role) and a query/render route. NO sim/ edit, NO composer edit yet (Step 2 only if this is GO). Tiny
vocab + numpy (CPU).

Biology: wh-question = a filler-gap dependency (front-1 C1) -- the fronted wh-word is the filler held in WM
(dlPFC NMDA latch / SAN-LAN syntactic working memory; Hagoort MUC Unification + on-line memory), the verb's frame
(MUC-Memory, the Tier-0.1 lexicon) says which argument slot is GAPPED, and the wh-word selects which role to query.

Run:  SIM_BACKEND=numpy python -u -m research.runners._tier0_wh_questions_derisk
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.argstructure_composer import (  # noqa: E402
    ArgStructureComposer, FRAME_ROLES, frame_for, reparse_to_fact, TENSE_3SG, FUNCTION_WORDS,
)


# ===================================================================================================================
# THE WH->ROLE MAPPING (the filler-gap lexicon). A fronted wh-word questions ONE thematic role; the verb's frame
# (Tier 0.1) CONSTRAINS the ambiguous ones (where -> GOAL for `go`, LOCATION for `put`). This is the host-side
# lexical scaffold (the dictionary, like the parser's morphology) -- the COGNITION is the composer's spiking
# query_role on the resolved role.
#   who   -> agent (subject question) | RECIPIENT (in "who does X give to")
#   what  -> patient | THEME (whichever the verb licenses)
#   where -> GOAL | LOCATION (whichever the verb licenses)
#   when  -> TIME
#   where-from / from-where -> SOURCE
#   with-what -> INSTRUMENT
#   whom / to-whom -> RECIPIENT
# ===================================================================================================================
# Each wh-word maps to an ORDERED list of candidate roles; the resolver picks the FIRST one the verb's frame
# licenses (FRAME_ROLES from the Tier-0.1 lexicon). This is the "verb-frame says which role is gapped" rule.
WH_ROLE_CANDIDATES = {
    "who": ["agent", "RECIPIENT"],          # default subject question -> the agent gap
    "who_to": ["RECIPIENT", "agent"],       # "who does X give TO" (a trailing to-PP) -> the recipient gap
    "what": ["patient", "THEME"],
    "where": ["GOAL", "LOCATION"],
    "when": ["TIME"],
    "whom": ["RECIPIENT"],
    "with": ["INSTRUMENT"],                 # "with what" -> INSTRUMENT (the trailing "what" is consumed by the regex)
}
# multiword wh-cues that fix the role unambiguously (checked before the single-word map).
WH_MULTIWORD = {
    ("where", "from"): "SOURCE", ("from", "where"): "SOURCE",
    ("to", "whom"): "RECIPIENT", ("with", "what"): "INSTRUMENT",
}

# The set of trailing prepositions that mark the gap SITE in a wh-question ("who does the girl give TO" -> the
# RECIPIENT gap). Kept out of the optional-object `\w+` so a bare trailing prep is captured as `trailprep`, not
# swallowed as an object word.
_TRAIL_PREPS = "to|on|in|with|from|at|by"

# FORM 1 -- "WH [does|do|did] AGENT VERB (OBJ)? (PREP)?" -- the auxiliary wh-question (the common surface form).
# An optional non-prep object word may sit before the optional trailing preposition (the gap-site marker). The
# object alternative is `(?!PREP)\w+` so "to" in "give to" is NOT consumed as the object.
_WH_AUX_RE = re.compile(
    r"^\s*(?P<wh>where\s+from|from\s+where|to\s+whom|with\s+what|who|whom|what|where|when)\b"
    r"\s+(?:does|do|did)\s+(?:the\s+|a\s+|an\s+)?(?P<agent>\w+)\s+(?P<verb>\w+)"
    r"(?:\s+(?:the\s+|a\s+|an\s+)?(?!(?:" + _TRAIL_PREPS + r")\b)\w+)?"
    r"(?:\s+(?P<trailprep>" + _TRAIL_PREPS + r"))?\s*\??\s*$",
    re.IGNORECASE,
)

# FORM 2 -- the BARE SUBJECT wh-question "WHO VERB (the) OBJECT?" (no auxiliary; the wh-word IS the subject, e.g.
# "who chase river?"). Only who/what front a subject question; the gapped role is the AGENT (subject).
_WH_SUBJ_RE = re.compile(
    r"^\s*(?P<wh>who|what)\s+(?P<verb>\w+)\s+(?:the\s+|a\s+|an\s+)?(?P<patient>\w+)\s*\??\s*$",
    re.IGNORECASE,
)

# de-inflect a surface 3sg verb back to its base (goes->go) so "where does the boy goes" still cues (robustness).
_INV_TENSE = {v: k for k, v in TENSE_3SG.items()}


def _resolve_wh_role(wh_cue, verb, trailprep=None, role_map=None):
    """Map a parsed wh-cue + verb-frame to the GAPPED typed role (the filler-gap resolution).

    `role_map` (default = WH_ROLE_CANDIDATES) is the wh->candidate-roles table; the PERMUTED-MAPPING anti-cheat
    passes a WRONG table here. Returns the role string, or None if the verb's frame licenses none of the candidates
    (e.g. "when does X go" -> TIME, but `go`'s frame has no TIME slot -> abstain by construction)."""
    role_map = WH_ROLE_CANDIDATES if role_map is None else role_map
    licensed = set(FRAME_ROLES.get(verb, FRAME_ROLES["_default"]))
    wh_cue = wh_cue.lower().strip()
    wh_tokens = tuple(wh_cue.split())
    # multiword wh cues (where-from, to-whom, with-what) fix the role; still must be frame-licensed.
    if wh_tokens in WH_MULTIWORD:
        r = WH_MULTIWORD[wh_tokens]
        return r if r in licensed else None
    # "who ... <trailing prep>" -> the recipient gap (the to-PP) under the `who_to` key, else the subject `who`
    # key. BOTH go THROUGH role_map so the permuted-mapping anti-cheat can derange them (no hard-wired override --
    # the wh->role MAPPING is what carries the meaning, including the to-PP recipient resolution).
    head = wh_tokens[0]
    key = "who_to" if (head == "who" and trailprep) else head
    candidates = role_map.get(key, [])
    for r in candidates:
        if r in licensed:
            return r
    return None


def parse_wh_question(text, role_map=None):
    """Parse a natural wh-question into a parse dict, or None if it isn't a wh-question. The verb's Tier-0.1 frame
    resolves which role the wh-word gaps. `role_map` lets the anti-cheat inject a wrong table.

    Returns a dict {role, cue} where `cue` is the {role: filler} the query matches on (the KNOWN arguments) and
    `role` is the GAPPED role to read back. Two surface forms:
      * auxiliary form ("where does the boy go?"): cue = {agent}, role = the wh-gapped oblique/patient;
      * bare subject form ("who chase river?"): cue = {action, patient}, role = agent (the subject is the gap).
    `role` is "__UNLICENSED__" when the verb frame licenses none of the wh-word's candidate roles (-> abstain)."""
    m = _WH_AUX_RE.search(text)
    if m:
        wh = m.group("wh").lower()
        agent = m.group("agent").lower()
        verb = _INV_TENSE.get(m.group("verb").lower(), m.group("verb").lower())
        trailprep = (m.group("trailprep") or "").lower() or None
        role = _resolve_wh_role(wh, verb, trailprep=trailprep, role_map=role_map)
        if role is None:
            return {"role": "__UNLICENSED__", "cue": {"agent": agent, "action": verb}, "agent": agent, "verb": verb}
        return {"role": role, "cue": {"agent": agent, "action": verb}, "agent": agent, "verb": verb}
    # bare subject question: WHO/WHAT VERB OBJECT  (the wh-word is the subject -> the AGENT gap)
    ms = _WH_SUBJ_RE.search(text)
    if ms:
        wh = ms.group("wh").lower()
        verb = _INV_TENSE.get(ms.group("verb").lower(), ms.group("verb").lower())
        patient = ms.group("patient").lower()
        # the candidate-role table still gates the subject question (who->agent first); a PERMUTED table that maps
        # who to a non-agent role makes this abstain/mis-answer -- so the anti-cheat bites here too.
        candidates = (role_map or WH_ROLE_CANDIDATES).get(wh, [])
        role = next((r for r in candidates if r in ("agent",)), None)
        if role is None:
            return {"role": "__UNLICENSED__", "cue": {"action": verb, "patient": patient}, "agent": None,
                    "verb": verb}
        return {"role": "agent", "cue": {"action": verb, "patient": patient}, "agent": None, "verb": verb}
    return None


def answer_wh(comp, text, role_map=None):
    """The full wh-question route: parse -> resolve the gapped role -> query_role on the composer -> (filler, role).
    Returns (filler, role, parse) where filler is None on abstain (the no-confab moat: an unanswerable/unstored wh
    returns None). `parse` is the parse dict for downstream render, or None if not a wh-question."""
    parse = parse_wh_question(text, role_map=role_map)
    if parse is None:
        return None, None, None
    if parse["role"] == "__UNLICENSED__":
        return None, None, parse                            # frame licenses no such role -> abstain (moat)
    filler = comp.query_role(parse["role"], **parse["cue"])
    return filler, parse["role"], parse


def _bare_answer(role, filler):
    """The short natural answer to a wh-question ('to the park' for a GOAL, 'a ball' for a patient/THEME)."""
    lead = {"GOAL": "to the", "RECIPIENT": "to the", "LOCATION": "on the", "SOURCE": "from the",
            "INSTRUMENT": "with the", "THEME": "the", "patient": "the", "agent": "the", "TIME": "at"}
    return f"{lead.get(role, 'the')} {filler}".strip()


# ===================================================================================================================
# THE DE-RISK
# ===================================================================================================================
def run_seed(seed, D=64, verbose=True):
    vocab = ["boy", "girl", "dog", "cat", "go", "give", "put", "chase", "eat",
             "park", "house", "ball", "bone", "table", "shelf", "river"]
    comp = ArgStructureComposer(seed=seed, D=D, vocab=vocab)
    facts = [
        {"agent": "boy", "action": "go", "GOAL": "park"},
        {"agent": "girl", "action": "give", "THEME": "ball", "RECIPIENT": "dog"},
        {"agent": "dog", "action": "put", "THEME": "bone", "LOCATION": "table"},
        {"agent": "cat", "action": "chase", "patient": "river"},          # default transitive (bare patient)
    ]
    for f in facts:
        comp.store_fact(f)

    results = {"seed": seed}

    # --- (1) wh-question comprehension + answer: the natural questions over the stored arg-structure facts ---
    wh_cases = [
        ("where does the boy go?",            "GOAL",      "park"),       # the headline case
        ("what does the girl give?",          "THEME",     "ball"),       # what -> THEME (give licenses THEME)
        ("who does the girl give to?",        "RECIPIENT", "dog"),        # who + to-gap -> RECIPIENT
        ("where does the dog put?",           "LOCATION",  "table"),      # where -> LOCATION (put licenses LOCATION)
        ("what does the dog put?",            "THEME",     "bone"),       # what -> THEME (put licenses THEME)
        ("what does the cat chase?",          "patient",   "river"),      # what -> patient (default transitive)
        ("who chase river?",                  "agent",     "cat"),        # who -> agent (subject question, no does)
    ]
    wh_ok = []
    for q, exp_role, exp_filler in wh_cases:
        filler, role, parse = answer_wh(comp, q)
        ok = (role == exp_role and filler == exp_filler)
        wh_ok.append((q, ok, role, filler, exp_role, exp_filler))
    # the "who chase river?" subject question has no "does"; support a bare-form regex variant too.
    n_wh = sum(1 for _, ok, *_ in wh_ok if ok)
    results["wh"] = {"n_ok": n_wh, "n_total": len(wh_ok),
                     "detail": [(q, bool(ok), str(role), str(f), er, str(ef)) for q, ok, role, f, er, ef in wh_ok]}
    if verbose:
        print(f"  [seed {seed}] WH-ANSWER {n_wh}/{len(wh_ok)}:", flush=True)
        for q, ok, role, f, er, ef in wh_ok:
            print(f"      {'OK ' if ok else 'XX '} \"{q}\" -> role={role} filler={f}  (want {er}={ef})", flush=True)

    # --- (2) RENDER the wh-answer fluently via the Tier-0.1 frame render + the bare natural answer ---
    # "where does the boy go?" -> render the whole fact + the bare answer.
    filler, role, parse = answer_wh(comp, "where does the boy go?")
    boy_fact, boy_comp = comp.kb[0]
    full_render = comp.render(boy_fact, boy_comp)
    bare = _bare_answer(role, filler) if filler else None
    results["render_where_boy_go"] = {"full": full_render, "bare": bare, "filler": str(filler), "role": str(role)}
    full_ok = (full_render == "the boy goes to the park")
    bare_ok = (bare == "to the park")
    if verbose:
        print(f"  [seed {seed}] RENDER \"where does the boy go?\" -> full=\"{full_render}\"  bare=\"{bare}\"  "
              f"({'OK' if full_ok and bare_ok else 'MISMATCH'})", flush=True)

    # --- (3) MOAT: an unanswerable / unstored wh -> ABSTAIN (None); 0 false-accepts ---
    moat_cases = [
        ("where does the boy go?",   "park"),       # stored -> answers (positive control)
        ("where does the boy give?", None),          # boy+give has no GOAL stored -> abstain
        ("where does the cat go?",   None),          # cat+go not stored -> abstain
        ("what does the boy give?",  None),          # boy+give not stored (girl gives) -> abstain
        ("who does the dog give to?", None),         # dog+give not stored -> abstain
        ("when does the boy go?",    None),          # go's frame has no TIME slot -> abstain (unlicensed)
        ("where does the cat chase?", None),         # chase (default) has no GOAL/LOCATION -> abstain (unlicensed)
    ]
    moat_detail, false_accepts = [], 0
    for q, exp in moat_cases:
        filler, role, parse = answer_wh(comp, q)
        if exp is None and filler is not None:
            false_accepts += 1
        moat_detail.append((q, str(filler), str(exp), bool((filler == exp) if exp else (filler is None))))
    moat_recall_ok = (answer_wh(comp, "where does the boy go?")[0] == "park")
    n_abstain = sum(1 for _, exp in moat_cases if exp is None)
    abstain_ok = sum(1 for q, exp in moat_cases if exp is None and answer_wh(comp, q)[0] is None)
    results["moat"] = {"false_accepts": int(false_accepts), "recall_ok": bool(moat_recall_ok),
                       "abstain_ok": int(abstain_ok), "n_abstain": int(n_abstain), "detail": moat_detail}
    if verbose:
        print(f"  [seed {seed}] MOAT: recall_ok={moat_recall_ok}, abstain {abstain_ok}/{n_abstain}, "
              f"false_accepts={false_accepts}", flush=True)

    # --- (3b) VERIFY: the rendered wh-answer re-parses to the stored typed fact ---
    reparse_ok = reparse_to_fact(full_render, boy_fact)
    results["verify_reparse"] = bool(reparse_ok)
    if verbose:
        print(f"  [seed {seed}] VERIFY re-parse of rendered answer -> stored fact: "
              f"{'OK' if reparse_ok else 'FAIL'}", flush=True)

    # --- (4) PERMUTED-MAPPING anti-cheat (LOAD-BEARING): a WRONG wh->role table must give a wrong/abstaining answer.
    # We rotate the candidate roles so where->patient, what->GOAL, who->THEME (a deranged mapping). With the
    # TRUE mapping "where does the boy go?" -> GOAL -> park. With the wrong mapping, "where" -> patient, but go's
    # frame has no patient slot -> abstain (None) [a wrong-but-safe answer]; and where the wrong role IS licensed it
    # must return the WRONG filler. The control PASSES iff the wrong mapping does NOT reproduce the true answers.
    PERMUTED = {
        "who": ["THEME", "patient"],           # who -> THEME/patient (wrong)
        "who_to": ["THEME", "patient"],        # who+to -> THEME/patient (wrong; not RECIPIENT)
        "what": ["GOAL", "LOCATION"],          # what -> GOAL/LOCATION (wrong)
        "where": ["patient", "THEME"],         # where -> patient/THEME (wrong)
        "when": ["agent"], "whom": ["agent"], "with": ["agent"],
    }
    permuted_detail, permuted_matches_true = [], 0
    for q, _exp_role, exp_filler in wh_cases:
        true_filler, _tr, _tp = answer_wh(comp, q)                      # the TRUE-mapping answer
        wrong_filler, wrong_role, _wp = answer_wh(comp, q, role_map=PERMUTED)   # the WRONG-mapping answer
        # the control FAILS for this case if the wrong mapping reproduces the true (correct, non-None) filler.
        reproduced = (true_filler is not None and wrong_filler == true_filler)
        if reproduced:
            permuted_matches_true += 1
        permuted_detail.append((q, str(true_filler), str(wrong_filler), str(wrong_role), bool(reproduced)))
    permuted_ok = (permuted_matches_true == 0)                          # NO true answer reproduced under the wrong map
    results["permuted_mapping"] = {"matches_true": int(permuted_matches_true), "ok": bool(permuted_ok),
                                   "detail": permuted_detail}
    if verbose:
        print(f"  [seed {seed}] PERMUTED-MAPPING anti-cheat: wrong-table reproduces {permuted_matches_true} true "
              f"answers (must be 0) -> {'OK' if permuted_ok else 'FAIL'}", flush=True)
        for q, tf, wf, wr, rep in permuted_detail:
            print(f"      \"{q}\": true={tf}  wrong-map({wr})={wf}  {'REPRODUCED!' if rep else 'differs'}",
                  flush=True)

    seed_go = (n_wh == len(wh_ok) and full_ok and bare_ok and false_accepts == 0 and moat_recall_ok
               and abstain_ok == n_abstain and reparse_ok and permuted_ok)
    results["seed_go"] = bool(seed_go)
    return results


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    print("[Tier 0.3 wh-questions DE-RISK] natural wh-questions as a filler-gap dependency: the wh-word selects "
          "the GAPPED typed role (verb-frame-constrained), queried on the Tier-0.1 ArgStructureComposer; render "
          "via the 0.1 frame; the no-confab moat + the PERMUTED-MAPPING anti-cheat.", flush=True)
    print("  (cheap-first HARD GATE before the full build; tiny vocab; numpy/CPU)\n", flush=True)
    seeds = (42, 43, 44, 45, 46, 47)
    rows = [run_seed(s) for s in seeds]

    n_go = sum(1 for r in rows if r["seed_go"])
    all_wh = all(r["wh"]["n_ok"] == r["wh"]["n_total"] for r in rows)
    all_render = all(r["render_where_boy_go"]["full"] == "the boy goes to the park"
                     and r["render_where_boy_go"]["bare"] == "to the park" for r in rows)
    total_fa = sum(r["moat"]["false_accepts"] for r in rows)
    all_abstain = all(r["moat"]["abstain_ok"] == r["moat"]["n_abstain"] for r in rows)
    all_reparse = all(r["verify_reparse"] for r in rows)
    all_permuted = all(r["permuted_mapping"]["ok"] for r in rows)

    print(f"\n{'='*100}", flush=True)
    print(f"  SUMMARY ({len(seeds)} seeds): GO {n_go}/{len(seeds)}", flush=True)
    print(f"    wh-answer all-correct:     {all_wh}", flush=True)
    print(f"    render where-boy-go:       {all_render}  (e.g. full=\"{rows[0]['render_where_boy_go']['full']}\" "
          f"bare=\"{rows[0]['render_where_boy_go']['bare']}\")", flush=True)
    print(f"    moat false-accepts total:  {total_fa}  (must be 0)", flush=True)
    print(f"    moat abstain all:          {all_abstain}", flush=True)
    print(f"    verify re-parse all:       {all_reparse}", flush=True)
    print(f"    permuted-mapping anti-cheat (wrong table != true answers): {all_permuted}", flush=True)
    print(f"{'='*100}", flush=True)

    go = (n_go == len(seeds) and all_wh and all_render and total_fa == 0 and all_abstain
          and all_reparse and all_permuted)
    if go:
        print(f"  GO: natural wh-questions parse to a verb-frame-constrained typed-role query on the composer, "
              f"answer \"where does the boy go?\" -> \"to the park\", preserve the no-confab moat (0 false-accepts; "
              f"unanswerable/unlicensed wh abstain), the answer re-parses to the stored fact, and the "
              f"PERMUTED-MAPPING control proves the wh->role mapping carries the meaning. ==> PROCEED to the full "
              f"build (the wh-parser + console route).", flush=True)
    else:
        print(f"  NO-GO: wh-questions do NOT answer correctly OR break the moat OR the permuted-mapping control is "
              f"decorative. STOP -- this is a valid NEGATIVE that re-scopes 0.3.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)

    out = {"go": bool(go), "n_go": int(n_go), "n_seeds": len(seeds), "all_wh": bool(all_wh),
           "all_render": bool(all_render), "total_false_accepts": int(total_fa), "all_abstain": bool(all_abstain),
           "all_reparse": bool(all_reparse), "all_permuted_mapping": bool(all_permuted), "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_tier0_wh_questions_derisk.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
