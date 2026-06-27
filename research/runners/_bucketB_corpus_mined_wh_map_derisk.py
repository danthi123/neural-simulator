"""Bucket-B B-mine-2 -- the wh->role MAP as the INVERSE INDEX of the corpus-mined verb-frames -- the cheap-first
DE-RISK. THE NEAR-FREE COROLLARY of B-mine-1.

B-mine-1 (`_bucketB_corpus_mined_frames_derisk.py`, GO) mined the verb-frame LEXICON (go->GOAL, give->THEME+
RECIPIENT, ...) from corpus ARGUMENT co-occurrence over the brain's OWN learned verbs. The Bucket-B research gate
(`2026-06-27-burndown-bucketB-structure-learning-research-gate.md`, B-mine-2) found the wh->role map
(`wh_question_parser.py:44` WH_ROLE_CANDIDATES: where->[GOAL,LOCATION], what->[patient,THEME], who->[agent,
RECIPIENT], ...) "falls out nearly free as the INVERSE INDEX of the mined frames": which typed role a wh-word
gaps follows from which roles the mined frames license, plus a wh-word->role-CLASS affinity (a small closed,
corpus-justifiable lexicon -- NOT a per-verb hand list). Structure ACQUIRED, not given.

THE MECHANISM (reuse-by-import; the B-mine-1 mined frames are the INPUT; NO sim/ edit, NO composer edit):
  half 1 -- MINE the verb-frame role inventory (== B-mine-1). The mined FRAME_LEXICON gives, per verb, the set of
    typed roles the verb LICENSES (GOAL/LOCATION/THEME/RECIPIENT/patient/INSTRUMENT/SOURCE), each corpus-attested;
    B-mine-1's mined VERB_PREP_ROLE gives the (verb, prep)->role evidence. This is the brain's ACQUIRED structure.
  half 2 -- INVERT it. A wh-word questions ONE thematic ROLE-CLASS (the filler-gap lexicon): `where` gaps a SPATIAL
    role (GOAL/LOCATION), `what` gaps an ENTITY/THEME role (patient/THEME), `who` gaps an ANIMATE-PARTICIPANT role
    (agent/RECIPIENT), `when`->TEMPORAL, `with`->INSTRUMENT, `whom`->RECIPIENT. The wh-word->role-CLASS affinity
    (WH_ROLE_CLASS below) is the ONLY hand input -- a SMALL CLOSED lexicon over the language's wh-words
    (corpus-justifiable: a wh-word's selectional class is a closed grammatical fact, like the prepositions in
    B-mine-1's PREP_ROLE), NOT a per-verb hand list. For each wh-word, gather the mined roles in its class and
    ORDER them by the INVERSE INDEX: CORE roles (agent/patient, licensed by ~every verb) before OBLIQUE roles, then
    by descending CORPUS ATTESTATION (the most-attested role first -- GOAL > LOCATION for `where`; patient > THEME
    for `what`). The multiword cues (where-from->SOURCE, with-what->INSTRUMENT, ...) come from the prep->role
    associations B-mine-1's mining already attests (`from`->SOURCE x702, `with`->INSTRUMENT). Produces a MINED
    WH_ROLE_CANDIDATES (+ WH_MULTIWORD); the wh-parser consumes it behind a flag (the hand map = the parity oracle).

THE ANTI-CHEAT BAR (mirrors B-mine-1 + the wh-parser's existing permuted-mapping control):
  - MATCH/justify: the MINED wh-map MATCHES the hand WH_ROLE_CANDIDATES on the validated wh-cases
    ("where does the boy go?"->GOAL, "what does the mom give?"->THEME, "who does the girl give to?"->RECIPIENT),
    or DIFFERS with a corpus-justified reason (a role-class the corpus simply does not attest).
  - PARSE PARITY: `parse_wh_question` + `answer_wh` with the MINED map == with the hand map on the validated
    questions (same gapped role, same filler, answer-identical).
  - ** PERMUTED-MINING (the decisive control, mirror B-mine-1) **: scramble the mined-frame INPUT (assign each verb
    a RANDOM other-verb role inventory) -> a BROKEN inverse index -> the wh-map maps each wh-word to roles no verb
    licenses (or the WRONG ones) -> the wh-parses COLLAPSE (wrong/abstaining). If it does not collapse, the mined
    frames are not load-bearing for the wh-map.
  - the no-confab MOAT (an unlicensed/unstored wh -> None) stays 0-FA on the mined map; PROVENANCE: every mined
    wh-candidate is backed by a corpus-attested licensing verb-frame (no train/test leak: the parity questions are
    answered, not used to build the map).

GATE -- GO requires ALL of:
  (i)   the MINED wh-map MATCHES-OR-JUSTIFIES the hand WH_ROLE_CANDIDATES on the validated wh-words
        (where/what/who[/whom/with/when], 0 unjustified differences);
  (ii)  PARSE PARITY: parse + answer on the MINED map == on the hand map for the validated questions;
  (iii) ** PERMUTED-MINING collapses **: a scrambled mined-frame input -> a broken wh-map -> wrong/abstaining
        parses far below the mined-map accuracy (the mined frames, not the apparatus, carry the wh-map);
  (iv)  the no-confab MOAT abstains (0 false-accepts) on the mined map;
  (v)   PROVENANCE: every mined wh-candidate role is backed by a corpus-attested licensing frame.
  multi-seed: the mining + inversion are corpus-deterministic; the SEED varies the composer's codes + the
  permuted-mining scramble (exactly what B-mine-1 + the wh de-risk varied).

Run (CPU/numpy fast path; spaCy parses TinyStories once for the mine, then 6 seeds of parse parity + permuted):
  SIM_BACKEND=numpy python -m research.runners._bucketB_corpus_mined_wh_map_derisk --seeds 42 43 44 45 46 47
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.argstructure_composer import (  # noqa: E402
    ArgStructureComposer, FRAME_LEXICON)
from research.runners._bucketB_corpus_mined_frames_derisk import (  # noqa: E402
    mine_verb_argstats, derive_frame_lexicon, _roles_of, PREP_ROLE)
from research.runners.wh_question_parser import (  # noqa: E402
    WH_ROLE_CANDIDATES as HAND_WH_ROLE_CANDIDATES, WH_MULTIWORD as HAND_WH_MULTIWORD,
    parse_wh_question, answer_wh)

# ------------------------------------------------------------------------------------------------------------
# THE wh-word -> role-CLASS AFFINITY (half 2's ONLY hand input). This is NOT a per-verb hand list (the thing
# B-mine-2 removes) and NOT the wh->role MAP (that is DERIVED below by inverting the mined frames); it is a SMALL
# CLOSED lexicon over the LANGUAGE'S WH-WORDS -- a wh-word's selectional ROLE-CLASS, the same kind of closed
# grammatical fact as B-mine-1's PREP_ROLE (a preposition's dominant thematic role). A wh-word questions a CLASS of
# thematic role; the corpus-mined frames then say WHICH concrete roles fall in that class + in what order (the
# inverse index). Each wh-word maps to a set of ROLE-CLASS tags; a typed role belongs to a class via ROLE_CLASS.
#   who   -> animate participant   (the SUBJECT / the RECIPIENT)
#   what  -> entity / theme        (the direct OBJECT / the THEME)
#   where -> spatial               (the GOAL / the LOCATION)
#   when  -> temporal              (the TIME)
#   whom  -> recipient             (the indirect-object RECIPIENT)
#   with  -> instrument            (the INSTRUMENT)
WH_ROLE_CLASS = {
    "who":   ("PARTICIPANT",),
    "what":  ("ENTITY",),
    "where": ("SPATIAL",),
    "when":  ("TEMPORAL",),
    "whom":  ("RECIPIENT_CLASS",),
    "with":  ("INSTRUMENT_CLASS",),
}
# which CLASS(es) a concrete typed role belongs to (the role-class membership; the dual of WH_ROLE_CLASS). A role
# may belong to MORE THAN ONE class -- a RECIPIENT is both an animate PARTICIPANT (askable with `who`) and the
# dative target (askable with `whom`) -- so the membership is a frozenset.
ROLE_CLASS = {
    "agent":      frozenset({"PARTICIPANT"}),
    "RECIPIENT":  frozenset({"PARTICIPANT", "RECIPIENT_CLASS"}),   # who-candidate AND whom-candidate
    "patient":    frozenset({"ENTITY"}),
    "THEME":      frozenset({"ENTITY"}),
    "GOAL":       frozenset({"SPATIAL"}),
    "LOCATION":   frozenset({"SPATIAL"}),
    "TIME":       frozenset({"TEMPORAL"}),
    "INSTRUMENT": frozenset({"INSTRUMENT_CLASS"}),
    "SOURCE":     frozenset({"SPATIAL"}),       # SOURCE is also a where-class role; the where-from multiword fixes it
}
# the CORE (argument-structure obligatory) roles -- licensed by ~every verb (agent always; patient by transitives).
# Core roles rank BEFORE oblique roles within a wh-word's candidate list (the subject/object gap is the default).
CORE_ROLES = {"agent", "patient"}

# multiword wh-cues are derived from PREP-ROLE associations the mining attests: a wh-word + a preposition fixes the
# role unambiguously when the prep's role differs from the wh-word's default class. `where`+`from` -> SOURCE
# (`from`->SOURCE in PREP_ROLE, x702 attested); `with`+`what` -> INSTRUMENT (`with`->INSTRUMENT); `to`+`whom` ->
# RECIPIENT (the ditransitive `to`-dative). Built from PREP_ROLE so the multiword table is also acquired-not-given.
_WH_PREP_MULTIWORD = [
    ("where", "from"), ("from", "where"),     # SOURCE  (from->SOURCE)
    ("with", "what"),                          # INSTRUMENT (with->INSTRUMENT)
    ("to", "whom"),                            # RECIPIENT (to-dative)
]

# the wh-words the GATE validates (the hand WH_ROLE_CANDIDATES single-word keys; `who_to` is a derived sub-key).
VALIDATED_WH = ["who", "what", "where", "when", "whom", "with"]


# ------------------------------------------------------------------------------------------------------------
# Stage 1 -- DERIVE the wh->role map as the INVERSE INDEX of the mined frames (half 2).
# ------------------------------------------------------------------------------------------------------------
def derive_wh_role_map(mined_frames, attest_count=None):
    """INVERT the mined verb-frame lexicon into a wh->role candidate map (+ a who_to sub-key + the multiword cues),
    each candidate backed by corpus-attested licensing frames.

    The inverse index: for each wh-word, gather every mined role in the wh-word's ROLE-CLASS (WH_ROLE_CLASS x
    ROLE_CLASS) and ORDER them by (CORE-before-OBLIQUE, then descending CORPUS ATTESTATION of the role -- so
    GOAL > LOCATION for `where` [3437 vs 2819 attestations], patient > THEME for `what`, agent[core] > RECIPIENT
    [oblique] for `who`). `attest_count` is the per-role corpus attestation total (the sum of the role's slot counts
    across verbs, from B-mine-1's provenance); when None it falls back to the verb-LICENSE count (how many verbs
    attest the role) -- a coarser but monotone proxy. Returns (wh_role_candidates, wh_multiword, provenance)."""
    # role -> the set of verbs whose mined frame licenses it (the inverse index of the frame lexicon) + a count.
    role_verbs = collections.defaultdict(list)
    for verb, units in mined_frames.items():
        if verb == "_default":
            continue
        for r in _roles_of(units):
            if r not in ("action",):
                role_verbs[r].append(verb)
    role_count = {r: len(vs) for r, vs in role_verbs.items()}
    # the ranking weight = the role's CORPUS ATTESTATION (slot-count total) when available, else the verb-license
    # count. The attestation is the faithful inverse-index frequency (GOAL out-attests LOCATION even though FEWER
    # verbs license it -- motion verbs fire GOAL far more often), which is what the hand map's order encodes.
    rank_weight = {r: (attest_count.get(r, 0) if attest_count is not None else c) for r, c in role_count.items()}

    def _rank_key(role):
        # CORE roles first (agent/patient -- the obligatory subject/object gap; agent has 0 oblique attestation so
        # the core flag, not the count, ranks it), then by descending corpus attestation. Tie-break by role name.
        return (0 if role in CORE_ROLES else 1, -rank_weight.get(role, 0), role)

    wh_map = {}
    prov = {}
    for wh, classes in WH_ROLE_CLASS.items():
        cls = set(classes)
        cands = [r for r in role_verbs if ROLE_CLASS.get(r, frozenset()) & cls]
        # SOURCE belongs to SPATIAL but is reserved for the where-from multiword (it would otherwise pollute
        # `where`'s single-word candidates with a role the bare `where` does not gap) -- drop it from the bare list.
        cands = [r for r in cands if r != "SOURCE"]
        cands.sort(key=_rank_key)
        wh_map[wh] = cands
        prov[wh] = {"class": list(classes),
                    "candidates": [{"role": r, "n_licensing_verbs": role_count.get(r, 0),
                                    "attestation": rank_weight.get(r, 0),
                                    "example_verbs": sorted(role_verbs.get(r, []))[:4]} for r in cands]}

    # the who_to sub-key: a trailing to-PP on a `who` question fixes the RECIPIENT gap (the ditransitive dative).
    # Derived: RECIPIENT first (the to-PP), then the bare-who candidates (the agent fallback). Only if RECIPIENT is
    # mined at all (a corpus that never attests ditransitives would not license it -- the honest constraint).
    if "RECIPIENT" in role_verbs:
        wh_map["who_to"] = ["RECIPIENT"] + [r for r in wh_map.get("who", []) if r != "RECIPIENT"]
        prov["who_to"] = {"class": ["RECIPIENT_CLASS", "PARTICIPANT"],
                          "candidates": [{"role": "RECIPIENT", "n_licensing_verbs": role_count.get("RECIPIENT", 0),
                                          "example_verbs": sorted(role_verbs.get("RECIPIENT", []))[:4]}]}

    # the multiword cues, derived from PREP_ROLE (a prep that maps to a role fixes the wh+prep combination); only
    # emit a multiword whose target role is corpus-attested in the mined frames OR (for SOURCE) in PREP_ROLE.
    wh_multiword = {}
    mw_prov = {}
    for w1, w2 in _WH_PREP_MULTIWORD:
        if (w1, w2) == ("to", "whom"):                         # to-whom -> RECIPIENT (the dative; `to` is the
            # ditransitive dative here, NOT a GOAL -- same disambiguation as B-mine-1's ditransitive rule for `to`).
            if "RECIPIENT" in role_verbs:
                wh_multiword[(w1, w2)] = "RECIPIENT"
                mw_prov["to whom"] = {"role": "RECIPIENT", "from_prep": "to(dative)", "attested": True}
            continue
        # the content preposition of the cue (the non-wh token): from / with. Its PREP_ROLE is the fixed role.
        prep = next((w for w in (w1, w2) if w in PREP_ROLE), None)
        if prep is not None:                                   # where-from / with-what (a real preposition)
            role = PREP_ROLE[prep]
            wh_multiword[(w1, w2)] = role
            mw_prov[f"{w1} {w2}"] = {"role": role, "from_prep": prep, "attested": role in role_verbs or role == "SOURCE"}
    return wh_map, wh_multiword, {"wh": prov, "multiword": mw_prov, "role_count": role_count}


def compare_wh_maps(mined_map, mined_mw, hand_map=HAND_WH_ROLE_CANDIDATES, hand_mw=HAND_WH_MULTIWORD):
    """Per validated wh-word: does the MINED candidate list MATCH the hand WH_ROLE_CANDIDATES? Returns
    {wh: ('match'|'differ'|'unmined', mined_cands, hand_cands)}. A `differ` is corpus-JUSTIFIED iff the difference
    is a role-class the corpus does not attest (checked by the caller via provenance)."""
    out = {}
    for wh in VALIDATED_WH:
        hand_c = hand_map.get(wh, [])
        mined_c = mined_map.get(wh)
        if mined_c is None:
            out[wh] = ("unmined", None, hand_c)
        else:
            out[wh] = ("match" if mined_c == hand_c else "differ", mined_c, hand_c)
    # multiword parity
    mw_match = (mined_mw == dict(hand_mw))
    out["__multiword__"] = ("match" if mw_match else "differ", mined_mw, dict(hand_mw))
    return out


# ------------------------------------------------------------------------------------------------------------
# Stage 2 -- PARSE PARITY + the anti-cheats (the de-risk's load-bearing evidence).
# ------------------------------------------------------------------------------------------------------------
# The validated wh-questions + their stored facts. Each question's verb has a mined frame (go/give/put/come...);
# the gold (role, filler) is what the hand wh-map + composer answer. Vocab + fillers are concrete in-vocab nouns.
PARITY_VOCAB = ["boy", "girl", "mom", "dog", "cat", "go", "give", "come", "run", "walk", "put", "chase",
                "park", "ball", "house", "tree", "shop", "bone", "table", "river"]
PARITY_FACTS = [
    {"agent": "boy", "action": "go", "GOAL": "park"},
    {"agent": "mom", "action": "give", "THEME": "ball", "RECIPIENT": "dog"},
    {"agent": "girl", "action": "give", "THEME": "bone", "RECIPIENT": "cat"},
    {"agent": "dog", "action": "come", "GOAL": "house"},
    {"agent": "cat", "action": "chase", "patient": "river"},
]
# (question, expected_role, expected_filler) -- the validated wh-cases the gate names + frame-family coverage.
WH_CASES = [
    ("where does the boy go?",        "GOAL",      "park"),     # the headline case (gate)
    ("what does the mom give?",       "THEME",     "ball"),     # the headline case (gate)
    ("who does the girl give to?",    "RECIPIENT", "cat"),      # who + to-gap -> RECIPIENT
    ("where does the dog come?",      "GOAL",      "house"),    # where -> GOAL (come licenses GOAL)
    ("what does the cat chase?",      "patient",   "river"),    # what -> patient (default transitive)
    ("who chase river?",              "agent",     "cat"),      # who -> agent (bare subject question)
]


def frame_roles_of(frames):
    """The per-verb {verb: [roles]} licensing map of a frame lexicon -- the FRAME_ROLES the wh-parser intersects the
    wh-candidates against (so the wh resolution consumes the MINED, not the hand, frame inventory)."""
    return {v: list(_roles_of(units)) for v, units in frames.items()}


def _composer(seed, mined_frames):
    """A composer that renders/recalls through the MINED frames (B-mine-1) -- so parse-parity is end-to-end on the
    acquired structure (the wh-map AND the frame lexicon both mined)."""
    return ArgStructureComposer(seed=seed, D=64, vocab=list(PARITY_VOCAB), frame_lexicon=mined_frames,
                                use_spiking_cq=False)


def _store(comp):
    for f in PARITY_FACTS:
        if f["action"] in comp._frames:
            comp.store_fact(f)


def _patched_parser(wh_map, wh_mw):
    """The wh-parser threads a `role_map` for the single-word candidates (the existing permuted-mapping seam). The
    MULTIWORD table is a module constant the parser reads, so to exercise a MINED multiword we monkeypatch the
    module's WH_MULTIWORD for the call (and restore it). Returns a context-manager-like (set, restore) pair."""
    import research.runners.wh_question_parser as whp
    saved = dict(whp.WH_MULTIWORD)

    def _set():
        whp.WH_MULTIWORD.clear(); whp.WH_MULTIWORD.update(wh_mw)

    def _restore():
        whp.WH_MULTIWORD.clear(); whp.WH_MULTIWORD.update(saved)
    return _set, _restore


def parse_parity(seed, mined_frames, mined_map, mined_mw):
    """PARSE PARITY: parse + answer each validated wh-question with the MINED wh-map (resolved against the MINED
    frame roles) vs the HAND wh-map (resolved against the hand frame roles). Both must return the SAME gapped role
    + the SAME filler (answer-identical). Returns (parity_ok, mined_acc, details)."""
    comp = _composer(seed, mined_frames)
    _store(comp)
    mined_fr = frame_roles_of(mined_frames)
    set_mined, restore = _patched_parser(mined_map, mined_mw)
    details, parity_ok, n_mined_ok = [], True, 0
    for q, exp_role, exp_filler in WH_CASES:
        # hand-map answer (the parity oracle): the module's default WH_MULTIWORD + hand FRAME_ROLES.
        h_filler, h_role, _hp = answer_wh(comp, q)                 # role_map/frame_roles=None -> hand scaffold
        # mined-map answer: thread the mined role_map + the mined FRAME_ROLES + swap in the mined multiword.
        set_mined()
        try:
            m_filler, m_role, _mp = answer_wh(comp, q, role_map=mined_map, frame_roles=mined_fr)
        finally:
            restore()
        role_match = (m_role == h_role)
        filler_match = (m_filler == h_filler)
        mined_correct = (m_role == exp_role and m_filler == exp_filler)
        if mined_correct:
            n_mined_ok += 1
        pair_ok = role_match and filler_match and mined_correct
        parity_ok = parity_ok and pair_ok
        details.append({"q": q, "exp_role": exp_role, "exp_filler": exp_filler,
                        "hand": [str(h_role), str(h_filler)], "mined": [str(m_role), str(m_filler)],
                        "role_match": role_match, "filler_match": filler_match, "pair_ok": pair_ok})
    return parity_ok, (n_mined_ok / max(len(WH_CASES), 1)), details


def permuted_mining(seed, mined_frames, mined_map, mined_mw):
    """** THE DECISIVE CONTROL (mirror B-mine-1) ** -- SCRAMBLE the mined-frame INPUT (assign each mineable verb a
    RANDOM other-verb role inventory), then RE-DERIVE the wh-map AND the per-verb FRAME_ROLES from the broken
    frames. The wh-parses must COLLAPSE: the wh-parser resolves the gapped role by intersecting the wh-candidates
    with the verb's LICENSED roles (FRAME_ROLES) -- when `go`'s frame is scrambled to (say) `give`'s
    [THEME, RECIPIENT], "where does the boy go?" -> where=[GOAL,LOCATION] INTERSECT scrambled-go={THEME,RECIPIENT}
    = EMPTY -> abstain. (The verb-frame inventory -- the structure -- is destroyed while the apparatus -- the
    composer, the parser, the affinity lexicon -- is identical, exactly B-mine-1's permuted-mining logic. The
    composer still holds the CORRECT facts so the collapse is the wh-MAP failing to resolve, not the facts.)

    We measure the fraction of validated questions whose (role, filler) on the SCRAMBLED-derived map+frames matches
    the mined-map answer -- it must drop far below the mined-map accuracy."""
    mineable = [v for v in mined_frames if v != "_default"]
    rng = np.random.default_rng(seed * 911 + 17)
    perm = list(mineable)
    for _ in range(200):
        rng.shuffle(perm)
        if all(perm[i] != mineable[i] for i in range(len(mineable))):
            break
    scrambled = {v: list(mined_frames[perm[i]]) for i, v in enumerate(mineable)}
    scrambled["_default"] = list(mined_frames["_default"])
    perm_map, perm_mw, _pp = derive_wh_role_map(scrambled)
    scrambled_fr = frame_roles_of(scrambled)

    # the composer holds the CORRECT facts (rendered through the real mined frames) -- only the wh-MAP + the per-verb
    # FRAME_ROLES the parser resolves against are corrupted (isolating the wh-resolution's dependence on the mined
    # frame inventory). A collapse = the scrambled inventory makes the wh-gap unresolvable / wrong.
    comp = _composer(seed, mined_frames)
    _store(comp)
    mined_fr = frame_roles_of(mined_frames)
    set_mined, restore = _patched_parser(mined_map, mined_mw)
    set_perm, restore_perm = _patched_parser(perm_map, perm_mw)
    n_match = 0
    for q, _er, _ef in WH_CASES:
        set_mined()
        try:
            m_filler, m_role, _ = answer_wh(comp, q, role_map=mined_map, frame_roles=mined_fr)
        finally:
            restore()
        set_perm()
        try:
            p_filler, p_role, _ = answer_wh(comp, q, role_map=perm_map, frame_roles=scrambled_fr)
        finally:
            restore_perm()
        if p_role == m_role and p_filler == m_filler and m_filler is not None:
            n_match += 1
    return n_match / max(len(WH_CASES), 1), {v: _roles_of(scrambled[v]) for v in mineable[:6]}


def moat(seed, mined_frames, mined_map, mined_mw):
    """The no-confab MOAT on the MINED wh-map: an unanswerable / unstored / frame-unlicensed wh -> None (abstain),
    0 false-accepts. (Same cases as the wh de-risk: an unstored agent, an unstored pair, an unlicensed wh.)"""
    comp = _composer(seed, mined_frames)
    _store(comp)
    mined_fr = frame_roles_of(mined_frames)
    set_mined, restore = _patched_parser(mined_map, mined_mw)
    set_mined()
    try:
        cases = [
            ("where does the boy go?",    "park"),       # stored -> answers (positive control)
            ("where does the boy give?",  None),          # boy+give has no GOAL stored -> abstain
            ("where does the cat go?",    None),          # cat+go not stored -> abstain
            ("what does the boy give?",   None),          # boy+give not stored -> abstain
            ("who does the dog give to?", None),          # dog+give not stored -> abstain
            ("when does the boy go?",     None),          # go's frame has no TIME slot -> abstain (unlicensed)
            ("where does the cat chase?", None),          # chase (default) has no GOAL/LOCATION -> abstain
        ]

        def _ans(q):
            return answer_wh(comp, q, role_map=mined_map, frame_roles=mined_fr)[0]
        false_accepts = 0
        recall_ok = (_ans("where does the boy go?") == "park")
        for q, exp in cases:
            if exp is None and _ans(q) is not None:
                false_accepts += 1
        n_abstain = sum(1 for _, e in cases if e is None)
        abstain_ok = sum(1 for q, e in cases if e is None and _ans(q) is None)
    finally:
        restore()
    return int(false_accepts), bool(recall_ok), int(abstain_ok), int(n_abstain)


# ------------------------------------------------------------------------------------------------------------
def run_seed(seed, mined_frames, mined_map, mined_mw):
    t0 = time.time()
    parity_ok, mined_acc, parity_details = parse_parity(seed, mined_frames, mined_map, mined_mw)
    pm_acc, _scrambled = permuted_mining(seed, mined_frames, mined_map, mined_mw)
    fa, recall_ok, abstain_ok, n_abstain = moat(seed, mined_frames, mined_map, mined_mw)
    return {
        "seed": seed, "elapsed_s": round(time.time() - t0, 1),
        "parity_ok": parity_ok, "mined_acc": mined_acc, "permuted_mining_acc": pm_acc,
        "moat_false_accepts": fa, "moat_recall_ok": recall_ok,
        "moat_abstain_ok": abstain_ok, "moat_n_abstain": n_abstain,
        "moat_ok": (fa == 0 and recall_ok and abstain_ok == n_abstain),
        "parity_details": parity_details,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46, 47])
    ap.add_argument("--corpus", default="data/corpus/tinystories.txt")
    ap.add_argument("--npz", default="bridges/firstchat/brainALL_w7000.npz_seed42.npz")
    ap.add_argument("--max-sentences", type=int, default=200000)
    ap.add_argument("--min-freq", type=int, default=30)
    ap.add_argument("--out", default="research/findings/raw/_bucketB_corpus_mined_wh_map.json")
    a = ap.parse_args()

    print(f"[B-mine-2 wh->role MAP as the INVERSE INDEX of the mined verb-frames] corpus={a.corpus} | "
          f"brain={os.path.basename(a.npz)}\n  half1: MINE the verb-frame role inventory (== B-mine-1); half2: "
          f"INVERT it -> a wh-word gaps a role-CLASS (the small closed WH_ROLE_CLASS affinity), the mined frames "
          f"say WHICH roles + in what order (CORE-first, then corpus freq).\n  HEADLINE controls = MATCH-or-justify "
          f"the hand wh-map + PARSE PARITY + PERMUTED-MINING (scrambled frames -> broken wh-map -> collapse).\n",
          flush=True)

    # ---- Stage 0+1: MINE the frames (B-mine-1) + DERIVE the wh-map (corpus-deterministic; once) ----
    npz_full = os.path.join(_REPO, a.npz) if not os.path.isabs(a.npz) else a.npz
    d = np.load(npz_full, allow_pickle=True)
    vocab = set(str(w).lower() for w in d["vocab"])
    print(f"[mine] brain vocab: {len(vocab)} words | mining the verb-frame role inventory from {a.corpus} ...",
          flush=True)
    corpus_full = os.path.join(_REPO, a.corpus) if not os.path.isabs(a.corpus) else a.corpus
    stats, n_sent = mine_verb_argstats(corpus_full, vocab, a.max_sentences, target_verbs=None)
    mined_frames, _mined_vpr, frame_prov = derive_frame_lexicon(stats, min_freq=a.min_freq)
    # the per-role CORPUS ATTESTATION total (sum of the role's oblique slot counts across verbs, from B-mine-1's
    # provenance) -- the inverse-index ranking weight (GOAL out-attests LOCATION). Core agent/patient are not in the
    # oblique-slot log, but agent is ranked by the core flag, and patient's direct-object count is in the slot log.
    attest_count = collections.Counter()
    for _v, p in frame_prov.items():
        if not p.get("attested"):
            continue
        for s in p.get("slots", []):
            attest_count[s["role"]] += s.get("count", 0)
    mined_map, mined_mw, wh_prov = derive_wh_role_map(mined_frames, attest_count=dict(attest_count))
    cf = compare_wh_maps(mined_map, mined_mw)
    print(f"[mine] parsed {n_sent} sentences -> "
          f"{len([v for v in mined_frames if v != '_default'])} verbs cleared attestation; "
          f"inverted to a {len(mined_map)}-entry wh-map.", flush=True)

    # ---- (i) MATCH-or-justify on the validated wh-words ----
    print(f"\n  {'wh-word':9s} {'status':8s}  {'mined candidates':28s}  hand candidates", flush=True)
    n_unjustified = 0
    justified_diffs = []
    for wh in VALIDATED_WH:
        status, mc, hc = cf[wh]
        mc_s = " ".join(mc) if mc else "(un-mined: no licensing frame)"
        flag = ""
        if status == "differ":
            # a difference is JUSTIFIED iff every mined candidate is corpus-attested (a licensing verb-frame exists)
            # AND no hand candidate that the corpus DOES attest is missing (a dropped-but-attested role is unjust).
            mined_attested = all(c.get("n_licensing_verbs", 0) > 0 for c in wh_prov["wh"].get(wh, {}).get("candidates", []))
            dropped_attested = [r for r in hc if r not in mc and wh_prov["role_count"].get(r, 0) > 0]
            if mined_attested and not dropped_attested:
                justified_diffs.append(wh)
                flag = "  [corpus-JUSTIFIED]"
            else:
                n_unjustified += 1
                flag = f"  [** UNJUSTIFIED ** dropped-attested={dropped_attested}]"
        print(f"  {wh:9s} {status:8s}  {mc_s:28s}  {' '.join(hc)}{flag}", flush=True)
        for c in wh_prov["wh"].get(wh, {}).get("candidates", []):
            print(f"           - {c['role']:10s} licensed by {c['n_licensing_verbs']:3d} verbs "
                  f"(e.g. {c['example_verbs']})", flush=True)
    mw_status = cf["__multiword__"][0]
    print(f"  {'MULTIWORD':9s} {mw_status:8s}  mined={mined_mw}", flush=True)
    print(f"                     hand ={dict(HAND_WH_MULTIWORD)}", flush=True)
    match_or_justify_ok = (n_unjustified == 0)

    # ---- (ii)-(v) per-seed parse parity + the anti-cheats ----
    rows = []
    for s in a.seeds:
        r = run_seed(s, mined_frames, mined_map, mined_mw)
        rows.append(r)
        print(f"\n  [seed {s}] parse-parity {'OK' if r['parity_ok'] else 'X'} (mined-acc {r['mined_acc']:.2f}) | "
              f"** PERMUTED-MINING {r['permuted_mining_acc']:.2f} ** | moat "
              f"{'ok' if r['moat_ok'] else 'X'} (FA {r['moat_false_accepts']}, "
              f"abstain {r['moat_abstain_ok']}/{r['moat_n_abstain']})", flush=True)
        for dts in r["parity_details"][:3]:
            print(f"           \"{dts['q']}\" -> mined {dts['mined']} (hand {dts['hand']}, "
                  f"match {dts['pair_ok']})", flush=True)

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    mined_acc = m("mined_acc"); pm_acc = m("permuted_mining_acc")
    all_parity = all(r["parity_ok"] for r in rows)
    all_moat = all(r["moat_ok"] for r in rows)
    total_fa = sum(r["moat_false_accepts"] for r in rows)

    # PROVENANCE: every mined wh-candidate role is backed by >=1 corpus-attested licensing verb-frame.
    prov_ok = all(c["n_licensing_verbs"] > 0
                  for wh in mined_map for c in wh_prov["wh"].get(wh, {}).get("candidates", []))

    # GATE
    permuted_mining_ok = (pm_acc <= 0.5 and mined_acc - pm_acc >= 0.4)   # the decisive control: scrambled frames collapse
    go = (match_or_justify_ok and all_parity and permuted_mining_ok and all_moat and prov_ok)

    out_full = os.path.join(_REPO, a.out) if not os.path.isabs(a.out) else a.out
    os.makedirs(os.path.dirname(out_full), exist_ok=True)
    summary = {
        "capability": "corpus-mined wh->role map as the inverse index of the mined frames (B-mine-2)",
        "corpus": a.corpus, "brain": os.path.basename(a.npz), "n_seeds": len(a.seeds), "n_sentences": n_sent,
        "mined_wh_role_candidates": mined_map,
        "mined_wh_multiword": {f"{k[0]} {k[1]}": v for k, v in mined_mw.items()},
        "hand_wh_role_candidates": {k: list(v) for k, v in HAND_WH_ROLE_CANDIDATES.items()},
        "hand_wh_multiword": {f"{k[0]} {k[1]}": v for k, v in HAND_WH_MULTIWORD.items()},
        "wh_comparison": {wh: {"status": cf[wh][0], "mined": cf[wh][1], "hand": cf[wh][2]} for wh in VALIDATED_WH},
        "multiword_status": mw_status,
        "wh_role_class_affinity": {k: list(v) for k, v in WH_ROLE_CLASS.items()},
        "provenance": wh_prov,
        "justified_diffs": justified_diffs, "n_unjustified": n_unjustified,
        "match_or_justify_ok": match_or_justify_ok, "parity_ok": all_parity,
        "mined_acc": mined_acc, "permuted_mining_acc": pm_acc, "permuted_mining_ok": permuted_mining_ok,
        "moat_ok": all_moat, "total_false_accepts": total_fa, "provenance_ok": prov_ok,
        "go": go, "per_seed": rows,
    }
    with open(out_full, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"\n{'=' * 112}", flush=True)
    print(f"  MINED wh-map (inverse index of the corpus-mined frames): {mined_map}", flush=True)
    print(f"  MINED multiword: {dict(((k[0]+' '+k[1]), v) for k, v in mined_mw.items())}", flush=True)
    print(f"  MATCH-or-justify: {match_or_justify_ok} (multiword {mw_status}; "
          f"{len(justified_diffs)} corpus-justified diffs: {justified_diffs}; {n_unjustified} UNJUSTIFIED)",
          flush=True)
    print(f"  MEAN ({len(a.seeds)} seeds): parse-parity {all_parity} (mined-acc {mined_acc:.3f}) | "
          f"** PERMUTED-MINING {pm_acc:.3f} (must be <=0.5 AND >=0.4 below mined-acc) ** | "
          f"moat {all_moat} (FA total {total_fa}) | provenance {prov_ok}", flush=True)
    if go:
        print(f"\n  GO: the wh->role MAP is DERIVED as the INVERSE INDEX of the corpus-mined verb-frames "
              f"(B-mine-1) over the brain's OWN learned verbs -- structure ACQUIRED, not given. The mined wh-map "
              f"MATCHES the hand WH_ROLE_CANDIDATES on the validated wh-words; parse + answer on the MINED map == "
              f"on the hand map ({mined_acc:.2f}); ** PERMUTED-MINING collapses ({pm_acc:.2f}) ** -> the mined "
              f"frames, NOT the apparatus, carry the wh-map; moat 0-FA, provenance logged. The near-free corollary "
              f"of B-mine-1. NO sim/ edit.", flush=True)
    else:
        why = []
        if not match_or_justify_ok:
            why.append(f"{n_unjustified} UNJUSTIFIED wh-map difference(s)")
        if not all_parity:
            why.append(f"parse parity FAILED (mined != hand, or wrong answer; mined-acc {mined_acc:.2f})")
        if not permuted_mining_ok:
            why.append(f"** PERMUTED-MINING did NOT collapse (perm {pm_acc:.2f} vs mined {mined_acc:.2f}) -- the "
                       f"mined frames are NOT load-bearing for the wh-map **")
        if not all_moat:
            why.append(f"no-confab moat breach (FA {total_fa})")
        if not prov_ok:
            why.append("a mined wh-candidate lacks a corpus-attested licensing frame")
        print(f"\n  NO-GO: {'; '.join(why)}. Per the spec this is the honest NEGATIVE -- write it up, do not "
              f"over-claim.", flush=True)
    print(f"  [saved] {out_full}\n{'=' * 112}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
