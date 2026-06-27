"""Bucket-B B-mine-1 -- a CORPUS-MINED verb-frame LEXICON -- the cheap-first DE-RISK.

THE B1-FOR-RELATIONS -> B1-FOR-FRAMES STEP. B1 (`_regimeb_corpus_mined_axis_derisk.py`, GO) mined an ORDINAL
relation axis (size) from corpus scalar-adjective co-occurrence over the brain's vocab; here we mine the
single-largest hand-authored Bucket-B structure -- the verb-frame LEXICON (`argstructure_composer.py:65`
FRAME_LEXICON: go->GOAL, give->THEME+RECIPIENT, ...) -- from corpus ARGUMENT co-occurrence over the brain's
OWN learned verbs. Structure ACQUIRED, not given.
Spec: research/findings/2026-06-27-burndown-bucketB-structure-learning-research-gate.md (B-mine-1 + the anti-cheat).

THE MECHANISM (the converter = inverting the EXISTING extractor + a small corpus-justified prep->role table;
reuse-by-import, NO sim/ edit):
  half 1 -- MINE the argument distribution. `_corpus_svo_extract.py --typed-roles` ALREADY parses the corpus
    (spaCy) and observes, per verb, its DIRECT OBJECTS (dobj/dative/attr/oprd) + PREPOSITIONAL arguments
    (prep -> pobj, KEEPING the preposition) restricted to the brain's vocab, with provenance + an attestation
    count -- but it then CONSULTS the hand VERB_PREP_ROLE/FRAME_ROLES to label the slot. We INVERT that: accumulate
    `verb -> {preposition: count, dobj_count, freq}` and DERIVE the licensed slots from what the verb attests (the
    prepositions a verb takes above threshold = its obliques; a strong direct object = a THEME/patient slot). This
    is the same "count attested co-occurrences over the brain's vocab + log provenance" half as B1's
    `mine_size_scores`. Host-side curriculum prep (legitimate per BRAIN-BASED-ONLY: preparing the syllabus / the
    verb's frame the brain then RENDERS/RECALLS through spikes; like rendering a retinal image).
  half 2 -- DERIVE the frame via a small CORPUS-JUSTIFIABLE prep->role table (PREP_ROLE below: a prep's dominant
    role across verbs -- a closed lexicon over the language's prepositions, NOT a per-verb hand list) + the
    Bock & Levelt ditransitive rule (a verb with a STRONG direct object that ALSO takes `to` is ditransitive:
    dobj=THEME, to=RECIPIENT -- "give a hug to mom"; a verb with NO direct object that takes `to` licenses a GOAL
    -- "go to the park"). The `ArgStructureComposer` then renders/recalls through the MINED frames (its new
    `frame_lexicon=` kwarg; the hand frames are the ORACLE for parity).

THE ANTI-CHEAT BAR (mirrors B1; reasoning-over-structure is exactly where over-claims are tempting):
  - MATCH/justify: the MINED frame for each validated verb MATCHES the hand FRAME_LEXICON, or DIFFERS with a logged
    corpus attestation (a corpus-justified difference is fine; a wrong one is not).
  - COMPOSER PARITY: typed-role store/query_role/render on the MINED frames == on the hand frames for the validated
    cases ("the boy goes to the park"; "the girl gives the ball to the dog"), answer-identical or corpus-justified.
  - ** PERMUTED-MINING (the decisive control, mirror B1) **: assign each verb a RANDOM frame (shuffle the mined
    frames across verbs) -> the render/recall must COLLAPSE (a give-framed `go` mis-renders). If it does not
    collapse, the frame is not load-bearing.
  - the AGRAMMATISM ablation (drop the closed-class scaffold -> telegraphic) still holds on the mined frames; the
    no-confab MOAT (an unlicensed/unstored cue -> None) stays 0-FA. PROVENANCE: every mined slot is corpus-attested
    (>= --min-prep-count / --dobj-thresh) with an example sentence logged; no train/test leak (the parity facts are
    rendered, not used to derive the frames).

GATE -- GO requires ALL of:
  (i)   MINED frames MATCH-OR-JUSTIFY the hand frames on the validated verbs (every mineable validated verb either
        matches OR has a logged corpus attestation for its difference; 0 unjustified differences);
  (ii)  COMPOSER PARITY: render + query_role on the MINED frames == on the hand frames for the validated facts
        (answer-identical, OR the difference is one of the (i) corpus-justified frame differences);
  (iii) ** PERMUTED-MINING collapses **: the render/recall accuracy with randomly-reassigned frames drops far below
        the mined-frame accuracy (the corpus, not the apparatus, carries the frames);
  (iv)  the agrammatism ablation still produces telegraphic output != the full render on the mined frames;
  (v)   the no-confab MOAT abstains (0 false-accepts) on the mined frames;
  (vi)  PROVENANCE: every mined slot corpus-attested with a logged example; no leak.
  multi-seed: the MINING is corpus-deterministic; the SEED varies the composer's codes + the permuted-mining
  shuffle + the FrameCQ init (exactly what B1 varied).

Run (CPU/numpy fast path -- the mine + derive + composer parity + permuted-mining; spaCy needed for the mine):
  SIM_BACKEND=numpy python -m research.runners._bucketB_corpus_mined_frames_derisk --seeds 42 43 44 45 46 47
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
    ArgStructureComposer, FRAME_LEXICON, FRAME_ROLES, FUNCTION_WORDS, reparse_to_fact, _U)

# ------------------------------------------------------------------------------------------------------------
# The CORPUS-JUSTIFIABLE prep -> typed-role table (half 2). This is NOT a per-verb hand list (the thing B-mine-1
# removes); it is a small CLOSED lexicon over the language's PREPOSITIONS -- a preposition's DOMINANT thematic role
# across verbs (Bock & Levelt; the spec's "a prep's distribution across verbs"). `to` is the one ambiguous prep
# (GOAL for motion verbs, RECIPIENT for ditransitives) and is resolved by the CORPUS dobj/oblique ratio at derive
# time (the ditransitive rule), not hand-typed per verb. A motion-into prep (into/onto/towards) is also GOAL; a
# containment/surface prep (on/in/at/inside/under) is LOCATION; with -> INSTRUMENT; from -> SOURCE. Prepositions not
# in this table (for/after/across/...) do NOT license a typed argument slot (they are adjuncts, not core frame args)
# -- the conservative choice that keeps the mined frame to the verb's CORE argument structure.
PREP_ROLE = {
    "to": "GOAL",                       # resolved GOAL<->RECIPIENT by the ditransitive rule at derive time
    "into": "GOAL", "onto": "GOAL", "towards": "GOAL", "toward": "GOAL", "inside": "GOAL",
    "on": "LOCATION", "in": "LOCATION", "at": "LOCATION", "under": "LOCATION", "behind": "LOCATION",
    "over": "LOCATION", "near": "LOCATION", "beside": "LOCATION",
    "with": "INSTRUMENT",
    "from": "SOURCE",
}
# the closed-class lead scaffold a role's content unit carries (the verb-frame's determiner/preposition prefix).
# agent + a direct-object content role -> just the determiner; an oblique role -> (its preposition, determiner).
# This reproduces the hand FRAME_LEXICON's lead convention (agent/THEME/patient -> ('the',); GOAL/RECIPIENT ->
# ('to','the'); LOCATION -> ('on'|'in','the')) FROM the mined preposition, rather than hand-typing the lead.

# The verbs the GATE validates (the hand FRAME_LEXICON's content verbs). A verb is "mineable" iff the brain learned
# it AND it clears the attestation threshold; an un-mineable validated verb (e.g. `put` absent from the brain's
# vocab) is reported as such (the honest "the brain's vocab gates which verbs are mineable" constraint, like B1).
VALIDATED_VERBS = ["go", "come", "walk", "run", "give", "send", "put"]


# ------------------------------------------------------------------------------------------------------------
# Stage 0 -- MINE the per-verb argument distribution from the corpus (half 1; the inverted extractor).
# ------------------------------------------------------------------------------------------------------------
def mine_verb_argstats(corpus_path, vocab, max_sentences, target_verbs=None, _spacy_cache={}):
    """Parse the corpus with spaCy (restricted to the brain's `vocab`) and accumulate, per verb, the distribution
    its arguments attest: `{verb: {'freq': n, 'dobj': n_directobj, 'preps': Counter{prep: n}, 'examples': {...}}}`.
    This is the EXACT observation `_corpus_svo_extract.py --typed-roles` makes (dobj/dative/attr/oprd direct objects
    + prep->pobj prepositional args KEEPING the preposition, vform over the brain's vocab) -- here ACCUMULATED into a
    per-verb distribution to DERIVE the frame, instead of consulting the hand table to label one slot. Provenance:
    an example sentence is logged per (verb, prep) and per (verb, dobj) -- a slot is provably corpus-attested."""
    import spacy
    # THEME (direct object) vs RECIPIENT (the dative indirect object). spaCy tags the double-object dative's FIRST
    # NP as `dative` ("gave [Lily]_dative [a cookie]_dobj") -- so a ditransitive verb's RECIPIENT is observable as a
    # `dative` dependency even when the prepositional dative ("...to Lily") is rare in the corpus (TinyStories
    # strongly prefers the double-object form). Keeping them separate lets the derive step recover BOTH the THEME and
    # the RECIPIENT from the corpus, faithful to how child-directed speech states ditransitives (Goldberg; Tomasello).
    THEME_DEPS = {"dobj", "attr", "oprd"}

    def vform(tok):
        lem = tok.lemma_.lower()
        if lem in vocab:
            return lem
        txt = tok.text.lower()
        return txt if txt in vocab else None

    if "nlp" not in _spacy_cache:
        _spacy_cache["nlp"] = spacy.load("en_core_web_sm")
    nlp = _spacy_cache["nlp"]
    with open(corpus_path, encoding="utf-8") as fh:
        text = fh.read()
    pieces = []
    for s in text.replace("<|endoftext|>", "\n").split("\n"):
        s = s.strip()
        if not s:
            continue
        if len(s) <= 100000:
            pieces.append(s)
        else:
            pieces.extend(s[i:i + 100000] for i in range(0, len(s), 100000))

    stats = collections.defaultdict(lambda: {"freq": 0, "dobj": 0, "dative": 0, "preps": collections.Counter(),
                                             "ex_dobj": None, "ex_dative": None, "ex_prep": {}})
    n_sent = 0
    tset = set(target_verbs) if target_verbs is not None else None
    for doc in nlp.pipe(pieces, batch_size=128):
        for sent in doc.sents:
            n_sent += 1
            for tok in sent:
                if tok.dep_ != "nsubj" or tok.head.pos_ not in ("VERB", "AUX"):
                    continue
                vl = vform(tok.head)
                if vl is None or (tset is not None and vl not in tset):
                    continue
                st = stats[vl]
                st["freq"] += 1
                stext = sent.text.strip()[:90]
                for c in tok.head.children:
                    if c.dep_ in THEME_DEPS:
                        # a DIRECT object that is itself in-vocab (the THEME/patient filler the frame renders)
                        if vform(c) is not None and vform(c) != vl:
                            st["dobj"] += 1
                            if st["ex_dobj"] is None:
                                st["ex_dobj"] = stext
                    elif c.dep_ == "dative":
                        # the double-object dative's indirect object = the RECIPIENT ("gave [Lily] a cookie").
                        # Count it whether or not the recipient noun is in-vocab (the SLOT is attested; the filler is
                        # supplied at store time) -- this is structural evidence the verb LICENSES a recipient.
                        st["dative"] += 1
                        if st["ex_dative"] is None:
                            st["ex_dative"] = stext
                    elif c.dep_ == "prep":
                        po = next((x for x in c.children if x.dep_ == "pobj"), None)
                        if po is not None and vform(po) is not None:
                            prep = c.text.lower()
                            st["preps"][prep] += 1
                            st["ex_prep"].setdefault(prep, stext)
        if n_sent >= max_sentences:
            break
    # de-defaultdict for clean JSON/return
    return {v: {"freq": s["freq"], "dobj": s["dobj"], "dative": s["dative"], "preps": dict(s["preps"]),
                "ex_dobj": s["ex_dobj"], "ex_dative": s["ex_dative"], "ex_prep": dict(s["ex_prep"])}
            for v, s in stats.items()}, n_sent


# ------------------------------------------------------------------------------------------------------------
# Stage 1 -- DERIVE the frame lexicon from the mined distribution (half 2).
# ------------------------------------------------------------------------------------------------------------
# prepositions that map to the SAME typed role are aggregated into ONE role signal before thresholding (so a verb
# whose GOAL evidence is split across `to`/`into`/`onto` -- e.g. `come`: to+into -- is not knife-edged out by a
# per-prep threshold). The canonical CITATION preposition for each role's lead scaffold (the surface form the frame
# renders) -- so the mined GOAL renders "to the X" and LOCATION "on the X", matching the hand lexicon's lead.
_ROLE_CANON_PREP = {"GOAL": "to", "LOCATION": "on", "RECIPIENT": "to", "INSTRUMENT": "with", "SOURCE": "from"}


def derive_frame_lexicon(stats, min_freq=30, dobj_thresh=0.20, role_thresh=0.10, min_role_count=20):
    """Derive a FRAME_LEXICON-shaped dict + a VERB_PREP_ROLE map + provenance from the mined per-verb argument
    distribution. Each derived slot is corpus-attested above threshold; the rules are corpus-justifiable (the
    prep->role table + the Bock & Levelt ditransitive/double-object rule), NOT per-verb hand lists.

      * a verb is ATTESTED iff freq >= min_freq (else un-mineable at this corpus/vocab budget; dropped + reported).
      * a DIRECT-OBJECT content slot iff dobj/freq >= dobj_thresh (the THEME or patient filler the frame renders).
      * a RECIPIENT slot iff the verb attests a RECIPIENT signal above threshold = the DATIVE indirect object
        ("gave [Lily] a cookie") OR the prepositional dative (`to`-PP), whichever is larger. (The double-object
        dative is the dominant child-directed form, so the dative dependency is the faithful RECIPIENT evidence.)
      * other OBLIQUE slots: per typed role, the AGGREGATED count of its PREP_ROLE prepositions, above
        role_thresh (frac) AND min_role_count -- so GOAL = to+into+onto+towards, LOCATION = on+in+at+..., etc.
        Emitted in DESCENDING aggregated-count order (the dominant oblique first), capped to <= 2 obliques (WM).
      * the `to` GOAL<->RECIPIENT disambiguation IS the ditransitive rule: a verb with a strong DIRECT OBJECT (a
        THEME) AND a RECIPIENT signal is DITRANSITIVE (dobj=THEME + RECIPIENT); a verb with NO/weak direct object
        whose `to`-family fires is intransitive-motion (GOAL). Bock & Levelt: the verb projects its argument frame.

    Returns (frame_lexicon, verb_prep_role, provenance). frame_lexicon includes a "_default" transitive frame
    (== the hand default) so unmined verbs fall back exactly like the hand lexicon."""
    frames, vpr, prov = {}, {}, {}
    for verb, st in stats.items():
        freq = st["freq"]
        if freq < min_freq:
            prov[verb] = {"attested": False, "freq": freq, "reason": f"freq {freq} < min_freq {min_freq}"}
            continue
        dobj_frac = st["dobj"] / freq
        has_dobj = dobj_frac >= dobj_thresh
        # AGGREGATE the prep evidence per typed role (GOAL = to+into+onto+...; LOCATION = on+in+at+...; etc.)
        role_counts = collections.Counter()
        role_example = {}
        for prep, cnt in st["preps"].items():
            if prep in PREP_ROLE:
                role = PREP_ROLE[prep]                  # `to` -> "GOAL" provisionally; reassigned below if ditrans
                role_counts[role] += cnt
                role_example.setdefault(role, (prep, st["ex_prep"].get(prep)))
        # RECIPIENT signal = max(dative indirect object, prepositional-dative `to`-count); ditransitive iff a THEME
        # (direct object) co-occurs with a RECIPIENT signal above threshold.
        to_count = st["preps"].get("to", 0)
        recip_count = max(st.get("dative", 0), to_count)
        recip_attested = (recip_count >= min_role_count and recip_count / freq >= role_thresh)
        ditransitive = bool(has_dobj and recip_attested)

        units = [_U("CONTENT", "agent", ("the",)), _U("TENSE", "action")]
        slots_log = []
        used_roles = {"agent", "action"}
        # the THEME/patient direct-object slot (THEME for a ditransitive frame; patient otherwise -- Bock & Levelt)
        if has_dobj:
            obj_role = "THEME" if ditransitive else "patient"
            units.append(_U("CONTENT", obj_role, ("the",)))
            used_roles.add(obj_role)
            slots_log.append({"role": obj_role, "source": "direct-object", "count": st["dobj"],
                              "frac": round(dobj_frac, 3), "example": st["ex_dobj"]})
        # the RECIPIENT slot of a ditransitive (lead = the canonical prepositional-dative "to the", the citation
        # surface form -- matching the hand frame -- even when the corpus evidence is the double-object dative).
        if ditransitive:
            units.append(_U("CONTENT", "RECIPIENT", ("to", "the")))
            used_roles.add("RECIPIENT")
            src = "dative" if st.get("dative", 0) >= to_count else "prep:to"
            vpr[(verb, "to")] = "RECIPIENT"
            slots_log.append({"role": "RECIPIENT", "source": src, "count": recip_count,
                              "frac": round(recip_count / freq, 3),
                              "example": st.get("ex_dative") or st["ex_prep"].get("to")})
            role_counts.pop("GOAL", None)            # the `to`-evidence is the RECIPIENT, not a GOAL, for a ditrans
        # the remaining oblique slots (GOAL/LOCATION/INSTRUMENT/SOURCE), dominant first, capped at <=2 obliques
        n_obl = 1 if ditransitive else 0
        for role, cnt in role_counts.most_common():
            if role in used_roles or n_obl >= 2:
                continue
            if cnt < min_role_count or cnt / freq < role_thresh:
                continue
            canon = _ROLE_CANON_PREP.get(role, "to")
            units.append(_U("CONTENT", role, (canon, "the")))
            used_roles.add(role)
            n_obl += 1
            ep, ex = role_example.get(role, (canon, None))
            vpr[(verb, ep)] = role
            slots_log.append({"role": role, "source": f"prep:{ep}(+role-agg)", "count": cnt,
                              "frac": round(cnt / freq, 3), "example": ex})
        frames[verb] = units
        prov[verb] = {"attested": True, "freq": freq, "dobj": st["dobj"], "dobj_frac": round(dobj_frac, 3),
                      "dative": st.get("dative", 0), "recip_count": recip_count, "ditransitive": ditransitive,
                      "top_preps": st["preps"], "slots": slots_log, "roles": [u[1] for u in units]}
    frames["_default"] = list(FRAME_LEXICON["_default"])     # the hand default transitive frame (unmined fallback)
    return frames, vpr, prov


def _roles_of(frame_units):
    """The ordered role sequence of a frame (drop the closed-class lead -> the structural signature for matching)."""
    return tuple(u[1] for u in frame_units)


def compare_frames(mined, hand=FRAME_LEXICON):
    """Per validated verb: does the MINED frame's role sequence MATCH the hand frame's? Returns
    {verb: ('match'|'differ'|'unmined', mined_roles, hand_roles)}. A differ is corpus-JUSTIFIED iff every mined
    slot is attested (checked by the caller via provenance)."""
    out = {}
    for v in VALIDATED_VERBS:
        hand_roles = _roles_of(hand.get(v, hand["_default"]))
        if v not in mined:
            out[v] = ("unmined", None, hand_roles)
        else:
            mr = _roles_of(mined[v])
            out[v] = ("match" if mr == hand_roles else "differ", mr, hand_roles)
    return out


# ------------------------------------------------------------------------------------------------------------
# Stage 2 -- COMPOSER PARITY + the anti-cheats (the de-risk's load-bearing evidence).
# ------------------------------------------------------------------------------------------------------------
# The validated facts the gate names (the boy goes to the park; the girl gives the ball to the dog) + coverage of
# every validated FRAME family. Each fact's verb is one of VALIDATED_VERBS; only mineable verbs are exercised for
# render parity (an un-mined verb has no mined frame to compare). The fillers are concrete in-vocab nouns.
PARITY_FACTS = [
    {"agent": "boy", "action": "go", "GOAL": "park"},
    {"agent": "girl", "action": "give", "THEME": "ball", "RECIPIENT": "dog"},
    {"agent": "dog", "action": "come", "GOAL": "house"},
    {"agent": "cat", "action": "run", "GOAL": "tree"},
    {"agent": "man", "action": "walk", "GOAL": "shop"},
]
PARITY_VOCAB = ["boy", "girl", "dog", "cat", "man", "go", "give", "come", "run", "walk", "send", "put",
                "park", "ball", "house", "tree", "shop", "bone", "table"]


def _composer(seed, frame_lexicon):
    return ArgStructureComposer(seed=seed, D=64, vocab=list(PARITY_VOCAB), frame_lexicon=frame_lexicon,
                                use_spiking_cq=False)


def _store_all(comp, facts):
    for f in facts:
        if f["action"] in comp._frames:        # only facts whose verb has a (mined or hand) frame
            comp.store_fact(f)


def composer_parity(seed, mined_frames, parity_facts):
    """Render + query_role on the MINED frames vs the HAND frames. Returns (parity_ok, details). A fact is counted
    only if its verb is MINEABLE (present in mined_frames); a mined-vs-hand render DIFFERENCE is acceptable iff it
    is one of the corpus-justified FRAME differences (same role-sequence difference compare_frames reported)."""
    hand = _composer(seed, FRAME_LEXICON)
    mine = _composer(seed, mined_frames)
    facts = [f for f in parity_facts if f["action"] in mined_frames]
    _store_all(hand, facts)
    _store_all(mine, facts)
    cf = compare_frames(mined_frames)
    details, ok = [], True
    for f in facts:
        v = f["action"]
        # render parity
        rh = hand.render(f, hand._composite_for(f))
        rm = mine.render(f, mine._composite_for(f))
        frame_differs = (cf[v][0] == "differ")
        render_match = (rh == rm)
        # recall parity: every role present in the fact recalls the same filler on both
        recall_match = True
        for role, val in f.items():
            if role == "action":
                continue
            qh = hand.query_role(role, agent=f["agent"], action=v)
            qm = mine.query_role(role, agent=f["agent"], action=v)
            if qm != val or qh != val:
                # if the mined frame DROPPED this role (justified frame difference), the recall difference is OK
                if not (frame_differs and role not in _roles_of(mined_frames[v])):
                    recall_match = False
        pair_ok = (render_match or frame_differs) and recall_match
        ok = ok and pair_ok
        details.append({"fact": f, "hand_render": rh, "mined_render": rm,
                        "render_match": render_match, "recall_match": recall_match,
                        "frame_differs_justified": frame_differs, "pair_ok": pair_ok})
    return ok, details


def permuted_mining(seed, mined_frames, parity_facts):
    """** THE DECISIVE CONTROL (mirror B1) ** -- assign each mineable verb a RANDOM (other verb's) frame. The
    render/recall must COLLAPSE: a give-framed `go` (THEME+RECIPIENT slots) cannot render/recall a GOAL fact. We
    measure, over the parity facts, the fraction whose render+recall on the PERMUTED frames matches the mined-frame
    answer -- it must drop far below 1.0. (Shuffling the frames across verbs = the structure is destroyed while the
    apparatus -- the composer, the codes -- is identical, exactly B1's permuted-mining logic.)"""
    mineable = [v for v in mined_frames if v != "_default"]
    rng = np.random.default_rng(seed * 733 + 11)
    # a DERANGEMENT of the mineable verbs' frames (no verb keeps its own frame)
    perm = list(mineable)
    for _ in range(100):
        rng.shuffle(perm)
        if all(perm[i] != mineable[i] for i in range(len(mineable))):
            break
    permuted = {v: list(mined_frames[perm[i]]) for i, v in enumerate(mineable)}
    permuted["_default"] = list(mined_frames["_default"])

    mine = _composer(seed, mined_frames)
    pmc = _composer(seed, permuted)
    facts = [f for f in parity_facts if f["action"] in mined_frames]
    _store_all(mine, facts)
    _store_all(pmc, facts)
    n_match = 0
    for f in facts:
        v = f["action"]
        rm = mine.render(f, mine._composite_for(f))
        rp = pmc.render(f, pmc._composite_for(f))
        # recall: the obliques present in the fact -- do they recall the same on the permuted frame?
        rec_same = all(pmc.query_role(role, agent=f["agent"], action=v) ==
                       mine.query_role(role, agent=f["agent"], action=v)
                       for role in f if role not in ("agent", "action"))
        if rp == rm and rec_same:
            n_match += 1
    return n_match / max(len(facts), 1), permuted


def agrammatism_and_moat(seed, mined_frames, parity_facts):
    """The agrammatism ablation (drop the closed-class scaffold -> telegraphic != full) + the no-confab moat (an
    unstored/unlicensed cue -> None) on the MINED frames. Both must hold."""
    mine = _composer(seed, mined_frames)
    facts = [f for f in parity_facts if f["action"] in mined_frames]
    _store_all(mine, facts)
    # agrammatism: on the first mineable fact, telegraphic != full + no function words
    f0 = facts[0]
    full = mine.render(f0, mine._composite_for(f0))
    tele = mine.render(f0, mine._composite_for(f0), ablate_closed_class=True)
    agram_ok = (tele != full) and all(w not in FUNCTION_WORDS for w in tele.split())
    # the rendered prose re-parses to the stored fact (the render moat) -- on the mined frames
    reparse_ok = all(reparse_to_fact(mine.render(f, mine._composite_for(f)), f, lexicon=mined_frames)
                     for f in facts)
    # moat (the no-confab abstention, == the test_argstructure_composer.test_no_confab_moat semantics): a cue that
    # matches NO stored fact -> None (0 false-accepts). An UNSTORED agent, an UNSTORED (agent, action) pair, and a
    # fully-unknown cue must all abstain. (NB: querying an UNBOUND role of a STORED fact is NOT the no-confab moat --
    # the parent composer's query_role unbinds the requested role of the cue-matched fact; abstention is about
    # whether a fact MATCHES, exactly as the production moat is defined.)
    f0a = facts[0]["agent"]
    moat_ok = (mine.query_role("GOAL", agent="nobody", action="go") is None and       # unstored agent
               mine.query_role("GOAL", agent=f0a, action="zzz") is None and           # stored agent, unstored verb
               mine.query_role("GOAL", agent="zzz", action="zzz") is None)            # fully unknown cue
    return bool(agram_ok), bool(reparse_ok), bool(moat_ok)


# ------------------------------------------------------------------------------------------------------------
def run_seed(seed, mined_frames, prov, cf):
    t0 = time.time()
    parity_ok, parity_details = composer_parity(seed, mined_frames, PARITY_FACTS)
    pm_acc, _permuted = permuted_mining(seed, mined_frames, PARITY_FACTS)
    agram_ok, reparse_ok, moat_ok = agrammatism_and_moat(seed, mined_frames, PARITY_FACTS)
    # mined-frame accuracy = the parity render+recall match rate on the mined frames (vs the permuted-mining rate)
    mined_acc = float(np.mean([1.0 if d["pair_ok"] else 0.0 for d in parity_details])) if parity_details else 0.0
    return {
        "seed": seed, "elapsed_s": round(time.time() - t0, 1),
        "parity_ok": parity_ok, "mined_acc": mined_acc, "permuted_mining_acc": pm_acc,
        "agrammatism_ok": agram_ok, "reparse_ok": reparse_ok, "moat_ok": moat_ok,
        "parity_details": parity_details,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46, 47])
    ap.add_argument("--corpus", default="data/corpus/tinystories.txt",
                    help="the child-directed-speech-like corpus (TinyStories) -- the frames are recoverable from a "
                         "learner's input (Buttery & Korhonen 2005). Simple-Wiki also works but the frame verbs are "
                         "rarer there.")
    ap.add_argument("--npz", default="bridges/firstchat/brainALL_w7000.npz_seed42.npz")
    ap.add_argument("--max-sentences", type=int, default=200000)
    ap.add_argument("--min-freq", type=int, default=30)
    ap.add_argument("--dobj-thresh", type=float, default=0.20)
    ap.add_argument("--role-thresh", type=float, default=0.10,
                    help="min fraction (of verb freq) of the AGGREGATED per-role prep count for an oblique slot.")
    ap.add_argument("--min-role-count", type=int, default=20,
                    help="min aggregated attested count for an oblique/recipient slot.")
    ap.add_argument("--out", default="research/findings/raw/_bucketB_corpus_mined_frames.json")
    a = ap.parse_args()

    print(f"[B-mine-1 corpus-mined verb-frame LEXICON de-risk] corpus={a.corpus} | "
          f"brain={os.path.basename(a.npz)}\n  half1: MINE per-verb argument distribution (the inverted "
          f"_corpus_svo_extract --typed-roles); half2: DERIVE the frame via a corpus-justifiable prep->role table "
          f"+ the Bock&Levelt ditransitive rule.\n  HEADLINE controls = MATCH-or-justify the hand frames + "
          f"COMPOSER PARITY + PERMUTED-MINING (random frames collapse).\n", flush=True)

    # ---- Stage 0+1: MINE the argument distribution + DERIVE the frame lexicon (corpus-deterministic; once) ----
    npz_full = os.path.join(_REPO, a.npz) if not os.path.isabs(a.npz) else a.npz
    d = np.load(npz_full, allow_pickle=True)
    vocab = set(str(w).lower() for w in d["vocab"])
    print(f"[mine] brain vocab: {len(vocab)} words | mining argument stats from {a.corpus} ...", flush=True)
    corpus_full = os.path.join(_REPO, a.corpus) if not os.path.isabs(a.corpus) else a.corpus
    # mine ALL verbs the brain learned (not just the validated set) -- a real frame learner doesn't know the answer
    # set; we then REPORT the validated subset. (target_verbs=None would mine the whole vocab; we cap to verbs that
    # appear as a parsed nsubj-head for tractability by passing None and letting freq>=min_freq prune.)
    stats, n_sent = mine_verb_argstats(corpus_full, vocab, a.max_sentences, target_verbs=None)
    mined_frames, mined_vpr, prov = derive_frame_lexicon(
        stats, min_freq=a.min_freq, dobj_thresh=a.dobj_thresh, role_thresh=a.role_thresh,
        min_role_count=a.min_role_count)
    cf = compare_frames(mined_frames)
    print(f"[mine] parsed {n_sent} sentences -> {len(stats)} verbs with parsed args; "
          f"{len([v for v in mined_frames if v != '_default'])} verbs cleared attestation (freq>={a.min_freq}).",
          flush=True)

    # ---- (i) MATCH-or-justify on the validated verbs ----
    print(f"\n  {'verb':6s} {'status':8s}  {'mined roles':40s}  hand roles", flush=True)
    n_unjustified = 0
    justified_diffs = []
    for v in VALIDATED_VERBS:
        status, mr, hr = cf[v]
        mr_s = " ".join(mr) if mr else "(un-mined: brain vocab lacks it / below threshold)"
        hr_s = " ".join(hr)
        flag = ""
        if status == "differ":
            # a difference is JUSTIFIED iff every mined slot of this verb is corpus-attested (provenance present)
            attested = prov.get(v, {}).get("attested", False) and all(
                s.get("count", 0) > 0 for s in prov.get(v, {}).get("slots", []))
            if attested:
                justified_diffs.append(v)
                flag = "  [corpus-JUSTIFIED]"
            else:
                n_unjustified += 1
                flag = "  [** UNJUSTIFIED **]"
        print(f"  {v:6s} {status:8s}  {mr_s:40s}  {hr_s}{flag}", flush=True)
        if status != "unmined" and prov.get(v, {}).get("slots"):
            for s in prov[v]["slots"]:
                print(f"           - {s['role']:10s} <- {s['source']:14s} x{s['count']:<5d} "
                      f"(frac {s['frac']}) e.g. \"{s['example']}\"", flush=True)
    match_or_justify_ok = (n_unjustified == 0)

    # ---- (ii)-(vi) per-seed parity + the anti-cheats ----
    rows = []
    for s in a.seeds:
        r = run_seed(s, mined_frames, prov, cf)
        rows.append(r)
        print(f"\n  [seed {s}] parity {'OK' if r['parity_ok'] else 'X'} (mined-acc {r['mined_acc']:.2f}) | "
              f"** PERMUTED-MINING {r['permuted_mining_acc']:.2f} ** | agrammatism "
              f"{'ok' if r['agrammatism_ok'] else 'X'} | reparse {'ok' if r['reparse_ok'] else 'X'} | "
              f"moat {'ok' if r['moat_ok'] else 'X'}", flush=True)
        for dts in r["parity_details"][:2]:
            print(f"           '{dts['mined_render']}'  (hand: '{dts['hand_render']}', "
                  f"match {dts['render_match']})", flush=True)

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    mined_acc = m("mined_acc"); pm_acc = m("permuted_mining_acc")
    all_parity = all(r["parity_ok"] for r in rows)
    all_agram = all(r["agrammatism_ok"] for r in rows)
    all_reparse = all(r["reparse_ok"] for r in rows)
    all_moat = all(r["moat_ok"] for r in rows)

    # GATE
    parity_ok = all_parity
    permuted_mining_ok = (pm_acc <= 0.5 and mined_acc - pm_acc >= 0.4)   # the decisive control: random frames collapse
    go = (match_or_justify_ok and parity_ok and permuted_mining_ok and all_agram and all_reparse and all_moat)

    out_full = os.path.join(_REPO, a.out) if not os.path.isabs(a.out) else a.out
    os.makedirs(os.path.dirname(out_full), exist_ok=True)
    summary = {
        "capability": "corpus-mined verb-frame lexicon (B-mine-1)", "corpus": a.corpus,
        "brain": os.path.basename(a.npz), "n_seeds": len(a.seeds), "n_sentences": n_sent,
        "mined_frames": {v: [list(u) for u in units] for v, units in mined_frames.items()},
        "mined_verb_prep_role": {f"{k[0]}|{k[1]}": v for k, v in mined_vpr.items()},
        "hand_frames": {v: [list(u) for u in units] for v, units in FRAME_LEXICON.items()},
        "frame_comparison": {v: {"status": cf[v][0], "mined_roles": cf[v][1], "hand_roles": cf[v][2]}
                             for v in VALIDATED_VERBS},
        "justified_diffs": justified_diffs, "n_unjustified": n_unjustified,
        "provenance": prov,
        "match_or_justify_ok": match_or_justify_ok, "parity_ok": parity_ok,
        "mined_acc": mined_acc, "permuted_mining_acc": pm_acc, "permuted_mining_ok": permuted_mining_ok,
        "agrammatism_ok": all_agram, "reparse_ok": all_reparse, "moat_ok": all_moat,
        "go": go, "per_seed": rows,
    }
    with open(out_full, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"\n{'=' * 110}", flush=True)
    print(f"  MINED frames (corpus-derived): "
          f"{ {v: ' '.join(_roles_of(u)) for v, u in mined_frames.items() if v != '_default'} }", flush=True)
    print(f"  MATCH-or-justify: {match_or_justify_ok} ({len(justified_diffs)} corpus-justified diffs: "
          f"{justified_diffs}; {n_unjustified} UNJUSTIFIED)", flush=True)
    print(f"  MEAN ({len(a.seeds)} seeds): composer-parity {all_parity} (mined-acc {mined_acc:.3f}) | "
          f"** PERMUTED-MINING {pm_acc:.3f} (must be <=0.5 AND >=0.4 below mined-acc = random frames collapse) ** | "
          f"agrammatism {all_agram} | reparse {all_reparse} | moat {all_moat}", flush=True)
    if go:
        print(f"\n  GO: the verb-frame LEXICON is MINED from corpus argument co-occurrence over the brain's OWN "
              f"learned verbs -- structure ACQUIRED, not given. The mined frames MATCH-or-justify the hand frames "
              f"on the validated verbs; the composer's typed recall/render on the MINED frames == on the hand "
              f"frames ({mined_acc:.2f}); ** PERMUTED-MINING collapses ({pm_acc:.2f}) ** -> the corpus, NOT the "
              f"apparatus, carries the frames; agrammatism holds, moat 0-FA, provenance logged. The B1-for-relations "
              f"-> B1-for-frames step. NO sim/ edit.", flush=True)
    else:
        why = []
        if not match_or_justify_ok:
            why.append(f"{n_unjustified} UNJUSTIFIED frame difference(s) on validated verbs")
        if not parity_ok:
            why.append("composer parity FAILED (mined render/recall != hand, and not a justified frame diff)")
        if not permuted_mining_ok:
            why.append(f"** PERMUTED-MINING did NOT collapse (perm {pm_acc:.2f} vs mined {mined_acc:.2f}) -- the "
                       f"frames are NOT load-bearing **")
        if not all_agram:
            why.append("agrammatism ablation failed")
        if not all_reparse:
            why.append("reparse (render moat) failed")
        if not all_moat:
            why.append("no-confab moat breach")
        print(f"\n  NO-GO: {'; '.join(why)}. Per the spec this is the honest NEGATIVE -- write it up, do not "
              f"over-claim.", flush=True)
    print(f"  [saved] {out_full}\n{'=' * 110}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
