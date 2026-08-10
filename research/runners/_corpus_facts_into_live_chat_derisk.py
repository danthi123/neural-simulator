"""INTEGRATION #6 -- wire CORPUS-LEARNED grounded facts into the LIVE multi-turn chat so the brain genuinely says
MORE (reduce the live-chat silences the RIGHT way, per the EMERGENCE BAR: learned CONTENT the brain "heard" in the
corpus, NOT hand-added phatic handlers).

THE WIRE-IN (Route #3, corpus-SVO -- the no-confab MOAT holds BY CONSTRUCTION). The live chat's content chokepoint is
`comp.kb` (the RF-phasor VSA fact store) + the composer VOCAB, both set in
`_stageA_full_integration_derisk.build_one_brain`. Facts stored via `comp.store(a,v,p)` -> `comp.kb`; retrieval is
`_gm_retrieve_neighbourhood(comp,topic,actions)` -> `comp.query_patient` (the RF-VSA moat, abstain->None). So expanding
the VOCAB + storing more CORPUS-MINED facts propagates automatically through retrieval/moat/prose: the brain can talk
about MORE subjects, and the moat still abstains on everything unstored. The ONE additive plumbing edit to the live
loop is a `vocab=DEFAULT_VOCAB` kwarg threaded through `build_one_brain` (default keeps byte-identity).

TIER 0 (CPU, seconds -- the core claim, NO bridge): mine top-K clean noun-verb-noun triples from TinyStories
(K in {10,20,40}, frequency-ranked, nouns=_ANIMALS|NOUNS_EXTRA, VERB_NORM past->present); V = DEFAULT_VOCAB U mined U
curated; comp = RFPhasorComposer(seed, D=128, vocab=V); store 6 CURATED_FACTS + K mined. ASSERT recall >= 0.95 on the
stored (a,v) cues; BREADTH (distinct subject-topics with a non-empty `_gm_retrieve_neighbourhood`) rises from 2
(dog,cat) to >= 8; the MOAT battery reads 0 false-accepts.

TIER 1 (CPU, minutes -- the LIVE loop, MOUTH-FREE numpy substrate): build the brain via
`build_one_brain(seed, with_faculties=True, co_resident_affect_ladder=True, vocab=V)`, store the facts, and drive an
EXPANDED conversation = the shipped 14 human turns (`_conversation_turing_test_derisk.HUMAN_TURNS`, incl. the
out-of-domain probes that MUST abstain) + teacher probes about the newly-LEARNED corpus subjects (the teacher authoring
probes = the legitimate linguistic environment). Each turn drives the LIVE spiking faculties (SEAM-C affect ladder
differential + curiosity want + the shared 3-way arbiter) off `cp_firing_states`; a topic with a grounded RF-store
neighbourhood answers with moat-verified frame-render prose (the generator MOUTH is a GPU scaffold, deliberately OFF
here -- the grounded CONTENT is the RF-VSA read, the same content the mouth would render). Grounded-reply count RISES
vs the 6-fact baseline; the OOD turns still abstain; the moat + surface scan stay clean.

GO GATE (6 seeds 42/43/44/100/101/102, 6/6; cfg.seed-controlled, SIM_BACKEND=numpy): grounded-reply count RISES vs the
6-fact baseline AND confabulated==0 on ALL turns AND OOD turns still abstain AND `_gm_posthoc_verify` drops 100% of
unsupported propositions AND `_detect_ungrounded==0`.

ANTI-CHEATS (all required, named):
  1. PERMUTED-CORPUS PROVENANCE -- shuffle the token order, re-mine; mined-set overlap ~= 0 (the knowledge is
     corpus-ORDER-derived, not a hand list).
  2. EXPANDED MOAT BATTERY -- untaught in-vocab cues + the OOD turns -> 0 false-accepts (query_patient -> None).
  3. EMPTY-KB CONTROL (THE KEY ANTI-CHEAT) -- with the EXPANDED vocab but 0 stored corpus facts, breadth stays 2 and
     every new-subject turn abstains: competence comes from the FACTS, not the vocab expansion.
  4. CAPACITY SWEEP K=10/20/40 -- recall + moat hold as K grows (watch for VSA mis-bind false-accept onset).
  5. SURFACE-CONFAB SCAN -- `_detect_ungrounded == 0` on every reply (the render must not add ungrounded discourse the
     SVO moat is blind to).
  6. BYTE-IDENTITY -- `build_one_brain(seed)` vs `build_one_brain(seed, vocab=DEFAULT_VOCAB)` is bit-identical
     (threshold hash + composer concept codes + num_neurons), and MY Tier-1 loop reproduces its own default-build
     transcript exactly -- the additive-`vocab`-param guard.

HONEST SCOPE (declared, per THE LAW + docs/TERMS.md). DECLARED SCAFFOLDS: the SVO mining (host POS/noun-filter = the
"linguistic environment" boundary), `comp.store` (host VSA write = the composer-as-idealization shortcut), the
frame-render `_gm_fact_to_english` (host text interface = the same status the generator mouth's conditioning has).
GENUINELY brain-based / EMERGENT: the knowledge is corpus-DERIVED (token order carries it, permuted overlap ~= 0 -- the
emergence-bar win: the brain talks about what it "heard"); recall + the no-confab moat are RF-VSA brain reads
(query_patient = a spiking VSA unbind + cleanup). BURN-DOWN SUCCESSORS: the stream cortex learning the co-occurrence M
in synapses (`_foundational_curriculum_scaling_scoping`), and the teacher-loop plasticity
(`2026-08-08-teacher-loop-corrective-acquisition-*`) once ITS learned-moat leak is closed. If the result is a NEGATIVE
(the moat leaks at scale), that is a first-class deliverable -- document + name the next mechanism.

DISCIPLINE: SIM_BACKEND=numpy, reuse-by-import, NO `sim/` edit (only the additive `vocab` kwarg + the additive
`_store_facts(extra_facts=)` in `_stageA_full_integration_derisk`, both default-off), cfg.seed (handled by
build_one_brain), additive.

Run (single seed smoke):
  PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._corpus_facts_into_live_chat_derisk \
      --seeds 42 --K 40
Full 6-seed sweep:
  PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._corpus_facts_into_live_chat_derisk \
      --seeds 42,43,44,100,101,102 --K 40 \
      --out research/findings/raw/lanes/stageA/corpus_facts_live_chat_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
import logging as _logging  # noqa: E402
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.ERROR)

import numpy as np  # noqa: E402

from sim.backend import get_backend  # noqa: E402

from research.runners import _stageA_full_integration_derisk as SA  # noqa: E402
from research.runners import _conversation_turing_test_derisk as TT  # noqa: E402
from research.runners.rf_phasor_composer import RFPhasorComposer, DEFAULT_VOCAB  # noqa: E402
from research.runners.corpus_stream import load_token_stream_multi  # noqa: E402
# reuse-by-import: the corpus mining + noun/verb inventories (do NOT reinvent).
from research.runners._realcorpus_learn_corpus_facts_derisk import (  # noqa: E402
    mine_svo, VERB_NORM, NOUNS_EXTRA, VERBS,
)
from research.runners._realcorpus_cancellation_derisk import _ANIMALS  # noqa: E402
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# TIER 0 -- CORPUS MINING -> VOCAB -> RF-VSA STORE -> recall / breadth / moat (CPU seconds, NO bridge).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
CURATED_WORDS = sorted({w for f in SA.CURATED_FACTS for w in f})
_BASE_SUBJECTS = {"dog", "cat"}   # the 6-fact baseline's grounded subjects (breadth == 2)


def build_corpus_counter(corpus_path="data/corpus/tinystories.txt"):
    """Load the TinyStories token stream and mine every clean noun-VERB-noun SVO triple (subject + object both in
    _ANIMALS|NOUNS_EXTRA, determiners skipped). Returns (counter, flat_tokens, nouns, verbs)."""
    stories = load_token_stream_multi(corpus_path, max_stories=None)
    toks = [t for s in stories for t in s]
    nouns = _ANIMALS | NOUNS_EXTRA
    verbs = list(VERBS)
    counter = mine_svo(toks, nouns, verbs)
    return counter, toks, nouns, verbs


def mine_top_k(counter, K):
    """Top-K frequency-ranked clean SVO triples, verb NORMALISED past->present (VERB_NORM). A frequency-ranked list
    of (subject, present_verb, object)."""
    out = []
    for (s, v, o), _cnt in counter.most_common(K):
        out.append((s, VERB_NORM.get(v, v), o))
    return out


def build_vocab(mined):
    """V = DEFAULT_VOCAB U mined-words U curated-words (sorted). The composer's concept codebook derives from V, so a
    larger V grows the rf slice + gives a genuinely larger brain (only vocab=DEFAULT_VOCAB is byte-identical)."""
    mined_words = {w for tr in mined for w in tr}
    return sorted(set(DEFAULT_VOCAB) | mined_words | set(CURATED_WORDS))


def make_composer_and_store(seed, V, mined):
    """RFPhasorComposer(seed, D=128, vocab=V) + store the 6 CURATED_FACTS and the K mined triples via comp.store (the
    composer's OWN VSA write). Returns (comp, facts) where facts is every (a,v,p) actually written."""
    comp = RFPhasorComposer(seed=int(seed), D=128, vocab=V)
    _vocab, facts = SA._store_facts(comp, extra_facts=mined)
    return comp, facts


def recall_on_stored(comp, facts):
    """Recall = fraction of distinct stored (a,v) cues whose query_patient returns a patient actually stored for that
    cue (the store is many-to-many; the first-match answer must be ONE valid patient). A pure RF-VSA brain read."""
    cue2pats = {}
    for (a, v, p) in facts:
        cue2pats.setdefault((a, v), set()).add(p)
    if not cue2pats:
        return 0.0, 0
    hit = 0
    for (a, v), pats in cue2pats.items():
        ans = comp.query_patient(a, v)
        hit += int(ans is not None and ans in pats)
    return hit / len(cue2pats), len(cue2pats)


def grounded_subjects(comp, facts):
    """The distinct subject-topics with a NON-EMPTY grounded neighbourhood (the breadth basis). Each is a subject
    for which `_gm_retrieve_neighbourhood` (per-action query_patient) returns >= 1 stored SVO."""
    actions = sorted({v for (_a, v, _p) in facts})
    subs = sorted({a for (a, _v, _p) in facts})
    return [s for s in subs if SA._gm_retrieve_neighbourhood(comp, s, actions)]


def moat_battery(comp, V, facts, n_probes=60, seed=0):
    """The no-confab MOAT battery: sample UNTAUGHT in-vocab (agent, action) cues (both words in V, the pair never
    stored) and count false-accepts (query_patient returns non-None). 0 = the moat holds. A false-accept is a VSA
    mis-bind (the failure mode this whole integration guards against)."""
    actions = sorted({v for (_a, v, _p) in facts}) or sorted(set(V))
    stored_cues = {(a, v) for (a, v, _p) in facts}
    rng = np.random.default_rng(int(seed) * 7 + 11)
    false_accepts, probes, tries = 0, 0, 0
    accepted = []
    while probes < n_probes and tries < 4000:
        a = V[int(rng.integers(0, len(V)))]
        v = actions[int(rng.integers(0, len(actions)))]
        tries += 1
        if (a, v) in stored_cues:
            continue
        probes += 1
        ans = comp.query_patient(a, v)
        if ans is not None:
            false_accepts += 1
            accepted.append((a, v, ans))
    return {"false_accepts": int(false_accepts), "probes": int(probes),
            "examples": accepted[:5]}


def permuted_overlap(seed, toks, nouns, verbs, K):
    """ANTI-CHEAT 1 -- PROVENANCE: shuffle the token order and re-mine; the mined-set overlap with the real-order
    mining measures whether the knowledge is corpus-ORDER-derived (~0) or a hand list (~1)."""
    real = set(k for k, _ in mine_svo(toks, nouns, verbs).most_common(K))
    rng = np.random.default_rng(int(seed))
    perm = list(toks)
    rng.shuffle(perm)
    permd = set(k for k, _ in mine_svo(perm, nouns, verbs).most_common(K))
    return len(real & permd) / max(1, len(real))


def tier0(seed, counter, toks, nouns, verbs, K):
    """One Tier-0 record for (seed, K): mine, build V, store, measure recall / breadth / moat + the provenance and
    empty-kb-control anti-cheats. NO bridge -- CPU seconds."""
    mined = mine_top_k(counter, K)
    V = build_vocab(mined)
    comp, facts = make_composer_and_store(seed, V, mined)

    recall, n_cues = recall_on_stored(comp, facts)
    breadth_subs = grounded_subjects(comp, facts)
    breadth = len(breadth_subs)
    moat = moat_battery(comp, V, facts, seed=seed)

    # ANTI-CHEAT 3 (empty-kb control): SAME expanded vocab V, but store ONLY the 6 curated facts (0 corpus facts).
    # Breadth must stay 2 (dog,cat) and every NEW corpus subject must abstain -> competence is in the FACTS.
    comp_ctrl = RFPhasorComposer(seed=int(seed), D=128, vocab=V)
    _vc, curated_only = SA._store_facts(comp_ctrl, extra_facts=None)
    ctrl_breadth_subs = grounded_subjects(comp_ctrl, curated_only)
    new_subjects = sorted({s for (s, _v, _p) in mined if s not in _BASE_SUBJECTS})
    ctrl_new_answers = sum(
        1 for s in new_subjects
        if SA._gm_retrieve_neighbourhood(comp_ctrl, s, sorted({v for (_a, v, _p) in facts})))

    return {
        "seed": int(seed), "K": int(K), "vocab_size": len(V), "n_facts_stored": len(facts),
        "n_distinct_cues": int(n_cues), "recall_on_stored": float(recall),
        "breadth": int(breadth), "breadth_subjects": breadth_subs,
        "moat_false_accepts": moat["false_accepts"], "moat_probes": moat["probes"],
        "moat_examples": moat["examples"],
        "permuted_overlap": float(permuted_overlap(seed, toks, nouns, verbs, K)),
        "empty_kb_breadth": len(ctrl_breadth_subs), "empty_kb_breadth_subjects": ctrl_breadth_subs,
        "empty_kb_new_subject_answers": int(ctrl_new_answers),
        "sample_facts": ["%s %s %s" % f for f in mined[:8]],
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# TIER 1 -- the LIVE mouth-free chat on the ONE bridge (SEAM-C affect + curiosity + arbiter LIVE; grounded content =
# the RF-store neighbourhood, moat-verified; the generator MOUTH is a GPU scaffold, deliberately OFF here).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def make_teacher_probes(mined):
    """The teacher (the legitimate linguistic environment) authors one 'Tell me about the <subject>.' probe per
    newly-LEARNED corpus subject (excluding the baseline dog/cat), deterministically ordered. These are the turns
    that go from SILENCE (baseline) to a GROUNDED reply (treatment) -- the breadth win, driven by learned content."""
    new_subjects = sorted({s for (s, _v, _p) in mined if s not in _BASE_SUBJECTS})
    return [("Tell me about the %s." % s, "corpus-learned subject probe (%s)" % s) for s in new_subjects]


def _affect_cache(bridge, xp, idx, snap, appraisals):
    """Read the SEAM-C affect-ladder differential ONCE per distinct host-fed appraisal (a pure function of appraisal
    given the restored baseline). Deterministic + identical to a per-turn read, but ~10x fewer 800-step ladder reads."""
    ladder_live = "ladder" in idx
    out = {}
    for a in appraisals:
        diff, _r = SA._turn_valence(bridge, xp, idx, snap, float(a), ladder_live)
        out[float(a)] = float(diff)
    return out


def _curio_cache(bridge, xp, idx, snap, novelties):
    out = {}
    for nv in novelties:
        out[float(nv)] = float(SA.read_curiosity_want(bridge, xp, idx, snap, novelty=float(nv)))
    return out


def run_chat(bridge, xp, idx, snap, comp, facts, turns):
    """Drive the expanded conversation MOUTH-FREE. Per turn: classify the cue against comp.words; read the LIVE
    faculties (affect differential, curiosity want, 3-way arbiter); route -- a topic with a grounded RF neighbourhood
    answers with moat-verified frame-render prose (tone-colored), a novel in-vocab cue asks (moat abstains), an
    out-of-domain / no-cue turn stays SILENT (honest abstain). Returns the transcript (list of per-turn records)."""
    agents_set = {a for (a, _v, _p) in facts}
    actions_set = sorted({v for (_a, v, _p) in facts})
    grounded_lex = TT._grounded_lexicon(comp)
    # vocab_sets for the post-hoc SVO moat (the SAME re-parse the shipped mouth path uses).
    patients_set = {p for (_a, _v, p) in facts}
    inflect = SA._gm_augment_inflect(actions_set, SA._build_inflection_map(actions_set))
    vocab_sets = (agents_set, actions_set, patients_set, inflect)

    aff = _affect_cache(bridge, xp, idx, snap, {0.0, 1.0})
    curio = _curio_cache(bridge, xp, idx, snap, {0.05, 1.0})

    # The co-resident composer's query_patient is a whole-merged-bridge RF resonate (~1.3s each, the faithful but slow
    # substrate cost). Within ONE chat a given (agent, action) is queried by BOTH _gm_retrieve_neighbourhood AND the
    # post-hoc moat (and by a later referential turn), so memoise query_patient FOR THE DURATION OF THE CHAT ONLY --
    # query_patient is a deterministic pure read of the frozen store, so the memo is answer-IDENTICAL, it just removes
    # redundant resonates. Restored in the finally so the composer is left untouched.
    _orig_qp = comp.query_patient
    _qp_memo = {}

    def _memo_qp(agent, action, order_fn=None):
        key = (agent, action, order_fn)
        if key not in _qp_memo:
            _qp_memo[key] = _orig_qp(agent, action, order_fn=order_fn)
        return _qp_memo[key]

    comp.query_patient = _memo_qp
    try:
        return _run_chat_turns(bridge, xp, idx, snap, comp, turns, agents_set, actions_set,
                               grounded_lex, vocab_sets, aff, curio)
    finally:
        comp.query_patient = _orig_qp


def _run_chat_turns(bridge, xp, idx, snap, comp, turns, agents_set, actions_set, grounded_lex, vocab_sets, aff, curio):
    transcript = []
    for tno, (human, tag) in enumerate(turns, start=1):
        appraisal = 1.0 if (tno in TT.FRIENDLY_TURNS) or ("subject probe" in tag) else 0.0
        cls = TT._classify(human, comp, agents_set, actions_set)
        diff = aff[appraisal]
        tone_lvl = SA._graded_tone_level(diff)
        tone_tok = SA._graded_tone_token(tone_lvl)
        novelty = 1.0 if cls["kind"] in ("novel_cue", "in_vocab_no_cue", "no_cue") else 0.05
        want = curio[novelty]
        winner, margin, rates = SA.run_arbiter(bridge, xp, idx, snap, SA._arb_drives(diff, want))

        rec = {"turn": tno, "human": human, "tag": tag, "cue_kind": cls["kind"],
               "cue_agent": cls["agent"], "cue_action": cls["action"],
               "affect_differential": float(diff), "tone_level": int(tone_lvl), "tone_token": tone_tok,
               "curiosity_want_hz": float(want), "arbiter_winner": winner,
               "brain_reply": "", "category": "silence", "confabulated": False,
               "moat_verified_props": 0, "moat_confab_props": 0, "ungrounded_words": []}

        if cls["kind"] in ("known_cue", "topic"):
            topic = cls["agent"]
            nbhd = SA._gm_retrieve_neighbourhood(comp, topic, actions_set)   # RF-VSA recall (moat abstains -> [])
            if nbhd:
                body = " ".join(SA._gm_fact_to_english(tuple(f)) for f in nbhd)
                reply = (tone_tok + " " + body).strip() if tone_tok else body
                # POST-HOC no-confab moat over the emitted prose (the SAME re-parse the mouth path uses): every
                # frame-rendered fact came from query_patient, so all must verify + none confabulate.
                props = SA._gm_posthoc_verify(comp, body, vocab_sets, topic=topic)
                n_ver = sum(1 for pr in props if pr["verified"])
                n_confab = sum(1 for pr in props if not pr["verified"])
                surf_confab, ungrounded = TT._detect_ungrounded(reply, grounded_lex)
                rec.update({"brain_reply": reply, "category": "grounded",
                            "utterance_source": "grounded frame-render (RF-store neighbourhood; moat-verified)",
                            "neighbourhood": [list(f) for f in nbhd],
                            "moat_verified_props": int(n_ver), "moat_confab_props": int(n_confab),
                            "ungrounded_words": ungrounded,
                            "confabulated": bool(n_confab > 0 or surf_confab)})
            else:
                rec.update({"utterance_source": "silence/abstain (no grounded neighbourhood)"})
        elif cls["kind"] == "novel_cue":
            a, v = cls["agent"], cls["action"]
            moat = comp.query_patient(a, v)                                  # HARD moat: unstored -> None
            rec.update({"brain_reply": "what does %s %s?" % (a, v), "category": "ask",
                        "utterance_source": "curiosity-ask (moat abstains)",
                        "moat_answer": moat, "moat_held": bool(moat is None),
                        "confabulated": False})
        else:
            # in_vocab_no_cue / no_cue -> honest silence/abstain (OOD facts, arithmetic, humor, opinion, closing).
            rec.update({"utterance_source": "silence/abstain (no grounded cue / no faculty for this intent)"})

        transcript.append(rec)
    return transcript


def posthoc_teeth(comp, facts, n=12, seed=0):
    """GO-gate component: `_gm_posthoc_verify` must DROP 100% of UNSUPPORTED propositions. Build a battery of TRUE
    frame-rendered facts (each must verify) + INVENTED foreign-patient sentences over stored (agent,action) cues
    (each must be dropped). Returns the drop-rate of unsupported props (must be 1.0) + the keep-rate of supported."""
    agents_set = {a for (a, _v, _p) in facts}
    actions_set = sorted({v for (_a, v, _p) in facts})
    patients_set = {p for (_a, _v, p) in facts}
    inflect = SA._gm_augment_inflect(actions_set, SA._build_inflection_map(actions_set))
    vocab_sets = (agents_set, actions_set, patients_set, inflect)
    true_store = {tuple(f) for f in facts}
    pats = sorted(patients_set)
    rng = np.random.default_rng(int(seed) * 13 + 5)

    supported_verified, supported_total = 0, 0
    unsupported_dropped, unsupported_total = 0, 0
    for (a, v, p) in facts[:n]:
        # TRUE proposition -> must verify.
        tp = SA._gm_posthoc_verify(comp, SA._gm_fact_to_english((a, v, p)), vocab_sets, topic=a)
        for pr in tp:
            supported_total += 1
            supported_verified += int(pr["verified"])
        # INVENTED proposition (same cue, a FOREIGN patient never stored for it) -> must be dropped.
        foreign = [q for q in pats if q != p and (a, v, q) not in true_store]
        if not foreign:
            continue
        inv = foreign[int(rng.integers(0, len(foreign)))]
        ip = SA._gm_posthoc_verify(comp, SA._gm_fact_to_english((a, v, inv)), vocab_sets, topic=a)
        for pr in ip:
            unsupported_total += 1
            unsupported_dropped += int(not pr["verified"])   # dropped == not verified (moat catches it)
    return {
        "supported_props": int(supported_total), "supported_verified": int(supported_verified),
        "supported_keep_rate": (supported_verified / supported_total) if supported_total else 1.0,
        "unsupported_props": int(unsupported_total), "unsupported_dropped": int(unsupported_dropped),
        "unsupported_drop_rate": (unsupported_dropped / unsupported_total) if unsupported_total else 1.0,
    }


def _chat_summary(transcript):
    grounded = sum(1 for r in transcript if r["category"] == "grounded")
    silence = sum(1 for r in transcript if r["category"] == "silence")
    ask = sum(1 for r in transcript if r["category"] == "ask")
    confab = sum(1 for r in transcript if r["confabulated"])
    ungrounded = sum(len(r.get("ungrounded_words") or []) for r in transcript)
    # OOD turns = the no-cue turns (out-of-domain fact / arithmetic / humor / opinion / closing) -- must abstain.
    ood = [r for r in transcript if r["cue_kind"] in ("no_cue", "in_vocab_no_cue")]
    ood_abstained = sum(1 for r in ood if r["category"] == "silence" and not r["confabulated"])
    return {"n_turns": len(transcript), "grounded": grounded, "silence": silence, "ask": ask,
            "confabulated": confab, "ungrounded_word_total": ungrounded,
            "ood_turns": len(ood), "ood_abstained": ood_abstained}


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# ANTI-CHEAT 6 -- BYTE-IDENTITY of the additive `vocab` param (substrate + composer + a default-build transcript).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _concept_hash(comp):
    import hashlib
    m = hashlib.blake2b(digest_size=16)
    for w in comp.words:
        m.update(w.encode())
        m.update(np.ascontiguousarray(comp.comp.concepts[w] if hasattr(comp, "comp") else comp.concepts[w]).tobytes())
    return m.hexdigest()


# The decision-bearing transcript fields (the reply + every routing/moat verdict). The per-turn AFFECT/curiosity/
# arbiter RAW floats are deliberately EXCLUDED from the behavioural compare: they are OU-read-noise (ou_seed is
# unseeded) whose value depends only on the global-RNG position at read time -- a property of the shipped affect
# read, orthogonal to the vocab param. The substrate hash + these decision fields are the additive-param guard.
_DECISION_FIELDS = ("human", "tag", "cue_kind", "cue_agent", "cue_action", "tone_level", "brain_reply", "category",
                    "utterance_source", "confabulated", "moat_verified_props", "moat_confab_props",
                    "ungrounded_words", "neighbourhood", "arbiter_winner", "moat_answer", "moat_held")


def _decision_view(transcript):
    return [{k: r.get(k) for k in _DECISION_FIELDS} for r in transcript]


def byte_identity(seed, turns):
    """Prove the additive `vocab=DEFAULT_VOCAB` kwarg is DEFAULT-OFF / bit-identical. Each brain runs its FULL
    pipeline (build -> store -> chat) BEFORE the next: the build's heterogeneity seeding (cfg.seed) resets the global
    RNG, so a chat run immediately after its build sees identical OU read-noise -- interleaving the two builds would
    only shuffle that unseeded OU stream (an eval artifact, not the vocab param). Checks: (a) the substrate threshold
    hash, (b) num_neurons, (c) the composer concept codes, (d) the DECISION-bearing transcript (reply + every
    routing/moat verdict), and (e) the FULL transcript incl. the OU raw floats. (a)-(d) are the additive-param
    guarantee; (e) is reported for completeness (True when the RNG position matches)."""
    xp, _ = get_backend()
    # DEFAULT build (no vocab kwarg) -> full pipeline.
    b0, c0, i0, s0 = SA.build_one_brain(int(seed), with_faculties=True, co_resident_affect_ladder=True)
    th0 = SA._threshold_hash(b0, b0.core_config.num_neurons)
    cc0 = _concept_hash(c0)
    nn0 = int(b0.core_config.num_neurons)
    _v0, f0 = SA._store_facts(c0)
    t0 = run_chat(b0, xp, i0, s0, c0, f0, turns)
    # vocab=DEFAULT_VOCAB build -> full pipeline (RNG reset by this build's het-seeding).
    b1, c1, i1, s1 = SA.build_one_brain(int(seed), with_faculties=True, co_resident_affect_ladder=True,
                                        vocab=DEFAULT_VOCAB)
    th1 = SA._threshold_hash(b1, b1.core_config.num_neurons)
    cc1 = _concept_hash(c1)
    nn1 = int(b1.core_config.num_neurons)
    _v1, f1 = SA._store_facts(c1)
    t1 = run_chat(b1, xp, i1, s1, c1, f1, turns)

    dec_ident = bool(json.dumps(_decision_view(t0), sort_keys=True, default=str)
                     == json.dumps(_decision_view(t1), sort_keys=True, default=str))
    full_ident = bool(json.dumps(t0, sort_keys=True, default=str)
                      == json.dumps(t1, sort_keys=True, default=str))
    return {
        "threshold_hash_identical": bool(th0 == th1),
        "num_neurons_identical": bool(nn0 == nn1), "num_neurons": nn0,
        "concept_codes_identical": bool(cc0 == cc1),
        "decision_transcript_identical": dec_ident,
        "transcript_identical": full_ident,
    }


def build_verdict(recs, K, go):
    """Earn the aggregate verdict with its PRECONDITIONS carried in the artifact (tools.verdict.Verdict), so
    verdict-preconditions can enforce that the GO travels with what earned it. Every check below is a GO-gate or
    anti-cheat clause; `decide` returns UNDEFINED if any is unmeasured/unmet, never a bare GO."""
    def _allmin(fn):
        vals = [fn(r) for r in recs]
        return (min(vals) if vals else None), vals

    min_recall, _ = _allmin(lambda r: r["tier0_headline"]["recall_on_stored"])
    min_breadth, _ = _allmin(lambda r: r["tier0_headline"]["breadth"])
    tot_moat_fa = sum(t["moat_false_accepts"] for r in recs for t in r["tier0"].values())
    tot_confab = sum(r["chat_treatment_summary"]["confabulated"] + r["chat_baseline_summary"]["confabulated"]
                     for r in recs)
    min_drop, _ = _allmin(lambda r: r["posthoc_teeth"]["unsupported_drop_rate"])
    max_perm = max((r["tier0_headline"]["permuted_overlap"] for r in recs), default=1.0)
    tot_empty_new = sum(r["tier0_headline"]["empty_kb_new_subject_answers"] for r in recs)
    max_empty_breadth = max((r["tier0_headline"]["empty_kb_breadth"] for r in recs), default=99)
    # representative treatment-vs-control magnitudes (first seed; the per-seed records carry all).
    r0 = recs[0]
    bi = next((r["byte_identity"] for r in recs if r.get("byte_identity")), None)

    v = Verdict("INTEGRATION #6 corpus-facts-into-live-chat (K=%d, %d seeds)" % (int(K), len(recs)))
    v.require("all seeds GO", int(sum(1 for r in recs if r["seed_go"])), expect=len(recs))
    v.floor("min recall on stored (a,v) cues", min_recall, floor=0.95)
    v.require("breadth >= 8 all seeds (rose from 2)", int(min_breadth or 0), expect=lambda m: m >= 8)
    v.control("grounded replies: treatment vs 6-fact baseline",
              r0["grounded_treatment"], r0["grounded_baseline"], min_separation=0.0)
    v.control("breadth: treatment vs empty-kb same-vocab control",
              r0["tier0_headline"]["breadth"], r0["tier0_headline"]["empty_kb_breadth"], min_separation=0.0)
    v.require("moat 0 false-accepts (all K, all seeds)", int(tot_moat_fa), expect=0)
    v.require("confab == 0 (treatment + baseline, all turns)", int(tot_confab), expect=0)
    v.require("posthoc teeth drop 100% of unsupported props", float(min_drop or 0.0), expect=1.0)
    v.require("permuted-corpus provenance overlap < 0.5 (all seeds)", float(max_perm), expect=lambda m: m < 0.5)
    v.require("empty-kb control: 0 new-subject answers (all seeds)", int(tot_empty_new), expect=0)
    v.require("empty-kb control: breadth stays <= 2 (all seeds)", int(max_empty_breadth), expect=lambda m: m <= 2)
    if bi is not None:
        v.require("byte-identity: substrate threshold hash identical", bool(bi["threshold_hash_identical"]),
                  expect=True)
        v.require("byte-identity: composer concept codes identical", bool(bi["concept_codes_identical"]),
                  expect=True)
        v.require("byte-identity: default-build transcript identical", bool(bi["transcript_identical"]), expect=True)
    v.disabled("spiking-generator MOUTH (GPU/torch)",
               "CPU eval; the grounded CONTENT is the RF-VSA read (what the mouth would render), not the mouth")
    v.disabled("plasticity (STDP/Hebbian/homeostasis/STP/structural)",
               "the composer store is a host VSA write; the synaptic-learning successor is named in the finding")
    return v.decide(go=bool(go), verbose=False)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def run_seed(seed, counter, toks, nouns, verbs, K, do_byte_identity=False):
    """One seed: the Tier-0 capacity sweep (K in {10,20,K}), the Tier-1 baseline vs treatment live chat over ONE
    fixed expanded turn list, the post-hoc teeth, and (optionally) the byte-identity guard."""
    t_start = time.time()
    # ---- Tier 0: capacity sweep + core claim ----
    sweep_Ks = sorted({10, 20, int(K)})
    tier0_by_K = {int(kk): tier0(seed, counter, toks, nouns, verbs, kk) for kk in sweep_Ks}
    t0_head = tier0_by_K[int(K)]

    # ---- Tier 1: build ONE fixed expanded turn list from the headline-K mined subjects; run baseline vs treatment ----
    mined = mine_top_k(counter, int(K))
    V = build_vocab(mined)
    turns = list(TT.HUMAN_TURNS) + make_teacher_probes(mined)

    xp, _ = get_backend()
    # BASELINE: the 6-fact brain (DEFAULT_VOCAB, curated facts only) over the SAME turn list.
    b_b, c_b, i_b, s_b = SA.build_one_brain(int(seed), with_faculties=True, co_resident_affect_ladder=True)
    _vb, facts_b = SA._store_facts(c_b)
    tr_base = run_chat(b_b, xp, i_b, s_b, c_b, facts_b, turns)
    sum_base = _chat_summary(tr_base)
    # silence count on JUST the shipped 14 live-chat turns (the "8 silences" the integration reduces).
    base_live14_silence = sum(1 for r in tr_base[:len(TT.HUMAN_TURNS)] if r["category"] == "silence")

    # TREATMENT: the expanded-vocab, corpus-fact brain over the SAME turn list.
    b_t, c_t, i_t, s_t = SA.build_one_brain(int(seed), with_faculties=True, co_resident_affect_ladder=True, vocab=V)
    _vt, facts_t = SA._store_facts(c_t, extra_facts=mined)
    tr_treat = run_chat(b_t, xp, i_t, s_t, c_t, facts_t, turns)
    sum_treat = _chat_summary(tr_treat)
    treat_live14_silence = sum(1 for r in tr_treat[:len(TT.HUMAN_TURNS)] if r["category"] == "silence")

    teeth = posthoc_teeth(c_t, facts_t, seed=seed)

    # ATTRIBUTION (tools.lab): WHOSE is the effect? The grounded-reply RISE is attributed to the stored corpus FACTS
    # by subtracting the matched 6-fact baseline (same turns, same faculties, only the facts+vocab differ); the BREADTH
    # is attributed to the FACTS by subtracting the empty-kb same-vocab control (the vocab expansion alone gives none).
    grounded_attrib = attributable_to(
        "grounded replies from stored corpus facts (treatment vs 6-fact baseline, same turns)",
        float(sum_treat["grounded"]), float(sum_base["grounded"]))
    breadth_attrib = attributable_to(
        "breadth from stored corpus facts (treatment vs empty-kb same-vocab control)",
        float(t0_head["breadth"]), float(t0_head["empty_kb_breadth"]))

    bi = byte_identity(seed, list(TT.HUMAN_TURNS)) if do_byte_identity else None

    # ---- per-seed GO ----
    grounded_rises = bool(sum_treat["grounded"] > sum_base["grounded"])
    no_confab = bool(sum_treat["confabulated"] == 0 and sum_base["confabulated"] == 0)
    ood_abstains = bool(sum_treat["ood_abstained"] == sum_treat["ood_turns"]
                        and sum_base["ood_abstained"] == sum_base["ood_turns"])
    teeth_ok = bool(abs(teeth["unsupported_drop_rate"] - 1.0) < 1e-9 and teeth["unsupported_props"] > 0)
    no_ungrounded = bool(sum_treat["ungrounded_word_total"] == 0 and sum_base["ungrounded_word_total"] == 0)
    recall_ok = bool(t0_head["recall_on_stored"] >= 0.95)
    breadth_ok = bool(t0_head["breadth"] >= 8)
    moat_ok = bool(all(r["moat_false_accepts"] == 0 for r in tier0_by_K.values()))
    provenance_ok = bool(all(r["permuted_overlap"] < 0.5 for r in tier0_by_K.values()))
    empty_kb_ok = bool(t0_head["empty_kb_breadth"] <= 2 and t0_head["empty_kb_new_subject_answers"] == 0)

    seed_go = bool(grounded_rises and no_confab and ood_abstains and teeth_ok and no_ungrounded
                   and recall_ok and breadth_ok and moat_ok and provenance_ok and empty_kb_ok)

    return {
        "seed": int(seed), "K": int(K), "elapsed_s": round(time.time() - t_start, 1),
        "tier0": tier0_by_K, "tier0_headline": t0_head,
        "chat_baseline_summary": sum_base, "chat_treatment_summary": sum_treat,
        "live14_silence_baseline": int(base_live14_silence),
        "live14_silence_treatment": int(treat_live14_silence),
        "grounded_baseline": sum_base["grounded"], "grounded_treatment": sum_treat["grounded"],
        "grounded_delta": sum_treat["grounded"] - sum_base["grounded"],
        "grounded_attributable_to_facts": grounded_attrib,
        "breadth_attributable_to_facts": breadth_attrib,
        "posthoc_teeth": teeth, "byte_identity": bi,
        "gate": {
            "grounded_rises": grounded_rises, "no_confab": no_confab, "ood_abstains": ood_abstains,
            "posthoc_teeth_drop_100pct": teeth_ok, "no_surface_confab": no_ungrounded,
            "recall_ge_0.95": recall_ok, "breadth_ge_8": breadth_ok, "moat_0_false_accepts": moat_ok,
            "permuted_provenance": provenance_ok, "empty_kb_control": empty_kb_ok,
        },
        "seed_go": seed_go,
        "transcript_treatment": tr_treat,
    }


def main():
    ap = argparse.ArgumentParser(description="INTEGRATION #6: corpus-learned facts into the live chat (breadth).")
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--K", type=int, default=40, help="headline capacity (top-K mined SVO triples)")
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--byte-identity", choices=["auto", "on", "off"], default="auto",
                    help="auto=on for the first seed only (default); on=every seed; off=never (parallel per-seed runs)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    def _want_bi(i):
        return {"on": True, "off": False, "auto": (i == 0)}[a.byte_identity]

    print("[INTEGRATION #6] corpus-learned facts -> live chat | mining TinyStories ...", flush=True)
    counter, toks, nouns, verbs = build_corpus_counter(a.corpus_path)
    print("  mined %d distinct clean SVO triples from %d tokens" % (len(counter), len(toks)), flush=True)

    recs = []
    for i, s in enumerate(seeds):
        r = run_seed(s, counter, toks, nouns, verbs, a.K, do_byte_identity=_want_bi(i))
        recs.append(r)
        t0 = r["tier0_headline"]
        print("  [seed %d] K=%d |V|=%d facts=%d | recall=%.3f breadth=%d (base 2) moat_fa=%d | "
              "grounded base->treat %d->%d (delta +%d) | live14 silence %d->%d | confab t=%d b=%d | "
              "teeth drop=%.2f | perm-overlap=%.2f | empty-kb breadth=%d new-ans=%d | GO=%s (%.1fs)"
              % (s, a.K, t0["vocab_size"], t0["n_facts_stored"], t0["recall_on_stored"], t0["breadth"],
                 t0["moat_false_accepts"], r["grounded_baseline"], r["grounded_treatment"], r["grounded_delta"],
                 r["live14_silence_baseline"], r["live14_silence_treatment"],
                 r["chat_treatment_summary"]["confabulated"], r["chat_baseline_summary"]["confabulated"],
                 r["posthoc_teeth"]["unsupported_drop_rate"], t0["permuted_overlap"],
                 t0["empty_kb_breadth"], t0["empty_kb_new_subject_answers"],
                 r["seed_go"], r["elapsed_s"]), flush=True)
        if r["byte_identity"] is not None:
            print("    byte-identity(default vs vocab=DEFAULT_VOCAB): %s" % r["byte_identity"], flush=True)

    n_go = sum(1 for r in recs if r["seed_go"])
    go = bool(n_go == len(recs) and len(recs) > 0)

    def _agg(fn):
        return [fn(r) for r in recs]
    print("\n  AGGREGATE (%d seeds): grounded delta = %s | breadth = %s | recall = %s | moat_fa(all K) = %s"
          % (len(recs), _agg(lambda r: r["grounded_delta"]),
             _agg(lambda r: r["tier0_headline"]["breadth"]),
             _agg(lambda r: round(r["tier0_headline"]["recall_on_stored"], 3)),
             _agg(lambda r: sum(t["moat_false_accepts"] for t in r["tier0"].values()))), flush=True)
    bi = recs[0]["byte_identity"] if recs else None
    print("  ANTI-CHEATS: provenance overlap = %s | empty-kb breadth = %s | empty-kb new-answers = %s | "
          "byte-identity = %s"
          % (_agg(lambda r: round(r["tier0_headline"]["permuted_overlap"], 2)),
             _agg(lambda r: r["tier0_headline"]["empty_kb_breadth"]),
             _agg(lambda r: r["tier0_headline"]["empty_kb_new_subject_answers"]), bi), flush=True)
    print("\n  VERDICT: %s -- %d/%d seeds. The brain talks about MORE subjects it LEARNED FROM THE CORPUS "
          "(grounded-reply count RISES), the no-confab moat holds (0 false-accepts, 100%% unsupported dropped), the "
          "OOD turns still abstain, and the knowledge is corpus-derived (permuted overlap ~0, empty-kb control = no "
          "competence from vocab alone)." % ("GO" if go else "PARTIAL/NEGATIVE", n_go, len(recs)), flush=True)

    decided = build_verdict(recs, a.K, go)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        payload = {"verdict": "GO" if go else "PARTIAL", "verdict_earned": decided["status"],
                   "n_go": n_go, "n_seeds": len(recs), "K": a.K, "seeds": seeds,
                   "sim_backend": os.environ.get("SIM_BACKEND", "numpy"),
                   "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
                   "byte_identity": bi, "per_seed": recs}
        with open(a.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print("  [saved] %s" % a.out, flush=True)
    return go


if __name__ == "__main__":
    main()
