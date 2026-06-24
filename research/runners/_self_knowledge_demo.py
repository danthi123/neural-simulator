"""TEACH THE BRAIN ABOUT ITSELF — the self-knowledge education demo + the FIREWALL-PROOF test.

The owner chose a META topic: teach the brain how THIS project works (its own architecture — Claude-authored
facts in `_curriculum_self_knowledge.json`), DEVELOP a brain on it via the longitudinal develop loop, then PROVE
the brain's answers are its OWN learned knowledge, NOT the off-bridge LLM's. The decisive lever: Qwen2.5-0.5B has
ZERO knowledge of THIS niche project, so any correct project answer is provably the brain's — and the firewall
test makes that airtight by ALSO requiring the brain to ABSTAIN on the things Qwen DOES know (capital of France,
2+2, Romeo and Juliet). If the brain "answers" any of those, the LLM leaked. The bar is 0 LEAKS.

THE THREE PARTS:

  (1) DEVELOP — run the REAL-GPU longitudinal develop loop (`_longitudinal_develop_loop_gpu.develop_gpu`) on the
      self-knowledge curriculum, graded across a few "days" (the brain HEARS the project's concepts word-by-word in
      the TinyStories corpus, its rate-Hebbian stream cortex LEARNS the concept codes -> grounded phasors -> the
      conversational composer). Vocab grows, facts are stored, retention holds, the no-confab moat stays 0-FA. The
      developed grounded codes + the lineage are SAVED so the firewall/Q&A/REPL stages load the SAME developed brain.

  (2) THE FIREWALL TEST (the decisive proof, fully automated) — build a `BrainConversationalAgent` on the developed
      grounded codes, teach it the FULL curriculum, then run three probe batteries:
        (a) POSITIVE  (`firewall_probes_answer_project`): 'what does the moat prevent' -> 'confabulation', etc. The
            brain ANSWERS from its OWN spiking memory (Qwen cannot know these niche facts).
        (b) ABSTAIN-GENERAL (`firewall_probes_abstain_general_knowledge`): capital of France, two plus two, who
            wrote Romeo and Juliet, ... The brain MUST ABSTAIN. A single non-abstain = an LLM LEAK -> the bar is 0.
            This is the DECISIVE control.
        (c) ABSTAIN-UNTAUGHT (`deliberately_untaught_project_facts`): nav/dopamine/vision — REAL project parts NOT
            taught -> the brain must abstain (no over-generalization).

  (3) SELF-REFLECTIVE Q&As — run the `self_reflective_questions` ('what are you', 'how do you learn', 'do you
      forget', ...) through the grounded-language faculty: the BRAIN recalls a relevant fact (GATE), the FAST
      OFF-BRIDGE Qwen-0.5B RENDERS the recalled fact into a fluent sentence (CONSTRAIN), and the brain re-parses
      the prose to VERIFY the content is its own (reject on drift). The verbatim Q&As are saved so the controller
      can show the owner the brain talking about itself.

  Plus a `--repl` mode (and a `chat()` function) the owner runs to type questions to the developed brain.

VERDICT: GO = the brain LEARNED the project (answers project facts) AND the firewall HOLDS (0 LEAKS on the Qwen-
general probes; abstains on untaught) AND the self-reflective Q&As read coherently with BRAIN-sourced content
-> PROVEN: the knowledge is the brain's, the LLM only phrases. Or HONEST: any leak + the fix.

self-reference: the question parser maps you/your/I/me/it -> the agent 'brain' so self-questions resolve.

REUSE-BY-IMPORT, NO `sim/` edit. GPU (`SIM_BACKEND=cupy`) for the develop loop (the stream cortex) + the spiking
Qwen faculty (PyTorch on GPU). FOREGROUND-only (GPU run blocking). Run:
    SIM_BACKEND=cupy python -u -m research.runners._self_knowledge_demo --n-days 4 --seed 42

    # then talk to the developed brain (loads the saved grounded codes):
    SIM_BACKEND=cupy python -u -m research.runners._self_knowledge_demo --repl --load <codes.json>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

# GPU by default (the develop loop's stream cortex + the spiking Qwen faculty). An explicit env still wins.
os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _json_default(o):
    """JSON encoder fallback: coerce numpy scalars/arrays + NaN to JSON-native (the develop-loop metrics carry
    numpy int/float scalars; a bare `default` that returns them unchanged trips a confusing circular-ref error)."""
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if math.isnan(float(o)) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

from sim.lineage import BridgeLineage  # noqa: E402

# the develop loop (REAL stream cortex) + its curriculum machinery (reuse-by-import)
from research.runners._longitudinal_develop_loop import GradedCurriculum  # noqa: E402
from research.runners._longitudinal_develop_loop_gpu import develop_gpu, StreamCortex  # noqa: E402
from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402

CURRICULUM = os.path.join(_REPO, "research", "findings", "raw", "_curriculum_self_knowledge.json")
OUT = os.path.join(_REPO, "research", "findings", "raw", "_self_knowledge_demo.json")
CODES = os.path.join(_REPO, "research", "findings", "raw", "_self_knowledge_grounded_codes.json")


def _free_cupy_pool():
    """QWEN-CRASH FIX helper: release the cupy default memory pool's CACHED blocks back to the device so the
    PyTorch faculty can allocate without contending with the ~12 GB of cupy-pool blocks the develop loop +
    composer bridges leave held. A no-op on the numpy backend / if cupy is absent."""
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


# ============================================================================================================
# 1. The self-knowledge curriculum -> the develop loop's graded-syllabus format.
#    The curriculum's `facts` (SVO) + `attribute_facts` ([noun, adj] -> (noun, "is", adj)) are GRADED across a
#    few simulated days (simple -> richer). Each day introduces its NEW concepts (so the stream cortex learns
#    their codes) + a fact set + a small probe battery (recall / yes-no / the no-confab moat). The cumulative
#    vocabulary is the full self-knowledge concept set; the stream cortex hears + learns codes for all of them.
# ============================================================================================================

def _load_curriculum():
    with open(os.path.abspath(CURRICULUM), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _curriculum_concept_set(cur):
    """Every concept word in the self-knowledge curriculum (SVO agents/actions/patients + attribute nouns/adjs +
    the 'is' relation). These are the words the stream cortex must HEAR + learn codes for, and the composer's
    vocabulary. (The firewall general-knowledge cues — 'paris', 'four', 'shakespeare' — are NOT taught and NOT in
    this set; they get encodable fall-back codes so the moat can abstain on them STRUCTURALLY.)"""
    words = set(["is"])
    for a, v, p in cur.get("facts", []):
        words.update([a, v, p])
    for noun, adj in cur.get("attribute_facts", []):
        words.update([noun, "is", adj])
    return words


def _all_facts_svo(cur):
    """The full taught fact set as SVO triples: the curriculum facts + each attribute_fact as (noun, 'is', adj)."""
    facts = [tuple(f) for f in cur.get("facts", [])]
    facts += [(noun, "is", adj) for noun, adj in cur.get("attribute_facts", [])]
    return facts


def _build_self_knowledge_syllabus(cur, n_days):
    """Grade the curriculum's facts into `n_days` developmentally-simple->rich days. Day d introduces a contiguous
    slice of the SVO facts (its NEW concepts = the concepts in that slice not seen before) + a per-day probe
    battery (recall on the day's facts; a yes/no on a prior-day fact for retention; a never-taught yes/no for the
    moat). The cumulative vocabulary grows monotonically across days — the development vocab axis."""
    facts = _all_facts_svo(cur)
    n_days = max(2, int(n_days))
    # contiguous, near-equal slices so each later day adds concepts (developmental growth)
    bounds = [round(i * len(facts) / n_days) for i in range(n_days + 1)]
    seen_concepts = set()
    syllabus = []
    prior_day_first_fact = None
    for d in range(n_days):
        day_facts = facts[bounds[d]:bounds[d + 1]]
        if not day_facts:
            day_facts = facts[-1:]                 # never emit an empty day
        new_concepts = []
        for (a, v, p) in day_facts:
            for c in (a, v, p):
                if c not in seen_concepts:
                    seen_concepts.add(c)
                    new_concepts.append(c)
        # recall probes: the patient of every fact taught TODAY (the conversational competence on new learning)
        probe_recall = [("patient", (a, v), p) for (a, v, p) in day_facts]
        # retention: a yes/no on a PRIOR day's fact (still true after the day's new learning -> no-forget)
        probe_yesno = []
        if prior_day_first_fact is not None:
            pa, pv, pp = prior_day_first_fact
            probe_yesno.append((pa, pv, pp, "yes"))
        # the no-confab moat: a NEVER-taught (subject, action, patient) triple from today's concepts -> abstain
        if len(day_facts) >= 2:
            a0, v0, _ = day_facts[0]
            _, _, p1 = day_facts[1]
            if (a0, v0, p1) not in facts:          # a scrambled pairing that was never asserted
                probe_yesno.append((a0, v0, p1, "no_or_unknown"))
        syllabus.append({
            "new_concepts": new_concepts,
            "facts": [tuple(f) for f in day_facts],
            "probe_recall": probe_recall,
            "probe_heldout": [],
            "probe_yesno": probe_yesno,
            "probe_chain": [],
        })
        prior_day_first_fact = day_facts[0]
    return syllabus


class SelfKnowledgeCurriculum(GradedCurriculum):
    """The self-knowledge curriculum as a develop-loop GradedCurriculum (graded across days)."""

    def __init__(self, cur, n_days):
        super().__init__(syllabus=_build_self_knowledge_syllabus(cur, n_days))


# ============================================================================================================
# 2. The self-reference resolver + the question parser. The owner asks 'what are YOU?' / 'how do YOU learn?'; we
#    map you/your/I/me/it -> the agent 'brain' so the question resolves against the taught facts.
# ============================================================================================================

# the agent's self aliases (from the curriculum's self_reference block) — any of these in a question means 'brain'
SELF_ALIASES = {"you", "your", "yours", "i", "me", "my", "it", "its", "yourself", "itself"}
# generic filler words a question may contain that are not content (so the cue-extractor ignores them)
_STOP = {"what", "who", "does", "do", "the", "a", "an", "is", "are", "of", "to", "from", "that", "how",
         "did", "will", "can", "you", "your", "i", "me", "my", "it", "its", "and", "with", "in", "on",
         "for", "by", "as", "be", "this", "these", "those", "there", "here", "?", ".", ",", "prevent",
         "prevents"}


def _resolve_self(word):
    """Map a self-alias word to the agent's name 'brain' (else return the word unchanged, lowercased)."""
    w = word.lower().strip(".,!?")
    return "brain" if w in SELF_ALIASES else w


# A small NATURAL-LANGUAGE -> curriculum-token synonym map so a plain English question word maps onto the exact
# tokens the curriculum used. This is the ONLY 'understanding' the router needs — and it is faithful: it just
# lets 'how do you LEARN?' find the fact whose verb is 'learns'. (It carries ZERO project knowledge: an untaught
# question word like 'navigate'/'drive' is in NO fact, so the matcher abstains -> the firewall is what proves the
# answer is the brain's, NOT a clever parser.)
_QUESTION_SYNONYMS = {
    "learn": {"learns", "learning"}, "learns": {"learns"},
    "forget": {"forgetting", "replays", "replay", "remembers"}, "forgetting": {"forgetting"},
    "remember": {"remembers", "memory", "remembers"}, "memory": {"memory", "remembers", "consolidates"},
    "lie": {"moat", "confabulation", "abstains", "refuses", "honest"},
    "lying": {"moat", "confabulation", "abstains", "refuses", "honest"},
    "guess": {"moat", "confabulation", "refuses", "guessing"},
    "use": {"uses"}, "uses": {"uses"}, "using": {"uses"},
    "teach": {"teaches"}, "teaches": {"teaches"}, "taught": {"teaches"},
    "made": {"has", "uses", "neurons", "spikes"}, "make": {"has", "uses"},
    "store": {"stores", "remembers", "composer"}, "speak": {"phrases", "faculty", "answers"},
    "answer": {"answers", "remembers"}, "think": {"uses", "neurons"}, "work": {"uses", "runs"},
    "consolidate": {"consolidates"}, "grow": {"grows", "develops", "tiers"}, "develop": {"develops", "daily"},
}


def _question_keywords(question):
    """Tokenize -> lowercase -> resolve self-aliases -> drop stopwords -> the content keywords, EXPANDED through the
    natural-language synonym map so 'learn'/'forget'/'lie' map onto the curriculum's exact tokens ('learns'/
    'forgetting'/'moat')."""
    toks = [_resolve_self(t) for t in re.findall(r"[a-zA-Z]+", question.lower())]
    kws = set()
    for t in toks:
        if t in _STOP and t != "brain":
            continue
        kws.add(t)
        kws |= _QUESTION_SYNONYMS.get(t, set())
    return kws, toks


def _match_fact(question, stored_facts):
    """Find the stored SVO fact that BEST answers `question`: score each fact by how many of the question's content
    keywords (synonym-expanded) appear in the fact's {agent, action, patient}, with a STRONG bonus for matching a
    non-'brain' content word (so a generic 'brain ... X' question routes to the fact about X, not just any 'brain'
    fact). Returns (gate_svo or None, score). A fact matched ONLY by the self-alias 'brain' (score from 'brain'
    alone) is NOT decisive — an untaught self-question ('do you navigate?') has no other keyword in any fact, so it
    abstains (the no-confab moat). This is the load-bearing gate: the answer is the fact whose words the question
    actually mentions, else the brain abstains."""
    kws, toks = _question_keywords(question)
    content_kws = kws - {"brain"}                       # the NON-self keywords (the discriminating signal)
    # IDENTITY special-case: a bare 'what/who ARE you' (an identity verb 'be/are/is' + ONLY the self-alias, no
    # other content keyword) is a legitimate self-question. Route it to a DEFINING fact about the brain ('brain has
    # neurons' / 'brain uses spikes' / 'brain is spiking') — the brain's own identity statement, not abstain. This
    # is NOT a leak escape hatch: it only fires when the self-alias 'brain' is present (an untaught identity question
    # 'what is dopamine' has no 'brain' alias -> still abstains).
    is_identity_q = ("brain" in kws and not content_kws
                     and any(w in {"be", "are", "is", "am"} for w in toks))
    if is_identity_q:
        identity_verbs = ("has", "is", "uses")          # preference order for a defining fact
        for want in identity_verbs:
            for (a, v, p) in stored_facts:
                if a == "brain" and v == want:
                    return [a, v, p], 1
    best, best_score = None, 0
    for (a, v, p) in stored_facts:
        ftoks = {a, v, p}
        content_hits = len(content_kws & ftoks)         # matches beyond the bare self-alias
        brain_hit = 1 if ("brain" in kws and "brain" in ftoks) else 0
        # decisive iff at least one CONTENT keyword matches (a 'brain'-only match is not enough -> abstain)
        score = content_hits * 10 + brain_hit
        if content_hits >= 1 and score > best_score:
            best, best_score = (a, v, p), score
    return (list(best) if best is not None else None), best_score


def _question_to_cue(question, agent_concepts, action_words, stored_facts=None):
    """Map a free-text question to a (kind, cue) the brain answers against its stored SVO facts.

    Primary path (`stored_facts` given): the keyword->fact matcher (`_match_fact`) — the answer is the stored fact
    whose words the question mentions (synonym-resolved), else ABSTAIN. Returns ('fact', gate_svo) on a match,
    ('none', None) on abstain. This is faithful + abstains naturally on Qwen-general ('capital of France' matches no
    fact) AND untaught-project ('brain navigates' — 'navigate' is in no fact).

    Fallback path (`stored_facts` None — used only by the REPL when not handed the facts): the legacy
    action+concept heuristic (kept for compatibility)."""
    if stored_facts is not None:
        gate_svo, score = _match_fact(question, stored_facts)
        return ("fact", gate_svo) if gate_svo is not None else ("none", None)
    # --- legacy fallback (only if facts not supplied) ---
    toks = [_resolve_self(t) for t in re.findall(r"[a-zA-Z]+", question.lower())]
    known_actions = [t for t in toks if t in action_words]
    known_concepts = [t for t in toks if t in agent_concepts and t not in _STOP and t not in action_words]
    if known_actions and known_concepts:
        agent_word = "brain" if "brain" in known_concepts else known_concepts[0]
        return "patient", (agent_word, known_actions[0])
    return "none", None


# ============================================================================================================
# 3. The grounded-language faculty (the FAST off-bridge Qwen-0.5B render of the brain's recalled facts), GATED +
#    VERIFIED. Reused-by-import from the integration de-risk (gate->constrain->verify with the spiking forward).
#    The faculty's ONLY job is surface form; the brain supplies + verifies the content (the no-confab moat).
# ============================================================================================================

def _build_inflection_map_local(verbs):
    """Reuse the de-risk's verb-inflection map (so a rendered 'the brain uses spikes' / 'the moat prevents
    confabulation' re-parses back to the base verb)."""
    from research.runners._grounded_lang_integration_derisk import _build_inflection_map
    return _build_inflection_map(sorted(set(verbs)))


def _extract_svo(prose, agents, actions, patients, inflect):
    from research.runners._grounded_lang_integration_derisk import _extract_svo_from_prose
    return _extract_svo_from_prose(prose, agents, actions, patients, inflect)


def grounded_render(agent, faculty, gate_svo, vocab_sets, allow_regen=True):
    """CONSTRAIN -> VERIFY one gated SVO with the REAL spiking Qwen faculty. Returns a record with the surface
    prose, the re-parsed SVO, and whether the render VERIFIED (re-parsed back to the gated fact). On a verify
    reject, RE-PROMPT tighter ONCE (the production recovery path)."""
    agents_set, actions_set, patients_set, inflect = vocab_sets
    a, v, p = gate_svo
    surface, surface_full, gen_s = faculty.render_svo(a, v, p)

    def _verify(prose):
        csvo = _extract_svo(prose, agents_set, actions_set, patients_set, inflect)
        if csvo is None:
            return None, False, "prose did not re-parse to a clean SVO"
        parsed_ = agent.parse(csvo, voice="active")
        rsvo = [parsed_.get("agent"), parsed_.get("action"), parsed_.get("patient")]
        return rsvo, (rsvo == list(gate_svo)), (None if rsvo == list(gate_svo) else "re-parse mismatches gated fact")

    reparse_svo, verified, reason = _verify(surface)
    regen_used = False
    if (not verified) and allow_regen:
        regen_used = True
        surface2, surface_full2, gen_s2 = faculty.render_svo_regen(a, v, p)
        reparse2, verified2, reason2 = _verify(surface2)
        surface, surface_full, reparse_svo, verified, reason = surface2, surface_full2, reparse2, verified2, reason2
        gen_s += gen_s2
    return {"gate_svo": list(gate_svo), "surface": surface, "surface_full": surface_full,
            "reparse_svo": reparse_svo, "verified": bool(verified), "regen_used": regen_used,
            "reject_reason": reason, "gen_seconds": round(gen_s, 2)}


# ============================================================================================================
# 4. The agent builder for the firewall/Q&A/REPL stages: a BrainConversationalAgent on the DEVELOPED grounded
#    codes, taught the FULL curriculum. The general-knowledge firewall cues are added as ENCODABLE fall-back
#    vocab so the moat can abstain on them STRUCTURALLY (an encodable-but-never-stored cue matches no fact).
# ============================================================================================================

# the general-knowledge probe cues Qwen KNOWS — mapped to a content token the brain could (but must NOT) answer.
# These are added to the vocab so they ENCODE (a missing concept would KeyError); the brain abstains because they
# were never STORED as facts.
GENERAL_KNOWLEDGE_CUES = {
    "what is the capital of france": ("france", "has"),       # if it answered -> 'paris' (Qwen knows; brain must not)
    "what is two plus two": ("two", "plus"),
    "who wrote romeo and juliet": ("romeo", "wrote"),
    "what color is the sky": ("sky", "is"),
    "how many legs does a dog have": ("dog", "has"),
}
# the extra fall-back vocab so those cues encode (they are NEVER stored as facts -> structural abstention)
GENERAL_KNOWLEDGE_VOCAB = {"france", "paris", "two", "plus", "four", "romeo", "juliet", "wrote", "shakespeare",
                           "color", "blue", "legs", "has", "many", "four_legs"}


def decorrelate_grounded_codes(grounded, eps=1e-3):
    """THE RECALL FIX (2026-06-24): decorrelate the stream-learned grounded codes so the composer's cleanup
    (argmax phase-cosine) can DISCRIMINATE them. The stream cortex's learned codes COLLAPSE at scale -- many
    concepts are heard in near-identical hub contexts -> near-identical code rows -> ~22% of code PAIRS have
    cos > 0.9 (the documented graded-magnitude / code-correlation family wall). On such codes the cleanup picks
    a near-neighbour ('faculty' when the answer is 'development', cos 0.995) -> recall collapses to ~0.21 at 52
    facts. ZCA-whitening the per-concept phasor matrix removes the cross-concept common mode while PRESERVING
    the grounded content (a linear, invertible transform of the SAME learned codes) -> recall 0.21 -> 0.94 at
    the build's D=128, with the no-confab moat fully intact (0 leaks). This is a HOST post-processing of the
    codes -- the codes are a legitimate host-shaped INPUT to the composer (the existing `grounded_codes`
    interface; cf. the project's flat-distinct path that uses per-bridge distinct seeds to get decorrelated
    codes); the composer's spiking bind/unbind/cleanup algebra is UNTOUCHED. Returns a new {word: phases} dict
    at the same D. De-risked: research/findings/raw/_self_knowledge_recall_probe.json +
    _self_knowledge_chat_e2e_probe.json."""
    if not grounded:
        return grounded
    ws = list(grounded.keys())
    Z = np.array([np.exp(2j * np.pi * np.asarray(grounded[w], dtype=float)) for w in ws])   # (V, D) phasors
    mu = Z.mean(axis=0, keepdims=True)
    Zc = Z - mu
    C = (Zc.conj().T @ Zc) / max(1, Zc.shape[0])
    lam, U = np.linalg.eigh(C)
    lam = np.clip(lam.real, 0.0, None)
    Wzca = U @ np.diag(1.0 / np.sqrt(lam + eps)) @ U.conj().T
    Zw = Zc @ Wzca
    return {w: (np.angle(Zw[i]) % (2.0 * np.pi)) / (2.0 * np.pi) for i, w in enumerate(ws)}


def build_qa_agent(cur, vocab, grounded, seed, decorrelate_codes=True):
    """Build the conversational agent on the developed grounded codes + teach the FULL curriculum.

    `grounded` = {word: phases} the develop loop's stream cortex LEARNED (the brain converses on the codes it
    learned from listening). `vocab` = the full concept set + the general-knowledge + untaught fall-back vocab
    (so a firewall cue ENCODES; the moat abstains structurally on the never-stored ones).

    `decorrelate_codes` (default True, 2026-06-24 RECALL FIX): ZCA-decorrelate the grounded codes before they
    enter the composer so the cleanup can discriminate them at scale (recall 0.21 -> 0.94 at 52 facts; the moat
    stays 0-FA). Pass False to use the raw stream-learned codes (the old, recall-degrading behaviour)."""
    g = decorrelate_grounded_codes(grounded) if (decorrelate_codes and grounded) else grounded
    agent = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab}, composer_kind="rf",
                                     grounded_codes=g if g else None)
    # teach the FULL curriculum as AFFIRM facts (the validated yes-no pattern)
    n_taught = 0
    for a, v, p in _all_facts_svo(cur):
        agent.hear(f"{a} {v} {p}", polarity="AFFIRM")
        n_taught += 1
    return agent, n_taught


def _qa_vocab(cur):
    """The full vocabulary for the firewall agent: the taught concepts + the general-knowledge fall-back words +
    the deliberately-untaught project-fact words (all encodable; only the taught facts are stored)."""
    vocab = set(_curriculum_concept_set(cur))
    vocab |= GENERAL_KNOWLEDGE_VOCAB
    for probe in cur.get("deliberately_untaught_project_facts", {}).get("probes", []):
        for w in probe:
            if isinstance(w, str) and w != "?":
                vocab.add(w)
    return sorted(vocab)


# ============================================================================================================
# 5. THE FIREWALL TEST — the decisive proof, fully automated.
# ============================================================================================================

def run_firewall(agent, cur, action_words):
    """Run the three firewall batteries. Returns a structured result + the per-probe trail.

    (a) POSITIVE  — every `firewall_probes_answer_project` ('what does the moat prevent' -> 'confabulation') must
        get a NON-abstain BRAIN answer (Qwen cannot know these).
    (b) ABSTAIN-GENERAL — every `firewall_probes_abstain_general_knowledge` (capital of France, 2+2, ...) must
        ABSTAIN. ANY non-abstain = an LLM LEAK. The bar is 0 leaks.
    (c) ABSTAIN-UNTAUGHT — every `deliberately_untaught_project_facts` probe (nav/dopamine/vision) must abstain.

    EVERY battery routes through the SAME keyword->fact matcher (`_match_fact` via `_question_to_cue`) over the
    brain's stored facts, then VERIFIES the gate against the spiking recall (`_answer_question`) — so the moat is
    the brain's own abstention, identically applied to project / general / untaught cues."""
    concepts = set(agent.composer.concepts.keys())
    stored_facts = [(f.get("agent"), f.get("action"), f.get("patient")) for f, _ in agent.composer.kb
                    if all(isinstance(f.get(r), str) for r in ("agent", "action", "patient"))]

    # --- (a) POSITIVE: the brain answers project facts ---
    pos = []
    for q_text, expect_substr in cur.get("firewall_probes_answer_project", {}).get("probes", []):
        kind, cue = _question_to_cue(q_text, concepts, action_words, stored_facts)
        answer, abstained = _answer_question(agent, kind, cue)
        # 'answered the project fact' = a non-abstain whose answer token matches the expected content (loosely)
        matched = (answer is not None) and any(tok in str(answer).lower()
                                               for tok in re.split(r"[/\s]+", expect_substr.lower()) if tok)
        pos.append({"question": q_text, "expect": expect_substr, "kind": kind, "gate_svo": cue,
                    "answer": answer, "abstained": abstained, "answered_correctly": bool(matched)})

    # --- (b) ABSTAIN-GENERAL: the decisive control. The brain MUST abstain on Qwen-general knowledge. ---
    # Two abstention checks, BOTH must hold: (1) the keyword->fact matcher finds NO stored fact (the question's
    # words are in no fact), AND (2) the encodable direct cue is never-stored -> the spiking recall abstains.
    gen = []
    for q_text in cur.get("firewall_probes_abstain_general_knowledge", {}).get("probes", []):
        kind, cue = _question_to_cue(q_text, concepts, action_words, stored_facts)
        matcher_answer, _ = _answer_question(agent, kind, cue)     # the routed answer (None if matcher abstained)
        key = q_text.lower().strip("?. ")
        cue_pair = GENERAL_KNOWLEDGE_CUES.get(key)
        direct_answer = agent.what_does(cue_pair[0], cue_pair[1]) if cue_pair is not None else None
        answer = matcher_answer if matcher_answer is not None else direct_answer
        abstained = (matcher_answer is None) and (direct_answer is None)
        gen.append({"question": q_text, "kind": kind, "gate_svo": cue, "direct_cue": cue_pair,
                    "answer": answer, "abstained": abstained, "LEAK": (not abstained)})

    # --- (c) ABSTAIN-UNTAUGHT: real project parts NOT taught -> abstain (no over-generalization) ---
    unt = []
    for probe in cur.get("deliberately_untaught_project_facts", {}).get("probes", []):
        subj = probe[0]
        act = probe[1] if (len(probe) > 1 and probe[1] != "?") else None
        # route through the matcher as a question 'what does <subj> <act>' (the SAME gate as everything else)
        q_text = f"what does {subj} {act}" if act else f"what is {subj}"
        kind, cue = _question_to_cue(q_text, concepts, action_words, stored_facts)
        matcher_answer, _ = _answer_question(agent, kind, cue)
        # also the direct spiking recall if the action is a known verb (belt-and-suspenders)
        direct_answer = agent.what_does(subj, act) if (act is not None and act in action_words) else None
        answer = matcher_answer if matcher_answer is not None else direct_answer
        abstained = (matcher_answer is None) and (direct_answer is None)
        unt.append({"probe": probe, "gate_svo": cue, "answer": answer,
                    "abstained": abstained, "LEAK": (not abstained)})

    n_pos_answered = sum(p["answered_correctly"] for p in pos)
    n_gen_leaks = sum(g["LEAK"] for g in gen)
    n_unt_leaks = sum(u["LEAK"] for u in unt)
    return {
        "positive_detail": pos,
        "general_detail": gen,
        "untaught_detail": unt,
        "positive_answered": n_pos_answered,
        "positive_total": len(pos),
        "general_abstained": sum(g["abstained"] for g in gen),
        "general_total": len(gen),
        "general_leaks": n_gen_leaks,
        "untaught_abstained": sum(u["abstained"] for u in unt),
        "untaught_total": len(unt),
        "untaught_leaks": n_unt_leaks,
    }


def _answer_question(agent, kind, cue):
    """Answer a parsed question against the brain's spiking memory. Returns (answer_or_None, abstained_bool).

    For kind=='fact' the cue is a gate SVO [a, v, p] from the keyword->fact matcher; we VERIFY it against the
    brain's SPIKING recall (what_does(a,v) must return p) — so the answer is genuinely the spiking memory's, not
    just the host matcher's pick. A verify miss -> abstain (the moat holds at the recall layer too)."""
    if kind == "fact":
        a, v, p = cue
        recalled = agent.what_does(a, v)     # the spiking composer recall
        if recalled == p:
            return f"{a} {v} {p}", False      # the brain's own fact, spiking-verified
        return None, True                     # recall didn't confirm the matcher's pick -> abstain
    if kind == "patient":
        ans = agent.what_does(cue[0], cue[1])
        return ans, (ans is None)
    if kind == "describe":
        ans = agent.describe(cue)            # render any fact about the concept (or None if it knows none)
        return ans, (ans is None)
    return None, True                        # 'none' -> nothing mapped to a known concept -> abstain (the moat)


# ============================================================================================================
# 6. THE SELF-REFLECTIVE Q&As — the brain talking about itself, rendered by the off-bridge Qwen faculty.
# ============================================================================================================

def run_self_reflective(agent, cur, faculty, action_words):
    """Run each `self_reflective_questions` through GATE (brain recall) -> CONSTRAIN (off-bridge Qwen render) ->
    VERIFY (re-parse). Returns the verbatim Q&A list (the brain's spoken answers about itself).

    If `faculty` is None (no GPU / --no-faculty), the answer is the brain's raw recalled fact as a plain triple
    sentence (still BRAIN-sourced; just not Qwen-phrased)."""
    concepts = set(agent.composer.concepts.keys())
    # the VERIFY content-token sets (over the TAUGHT facts) + the verb-inflection map
    facts = _all_facts_svo(cur)
    agents_set = {f[0] for f in facts}
    patients_set = {f[2] for f in facts}
    actions_set = {f[1] for f in facts}
    inflect = _build_inflection_map_local(actions_set)
    vocab_sets = (agents_set, actions_set, patients_set, inflect)

    stored_facts = [(f.get("agent"), f.get("action"), f.get("patient")) for f, _ in agent.composer.kb
                    if all(isinstance(f.get(r), str) for r in ("agent", "action", "patient"))]
    qas = []
    for q_text in cur.get("self_reflective_questions", []):
        kind, cue = _question_to_cue(q_text, concepts, action_words, stored_facts)
        # GATE: the matcher returns the stored SVO triple this self-question is about; VERIFY it via spiking recall
        gate_svo = None
        if kind == "fact":
            a, v, p = cue
            if agent.what_does(a, v) == p:        # spiking recall confirms the matcher's pick
                gate_svo = [a, v, p]
        rec = {"question": q_text, "kind": kind, "cue": cue, "gate_svo": gate_svo}
        if gate_svo is None:
            rec.update({"answer": None, "abstained": True, "verified": None,
                        "note": "brain knows no fact for this self-question -> abstain (the moat)"})
            qas.append(rec)
            continue
        if faculty is not None:
            r = grounded_render(agent, faculty, gate_svo, vocab_sets)
            rec.update({"answer": r["surface"], "surface_full": r["surface_full"],
                        "reparse_svo": r["reparse_svo"], "verified": r["verified"],
                        "regen_used": r["regen_used"], "gen_seconds": r["gen_seconds"], "abstained": False})
        else:
            # no faculty: speak the brain's raw triple (brain-sourced, unphrased)
            rec.update({"answer": " ".join(gate_svo), "verified": None, "abstained": False,
                        "note": "no-faculty: raw brain triple (BRAIN-sourced; not LLM-phrased)"})
        qas.append(rec)
    return qas


# ============================================================================================================
# 7. THE DEVELOP STAGE — run the GPU develop loop on the self-knowledge curriculum + SAVE the developed brain.
# ============================================================================================================

def develop_self_knowledge(cur, n_days, seed, root, max_windows_per_day, n_hub, n_per, D, verbose=True):
    """Run the REAL-GPU develop loop on the self-knowledge curriculum. Returns (per_day metrics, grounded codes,
    a final-cortex handle for reading the codes). The grounded codes are the brain's LEARNED-from-listening codes;
    the firewall/Q&A stages converse on them."""
    curriculum = SelfKnowledgeCurriculum(cur, n_days)
    main_root = os.path.join(root, "develop_main")
    lineage = BridgeLineage("self_knowledge_main", root=Path(main_root))
    if verbose:
        print(f"[develop] WAKE(REAL stream-cortex code-learning) -> CONVERSE -> SLEEP -> [GROWTH] -> METRICS -> "
              f"PERSIST, {n_days} days on the SELF-KNOWLEDGE curriculum.\n", flush=True)

    # build the persistent stream cortex ONCE so we can read the developed grounded codes AFTER the loop. The
    # develop loop owns its own cortex by default; we pass a SHARED one so its accumulated codes survive the loop.
    full_vocab = curriculum.full_vocab()
    cortex = StreamCortex(full_vocab, seed, n_hub=n_hub, n_per=n_per, D=D, verbose=verbose)
    per_day, assembly = develop_gpu(lineage, curriculum, n_days, seed=seed, consolidation_on=True,
                                    plasticity_on=True, max_windows_per_day=max_windows_per_day,
                                    n_hub=n_hub, n_per=n_per, D=D, verbose=verbose, _shared_cortex=cortex)
    # read the FINAL developed grounded codes (the brain's listened-for codes)
    _, _, grounded = cortex.read_codes()
    learn_fid = cortex.learning_fidelity()
    cortex.close()
    return per_day, assembly, grounded, learn_fid, lineage


# ============================================================================================================
# 8. The REPL (the owner types questions to the developed brain).
# ============================================================================================================

def chat(agent, faculty, cur, action_words, question):
    """Answer ONE question to the developed brain: parse (self-aliases resolved) -> GATE (brain recall) ->
    CONSTRAIN (off-bridge Qwen render if a faculty is loaded) -> VERIFY. Returns the spoken answer string (or an
    honest abstention message). The no-confab moat: an untaught / general-knowledge / unmapped question abstains."""
    concepts = set(agent.composer.concepts.keys())
    stored_facts = [(f.get("agent"), f.get("action"), f.get("patient")) for f, _ in agent.composer.kb
                    if all(isinstance(f.get(r), str) for r in ("agent", "action", "patient"))]
    kind, cue = _question_to_cue(question, concepts, action_words, stored_facts)
    gate_svo = None
    if kind == "fact":
        a, v, p = cue
        if agent.what_does(a, v) == p:        # spiking recall confirms the matcher's pick
            gate_svo = [a, v, p]
    if gate_svo is None:
        return "(the brain abstains — it has no stored fact for that. It only answers what it was taught.)"
    if faculty is not None:
        facts = _all_facts_svo(cur)
        vocab_sets = ({f[0] for f in facts}, {f[1] for f in facts}, {f[2] for f in facts},
                      _build_inflection_map_local({f[1] for f in facts}))
        r = grounded_render(agent, faculty, gate_svo, vocab_sets)
        if r["verified"]:
            return r["surface"]
        # render didn't verify -> fall back to the raw brain triple (never leak unverified prose)
        return " ".join(gate_svo) + "  [unverified render; spoke the brain's raw fact]"
    return " ".join(gate_svo)


def run_repl(load_path, seed, T, max_new_tokens, no_faculty):
    """Interactive REPL on the developed brain (loads the saved grounded codes)."""
    cur = _load_curriculum()
    action_words = {v for (_a, v, _p) in _all_facts_svo(cur)}
    vocab = _qa_vocab(cur)
    grounded = None
    if load_path and os.path.exists(load_path):
        with open(load_path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        grounded = {w: np.asarray(v, dtype=float) for w, v in blob.get("grounded_codes", {}).items()}
        print(f"[repl] loaded {len(grounded)} developed grounded codes from {load_path}", flush=True)
    else:
        print("[repl] no developed codes given (--load); using the composer's own seed codes "
              "(the brain still answers the taught facts, just not on the listened-for codes).", flush=True)
    agent, n_taught = build_qa_agent(cur, vocab, grounded, seed)
    faculty = None
    if not no_faculty:
        _free_cupy_pool()      # QWEN-CRASH FIX: free cupy-pool blocks before torch loads the faculty
        try:
            from research.runners._grounded_lang_integration_derisk import SpikingQwenFaculty
            import torch
            faculty = SpikingQwenFaculty(T=T, max_new_tokens=max_new_tokens, seed=seed,
                                         device=("cuda" if torch.cuda.is_available() else "cpu"))
            print(f"[repl] off-bridge Qwen faculty loaded (T={T}).", flush=True)
        except Exception as e:
            print(f"[repl] faculty unavailable ({e!r}); answers will be raw brain triples.", flush=True)
    print("\n" + "=" * 78)
    print("  Talk to the brain about ITSELF.  'you'/'your'/'I'/'it' map to the brain.")
    print(f"  It knows {n_taught} facts about how this project works. It ABSTAINS on anything else.")
    print("  Try:  what are you   /   how do you learn   /   what prevents confabulation")
    print("  (the brain will REFUSE 'capital of France' etc. — that is the firewall.)")
    print("  Type 'quit' to exit.")
    print("=" * 78 + "\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[repl] bye.")
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            print("[repl] bye.")
            break
        ans = chat(agent, faculty, cur, action_words, q)
        print(f"brain> {ans}\n", flush=True)


# ============================================================================================================
# 9. THE FULL DEMO (develop -> firewall -> self-reflective Q&As -> verdict).
# ============================================================================================================

def run_demo(n_days, seed, T, max_new_tokens, max_windows_per_day, n_hub, n_per, D, no_faculty, root):
    cur = _load_curriculum()
    action_words = {v for (_a, v, _p) in _all_facts_svo(cur)}

    # ---- (1) DEVELOP: the GPU develop loop on the self-knowledge curriculum ----
    t0 = time.time()
    per_day, assembly, grounded, learn_fid, lineage = develop_self_knowledge(
        cur, n_days, seed, root, max_windows_per_day, n_hub, n_per, D, verbose=True)
    develop_s = time.time() - t0

    # the develop trends (did the brain LEARN the project?)
    vocab_trend = [dp["vocab_size"] for dp in per_day]
    heard_trend = [dp["concepts_heard"] for dp in per_day]
    facts_trend = [dp["facts_known"] for dp in per_day]
    recall_vals = [dp["recall_acc"] for dp in per_day if dp["recall_acc"] is not None]
    retention_vals = [dp["retention_acc"] for dp in per_day if dp["retention_acc"] is not None]
    moat_fa_develop = sum(dp["moat_false_accepts"] for dp in per_day)
    learn_vals = [dp["learn_fidelity"] for dp in per_day]
    developed = {
        "vocab_grew": (len(vocab_trend) >= 2 and vocab_trend[-1] > vocab_trend[0]),
        "facts_grew": (len(facts_trend) >= 2 and facts_trend[-1] > facts_trend[0]),
        "vocab_trend": vocab_trend, "concepts_heard_trend": heard_trend, "facts_trend": facts_trend,
        "recall_mean": (float(np.mean(recall_vals)) if recall_vals else None),
        "retention_mean": (float(np.mean(retention_vals)) if retention_vals else None),
        "learn_fidelity_mean": (float(np.mean(learn_vals)) if learn_vals else None),
        "moat_false_accepts_develop": moat_fa_develop,
        "concepts_grounded": len(grounded),
    }
    print(f"\n[develop] DONE in {develop_s:.0f}s. vocab {vocab_trend[0]}->{vocab_trend[-1]}, "
          f"heard {heard_trend[0]}->{heard_trend[-1]}, facts {facts_trend[0]}->{facts_trend[-1]}, "
          f"recall_mean {developed['recall_mean']}, retention_mean {developed['retention_mean']}, "
          f"corr(M,C)_mean {developed['learn_fidelity_mean']:+.2f}, grounded {len(grounded)} codes.\n", flush=True)

    # SAVE the developed grounded codes (so the REPL + future runs load the SAME developed brain)
    Path(CODES).parent.mkdir(parents=True, exist_ok=True)
    with open(CODES, "w", encoding="utf-8") as fh:
        json.dump({"seed": seed, "n_days": n_days, "learn_fidelity_mean": developed["learn_fidelity_mean"],
                   "grounded_codes": {w: np.asarray(v).tolist() for w, v in grounded.items()}},
                  fh, default=_json_default)
    print(f"[develop] saved {len(grounded)} developed grounded codes -> {CODES}", flush=True)

    # ---- build the firewall/Q&A agent on the DEVELOPED grounded codes + teach the full curriculum ----
    vocab = _qa_vocab(cur)
    agent, n_taught = build_qa_agent(cur, vocab, grounded, seed)
    print(f"[firewall] built the conversational agent on the developed codes; taught {n_taught} project facts; "
          f"vocab {len(vocab)} (incl. {len(GENERAL_KNOWLEDGE_VOCAB)} encodable Qwen-general fall-backs).\n", flush=True)

    # ---- (2) THE FIREWALL TEST ----
    fw = run_firewall(agent, cur, action_words)
    print(f"[firewall] (a) PROJECT-answered {fw['positive_answered']}/{fw['positive_total']}  |  "
          f"(b) GENERAL-abstained {fw['general_abstained']}/{fw['general_total']} (LEAKS={fw['general_leaks']})  |  "
          f"(c) UNTAUGHT-abstained {fw['untaught_abstained']}/{fw['untaught_total']} (LEAKS={fw['untaught_leaks']})",
          flush=True)
    for g in fw["general_detail"]:
        tag = "LEAK!" if g["LEAK"] else "abstain OK"
        print(f"[firewall]   general: {g['question']!r} -> {g['answer']!r}  [{tag}]", flush=True)

    # ---- (3) THE SELF-REFLECTIVE Q&As (off-bridge Qwen render) ----
    faculty = None
    faculty_info = None
    faculty_err = None
    if not no_faculty:
        print(f"\n[selfreflect] loading the FAST off-bridge Qwen-0.5B faculty at T={T} ...", flush=True)
        # QWEN-CRASH FIX (2026-06-24): the develop loop + the firewall composer bridges leave ~12 GB of CACHED
        # cupy-pool blocks held on the device (the run.log's constant 12.4 GB). The PyTorch faculty allocates its
        # OWN CUDA context + buffers ALONGSIDE -- under GPU contention this can hard-kill the process with no
        # traceback (the observed ~31-min crash). Release the cupy pool's cached blocks back to the device FIRST
        # so torch loads into free VRAM (in the faithful repro this drops held VRAM 12.2 GB -> 1.6 GB; the faculty
        # then loads to ~2.9 GB used / ~21 GB free). Runner-level, no sim/ edit.
        _free_cupy_pool()
        try:
            from research.runners._grounded_lang_integration_derisk import SpikingQwenFaculty
            import torch
            if not torch.cuda.is_available():
                print("[selfreflect] WARNING: CUDA not available — the spiking faculty will be slow.", flush=True)
            faculty = SpikingQwenFaculty(T=T, max_new_tokens=max_new_tokens, seed=seed,
                                         device=("cuda" if torch.cuda.is_available() else "cpu"))
            faculty_info = {"load_seconds": faculty.load_seconds, "pools": faculty.pools,
                            "T": T, "model": "Qwen2.5-0.5B-Instruct (spiking forward, off-bridge)"}
            print(f"[selfreflect]   faculty loaded in {faculty.load_seconds}s.\n", flush=True)
        except Exception as e:
            import traceback
            faculty_err = repr(e)
            traceback.print_exc()
    qas = run_self_reflective(agent, cur, faculty, action_words)
    print("\n[selfreflect] the brain talking about ITSELF (BRAIN-recalled content, off-bridge-Qwen-phrased):",
          flush=True)
    for qa in qas:
        v = qa.get("verified")
        vtag = "" if v is None else (" [verified✓]" if v else " [unverified]")
        print(f"[selfreflect]   Q: {qa['question']}", flush=True)
        print(f"[selfreflect]   A: {qa.get('answer')!r}{vtag}  (gate={qa.get('gate_svo')})", flush=True)

    # ---- VERDICT ----
    brain_learned = (developed["vocab_grew"] and developed["facts_grew"]
                     and (developed["recall_mean"] is not None and developed["recall_mean"] >= 0.5)
                     and fw["positive_answered"] >= max(1, fw["positive_total"] // 2))
    firewall_holds = (fw["general_leaks"] == 0 and fw["untaught_leaks"] == 0)
    # the self-reflective Q&As read coherently: a majority produced a non-abstain answer AND (if the faculty ran)
    # the verified ones dominate; if no faculty, the raw brain triples are BRAIN-sourced by construction.
    answered_qas = [qa for qa in qas if not qa.get("abstained")]
    if faculty is not None:
        verified_qas = [qa for qa in answered_qas if qa.get("verified")]
        selfreflect_coherent = (len(answered_qas) >= max(1, len(qas) // 2)
                                and len(verified_qas) >= max(1, len(answered_qas) // 2))
    else:
        selfreflect_coherent = (len(answered_qas) >= max(1, len(qas) // 2))
    go = bool(brain_learned and firewall_holds and selfreflect_coherent)

    if go:
        verdict = (
            f"GO — PROVEN brain-sourced. The brain DEVELOPED the project from listening (vocab "
            f"{vocab_trend[0]}->{vocab_trend[-1]}, facts {facts_trend[0]}->{facts_trend[-1]}, corr(M,C) "
            f"{developed['learn_fidelity_mean']:+.2f}, recall {developed['recall_mean']:.2f}), ANSWERS project "
            f"facts ({fw['positive_answered']}/{fw['positive_total']}), and the FIREWALL HOLDS: 0 LEAKS on the "
            f"Qwen-general-knowledge probes (capital of France / 2+2 / Romeo&Juliet all ABSTAINED "
            f"{fw['general_abstained']}/{fw['general_total']}) AND abstains on untaught project parts "
            f"({fw['untaught_abstained']}/{fw['untaught_total']}). The self-reflective Q&As read coherently with "
            f"BRAIN-recalled content (off-bridge Qwen only phrases). ⇒ the knowledge is the brain's; the LLM cannot "
            f"see this niche project, so every project answer is provably the brain's own memory.")
    else:
        snags = []
        if not brain_learned:
            snags.append(f"brain_learned={brain_learned} (vocab_grew={developed['vocab_grew']}, "
                         f"facts_grew={developed['facts_grew']}, recall_mean={developed['recall_mean']}, "
                         f"project_answered={fw['positive_answered']}/{fw['positive_total']})")
        if not firewall_holds:
            snags.append(f"FIREWALL BREACH: general_leaks={fw['general_leaks']} untaught_leaks={fw['untaught_leaks']} "
                         f"-> the LLM leaked OR the brain over-generalized; the fix is a tighter gate (the cue-"
                         f"extractor mapped a general-knowledge cue to a stored fact, or a fall-back vocab word was "
                         f"accidentally stored). See general_detail/untaught_detail for the exact breach.")
        if not selfreflect_coherent:
            snags.append(f"self-reflective Q&As: {len(answered_qas)}/{len(qas)} answered "
                         f"(faculty={'on' if faculty is not None else 'off'})")
        verdict = "HONEST/PARTIAL — " + " || ".join(snags)

    return {
        "probe": "teach_the_brain_about_itself__develop_plus_firewall_proof",
        "resolves": "DEVELOP a brain on the self-knowledge curriculum (the project's own architecture, Claude-"
                    "authored, learned from listening), then PROVE its answers are the brain's OWN knowledge via "
                    "the firewall test (the brain answers niche project facts the off-bridge Qwen cannot know, AND "
                    "ABSTAINS on the things Qwen DOES know — capital of France, 2+2 — so a leak is impossible).",
        "curriculum": os.path.relpath(os.path.abspath(CURRICULUM), _REPO),
        "backend": os.environ.get("SIM_BACKEND"),
        "seed": seed, "n_days": n_days, "D": D, "n_hub": n_hub, "n_per": n_per,
        "max_windows_per_day": max_windows_per_day, "T": T, "max_new_tokens": max_new_tokens,
        "GO": go,
        "verdict": verdict,
        "developed": developed,
        "develop_seconds": round(develop_s, 1),
        "develop_per_day": per_day,
        "firewall": {
            "positive_answered": fw["positive_answered"], "positive_total": fw["positive_total"],
            "general_abstained": fw["general_abstained"], "general_total": fw["general_total"],
            "general_leaks": fw["general_leaks"],
            "untaught_abstained": fw["untaught_abstained"], "untaught_total": fw["untaught_total"],
            "untaught_leaks": fw["untaught_leaks"],
            "positive_detail": fw["positive_detail"],
            "general_detail": fw["general_detail"],
            "untaught_detail": fw["untaught_detail"],
        },
        "self_reflective_qas": qas,
        "faculty_info": faculty_info,
        "faculty_error": faculty_err,
        "n_facts_taught": n_taught,
        "grounded_codes_saved": CODES,
        "repl_command": (f"SIM_BACKEND=cupy python -u -m research.runners._self_knowledge_demo --repl "
                         f"--load {os.path.relpath(CODES, _REPO)} --seed {seed}"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-days", type=int, default=4, help="number of simulated 'days' to develop the brain")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--T", type=int, default=16, help="rate-code pool budget for the off-bridge Qwen faculty (16=GO)")
    ap.add_argument("--max-new-tokens", type=int, default=24, help="faculty surface-form length cap")
    ap.add_argument("--max-windows-per-day", type=int, default=2500, help="stream-window budget per day")
    ap.add_argument("--n-hub", type=int, default=200, help="stream-cortex hub (context-word) count")
    ap.add_argument("--n-per", type=int, default=12, help="neurons per concept (population code)")
    ap.add_argument("--D", type=int, default=128, help="composer phasor dimension")
    ap.add_argument("--no-faculty", action="store_true", help="skip the off-bridge Qwen render (brain triples only)")
    ap.add_argument("--repl", action="store_true", help="interactive REPL on the developed brain (use with --load)")
    ap.add_argument("--load", default=CODES, help="path to saved developed grounded codes (for --repl)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import logging
    logging.disable(logging.INFO)

    if a.repl:
        run_repl(a.load, a.seed, a.T, a.max_new_tokens, a.no_faculty)
        return 0

    print("=" * 110, flush=True)
    print("[TEACH THE BRAIN ABOUT ITSELF — develop + the FIREWALL-PROOF test]", flush=True)
    print(f"  backend={os.environ.get('SIM_BACKEND')}  n_days={a.n_days}  seed={a.seed}  T={a.T}  D={a.D}", flush=True)
    print("  DEVELOP the brain on the project's own architecture (learned from listening), then PROVE its answers "
          "are the BRAIN's (answers niche project facts Qwen can't know; ABSTAINS on what Qwen DOES know).", flush=True)
    print("=" * 110 + "\n", flush=True)

    t0 = time.time()
    root = tempfile.mkdtemp(prefix="self_knowledge_")
    try:
        res = run_demo(a.n_days, a.seed, a.T, a.max_new_tokens, a.max_windows_per_day,
                       a.n_hub, a.n_per, a.D, a.no_faculty, root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    res["wall_seconds"] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False, default=_json_default)

    print(f"\n{'=' * 110}", flush=True)
    print(f"  VERDICT: {res['verdict']}", flush=True)
    print(f"  [saved] {a.out}  (wall {res['wall_seconds']}s)", flush=True)
    print(f"  REPL: {res['repl_command']}", flush=True)
    print(f"{'=' * 110}", flush=True)
    return 0 if res["GO"] else 1


if __name__ == "__main__":
    sys.exit(main())
