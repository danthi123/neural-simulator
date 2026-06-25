"""COMMUNICABLE-BRAIN PROBE 1 -- "what do you think about X?" -> a NOVEL, GROUNDED, FLAGGED-as-hypothesis
turn, GENERATED on the brain's OWN substrate, the LLM fluency-only (NEVER the content).

The cheapest-first de-risk of the generative+inferential frontier (scoping
`research/findings/raw/_generative_inferential_frontier_scoping.md`, RECOMMENDED Option 1): probe the
COMPOSITION of three INDIVIDUALLY-GO pieces into a conversational turn -- NOT a new mechanism. The open
question the scoping poses: can a "what do you think about X" turn drive
  ASSIMILATE-X-to-the-learned-graph -> generative-replay a NOVEL-but-plausible proposition about X
  -> RENDER it (neural word order + fluency-only LLM surface form) -> VERIFY it matches the brain's
     proposed proposition (catch LLM drift)
  -> EMIT it as a GRADED-CONFIDENCE HYPOTHESIS ("I'm not sure, but I'd guess ...")
such that the output is NOVEL (a generated proposition, not a recalled verbatim fact), GROUNDED (the
brain's LEARNED graph is load-bearing -- beats a SHUFFLED-graph control >=3x), FLAGGED + LEAK-FREE (the
proposition is NOT stored as a fact; a who/what query on the un-stored fact still ABSTAINS -> the moat is
RELAXED to speak-while-flagging, NOT removed), and CALIBRATED (the stated confidence tracks the proposal's
plausibility), AND the LESION (sever the brain's proposal, let the LLM free-generate the content) COLLAPSES
to a hallucination that VERIFY REJECTS (proving the content is the BRAIN's, not the LLM's).

THE THREE GO PIECES COMPOSED (reuse-by-import, NO sim/ edit):
  - the b2 generative-replay PROPOSER (`_genfrontier_b2_generative_replay_derisk`: the
    GenerativeReplayProposer + the PPMI co-occurrence cortex + the RF composer's moat-safe accept gate,
    GO 6-seed: novel-composition 0.752, 17x chance, shuffled-graph collapses, 0 leaks) -> ASSIMILATE + PROPOSE.
  - the NEURAL serial-order renderer (`neural_serial_order_renderer.NeuralSerialOrderRenderer`, GO 6/6:
    spiking competitive-queuing produces the SVO word order, NOT a host sort) -> the cognitive WORD ORDER.
    [OPT-IN via --neural-order; default off so the turn pipeline runs CPU-fast without building the GPU pool
     bridge. The word ORDER for an SVO frame is fixed [agent, action, patient]; the renderer's GO is that the
     order is produced NEURALLY, exercised here when --neural-order is set.]
  - the gate->constrain->VERIFY loop (`_grounded_lang_p3_derisk`: the content-extractor `_extract_svo_from_prose`
    + the brain's BridgeParser re-parse via `BrainConversationalAgent.parse`, GO 3-seed: grounded->fluent,
    untaught->abstain, confab->caught) -> RENDER surface form + VERIFY the rendered sentence matches the
    brain's proposed proposition.

THE FLUENCY-ONLY FACULTY (CPU, per the cheap-first constraint -- the real spiking-Qwen integration runner is
GPU-only and option-A's GPU CI is running). The fluency faculty's ONLY job is the SURFACE FORM; the BRAIN
supplies the proposition (the content + the choice) and VERIFY re-parses the faculty's prose. Two CPU
stand-ins, both reused from the GO P3 loop (the SAME VERIFY contract the real Qwen integration passed):
  - the GROUNDED faculty (`P3.TemplateStubFaculty`): renders the brain's proposed triple into fluent prose,
    content-locked to the triple's own words (its freedom is grammar/determiners/inflection). This is the
    fluency-only LLM with the moat in place.
  - the LESION faculty (`P3.InjectingStubFaculty`): the brain's proposal is SEVERED -> the faculty FREE-
    GENERATES the content itself (a self-chosen patient, NOT the brain's) -> VERIFY MUST reject it. This is
    the decisive provenance anti-cheat: if the turn still emits a sensible grounded reply with the brain's
    proposal severed, the LLM is doing the cognition (FAIL). Mirrors the b2 lesion + the P3 confab-catch.

THE GRADED-CONFIDENCE MOAT-RELAX (owner-sanctioned, `feedback_moat_not_hard_lossy_memory_ok`): the emitted
proposition is FLAGGED as a hypothesis with a hedge derived from its plausibility score (the PPMI pair
strengths -- the brain's learned relatedness). High plausibility -> assertive hedge ("I'd say ..."); low ->
tentative ("I'm not sure, but maybe ..."). CONSERVATIVE relax: flagged-hypothesis ONLY, never asserted as a
known fact; the known-fact channel stays HARD-gated (a who/what query on an un-stored proposed triple still
ABSTAINS -> 0 leaks).

VERDICT (falsification, >=3 seeds; the controller runs 6-seed if GO): GO requires ALL of
  (1) NOVEL       -- every emitted proposition is a GENERATED proposition (never stored), beating the 0.0
                     retrieval-novelty ceiling (`2026-06-22-generation-novelty-categorical-gap-MEASURED.md`);
  (2) GROUNDED    -- the turn's proposals' plausibility beats a SHUFFLED-PPMI-graph control by >= 3x (the
                     brain's LEARNED structure is load-bearing, not noise / a template artifact);
  (3) FLAGGED + leak-free -- 0 leaks to the known-fact channel (the proposition is NOT stored; a subsequent
                     who/what query on it still abstains -> the moat is RELAXED to speak-while-flagging);
  (4) CALIBRATED  -- the stated confidence tracks plausibility (higher PPMI-support -> higher stated
                     confidence; monotone bins, positive rank correlation).
ANTI-CHEATS: LESION/PROVENANCE (sever the proposal -> LLM free-generation -> VERIFY rejects) + the
shuffled-graph control + known-fact-channel integrity (0 leaks) + single-substrate (the PPMI cortex + the
RF composer are the brain; the host does the recombination bookkeeping + routes which assembly fired; the
fluency faculty is the surface form only).

HONEST: if the brain's proposal cannot be rendered+verified into a coherent flagged hypothesis (render/VERIFY
rejects the brain's novel proposition), this reports it PRECISELY -- that localizes the next bounded build.
NOT letting the LLM free-generate the content to "pass" (that IS the cheat -- it is exactly the LESION arm,
which MUST collapse).

CPU (`SIM_BACKEND=numpy`); reuse-by-import; NO `sim/` edit. Run:
  SIM_BACKEND=numpy python -u -m research.runners._communicable_brain_probe1_whatdoyouthink \
      --seeds 42,43,44 --out research/findings/raw/_communicable_brain_probe1_whatdoyouthink.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

# the whole pipeline is the numpy-CPU brain (PPMI cortex + RF composer + parser) + a CPU fluency stand-in.
os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# --- the b2 generative-replay machinery (the GO PROPOSE piece) -- reused VERBATIM ---
from research.runners._genfrontier_b2_generative_replay_derisk import (  # noqa: E402
    GenerativeReplayProposer,
    build_plausibility,
    build_stored_facts,
    enumerate_plausible,
    shuffle_graph,
    _category_pools,
)
from research.runners.option_c_real_cooccurrence_derisk import (  # noqa: E402
    TAXONOMY_8x8,
    taxonomy_to_vocab_categories,
    build_real_cooccurrence,
)
# --- the RF composer + the no-confab moat (the brain's KNOWN-fact store) ---
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402
# --- the gate->constrain->VERIFY loop (the GO RENDER+VERIFY piece) -- the content-extractor + the parser ---
from research.runners._grounded_lang_p3_derisk import (  # noqa: E402
    TemplateStubFaculty,
    InjectingStubFaculty,
)
from research.runners._grounded_lang_integration_derisk import (  # noqa: E402
    _build_inflection_map,
    _extract_svo_from_prose,
)
from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402


# ===========================================================================
# Graded-confidence: the plausibility score -> a calibrated linguistic hedge.
# ===========================================================================
def plausibility_score(P, row, a, ac, p):
    """The proposal's plausibility = the geometric mean of its PPMI pair strengths (the brain's LEARNED
    relatedness). agent~action and action~patient are the selectional-preference pairs (the hard gate); the
    agent~patient relation is the coherence bonus. A scalar in [0, inf) the hedge maps from -- the SAME
    quantity the proposer gated on, read out as a graded confidence (the Bogacz-Brown familiarity signal
    read GRADED, not thresholded)."""
    s_aa = float(P[row[a], row[ac]])
    s_ap = float(P[row[ac], row[p]])
    s_xp = float(P[row[a], row[p]])
    # geometric mean of the two selectional pairs (the load-bearing ones) + a small weight on the direct pair
    base = (max(s_aa, 1e-9) * max(s_ap, 1e-9)) ** 0.5
    return base * (1.0 + 0.25 * s_xp)


# the hedge bands: confidence percentile -> a linguistic hedge. Calibrated against the proposal-population
# plausibility distribution (per-seed), so the band reflects relative graph support, not an absolute cutoff.
HEDGE_BANDS = [
    (0.66, "I'd say"),                          # high relative plausibility -> assertive (still a guess)
    (0.33, "I think maybe"),                     # mid
    (0.00, "I'm not sure, but I'd guess"),       # low -> tentative
]


def hedge_for(score, lo, hi):
    """Map a plausibility score into a hedge given the population [lo, hi] (per-seed min/max). Returns
    (hedge_text, normalized_confidence in [0,1])."""
    if hi <= lo:
        conf = 0.5
    else:
        conf = (score - lo) / (hi - lo)
    conf = float(min(1.0, max(0.0, conf)))
    for thr, txt in HEDGE_BANDS:
        if conf >= thr:
            return txt, conf
    return HEDGE_BANDS[-1][1], conf


# ===========================================================================
# The "what do you think about X" TURN -- the composition of the three GO pieces.
# ===========================================================================
class WhatDoYouThinkTurn:
    """One conversational turn. The BRAIN does the cognition; the LLM (CPU stand-in) does the surface form.

    Pipeline:
      ASSIMILATE(X) -- place the topic X in the learned PPMI graph: its graph neighborhood (the related
                       fillers) is what makes a proposition ABOUT X plausible. (The proposer already samples
                       from this neighborhood; assimilation = seeding the replay on X + reading X's relatedness.)
      PROPOSE       -- the b2 GenerativeReplayProposer samples a NOVEL-but-plausible proposition with X in a
                       role (agent or patient), graph-biased + moat-gated + non-contradictory.
      RENDER+VERIFY -- the fluency faculty renders the proposed triple into prose (word order optionally NEURAL
                       via the serial-order renderer); the brain re-parses the prose (content extractor + the
                       BridgeParser) and checks the re-parsed SVO == the proposed triple. Reject on drift.
      EMIT          -- a graded-confidence FLAGGED hypothesis (hedge from the plausibility score). NOT stored.
    """

    def __init__(self, proposer, comp, agent, P, row, vocab_sets, faculty,
                 full_pools=None, neural_renderer=None, rng=None):
        self.proposer = proposer
        self.comp = comp                 # the RF composer (the KNOWN-fact store + the moat)
        self.agent = agent               # the BrainConversationalAgent (its .parse is the VERIFY re-parse)
        self.P, self.row = P, row
        self.agents_set, self.actions_set, self.patients_set, self.inflect = vocab_sets
        self.faculty = faculty           # the fluency-only LLM stand-in (CPU)
        self.neural_renderer = neural_renderer   # optional NeuralSerialOrderRenderer (neural WORD ORDER)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        # the FULL taxonomy category pools (agent/action/patient) -- a topic the brain knows the WORD for can be
        # reasoned about (ASSIMILATE places it in graph space) even if no STORED fact used it. Generative replay
        # walks the LEARNED GRAPH (PPMI over the WHOLE vocab the brain heard), not only the few stored facts'
        # fillers -- so the partner sampling pools are the FULL known vocab, graph-BIASED (so plausibility /
        # groundedness stays load-bearing). The VERIFY content-token sets are likewise the full pools (the
        # brain's comprehension vocab is the whole vocab it learned). (full_agents, full_actions, full_patients)
        fa, fac, fp = full_pools if full_pools else (set(proposer.agents), set(proposer.actions),
                                                     set(proposer.patients))
        self.full_agents, self.full_actions, self.full_patients = set(fa), set(fac), set(fp)
        # ordered lists for graph-biased sampling over the FULL pools
        self._fa_list = sorted(self.full_agents)
        self._fac_list = sorted(self.full_actions)
        self._fp_list = sorted(self.full_patients)

    # --- ASSIMILATE: place X in the learned graph (read its neighborhood) ---
    def assimilate(self, topic):
        """Return the topic's graph neighborhood: the actions + patients it is most related to (PPMI). This is
        'where X sits in what the brain learned' -- the assimilation that makes a proposition ABOUT X plausible.
        For an unknown topic (not in vocab) this returns empty -> the turn will have nothing graph-supported to
        say (the honest 'I don't really have a view' case)."""
        if topic not in self.row:
            return {"in_graph": False, "related_actions": [], "related_patients": []}
        ti = self.row[topic]
        rel_ac = sorted(self.proposer.actions, key=lambda w: -self.P[ti, self.row[w]])
        rel_pt = sorted(self.proposer.patients, key=lambda w: -self.P[ti, self.row[w]])
        return {"in_graph": True,
                "related_actions": [(w, round(float(self.P[ti, self.row[w]]), 3)) for w in rel_ac[:4]],
                "related_patients": [(w, round(float(self.P[ti, self.row[w]]), 3)) for w in rel_pt[:4]]}

    # --- PROPOSE: a novel-but-plausible proposition ABOUT X via generative replay ---
    def propose_about(self, topic, n_attempts=400):
        """Seed the generative-replay proposer to produce a NOVEL-but-plausible proposition that is genuinely
        ABOUT `topic` -- the topic MUST appear in the proposition (as agent if it is an agent-category word, else
        as the patient). This makes topic-relevance LOAD-BEARING (a stronger grounded claim than 'any plausible
        proposition'): the brain's CHOSEN view about X must mention X. PARTNERS are sampled graph-biased from the
        EXPERIENCED pools (replay recombines what the brain experienced); among the topic-containing plausible
        proposals, the highest-plausibility one is the brain's view. Returns (triple, score) or None -- the
        honest 'I don't really have a view on X' case (no topic-containing proposition is graph-plausible)."""
        if topic not in self.row:
            return None                                    # the brain doesn't know the word -> no view
        topic_is_agent = topic in self.full_agents
        topic_is_patient = topic in self.full_patients
        if not (topic_is_agent or topic_is_patient):
            return None                                    # the topic plays no SVO content role here
        best = None
        for _ in range(n_attempts):
            if topic_is_agent:
                a = topic                                  # X is the SUBJECT of the view ('X does ...')
                ac = self.proposer._sample_weighted(
                    self._fac_list, self.proposer._weight_partner((a,), self._fac_list))
                p = self.proposer._sample_weighted(
                    self._fp_list, self.proposer._weight_partner((a, ac), self._fp_list))
            else:
                # X is the THING the view is about ('... does X' / '... is X'): sample an agent + action that
                # graph-plausibly TAKE X as the patient, biased toward X's neighborhood.
                p = topic
                a = self._sample_partner_for_patient(p)
                ac = self.proposer._sample_weighted(
                    self._fac_list, self.proposer._weight_partner((a, p), self._fac_list))
            triple = (a, ac, p)
            if topic not in triple:
                continue                                   # the proposition must be ABOUT X (topic-relevance)
            if triple in self.proposer.all_stored:
                continue                                   # must be NOVEL (never told)
            if not self.proposer._plausible(a, ac, p):
                continue                                   # must be graph-PLAUSIBLE (selectional preference)
            if self.proposer._contradicts(a, ac, p):
                continue                                   # must NOT contradict an explicitly-negated fact
            sc = plausibility_score(self.P, self.row, a, ac, p)
            if (best is None) or (sc > best[1]):
                best = (triple, sc)
        return best

    def _sample_partner_for_patient(self, patient):
        """Graph-biased sample of an AGENT plausibly associated with `patient` (the subject of a view ABOUT the
        patient X). Weighted by PPMI(agent, patient) over the full agent pool -- the learned relatedness biases
        which subject the brain considers (replay over X's neighborhood)."""
        w = self.proposer._weight_partner((patient,), self._fa_list)
        return self.proposer._sample_weighted(self._fa_list, w)

    # --- RENDER + VERIFY: surface form (fluency faculty) + re-parse the prose back to an SVO ---
    def render_and_verify(self, triple, faculty, faculty_mode="grounded"):
        """Render `triple` into prose with the fluency faculty, then VERIFY by re-parsing the PROSE back into an
        SVO (the brain's content-extractor + BridgeParser) and checking it equals `triple`. Returns a record with
        the surface prose, the re-parsed SVO, and whether VERIFY accepted. `faculty_mode='lesion'` -> the faculty
        FREE-GENERATES the content (the brain's proposal severed); VERIFY must then reject.

        The optional NEURAL serial-order renderer produces the WORD ORDER (the cognitive parallel->serial step);
        the faculty supplies determiners/inflection (surface fluency)."""
        a, v, p = triple
        if faculty_mode == "lesion":
            # the proposal is SEVERED: the faculty chooses its OWN patient (free generation), NOT the brain's.
            surface, asserted = faculty.render_svo(a, v, p)     # InjectingStubFaculty swaps in a self-chosen p
        else:
            surface, asserted = faculty.render_svo(a, v, p)     # TemplateStubFaculty: content-locked fluent prose

        # NEURAL word order (optional): re-order the surface content words by the spiking serial-order read-out.
        # The renderer asserts the SVO frame order [agent, action, patient]; we surface the faculty's prose but
        # record the neural order for the trail (the order the brain produced).
        neural_order = None
        if self.neural_renderer is not None and faculty_mode != "lesion":
            try:
                # spell-by-identity over the 3 content tokens; the renderer ranks them by spiking rate
                idx = {0: a, 1: v, 2: p}
                ordered = self.neural_renderer.order([0, 1, 2])
                neural_order = [idx[i] for i in ordered]
            except Exception:
                neural_order = None

        # VERIFY: re-parse the faculty's PROSE (content extraction + the brain's role assignment).
        csvo = _extract_svo_from_prose(surface, self.agents_set, self.actions_set, self.patients_set,
                                       self.inflect)
        if csvo is None:
            return {"surface": surface, "asserted_svo": asserted, "reparse_svo": None,
                    "verified": False, "neural_order": neural_order,
                    "reject_reason": "prose did not re-parse to a clean SVO"}
        parsed = self.agent.parse(csvo, voice="active")          # the brain's comprehension of the recovered SVO
        rsvo = [parsed.get("agent"), parsed.get("action"), parsed.get("patient")]
        verified = (rsvo == list(triple))
        return {"surface": surface, "asserted_svo": asserted, "reparse_svo": rsvo,
                "verified": bool(verified), "neural_order": neural_order,
                "reject_reason": None if verified else "re-parsed SVO mismatches the brain's proposed proposition"}

    # --- the full turn ---
    def turn(self, topic, conf_lo, conf_hi, n_attempts=400):
        """Run the full 'what do you think about X' turn. Returns a structured record."""
        assim = self.assimilate(topic)
        prop = self.propose_about(topic, n_attempts=n_attempts)
        rec = {"topic": topic, "assimilation": assim}
        if prop is None:
            # the brain has nothing graph-supported to say -> the honest 'I don't really have a view' (no emission)
            rec.update({"proposed_triple": None, "plausibility": None, "rendered": None, "verified": None,
                        "emitted": False, "reply": "I don't really have a view on that.",
                        "hedge": None, "confidence": None, "abstained_opinion": True})
            return rec
        triple, score = prop
        topic_in_prop = topic in triple
        rv = self.render_and_verify(triple, self.faculty, faculty_mode="grounded")
        hedge, conf = hedge_for(score, conf_lo, conf_hi)
        # the spoken word order is the NEURAL serial-order read-out when available, else the SVO frame (the
        # surface words are the brain's proposed triple; the LLM faculty supplied determiners/inflection in
        # rv['surface'], which VERIFY confirmed re-parses back to the proposed triple).
        order_words = rv.get("neural_order") or list(triple)
        if rv["verified"]:
            # EMIT the flagged hypothesis (NOT stored as a fact)
            rec.update({"proposed_triple": list(triple), "topic_in_proposition": topic_in_prop,
                        "plausibility": round(score, 4),
                        "rendered": rv, "verified": True, "emitted": True,
                        "reply": f"{hedge} {' '.join(order_words)}.",
                        "hedge": hedge, "confidence": round(conf, 3), "abstained_opinion": False})
        else:
            # render/VERIFY rejected the brain's proposition -> NO emission (the honest localization case)
            rec.update({"proposed_triple": list(triple), "topic_in_proposition": topic_in_prop,
                        "plausibility": round(score, 4),
                        "rendered": rv, "verified": False, "emitted": False,
                        "reply": None, "hedge": hedge, "confidence": round(conf, 3),
                        "abstained_opinion": False})
        return rec


# ===========================================================================
# Per-seed run: build the grounded brain (b2 testbed), run the turn over held-out topics, measure the 4 gates
# + the lesion/provenance anti-cheat + the shuffled-graph control.
# ===========================================================================
def run_seed(seed, vocab, corpus, a):
    rng = np.random.default_rng(seed)
    agents, actions, patients = _category_pools(TAXONOMY_8x8)
    P, row = build_plausibility(corpus, vocab)
    pos = P[P > 0]
    tau = float(np.percentile(pos, a.tau_pct)) if pos.size else 0.0

    affirmed, negated, plausible_all = build_stored_facts(
        agents, actions, patients, P, row, tau, a.n_facts, a.n_negated, rng)
    all_stored = set(affirmed) | set(negated)

    # the brain's KNOWN-fact store (RF composer; the no-confab moat intact)
    comp = RFPhasorComposer(seed=seed, D=a.D, vocab=vocab)
    for ag, ac, pt in affirmed:
        comp.store(ag, ac, pt, polarity="AFFIRM")
    for ag, ac, pt in negated:
        comp.store(ag, ac, pt, polarity="NEGATE")

    # the BrainConversationalAgent over the SAME vocab -- its .parse is the VERIFY re-parse, and its
    # what_does/who_does/is_it_true is the KNOWN-fact channel the moat must keep hard-gated.
    bc_agent = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab},
                                        composer=comp, composer_kind="rf")

    # the generative-replay proposer (the GO PROPOSE piece) over the TRUE learned graph. The generative DRAW is the
    # validated spiking soft-WTA by DEFAULT (use_spiking_sampler=True -> the brain's spiking generative act); the
    # --host-oracle-sampler escape pins it to the host np.random.choice ORACLE (the prior behavior, for parity).
    use_spiking = not a.host_oracle_sampler
    proposer = GenerativeReplayProposer(comp, affirmed, negated, P, row, tau,
                                        np.random.default_rng(seed * 7 + 1),
                                        use_spiking_sampler=use_spiking, spiking_seed=seed)

    # content-token sets for the VERIFY re-parse. The brain's COMPREHENSION vocab is the WHOLE vocab it
    # learned (its parser reads any known word), so the VERIFY sets are the FULL taxonomy category pools --
    # NOT only the experienced fillers. (Using the experienced fillers spuriously rejects a TRUE rendered
    # proposition whose topic word wasn't a stored patient, e.g. 'frog jump head' when 'head' was never a
    # stored patient -- a VERIFY-set artifact, not a real drift.)
    agents_set = set(agents)
    actions_set = set(actions)
    patients_set = set(patients)
    inflect = _build_inflection_map(sorted(actions_set))
    vocab_sets = (agents_set, actions_set, patients_set, inflect)

    # the fluency-only faculty (CPU): GROUNDED = content-locked template; renders the brain's proposed triple.
    grounded_faculty = TemplateStubFaculty()

    # the FULL taxonomy category pools (agents/actions/patients) -- a topic the brain knows the WORD for can be
    # reasoned about even if no STORED fact used it as that role. `_category_pools(TAXONOMY_8x8)` returns the
    # full pools; the proposer's experienced pools (proposer.agents/.patients) supply the graph-biased PARTNERS.
    full_agents, full_actions, full_patients = agents, actions, patients
    full_pools = (set(full_agents), set(full_actions), set(full_patients))

    # optional NEURAL word-order renderer (GPU pool bridge) -- default OFF (CPU-fast); --neural-order to exercise.
    neural_renderer = None
    if a.neural_order:
        try:
            from research.runners.neural_serial_order_renderer import NeuralSerialOrderRenderer
            neural_renderer = NeuralSerialOrderRenderer(seed=seed)
        except Exception as e:
            print(f"  [warn] neural renderer unavailable ({e!r}); word order falls back to the SVO frame.", flush=True)

    turn = WhatDoYouThinkTurn(proposer, comp, bc_agent, P, row, vocab_sets, grounded_faculty,
                              full_pools=full_pools, neural_renderer=neural_renderer,
                              rng=np.random.default_rng(seed * 23 + 9))

    # held-out TOPICS: vocab words NOT used as the agent of any stored fact (so a proposition about them is
    # genuinely novel, not a re-statement). Pick a spread across agent + patient categories.
    stored_agents = {f[0] for f in affirmed}
    topic_pool = [w for w in (agents + patients) if w not in stored_agents]
    rng.shuffle(topic_pool)
    topics = topic_pool[:a.n_topics]

    # --- pre-pass to calibrate the hedge bands: score the proposals for all topics first ---
    pre = []
    for t in topics:
        pr = turn.propose_about(t, n_attempts=a.n_attempts)
        pre.append((t, pr))
    scores = [pr[1] for (_t, pr) in pre if pr is not None]
    conf_lo = float(min(scores)) if scores else 0.0
    conf_hi = float(max(scores)) if scores else 1.0

    # --- run the turns ---
    turns = [turn.turn(t, conf_lo, conf_hi, n_attempts=a.n_attempts) for t in topics]
    emitted = [r for r in turns if r["emitted"]]
    n_emitted = len(emitted)

    # =====================================================================
    # (1) NOVEL: every emitted proposition is GENERATED (never stored) -> beats the 0.0 retrieval ceiling.
    # AND topic-relevant (the proposition is genuinely ABOUT its topic X) -- a stronger grounded claim.
    # =====================================================================
    n_novel = sum(1 for r in emitted if tuple(r["proposed_triple"]) not in all_stored)
    all_emitted_novel = (n_emitted > 0) and (n_novel == n_emitted)
    # distinct-from-stored ratio (every emitted triple is distinct from every stored fact -> > 1.0 means
    # the turn produces content the store does not contain)
    novel_ratio = (n_novel / max(1, n_emitted)) if n_emitted else 0.0
    # topic-relevance: every emitted proposition mentions its topic X (the 'about X' semantics)
    n_topic_relevant = sum(1 for r in emitted if r.get("topic_in_proposition"))
    all_topic_relevant = (n_emitted > 0) and (n_topic_relevant == n_emitted)
    # how many topics the brain had a view about (proposed) vs abstained-from-opinion (honest 'no view')
    n_opinion_abstained = sum(1 for r in turns if r.get("abstained_opinion"))

    # =====================================================================
    # (2) GROUNDED: the SHUFFLED-PPMI-graph control. Re-build the proposer over a shuffled graph (destroys
    # neighborhoods, preserves marginals); the proposals' TRUE-graph plausibility must collapse to the random
    # floor -> the LEARNED structure is load-bearing. Measured as the b2-style advantage RATIO.
    # =====================================================================
    # the TRUE-graph plausible-fraction of the brain's biased replay (over novel samples)
    rep_true = proposer.propose(a.n_attempts_grounded)
    replay_frac = rep_true["plausible_fraction_of_novel"]
    # the SHUFFLED-graph proposer, scored under the TRUE graph (does shuffling kill the advantage?)
    P_shuf = shuffle_graph(P, np.random.default_rng(seed * 17 + 5))
    pos_s = P_shuf[P_shuf > 0]
    tau_s = float(np.percentile(pos_s, a.tau_pct)) if pos_s.size else 0.0
    proposer_shuf = GenerativeReplayProposer(comp, affirmed, negated, P_shuf, row, tau_s,
                                             np.random.default_rng(seed * 19 + 7),
                                             use_spiking_sampler=use_spiking, spiking_seed=seed * 19 + 7)
    shuf_true_plausible, shuf_novel = 0, 0
    for _ in range(a.n_attempts_grounded):
        ag = proposer_shuf.agents[int(proposer_shuf.rng.integers(len(proposer_shuf.agents)))]
        acn = proposer_shuf._sample_weighted(
            proposer_shuf.actions, proposer_shuf._weight_partner((ag,), proposer_shuf.actions))
        ptn = proposer_shuf._sample_weighted(
            proposer_shuf.patients, proposer_shuf._weight_partner((ag, acn), proposer_shuf.patients))
        if (ag, acn, ptn) in proposer_shuf.all_stored:
            continue
        shuf_novel += 1
        if proposer._plausible(ag, acn, ptn):           # TRUE-graph plausibility of a shuffled-replay sample
            shuf_true_plausible += 1
    shuf_frac = shuf_true_plausible / max(1, shuf_novel)
    grounded_advantage = replay_frac / max(shuf_frac, 1.0 / max(1, shuf_novel))
    grounded_ok = grounded_advantage >= a.advantage_bar

    # =====================================================================
    # (3) FLAGGED + LEAK-FREE: the proposition is NOT stored. A who/what query on the proposed (un-stored)
    # triple must STILL ABSTAIN -> the moat is RELAXED to speak-while-flagging, NOT removed.
    # =====================================================================
    moat_leaks = 0
    for r in emitted:
        a_, v_, p_ = r["proposed_triple"]
        # the known-fact channel must NOT have learned the proposed (un-stored) fact:
        known_p = bc_agent.what_does(a_, v_)             # must be None OR != p_ (the cue was never stored as this)
        yn = bc_agent.is_it_true(a_, v_, p_)             # must be 'unknown' (never stored -> not a known fact)
        if known_p == p_:
            moat_leaks += 1
        if yn == "yes":
            moat_leaks += 1
    # every emitted reply must carry a hedge (it is FLAGGED, never asserted as a known fact)
    all_flagged = (n_emitted > 0) and all(r["hedge"] is not None for r in emitted)
    flagged_leakfree = (moat_leaks == 0) and all_flagged

    # =====================================================================
    # (4) CALIBRATED: the stated confidence tracks the proposal's plausibility -- the moat-relax reliability
    # test (anti-cheat #3). TWO checks:
    #   (i) rank-consistency: Spearman(plausibility, stated-confidence). The hedge is a monotone map of the
    #       plausibility SCORE by construction, so this confirms the MAPPING is order-preserving across the real
    #       population (it would break only if a reply's confidence were decoupled from its plausibility).
    #   (ii) INDEPENDENT reliability bin: bin emitted propositions by stated confidence; the high-confidence bin
    #       must have a HIGHER incidence of an INDEPENDENT graph-support property -- the b2 STRONG-plausibility
    #       indicator (ALL THREE PPMI pairs graph-related, a binary property the hedge does NOT read; the hedge
    #       maps only the continuous selectional-pair score). A flat curve here = the hedge is decorative. This
    #       is the genuinely-non-tautological reliability check: empirical graph-strength rises with the hedge.
    # =====================================================================
    cal_rows = [(r["plausibility"], r["confidence"],
                 int(proposer._strong_plausible(*tuple(r["proposed_triple"]))))
                for r in emitted if r["plausibility"] is not None and r["confidence"] is not None]
    if len(cal_rows) >= 3:
        ps = np.array([x[0] for x in cal_rows], dtype=float)
        cs = np.array([x[1] for x in cal_rows], dtype=float)
        strong = np.array([x[2] for x in cal_rows], dtype=float)   # INDEPENDENT binary graph-support
        pr = np.argsort(np.argsort(ps)).astype(float)
        cr = np.argsort(np.argsort(cs)).astype(float)
        if pr.std() > 0 and cr.std() > 0:
            spearman = float(np.corrcoef(pr, cr)[0, 1])
        else:
            spearman = 1.0          # constant (all equally plausible) -> trivially consistent
        order = np.argsort(cs)
        n = len(cs)
        lo_idx = order[: max(1, n // 3)]
        hi_idx = order[-max(1, n // 3):]
        # (i) mean plausibility-score monotone across the conf bins
        monotone = bool(ps[hi_idx].mean() >= ps[lo_idx].mean())
        # (ii) INDEPENDENT strong-plausibility incidence rises (or holds) with confidence -- >= not > so a
        # population with all-strong (or none-strong) does not spuriously fail; the binding test is when there
        # is a mix and the high-conf bin must carry more of the strong ones.
        strong_reliability = bool(strong[hi_idx].mean() >= strong[lo_idx].mean())
        strong_lo = float(strong[lo_idx].mean())
        strong_hi = float(strong[hi_idx].mean())
        calibrated_ok = (spearman >= a.calib_spearman_bar) and monotone and strong_reliability
    else:
        spearman = None
        monotone = None
        strong_reliability = None
        strong_lo = strong_hi = None
        calibrated_ok = None        # too few emissions to assess calibration (reported, not failed silently)

    # =====================================================================
    # ANTI-CHEAT: LESION / PROVENANCE. Sever the brain's proposal -> the faculty FREE-GENERATES the content
    # (a self-chosen patient, NOT the brain's) -> VERIFY MUST reject it (the content is the BRAIN's, not the
    # LLM's). Run on the emitted topics: gate the TRUE proposed triple, but hand the faculty a free hand to
    # pick its OWN patient. caught iff VERIFY rejects (re-parsed SVO != the brain's proposed triple).
    # =====================================================================
    # build the free-generation (lesion) faculty: it swaps the brain's patient for a self-chosen one.
    all_pats = sorted(patients_set)
    lesion_caught, lesion_total = 0, 0
    lesion_detail = []
    for r in emitted:
        a_, v_, p_ = r["proposed_triple"]
        wrong = next((x for x in all_pats if x != p_), p_ + "_X")    # the faculty's OWN (wrong) content
        # the lesion faculty free-generates: assert (a, v, wrong) -- the brain's proposal severed
        lesion_faculty = InjectingStubFaculty({p_: wrong}, swap_role="patient")
        rv = turn.render_and_verify((a_, v_, p_), lesion_faculty, faculty_mode="lesion")
        # caught iff VERIFY refused (the re-parsed prose != the brain's proposed triple)
        caught = (not rv["verified"])
        lesion_caught += int(caught)
        lesion_total += 1
        lesion_detail.append({"brain_proposed": [a_, v_, p_], "llm_free_generated_patient": wrong,
                              "surface": rv["surface"], "reparse_svo": rv["reparse_svo"],
                              "verify_rejected": caught})
    lesion_ok = (lesion_total > 0) and (lesion_caught == lesion_total)

    # the grounded RENDER itself must succeed (the brain's TRUE proposition renders+verifies) -- otherwise
    # there is nothing to flag (the honest BOUNDARY case: generation works but the round-trip strips it).
    n_render_ok = sum(1 for r in turns if (r["proposed_triple"] is not None and r["verified"] is True))
    n_proposed = sum(1 for r in turns if r["proposed_triple"] is not None)
    render_rate = (n_render_ok / max(1, n_proposed)) if n_proposed else 0.0

    print(f"\n[probe1 seed {seed}] stored {len(affirmed)} facts ({len(negated)} negated) | "
          f"topics {len(topics)} | tau(P{a.tau_pct})={tau:.3f}", flush=True)
    print(f"  TURNS: proposed {n_proposed}/{len(topics)} | render+VERIFY ok {n_render_ok}/{n_proposed} "
          f"(rate {render_rate:.3f}) | EMITTED {n_emitted}", flush=True)
    print(f"  (1) NOVEL: {n_novel}/{n_emitted} emitted are GENERATED (never stored) -> all-novel "
          f"{all_emitted_novel} (vs 0.0 retrieval ceiling) | topic-relevant {n_topic_relevant}/{n_emitted} "
          f"-> {all_topic_relevant} | opinion-abstained {n_opinion_abstained}/{len(topics)}", flush=True)
    print(f"  (2) GROUNDED: replay TRUE-frac {replay_frac:.3f} vs shuffled {shuf_frac:.4f} -> advantage "
          f"{grounded_advantage:.1f}x (>= {a.advantage_bar}x: {grounded_ok})", flush=True)
    print(f"  (3) FLAGGED+leak-free: {moat_leaks} known-fact-channel leaks (must be 0) | all-flagged "
          f"{all_flagged} -> {flagged_leakfree}", flush=True)
    print(f"  (4) CALIBRATED: spearman(plausibility,confidence)={spearman} monotone={monotone} | "
          f"INDEPENDENT strong-plausible incidence lo-conf {strong_lo} -> hi-conf {strong_hi} "
          f"(rises: {strong_reliability}) -> {calibrated_ok}", flush=True)
    print(f"  ANTI-CHEAT LESION/PROVENANCE: LLM free-generation caught-by-VERIFY {lesion_caught}/{lesion_total} "
          f"-> {lesion_ok}", flush=True)
    if emitted:
        print(f"  example flagged hypotheses:", flush=True)
        for r in emitted[:5]:
            print(f"     X={r['topic']!r:>10} -> {r['reply']!r}  (conf {r['confidence']}, "
                  f"plaus {r['plausibility']})", flush=True)

    return {
        "seed": seed,
        "n_stored": len(affirmed),
        "n_negated": len(negated),
        "n_topics": len(topics),
        "topics": topics,
        "tau": tau,
        # turn outcomes
        "n_proposed": n_proposed,
        "n_render_verify_ok": n_render_ok,
        "render_verify_rate": render_rate,
        "n_emitted": n_emitted,
        # (1) NOVEL
        "n_novel_emitted": n_novel,
        "all_emitted_novel": all_emitted_novel,
        "novel_ratio": novel_ratio,
        "n_topic_relevant": n_topic_relevant,
        "all_topic_relevant": all_topic_relevant,
        "n_opinion_abstained": n_opinion_abstained,
        # (2) GROUNDED
        "replay_true_plausible_frac": replay_frac,
        "shuffled_true_plausible_frac": shuf_frac,
        "grounded_advantage_ratio": grounded_advantage,
        "grounded_ok": grounded_ok,
        # (3) FLAGGED + leak-free
        "moat_leaks": moat_leaks,
        "all_flagged": all_flagged,
        "flagged_leakfree": flagged_leakfree,
        # (4) CALIBRATED
        "calib_spearman": spearman,
        "calib_monotone": monotone,
        "calib_strong_reliability": strong_reliability,
        "calib_strong_incidence_lo_conf": strong_lo,
        "calib_strong_incidence_hi_conf": strong_hi,
        "calibrated_ok": calibrated_ok,
        # ANTI-CHEAT lesion/provenance
        "lesion_caught": lesion_caught,
        "lesion_total": lesion_total,
        "lesion_ok": lesion_ok,
        "lesion_detail": lesion_detail[:6],
        # the turns themselves (trimmed)
        "turns": [{k: v for k, v in r.items() if k != "rendered"} | {
            "rendered_surface": (r.get("rendered") or {}).get("surface"),
            "rendered_reparse": (r.get("rendered") or {}).get("reparse_svo"),
            "rendered_neural_order": (r.get("rendered") or {}).get("neural_order"),
        } for r in turns],
    }


def decide_verdict(rows, a):
    """GO iff, across ALL seeds: (1) every EMITTED proposition is GENERATED (never stored) and at least
    `min_emit` are emitted; (2) the brain's LEARNED graph is load-bearing (shuffled-graph advantage >=
    advantage_bar); (3) the moat is RELAXED-NOT-REMOVED (0 known-fact-channel leaks + every emission flagged);
    (4) confidence tracks plausibility (calibrated_ok where assessable); AND the LESION/PROVENANCE anti-cheat
    holds (LLM free-generation caught-by-VERIFY on every emitted topic -> the content is the BRAIN's).
    Else HONEST_NEGATIVE / BOUNDARY + why."""
    def col(k):
        return [r[k] for r in rows]

    n_emit = np.array(col("n_emitted"))
    novel_all = all(col("all_emitted_novel")) and all(col("all_topic_relevant"))
    grounded_all = all(col("grounded_ok"))
    leakfree_all = all(col("flagged_leakfree"))
    lesion_all = all(col("lesion_ok"))
    cal = col("calibrated_ok")
    # calibration: GO requires it where assessable; None (too-few-emissions) does not fail the gate but is noted
    cal_assessable = [c for c in cal if c is not None]
    calibrated_all = (len(cal_assessable) > 0) and all(cal_assessable)
    enough_emit = bool(np.all(n_emit >= a.min_emit))

    detail = {
        "n_emitted_mean": float(n_emit.mean()),
        "n_emitted_min": int(n_emit.min()),
        "render_verify_rate_mean": float(np.mean(col("render_verify_rate"))),
        "novel_all_seeds": bool(novel_all),
        "novel_ratio_mean": float(np.mean(col("novel_ratio"))),
        "topic_relevant_all_seeds": bool(all(col("all_topic_relevant"))),
        "n_topic_relevant_total": int(np.sum(col("n_topic_relevant"))),
        "n_opinion_abstained_total": int(np.sum(col("n_opinion_abstained"))),
        "grounded_advantage_mean": float(np.mean(col("grounded_advantage_ratio"))),
        "grounded_advantage_min": float(np.min(col("grounded_advantage_ratio"))),
        "grounded_all_seeds": bool(grounded_all),
        "moat_leaks_total": int(np.sum(col("moat_leaks"))),
        "flagged_leakfree_all_seeds": bool(leakfree_all),
        "calib_spearman_mean": float(np.mean([s for s in col("calib_spearman") if s is not None])
                                     if any(s is not None for s in col("calib_spearman")) else float("nan")),
        "calibrated_all_seeds": bool(calibrated_all),
        "calibrated_assessable_seeds": len(cal_assessable),
        "lesion_caught_total": int(np.sum(col("lesion_caught"))),
        "lesion_total": int(np.sum(col("lesion_total"))),
        "lesion_all_seeds": bool(lesion_all),
        "advantage_bar": float(a.advantage_bar),
        "calib_spearman_bar": float(a.calib_spearman_bar),
        "min_emit": int(a.min_emit),
    }

    if not enough_emit:
        verdict = "HONEST_BOUNDARY_too_few_emitted"
    elif not novel_all:
        verdict = "HONEST_NEGATIVE_not_novel"
    elif not grounded_all:
        verdict = "HONEST_NEGATIVE_not_grounded"
    elif not leakfree_all:
        verdict = "HONEST_NEGATIVE_moat_leak"
    elif not lesion_all:
        verdict = "HONEST_NEGATIVE_lesion_not_caught"      # the LLM is doing the cognition -> the cheat
    elif not calibrated_all:
        verdict = "BOUNDARY_uncalibrated"                  # generates+flags, but the hedge doesn't track plausibility
    else:
        verdict = "GO"
    return verdict, detail


def main():
    p = argparse.ArgumentParser(description="Communicable-brain probe 1: a 'what do you think about X' turn "
                                            "that GENERATES a novel grounded flagged-as-hypothesis response.")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--D", type=int, default=64, help="phasor dimension for the RF composer store")
    p.add_argument("--n-facts", type=int, default=24, help="AFFIRMED facts the brain is TOLD")
    p.add_argument("--n-negated", type=int, default=12, help="NEGATED facts (non-contradiction gate work)")
    p.add_argument("--n-topics", type=int, default=30, help="held-out 'what do you think about X' topics")
    p.add_argument("--n-attempts", type=int, default=500, help="generative-replay samples per topic")
    p.add_argument("--n-attempts-grounded", type=int, default=3000,
                   help="replay samples for the shuffled-graph grounded control (the b2-style advantage ratio)")
    p.add_argument("--tau-pct", type=float, default=50.0, help="graph-related threshold = percentile of +PPMI")
    p.add_argument("--advantage-bar", type=float, default=3.0, help="grounded shuffled-graph advantage ratio bar")
    p.add_argument("--calib-spearman-bar", type=float, default=0.5,
                   help="min Spearman(plausibility, stated confidence) for CALIBRATED")
    p.add_argument("--min-emit", type=int, default=3, help="min emitted flagged hypotheses per seed")
    p.add_argument("--neural-order", action="store_true",
                   help="exercise the NEURAL serial-order renderer for word order (builds a GPU pool bridge)")
    p.add_argument("--host-oracle-sampler", action="store_true",
                   help="pin the generative DRAW to the host np.random.choice ORACLE (default OFF = the validated "
                        "spiking soft-WTA draw is default-on); for the host-parity escape check")
    p.add_argument("--max-bytes", type=int, default=4_000_000)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--repeat-cap", type=int, default=40)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")

    seeds = [int(s.strip()) for s in a.seeds.split(",")]
    t0 = time.time()
    print(f"[probe1] seeds={seeds} D={a.D} n_facts={a.n_facts} n_topics={a.n_topics} -- can a 'what do you "
          f"think about X' turn GENERATE a novel grounded flagged-as-hypothesis reply on the brain's own "
          f"substrate (LLM fluency-only)?", flush=True)

    vocab, cat_ids, cat_names = taxonomy_to_vocab_categories(TAXONOMY_8x8)
    corpus_path = os.path.join(_REPO, "data", "corpus", "tinystories.txt")
    if not os.path.exists(corpus_path):
        print(f"[ERROR] corpus not found: {corpus_path}", flush=True)
        sys.exit(2)
    corpus = build_real_cooccurrence(corpus_path, vocab, cat_ids, window=a.window, repeat_cap=a.repeat_cap,
                                     seed=42, max_bytes=a.max_bytes, freq_floor=30,
                                     min_facts_per_category=20, verbose=True)

    rows = [run_seed(s, vocab, corpus, a) for s in seeds]
    verdict, detail = decide_verdict(rows, a)

    print(f"\n{'='*100}", flush=True)
    print(f"  OVERALL VERDICT: {verdict}", flush=True)
    print(f"  EMITTED flagged hypotheses: mean {detail['n_emitted_mean']:.1f} (min {detail['n_emitted_min']}); "
          f"render+VERIFY rate {detail['render_verify_rate_mean']:.3f}", flush=True)
    print(f"  (1) NOVEL all seeds: {detail['novel_all_seeds']} (novel-ratio mean {detail['novel_ratio_mean']:.2f}; "
          f"vs 0.0 retrieval ceiling)", flush=True)
    print(f"  (2) GROUNDED all seeds: {detail['grounded_all_seeds']} (shuffled-graph advantage mean "
          f"{detail['grounded_advantage_mean']:.1f}x, min {detail['grounded_advantage_min']:.1f}x; bar "
          f"{detail['advantage_bar']}x)", flush=True)
    print(f"  (3) FLAGGED+leak-free all seeds: {detail['flagged_leakfree_all_seeds']} "
          f"({detail['moat_leaks_total']} total known-fact-channel leaks)", flush=True)
    print(f"  (4) CALIBRATED all seeds: {detail['calibrated_all_seeds']} "
          f"(spearman mean {detail['calib_spearman_mean']:.3f}; assessable seeds "
          f"{detail['calibrated_assessable_seeds']}/{len(seeds)})", flush=True)
    print(f"  ANTI-CHEAT LESION/PROVENANCE all seeds: {detail['lesion_all_seeds']} "
          f"(LLM free-generation caught-by-VERIFY {detail['lesion_caught_total']}/{detail['lesion_total']})",
          flush=True)
    print(f"  elapsed {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*100}\n", flush=True)

    out = {
        "probe": "communicable_brain_probe1_whatdoyouthink",
        "verdict": verdict,
        "seeds": seeds,
        "config": {"D": a.D, "n_facts": a.n_facts, "n_negated": a.n_negated, "n_topics": a.n_topics,
                   "n_attempts": a.n_attempts, "tau_pct": a.tau_pct, "advantage_bar": a.advantage_bar,
                   "calib_spearman_bar": a.calib_spearman_bar, "min_emit": a.min_emit,
                   "neural_order": a.neural_order, "host_oracle_sampler": a.host_oracle_sampler,
                   "use_spiking_sampler": (not a.host_oracle_sampler), "max_bytes": a.max_bytes},
        "baseline_to_beat": {"measured_retrieval_novel_composition": 0.0,
                             "source": "2026-06-22-generation-novelty-categorical-gap-MEASURED.md"},
        "pipeline": (
            "ASSIMILATE(X) [PPMI cortex neighborhood] -> PROPOSE [b2 GenerativeReplayProposer: novel-but-plausible "
            "graph-biased, moat-gated, non-contradictory] -> RENDER [neural serial-order word order (opt-in) + "
            "fluency-only faculty surface form] + VERIFY [content-extractor + BridgeParser re-parse of the PROSE; "
            "reject on drift] -> EMIT [graded-confidence FLAGGED hypothesis; NOT stored; known-fact channel stays "
            "hard-gated]."),
        "go_pieces_composed": {
            "propose": "research/runners/_genfrontier_b2_generative_replay_derisk.py (GenerativeReplayProposer) "
                       "[2026-06-23-genfrontier-b2-generative-replay-derisk.md, GO 6-seed]",
            "render_word_order": "research/runners/neural_serial_order_renderer.py (NeuralSerialOrderRenderer) "
                                 "[2026-06-16-sentence-generation-serial-order-cheap-first-GO.md, GO 6/6]",
            "render_verify": "research/runners/_grounded_lang_p3_derisk.py (gate->constrain->VERIFY; "
                             "_grounded_lang_integration_derisk._extract_svo_from_prose + the BridgeParser) "
                             "[2026-06-23-grounded-lang-INTEGRATION-GO.md, GO]",
        },
        "fluency_only_note": (
            "the fluency faculty supplies SURFACE FORM only; the BRAIN (PPMI cortex + RF composer) supplies the "
            "proposition (content + choice) and VERIFY re-parses the faculty's prose. CPU stand-ins reused from "
            "the GO P3 loop (the SAME VERIFY contract the real spiking-Qwen integration passed): GROUNDED = "
            "TemplateStubFaculty (content-locked); LESION = InjectingStubFaculty (free-generates content -> VERIFY "
            "MUST reject -> the content is the BRAIN's, not the LLM's). The real spiking-Qwen faculty "
            "(_grounded_lang_integration_derisk.SpikingQwenFaculty) is the GPU follow-on; this CPU probe is the "
            "cheap-first composition de-risk per the scoping."),
        "detail": detail,
        "per_seed": rows,
        "elapsed_total_s": time.time() - t0,
    }
    if a.out is None:
        a.out = os.path.join(_REPO, "research", "findings", "raw",
                             "_communicable_brain_probe1_whatdoyouthink.json")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {a.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
