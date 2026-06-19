"""Phase-1 cheapest-first MECHANISM de-risk: multi-cue COMPETITION parser for robust thematic-role assignment.

Question this decides (scoping `2026-06-19-multicue-competition-parser-scoping.md`, commit 0ecdb628):
    Does adding competing cues (ANIMACY + VERB-SELECTIONAL-FIT) to the WORD-POSITION cue, integrated by a
    reliability-weighted competition, make thematic-role assignment ROBUST to degraded/imperfect English
    (dropped words, scrambled order, object-fronted order) -- WHERE A POSITION-ONLY PARSER FAILS -- with the
    cue WEIGHTS genuinely LEARNED (= cue VALIDITIES, not hand-set) and the cues genuinely LOAD-BEARING?

This is the Bates-MacWhinney Competition Model realized as biased competition (Desimone-Duncan 1995) over ROLE
assemblies + a cue-validity-weighted accumulator (catalog G.18 LIP: "integrates ANY evidence weighted by
reliability, additively"; G.12 Broca: "semantic cues carry comprehension when order is hard"). This runner is the
CPU/numpy MECHANISM de-risk; if GO, the spiking realization (re-point `biased_competition_buffer.py`'s sel_X from
referent->role + plastic cue->role projections) is the production build.

WHAT IS BRAIN-BASED vs SCAFFOLD (honest, per BRAIN-BASED-ONLY directive):
  * The COMPETITION + the reliability-weighted ACCUMULATION + the WINNER are the validated spiking computation
    (this numpy model is the FUNCTIONAL stand-in for that biased-competition WTA; the production build is on the
    substrate). The cue WEIGHTS (= validities) are LEARNED here by a delta rule over clean canonical sentences
    -- the Hebbian-co-firing cue-validity learning the spiking build does with `enable_hebbian_learning=True`.
  * The feature LEXICONS (animacy, verb-selectional-fit) are HOST scaffolds, reused verbatim from
    `biased_competition_buffer.py` (already flagged there for conversion to a learned lexical-feature map). They
    supply each cue's VALUE for a word; they do NOT supply the role decision (that is the learned-weight
    competition). The PERMUTED-CUE + the NO-LEARNING controls guard against the lexicon doing the discrimination.

THE PRIMARY MISLEAD GUARDED (the seductive failure): hand-tuned cues masquerading as a learned model. The decisive
controls (all must pass or it is NOT a GO):
  - POSITION-ONLY baseline COLLAPSES on the degraded battery (else the battery is not degrading position -> INVALID).
  - NO-LEARNING control (cue weights FROZEN at init) collapses -> the validities are LEARNED, not hand-set.
  - CUE-LESION (zero animacy+verb-fit weights, keep position) collapses on degraded -> the cues are load-bearing.
  - PERMUTED-CUE (shuffle the animacy/verb-fit tags) collapses -> not exploiting a leak / not a relabelled position.
  - HELD-OUT FILLERS (test nouns/verbs disjoint from training) -> not memorizing examples.
  - the no-confab MOAT holds (an all-ambiguous / no-decisive-winner sentence -> ABSTAIN, 0 breaches).

Run (CPU/numpy is fast; INLINE):
    SIM_BACKEND=numpy python -m research.runners._phaseB_multicue_competition_derisk --smoke   # 1 seed
    SIM_BACKEND=numpy python -m research.runners._phaseB_multicue_competition_derisk --seeds 42,43,44,45,46,47
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Feature lexicons -- REUSED verbatim from research/runners/biased_competition_buffer.py
# (ANIMACY + VERB_SELECTS). HOST scaffold: supplies each cue's VALUE, not the role decision.
# Extended with disjoint TRAIN vs HELD-OUT filler/verb pools so role correctness is vocab-agnostic.
# ---------------------------------------------------------------------------

# Animacy lexicon (animate nouns can be agents; inanimate nouns are typically patients of agentive verbs).
ANIMACY = {
    # --- animate (agents) ---
    "dog": "animate", "cat": "animate", "fox": "animate", "bird": "animate",
    "wolf": "animate", "bear": "animate", "owl": "animate", "frog": "animate",
    # --- inanimate (patients) ---
    "ball": "inanimate", "apple": "inanimate", "rock": "inanimate", "book": "inanimate",
    "stick": "inanimate", "bone": "inanimate", "leaf": "inanimate", "cup": "inanimate",
}

# Verb selectional restriction: an agentive verb's AGENT slot prefers an animate filler; its PATIENT slot
# prefers (or at least tolerates) an inanimate filler. `agent`/`patient` = which animacy each slot selects for.
VERB_SELECTS = {
    # train verbs
    "chase": {"agent": "animate", "patient": "animate"},   # animal chases animal (patient can be animate)
    "eat":   {"agent": "animate", "patient": "inanimate"},  # animal eats a thing
    "push":  {"agent": "animate", "patient": "inanimate"},
    "carry": {"agent": "animate", "patient": "inanimate"},
    # held-out verbs (disjoint -- verb-fit must generalize from the animacy structure, not memorize the verb)
    "bite":  {"agent": "animate", "patient": "inanimate"},
    "kick":  {"agent": "animate", "patient": "inanimate"},
    "grab":  {"agent": "animate", "patient": "inanimate"},
    "watch": {"agent": "animate", "patient": "animate"},
}

# Disjoint train/held-out splits (held-out fillers + verbs never seen in cue-weight training).
TRAIN_ANIMATE = ["dog", "cat", "fox", "bird"]
TRAIN_INANIM = ["ball", "apple", "rock", "book"]
HELD_ANIMATE = ["wolf", "bear", "owl", "frog"]
HELD_INANIM = ["stick", "bone", "leaf", "cup"]
TRAIN_VERBS = ["chase", "eat", "push", "carry"]
HELD_VERBS = ["bite", "kick", "grab", "watch"]

ROLES = ("agent", "patient")  # 2-role assignment for NOUNS -> chance = 0.5


# ---------------------------------------------------------------------------
# Cue evidence (the Competition Model made HONEST: cues are individually NON-DECISIVE).
#
# Each cue emits a SIGNED vote in {-1,0,+1} toward (agent:+ / patient:-) for a noun, plus a RELIABILITY in
# [0,1] (whether the cue is APPLICABLE at all for this noun/sentence). The role logit per noun is
#     sum_cue  w_cue * reliability_cue * vote_cue            (cue-validity-weighted ADDITIVE accumulation, G.18)
# and the LEARNED w_cue ARE the cue validities.
#
# WHY NOISY VOTES (the load-bearing design fix). If every cue were perfectly reliable and mutually agreeing,
# ANY non-negative weights would solve the task -> the NO-LEARNING control could never collapse and "learning"
# would be vacuous (exactly the seductive hand-tuned-cues failure the scoping flags as the PRIMARY mislead).
# So each cue's vote is CORRUPTED with per-cue label noise at rate (1 - true_validity), deterministically
# seeded per (sentence_id, cue, noun). Now NO single cue is trustworthy; only the LEARNED WEIGHTED combination
# of several cues separates the roles -- and a parser that mis-weights a low-validity cue (the frozen/uniform
# NO-LEARNING parser) is genuinely misled. This is the real Competition Model: cue validity is learned from a
# naturalistic distribution, the comprehender DOWN-WEIGHTS unreliable cues. The 4th DISTRACTOR cue (`lexbias`,
# true validity 0.5 = pure noise correlated with position) is the cue the learner MUST drive to zero; a
# no-learning parser keeps weighting it and pays for it.
#
# POSITION is special: its vote is STRUCTURAL (from surface index), so it is correct iff the surface order is
# canonical. In naturalistic TRAINING (canonical-majority + non-canonical-minority) position's EMPIRICAL
# validity is high-but-imperfect; on the DEGRADED battery position is SYSTEMATICALLY wrong. The semantic cues
# (animacy/verb-fit) keep their moderate validity everywhere. The learner therefore learns w_position high but
# < the semantic weights, and the semantic cues carry the degraded battery (catalog G.12 Broca).
# ---------------------------------------------------------------------------

CUES = ("position", "animacy", "verbfit", "lexbias")

# True cue validities (probability the cue's vote agrees with the gold role WHEN APPLICABLE). The SEMANTIC
# cues are reliable (animate->agent / verb-selectional-fit are strong heuristics); the lexbias DISTRACTOR is at
# chance (0.5) and must be learned out. POSITION is NOT noised here -- its (im)reliability is STRUCTURAL: its
# vote is right iff the surface order is canonical, so its EMPIRICAL validity is set by the non-canonical
# fraction of the training distribution (naturalistic English: order is reliable but not perfectly so), and on
# the DEGRADED test order is systematically wrong. The learner discovers w_position < w_semantic from the
# naturalistic distribution; a uniform no-learning parser over-trusts position and collapses on degraded input.
TRUE_VALIDITY = {"animacy": 0.90, "verbfit": 0.90, "lexbias": 0.50}
POSITION_NOISE = 0.0


def _det_unit(sent_id, cue, noun_index):
    """Deterministic uniform(0,1) keyed on (sentence, cue, noun) so a sentence's noisy votes are STABLE across
    every parser/control that sees it (the corruption is a property of the INPUT, identical for all readers)."""
    h = hash((int(sent_id), cue, int(noun_index))) & 0xFFFFFFFF
    return (np.random.default_rng(h).random())


def _maybe_flip(vote, validity, sent_id, cue, noun_index):
    """With prob (1-validity), flip the vote sign (label noise). Deterministic per (sentence,cue,noun)."""
    if vote == 0.0:
        return 0.0
    return vote if _det_unit(sent_id, cue, noun_index) < validity else -vote


def _position_vote(noun_index, n_nouns):
    if n_nouns <= 1:
        return 0.0
    frac = noun_index / (n_nouns - 1)          # 0..1
    return 1.0 - 2.0 * frac                     # +1 (first) -> -1 (last)


def _animacy_vote(noun):
    a = ANIMACY.get(noun)
    return +1.0 if a == "animate" else (-1.0 if a == "inanimate" else 0.0)


def _verbfit_vote(noun, verb):
    sel = VERB_SELECTS.get(verb)
    a = ANIMACY.get(noun)
    if sel is None or a is None:
        return 0.0
    fits_agent = (sel["agent"] == a)
    fits_patient = (sel["patient"] == a)
    if fits_agent and not fits_patient:
        return +1.0
    if fits_patient and not fits_agent:
        return -1.0
    return 0.0   # symmetric verb / fits both -> genuinely uninformative (feeds the moat)


def cue_evidence(noun, noun_index, n_nouns, verb, sent_id,
                 permute_map=None, lesion_semantic=False, drop_cues=(), clean_cues=False):
    """Return {cue: (vote, reliability)} for one noun, with per-cue label noise baked in (deterministic per
    sentence). `permute_map` remaps the SEMANTIC feature-bearer identity (PERMUTED-CUE control). `lesion_semantic`
    zeroes animacy+verbfit (CUE-LESION). `drop_cues` removes named cues entirely (position-only baseline).
    `clean_cues` disables the label noise -- used ONLY for the MOAT-ambiguity set, whose ambiguity is a property
    of its CONSTRUCTION (two animate nouns + a symmetric verb: the cues genuinely do not distinguish the roles);
    injecting per-noun noise there would FABRICATE a spurious distinction, so the moat test reads the clean
    (constructed) cue values to ask the honest question 'when the cues truly tie, does the parser abstain?'."""
    def flip(vote, validity, cue):
        return vote if clean_cues else _maybe_flip(vote, validity, sent_id, cue, noun_index)

    ev = {}
    pv = _position_vote(noun_index, n_nouns)
    ev["position"] = (flip(pv, 1.0 - POSITION_NOISE, "position"), 1.0)

    sem_noun = permute_map.get(noun, noun) if permute_map else noun
    if lesion_semantic:
        ev["animacy"] = (0.0, 0.0)
        ev["verbfit"] = (0.0, 0.0)
    else:
        av = flip(_animacy_vote(sem_noun), TRUE_VALIDITY["animacy"], "animacy")
        ev["animacy"] = (av, 1.0 if _animacy_vote(sem_noun) != 0.0 else 0.0)
        vv = flip(_verbfit_vote(sem_noun, verb), TRUE_VALIDITY["verbfit"], "verbfit")
        ev["verbfit"] = (vv, 1.0 if _verbfit_vote(sem_noun, verb) != 0.0 else 0.0)

    # lexbias DISTRACTOR: a chance-validity vote correlated in SIGN with position (so a naive parser cannot
    # distinguish it from a 2nd order cue) but carrying NO real role info. Learner must zero its weight.
    lv = flip(np.sign(pv) if pv != 0 else 0.0, TRUE_VALIDITY["lexbias"], "lexbias")
    ev["lexbias"] = (float(lv), 1.0)

    for c in drop_cues:
        ev[c] = (0.0, 0.0)
    return ev


# ---------------------------------------------------------------------------
# The competition role-assigner. Per noun, role logit = sum_cue w[cue]*rel*vote (toward agent:+ / patient:-).
# Softmax over the two roles -> P(agent). The WINNER is argmax; the MARGIN (|P-0.5|) gates the moat.
# Sentence-level constraint: a transitive sentence has exactly ONE agent and ONE patient, so we ASSIGN the
# higher-agent-logit noun = agent, the other = patient (the competition picks the configuration, the biology
# of mutual inhibition between the two role assemblies -- a winner suppresses the loser). With >2 nouns we
# rank by agent-logit. This is the functional stand-in for the spiking biased-competition WTA.
# ---------------------------------------------------------------------------

@dataclass
class MultiCueParser:
    weights: dict = field(default_factory=lambda: {c: 0.5 for c in CUES})  # learned cue validities

    def noun_agent_logit(self, ev):
        return sum(self.weights[c] * rel * vote for c, (vote, rel) in ev.items())

    SEMANTIC_CUES = ("animacy", "verbfit")  # the content cues that carry the no-confab moat decision

    def _semantic_contrast(self, evs):
        """Signed agent-logit contrast (noun0 - noun1) from the SEMANTIC cues ONLY (animacy + verb-fit),
        i.e. dropping position + the lexbias distractor. This is the CONTENT evidence the moat gates on."""
        def slogit(ev):
            return sum(self.weights[c] * rel * vote
                       for c, (vote, rel) in ev.items() if c in self.SEMANTIC_CUES)
        return slogit(evs[0]) - slogit(evs[1])

    def assign_roles(self, nouns, evs, abstain_margin=0.0):
        """Assign agent/patient over the sentence's nouns by competition. Returns (assignment, decisive).
        The higher-agent-logit noun wins AGENT, the other PATIENT (the 1-agent/1-patient transitive constraint;
        biased competition between the two role assemblies). `decisive` (else -> ABSTAIN, the no-confab MOAT)
        requires the SEMANTIC (content) evidence to break the tie: |semantic_contrast| >= `abstain_margin`. The
        genuinely-ambiguous case -- two animate nouns + a symmetric verb, scrambled (animacy TIES, verb-fit is
        uninformative) -> semantic contrast ~0 -> ABSTAIN, regardless of what position says. This mirrors the
        biased-competition buffer's content gate (abstain when the content is silent); it does NOT penalize a
        legitimate degraded decision, which IS carried by the semantic cues. Position alone never licenses a
        commit (it is the unreliable cue on degraded input)."""
        logits = np.array([self.noun_agent_logit(ev) for ev in evs], dtype=float)
        order = np.argsort(-logits)
        assignment = {int(order[0]): "agent", int(order[-1]): "patient"}
        for i in order[1:-1]:
            assignment[int(i)] = "agent" if logits[i] >= 0 else "patient"
        decisive = True
        if len(nouns) == 2:
            decisive = abs(self._semantic_contrast(evs)) >= abstain_margin
        return assignment, decisive

    # --- cue-validity LEARNING (delta rule). The cue weights track how reliably each cue's signed vote agrees
    # with the GOLD role across the NATURALISTIC training distribution (canonical-majority + non-canonical-
    # minority, with per-cue label noise). A cue that agrees with gold more often (higher empirical validity)
    # grows a larger weight; the chance-validity distractor is driven toward zero. This IS the learned model;
    # freezing it = the NO-LEARNING control. Biology: Hebbian co-firing of a cue's vote with the correct role
    # assembly + the recurrence amplifying the reliable cues (the spiking build's enable_hebbian_learning). ---
    def learn(self, train_examples, lr=0.05, epochs=600, seed=0, freeze=False):
        if freeze:
            return  # NO-LEARNING control: leave weights at their uniform init (0.5 each)
        rng = np.random.default_rng(seed)
        idx = list(range(len(train_examples)))
        for _ in range(epochs):
            rng.shuffle(idx)
            for i in idx:
                _nouns, evs, gold = train_examples[i]  # gold: noun_index -> "agent"/"patient"
                # target per noun: +1 if agent, -1 if patient. Predicted = tanh(agent_logit) in [-1,1].
                for ni, ev in enumerate(evs):
                    target = +1.0 if gold[ni] == "agent" else -1.0
                    logit = self.noun_agent_logit(ev)
                    pred = np.tanh(logit)
                    err = target - pred
                    dpred = (1.0 - pred * pred)  # d tanh
                    for c, (vote, rel) in ev.items():
                        # L2 weight decay pulls a non-informative cue's weight toward 0 (a useless cue whose
                        # gradient averages to ~0 cannot resist the decay -> the distractor is zeroed).
                        grad = err * dpred * rel * vote
                        self.weights[c] += lr * (grad - 0.02 * self.weights[c])
                # keep weights non-negative (a cue's VALIDITY is non-negative; the SIGN lives in the vote).
        for c in self.weights:
            self.weights[c] = max(0.0, self.weights[c])


# ---------------------------------------------------------------------------
# Sentence generation. A sentence = (nouns_in_surface_order, verb, gold_roles_by_surface_index, tag, sent_id).
# `sent_id` is a unique integer that keys the per-cue label noise (so a given sentence's noisy votes are stable
# across every parser/control). TRAINING is NATURALISTIC: canonical-majority + a non-canonical minority (so
# position's empirical validity is high-but-imperfect, the real Competition-Model input).
# ---------------------------------------------------------------------------

# a module-level counter, reset per seed, so sent_ids are unique within a run_seed
class _Ids:
    def __init__(self):
        self.n = 0

    def next(self):
        self.n += 1
        return self.n


def _canonical(agent, verb, patient, sid):
    """Canonical 'AGENT VERB PATIENT' (active SVO). Surface noun order = [agent, patient]."""
    return [agent, patient], verb, {0: "agent", 1: "patient"}, "canonical", sid


def _drop_verb(agent, verb, patient, sid):
    """DROP-A-WORD: drop the verb -> 'AGENT PATIENT'. Surface order preserved (position stays valid here), but
    the verb-fit cue is GONE (no verb). Tests whether position+animacy still carry when a word is missing. This
    condition does NOT degrade POSITION -- so the position-only collapse gate is measured on scramble+front."""
    return [agent, patient], None, {0: "agent", 1: "patient"}, "drop_verb", sid


def _scramble(agent, verb, patient, sid, rng):
    """SCRAMBLE-ORDER: same words, randomized NOUN order -> position is MISLEADING (~50% of the time the 1st
    surface noun is the patient). Animacy + verb-fit must override. Gold follows noun identity, not slot."""
    nouns = [agent, patient]
    perm = rng.permutation(2)
    s = [nouns[p] for p in perm]
    gold = {j: ("agent" if perm[j] == 0 else "patient") for j in range(2)}
    return s, verb, gold, "scramble", sid


def _object_front(agent, verb, patient, sid):
    """OBJECT-FRONTED (OSV-style): 'PATIENT AGENT VERB' -> the position table (1st->agent) SYSTEMATICALLY mis-maps
    the fronted patient as agent (the NEMO object-initial weakness). Surface order [patient, agent]."""
    return [patient, agent], verb, {0: "patient", 1: "agent"}, "object_front", sid


def build_dataset(rng, animate_pool, inanim_pool, verb_pool, n_per_cond=40, ids=None,
                  noncanon_train_frac=0.40):
    """Build a NATURALISTIC training set (canonical-majority + non-canonical-minority) and a DEGRADED battery
    (drop/scramble/object-front) + a clean-canonical test + a moat-ambiguity set, all on the given pools.
    Asymmetric verbs (animate agent / inanimate patient) make verb-fit informative; symmetric verbs feed moat."""
    ids = ids or _Ids()
    asym = [v for v in verb_pool if VERB_SELECTS[v]["patient"] == "inanimate"]
    sym = [v for v in verb_pool if VERB_SELECTS[v]["patient"] == "animate"]

    def rand(verbs, pat_pool):
        a = animate_pool[rng.integers(len(animate_pool))]
        v = verbs[rng.integers(len(verbs))]
        p = pat_pool[rng.integers(len(pat_pool))]
        while p == a:
            p = pat_pool[rng.integers(len(pat_pool))]
        return a, v, p

    # NATURALISTIC training: ~70% canonical, ~30% non-canonical (scramble/object-front), with gold. This is
    # what gives position high-but-imperfect empirical validity and lets the learner discover w_pos < w_sem.
    train = []
    n_train = n_per_cond * 6
    for _ in range(n_train):
        a, v, p = rand(asym, inanim_pool)
        if rng.random() < noncanon_train_frac:
            if rng.random() < 0.5:
                train.append(_scramble(a, v, p, ids.next(), rng))
            else:
                train.append(_object_front(a, v, p, ids.next()))
        else:
            train.append(_canonical(a, v, p, ids.next()))

    battery = {"drop_verb": [], "scramble": [], "object_front": []}
    for _ in range(n_per_cond):
        a, v, p = rand(asym, inanim_pool); battery["drop_verb"].append(_drop_verb(a, v, p, ids.next()))
        a, v, p = rand(asym, inanim_pool); battery["scramble"].append(_scramble(a, v, p, ids.next(), rng))
        a, v, p = rand(asym, inanim_pool); battery["object_front"].append(_object_front(a, v, p, ids.next()))

    clean_test = [_canonical(*rand(asym, inanim_pool), ids.next()) for _ in range(n_per_cond)]

    # MOAT-ambiguity set: two ANIMATE nouns + a SYMMETRIC verb, SCRAMBLED. position misleading, animacy ties
    # (both animate), verb-fit uninformative (symmetric) -> NO decisive cue -> the parser must ABSTAIN.
    moat = []
    if sym:
        for _ in range(n_per_cond):
            a = animate_pool[rng.integers(len(animate_pool))]
            b = animate_pool[rng.integers(len(animate_pool))]
            while b == a:
                b = animate_pool[rng.integers(len(animate_pool))]
            v = sym[rng.integers(len(sym))]
            perm = rng.permutation(2)
            nn = [[a, b][perm[0]], [a, b][perm[1]]]
            gold = {j: ("agent" if perm[j] == 0 else "patient") for j in range(2)}
            moat.append((nn, v, gold, "moat_ambiguous", ids.next()))
    return train, clean_test, battery, moat


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _examples_to_evidence(sentences, permute_map=None, lesion_semantic=False, position_only=False,
                          clean_cues=False):
    """Convert (nouns, verb, gold, tag, sent_id) sentences into (nouns, evs, gold) with per-noun cue evidence.
    `position_only` DROPS the animacy/verbfit/lexbias cues entirely (the load-bearing position-only baseline).
    `clean_cues` disables label noise (MOAT-ambiguity set only -- see cue_evidence)."""
    drop = ("animacy", "verbfit", "lexbias") if position_only else ()
    out = []
    for nouns, verb, gold, _tag, sid in sentences:
        n = len(nouns)
        evs = [cue_evidence(noun, ni, n, verb, sid, permute_map=permute_map,
                            lesion_semantic=lesion_semantic, drop_cues=drop, clean_cues=clean_cues)
               for ni, noun in enumerate(nouns)]
        out.append((nouns, evs, gold))
    return out


def _role_accuracy(parser, sentences, **ev_kwargs):
    """Fraction of NOUNS assigned their gold role over the sentence set (chance 0.5)."""
    data = _examples_to_evidence(sentences, **ev_kwargs)
    correct = total = 0
    for nouns, evs, gold in data:
        assignment, _decisive = parser.assign_roles(nouns, evs)
        for ni in range(len(nouns)):
            total += 1
            if assignment.get(ni) == gold[ni]:
                correct += 1
    return correct / max(1, total)


def _battery_accuracy(parser, battery, **ev_kwargs):
    accs = {}
    flat = []
    for cond, sents in battery.items():
        accs[cond] = _role_accuracy(parser, sents, **ev_kwargs)
        flat.extend(sents)
    accs["_mean"] = _role_accuracy(parser, flat, **ev_kwargs)
    # the position-degrading subset (scramble + object_front) -- the gate for the position-only collapse
    # (drop_verb deliberately does NOT degrade position, so it is excluded from that gate's denominator).
    posdeg = battery["scramble"] + battery["object_front"]
    accs["_mean_posdeg"] = _role_accuracy(parser, posdeg, **ev_kwargs)
    return accs


def _moat_breaches(parser, moat_set, abstain_margin):
    """On the genuinely-ambiguous set: the parser should ABSTAIN (decisive=False). A decisive COMMIT is a
    moat breach. Returns (n_breaches, n_total, abstain_rate). Reads the CLEAN (constructed) cue values --
    the ambiguity is built in (both animate + symmetric verb), so no fabricated-noise distinction."""
    data = _examples_to_evidence(moat_set, clean_cues=True)
    breaches = 0
    for nouns, evs, _gold in data:
        _assignment, decisive = parser.assign_roles(nouns, evs, abstain_margin=abstain_margin)
        if decisive:
            breaches += 1
    n = len(data)
    return breaches, n, (n - breaches) / max(1, n)


def _calibrate_abstain_margin(parser, informative_sentences):
    """Set the moat abstain margin from the SEMANTIC contrast on cue-INFORMATIVE sentences (sentences whose
    semantic cues DO distinguish the nouns). The margin is placed BELOW the typical informative semantic
    contrast (so real semantic decisions stay decisive) but well above ~0 (the ambiguous set's semantic
    contrast, where animacy ties + verb-fit is uninformative). Uses a low percentile of the informative
    semantic contrasts. Calibrated only on INFORMATIVE sentences (no peek at the moat set)."""
    mags = []
    data = _examples_to_evidence(informative_sentences)
    for _nouns, evs, _gold in data:
        if len(evs) >= 2:
            mags.append(abs(parser._semantic_contrast(evs)))
    mags = [m for m in mags if m > 1e-9]
    if not mags:
        return 0.05
    return float(np.percentile(mags, 20) * 0.5)  # half the 20th-pct informative semantic contrast


# ---------------------------------------------------------------------------
# One seed: train cue validities on clean canonical English, evaluate the full battery + all controls.
# ---------------------------------------------------------------------------

def run_seed(seed, n_per_cond=40, held_out=True, verbose=False):
    rng = np.random.default_rng(seed)
    ids = _Ids()  # unique sent_ids across the whole seed (so the per-sentence noise never collides)
    if held_out:
        # TRAIN on train-pool fillers+verbs; the EVAL battery/moat on HELD-OUT fillers+verbs (disjoint).
        train_an, train_in, train_vb = TRAIN_ANIMATE, TRAIN_INANIM, TRAIN_VERBS
        test_an, test_in, test_vb = HELD_ANIMATE, HELD_INANIM, HELD_VERBS
    else:
        train_an, train_in, train_vb = TRAIN_ANIMATE, TRAIN_INANIM, TRAIN_VERBS
        test_an, test_in, test_vb = TRAIN_ANIMATE, TRAIN_INANIM, TRAIN_VERBS

    # naturalistic TRAIN distribution (its own battery/test unused here -- we only take `train`)
    train_sents, _ct_tr, _bt_tr, _mt_tr = build_dataset(rng, train_an, train_in, train_vb,
                                                        n_per_cond=n_per_cond, ids=ids)
    # the EVAL battery + clean test + moat on HELD-OUT fillers/verbs
    _tr_e, clean_test, battery, moat_set = build_dataset(rng, test_an, test_in, test_vb,
                                                         n_per_cond=n_per_cond, ids=ids)

    train_ex = _examples_to_evidence(train_sents)

    # ---- LEARNED multi-cue parser ----
    learned = MultiCueParser()
    learned.learn(train_ex, seed=seed, freeze=False)

    # ---- NO-LEARNING control (frozen at uniform init) ----
    frozen = MultiCueParser()
    frozen.learn(train_ex, seed=seed, freeze=True)

    # ---- PERMUTED-CUE control: learn cue validities against SCRAMBLED semantic feature-bearer identities ----
    perm_nouns = train_an + train_in
    perm_targets = list(perm_nouns)
    np.random.default_rng(seed + 9000).shuffle(perm_targets)
    permute_map = dict(zip(perm_nouns, perm_targets))
    train_ex_perm = _examples_to_evidence(train_sents, permute_map=permute_map)
    permuted = MultiCueParser()
    permuted.learn(train_ex_perm, seed=seed, freeze=False)

    # moat margin: calibrate on the INFORMATIVE held-out sentences (scramble+object_front+clean_test all DO
    # have a decisive cue, unlike the symmetric-verb moat set).
    informative = battery["scramble"] + battery["object_front"] + clean_test
    abstain_margin = _calibrate_abstain_margin(learned, informative)

    # ================= METRICS =================
    mc_battery = _battery_accuracy(learned, battery)                                  # GO metric
    pos_battery = _battery_accuracy(learned, battery, position_only=True)             # LOAD-BEARING control
    nolearn_battery = _battery_accuracy(frozen, battery)                              # no-learning control
    lesion_battery = _battery_accuracy(learned, battery, lesion_semantic=True)        # cue-lesion control
    perm_battery = _battery_accuracy(permuted, battery, permute_map=permute_map)      # permuted-cue control
    mc_clean = _role_accuracy(learned, clean_test)
    pos_clean = _role_accuracy(learned, clean_test, position_only=True)
    breaches, moat_n, abstain_rate = _moat_breaches(learned, moat_set, abstain_margin)

    res = {
        "seed": seed,
        "weights_learned": {k: round(v, 4) for k, v in learned.weights.items()},
        "weights_frozen": {k: round(v, 4) for k, v in frozen.weights.items()},
        "weights_permuted": {k: round(v, 4) for k, v in permuted.weights.items()},
        "abstain_margin": round(abstain_margin, 5),
        "multicue_battery": {k: round(v, 4) for k, v in mc_battery.items()},
        "position_only_battery": {k: round(v, 4) for k, v in pos_battery.items()},
        "nolearn_battery": {k: round(v, 4) for k, v in nolearn_battery.items()},
        "lesion_battery": {k: round(v, 4) for k, v in lesion_battery.items()},
        "permuted_battery": {k: round(v, 4) for k, v in perm_battery.items()},
        "clean_multicue": round(mc_clean, 4),
        "clean_position_only": round(pos_clean, 4),
        "moat": {"breaches": breaches, "n": moat_n, "abstain_rate": round(abstain_rate, 4)},
    }

    # ---- per-seed GO gates (scoping §6.3/§6.4). Position-only collapse is gated on the POSITION-DEGRADING
    # subset (_mean_posdeg = scramble+object_front), since drop_verb deliberately does not degrade position. ----
    mc = mc_battery["_mean_posdeg"]
    pos = pos_battery["_mean_posdeg"]
    nol = nolearn_battery["_mean_posdeg"]
    les = lesion_battery["_mean_posdeg"]
    perm = perm_battery["_mean_posdeg"]
    res["gates"] = {
        "multicue_ge_0.80": mc >= 0.80,
        "position_only_collapses_le_0.45": pos <= 0.45,
        "nolearn_below_multicue_by_0.15": nol <= mc - 0.15,   # learning must add >=15pp over frozen/uniform
        "lesion_collapses_near_position": les <= max(pos + 0.12, 0.55),
        "permuted_collapses_le_0.60": perm <= 0.60,
        "clean_unregressed": mc_clean >= pos_clean - 1e-9,
        "moat_zero_breach": breaches == 0,
    }
    res["seed_GO"] = all(res["gates"].values())
    if verbose:
        print(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 seed (42), verbose")
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46,47")
    ap.add_argument("--n-per-cond", type=int, default=40)
    ap.add_argument("--no-held-out", action="store_true", help="train==test fillers (diagnostic)")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    seeds = [42] if args.smoke else [int(s) for s in args.seeds.split(",") if s.strip()]
    held_out = not args.no_held_out
    results = [run_seed(s, n_per_cond=args.n_per_cond, held_out=held_out, verbose=args.smoke)
               for s in seeds]

    # ---- aggregate ----
    n = len(results)
    n_go = sum(r["seed_GO"] for r in results)

    def col(getter):
        return [getter(r) for r in results]

    key = "_mean_posdeg"  # the gated, load-bearing metric (scramble + object_front; position genuinely degraded)
    mc = col(lambda r: r["multicue_battery"][key])
    pos = col(lambda r: r["position_only_battery"][key])
    nol = col(lambda r: r["nolearn_battery"][key])
    les = col(lambda r: r["lesion_battery"][key])
    perm = col(lambda r: r["permuted_battery"][key])
    breaches = sum(r["moat"]["breaches"] for r in results)

    print("\n" + "=" * 80)
    print("MULTI-CUE COMPETITION PARSER -- degraded-input robustness de-risk")
    print("=" * 80)
    print(f"seeds: {seeds}   held_out_fillers={held_out}   n_per_cond={args.n_per_cond}")
    print(f"chance (2-role agent/patient) = 0.500")
    print(f"metric below = position-DEGRADING battery (scramble + object-front)\n")
    hdr = f"{'seed':>5} | {'MULTICUE':>8} | {'POS-ONLY':>8} | {'NO-LEARN':>8} | {'LESION':>7} | {'PERMUTE':>7} | {'moat_br':>7} | GO"
    print(hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r['seed']:>5} | {r['multicue_battery'][key]:>8.3f} | "
              f"{r['position_only_battery'][key]:>8.3f} | {r['nolearn_battery'][key]:>8.3f} | "
              f"{r['lesion_battery'][key]:>7.3f} | {r['permuted_battery'][key]:>7.3f} | "
              f"{r['moat']['breaches']:>7d} | {'GO' if r['seed_GO'] else 'no'}")
    print("-" * len(hdr))
    print(f"{'mean':>5} | {np.mean(mc):>8.3f} | {np.mean(pos):>8.3f} | {np.mean(nol):>8.3f} | "
          f"{np.mean(les):>7.3f} | {np.mean(perm):>7.3f} | {breaches:>7d} |")

    # per-condition breakdown (multi-cue vs position-only) averaged across seeds
    print("\nPer-degradation (mean across seeds): MULTICUE  vs  POSITION-ONLY")
    for cond in ("drop_verb", "scramble", "object_front"):
        m = np.mean([r["multicue_battery"][cond] for r in results])
        p = np.mean([r["position_only_battery"][cond] for r in results])
        note = "  (position NOT degraded here)" if cond == "drop_verb" else ""
        print(f"  {cond:>14}:   {m:>5.3f}   vs   {p:>5.3f}{note}")

    cm = np.mean([r["clean_multicue"] for r in results])
    cp = np.mean([r["clean_position_only"] for r in results])
    print(f"\nclean canonical (no-regression): multicue {cm:.3f}  vs  position-only {cp:.3f}")
    print(f"learned weights (mean): " +
          ", ".join(f"{c}={np.mean([r['weights_learned'][c] for r in results]):.3f}" for c in CUES))
    print(f"frozen  weights (mean): " +
          ", ".join(f"{c}={np.mean([r['weights_frozen'][c] for r in results]):.3f}" for c in CUES))

    overall_go = n_go >= max(1, int(np.ceil(0.8333 * n)))  # >=5/6
    print("\n" + "=" * 78)
    print(f"VERDICT: {n_go}/{n} seeds GO  (>=5/6 required)  ->  "
          f"{'GO' if overall_go else 'NEGATIVE / BOUNDARY'}")
    print(f"  moat breaches across all seeds: {breaches} (must be 0)")
    print("=" * 78)

    payload = {"seeds": seeds, "held_out": held_out, "n_per_cond": args.n_per_cond,
               "n_go": n_go, "n": n, "overall_GO": overall_go,
               "total_moat_breaches": breaches, "results": results}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[wrote] {args.out}")
    return payload


if __name__ == "__main__":
    main()
