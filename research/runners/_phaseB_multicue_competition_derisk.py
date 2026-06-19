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
# Cue evidence: each cue emits a SIGNED vote in [-1,+1] toward (agent:+ / patient:-) for a noun, plus a
# RELIABILITY in [0,1] (how confident this cue is for this noun in this sentence). The role logit per noun is
# sum_cue  w_cue * reliability_cue * vote_cue   (cue-validity-weighted ADDITIVE accumulation, catalog G.18).
# The learned w_cue ARE the cue validities (delta-rule learned on clean canonical sentences).
# ---------------------------------------------------------------------------

CUES = ("position", "animacy", "verbfit")


def _position_cue(noun_index, n_nouns):
    """Serial-position cue. 1st noun -> agent (+1), last noun -> patient (-1). Linear ramp between.
    Reliability is HIGH for canonical order and is what DEGRADES under scramble/drop/object-fronting
    (because the *index* the parser sees no longer reflects the true role)."""
    if n_nouns <= 1:
        return 0.0, 1.0
    # +1 at first slot, -1 at last slot
    frac = noun_index / (n_nouns - 1)          # 0..1
    vote = 1.0 - 2.0 * frac                     # +1 -> -1
    return float(vote), 1.0


def _animacy_cue(noun):
    """Animacy cue: animate -> agent-biased (+), inanimate -> patient-biased (-). Near-binary."""
    a = ANIMACY.get(noun)
    if a == "animate":
        return +1.0, 1.0
    if a == "inanimate":
        return -1.0, 1.0
    return 0.0, 0.0   # unknown -> no vote, zero reliability


def _verbfit_cue(noun, verb):
    """Verb-selectional-fit cue: does this noun fit the verb's AGENT slot better than its PATIENT slot?
    Compares the noun's animacy against what each slot selects for. Vote toward whichever slot it fits;
    if both slots select the same animacy (symmetric verb, e.g. chase/watch with two animates) -> 0 vote,
    LOW reliability (the cue is genuinely uninformative here -- the honest source of moat abstentions)."""
    sel = VERB_SELECTS.get(verb)
    a = ANIMACY.get(noun)
    if sel is None or a is None:
        return 0.0, 0.0
    fits_agent = (sel["agent"] == a)
    fits_patient = (sel["patient"] == a)
    if fits_agent and not fits_patient:
        return +1.0, 1.0
    if fits_patient and not fits_agent:
        return -1.0, 1.0
    return 0.0, 0.0   # fits both (or neither) -> uninformative


def cue_evidence(noun, noun_index, n_nouns, verb,
                 permute_map=None, lesion_semantic=False):
    """Return {cue: (vote, reliability)} for one noun. `permute_map` (a dict remapping nouns to scrambled
    feature-bearers) implements the PERMUTED-CUE control on the SEMANTIC cues only (position is structural).
    `lesion_semantic` zeroes the animacy+verbfit reliability (the CUE-LESION control)."""
    ev = {}
    ev["position"] = _position_cue(noun_index, n_nouns)
    if lesion_semantic:
        ev["animacy"] = (0.0, 0.0)
        ev["verbfit"] = (0.0, 0.0)
        return ev
    sem_noun = permute_map.get(noun, noun) if permute_map else noun
    ev["animacy"] = _animacy_cue(sem_noun)
    ev["verbfit"] = _verbfit_cue(sem_noun, verb)
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

    def assign_roles(self, nouns, evs, abstain_margin=0.0):
        """Assign agent/patient over the sentence's nouns by competition. Returns (assignment, decisive).
        `assignment` maps noun_index -> role. With the 1-agent/1-patient transitive constraint: the noun with
        the highest agent-logit wins AGENT, the lowest wins PATIENT. `decisive` is False (=> abstain) when the
        agent-logit separation between the top-2 nouns is below `abstain_margin` (no clear winner -> moat)."""
        logits = np.array([self.noun_agent_logit(ev) for ev in evs], dtype=float)
        order = np.argsort(-logits)  # descending agent-logit
        sep = float(logits[order[0]] - logits[order[1]]) if len(order) >= 2 else abs(float(logits[0]))
        decisive = sep >= abstain_margin
        assignment = {}
        assignment[int(order[0])] = "agent"
        assignment[int(order[-1])] = "patient"
        # any middle nouns (n>2) get the role of their nearer pole -- not exercised in this 2-noun battery.
        for i in order[1:-1]:
            assignment[int(i)] = "agent" if logits[i] >= 0 else "patient"
        return assignment, decisive

    # --- cue-validity LEARNING (delta rule). The cue weights track how reliably each cue's signed vote agrees
    # with the GOLD role on CLEAN canonical sentences. This IS the learned model; freezing it = the NO-LEARNING
    # control. Biology: Hebbian co-firing of a cue's vote with the correct role assembly (the spiking build). ---
    def learn(self, train_examples, lr=0.10, epochs=400, seed=0, freeze=False):
        if freeze:
            return  # NO-LEARNING control: leave weights at their uniform init (0.5 each)
        rng = np.random.default_rng(seed)
        idx = list(range(len(train_examples)))
        for _ in range(epochs):
            rng.shuffle(idx)
            for i in idx:
                nouns, evs, gold = train_examples[i]  # gold: noun_index -> "agent"/"patient"
                # target per noun: +1 if agent, -1 if patient. Predicted = tanh(agent_logit) in [-1,1].
                for ni, ev in enumerate(evs):
                    target = +1.0 if gold[ni] == "agent" else -1.0
                    logit = self.noun_agent_logit(ev)
                    pred = np.tanh(logit)
                    err = target - pred
                    dpred = (1.0 - pred * pred)  # d tanh
                    for c, (vote, rel) in ev.items():
                        self.weights[c] += lr * err * dpred * rel * vote
                # keep weights non-negative (a cue's VALIDITY is a non-negative reliability; the SIGN lives in
                # the vote). Clamp small floor->0 so a useless cue can be driven fully out (lesion-equivalent).
        for c in self.weights:
            self.weights[c] = max(0.0, self.weights[c])


# ---------------------------------------------------------------------------
# Sentence generation: clean canonical SVO (agent verb patient) + degraded variants.
# A sentence = (nouns_in_surface_order, verb, gold_roles_by_surface_index).
# ---------------------------------------------------------------------------

def _make_canonical(agent, verb, patient):
    """Canonical 'AGENT VERB PATIENT' (active SVO). Surface noun order = [agent, patient]."""
    nouns = [agent, patient]
    gold = {0: "agent", 1: "patient"}
    return nouns, verb, gold, "canonical"


def _degrade_drop(agent, verb, patient, rng):
    """DROP-A-WORD: drop the verb (a non-noun) -> 'AGENT PATIENT'. Surface order preserved BUT the position
    cue's reliability is unchanged (1st still agent) -- so to genuinely stress position we ALSO drop with a
    shifted frame in half the cases by dropping the FIRST noun's status marker... simpler + honest: dropping
    the verb removes the verb-fit cue's verb, weakening THAT cue, while position stays valid. That does NOT
    degrade position. To degrade POSITION we must change ORDER. So 'drop' here drops the verb AND we test it
    primarily for: does the parser still work when the verb (hence verb-fit) is gone? Position+animacy carry.
    Gold roles unchanged; surface order [agent, patient]."""
    nouns = [agent, patient]
    gold = {0: "agent", 1: "patient"}
    return nouns, None, gold, "drop_verb"


def _degrade_scramble(agent, verb, patient, rng):
    """SCRAMBLE-ORDER: same words, randomized NOUN order. Position cue becomes MISLEADING (the 1st surface
    noun is the patient in ~half the cases). Animacy + verb-fit must override. Gold roles follow the noun
    identity, not the surface slot."""
    nouns = [agent, patient]
    perm = rng.permutation(2)
    s_nouns = [nouns[p] for p in perm]
    # gold by surface index: surface slot j holds original noun perm[j]; its role is agent if perm[j]==0 else patient
    gold = {j: ("agent" if perm[j] == 0 else "patient") for j in range(2)}
    return s_nouns, verb, gold, "scramble"


def _degrade_object_front(agent, verb, patient, rng):
    """OBJECT-FRONTED (OSV-style): 'PATIENT AGENT VERB' surface order -> the position-only table (1st->agent)
    mis-maps the fronted patient as agent (the NEMO object-initial weakness). Surface noun order=[patient, agent]."""
    nouns = [patient, agent]
    gold = {0: "patient", 1: "agent"}
    return nouns, verb, gold, "object_front"


def build_dataset(rng, animate_pool, inanim_pool, verb_pool, n_per_cond=40):
    """Build clean canonical TRAIN sentences and a DEGRADED battery (drop/scramble/object-front).
    Every sentence: an ANIMATE agent + an agentive verb + a patient (inanimate for asymmetric verbs;
    we use the eat/push/carry/bite/kick/grab family so verb-fit is informative; chase/watch are symmetric
    and feed the MOAT-ambiguity set)."""
    # asymmetric verbs (agent animate, patient inanimate) -> verb-fit is informative
    asym_verbs = [v for v in verb_pool if VERB_SELECTS[v]["patient"] == "inanimate"]
    sym_verbs = [v for v in verb_pool if VERB_SELECTS[v]["patient"] == "animate"]

    def rand_sentence(verbs, pat_pool):
        a = animate_pool[rng.integers(len(animate_pool))]
        v = verbs[rng.integers(len(verbs))]
        p = pat_pool[rng.integers(len(pat_pool))]
        while p == a:
            p = pat_pool[rng.integers(len(pat_pool))]
        return a, v, p

    clean_train = []
    for _ in range(n_per_cond * 3):  # ample clean canonical examples for cue-validity learning
        a, v, p = rand_sentence(asym_verbs, inanim_pool)
        clean_train.append(_make_canonical(a, v, p))

    battery = {"drop_verb": [], "scramble": [], "object_front": []}
    for _ in range(n_per_cond):
        a, v, p = rand_sentence(asym_verbs, inanim_pool)
        battery["drop_verb"].append(_degrade_drop(a, v, p, rng))
        a, v, p = rand_sentence(asym_verbs, inanim_pool)
        battery["scramble"].append(_degrade_scramble(a, v, p, rng))
        a, v, p = rand_sentence(asym_verbs, inanim_pool)
        battery["object_front"].append(_degrade_object_front(a, v, p, rng))

    # clean canonical TEST (held-out fillers) for the no-regression check
    clean_test = []
    for _ in range(n_per_cond):
        a, v, p = rand_sentence(asym_verbs, inanim_pool)
        clean_test.append(_make_canonical(a, v, p))

    # MOAT-ambiguity set: two ANIMATE nouns + a SYMMETRIC verb (chase/watch), SCRAMBLED order. position is
    # misleading, animacy ties (both animate), verb-fit is uninformative (symmetric) -> NO decisive cue ->
    # the parser must ABSTAIN. Gold here is genuinely undetermined by the cues; a commit is a moat breach.
    moat_set = []
    if sym_verbs:
        for _ in range(n_per_cond):
            a = animate_pool[rng.integers(len(animate_pool))]
            b = animate_pool[rng.integers(len(animate_pool))]
            while b == a:
                b = animate_pool[rng.integers(len(animate_pool))]
            v = sym_verbs[rng.integers(len(sym_verbs))]
            nouns, perm_gold = _scrambled_two(a, b, rng)
            moat_set.append((nouns, v, perm_gold, "moat_ambiguous"))
    return clean_train, clean_test, battery, moat_set


def _scrambled_two(a, b, rng):
    nouns = [a, b]
    perm = rng.permutation(2)
    s = [nouns[p] for p in perm]
    gold = {j: ("agent" if perm[j] == 0 else "patient") for j in range(2)}
    return s, gold


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _examples_to_evidence(sentences, permute_map=None, lesion_semantic=False, position_only=False):
    """Convert (nouns, verb, gold, tag) sentences into (nouns, evs, gold) with per-noun cue evidence."""
    out = []
    for nouns, verb, gold, _tag in sentences:
        n = len(nouns)
        evs = []
        for ni, noun in enumerate(nouns):
            ev = cue_evidence(noun, ni, n, verb, permute_map=permute_map, lesion_semantic=lesion_semantic)
            if position_only:
                ev = {"position": ev["position"], "animacy": (0.0, 0.0), "verbfit": (0.0, 0.0)}
            out.append(None)
            evs.append(ev)
        out[-1] = (nouns, evs, gold)
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
    return accs


def _moat_breaches(parser, moat_set, abstain_margin):
    """On the genuinely-ambiguous set: the parser should ABSTAIN (decisive=False). A decisive COMMIT is a
    moat breach. Returns (n_breaches, n_total, abstain_rate)."""
    data = _examples_to_evidence(moat_set)
    breaches = 0
    for nouns, evs, _gold in data:
        _assignment, decisive = parser.assign_roles(nouns, evs, abstain_margin=abstain_margin)
        if decisive:
            breaches += 1
    n = len(data)
    return breaches, n, (n - breaches) / max(1, n)


def _calibrate_abstain_margin(parser, clean_train, battery):
    """Set the moat abstain margin from the DECISIVE separations on cue-INFORMATIVE sentences: pick a margin
    below the typical informative separation (so real decisions pass) but above the near-zero separation an
    all-ambiguous sentence produces. We use a low percentile of the informative-set separations."""
    seps = []
    informative = list(clean_train)
    for cond in ("scramble", "object_front", "drop_verb"):
        informative += battery[cond]
    for nouns, verb, gold, _tag in informative:
        n = len(nouns)
        evs = [cue_evidence(noun, ni, n, verb) for ni, noun in enumerate(nouns)]
        logits = np.array([parser.noun_agent_logit(ev) for ev in evs])
        if len(logits) >= 2:
            order = np.argsort(-logits)
            seps.append(float(logits[order[0]] - logits[order[1]]))
    if not seps:
        return 0.05
    return float(np.percentile(seps, 10) * 0.5)  # half the 10th-percentile informative separation


# ---------------------------------------------------------------------------
# One seed: train cue validities on clean canonical English, evaluate the full battery + all controls.
# ---------------------------------------------------------------------------

def run_seed(seed, n_per_cond=40, held_out=True, verbose=False):
    rng = np.random.default_rng(seed)
    if held_out:
        # TRAIN on train-pool fillers+verbs; TEST (battery/moat) on HELD-OUT fillers+verbs (disjoint).
        train_an, train_in, train_vb = TRAIN_ANIMATE, TRAIN_INANIM, TRAIN_VERBS
        test_an, test_in, test_vb = HELD_ANIMATE, HELD_INANIM, HELD_VERBS
    else:
        train_an, train_in, train_vb = TRAIN_ANIMATE, TRAIN_INANIM, TRAIN_VERBS
        test_an, test_in, test_vb = TRAIN_ANIMATE, TRAIN_INANIM, TRAIN_VERBS

    clean_train, _ct0, _b0, _m0 = build_dataset(rng, train_an, train_in, train_vb, n_per_cond=n_per_cond)
    # build the EVAL battery on held-out fillers/verbs
    _ctr, clean_test, battery, moat_set = build_dataset(rng, test_an, test_in, test_vb, n_per_cond=n_per_cond)

    # training examples as evidence (clean canonical, FULL cues)
    train_ex = _examples_to_evidence(clean_train)

    # ---- LEARNED multi-cue parser ----
    learned = MultiCueParser()
    learned.learn(train_ex, seed=seed, freeze=False)

    # ---- NO-LEARNING control (frozen at uniform init) ----
    frozen = MultiCueParser()
    frozen.learn(train_ex, seed=seed, freeze=True)

    # ---- PERMUTED-CUE control: learn cue validities against SCRAMBLED semantic tags ----
    # Permute the animacy/verb-fit feature-bearer identity within the training fillers so the semantic cues
    # carry NO real role information. If the learned weights then still "work", the cues were a leak.
    perm_nouns = train_an + train_in
    perm_targets = list(perm_nouns)
    rng_p = np.random.default_rng(seed + 9000)
    rng_p.shuffle(perm_targets)
    permute_map = dict(zip(perm_nouns, perm_targets))
    train_ex_perm = _examples_to_evidence(clean_train, permute_map=permute_map)
    permuted = MultiCueParser()
    permuted.learn(train_ex_perm, seed=seed, freeze=False)

    # calibrate the moat margin from the LEARNED parser's informative separations
    abstain_margin = _calibrate_abstain_margin(learned, clean_train, battery)

    # ================= METRICS =================
    # 1) MULTI-CUE on the degraded battery (the GO metric)
    mc_battery = _battery_accuracy(learned, battery)
    # 2) POSITION-ONLY baseline on the SAME degraded battery (the LOAD-BEARING control -> must collapse)
    pos_battery = _battery_accuracy(learned, battery, position_only=True)
    # 3) NO-LEARNING control on the degraded battery (must collapse)
    nolearn_battery = _battery_accuracy(frozen, battery)
    # 4) CUE-LESION on the degraded battery (zero semantic cues -> must collapse to ~position-only)
    lesion_battery = _battery_accuracy(learned, battery, lesion_semantic=True)
    # 5) PERMUTED-CUE on the degraded battery (must collapse to chance)
    perm_battery = _battery_accuracy(permuted, battery, permute_map=permute_map)
    # 6) clean canonical no-regression: multi-cue vs position-only on CLEAN held-out test
    mc_clean = _role_accuracy(learned, clean_test)
    pos_clean = _role_accuracy(learned, clean_test, position_only=True)
    # 7) MOAT: abstain on the genuinely-ambiguous set
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

    # ---- per-seed GO gates (mirrors scoping §6.3/§6.4) ----
    mc = mc_battery["_mean"]
    pos = pos_battery["_mean"]
    nol = nolearn_battery["_mean"]
    les = lesion_battery["_mean"]
    perm = perm_battery["_mean"]
    res["gates"] = {
        "multicue_ge_0.80": mc >= 0.80,
        "position_only_collapses_le_0.45": pos <= 0.45,
        "nolearn_collapses_le_0.60": nol <= 0.60,        # frozen/uniform must not reach the multi-cue level
        "lesion_collapses_near_position": les <= max(pos + 0.10, 0.55),
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

    mc = col(lambda r: r["multicue_battery"]["_mean"])
    pos = col(lambda r: r["position_only_battery"]["_mean"])
    nol = col(lambda r: r["nolearn_battery"]["_mean"])
    les = col(lambda r: r["lesion_battery"]["_mean"])
    perm = col(lambda r: r["permuted_battery"]["_mean"])
    breaches = sum(r["moat"]["breaches"] for r in results)

    print("\n" + "=" * 78)
    print("MULTI-CUE COMPETITION PARSER -- degraded-input robustness de-risk")
    print("=" * 78)
    print(f"seeds: {seeds}   held_out_fillers={held_out}   n_per_cond={args.n_per_cond}")
    print(f"chance (2-role agent/patient) = 0.500\n")
    hdr = f"{'seed':>5} | {'MULTICUE':>8} | {'POS-ONLY':>8} | {'NO-LEARN':>8} | {'LESION':>7} | {'PERMUTE':>7} | {'moat_br':>7} | GO"
    print(hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r['seed']:>5} | {r['multicue_battery']['_mean']:>8.3f} | "
              f"{r['position_only_battery']['_mean']:>8.3f} | {r['nolearn_battery']['_mean']:>8.3f} | "
              f"{r['lesion_battery']['_mean']:>7.3f} | {r['permuted_battery']['_mean']:>7.3f} | "
              f"{r['moat']['breaches']:>7d} | {'GO' if r['seed_GO'] else 'no'}")
    print("-" * len(hdr))
    print(f"{'mean':>5} | {np.mean(mc):>8.3f} | {np.mean(pos):>8.3f} | {np.mean(nol):>8.3f} | "
          f"{np.mean(les):>7.3f} | {np.mean(perm):>7.3f} | {breaches:>7d} |")

    # per-condition breakdown (multi-cue vs position-only) averaged across seeds
    print("\nPer-degradation (mean across seeds): MULTICUE  vs  POSITION-ONLY")
    for cond in ("drop_verb", "scramble", "object_front"):
        m = np.mean([r["multicue_battery"][cond] for r in results])
        p = np.mean([r["position_only_battery"][cond] for r in results])
        print(f"  {cond:>14}:   {m:>5.3f}   vs   {p:>5.3f}")

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
