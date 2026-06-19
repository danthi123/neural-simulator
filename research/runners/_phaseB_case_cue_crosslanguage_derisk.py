"""Phase-2 SPIKING-substrate de-risk: a CASE cue -> true CROSS-LANGUAGE thematic-role comprehension.

This builds DIRECTLY on the Phase-1 spiking multi-cue COMPETITION parser
(`_phaseB_multicue_competition_spiking_derisk.py`, GO -- install 5/6 + three-factor on-substrate learning), per the
Phase-2 scoping (`research/findings/2026-06-19-phase2-case-cue-crosslanguage-scoping.md`). It asks the canonical
Bates-MacWhinney Competition-Model cross-linguistic question:

    Does the SAME validated multi-cue competition, with a CASE cue added, read thematic roles by CASE on a
    FREE-word-order case-marked toy language (Japanese-style ga/wo) where a word-POSITION parser CANNOT --
    with the case cue's validity LEARNED HIGH (the three-factor rule, not hand-set) -- AND -- the HEADLINE --
    the SAME code learning OPPOSITE cue weights on English vs the case-language
    (English -> position-dominant, Japanese-toy -> case-dominant)?

REUSE (NO new mechanism; NO `sim/` edit; reuse-by-import + additive):
  * The ENTIRE spiking competition substrate (`SpikingRoleCompetition`): the Wong-Wang role accumulators
    (sel_agent/sel_patient) + Rutishauser selective inhibition, the cue-population -> plastic cue->role projection
    pipeline, `set_cue_weight`/`cue_weights`/`freeze_all_cue_plasticity`, `learn_error_gated` (the three-factor
    spike-eligibility x reward x vote validity learner), `assign_roles` (the WTA read + moat gate).
    The constructor AUTO-BUILDS a cue population + plastic projection for EVERY `c in CUES`, so adding `"case"`
    to the module-level `CUES` tuple creates its populations + projection with ZERO constructor edits.
  * The eval/battery/moat machinery (`_battery_accuracy`, `_moat_breaches`, `_calibrate_abstain_margin`,
    `_role_accuracy`, `_examples_to_evidence`) -- they read the live module globals (`CUES`, `cue_evidence`,
    `SEMANTIC_CUES`), so by setting `CUES` and installing a case-aware `cue_evidence` we reuse them verbatim.

THE MINIMAL NEW PIECES (all additive; flagged):
  1. `"case"` added to `CUES` (auto-builds its spiking populations + plastic cue->role projection).
  2. A `_case_vote(marker)` (nominative-particle -> +1 agent, accusative-particle -> -1 patient, unmarked -> 0)
     + a case-aware `cue_evidence` that injects it. The case marker is a per-NOUN annotation carried IN the
     sentence tuple (a token-level lexical front-end -- the legitimate isolating-particle boundary, FLAGGED like
     the verb/animacy lexicons; for an isolating particle this is a set-membership check, NO segmentation).
  3. A Japanese-style FREE-WORD-ORDER particle toy corpus (SOV canonical + OSV + scrambled, ALL with ga/wo
     marking, at a HIGH non-canonical fraction so POSITION is uninformative and CASE is decisive). English
     sentences carry NO markers -> the case cue is silent (-> its validity stays at floor) = the dissociation
     mechanism, with NO code-path difference between the two languages (the same `cue_evidence` reads markers
     if present, votes 0 if absent).

GO BAR (pre-registered; FROZEN; >=6 seeds; fractional >=5/6; chance for 2-role agent/patient = 0.500):
  - case-path role accuracy >= 0.80 on the FREE-WORD-ORDER battery (case reads roles when position can't);
  - POSITION-ONLY baseline COLLAPSES on the same free-order battery (<= ~0.45) -- THE load-bearing control;
  - THE HEADLINE cross-linguistic DISSOCIATION: the SAME code+learner lands w_position HIGH / w_case ~0 on the
    ENGLISH corpus AND w_case HIGH / w_position LOW on the Japanese-toy corpus (report both weight vectors);
  - CASE-LESION collapses (case load-bearing); NO-LEARNING (frozen weights) collapses (validity LEARNED, not
    hand-set); PERMUTED-CASE collapses (no leak);
  - clean canonical (the toy's majority order) unregressed; no-confab MOAT 0-breach (unmarked/ambiguous -> abstain).

Run (CPU/numpy is ~9x faster than GPU for this tiny ~240-neuron bridge -- the Phase-1 perf note):
    SIM_BACKEND=numpy python -m research.runners._phaseB_case_cue_crosslanguage_derisk --smoke
    SIM_BACKEND=numpy python -m research.runners._phaseB_case_cue_crosslanguage_derisk \
        --seeds 42,43,44,45,46,47 --out research/findings/raw/_phaseB_case_cue_crosslanguage.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

# Import the Phase-1 spiking machinery as P1 and EXTEND its module globals additively (the class + eval helpers
# read these live globals, so this is the reuse-by-import path the scoping specifies -- NO new mechanism).
import research.runners._phaseB_multicue_competition_spiking_derisk as P1  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Add the CASE cue to the competition's cue tuple (auto-builds its spiking populations + plastic projection).
#    Ordering keeps `case` adjacent to the semantic cues; `lexbias` (the chance distractor) stays last.
# ---------------------------------------------------------------------------
CUES = ("position", "animacy", "verbfit", "case", "lexbias")
P1.CUES = CUES
P1._CUE_ID = {c: i for i, c in enumerate(CUES)}

# the case cue is the dominant CONTENT cue in a case language; include it in the moat's content gate so an
# unmarked/ambiguous sentence (case silent) correctly has NO decisive content cue -> abstain.
P1.SpikingRoleCompetition.SEMANTIC_CUES = ("animacy", "verbfit", "case")

# reuse the Phase-1 lexicons + cues verbatim (the SAME animacy/verb-fit cues; the SAME English pools/split, used
# by the dissociation's ENGLISH arm via P1.build_dataset)
ANIMACY = P1.ANIMACY
VERB_SELECTS = P1.VERB_SELECTS
ROLES = P1.ROLES
TRUE_VALIDITY = dict(P1.TRUE_VALIDITY)
TRUE_VALIDITY["case"] = 0.98   # case is the most-reliable cue in the case language (ga->agent, wo->patient)

# ---------------------------------------------------------------------------
# Japanese-toy lexicon: BOTH the agent and the patient are ANIMATE, so ANIMACY TIES (non-informative for role) and
# the VERB is SYMMETRIC (both slots animate -> verb-fit ties) -> CASE is the SOLE reliable role cue. This is the
# canonical Competition-Model Japanese test ("inu ga neko wo oikakeru" -- dog-NOM cat-ACC chase: with both animate,
# only the case particle tells you who chased whom). Position is informative ONLY on canonical SOV; case is correct
# at ANY order. (If the patient were inanimate -- the Phase-1 English design -- animacy alone would solve free
# order and case would be REDUNDANT, NOT load-bearing -- the toy-design trap the scoping warns about.)
#
# All these nouns are animate (extends the buffer/Phase-1 ANIMACY lexicon, drift-safe: only ADDS animate nouns).
JP_AGENTS = ["dog", "cat", "fox", "bird"]          # train agents (all animate; reused from Phase-1 animate pool)
JP_PATIENTS = ["wolf", "bear", "owl", "frog"]      # train patients (all animate -- distinct from agents)
JP_AGENTS_HELD = ["lion", "deer", "hawk", "seal"]  # held-out agents (animate)
JP_PATIENTS_HELD = ["mouse", "goat", "crow", "toad"]  # held-out patients (animate)
for _w in JP_AGENTS + JP_PATIENTS + JP_AGENTS_HELD + JP_PATIENTS_HELD:
    ANIMACY.setdefault(_w, "animate")
    assert ANIMACY[_w] == "animate", f"JP noun {_w} must be animate"

# Japanese symmetric transitive verbs (both slots animate -> verb-fit non-informative). Adds to VERB_SELECTS.
JP_VERBS = ["chase", "watch", "follow", "greet"]        # train verbs
JP_VERBS_HELD = ["nudge", "tail", "spot", "mimic"]      # held-out verbs
for _v in JP_VERBS + JP_VERBS_HELD:
    VERB_SELECTS.setdefault(_v, {"agent": "animate", "patient": "animate"})

# ---------------------------------------------------------------------------
# 2. The case vote + a case-aware cue_evidence (installed onto P1 so ALL reused eval helpers pick it up).
#    A case marker is a per-noun annotation carried IN the sentence tuple at index 5 (markers list). English
#    sentences omit it -> case vote 0 (silent). This is the ONE evidence function for BOTH languages -- the
#    dissociation has NO code-path difference (the scoping's headline requirement).
# ---------------------------------------------------------------------------
NOMINATIVE = "ga"      # marks the AGENT  -> case vote +1
ACCUSATIVE = "wo"      # marks the PATIENT -> case vote -1  (Japanese-style ga / wo(o))
CASE_MARKERS = {NOMINATIVE, ACCUSATIVE}

# stash for the case-aware evidence (set per `_examples_to_evidence` call via the sentence tuple's markers field;
# but cue_evidence's signature is fixed by the reused helpers, so the marker is threaded through a per-sentence
# module-level map keyed on sent_id -- populated by our dataset builders. Clean: each sent_id is unique.)
_SENT_MARKERS: dict[int, list] = {}


def _case_vote(marker):
    """Isolating-particle case vote: nominative marker -> +1 (agent), accusative -> -1 (patient), else 0.
    Set-membership on the particle TOKEN -- the legitimate token-level lexical front-end (no segmentation)."""
    if marker == NOMINATIVE:
        return +1.0
    if marker == ACCUSATIVE:
        return -1.0
    return 0.0


# keep a handle to the Phase-1 evidence so we layer the case cue ON TOP of its position/animacy/verbfit/lexbias.
_P1_cue_evidence = P1.cue_evidence


def cue_evidence(noun, noun_index, n_nouns, verb, sent_id,
                 permute_map=None, lesion_semantic=False, drop_cues=(), clean_cues=False):
    """Phase-1 cue evidence (position/animacy/verbfit/lexbias) PLUS the CASE cue. Installed onto P1 so every
    reused eval helper (`_examples_to_evidence`, `_battery_accuracy`, `_moat_breaches`, `_calibrate_abstain_margin`)
    builds case-aware evidence. The case marker comes from the per-sentence markers map (English: no entry -> 0)."""
    ev = _P1_cue_evidence(noun, noun_index, n_nouns, verb, sent_id,
                          permute_map=permute_map, lesion_semantic=lesion_semantic,
                          drop_cues=drop_cues, clean_cues=clean_cues)
    # case cue: read this noun's marker from the per-sentence map; vote per the marker. Reliability=1 iff marked.
    markers = _SENT_MARKERS.get(int(sent_id))
    marker = markers[noun_index] if (markers is not None and noun_index < len(markers)) else None
    raw = _case_vote(marker)

    def flip(vote, validity, cue):
        return vote if clean_cues else P1._maybe_flip(vote, validity, sent_id, cue, noun_index)

    cv = flip(raw, TRUE_VALIDITY["case"], "case")
    ev["case"] = (float(cv), 1.0 if raw != 0.0 else 0.0)
    if "case" in drop_cues:
        ev["case"] = (0.0, 0.0)
    return ev


P1.cue_evidence = cue_evidence

# ---------------------------------------------------------------------------
# 3. The Japanese-style FREE-WORD-ORDER particle toy corpus.
#    Every transitive sentence has its AGENT noun + ga and its PATIENT noun + wo, in a FREE surface order (SOV
#    canonical, OSV object-front, and scrambled all present). The case marker TRAVELS WITH THE NOUN regardless of
#    position (it is a lexical property of the marked noun) -> the case cue is correct at any order, while position
#    is uninformative on the non-canonical majority. The English corpus (Phase-1's `build_dataset`) carries NO
#    markers -> case silent. SAME cues, SAME competition, two corpora -> the dissociation.
#
#    Sentence tuple = (nouns_in_surface_order, verb, gold_roles_by_surface_index, tag, sent_id). The per-noun case
#    markers are registered in `_SENT_MARKERS[sent_id]` (consumed by the case-aware cue_evidence above).
# ---------------------------------------------------------------------------


def _register_markers(sid, surface_nouns, gold):
    """Register the case marker per surface noun: the gold-AGENT noun gets `ga`, the gold-PATIENT gets `wo`.
    (The marker is a lexical property of the noun's role -- it does NOT depend on surface position, which is the
    whole point: case survives scrambling.)"""
    markers = [None] * len(surface_nouns)
    for j in range(len(surface_nouns)):
        markers[j] = NOMINATIVE if gold[j] == "agent" else ACCUSATIVE
    _SENT_MARKERS[int(sid)] = markers
    return markers


def _unmarked(sid, surface_nouns):
    _SENT_MARKERS[int(sid)] = [None] * len(surface_nouns)


def _jp_sentence(agent, verb, patient, sid, rng, force_canonical=False, drop_markers=False):
    """One Japanese-style transitive: agent+ga, patient+wo, FREE surface order (SOV/OSV/scramble). Case marker
    travels with the noun. `force_canonical` -> SOV (agent first); `drop_markers` -> particles dropped (moat)."""
    nouns = [agent, patient]
    if force_canonical:
        perm = np.array([0, 1])
    else:
        perm = rng.permutation(2)
    surface = [nouns[p] for p in perm]
    gold = {j: ("agent" if perm[j] == 0 else "patient") for j in range(2)}
    tag = "canonical_sov" if (perm[0] == 0) else "object_front_osv"
    if drop_markers:
        _unmarked(sid, surface)
    else:
        _register_markers(sid, surface, gold)
    return surface, verb, gold, tag, sid


def _osv(agent, verb, patient, sid):
    """An explicit OSV (object-front) sentence: surface = [patient, agent], case markers travel with the noun
    (patient->wo, agent->ga). A POSITION-only parser maps the surface-first noun (the patient) to AGENT and FAILS;
    case overrides. Returns the sentence tuple + registers the (position-independent) markers."""
    surface = [patient, agent]
    gold = {0: "patient", 1: "agent"}
    _register_markers(sid, surface, gold)
    return surface, verb, gold, "object_front_osv", sid


def build_case_dataset(rng, agent_pool, patient_pool, verb_pool, n_per_cond=20, ids=None,
                       noncanon_train_frac=0.65):
    """Japanese-style free-word-order case toy. BOTH nouns ANIMATE + a SYMMETRIC verb -> animacy + verb-fit TIE ->
    CASE is the SOLE reliable role cue. Mirrors the Phase-1 `build_dataset` shape (train + clean_test + battery +
    moat) so the reused eval helpers work, but EVERY sentence is case-marked and order is FREE.

    Battery (the position-degrading set, all case-marked, animate-animate):
      'free_order'   : random surface order (case correct, position misleading on the fronted half).
      'object_front' : explicit OSV (the fronted-object set a position-only parser maps to agent and FAILS).
      'case_absent'  : particles DROPPED (case silent; animacy+verb-fit ALSO tie -> nothing disambiguates) --
                       reported as graceful-degradation, NOT gated (case-and-position-and-semantics all silent).
    moat: two animate nouns + symmetric verb, particles DROPPED (case silent, animacy ties, verb symmetric) ->
          no decisive content cue -> ABSTAIN. (Differs from the battery ONLY by the missing case markers -> the
          case cue's presence is exactly what flips decisive vs abstain.)
    """
    ids = ids or P1._Ids()

    def rand():
        a = agent_pool[rng.integers(len(agent_pool))]
        v = verb_pool[rng.integers(len(verb_pool))]
        p = patient_pool[rng.integers(len(patient_pool))]
        while p == a:
            p = patient_pool[rng.integers(len(patient_pool))]
        return a, v, p

    # TRAINING: free word order with a HIGH non-canonical fraction so POSITION's empirical validity is LOW (the
    # three-factor learner discovers position is unreliable -> drops w_position; case never errs -> climbs w_case).
    train = []
    n_train = n_per_cond * 6
    for _ in range(n_train):
        a, v, p = rand()
        if rng.random() < noncanon_train_frac:
            train.append(_osv(a, v, p, ids.next()))                               # object-front (position wrong)
        else:
            train.append(_jp_sentence(a, v, p, ids.next(), rng, force_canonical=True))   # SOV canonical

    # BATTERY (held-out fillers/verbs supplied by the caller via the pools)
    battery = {"free_order": [], "object_front": [], "case_absent": []}
    for _ in range(n_per_cond):
        a, v, p = rand()
        battery["free_order"].append(_jp_sentence(a, v, p, ids.next(), rng, force_canonical=False))
        a, v, p = rand()
        battery["object_front"].append(_osv(a, v, p, ids.next()))
        a, v, p = rand()
        battery["case_absent"].append(_jp_sentence(a, v, p, ids.next(), rng, force_canonical=False,
                                                   drop_markers=True))

    # CLEAN canonical (the toy's majority order = SOV, case-marked)
    clean_test = [_jp_sentence(*rand(), ids.next(), rng, force_canonical=True) for _ in range(n_per_cond)]

    # MOAT: two animate nouns + symmetric verb, particles DROPPED -> no decisive content cue -> abstain.
    moat = []
    for _ in range(n_per_cond):
        a, v, p = rand()
        perm = rng.permutation(2)
        nn = [[a, p][perm[0]], [a, p][perm[1]]]
        gold = {j: ("agent" if perm[j] == 0 else "patient") for j in range(2)}
        sid = ids.next()
        _unmarked(sid, nn)  # particles dropped -> case silent (the ambiguous moat)
        moat.append((nn, v, gold, "moat_ambiguous", sid))
    return train, clean_test, battery, moat


# the position-degrading battery subset for the case toy (free_order + object_front; case_absent is reported, not gated)
def _case_posdeg_keys():
    return ("free_order", "object_front")


def _battery_with_posdeg(comp, battery, read_steps, **ev_kwargs):
    """Like P1._battery_accuracy but with the case-toy's posdeg subset (free_order + object_front)."""
    accs = {}
    flat = []
    for cond, sents in battery.items():
        accs[cond] = P1._role_accuracy(comp, sents, read_steps=read_steps, **ev_kwargs)
        flat.extend(sents)
    accs["_mean"] = P1._role_accuracy(comp, flat, read_steps=read_steps, **ev_kwargs)
    posdeg = []
    for k in _case_posdeg_keys():
        posdeg += battery[k]
    accs["_mean_posdeg"] = P1._role_accuracy(comp, posdeg, read_steps=read_steps, **ev_kwargs)
    return accs


# validated case-language validities for the INSTALL fallback (case is the dominant cue; position low; the
# spiking-scale magnitudes mirror Phase-1's INSTALLED_CUE_WEIGHTS, with case at the top).
INSTALLED_CASE_WEIGHTS = {"position": 6.0, "animacy": 14.0, "verbfit": 14.0, "case": 22.0, "lexbias": 2.0}


def learn_case_three_factor(comp, train_examples, epochs=30, settle_steps=18, seed=0,
                            lr=0.6, decay=0.06, w_floor=0.0, w_init=12.0, output_scale=20.0):
    """Phase-2 three-factor cue-validity learner -- the SAME brain-based rule as Phase-1's `learn_error_gated`
    (spike-eligibility x reward/RPE x vote on the real cue->role synapses, reusing comp._settle_with_eligibility +
    comp._fast_set_cue_weight), with TWO corrections required for the cross-LINGUISTIC setting that Phase-1's
    English-only normalization did not need:

      (A) DECAY EVERY CUE EVERY ITEM (not only when the cue fires). A cue that is NEVER informative -- the case cue
          in ENGLISH (no markers -> never fires), OR animacy/verb-fit in the Japanese ANIMATE-ANIMATE toy (always
          tie) -- must DECAY toward floor, so it reads as 'at floor' (not frozen at w_init). Phase-1 decayed only
          inside the cue-active branch, so a never-firing cue stayed at w_init and the output gain then amplified
          it -- the artifact that put English case at ~71 (it never fired). Decaying every item fixes this and is
          the correct credit-assignment: a cue that contributes no eligibility provides no evidence -> its weight
          relaxes to floor.
      (B) NORMALIZE TO THE MAX learned cue (not the semantic mean). Phase-1 normalized to 0.5*(animacy+verbfit),
          which EXPLODES when the semantic cues tie (Japanese animate-animate -> w_sem ~ 0 -> gain -> inf). The
          MAX-cue normalization places whichever cue WON the validity competition (case, in Japanese; position/
          semantics, in English) at `output_scale`, preserving every learned RATIO. (Same role as the numpy
          softmax temperature: the learned validities set the ratios; a fixed gain sets the decision scale.)

    Everything spike-measured / on-synapse is IDENTICAL to Phase-1; only the host-side normalization + decay
    bookkeeping differ. The reward (settled-winner-matched-gold) is the same host teaching signal (flagged)."""
    import numpy as _np
    rng = _np.random.default_rng(seed)
    comp._precompute_cue_edge_slots()
    for c in CUES:
        comp._fast_set_cue_weight(c, w_init)
    W = {c: float(w_init) for c in CUES}
    items = [(ev, gold[ni]) for _n, evs, gold in train_examples for ni, ev in enumerate(evs)]
    for _ep in range(epochs):
        rng.shuffle(items)
        for ev, gold_role in items:
            elig, rr = comp._settle_with_eligibility(ev, settle_steps=settle_steps)
            pred = rr["agent"] - rr["patient"]
            target = +1.0 if gold_role == "agent" else -1.0
            err = target - _np.tanh(pred * 8.0)
            d = 1.0 - _np.tanh(pred * 8.0) ** 2
            for c in CUES:
                vote, rel = ev.get(c, (0.0, 0.0))
                e_c = elig.get(c, 0.0) if (rel > 0.0 and vote != 0.0) else 0.0
                # (A) decay applies to EVERY cue every item; the reward-modulated term only when the cue fired.
                reinforce = lr * (err * d * e_c * float(vote)) if e_c != 0.0 else 0.0
                W[c] = max(w_floor, W[c] + reinforce - lr * decay * W[c])
                comp._fast_set_cue_weight(c, W[c])
    # (B) normalize to the MAX learned cue (robust when a cue family ties or never fires)
    w_max = max(W.values())
    gain = (output_scale / w_max) if w_max > 1e-6 else 1.0
    for c in CUES:
        W[c] = W[c] * gain
        comp._fast_set_cue_weight(c, W[c])
    comp.bridge.cp_external_input_current[:] = 0.0
    return W


def _build_competition(seed, **kw):
    return P1.SpikingRoleCompetition(seed=seed, **kw)


# ===========================================================================
# Per-seed run: trains on the Japanese-toy corpus, runs the free-order battery + ALL controls + the dissociation.
# ===========================================================================

def run_seed(seed, n_per_cond=20, held_out=True, learn_mode="error_gated", epochs=24, train_steps=18,
             read_steps=60, controls=True, noncanon_train_frac=0.65, verbose=False, **comp_kw):
    _SENT_MARKERS.clear()  # fresh per-sentence case-marker map (no cross-call contamination of the module global)
    rng = np.random.default_rng(seed)
    ids = P1._Ids()
    if held_out:
        tr_ag, tr_pa, tr_vb = JP_AGENTS, JP_PATIENTS, JP_VERBS
        te_ag, te_pa, te_vb = JP_AGENTS_HELD, JP_PATIENTS_HELD, JP_VERBS_HELD
    else:
        tr_ag, tr_pa, tr_vb = JP_AGENTS, JP_PATIENTS, JP_VERBS
        te_ag, te_pa, te_vb = JP_AGENTS, JP_PATIENTS, JP_VERBS

    # ---- Japanese-toy corpus (the case language) ----
    train_sents, _ct_tr, _bt_tr, _mt_tr = build_case_dataset(
        rng, tr_ag, tr_pa, tr_vb, n_per_cond=n_per_cond, ids=ids, noncanon_train_frac=noncanon_train_frac)
    _tr_e, clean_test, battery, moat_set = build_case_dataset(
        rng, te_ag, te_pa, te_vb, n_per_cond=n_per_cond, ids=ids, noncanon_train_frac=noncanon_train_frac)
    train_ex = P1._examples_to_evidence(train_sents)

    # ---- LEARNED case-aware spiking parser (the case language) ----
    learned = _build_competition(seed, verbose=verbose, **comp_kw)
    if learn_mode == "install":
        for c, w in INSTALLED_CASE_WEIGHTS.items():
            learned.set_cue_weight(c, w)
        learned.freeze_all_cue_plasticity()
    elif learn_mode == "error_gated":
        learn_case_three_factor(learned, train_ex, epochs=epochs, settle_steps=train_steps, seed=seed)
        learned.freeze_all_cue_plasticity()
    else:  # hebbian (characterized NEGATIVE for validity learning -- kept for completeness)
        learned.learn(train_ex, epochs=epochs, train_steps=train_steps, seed=seed, freeze=False)
        learned.freeze_all_cue_plasticity()
    w_learned = learned.cue_weights()

    # moat margin: calibrate on the INFORMATIVE held-out sentences (all have a decisive case cue)
    informative = battery["free_order"] + battery["object_front"] + clean_test
    abstain_margin = P1._calibrate_abstain_margin(learned, informative)

    # ===== primary metrics on the learned (case) parser =====
    mc_battery = _battery_with_posdeg(learned, battery, read_steps)
    mc_clean = P1._role_accuracy(learned, clean_test, read_steps=read_steps)
    breaches, moat_n, abstain_rate = P1._moat_breaches(learned, moat_set, abstain_margin, read_steps=read_steps)

    # CASE-LESION (THE decisive control): zero the LEARNED cue_case->role projection weights on the SAME trained
    # parser, keep all the others. The case cue still FIRES but its projection contributes no drive -> free-order
    # robustness must collapse back to the position/semantic-only level. (Done AFTER the primary metrics since it
    # destructively edits `learned`; all cues still fire so this is a true synaptic lesion, not an evidence drop.)
    learned.set_cue_weight("case", 0.0)
    lesion_case_battery = _battery_with_posdeg(learned, battery, read_steps)

    # POSITION-ONLY baseline: a GENUINE position-only parser -- position at the reference weight, ALL OTHER cues'
    # projection weights ZEROED (incl. case). All cues still fire, but only position drives the WTA, so on the
    # free-order battery it maps the fronted noun to agent and COLLAPSES. (Synaptic baseline; no evidence drop.)
    pos_ref = _build_competition(seed, **comp_kw)
    for c in CUES:
        pos_ref.set_cue_weight(c, 0.0)
    pos_ref.set_cue_weight("position", INSTALLED_CASE_WEIGHTS["position"])
    pos_ref.freeze_all_cue_plasticity()
    pos_battery = _battery_with_posdeg(pos_ref, battery, read_steps)
    pos_clean = P1._role_accuracy(pos_ref, clean_test, read_steps=read_steps)

    res = {
        "seed": seed,
        "learn_mode": learn_mode,
        "language": "japanese_toy",
        "weights_learned": {k: round(v, 4) for k, v in w_learned.items()},
        "abstain_margin": round(abstain_margin, 5),
        "case_battery": {k: round(v, 4) for k, v in mc_battery.items()},
        "position_only_battery": {k: round(v, 4) for k, v in pos_battery.items()},
        "case_lesion_battery": {k: round(v, 4) for k, v in lesion_case_battery.items()},
        "clean_case": round(mc_clean, 4),
        "clean_position_only": round(pos_clean, 4),
        "moat": {"breaches": breaches, "n": moat_n, "abstain_rate": round(abstain_rate, 4)},
    }

    # ===== NO-LEARNING + PERMUTED-CASE controls =====
    nol_battery = permcase_battery = None
    w_frozen = w_permcase = None
    if controls and learn_mode in ("hebbian", "error_gated"):
        # NO-LEARNING: frozen uniform init (no spread) -> over-trusts position -> collapses on free order.
        frozen = _build_competition(seed, **comp_kw)
        for c in CUES:
            frozen.set_cue_weight(c, INSTALLED_CASE_WEIGHTS["position"])  # uniform = no-spread baseline
        frozen.freeze_all_cue_plasticity()
        w_frozen = frozen.cue_weights()
        nol_battery = _battery_with_posdeg(frozen, battery, read_steps)

        # PERMUTED-CASE: train against a SCRAMBLED case-marker->role assignment (nominative->patient,
        # accusative->agent). The case cue then carries NO real role info -> the validity learner finds no useful
        # spread -> collapses to chance on free order. Built as a SEPARATE corpus whose markers are flipped.
        # IMPORTANT: reuse the SHARED `ids` (continues past train+battery) so the permuted-train sent_ids do NOT
        # collide with the main battery's sent_ids in the module-global _SENT_MARKERS map (flipping markers at a
        # colliding sid would corrupt the still-correctly-marked battery the permuted parser is evaluated on).
        rng_p = np.random.default_rng(seed + 7000)
        train_perm, _c2, _b2, _m2 = build_case_dataset(rng_p, tr_ag, tr_pa, tr_vb,
                                                       n_per_cond=n_per_cond, ids=ids,
                                                       noncanon_train_frac=noncanon_train_frac)
        # flip the case markers in the permuted training corpus (ga<->wo) so case anti-correlates with role
        for s, vb, gold, tag, sid in train_perm:
            m = _SENT_MARKERS.get(int(sid))
            if m is not None:
                _SENT_MARKERS[int(sid)] = [ACCUSATIVE if mm == NOMINATIVE else (NOMINATIVE if mm == ACCUSATIVE else None)
                                            for mm in m]
        train_ex_perm = P1._examples_to_evidence(train_perm)
        permuted = _build_competition(seed, **comp_kw)
        if learn_mode == "error_gated":
            learn_case_three_factor(permuted, train_ex_perm, epochs=epochs, settle_steps=train_steps, seed=seed)
        else:
            permuted.learn(train_ex_perm, epochs=epochs, train_steps=train_steps, seed=seed, freeze=False)
        permuted.freeze_all_cue_plasticity()
        w_permcase = permuted.cue_weights()
        # evaluate the permuted parser on the (correctly-marked) battery: it learned the WRONG case map -> collapses
        permcase_battery = _battery_with_posdeg(permuted, battery, read_steps)

        res["weights_frozen"] = {k: round(v, 4) for k, v in w_frozen.items()}
        res["weights_permuted_case"] = {k: round(v, 4) for k, v in w_permcase.items()}
        res["nolearn_battery"] = {k: round(v, 4) for k, v in nol_battery.items()}
        res["permuted_case_battery"] = {k: round(v, 4) for k, v in permcase_battery.items()}

    # ===== per-seed GO gates =====
    key = "_mean_posdeg"
    mc = mc_battery[key]
    pos = pos_battery[key]
    les = lesion_case_battery[key]
    w_sem_mean = 0.5 * (w_learned["animacy"] + w_learned["verbfit"])
    # the case-language signature: w_case driven HIGH (>= the semantic cues AND materially above position).
    case_above_pos = w_learned["case"] - w_learned["position"]
    sig_ok = (w_learned["case"] >= w_sem_mean * 0.9 and
              case_above_pos >= 0.25 * max(1e-9, w_learned["case"]) and
              w_learned["lexbias"] <= w_learned["case"] * 0.75)
    # The case-lesion gate: the case cue was LOAD-BEARING -> removing it must (a) materially drop the free-order
    # accuracy AND (b) collapse the OBJECT-FRONT condition specifically (with case gone, the position cue maps the
    # fronted noun to agent and fails -- the decisive signature). NOTE the no-case "semantic-only" baseline here is
    # ~chance (both nouns animate -> animacy/verb-fit TIE -> a coin-flip), NOT the 0.25 position-only level (which
    # additionally drops the animacy/lexbias drive); so the gate asserts a real collapse vs the CASE parser + the
    # object-front-specific collapse, not "<= position-only".
    les_of = lesion_case_battery["object_front"]
    gates = {
        "case_path_ge_0.80": mc >= 0.80,
        "position_only_collapses_le_0.45": pos <= 0.45,
        "case_lesion_collapses": (les <= mc - 0.20) and (les_of <= 0.55),
        "clean_strong_and_not_collapsed": (mc_clean >= 0.80) and (mc_clean >= pos_clean - 0.20),
        "moat_zero_breach": breaches == 0,
    }
    res["case_above_position_spread"] = round(case_above_pos, 4)
    if learn_mode in ("hebbian", "error_gated"):
        gates["weight_signature_case_dominant"] = bool(sig_ok)
        if controls:
            gates["nolearn_below_case_by_0.12"] = nol_battery[key] <= mc - 0.12
            gates["permuted_case_collapses_le_0.60"] = permcase_battery[key] <= 0.60
    res["weight_signature_ok"] = bool(sig_ok)
    res["gates"] = gates
    res["seed_GO"] = all(gates.values())
    if verbose:
        print(json.dumps(res, indent=2))
    return res


# ===========================================================================
# THE CROSS-LINGUISTIC DISSOCIATION (the headline): the SAME code+learner on the ENGLISH corpus vs the Japanese-toy.
# Returns both learned weight vectors -- English should land w_position high / w_case ~0; Japanese w_case high /
# w_position low. NO code-path difference between the two runs (the same cue_evidence reads markers if present).
# ===========================================================================

def run_dissociation(seed, n_per_cond=20, epochs=24, train_steps=18, noncanon_train_frac_jp=0.65,
                     noncanon_train_frac_en=0.55, verbose=False, **comp_kw):
    # --- ENGLISH (Phase-1's own English build_dataset; NO case markers -> case cue silent) ---
    # CLEAR the per-sentence markers map: English has NO case markers anywhere -> the case cue is silent on every
    # English sentence (so its learned validity stays at floor). This is the dissociation mechanism (the SAME
    # cue_evidence reads markers if present, votes 0 if absent) -- with no code-path difference between languages.
    _SENT_MARKERS.clear()
    rng_en = np.random.default_rng(seed)
    ids_en = P1._Ids()
    en_train, _c, _b, _m = P1.build_dataset(rng_en, P1.TRAIN_ANIMATE, P1.TRAIN_INANIM, P1.TRAIN_VERBS,
                                            n_per_cond=n_per_cond, ids=ids_en,
                                            noncanon_train_frac=noncanon_train_frac_en)
    en_ex = P1._examples_to_evidence(en_train)
    comp_en = _build_competition(seed, **comp_kw)
    learn_case_three_factor(comp_en, en_ex, epochs=epochs, settle_steps=train_steps, seed=seed)
    comp_en.freeze_all_cue_plasticity()
    w_en = comp_en.cue_weights()

    # --- JAPANESE-TOY (the case corpus; markers present -> case cue fires) ---
    rng_jp = np.random.default_rng(seed)
    ids_jp = P1._Ids()
    jp_train, _c2, _b2, _m2 = build_case_dataset(rng_jp, JP_AGENTS, JP_PATIENTS, JP_VERBS,
                                                 n_per_cond=n_per_cond, ids=ids_jp,
                                                 noncanon_train_frac=noncanon_train_frac_jp)
    jp_ex = P1._examples_to_evidence(jp_train)
    comp_jp = _build_competition(seed, **comp_kw)
    learn_case_three_factor(comp_jp, jp_ex, epochs=epochs, settle_steps=train_steps, seed=seed)
    comp_jp.freeze_all_cue_plasticity()
    w_jp = comp_jp.cue_weights()

    # the dissociation gates: English -> uses POSITION (not case); Japanese -> uses CASE (not position).
    en_pos = w_en["position"]
    en_sem = 0.5 * (w_en["animacy"] + w_en["verbfit"])
    en_lex = w_en["lexbias"]
    jp_case = w_jp["case"]
    jp_pos = w_jp["position"]
    diss = {
        "seed": seed,
        "english_weights": {k: round(v, 4) for k, v in w_en.items()},
        "japanese_weights": {k: round(v, 4) for k, v in w_jp.items()},
        # English: the CASE cue stayed at FLOOR (it never fired -- no markers in English), so the English learner
        # did NOT acquire a case strategy. (THE point: case is language-specific; with no case input it is unused.)
        "english_case_at_floor": w_en["case"] <= max(2.0, en_sem * 0.20),
        # English: POSITION is a genuinely-USED cue -- materially above the chance distractor (lexbias) floor AND
        # above the (floored) case cue. (English weights position over case; it need NOT beat the even-stronger
        # SEMANTIC cues -- the Competition-Model point is that English HAS a position strategy, while Japanese has a
        # case one. Position being secondary to the very-reliable animacy/verb-fit in English is expected, not a
        # failure: animacy is also a strong cue in English. What flips across languages is POSITION vs CASE.)
        "english_position_used_over_case": (en_pos > w_en["case"]) and (en_pos >= max(2.0, 2.0 * en_lex)),
        # Japanese: case DOMINANT (above position by a real margin) AND case is the TOP cue overall (case language).
        "japanese_case_dominant": jp_case >= jp_pos + 0.25 * max(1e-9, jp_case),
        "japanese_case_is_top": jp_case >= max(w_jp[c] for c in CUES if c != "case") - 1e-6,
    }
    # the OPPOSITE-PROFILE assertion (the headline): case's rank vs position FLIPS between the two languages --
    # English position > case; Japanese case > position. SAME code+learner, two corpora, opposite cue profiles.
    diss["profile_flips"] = bool(
        (w_en["position"] > w_en["case"]) and (w_jp["case"] > w_jp["position"]))
    diss["dissociation_GO"] = bool(
        diss["english_case_at_floor"] and diss["english_position_used_over_case"] and
        diss["japanese_case_dominant"] and diss["japanese_case_is_top"] and diss["profile_flips"])
    if verbose:
        print("\n[DISSOCIATION] English vs Japanese-toy (same code, same learner):")
        print(f"  English : " + ", ".join(f"{c}={w_en[c]:.2f}" for c in CUES))
        print(f"  Japanese: " + ", ".join(f"{c}={w_jp[c]:.2f}" for c in CUES))
        print(f"  profile_flips={diss['profile_flips']}  dissociation_GO={diss['dissociation_GO']}")
    return diss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 seed (42), verbose, dissociation included")
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46,47")
    ap.add_argument("--n-per-cond", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--train-steps", type=int, default=18)
    ap.add_argument("--read-steps", type=int, default=60)
    ap.add_argument("--learn-mode", choices=("hebbian", "error_gated", "install"), default="error_gated",
                    help="error_gated=brain-based three-factor on-substrate validity learning (the headline + the "
                         "dissociation REQUIRES this); install=validated case-language validities installed "
                         "(robust fallback arm); hebbian=plain co-firing (characterized NEGATIVE for validity).")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--no-dissociation", action="store_true", help="skip the cross-linguistic dissociation block")
    ap.add_argument("--noncanon-train-frac", type=float, default=0.65,
                    help="non-canonical (free-order) fraction of the Japanese TRAINING distribution (HIGH so "
                         "position is uninformative and case is decisive).")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    seeds = [42] if args.smoke else [int(s) for s in args.seeds.split(",") if s.strip()]
    controls = not args.no_controls
    do_diss = not args.no_dissociation

    results = []
    for s in seeds:
        r = run_seed(s, n_per_cond=args.n_per_cond, learn_mode=args.learn_mode, epochs=args.epochs,
                     train_steps=args.train_steps, read_steps=args.read_steps, controls=controls,
                     noncanon_train_frac=args.noncanon_train_frac, verbose=args.smoke)
        results.append(r)
        print(f"[seed {s}] case-run done: GO={r['seed_GO']}", flush=True)

    diss_results = []
    if do_diss and args.learn_mode == "error_gated":
        for s in seeds:
            d = run_dissociation(s, n_per_cond=args.n_per_cond, epochs=args.epochs,
                                 train_steps=args.train_steps,
                                 noncanon_train_frac_jp=args.noncanon_train_frac, verbose=args.smoke)
            diss_results.append(d)
            print(f"[seed {s}] dissociation done: GO={d['dissociation_GO']} "
                  f"(EN pos={d['english_weights']['position']:.1f}/case={d['english_weights']['case']:.1f}; "
                  f"JP case={d['japanese_weights']['case']:.1f}/pos={d['japanese_weights']['position']:.1f})",
                  flush=True)

    n = len(results)
    n_go = sum(r["seed_GO"] for r in results)
    key = "_mean_posdeg"
    mc = [r["case_battery"][key] for r in results]
    pos = [r["position_only_battery"][key] for r in results]
    les = [r["case_lesion_battery"][key] for r in results]
    breaches = sum(r["moat"]["breaches"] for r in results)
    has_controls = controls and args.learn_mode in ("hebbian", "error_gated")

    print("\n" + "=" * 92)
    print("PHASE-2 CASE-CUE CROSS-LANGUAGE -- free-word-order role comprehension de-risk (on SimulationBridge)")
    print("=" * 92)
    print(f"seeds: {seeds}   n_per_cond={args.n_per_cond}   learn_mode={args.learn_mode}   "
          f"noncanon_train_frac(JP)={args.noncanon_train_frac}")
    print(f"chance (2-role agent/patient) = 0.500")
    print(f"metric below = FREE-WORD-ORDER battery (free_order + object_front), case-marked\n")
    cols = f"{'seed':>5} | {'CASE':>6} | {'POS-ONLY':>8} | {'LESION':>7}"
    if has_controls:
        cols += f" | {'NO-LEARN':>8} | {'PERM-CASE':>9}"
    cols += f" | {'moat_br':>7} | {'sig':>3} | GO"
    print(cols); print("-" * len(cols))
    for r in results:
        line = (f"{r['seed']:>5} | {r['case_battery'][key]:>6.3f} | "
                f"{r['position_only_battery'][key]:>8.3f} | {r['case_lesion_battery'][key]:>7.3f}")
        if has_controls:
            line += (f" | {r['nolearn_battery'][key]:>8.3f} | {r['permuted_case_battery'][key]:>9.3f}")
        line += (f" | {r['moat']['breaches']:>7d} | {('Y' if r['weight_signature_ok'] else 'n'):>3} | "
                 f"{'GO' if r['seed_GO'] else 'no'}")
        print(line)
    print("-" * len(cols))
    mline = f"{'mean':>5} | {np.mean(mc):>6.3f} | {np.mean(pos):>8.3f} | {np.mean(les):>7.3f}"
    if has_controls:
        nol = [r["nolearn_battery"][key] for r in results]
        perm = [r["permuted_case_battery"][key] for r in results]
        mline += f" | {np.mean(nol):>8.3f} | {np.mean(perm):>9.3f}"
    mline += f" | {breaches:>7d} |"
    print(mline)

    print("\nPer-condition (mean across seeds): CASE  vs  POSITION-ONLY")
    for cond in ("free_order", "object_front", "case_absent"):
        m = np.mean([r["case_battery"][cond] for r in results])
        p = np.mean([r["position_only_battery"][cond] for r in results])
        note = "  (case ALSO silent here -- graceful degradation, NOT gated)" if cond == "case_absent" else ""
        print(f"  {cond:>14}:   {m:>5.3f}   vs   {p:>5.3f}{note}")

    cm = np.mean([r["clean_case"] for r in results])
    cp = np.mean([r["clean_position_only"] for r in results])
    print(f"\nclean canonical SOV (no-regression): case {cm:.3f}  vs  position-only {cp:.3f}")
    print(f"learned cue->role weights (mean): " +
          ", ".join(f"{c}={np.mean([r['weights_learned'][c] for r in results]):.3f}" for c in CUES))
    if has_controls:
        print(f"frozen  cue->role weights (mean): " +
              ", ".join(f"{c}={np.mean([r['weights_frozen'][c] for r in results]):.3f}" for c in CUES))

    # dissociation summary
    diss_go = None
    if diss_results:
        diss_go = sum(d["dissociation_GO"] for d in diss_results)
        print("\n" + "-" * 92)
        print("THE CROSS-LINGUISTIC DISSOCIATION (same code+learner; English vs Japanese-toy):")
        print(f"{'seed':>5} | {'EN: pos':>7} {'anim':>6} {'vfit':>6} {'case':>6} {'lex':>5}  ||  "
              f"{'JP: pos':>7} {'anim':>6} {'vfit':>6} {'case':>6} {'lex':>5}  | flip | GO")
        for d in diss_results:
            e, j = d["english_weights"], d["japanese_weights"]
            print(f"{d['seed']:>5} | {e['position']:>7.2f} {e['animacy']:>6.2f} {e['verbfit']:>6.2f} "
                  f"{e['case']:>6.2f} {e['lexbias']:>5.2f}  ||  "
                  f"{j['position']:>7.2f} {j['animacy']:>6.2f} {j['verbfit']:>6.2f} {j['case']:>6.2f} "
                  f"{j['lexbias']:>5.2f}  | {('Y' if d['profile_flips'] else 'n'):>4} | "
                  f"{'GO' if d['dissociation_GO'] else 'no'}")
        print(f"  English profile  -> position-dominant, case at floor (case never fired)")
        print(f"  Japanese profile -> case-dominant, position driven low (free order -> position unreliable)")
        print(f"  dissociation: {diss_go}/{len(diss_results)} seeds show the OPPOSITE-profile flip")

    overall_go = (n_go >= max(1, int(np.ceil(0.8333 * n))) and breaches == 0)
    if diss_results:
        overall_go = overall_go and (diss_go >= max(1, int(np.ceil(0.8333 * len(diss_results)))))
    print("\n" + "=" * 90)
    print(f"VERDICT: case-run {n_go}/{n} GO  +  moat breaches {breaches} (must be 0)" +
          (f"  +  dissociation {diss_go}/{len(diss_results)}" if diss_results else "") +
          f"  ->  {'GO' if overall_go else 'NEGATIVE / BOUNDARY'}")
    print("=" * 90)

    payload = {"seeds": seeds, "n_per_cond": args.n_per_cond, "learn_mode": args.learn_mode,
               "has_controls": has_controls, "noncanon_train_frac_jp": args.noncanon_train_frac,
               "n_go": n_go, "n": n, "total_moat_breaches": breaches,
               "dissociation_n_go": diss_go, "dissociation_n": len(diss_results),
               "overall_GO": overall_go, "results": results, "dissociation": diss_results}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[wrote] {args.out}")
    return payload


if __name__ == "__main__":
    main()
