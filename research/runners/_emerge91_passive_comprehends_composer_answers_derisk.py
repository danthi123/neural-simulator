"""EMERGE-91 -- PASSIVE comprehension IN the spiking comprehension->composition pipeline: the brain HEARS a PASSIVE
sentence ("the O was Ved by the S"), COMPREHENDS it ON SPIKES (assigns AGENT to the by-phrase noun S and THEME to the
surface subject O via the emergent per-role SPIKING read-out, NOT the canonical form->role which would wrongly assign
AGENT to the surface subject O), the composer STORES the fact, and ANSWERS a query about it -- with the no-confab moat.

CONTEXT (established; NOT re-derived here).
  * EMERGE-90 (`_emerge90_objrel_comprehends_composer_answers_derisk`, GO 6/6) wired the emergent fronto-striatal spiking
    reservoir comprehension INTO the fully-spiking conversational pipeline: the brain comprehends + answers CANONICAL SVO
    AND OBJECT-RELATIVE ("the S1 that the S2 Vs" -> THEME=head) sentences, emergent, spiking, no-confab moat. KEY FINDING:
    the comprehension is ROBUST -- a canonical-only learned read generalizes to objrel because the reservoir feature +
    slotting do the heavy lifting, so it is likely a GENERAL form->role mechanism.
  * The emergent read-out is the `DopaminePlasticReadout` (N_ROLES3=3 per-role Dale-legal SPIKING BinaryRoleDetectors,
    graded-DA reward-modulated delta from a random Dale init -- EMERGENT, NOT the ridge), on the c2 SPIKING reservoir
    (`GT._build_emergent`); the objrel slot0 THEME/AGENT tie is broken answer-independently by `gradedtie`.

THE BUILD (this file). Extend the EMERGE-90 pipeline to PASSIVE constructions. Reuse-by-import EMERGE-90's
`ObjrelReservoirComprehender` (the objrel emergent read-out + gradedtie + content-lexeme slotting) + its pipeline scaffold
UNCHANGED; the ONLY additions are:
  (1) PASSIVE sentence generation (`_make_passive`): "the O was Ved by the S" -> content-lexeme slots [O, Ved, S] with
      roles {O=THEME (slot0), Ved=PREDICATE (slot1), S=AGENT (slot2)}. The passive function words {"was", "by"} are NEW
      OOV closed words (like objrel's "that": not in the discovered closed class -> the encoder maps them to the OPEN
      marker) -- so they are EXCLUDED from the content slots by the SAME content-lexeme filter EMERGE-90 uses, no new
      hand rule. The SURFACE order != the role order (the surface subject O is the PATIENT, not the agent), so a
      canonical/position read would MISREAD it (assign AGENT to O).
  (2) A read-out that has SEEN passives (`_train_emergent_with_passive`): reuse DP._train_dopamine UNCHANGED, but on slot
      features cached over train sentences that INCLUDE passive (alongside canonical + objrel + the rest). We report BOTH:
        * the EMERGE-90 read-out (canonical+objrel-trained, `GT._build_emergent`, UNCHANGED) on passives -- the
          GENERALIZATION test (does the objrel/canonical training transfer to the never-seen passive slot-role pattern?);
        * the passive-trained read-out on passives -- the capability if generalization is insufficient.
      Which one carries the GO is REPORTED HONESTLY (per the task).

  WHY PASSIVE is a DISTINCT slot-role pattern (the generality test). Over sorted content positions:
    * canonical "the S Vs the O"          -> slot roles [AGENT, PREDICATE, THEME]  (slot0=AGENT, slot2=THEME)
    * objrel    "the S1 that the S2 Vs"   -> slot roles [THEME, AGENT, PREDICATE]  (slot0=THEME, slot1=AGENT)
    * passive   "the O was Ved by the S"  -> slot roles [THEME, PREDICATE, AGENT]  (slot0=THEME, slot2=AGENT)  <- NEW
  Passive shares slot0=THEME with objrel, but slot2=AGENT is NEVER seen in canonical (slot2=THEME) NOR objrel (slot2=
  PREDICATE) at slot2 -- so passive is a genuinely NOVEL slot-role configuration. If the canonical+objrel read-out
  GENERALIZES to it, that is a strong generality result; if not, the passive-trained read-out is the capability. Either
  way HONEST.

ANTI-CHEATS (mandatory).
  * EMERGENT + GENUINELY SPIKING: the passive role assignment comes from the EMERGENT reservoir read-out (per-role Dale-
    legal SPIKING detectors, delta-rule from a random Dale init -- DP._train_dopamine, argmax over OUTPUT-LIF SPIKE
    COUNTS), NOT a hand-coded "passive->swap" rule. The content-lexeme slotting only EXCLUDES the function words {was, by}
    (lexical categorization), it does NOT assign the roles. A PRE-learning (epochs=0) passive parse is reported (~chance).
  * NECESSITY (the genuine contrast, per the EMERGE-90 lesson): a POSITION/surface-order read (agent=surface-subject,
    i.e. slot0=AGENT) MISREADS the passive (assigns AGENT to O, the patient) -> the passive recall via a POSITION read
    collapses. Reported as `position_read_recall_passive` (should be LOW). NB: a canonical-LEARNED read GENERALIZES the
    positional prior, so we use the strict POSITION read (slot0->AGENT, slot1->PREDICATE, slot2->THEME) as the necessity
    contrast, NOT a canonical-learned read.
  * EXACT-FACT parse (agent==∧action==∧patient== ground truth); HELD-OUT content (fresh draws, distinct rng); no-confab
    MOAT (never-stored (agent,action) -> abstain (None); non-None = false-accept).
  * COMPREHENSION-LESION: collapse the reservoir's closed-class identity (encoder lesion) -> the passive role read
    degrades -> passive recall collapses (comprehension load-bearing).

SMOKE (then STOP; the controller fans out the 6-seed sweep + adversarially verifies). 1 seed: HEAR one passive + one
canonical (sanity) sentence, comprehend on spikes (via the emergent read-out that has seen passives), store, query, +
the moat abstain + the position-read contrast (misreads the passive) + the lesion. Also report whether the EMERGE-90
(canonical+objrel-only) read-out GENERALIZES to passives WITHOUT passive training. Honest either way.

Reuse-by-import: EMERGE-90's `ObjrelReservoirComprehender` / `_recall_over_facts` / `_parse_hits` / `_D` scaffold; the
objrel emergent read-out (`GT._build_emergent`, `DP._train_dopamine`, `DP.DopaminePlasticReadout`) + the c2 spiking
reservoir (`PR._build`/`PR._feature`, `D._cache_slot_features`) + the `RFPhasorComposer`. NO sim/ edit; CPU/numpy.

Run (smoke):
  SIM_BACKEND=numpy python -u -m research.runners._emerge91_passive_comprehends_composer_answers_derisk \
      --seeds 42 --json research/findings/raw/_emerge91_passive_comprehends.json

Fan-out (controller; the 6-seed sweep -- one process PER seed, aggregated after):
  for s in 42 43 44 100 101 102; do
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SIM_BACKEND=numpy \
      python -u -m research.runners._emerge91_passive_comprehends_composer_answers_derisk \
        --seeds $s --json research/findings/raw/_emerge91_passive_seed$s.json \
        > research/findings/raw/_emerge91_passive_seed$s.log 2>&1 &
  done; wait; echo ALL DONE
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C  # noqa: E402
import research.runners._rungB1c_objrel_per_role_readout_derisk as PR  # noqa: E402
import research.runners._rungB1c_objrel_dann_readout_derisk as D  # noqa: E402
import research.runners._rungB1c_objrel_dopamine_plasticity_derisk as DP  # noqa: E402
import research.runners._rungB1c_objrel_emergent_gradedtie_smoke as GT  # noqa: E402
from research.runners._emerge88_reservoir_comprehends_composer_answers_derisk import _ROLE2FIELD, _D  # noqa: E402
from research.runners._emerge78_reservoir_form_to_role_derisk import (  # noqa: E402
    Encoder, _gen, _TRAIN_KINDS, _ROLE_IDX, _ROLES,
)
from research.runners._emerge90_objrel_comprehends_composer_answers_derisk import (  # noqa: E402
    ObjrelReservoirComprehender, _parse_hits, _recall_over_facts,
)
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402

N_ROLES3 = DP.N_ROLES3


# ── the RESERVOIR-SILENCE lesion comprehender (the AIRTIGHT load-bearing control). The existing ENCODER lesion (collapse
#    the reservoir's closed-class IDENTITY) does NOT collapse passive recall, because passive's function words {"was","by"}
#    are OOV (not in the discovered closed class), so the closed-class lesion never touches them (the documented EMERGE-78/
#    c2 encoder-lesion weakness). The genuinely load-bearing lesion is to SILENCE the reservoir FEATURE itself: zero the
#    `final_state(...)` reservoir vector before the read-out, keeping ONLY the +1 bias element -> f = [0,...,0, 1.0]. Then
#    the read-out has NO reservoir information -> it reads a constant/chance role -> the stored fact is wrong -> PASSIVE
#    recall must COLLAPSE. This is the direct test that the reservoir COMPREHENSION is load-bearing for the pipeline turn.
#    Reuse-by-import: a thin subclass of EMERGE-90's ObjrelReservoirComprehender, overriding ONLY comprehend to add the
#    `reservoir_silence` mode; the encoder-`lesion` path is UNCHANGED (kept as a SEPARATE mode). NO sim/ edit. ───────────
class ReservoirSilenceComprehender(ObjrelReservoirComprehender):
    """ObjrelReservoirComprehender with an extra `reservoir_silence` mode: SILENCE the reservoir feature (keep only the
    +1 bias) before the emergent read-out, so the read-out has no reservoir signal. Everything else (content-lexeme
    slotting, the emergent read-out + gradedtie, the role->field map) is EMERGE-90's, byte-for-byte. The encoder-`lesion`
    path is inherited UNCHANGED (a separate, orthogonal mode)."""

    def comprehend(self, tokens, lesion=False, reservoir_silence=False):
        f = np.concatenate([self.res.final_state(self.enc.encode(tokens, lesion=lesion)), [1.0]])
        if reservoir_silence:
            f = np.zeros_like(f)
            f[-1] = 1.0                                   # keep ONLY the +1 bias; the reservoir vector is zeroed
        content = [t for t, w in enumerate(tokens) if w in self.content_lex]
        fact = {}
        for k, t in enumerate(content):
            if k >= N_ROLES3:
                break
            ri = self._role_for_slot(k, f)
            if ri is None:
                continue
            field = _ROLE2FIELD.get(_ROLES[ri])
            if field is not None and field not in fact:
                fact[field] = tokens[t]
        return fact


def _recall_over_facts_ressilence(composer, comprehender, facts):
    """Like EMERGE-90's `_recall_over_facts` but comprehends with reservoir_silence=True (the airtight load-bearing
    lesion): COMPREHEND (reservoir silenced) -> STORE the parsed fact -> query_patient(agent, action) -> fraction
    recalling the true patient. If the reservoir comprehension is load-bearing, this COLLAPSES."""
    for toks, ag, ac, pt, _kind in facts:
        fact = comprehender.comprehend(toks, reservoir_silence=True)
        if {"agent", "action", "patient"} <= set(fact):
            composer.store(fact["agent"], fact["action"], fact["patient"])
    hit = 0
    for toks, ag, ac, pt, _kind in facts:
        hit += int(composer.query_patient(ag, ac) == pt)
    return hit / max(1, len(facts))


# ── PASSIVE sentence generation. "the O was Ved by the S". Content-lexeme slots (sorted content positions) = [O, Ved, S]
#    with roles {O=THEME (slot0), Ved=PREDICATE (slot1), S=AGENT (slot2)}. The function words {"was", "by"} are NEW OOV
#    closed words (like objrel's "that"): not in the discovered closed class -> the encoder maps them to the OPEN marker
#    -> the SAME content-lexeme filter that excludes "that" excludes them. Surface subject O is the PATIENT (role !=
#    position), so a canonical/position read misassigns AGENT to O. ─────────────────────────────────────────────────
def _make_passive(rng, subj, verb, obj):
    """One passive sentence + its {position: role} map (over ALL tokens; the comprehender/read-out re-index by content
    slot). "the O was Ved by the S": positions the(0) O(1) was(2) Ved(3) by(4) the(5) S(6)."""
    s = str(rng.choice(subj)); o = str(rng.choice(obj)); v = str(rng.choice(verb))
    v3 = v + "s"                                    # keep the same inflected surface form the read-out was trained on
    toks = ["the", o, "was", v3, "by", "the", s]
    roles = {1: "THEME", 3: "PREDICATE", 6: "AGENT"}   # O=THEME(slot0), Ved=PREDICATE(slot1), S=AGENT(slot2)
    return toks, roles


def _gen_passive(n, rng, subj, verb, obj):
    return [_make_passive(rng, subj, verb, obj) for _ in range(n)]


# ── the PASSIVE-aware emergent read-out (reuse DP._train_dopamine UNCHANGED; only the TRAIN sentence set includes
#    passives). Cache slot features over canonical+objrel+passive (+the rest), then train the SAME DopaminePlasticReadout
#    per slot -- EMERGENT (delta from random Dale init), NOT the ridge. Returns (ros_main, ros_pre, feat_dim). ─────────
def _train_emergent_with_passive(res, enc, seed, subj, verb, obj):
    """Train the emergent per-role SPIKING read-out on a train set that INCLUDES passive (alongside the EMERGE-90 kinds).
    Reuses D._cache_slot_features + DP._train_dopamine byte-for-byte (no new learning rule); only the SENTENCE SET grows.
    Returns (ros_main, ros_pre) where ros_pre is the epochs=0 random-Dale-init read (the emergent PRE-learning control)."""
    rng = np.random.default_rng(seed * 101 + 5)
    train = _gen(_TRAIN_KINDS, DP.N_TRAIN, rng, subj, verb, obj)                       # the EMERGE-90 train kinds
    train += _gen_passive(DP.N_TRAIN, rng, subj, verb, obj)                            # + PASSIVE
    slot_train = D._cache_slot_features(res, enc, train)
    feat_dim = next(iter(slot_train.values()))[0].shape[1]
    ros_main = DP._train_dopamine(slot_train, feat_dim, seed, epochs=DP.EPOCHS, salience=True, reward_on=True)
    ros_pre = DP._train_dopamine(slot_train, feat_dim, seed, epochs=0)                 # PRE-learning (random Dale init)
    return ros_main, ros_pre, feat_dim


# ── the strict POSITION-read comprehender (the NECESSITY contrast). slot0->AGENT, slot1->PREDICATE, slot2->THEME (the
#    canonical surface-order prior). On a passive it assigns AGENT to the surface subject O (WRONG -> O is the patient),
#    so its passive recall collapses. NOT a canonical-LEARNED read (which would generalize the prior); a hard position
#    map, so the contrast is unambiguous. Same content-lexeme slotting as the emergent comprehender (fair contrast). ──
_POSITION_SLOT_ROLE = {0: "AGENT", 1: "PREDICATE", 2: "THEME"}


class PositionReadComprehender:
    """The surface-order NECESSITY contrast: role = the canonical POSITION prior (slot0=AGENT, slot1=PREDICATE,
    slot2=THEME), regardless of form. On a passive the surface subject (slot0) is mis-assigned AGENT (it is the THEME),
    so the stored fact is wrong -> passive recall collapses. Uses the SAME content-lexeme slotting as the emergent
    comprehender (identical slots; only the role source differs)."""

    def __init__(self, content_lex):
        self.content_lex = set(content_lex)

    def comprehend(self, tokens, lesion=False):
        content = [t for t, w in enumerate(tokens) if w in self.content_lex]
        fact = {}
        for k, t in enumerate(content):
            if k >= N_ROLES3:
                break
            field = _ROLE2FIELD.get(_POSITION_SLOT_ROLE[k])
            if field is not None and field not in fact:
                fact[field] = tokens[t]
        return fact


# ── held-out test facts: PASSIVE + CANONICAL sentences, fresh content draws (held out from the read-out fit) ─────────
def _build_passive_test_facts(seed, subj, verb, obj, n=12):
    """CANONICAL + PASSIVE sentences with fresh CONTENT draws (distinct rng from the read-out fit). Each entry is
    (tokens, agent, action, patient, kind): the fact FIELDS the pipeline must recover.
      * canonical "the S Vs the O"          -> agent=S, action=Vs, patient=O
      * passive   "the O was Ved by the S"  -> agent=S, action=Ved, patient=O   (surface subject O is the PATIENT)
    Distinct (agent, action) within each kind so the who/what query is unambiguous."""
    trng = np.random.default_rng(seed * 733 + 11)
    canon, pasv = [], []
    cseen, pseen = set(), set()
    guard = 0
    while (len(canon) < n or len(pasv) < n) and guard < 20000:
        guard += 1
        s = str(trng.choice(subj)); o = str(trng.choice(obj)); v = str(trng.choice(verb)); v3 = v + "s"
        if len(canon) < n and (s, v3) not in cseen:
            cseen.add((s, v3))
            canon.append((["the", s, v3, "the", o], s, v3, o, "canonical"))
        if len(pasv) < n and (s, v3) not in pseen:
            pseen.add((s, v3))
            # passive "the O was Ved by the S": agent=S (by-phrase), patient=O (surface subject), action=Ved
            pasv.append((["the", o, "was", v3, "by", "the", s], s, v3, o, "passive"))
    return canon, pasv, trng


def _derisk_one(seed):
    t0 = time.time()
    corpus = C.setup_corpus(seed=42)                            # shared corpus (the objrel scaffold's own setup)
    subj, verb, obj = corpus["subj"], corpus["verb"], corpus["obj"]

    # ── the c2 SPIKING reservoir + the EMERGE-90 (canonical+objrel-trained) emergent read-out (UNCHANGED). ────────────
    (res, enc, _canon_gen, _objr_gen, ros_objrel, ros_objrel_pre, _ros_nr,
     dale_legal_objrel, slot0_counts) = GT._build_emergent(seed, corpus)

    # ── the PASSIVE-aware emergent read-out (reuse DP._train_dopamine; train set includes passive). Same reservoir. ────
    ros_pasv, ros_pasv_pre, _feat_dim = _train_emergent_with_passive(res, enc, seed, subj, verb, obj)
    dale_legal_pasv = all(ro.dale_legal()["legal"] for ro in ros_pasv.values())

    # ── held-out CANONICAL + PASSIVE test facts (fresh content, distinct rng) ────────────────────────────────────────
    canon_facts, pasv_facts, trng = _build_passive_test_facts(seed, subj, verb, obj, n=12)

    v3 = [v + "s" for v in verb]
    vocab = sorted(set(subj) | set(v3) | set(obj))
    content_lex = set(subj) | set(v3) | set(obj)               # content-word slot filter (excludes "was"/"by"/"that")

    # ── comprehenders. All reuse EMERGE-90's ObjrelReservoirComprehender (emergent read-out + gradedtie + slotting);
    #    only the READ-OUT differs. gradedtie handles the slot0 THEME/AGENT tie exactly as in objrel (passive slot0=THEME). ─
    pasv_comp = ReservoirSilenceComprehender(res, enc, ros_pasv, content_lex, gradedtie=True)         # passive-trained
    # ^ ReservoirSilenceComprehender IS-A ObjrelReservoirComprehender: identical behaviour on the normal + encoder-lesion
    #   paths (comprehend(reservoir_silence=False) is byte-for-byte EMERGE-90); it ONLY adds the reservoir_silence mode.
    pasv_pre_comp = ObjrelReservoirComprehender(res, enc, ros_pasv_pre, content_lex, gradedtie=True)  # PRE-learning
    objrel_comp = ObjrelReservoirComprehender(res, enc, ros_objrel, content_lex, gradedtie=True)      # EMERGE-90 read-out
    pos_comp = PositionReadComprehender(content_lex)                                                  # necessity contrast

    # ── PARSE accuracy (comprehension, composer-independent) ─────────────────────────────────────────────────────────
    pasv_parse_pasv = _parse_hits(pasv_comp, pasv_facts)            # capability: passive-trained read-out on PASSIVE
    pasv_parse_canon = _parse_hits(pasv_comp, canon_facts)         # passive-trained read-out on CANONICAL (must not break)
    objrel_parse_pasv = _parse_hits(objrel_comp, pasv_facts)       # GENERALIZATION: EMERGE-90 read-out on PASSIVE
    pos_parse_pasv = _parse_hits(pos_comp, pasv_facts)             # necessity: POSITION read MISREADS passive (LOW)
    pos_parse_canon = _parse_hits(pos_comp, canon_facts)          # position read on canonical (~high; sanity)
    pre_parse_pasv = _parse_hits(pasv_pre_comp, pasv_facts)       # EMERGENT: PRE-learning passive parse (~chance)

    # ── THE INTEGRATION: comprehend -> store -> who/what recall, per construction (fresh composer per condition) ───────
    comp_pasv = RFPhasorComposer(seed=seed, D=_D, vocab=vocab)
    pasv_recall_pasv = _recall_over_facts(comp_pasv, pasv_comp, pasv_facts, lesion=False)             # HEADLINE
    comp_canon = RFPhasorComposer(seed=seed, D=_D, vocab=vocab)
    pasv_recall_canon = _recall_over_facts(comp_canon, pasv_comp, canon_facts, lesion=False)          # canonical intact
    comp_objrel_on_pasv = RFPhasorComposer(seed=seed, D=_D, vocab=vocab)
    objrel_recall_pasv = _recall_over_facts(comp_objrel_on_pasv, objrel_comp, pasv_facts, lesion=False)  # generalization
    comp_pos_on_pasv = RFPhasorComposer(seed=seed, D=_D, vocab=vocab)
    position_recall_pasv = _recall_over_facts(comp_pos_on_pasv, pos_comp, pasv_facts, lesion=False)   # necessity contrast

    # ── NO-CONFAB MOAT: an (agent, action) NEVER stored -> abstain (None). Query the PASSIVE composer. ────────────────
    stored_keys = {(ag, ac) for _t, ag, ac, _pt, _k in pasv_facts}
    fa = tot = 0
    mguard = 0
    while tot < 40 and mguard < 4000:
        mguard += 1
        s = str(trng.choice(subj)); v3q = str(trng.choice(verb)) + "s"
        if (s, v3q) in stored_keys:
            continue
        tot += 1
        fa += int(comp_pasv.query_patient(s, v3q) is not None)
    moat_fa = fa / max(1, tot)

    # ── COMPREHENSION-LESION #1 (ENCODER lesion -- the existing WEAK one): collapse the reservoir's closed-class identity.
    #    Passive's {was,by} are OOV, so the closed-class lesion does not touch them -> passive recall does NOT collapse
    #    (the documented EMERGE-78/c2 encoder-lesion weakness). Reported honestly alongside the airtight lesion below. ────
    comp_les = RFPhasorComposer(seed=seed, D=_D, vocab=vocab)
    encoder_lesion_recall = _recall_over_facts(comp_les, pasv_comp, pasv_facts, lesion=True)

    # ── COMPREHENSION-LESION #2 (RESERVOIR-SILENCE -- the AIRTIGHT load-bearing control, the ADD): SILENCE the reservoir
    #    FEATURE itself (zero the final_state vector; keep only the +1 bias) -> the read-out has NO reservoir information ->
    #    reads a constant/chance role -> the stored fact is wrong -> passive recall MUST collapse (comprehension is
    #    load-bearing for the turn). This is the direct test the encoder lesion could not deliver on passives. ────────────
    comp_ressil = RFPhasorComposer(seed=seed, D=_D, vocab=vocab)
    reservoir_silence_recall = _recall_over_facts_ressilence(comp_ressil, pasv_comp, pasv_facts)

    elapsed = round(time.time() - t0, 1)
    return {
        "seed": int(seed), "n_canon": len(canon_facts), "n_pasv": len(pasv_facts),
        "slot0_class_counts": slot0_counts,
        "dale_legal_passive_readout": bool(dale_legal_pasv), "dale_legal_objrel_readout": bool(dale_legal_objrel),
        # PARSE (comprehension, composer-independent)
        "passive_readout_parse_passive": round(pasv_parse_pasv, 3),          # the capability
        "passive_readout_parse_canonical": round(pasv_parse_canon, 3),      # passive read-out does NOT break canonical
        "objrel_readout_parse_passive": round(objrel_parse_pasv, 3),        # GENERALIZATION (EMERGE-90 read-out -> passive)
        "position_read_parse_passive": round(pos_parse_pasv, 3),            # necessity: position MISREADS passive (LOW)
        "position_read_parse_canonical": round(pos_parse_canon, 3),
        "pre_learning_parse_passive": round(pre_parse_pasv, 3),             # EMERGENT: PRE-learning ~chance
        # RECALL (comprehend -> store -> answer)
        "passive_recall_passive": round(pasv_recall_pasv, 3),              # HEADLINE: hear passive -> store -> answer
        "passive_recall_canonical": round(pasv_recall_canon, 3),          # canonical case still answered
        "objrel_readout_recall_passive": round(objrel_recall_pasv, 3),    # generalization recall
        "position_read_recall_passive": round(position_recall_pasv, 3),   # necessity recall (LOW = passive read needed)
        "moat_false_accept": round(moat_fa, 3),                           # no-confab moat
        # COMPREHENSION-LESION -- report BOTH (per the task):
        "encoder_lesion_recall": round(encoder_lesion_recall, 3),        # the EXISTING WEAK one (does NOT collapse: {was,by} OOV)
        "reservoir_silence_recall": round(reservoir_silence_recall, 3),  # the AIRTIGHT one (MUST collapse -> load-bearing)
        # back-compat alias for the GO gate: the load-bearing lesion is now the reservoir-silence lesion (airtight)
        "lesion_recall_passive": round(reservoir_silence_recall, 3),
        "elapsed_s": elapsed,
    }


def _go(rows):
    def mean(k):
        return float(np.mean([r[k] for r in rows]))
    return {
        "n_seeds": len(rows),
        "passive_readout_parse_passive": mean("passive_readout_parse_passive"),
        "passive_readout_parse_canonical": mean("passive_readout_parse_canonical"),
        "objrel_readout_parse_passive": mean("objrel_readout_parse_passive"),
        "position_read_parse_passive": mean("position_read_parse_passive"),
        "pre_learning_parse_passive": mean("pre_learning_parse_passive"),
        "passive_recall_passive": mean("passive_recall_passive"),
        "passive_recall_canonical": mean("passive_recall_canonical"),
        "objrel_readout_recall_passive": mean("objrel_readout_recall_passive"),
        "position_read_recall_passive": mean("position_read_recall_passive"),
        "moat_false_accept": mean("moat_false_accept"),
        "encoder_lesion_recall": mean("encoder_lesion_recall"),          # existing WEAK lesion (reported, NOT gating)
        "reservoir_silence_recall": mean("reservoir_silence_recall"),    # AIRTIGHT load-bearing lesion (gates)
        "lesion_recall_passive": mean("lesion_recall_passive"),          # alias == reservoir_silence_recall (the GO gate)
        "dale_legal_all": all(r["dale_legal_passive_readout"] for r in rows),
        # GO: the passive-trained emergent read-out drives correct PASSIVE who/what answers (>=0.85) AND does not break
        # canonical (>=0.90) AND the no-confab moat holds (<=0.05) AND comprehension is load-bearing (the AIRTIGHT
        # RESERVOIR-SILENCE lesion collapses recall <=0.55 -- NOT the weak encoder lesion, which cannot bite on passives
        # since {was,by} are OOV) AND the passive read-out is NECESSARY (a POSITION read on passive is materially lower)
        # AND the read is EMERGENT (PRE-learning passive parse ~chance, so the plasticity did the work). `lesion_recall_
        # passive` == reservoir_silence_recall.
        "go": (mean("passive_recall_passive") >= 0.85 and mean("passive_recall_canonical") >= 0.90
               and mean("moat_false_accept") <= 0.05 and mean("lesion_recall_passive") <= 0.55
               and (mean("passive_recall_passive") - mean("position_read_recall_passive")) >= 0.30
               and (mean("passive_readout_parse_passive") - mean("pre_learning_parse_passive")) >= 0.15),
        # a SEPARATE (reported, NOT gating) flag: did the EMERGE-90 (canonical+objrel-only) read-out GENERALIZE to passive?
        "objrel_readout_generalizes_to_passive": mean("objrel_readout_recall_passive") >= 0.85,
    }


def _print_seed(s, d):
    print(f"[seed {s}] slot0-cls {d['slot0_class_counts']} dale-legal(pasv/objrel) "
          f"{d['dale_legal_passive_readout']}/{d['dale_legal_objrel_readout']} | "
          f"PARSE passive-read: PASSIVE {d['passive_readout_parse_passive']:.2f} / CANON "
          f"{d['passive_readout_parse_canonical']:.2f} || objrel-read GENERALIZE->PASSIVE "
          f"{d['objrel_readout_parse_passive']:.2f} | POSITION-read on PASSIVE (misread) "
          f"{d['position_read_parse_passive']:.2f} | PRE-LEARN PASSIVE {d['pre_learning_parse_passive']:.2f}  ==  "
          f"RECALL passive-read: PASSIVE {d['passive_recall_passive']:.2f} / CANON {d['passive_recall_canonical']:.2f} "
          f"|| objrel-read-on-PASSIVE {d['objrel_readout_recall_passive']:.2f} | position-read-on-PASSIVE "
          f"{d['position_read_recall_passive']:.2f} | moat-FA {d['moat_false_accept']:.2f} | "
          f"LESION recall(passive): encoder(weak) {d['encoder_lesion_recall']:.2f} / RESERVOIR-SILENCE(airtight) "
          f"{d['reservoir_silence_recall']:.2f} ({d['elapsed_s']}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[emerge91] PASSIVE comprehension IN the spiking comprehension->composition pipeline: HEAR a passive "
          f"sentence ('the O was Ved by the S') -> comprehend on spikes (emergent per-role read-out assigns AGENT to the "
          f"by-phrase noun S, THEME to the surface subject O) -> composer STORES -> ANSWER, with the no-confab moat. "
          f"Contrast: canonical still works; a POSITION read misreads the passive (assigns AGENT to O). Also reports "
          f"whether the EMERGE-90 canonical+objrel read-out GENERALIZES to passives. seeds {args.seeds}. NO sim/ edit; "
          f"CPU/numpy. SMOKE (controller fans out + verifies).", flush=True)

    rows = []
    for s in args.seeds:
        d = _derisk_one(s)
        rows.append(d)
        _print_seed(s, d)

    agg = _go(rows)
    agg["elapsed_seconds"] = round(time.time() - t0, 1)
    verdict = "GO" if agg["go"] else "NO-GO"
    print(f"\n[emerge91] VERDICT: {verdict} -- the PASSIVE sentence is comprehended ON SPIKES (emergent per-role read-out: "
          f"AGENT=by-phrase noun, THEME=surface subject) and ANSWERED: passive who/what recall "
          f"{agg['passive_recall_passive']:.3f} (canonical NOT broken {agg['passive_recall_canonical']:.3f}); a POSITION "
          f"read misreads the passive (recall {agg['position_read_recall_passive']:.3f}, so the emergent read-out is "
          f"NECESSARY); EMERGENT (PRE-learning parse {agg['pre_learning_parse_passive']:.3f} -> learned "
          f"{agg['passive_readout_parse_passive']:.3f}); no-confab moat {agg['moat_false_accept']:.3f} false-accept; "
          f"the AIRTIGHT RESERVOIR-SILENCE lesion collapses passive recall to {agg['reservoir_silence_recall']:.3f} "
          f"(the weak encoder lesion does NOT: {agg['encoder_lesion_recall']:.3f}, since passive {{was,by}} are OOV). "
          f"GENERALIZATION: the "
          f"EMERGE-90 canonical+objrel read-out on passive recall = {agg['objrel_readout_recall_passive']:.3f} "
          f"(generalizes={agg['objrel_readout_generalizes_to_passive']}) -- so passive training "
          f"{'was NOT strictly needed' if agg['objrel_readout_generalizes_to_passive'] else 'WAS needed'}.",
          flush=True)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "agg": agg}, fh, indent=2, default=str)
        print(f"[emerge91] wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
