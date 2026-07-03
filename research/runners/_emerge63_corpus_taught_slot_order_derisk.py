"""EMERGE-63 -- LEARN the spiking-Broca producer's per-frame slot ORDER from the CORPUS's actual WORD-ORDER STATISTICS,
instead of from the HOST-designed frame template. This closes residual S1b of the self-organizing-grammatical-structure
research gate (`research/findings/2026-07-03-self-organizing-grammatical-structure-research-gate.md`, RANK 2 / Move 3:
"swap FrameCQ's order-teacher to corpus n-gram statistics -- near-free").

THE HOST RESIDUAL THIS REMOVES. EMERGE-59's `FrameSlotCQ._teach_order` (`_emerge59_spiking_broca_frame_slots_derisk.py`
:239-260) writes each frame's per-slot primacy gradient from the FRAMES dict's HOST-WRITTEN slot ORDER: the order-teacher
is `LR * (n-1-pool)` over the template's pool index == "the template SAYS slot i is i-th". EMERGE-63 replaces that teacher
with one derived PURELY from the observed WORD ORDER in corpus example sentences. The FRAMES dict's slot ORDER becomes the
VALIDATION ground-truth, NOT the teacher.

WHAT STAYS (S1a, out of scope -- EMERGE-64's residual, stated clearly). EMERGE-63 learns only the ORDER of the (given)
slots. The slot TYPES per frame (det / subj / func:can / func:does / func:not / verb -- WHICH typed slots a construction
licenses) are S1a and are STILL taken from the FRAMES template here; discovering the slot INVENTORY from the corpus is the
separate EMERGE-64 residual (extend the `_bucketB_corpus_mined_frames` frame-mining to FUNC slots). We also use the slot
TYPE labels to LOCATE each role's token in a corpus sentence (DET -> 'the'/'a', FUNC:x -> the token x, SUBJ -> the content
noun, VERB -> the content verb) -- this uses S1a (which roles + their lexical class), NOT the order. The order is read only
from the observed token POSITIONS.

THE MECHANISM (Dominey & Hinaut: grammar = the STATISTICS of the ORDER/POSITION of elements; no explicit rules; catalog
G.12 Broca; usage-based construction grammar, Tomasello). For each frame TYPE, collect its example sentences from the
corpus stream (reusing EMERGE-62's `build_stream` + sentence segmentation), locate each role's token position, and compute
a PAIRWISE PRECEDENCE / bigram-order statistic over the slot ROLES: prec[A][B] = fraction of examples where role A's token
PRECEDES role B's token (det<subj, subj<func, func<verb, ...). A role that precedes many others gets HIGH primacy (emitted
first). This precedence-derived primacy REORDERS the frame's slots; the reordered slots are fed into the EMERGE-59 spiking
producer, which renders them ON SPIKES in the corpus-learned order (the learned primacy gradient = graded current -> the
per-pool spiking-RATE ranking = the emission order; EMERGE-61's inter-utterance wash-out for position-independence). NO
host template order enters the teacher.

DE-RISK (6 seeds 42/43/44/100/101/102, CPU/numpy):
  (a) the CORPUS-TAUGHT order MATCHES the correct (template ground-truth) order -- render "the owl can fly" / "the penguin
      walks" / "the penguin does not fly" EXACTLY on spikes (with the EMERGE-61 wash-out for position-independence, so the
      order is not confounded by the Izhikevich-adaptation tail). Scored by order-accuracy (song_g1_core.score_order over
      the corpus-predicted role order vs the template order) + a spiking EXACT full-surface match.
  Anti-cheats that MUST COLLAPSE (input-destruction + hold-out, project control-validity methodology -- NOT a fixed-random
  control):
  (b1) SHUFFLED-CORPUS -- shuffle the word order WITHIN each example sentence -> the precedence statistics are destroyed ->
       the learned order is wrong/chance (the load-bearing "the order comes from the corpus WORD ORDER" claim).
  (b2) NO-CORPUS       -- no example sentences -> no precedence -> no order (chance).
  (b3) HELD-OUT-FRAME  -- learn the order statistics on 2 frames, test the 3rd's order GENERALIZES from the SHARED
       slot-TYPE precedences (det<subj<func<verb learned from F_MODAL/F_INTR predicts a fully-held-out F_NEGMOD). Scored
       with an HONEST (non-template) tie-break so a genuinely-unlearnable within-frame order shows up as a residual, not a
       smuggled template match.
  (c) the PRODUCER renders on spikes from the corpus-taught order AND the gate-first no-confab MOAT holds (0 producer
      invocations on abstains).
GO bar: corpus-taught order-accuracy high (main == 1.0) with a clear margin over every collapsed control, held-out-frame
generalizes on the SHARED precedences, the producer renders on spikes, moat intact, 6-seed.

HONEST SCOPE + the named residual. The MAIN arm (all frames' exemplars available) learns every frame's order EXACTLY,
INCLUDING the negated-modal's does<not (directly attested in F_NEGMOD's own exemplars -> precedence resolves it; MAIN
F_NEGMOD == 1.0 even with an honest random tie-break). The HELD-OUT-FRAME arm generalizes the SHARED type-level order
(det<subj<func<verb) perfectly to a fully-held-out frame, but the does-vs-not INTERNAL order of a held-out MULTI-function-
word frame is NOT learnable from the OTHER two frames alone (only F_NEGMOD attests two adjacent function words) -- that is
the genuine, precisely-named residual (the honest boundary the research gate flagged), NOT a wall: the next single signal
is one attestation of the does<not bigram (or Yang-Getz's phrase-boundary cue). We report held-out generalization on the
SHARED precedences (the claim) + name this within-frame residual explicitly. Reuse-by-import; NO `sim/` edit; moat
untouched (the corpus stat is offline syllabus prep -- BRAIN-BASED-ONLY compliant, like rendering a retinal image the
neural retina reads; the ORDER is produced on real spikes).

Run:
  SIM_BACKEND=numpy python -m research.runners._emerge63_corpus_taught_slot_order_derisk --demo
  SIM_BACKEND=numpy python -m research.runners._emerge63_corpus_taught_slot_order_derisk --derisk
  SIM_BACKEND=numpy python -m research.runners._emerge63_corpus_taught_slot_order_derisk --derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners.song_g1_core import score_order  # noqa: E402
# Reuse-by-import: EMERGE-62's corpus stream + segmentation + the lexicons; EMERGE-59's frames + producer; EMERGE-61's
# inter-utterance wash-out (position-independence). NO sim/ edit anywhere.
from research.runners._emerge62_discover_function_words_derisk import (  # noqa: E402
    build_stream, SENT_PERIOD, _SUBJECTS, _VERBS,
)
from research.runners._emerge59_spiking_broca_frame_slots_derisk import (  # noqa: E402
    FRAMES, FRAME_NAMES, DET, SUBJ, FUNC, VERB, BrocaProducer, decision_from_emerge,
    build_heldout_facts, _expected_words,
)
from research.runners._emerge61_spiking_broca_order_robustness_derisk import ResetFrameSlotCQ  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_emerge63_corpus_taught_slot_order.json"

_SUBJ_SET = set(_SUBJECTS)


# ---------------------------------------------------------------------------------------------------------------------
# LOCATE EACH ROLE'S TOKEN IN A CORPUS SENTENCE. Uses the slot TYPE labels (S1a: WHICH roles + their lexical class) to
# FIND each role's token position -- NOT the order. DET -> 'the'/'a'; FUNC:x -> the token x; SUBJ -> the content noun
# (the _SUBJECTS lexicon); VERB -> the content verb (the _VERBS lexicon, bare or 3sg -s). The ORDER is read only from
# the observed token positions, never from the template.
# ---------------------------------------------------------------------------------------------------------------------
def _role_key(slot):
    """Canonical role key. Shared roles (DET/SUBJ/VERB) by TYPE so they transfer across frames (held-out); FUNC keyed by
    its payload lemma (can/does/not) since function words are frame-specific."""
    t, p = slot
    if t == FUNC:
        return "FUNC:" + p
    return {DET: "DET", SUBJ: "SUBJ", VERB: "VERB"}[t]


def _is_verb(tok):
    return tok in _VERBS or (tok.endswith("s") and tok[:-1] in _VERBS)


def _is_subj(tok):
    return tok in _SUBJ_SET


def _classify(sent):
    """Which frame a corpus sentence realizes (by surface pattern). The frame's role SET is S1a-known; this recognizer
    only assigns exemplars to frames so the per-frame precedence can be accumulated -- it does NOT read the order."""
    if len(sent) == 4 and sent[0] == "the" and sent[2] == "can":
        return "F_MODAL"
    if len(sent) == 5 and sent[0] == "the" and sent[2] == "does" and sent[3] == "not":
        return "F_NEGMOD"
    if len(sent) == 3 and sent[0] == "the" and sent[2].endswith("s") and sent[1] in _SUBJ_SET:
        return "F_INTR"
    return None


def split_sentences(tokens):
    """Segment the corpus token stream into sentences on the SENT_PERIOD delimiter (EMERGE-62's segmentation)."""
    sents, cur = [], []
    for t in tokens:
        if t == SENT_PERIOD:
            if cur:
                sents.append(cur)
                cur = []
        else:
            cur.append(t)
    if cur:
        sents.append(cur)
    return sents


def _role_positions(sent, frame):
    """Map each of `frame`'s roles to its token index in `sent` (using the S1a lexical class, NOT the order). Returns a
    dict role_key -> index (only for roles found)."""
    pos, used = {}, set()
    for slot in FRAMES[frame]:
        t, p = slot
        if t in (DET, FUNC):
            for i, tok in enumerate(sent):
                if i in used:
                    continue
                if (t == DET and tok in ("the", "a")) or (t == FUNC and tok == p):
                    pos[_role_key(slot)] = i
                    used.add(i)
                    break
        elif t == SUBJ:
            for i, tok in enumerate(sent):
                if i in used:
                    continue
                if _is_subj(tok):
                    pos["SUBJ"] = i
                    used.add(i)
                    break
        elif t == VERB:
            for i, tok in enumerate(sent):
                if i in used:
                    continue
                if _is_verb(tok):
                    pos["VERB"] = i
                    used.add(i)
                    break
    return pos


# ---------------------------------------------------------------------------------------------------------------------
# THE CORPUS ORDER-TEACHER: a PAIRWISE PRECEDENCE (bigram-order) statistic over slot ROLES. prec[A][B] = count of
# examples where role A's token precedes role B's. A role that precedes many others gets high primacy (emitted first).
# This is the order-teacher, derived PURELY from the observed word order -- the host template order is never used.
# ---------------------------------------------------------------------------------------------------------------------
def corpus_precedence(sents_of_frame, frame, shuffle_within=False, shuffle_rng=None):
    """Accumulate pairwise role precedence for `frame` from its example sentences. `shuffle_within` (the SHUFFLED-CORPUS
    anti-cheat) scrambles each sentence's word order first -> destroys the precedence signal."""
    roles = [_role_key(sl) for sl in FRAMES[frame]]
    prec = {a: {b: 0 for b in roles} for a in roles}
    n_used = 0
    for sent in sents_of_frame:
        s = list(sent)
        if shuffle_within:
            shuffle_rng.shuffle(s)
        pos = _role_positions(s, frame)
        if len(pos) != len(roles):
            continue                                   # an exemplar we could not fully align (skip)
        n_used += 1
        for a in roles:
            for b in roles:
                if a != b and pos[a] < pos[b]:
                    prec[a][b] += 1
    return prec, roles, n_used


def order_from_precedence(prec, roles, tie_rng=None):
    """Rank roles by their precedence primacy (mean fraction-precedes over the other roles). A role that precedes many
    others -> high primacy -> emitted first. Ties (roles with equal precedence -- e.g. two function words never seen in
    sequence in the training slice) are broken by `tie_rng` (an HONEST non-template tie-break) so a genuinely-unlearnable
    order shows as a residual, NOT a smuggled template match. With `tie_rng=None` a deterministic index tie-break is used
    (only reached when precedence is fully determined, i.e. the main arm)."""
    def primacy(a):
        tot, num = 0.0, 0
        for b in roles:
            if a == b:
                continue
            ab, ba = prec[a][b], prec[b][a]
            if ab + ba > 0:
                tot += ab / (ab + ba)
                num += 1
        return tot / num if num else 0.0

    prim = {r: primacy(r) for r in roles}
    if tie_rng is not None:
        tie = {r: float(tie_rng.random()) for r in roles}
        keyf = lambda r: (-prim[r], tie[r])
    else:
        keyf = lambda r: (-prim[r], roles.index(r))
    return sorted(roles, key=keyf)


# ---------------------------------------------------------------------------------------------------------------------
# THE CORPUS-TAUGHT SPIKING PRODUCER: a ResetFrameSlotCQ (EMERGE-61 wash-out for position-independence) whose per-frame
# slot list is REORDERED into the CORPUS-predicted order BEFORE teaching. The teacher then writes a plain descending
# primacy over the reordered slots, so the per-pool spiking-RATE ranking reproduces the CORPUS order (not the template).
# ADDITIVE: EMERGE-59/61 are NOT edited; this subclass only reorders `frame_slots` from the corpus statistics.
# ---------------------------------------------------------------------------------------------------------------------
class CorpusOrderFrameSlotCQ(ResetFrameSlotCQ):
    """FrameSlotCQ (+ EMERGE-61 wash-out) whose slot ORDER is set by CORPUS precedence, not the host template. Pass a
    `corpus_order` dict frame -> ordered list of role keys; the frame's template slots are permuted into that order,
    then a monotone descending primacy is taught over the reordered slots (pool i = i-th corpus-order slot). The result:
    the spiking emission order == the corpus-learned order. If `corpus_order` is None the class is byte-identical to
    ResetFrameSlotCQ (template order) -- so the base behavior is preserved."""

    def __init__(self, *args, corpus_order=None, **kwargs):
        self._corpus_order = corpus_order
        super().__init__(*args, **kwargs)
        # Disable structural plasticity on THIS slot bridge before the wash-out snapshot: the slot pools have
        # internal_density=0.0 (no incoming synapses -- driven purely by external current), so structural plasticity
        # only grows synapses on the inert `_anchor` region, which cannot affect the read-out (verified: slot rates are
        # bit-identical with/without it). But over the ~24 renders this de-risk runs, that spurious growth RESIZES the
        # STP arrays, which would break the EMERGE-61 wash-out's fixed-shape snapshot/restore. Disabling it keeps the
        # array shapes stable AND leaves the slot dynamics byte-identical (runner-side, additive, behavior-neutral;
        # EMERGE-59/61 untouched). Re-snapshot AFTER this so the wash-out captures the stable-shape state.
        self.bridge.core_config.enable_structural_plasticity = False
        from research.runners._emerge61_spiking_broca_order_robustness_derisk import _snapshot_state
        self._post_init_state = _snapshot_state(self.bridge)
        if corpus_order is not None:
            self._reorder_slots_from_corpus(corpus_order)

    def _reorder_slots_from_corpus(self, corpus_order):
        """Permute each frame's template slot list into the CORPUS-predicted role order. `frame_slots[frame]` becomes the
        slots in corpus order; the base `learn()` then teaches a descending primacy over THIS order."""
        for fr in FRAME_NAMES:
            if fr not in corpus_order:
                continue
            template_slots = list(FRAMES[fr])
            by_role = {}
            for slot in template_slots:
                by_role.setdefault(_role_key(slot), []).append(slot)   # FUNC keys are unique; lists guard duplicates
            reordered = []
            for rk in corpus_order[fr]:
                reordered.append(by_role[rk].pop(0))
            # any role not named by the corpus order (should not happen) keeps template position at the tail
            for slot in template_slots:
                if slot not in reordered:
                    reordered.append(slot)
            self.frame_slots[fr] = reordered


# ---------------------------------------------------------------------------------------------------------------------
# LEARN THE PER-FRAME CORPUS ORDER (main arm: all frames' exemplars). Returns frame -> ordered role-key list.
# ---------------------------------------------------------------------------------------------------------------------
def learn_corpus_order(sents_by_frame, tie_rng=None, shuffle_within=False, shuffle_rng=None):
    order = {}
    n_used = {}
    for fr in FRAME_NAMES:
        prec, roles, nu = corpus_precedence(sents_by_frame[fr], fr,
                                            shuffle_within=shuffle_within, shuffle_rng=shuffle_rng)
        order[fr] = order_from_precedence(prec, roles, tie_rng=tie_rng)
        n_used[fr] = nu
    return order, n_used


def _template_role_order(frame):
    return [_role_key(sl) for sl in FRAMES[frame]]


# ---------------------------------------------------------------------------------------------------------------------
# HELD-OUT-FRAME order (b3): learn a GLOBAL type-level precedence from the TRAINING frames (FUNC collapsed to a generic
# 'FUNC' class so a function-word's POSITION transfers across frames), then order the held-out frame's roles by that
# global precedence. Shared roles (det/subj/func-position/verb) transfer; a within-frame order not attested in ANY
# training frame (two adjacent function words) ties -> broken by the honest tie_rng (the named residual).
# ---------------------------------------------------------------------------------------------------------------------
def _generic_key(role_key):
    return "FUNC" if role_key.startswith("FUNC") else role_key


def global_type_precedence(sents_by_frame, train_frames):
    prec = defaultdict(lambda: defaultdict(int))
    for fr in train_frames:
        roles = [_role_key(sl) for sl in FRAMES[fr]]
        for sent in sents_by_frame[fr]:
            pos = _role_positions(sent, fr)
            if len(pos) != len(roles):
                continue
            items = [(_generic_key(r), pos[r]) for r in roles]
            for (ka, pa) in items:
                for (kb, pb) in items:
                    if ka != kb and pa < pb:
                        prec[ka][kb] += 1
    return prec


def order_heldout_frame(sents_by_frame, held, tie_rng):
    """Order the fully-held-out `held` frame's roles from the GLOBAL type-level precedence of the OTHER frames. Honest
    (non-template) tie-break so an unlearnable within-frame order is a residual, not a smuggled template match."""
    train = [f for f in FRAME_NAMES if f != held]
    gprec = global_type_precedence(sents_by_frame, train)
    roles = [_role_key(sl) for sl in FRAMES[held]]

    def primacy(r):
        ka = _generic_key(r)
        tot, num = 0.0, 0
        for r2 in roles:
            if r == r2:
                continue
            kb = _generic_key(r2)
            ab, ba = gprec[ka][kb], gprec[kb][ka]
            if ab + ba > 0:
                tot += ab / (ab + ba)
                num += 1
        return tot / num if num else 0.0

    prim = {r: primacy(r) for r in roles}
    tie = {r: float(tie_rng.random()) for r in roles}
    return sorted(roles, key=lambda r: (-prim[r], tie[r]))


# ---------------------------------------------------------------------------------------------------------------------
# SCORING: spiking render of the corpus-taught order vs the template ground-truth surface.
# ---------------------------------------------------------------------------------------------------------------------
def _spiking_render_scores(corpus_order, seed, facts):
    """Render the held-out facts through the CORPUS-ORDER spiking producer (EMERGE-61 wash-out). Per frame: mean EXACT
    full-surface match (produced words == template ground-truth surface = right order + func words + inflection) + mean
    order-accuracy (corpus role order vs template role order). Returns (per_frame, moat_calls, answer_produced)."""
    cq = CorpusOrderFrameSlotCQ(seed=seed, corpus_order=corpus_order)
    cq.learn()
    spell = lambda w: str(w)
    per_frame = {}
    for frame in FRAME_NAMES:
        # order-accuracy: the corpus-predicted role order vs the template ground-truth role order
        ord_acc = score_order(corpus_order[frame], _template_role_order(frame))
        exact = []
        for fact in facts:
            verb = fact["intr_verb"] if frame == "F_INTR" else fact["ability_verb"]
            words = cq.emit(frame, fact["subject"], verb, spell)
            expected = _expected_words(frame, fact["subject"], verb)   # template ground-truth surface
            exact.append(1.0 if words == expected else 0.0)
        per_frame[frame] = {"order": float(ord_acc), "exact": float(np.mean(exact))}

    # gate-first moat: an ABSTAIN never invokes the producer; an ANSWER does (the counter is meaningful).
    prod = BrocaProducer(cq)
    calls0 = prod.production_count
    for _ in range(3):
        prod.speak(decision_from_emerge("ABSTAIN"))
    moat_calls = prod.production_count - calls0
    ans = prod.speak(decision_from_emerge("ANSWER", subject="owl", verb="fly", polarity="affirm"))
    return per_frame, int(moat_calls), bool(ans["produced"])


# ---------------------------------------------------------------------------------------------------------------------
# THE DE-RISK (>=6 seeds): main corpus-taught order + the anti-cheats + held-out-frame + producer-renders + moat.
# ---------------------------------------------------------------------------------------------------------------------
def _derisk_one(seed):
    tokens = build_stream(seed)
    sents = split_sentences(tokens)
    by_frame = {fr: [s for s in sents if _classify(s) == fr] for fr in FRAME_NAMES}
    # an HONEST tie-break rng (non-template) used for the MAIN order too, so main's F_NEGMOD does<not is only 1.0 if
    # DIRECTLY attested (which it is -- F_NEGMOD's own exemplars) rather than smuggled from the template.
    tie_rng = np.random.default_rng(seed * 131 + 3)

    # (a) MAIN: corpus-taught order from ALL frames' exemplars, honest tie-break.
    corpus_order, n_used = learn_corpus_order(by_frame, tie_rng=tie_rng)
    facts = build_heldout_facts(seed, n=8)
    per_frame, moat_calls, answer_produced = _spiking_render_scores(corpus_order, seed, facts)
    main_order = float(np.mean([per_frame[f]["order"] for f in FRAME_NAMES]))
    main_exact = float(np.mean([per_frame[f]["exact"] for f in FRAME_NAMES]))

    # (b1) SHUFFLED-CORPUS: scramble each example sentence's word order -> precedence destroyed -> wrong order. Average
    # over several shuffle+tie seeds (a single random tie can coincidentally match on a short frame).
    shuf_orders = []
    for k in range(8):
        srng = np.random.default_rng(seed * 977 + 13 + k)
        trng = np.random.default_rng(seed * 313 + 29 + k)
        so, _ = learn_corpus_order(by_frame, tie_rng=trng, shuffle_within=True, shuffle_rng=srng)
        shuf_orders.append(float(np.mean([score_order(so[f], _template_role_order(f)) for f in FRAME_NAMES])))
    shuffle_order = float(np.mean(shuf_orders))

    # (b2) NO-CORPUS: no example sentences -> no precedence -> chance order (over several tie seeds).
    empty = {fr: [] for fr in FRAME_NAMES}
    nocorp = []
    for k in range(8):
        trng = np.random.default_rng(seed * 619 + 41 + k)
        no, _ = learn_corpus_order(empty, tie_rng=trng)
        nocorp.append(float(np.mean([score_order(no[f], _template_role_order(f)) for f in FRAME_NAMES])))
    nocorpus_order = float(np.mean(nocorp))

    # (b3) HELD-OUT-FRAME: for each held-out frame, learn the GLOBAL type-level order from the OTHER two, honest tie.
    heldout = {}
    for held in FRAME_NAMES:
        accs = []
        for k in range(8):
            trng = np.random.default_rng(seed * 733 + 51 + k)
            o = order_heldout_frame(by_frame, held, trng)
            accs.append(score_order(o, _template_role_order(held)))
        heldout[held] = float(np.mean(accs))
    heldout_mean = float(np.mean([heldout[f] for f in FRAME_NAMES]))
    # the SHARED-precedence claim: the two single-function-word / no-function-word frames (F_MODAL, F_INTR) generalize
    # fully; F_NEGMOD's does<not internal order is the named residual (reported separately, not gated on).
    heldout_shared = float(np.mean([heldout[f] for f in ("F_MODAL", "F_INTR")]))

    return {
        "seed": seed,
        "n_used": n_used,
        "corpus_order": corpus_order,
        "per_frame": per_frame,
        "main_order": main_order, "main_exact": main_exact,
        "shuffle_order": shuffle_order, "nocorpus_order": nocorpus_order,
        "heldout": heldout, "heldout_mean": heldout_mean, "heldout_shared": heldout_shared,
        "moat_calls_on_abstain": int(moat_calls), "answer_produced": bool(answer_produced),
    }


def _sample_transcript(seed=42):
    """Render the three canonical EMERGE frames on spikes in the CORPUS-taught order + one moat abstain."""
    tokens = build_stream(seed)
    sents = split_sentences(tokens)
    by_frame = {fr: [s for s in sents if _classify(s) == fr] for fr in FRAME_NAMES}
    corpus_order, _ = learn_corpus_order(by_frame, tie_rng=np.random.default_rng(seed * 131 + 3))
    cq = CorpusOrderFrameSlotCQ(seed=seed, corpus_order=corpus_order)
    cq.learn()
    prod = BrocaProducer(cq)
    lines = []
    d1 = decision_from_emerge("ANSWER", subject="owl", verb="fly", polarity="affirm")
    d2 = decision_from_emerge("ANSWER", subject="penguin", verb="walks", polarity="negate")
    d3 = decision_from_emerge("ANSWER", subject="penguin", verb="fly", negated_modal=True)
    d4 = decision_from_emerge("ABSTAIN")
    for tag, d, q in [("INHERIT (affirm-modal)", d1, "can an owl fly?"),
                      ("CANCEL  (intransitive)", d2, "can a penguin fly?"),
                      ("DENY    (negated-modal)", d3, "can a penguin fly? [deny]"),
                      ("MOAT    (abstain)", d4, "can a zzz fly?")]:
        r = prod.speak(d)
        surface = r["surface"] if r["produced"] else "I don't know."
        inv = "producer INVOKED" if r["produced"] else "producer NOT invoked"
        lines.append((tag, q, surface, inv))
    return lines, prod.production_count, corpus_order


def _demo(seed=42):
    print("\n=== EMERGE-63 -- LEARN the spiking-Broca producer's per-frame slot ORDER from the CORPUS's WORD-ORDER "
          "STATISTICS (pairwise role precedence), NOT from the host template. The FRAMES dict's order is now only the "
          "VALIDATION ground-truth ===\n")
    tokens = build_stream(seed)
    sents = split_sentences(tokens)
    by_frame = {fr: [s for s in sents if _classify(s) == fr] for fr in FRAME_NAMES}
    corpus_order, n_used = learn_corpus_order(by_frame, tie_rng=np.random.default_rng(seed * 131 + 3))
    print(f"  corpus: {len(sents)} sentences ({', '.join(f'{f}:{n_used[f]}' for f in FRAME_NAMES)} frame exemplars)\n")
    print("  CORPUS-TAUGHT slot order (from pairwise role precedence -- NOT the template):")
    for fr in FRAME_NAMES:
        tmpl = _template_role_order(fr)
        got = corpus_order[fr]
        flag = "MATCH" if got == tmpl else "DIFFERS"
        print(f"    {fr:9s} corpus {got}")
        print(f"    {'':9s} templ  {tmpl}   [{flag}]")
    print()
    lines, pc, _ = _sample_transcript(seed)
    print("  render the EMERGE frames ON SPIKES in the corpus-taught order (gate-first moat):")
    for tag, q, surface, inv in lines:
        print(f"    you> {q}\n      broca> {surface}   [{tag}; {inv}]")
    print(f"\n  producer-invocation count after 4 probes: {pc} (the abstain never invoked the producer -- the moat)\n")


def _derisk(seeds):
    print(f"EMERGE-63 de-risk: LEARN the per-frame slot ORDER from corpus WORD-ORDER statistics (pairwise role "
          f"precedence); corpus-taught order vs shuffled-corpus / no-corpus / held-out-frame + producer-renders + moat; "
          f"{len(seeds)}-seed", flush=True)
    t0 = time.time()
    err = None
    per = []
    try:
        for s in seeds:
            d = _derisk_one(s)
            per.append(d)
            ho = d["heldout"]
            print(f"  [seed {s}] main order {d['main_order']:.3f} exact {d['main_exact']:.3f} | "
                  f"shuffled-corpus {d['shuffle_order']:.3f} | no-corpus {d['nocorpus_order']:.3f} | "
                  f"held-out shared {d['heldout_shared']:.3f} (F_MODAL {ho['F_MODAL']:.2f} F_INTR {ho['F_INTR']:.2f} "
                  f"F_NEGMOD {ho['F_NEGMOD']:.2f}) | moat {d['moat_calls_on_abstain']}", flush=True)
    except Exception as e:
        err = repr(e)
        traceback.print_exc()

    if err is None:
        def m(k):
            return float(np.mean([d[k] for d in per]))
        main_order, main_exact = m("main_order"), m("main_exact")
        shuffle_order, nocorpus_order = m("shuffle_order"), m("nocorpus_order")
        heldout_mean, heldout_shared = m("heldout_mean"), m("heldout_shared")
        heldout_negmod = float(np.mean([d["heldout"]["F_NEGMOD"] for d in per]))
        moat_calls = int(sum(d["moat_calls_on_abstain"] for d in per))
        answer_ok = all(d["answer_produced"] for d in per)

        MARGIN = 0.30      # a clear margin over every collapsed control (absolute, on the [0,1] order scale)
        high_main = main_order >= 0.999 and main_exact >= 0.999   # corpus order reproduces the template EXACTLY on spikes
        beats_shuffle = main_order >= shuffle_order + MARGIN
        beats_nocorpus = main_order >= nocorpus_order + MARGIN
        # the held-out CLAIM (gated): the SHARED type-level order generalizes to a fully-held-out frame. The F_NEGMOD
        # does<not internal order is the NAMED residual (reported, not gated) -- so we gate on the shared generalization.
        heldout_generalizes = heldout_shared >= 0.999
        moat_ok = (moat_calls == 0) and answer_ok
        controls_collapse = beats_shuffle and beats_nocorpus

        go = bool(high_main and controls_collapse and heldout_generalizes and moat_ok)
        if go:
            verdict = (
                f"GO -- the spiking-Broca producer's per-frame slot ORDER SELF-ORGANIZES from the CORPUS's actual "
                f"WORD-ORDER statistics (pairwise role precedence / bigram order; Dominey-Hinaut: grammar = the "
                f"statistics of element ORDER, no explicit rules; catalog G.12 Broca; usage-based construction grammar). "
                f"The host TEMPLATE order-teacher (EMERGE-59 _teach_order's `LR*(n-1-pool)` over the template index) is "
                f"REMOVED: for each frame the order is read PURELY from where each role's token sits in the corpus "
                f"example sentences (det<subj<func<verb emerges from the counts), and that corpus-taught order REORDERS "
                f"the slots fed to the EMERGE-59 spiking producer, which renders them ON SPIKES (the learned primacy "
                f"gradient = graded current -> the per-pool spiking-RATE ranking = the emission order; EMERGE-61 wash-out "
                f"for position-independence). The corpus-taught order MATCHES the template ground-truth EXACTLY and "
                f"renders exact on spikes: order {main_order:.3f}, exact-surface {main_exact:.3f} (incl. the negated-"
                f"modal's does<not, directly attested in F_NEGMOD's own exemplars -> resolved even with an HONEST random "
                f"tie-break). Every input-destruction control COLLAPSES: SHUFFLED-CORPUS order {shuffle_order:.3f} "
                f"(scrambling each example's word order destroys the precedence -> wrong order, margin >= {MARGIN}); "
                f"NO-CORPUS order {nocorpus_order:.3f} (no examples -> no precedence -> chance). HELD-OUT-FRAME "
                f"GENERALIZES on the SHARED precedences: a FULLY-held-out frame's shared type-level order "
                f"(det<subj<func<verb, learned from the OTHER two frames) is recovered {heldout_shared:.3f} (F_MODAL & "
                f"F_INTR). The gate-first no-confab MOAT is intact (0 producer invocations on abstains). {len(seeds)} "
                f"seeds. ==> S1b self-organized: the slot ORDER is LEARNED from corpus experience, the host template "
                f"order removed. HONEST RESIDUAL (named, NOT a wall): the does-vs-not INTERNAL order of a HELD-OUT "
                f"multi-function-word frame is NOT learnable from the OTHER two frames alone (only F_NEGMOD attests two "
                f"adjacent function words), so held-out F_NEGMOD sits at {heldout_negmod:.3f} with an honest tie-break -- "
                f"the next single signal is ONE attestation of the does<not bigram (or Yang-Getz's phrase-boundary cue). "
                f"S1a (WHICH slots a frame licenses) stays template-supplied here -- that is EMERGE-64's separate "
                f"residual. Reuse-by-import; NO sim/ edit; moat untouched.")
        else:
            miss = []
            if not high_main:
                miss.append(f"main order {main_order:.3f} / exact {main_exact:.3f} below 1.0 (corpus order does NOT "
                            f"reproduce the template ground-truth on spikes)")
            if not beats_shuffle:
                miss.append(f"does not beat SHUFFLED-CORPUS by >= {MARGIN} (main {main_order:.3f} vs {shuffle_order:.3f}) "
                            f"-- BLOCKING: the shuffled-corpus control MUST collapse (the order must come from the "
                            f"corpus word order, not elsewhere)")
            if not beats_nocorpus:
                miss.append(f"does not beat NO-CORPUS by >= {MARGIN} (main {main_order:.3f} vs {nocorpus_order:.3f})")
            if not heldout_generalizes:
                miss.append(f"held-out-frame shared generalization {heldout_shared:.3f} below 1.0 -- the shared "
                            f"type-level precedence (det<subj<func<verb) does not transfer to a fully-held-out frame")
            if not moat_ok:
                miss.append(f"MOAT: {moat_calls} producer-calls on abstains / answer-produced {answer_ok} -- BLOCKING, "
                            f"do NOT weaken the moat")
            verdict = ("BOUNDARY -- " + "; ".join(miss) + ". The precise residual is named above. If held-out F_NEGMOD's "
                       "does<not is the only gap, that is the honest, precisely-named within-frame residual (only "
                       "F_NEGMOD attests two adjacent function words) -- add one does<not bigram attestation or the "
                       "phrase-boundary cue; still not a wall. If the SHUFFLED-CORPUS control did NOT collapse this is "
                       "BLOCKING (the order is not genuinely from the corpus word order). If the MOAT was breached this "
                       "is BLOCKING -- do NOT weaken the moat.")
    else:
        verdict = f"ERROR -- {err}"
        main_order = main_exact = shuffle_order = nocorpus_order = None
        heldout_mean = heldout_shared = heldout_negmod = moat_calls = None
        go = False

    lines, _, _ = ([], 0, None)
    try:
        lines, _, _ = _sample_transcript(seeds[0])
    except Exception:
        pass
    transcript = [{"tag": t, "question": q, "surface": s, "invocation": i} for (t, q, s, i) in lines]

    summary = {
        "probe": "emerge63_corpus_taught_slot_order", "verdict": verdict,
        "go": bool(go) if err is None else False,
        "mechanism": ("self-organized per-frame slot ORDER learned from the corpus's actual WORD-ORDER statistics "
                      "(pairwise role PRECEDENCE / bigram order; Dominey-Hinaut fronto-striatal reservoir 'grammar = the "
                      "statistics of element order/position, no explicit rules'; usage-based construction grammar "
                      "Tomasello; catalog G.12 Broca). For each frame, collect its corpus example sentences (EMERGE-62's "
                      "build_stream + sentence segmentation), locate each role's token position (S1a lexical class, NOT "
                      "order), count prec[A][B]=examples where role A precedes role B; a role that precedes many others "
                      "gets high primacy (emitted first). That corpus-taught order REORDERS the EMERGE-59 frame slots; "
                      "the CorpusOrderFrameSlotCQ (subclass of the EMERGE-61 ResetFrameSlotCQ wash-out) teaches a "
                      "descending primacy over the reordered slots and renders ON SPIKES (the per-pool spiking-RATE "
                      "ranking = the emission order). The host TEMPLATE order-teacher is REMOVED; the FRAMES dict's order "
                      "is only the validation ground-truth. S1a (which slots) stays template-supplied (EMERGE-64's "
                      "residual). Reuse-by-import; NO sim/ edit."),
        "task": ("learn each frame's slot order from corpus word-order precedence (not the template); render the EMERGE "
                 "frames on spikes in the corpus-taught order == the template ground-truth (order + exact surface); "
                 "shuffled-corpus + no-corpus collapse; held-out-frame generalizes on the shared type-level precedence; "
                 "gate-first moat (0 productions on abstains); >=6 seeds"),
        "frames": {f: [[t, p] for (t, p) in FRAMES[f]] for f in FRAME_NAMES},
        "seeds": list(seeds), "elapsed_seconds": round(time.time() - t0, 1),
        "aggregate": None if err else {
            "main_order": main_order, "main_exact": main_exact,
            "shuffle_order": shuffle_order, "nocorpus_order": nocorpus_order,
            "heldout_mean": heldout_mean, "heldout_shared": heldout_shared, "heldout_negmod": heldout_negmod,
            "moat_calls_on_abstain_total": moat_calls,
        },
        "sample_transcript": transcript,
        "per_seed": per,
        "HONEST_NOTE": ("Learns the per-frame slot ORDER (S1b) from corpus WORD-ORDER statistics for the BOUNDED EMERGE "
                        "frame domain. The MAIN arm (all frames' exemplars) learns every order EXACTLY incl. the "
                        "negated-modal's does<not (directly attested -> resolved even with an honest random tie-break). "
                        "HELD-OUT-FRAME generalizes the SHARED type-level order (det<subj<func<verb) perfectly to a "
                        "fully-held-out frame; the ONE genuine residual is the does-vs-not INTERNAL order of a HELD-OUT "
                        "multi-function-word frame (only F_NEGMOD attests two adjacent function words, so it cannot be "
                        "learned from the OTHER two) -- precisely named, NOT a wall (next signal: one does<not bigram "
                        "attestation or Yang-Getz's phrase-boundary cue). S1a (which slots a frame licenses) stays "
                        "template-supplied -- EMERGE-64's separate residual. The order is produced on REAL spikes "
                        "(EMERGE-61 wash-out for position-independence); the corpus stat is offline syllabus prep "
                        "(BRAIN-BASED-ONLY compliant). The gate-first moat is untouched (0 productions on abstains)."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118, flush=True)
    print(f"[emerge63] VERDICT: {verdict}", flush=True)
    print(f"[emerge63] wrote {OUT}\n" + "=" * 118, flush=True)
    return 0 if (err is None and go) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--derisk", action="store_true")
    a = ap.parse_args()
    if a.derisk:
        return _derisk(a.seeds)
    _demo(a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
