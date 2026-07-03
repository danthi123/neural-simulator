"""EMERGE-76 -- CLOSE the EMERGE-63/64/65 HELD-OUT DISTINCTIVE-SLOT residual: ONE attestation of a fully-held-out
frame's OWN distinctive element (its function word / inflection / internal bigram order) SUFFICES to recover that
frame's distinctive slot + order -- i.e. the residual is a SINGLE-EXEMPLAR *DATA* residual (one-shot / fast-mapping),
NOT a mechanism wall.

THE RESIDUAL THIS CLOSES (named across EMERGE-63/64/65). When a construction frame is FULLY held out of the corpus,
its SHARED backbone (det<subj<verb) still generalizes from the OTHER frames (the EMERGE-63/64 gated claim), but its
DISTINCTIVE slots are NOT recoverable from the other frames alone -- because only that frame attests them:
  * F_MODAL   distinctive = the `can` FUNC slot (+ its position: det<subj<can<verb).            [EMERGE-64 residual]
  * F_NEGMOD  distinctive = the `does`/`not` FUNC slots + their INTERNAL does<not order.          [EMERGE-63/64 residual]
  * F_INTR    distinctive = the `3sg` verb inflection (walks).                                    [EMERGE-64 residual]
EMERGE-63/64/65 honestly NAMED this as the next single signal: "ONE attestation of the held-out frame's own function
word / inflection / bigram suffices to recover its distinctive slot" (see `_emerge63:52-56`, `_emerge64:64-70`,
`_emerge65:58-66`). EMERGE-76 DE-RISKS exactly that claim.

THE MECHANISM (one-shot / fast-mapping; hippocampal one-shot encoding -- catalog D.03 Marr autoassociator / D.13
pattern completion, Carey-Bartlett fast-mapping, McClelland-McNaughton-O'Reilly CLS "novel schema-consistent items
are learned in ONE exposure"; Dominey-Hinaut grammar-as-order-statistics = the ORDER of a construction's elements is
read from as few as ONE well-formed exemplar once the SHARED backbone schema exists). The held-out frame's SHARED
backbone (det<subj<verb) is already a learned schema (from the OTHER frames); a SINGLE well-formed attestation of the
held-out frame slots its DISTINCTIVE element into that schema in one shot -- the distinctive FUNC slot's IDENTITY +
POSITION (or the verb's inflection, or the does<not internal order) is fixed by that one exemplar's WORD ORDER.

DE-RISK (6 seeds 42/43/44/100/101/102, CPU/numpy). Change ONE variable per rung; gate before scaling.
  For each held-out frame (F_MODAL / F_NEGMOD / F_INTR), build the corpus = the OTHER two frames' exemplars (the
  EMERGE-63/64 held-out baseline) + K attestations of the held-out frame's OWN canonical sentence, and MINE + ORDER
  the held-out frame from THAT corpus (using its now-attested distinctive element), then check its DISTINCTIVE slot +
  order are recovered EXACTLY (inventory slots_match == ground-truth AND corpus order == template order), and render
  it EXACT on spikes (EMERGE-59/63/64 producer + EMERGE-61 wash-out). The mining `min_count` is lowered to the
  attestation regime K (a construction attested K times is confidently mined) -- reported transparently; the number of
  attestations needed is THE finding.
  (a) MAIN -- K=1 (single attestation): does ONE attestation recover the held-out frame's distinctive slot + order +
      exact spiking render? Report the minimum K in {1,3,5} that recovers each frame (the single-vs-few-exemplar map).
  Anti-cheats that MUST COLLAPSE (input-destruction + hold-out, project control-validity methodology -- NOT a fixed-
  random control):
  (b1) ZERO-ATTESTATION  -- K=0 attestations (the held-out baseline): the distinctive slot/order STAYS at the EMERGE-
       63/64 residual (NOT recovered). This proves the single attestation is LOAD-BEARING (the recovery is not smuggled
       from the OTHER frames -- exactly the EMERGE-63/64 held-out residual reproduced here as the zero control).
  (b2) PERMUTED-ATTESTATION -- add K attestations but SHUFFLE each attestation's word order first: the distinctive
       ORDER is destroyed -> the recovery COLLAPSES (the distinctive slot's POSITION / the does<not internal order
       must come from the attestation's WORD ORDER, not merely its token presence). The decisive input-destruction
       control.
  (c) the PRODUCER renders the recovered held-out frame ON SPIKES AND the gate-first no-confab MOAT holds (0 producer
      invocations on abstains).
GO bar: K=1 (or a small K, reported) recovers the held-out frame's distinctive slot + order + exact spiking render
with a clear margin over BOTH the zero-attestation residual AND the permuted-attestation collapse, for every held-out
frame; moat intact; 6-seed. If ONE attestation recovers -> GO (the residual is a single-exemplar DATA residual, not a
wall). If it needs many attestations, or a genuine mechanism gap remains, name it as an HONEST BOUNDARY (do NOT force
a GO).

HONEST SCOPE. This closes the EMERGE-63/64/65 named held-out DISTINCTIVE-slot residual for the BOUNDED EMERGE frame
domain by showing it is a DATA (single-exemplar) residual: once the SHARED backbone schema exists (from the other
frames), ONE well-formed attestation of the held-out frame fixes its distinctive element (function word + position /
inflection / does<not internal order) in one shot. It does NOT make the domain open-ended (open prose R4 is the
separate deferred wall). The corpus/attestation is offline syllabus prep (BRAIN-BASED-ONLY compliant -- like rendering
a retinal image the neural retina reads); the recovered structure is rendered on REAL spikes (EMERGE-61 wash-out).
Reuse-by-import; NO `sim/` edit; the gate-first no-confab moat is untouched (0 productions on abstains). Cites
EMERGE-63/64/65; one-shot/fast-mapping biology (Carey-Bartlett, McClelland-McNaughton-O'Reilly CLS, catalog D.03/D.13);
Dominey-Hinaut (grammar = the statistics of element order, learnable from few well-formed exemplars once a schema
exists).

Run:
  SIM_BACKEND=numpy python -m research.runners._emerge76_heldout_one_attestation_derisk --demo
  SIM_BACKEND=numpy python -m research.runners._emerge76_heldout_one_attestation_derisk --derisk
  SIM_BACKEND=numpy python -m research.runners._emerge76_heldout_one_attestation_derisk --derisk --seeds 42 43 44 100 101 102
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
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners.song_g1_core import score_order  # noqa: E402
# Reuse-by-import ONLY -- NO sim/ edit, NO reinvention. The corpus stream + segmentation + discovery; the miner + the
# spiking producer; the corpus-order learner; the frames + producer + moat.
from research.runners._emerge62_discover_function_words_derisk import (  # noqa: E402
    build_stream, compute_stats, discover_closed_class, SENT_PERIOD, _SUBJECTS,
)
from research.runners._emerge59_spiking_broca_frame_slots_derisk import (  # noqa: E402
    FRAMES, FRAME_NAMES, DET, SUBJ, FUNC, VERB, BrocaProducer, decision_from_emerge,
    build_heldout_facts, _expected_words,
)
from research.runners._emerge63_corpus_taught_slot_order_derisk import (  # noqa: E402
    split_sentences, learn_corpus_order, _template_role_order, _role_key,
)
from research.runners._emerge64_mine_slot_inventory_derisk import (  # noqa: E402
    mine_inventory, match_inventory_to_frames, label_sentence, _slot_signature, _frame_signature,
    _frame_groundtruth_slots, _mined_to_emerge59_slots, MinedInventoryFrameSlotCQ, _verb_inflection,
)
from research.runners._emerge57_ra_refinetune_emerge_frames_derisk import emerge_v3  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_emerge76_heldout_one_attestation.json"

# READER-CONSISTENT attestation verbs. An attestation is a WELL-FORMED exemplar of the construction -- it must be one
# the EMERGE-64 miner can PARSE (else it is not a valid attestation of that construction). EMERGE-64's morphology reader
# `_verb_inflection` reads a verb's inflection by stripping a SINGLE trailing -s and checking the stem against its
# content-verb lexicon (`_VERB_SET`): so a verb whose 3sg surface is `stem+es` (perch->perches, EMERGE-59's renderer
# emerge_v3 handles -es) OR whose stem is absent from `_VERB_SET` (lurk/wait/sit/sleep) is mis-read as `bare` -> the
# F_INTR (3sg) attestation is mis-typed. This is an INHERITED EMERGE-64 morphology-reader limit (single-`s` strip +
# `_VERB_SET` coverage), NOT a one-shot mechanism gap. A canonical F_INTR attestation therefore uses a verb whose 3sg
# surface the reader reads BACK as 3sg -- computed here (reader round-trip), reported transparently as the named
# sub-residual. F_MODAL/F_NEGMOD use BARE verbs, which the reader always reads correctly, so ANY verb is a valid
# attestation there. Below, `_readable_intr_verbs` = the EMERGE-59 intransitive pool whose 3sg the reader parses.
from research.runners._emerge59_spiking_broca_frame_slots_derisk import _ABILITY, _INTR3SG  # noqa: E402


def _readable_3sg(verb3sg):
    """True iff EMERGE-64's `_verb_inflection` reads this 3sg SURFACE back as '3sg' (single-`s` strip -> stem in the
    reader's content-verb lexicon). A verb whose 3sg the reader cannot parse is not a well-formed F_INTR attestation."""
    return _verb_inflection(verb3sg) == "3sg"


# the EMERGE-59 intransitive pool (already-3sg surfaces) restricted to those the reader parses -> canonical F_INTR
# attestation verbs (a well-formed exemplar the miner can read). The rest are the named morphology-reader sub-residual.
_readable_intr_verbs = [v for v in _INTR3SG if _readable_3sg(v)]
_unreadable_intr_verbs = [v for v in _INTR3SG if not _readable_3sg(v)]

# The K-rung (change ONE variable per rung): how many attestations of the held-out frame's own canonical sentence to
# add. K=0 is the zero-attestation residual (baseline); K=1 is the single-attestation claim; K in {1,3,5} maps the
# minimum recovery threshold. `_ATTEST_RUNGS` is the ordered rung sweep.
_ATTEST_RUNGS = [0, 1, 3, 5]


# ---------------------------------------------------------------------------------------------------------------------
# BUILD the held-out corpus: the OTHER two frames' exemplars (the EMERGE-63/64 held-out baseline) + K attestations of
# the held-out frame's OWN canonical sentence. The attestations use the SAME lexical domain (a subject + a verb drawn
# from the EMERGE lexicons) rendered in the held-out frame's ground-truth surface (correct word order) -- one
# well-formed exemplar of the held-out construction, exactly what the held-out arm was DENIED in EMERGE-63/64.
# ---------------------------------------------------------------------------------------------------------------------
def _held_frame_of_sentence(sent, closed):
    """Return the frame id whose ground-truth SIGNATURE this labelled sentence matches, or None. Used to WITHHOLD the
    held frame's exemplars from the base corpus (a validation-time split by the sentence's own labelled signature -- the
    frame id is NOT smuggled into the miner, per EMERGE-64)."""
    slots = label_sentence(sent, closed)
    if slots is None:
        return None
    sig = _slot_signature(slots)
    for fr in FRAME_NAMES:
        if sig == _frame_signature(fr):
            return fr
    return None


def _canonical_attestation(frame, subject, verb3sg_or_bare):
    """One well-formed CANONICAL surface sentence of `frame` (the held-out frame's OWN distinctive element rendered in
    correct word order). Uses `_expected_words` (the ground-truth surface = right slots + order + function words +
    inflection) so the attestation is a genuine exemplar of the held-out construction. For F_INTR the verb is rendered
    3sg by `_expected_words`; for F_MODAL/F_NEGMOD it is bare. Returns a token list (no period)."""
    return list(_expected_words(frame, subject, verb3sg_or_bare))


def _attestation_sentences(held, seed, k, shuffle_within=False, shuffle_rng=None, unreadable_intr=False):
    """K attestation sentences of the held-out frame, drawn from the EMERGE-59 frame lexicon -- the SAME lexical domain
    as the EMERGE-63/64/65 residual this closes (distinct subjects/verbs so the attestation is not a single memorized
    string but K well-formed exemplars of the construction). BARE verbs (`_ABILITY`) for F_MODAL/F_NEGMOD (the reader
    always reads bare correctly -> any verb is a valid attestation there); READER-CONSISTENT 3sg verbs
    (`_readable_intr_verbs`) for F_INTR (a WELL-FORMED exemplar the EMERGE-64 morphology reader parses back as 3sg --
    see the module note). `shuffle_within` (the PERMUTED-ATTESTATION anti-cheat) scrambles each attestation's word order
    -> destroys the distinctive ORDER. `unreadable_intr` (transparency probe): use the reader-UNREADABLE F_INTR verbs
    (lurk/wait/sit/sleep) to expose the inherited morphology-reader sub-residual honestly (reported, not gated)."""
    if k <= 0:
        return []
    rng = np.random.default_rng(seed * 619 + 41 + (1 if shuffle_within else 0) + (7 if unreadable_intr else 0))
    if held == "F_INTR":
        verbs = _unreadable_intr_verbs if unreadable_intr else _readable_intr_verbs
    else:
        verbs = _ABILITY                                  # bare verbs (F_MODAL / F_NEGMOD) -- reader-agnostic
    out = []
    for _ in range(k):
        s = str(rng.choice(_SUBJECTS))
        v = str(rng.choice(verbs))
        snt = _canonical_attestation(held, s, v)
        if shuffle_within:
            snt = list(snt)
            shuffle_rng.shuffle(snt)
        out.append(snt)
    return out


def build_heldout_corpus_sentences(base_sents, closed, held, seed, k, shuffle_attest=False, shuffle_rng=None,
                                   unreadable_intr=False):
    """The held-out corpus SENTENCES: all base sentences EXCEPT the held frame's exemplars, PLUS k attestations of the
    held frame (optionally word-order-shuffled for the permuted-attestation control). Returns a list of token lists."""
    kept = [list(s) for s in base_sents if _held_frame_of_sentence(s, closed) != held]
    att = _attestation_sentences(held, seed, k, shuffle_within=shuffle_attest, shuffle_rng=shuffle_rng,
                                 unreadable_intr=unreadable_intr)
    return kept + att


# ---------------------------------------------------------------------------------------------------------------------
# RECOVER the held-out frame's DISTINCTIVE slot + order from the held-out corpus (base OTHER frames + k attestations).
# Mine the inventory (min_count = the attestation regime so a K-attested construction is confidently mined) + learn the
# order over the mined constructions; check the HELD frame's inventory + order are recovered EXACTLY.
# ---------------------------------------------------------------------------------------------------------------------
def _mine_with_attestation_regime(sents, closed, k):
    """Mine the inventory with `min_count` set to the attestation regime: a construction attested at least K times (K>=1)
    is confidently mined; with K=0 the held frame is simply absent from the corpus (its residual). min_dominance stays
    at the committed 0.80 -- the attestation's canonical (unshuffled) order dominates completely; a SHUFFLED attestation
    scatters below it (the permuted-attestation collapse). Transparent: the number of attestations K == min_count."""
    min_count = max(1, k)
    return mine_inventory(sents, closed, min_count=min_count, min_dominance=0.80)


def _held_inventory_recovered(inventory, held):
    """Does the mined inventory recover the held-out frame's inventory EXACTLY (found AND slots_match == ground-truth)?
    This is the DISTINCTIVE-SLOT recovery: the held frame's function-word slots (can / does,not) or 3sg inflection are
    part of its ground-truth slot list, so slots_match==True means the distinctive slot is recovered."""
    m = match_inventory_to_frames(inventory)
    return bool(m[held]["found"] and m[held]["slots_match"]), m


def _held_order_recovered(sents, closed, held, seed):
    """Learn the corpus slot-ORDER (EMERGE-63 pairwise role precedence) for the held-out frame from THIS corpus (base
    OTHER frames + the k attestations), and score it vs the template ground-truth order. Recovers the distinctive
    INTERNAL order (F_NEGMOD's does<not; F_MODAL's can position) ONLY if the held frame's own exemplars are attested --
    the OTHER frames alone cannot (the EMERGE-63 residual). Honest random tie-break (no template smuggling)."""
    # group the held frame's labelled exemplars in THIS corpus by its ground-truth signature
    held_sents = [list(s) for s in sents if _held_frame_of_sentence(s, closed) == held]
    by_frame = {fr: [] for fr in FRAME_NAMES}
    by_frame[held] = held_sents
    accs = []
    for j in range(8):
        trng = np.random.default_rng(seed * 733 + 51 + j)
        order, _n = learn_corpus_order(by_frame, tie_rng=trng)
        accs.append(score_order(order[held], _template_role_order(held)))
    return float(np.mean(accs)), (held_sents[0] if held_sents else None)


def _spiking_render_held(inventory, held, seed, facts):
    """Render the held-out frame ON SPIKES from the MINED inventory (EMERGE-64 MinedInventoryFrameSlotCQ + EMERGE-61
    wash-out). Returns (exact, moat_calls, answer_produced). If the held frame was not mined -> exact 0."""
    m = match_inventory_to_frames(inventory)
    mined_slots = {}
    for fr in FRAME_NAMES:
        if m[fr]["found"]:
            mined_slots[fr] = _mined_to_emerge59_slots([tuple(x) for x in m[fr]["mined_slots"]])
    cq = MinedInventoryFrameSlotCQ(seed=seed, mined_slots=mined_slots)
    cq.learn()
    spell = lambda w: str(w)
    if held not in mined_slots:
        exact = 0.0
    else:
        ex = []
        for fact in facts:
            verb = fact["intr_verb"] if held == "F_INTR" else fact["ability_verb"]
            words = cq.emit(held, fact["subject"], verb, spell)
            expected = _expected_words(held, fact["subject"], verb)
            ex.append(1.0 if words == expected else 0.0)
        exact = float(np.mean(ex))

    prod = BrocaProducer(cq)
    calls0 = prod.production_count
    for _ in range(3):
        prod.speak(decision_from_emerge("ABSTAIN"))
    moat_calls = prod.production_count - calls0
    ans = prod.speak(decision_from_emerge("ANSWER", subject="owl", verb="fly", polarity="affirm"))
    return exact, int(moat_calls), bool(ans["produced"])


def recover_held_frame(base_sents, closed, held, seed, k, facts, shuffle_attest=False, shuffle_rng=None,
                       unreadable_intr=False):
    """Full recovery pass for ONE held-out frame at K attestations: build the corpus, mine + order + render the held
    frame, return the distinctive-slot recovery (inventory + order + exact spiking render)."""
    sents = build_heldout_corpus_sentences(base_sents, closed, held, seed, k,
                                            shuffle_attest=shuffle_attest, shuffle_rng=shuffle_rng,
                                            unreadable_intr=unreadable_intr)
    inventory, _sig = _mine_with_attestation_regime(sents, closed, k)
    inv_ok, _m = _held_inventory_recovered(inventory, held)
    order_acc, _ex_sent = _held_order_recovered(sents, closed, held, seed)
    order_ok = order_acc >= 0.999
    exact, moat_calls, answer_produced = _spiking_render_held(inventory, held, seed, facts)
    # the DISTINCTIVE-slot recovery: inventory (which includes the distinctive func-word / inflection slots) recovered
    # EXACTLY, its order recovered, and it renders EXACT on spikes.
    recovered = bool(inv_ok and order_ok and exact >= 0.999)
    return {
        "held": held, "k": k,
        "inventory_recovered": bool(inv_ok), "order_acc": float(order_acc), "order_recovered": bool(order_ok),
        "exact": float(exact), "recovered": recovered,
        "moat_calls_on_abstain": int(moat_calls), "answer_produced": bool(answer_produced),
    }


# ---------------------------------------------------------------------------------------------------------------------
# THE DE-RISK (>=6 seeds): for each held-out frame, sweep K in {0,1,3,5} (+ the permuted-attestation control at each
# K>=1); report the minimum K that recovers the distinctive slot; gate on: K=1 recovers with a margin over the
# zero-attestation residual (K=0) AND the permuted-attestation collapse; moat intact.
# ---------------------------------------------------------------------------------------------------------------------
def _derisk_one(seed):
    tokens = build_stream(seed)
    base_sents = split_sentences(tokens)
    words, freq, cover, _c = compute_stats(tokens)
    closed, _p, _f, _cp = discover_closed_class(words, freq, cover)
    facts = build_heldout_facts(seed, n=8)

    per_frame = {}
    for held in FRAME_NAMES:
        rungs = {}
        for k in _ATTEST_RUNGS:
            rec = recover_held_frame(base_sents, closed, held, seed, k, facts)
            rungs[k] = rec
        # PERMUTED-ATTESTATION control at each K>=1 (average over shuffle seeds): the distinctive order destroyed.
        perm = {}
        for k in [kk for kk in _ATTEST_RUNGS if kk >= 1]:
            recs = []
            for j in range(4):
                srng = np.random.default_rng(seed * 977 + 13 + j + 100 * k)
                recs.append(recover_held_frame(base_sents, closed, held, seed, k, facts,
                                               shuffle_attest=True, shuffle_rng=srng))
            perm[k] = {
                "recovered_frac": float(np.mean([1.0 if r["recovered"] else 0.0 for r in recs])),
                "exact": float(np.mean([r["exact"] for r in recs])),
                "order_acc": float(np.mean([r["order_acc"] for r in recs])),
                "inventory_recovered_frac": float(np.mean([1.0 if r["inventory_recovered"] else 0.0 for r in recs])),
            }
        # minimum K that recovers the distinctive slot (None if none in the rung set)
        min_k = None
        for k in [kk for kk in _ATTEST_RUNGS if kk >= 1]:
            if rungs[k]["recovered"]:
                min_k = k
                break
        pf = {
            "rungs": rungs, "perm": perm, "min_k_recovers": min_k,
            "zero_recovered": bool(rungs[0]["recovered"]), "zero_exact": float(rungs[0]["exact"]),
            "zero_inventory_recovered": bool(rungs[0]["inventory_recovered"]),
            "one_recovered": bool(rungs[1]["recovered"]), "one_exact": float(rungs[1]["exact"]),
            "one_order_acc": float(rungs[1]["order_acc"]),
            "perm_one_recovered_frac": perm[1]["recovered_frac"], "perm_one_exact": perm[1]["exact"],
        }
        # TRANSPARENCY PROBE (F_INTR only): the inherited EMERGE-64 morphology-reader sub-residual. A SINGLE attestation
        # whose 3sg surface the reader CANNOT parse (lurk/wait/sit/sleep -- single-`s` strip mis-reads them as bare)
        # fails to recover F_INTR -- honestly reported here (NOT gated), showing the one-shot claim is about a WELL-FORMED
        # (reader-parseable) exemplar, and the residual verb pool is the named next data signal (extend _VERB_SET or use
        # a lemmatizer). For F_MODAL/F_NEGMOD (bare verbs) this probe is N/A (the reader always reads bare).
        if held == "F_INTR" and _unreadable_intr_verbs:
            recs_bad = [recover_held_frame(base_sents, closed, held, seed, 1, facts, unreadable_intr=True)
                        for _ in range(1)]
            pf["one_unreadable_verb_recovered"] = bool(recs_bad[0]["recovered"])
            pf["one_unreadable_verb_exact"] = float(recs_bad[0]["exact"])
        per_frame[held] = pf

    # moat: any recovered rung must have 0 moat calls + produced an answer (the gate-first moat holds throughout).
    moat_calls = 0
    answer_ok = True
    for held in FRAME_NAMES:
        r1 = per_frame[held]["rungs"][1]
        moat_calls += r1["moat_calls_on_abstain"]
        answer_ok = answer_ok and r1["answer_produced"]

    return {
        "seed": seed,
        "n_closed": len(closed), "closed": sorted(closed),
        "per_frame": per_frame,
        "moat_calls_on_abstain": int(moat_calls), "answer_produced": bool(answer_ok),
    }


def _sample_transcript(seed=42):
    """Render a held-out frame (F_NEGMOD -- the hardest, its does<not internal order) ON SPIKES after ONE attestation
    recovers it, + one moat abstain. Demonstrates the one-shot recovery end-to-end."""
    tokens = build_stream(seed)
    base_sents = split_sentences(tokens)
    words, freq, cover, _c = compute_stats(tokens)
    closed, _p, _f, _cp = discover_closed_class(words, freq, cover)
    held = "F_NEGMOD"
    sents = build_heldout_corpus_sentences(base_sents, closed, held, seed, k=1)
    inventory, _sig = _mine_with_attestation_regime(sents, closed, 1)
    m = match_inventory_to_frames(inventory)
    mined_slots = {fr: _mined_to_emerge59_slots([tuple(x) for x in m[fr]["mined_slots"]])
                   for fr in FRAME_NAMES if m[fr]["found"]}
    cq = MinedInventoryFrameSlotCQ(seed=seed, mined_slots=mined_slots)
    cq.learn()
    prod = BrocaProducer(cq)
    lines = []
    d1 = decision_from_emerge("ANSWER", subject="owl", verb="fly", polarity="affirm")
    d2 = decision_from_emerge("ANSWER", subject="penguin", verb="walks", polarity="negate")
    d3 = decision_from_emerge("ANSWER", subject="penguin", verb="fly", negated_modal=True)
    d4 = decision_from_emerge("ABSTAIN")
    for tag, d, q in [("INHERIT (affirm-modal)", d1, "can an owl fly?"),
                      ("CANCEL  (intransitive)", d2, "can a penguin fly?"),
                      ("DENY    (negated-modal, held-out+1-attest)", d3, "can a penguin fly? [deny]"),
                      ("MOAT    (abstain)", d4, "can a zzz fly?")]:
        r = prod.speak(d)
        surface = r["surface"] if r["produced"] else "I don't know."
        inv = "producer INVOKED" if r["produced"] else "producer NOT invoked"
        lines.append((tag, q, surface, inv))
    return lines, prod.production_count


def _demo(seed=42):
    print("\n=== EMERGE-76 -- ONE attestation of a fully-HELD-OUT frame's OWN distinctive element (function word / "
          "inflection / does<not order) SUFFICES to recover its distinctive slot + order (one-shot / fast-mapping). "
          "Closes the EMERGE-63/64/65 held-out DISTINCTIVE-slot residual as a SINGLE-EXEMPLAR DATA residual ===\n")
    tokens = build_stream(seed)
    base_sents = split_sentences(tokens)
    words, freq, cover, _c = compute_stats(tokens)
    closed, _p, _f, _cp = discover_closed_class(words, freq, cover)
    facts = build_heldout_facts(seed, n=8)
    print(f"  corpus: {len(base_sents)} base sentences | discovered closed class: {sorted(closed)}\n")
    for held in FRAME_NAMES:
        gt = [list(x) for x in _frame_groundtruth_slots(held)]
        print(f"  HELD-OUT {held}  (ground-truth {gt})")
        for k in _ATTEST_RUNGS:
            rec = recover_held_frame(base_sents, closed, held, seed, k, facts)
            tag = "residual" if k == 0 else "attest x%d" % k
            print(f"    K={k} ({tag:9s}): inventory {'REC' if rec['inventory_recovered'] else 'no ':3s} "
                  f"order {rec['order_acc']:.2f} exact {rec['exact']:.2f} -> "
                  f"{'RECOVERED' if rec['recovered'] else 'not recovered'}")
        # permuted-attestation at K=1
        recs = []
        for j in range(4):
            srng = np.random.default_rng(seed * 977 + 13 + j + 100)
            recs.append(recover_held_frame(base_sents, closed, held, seed, 1, facts,
                                           shuffle_attest=True, shuffle_rng=srng))
        pf = float(np.mean([1.0 if r["recovered"] else 0.0 for r in recs]))
        print(f"    K=1 PERMUTED-attestation (word order shuffled): recovered-frac {pf:.2f} (must collapse)\n")
    lines, pc = _sample_transcript(seed)
    print("  render the EMERGE frames ON SPIKES after ONE attestation recovers held-out F_NEGMOD (gate-first moat):")
    for tag, q, surface, inv in lines:
        print(f"    you> {q}\n      broca> {surface}   [{tag}; {inv}]")
    print(f"\n  producer-invocation count after 4 probes: {pc} (the abstain never invoked the producer -- the moat)\n")


def _derisk(seeds):
    print(f"EMERGE-76 de-risk: ONE attestation of a fully-held-out frame's OWN distinctive element recovers its "
          f"distinctive slot + order (one-shot / fast-mapping); K=1 vs zero-attestation residual (K=0) + permuted-"
          f"attestation collapse + spiking render + moat; {len(seeds)}-seed", flush=True)
    t0 = time.time()
    err = None
    per = []
    try:
        for s in seeds:
            d = _derisk_one(s)
            per.append(d)
            for held in FRAME_NAMES:
                pf = d["per_frame"][held]
                print(f"  [seed {s}] {held:9s}: zero(K=0) rec={pf['zero_recovered']} exact={pf['zero_exact']:.2f} | "
                      f"one(K=1) rec={pf['one_recovered']} exact={pf['one_exact']:.2f} order={pf['one_order_acc']:.2f} | "
                      f"perm(K=1) rec-frac={pf['perm_one_recovered_frac']:.2f} | min-K={pf['min_k_recovers']}",
                      flush=True)
            print(f"  [seed {s}] moat {d['moat_calls_on_abstain']} answer-ok {d['answer_produced']}", flush=True)
    except Exception as e:
        err = repr(e)
        traceback.print_exc()

    if err is None:
        # aggregate per held-out frame
        def frac(held, key):
            return float(np.mean([1.0 if d["per_frame"][held][key] else 0.0 for d in per]))

        def meanf(held, key):
            return float(np.mean([d["per_frame"][held][key] for d in per]))

        agg = {}
        for held in FRAME_NAMES:
            agg[held] = {
                "zero_recovered_frac": frac(held, "zero_recovered"),
                "zero_exact": meanf(held, "zero_exact"),
                "one_recovered_frac": frac(held, "one_recovered"),
                "one_exact": meanf(held, "one_exact"),
                "one_order_acc": meanf(held, "one_order_acc"),
                "perm_one_recovered_frac": meanf(held, "perm_one_recovered_frac"),
                "perm_one_exact": float(np.mean([d["per_frame"][held]["perm"][1]["exact"] for d in per])),
                "min_k_all": sorted({d["per_frame"][held]["min_k_recovers"] for d in per},
                                    key=lambda x: (x is None, x)),
                "min_k_worst": max((d["per_frame"][held]["min_k_recovers"] or 99) for d in per),
            }
        moat_calls = int(sum(d["moat_calls_on_abstain"] for d in per))
        answer_ok = all(d["answer_produced"] for d in per)

        MARGIN = 0.30
        # GO gates:
        #  (1) K=1 recovers the distinctive slot + order + exact spiking render for EVERY held-out frame, all seeds.
        one_recovers_all = all(agg[h]["one_recovered_frac"] >= 0.999 and agg[h]["one_exact"] >= 0.999
                               for h in FRAME_NAMES)
        #  (2) the ZERO-attestation control (the held-out residual) does NOT recover the distinctive slot -- load-bearing.
        #      (F_MODAL/F_NEGMOD are missing entirely at K=0 -> exact 0; F_INTR is the only frame whose SHARED backbone
        #       fully IS its inventory -- but its 3sg inflection is the distinctive residual, so its exact still 0 when
        #       held out. We require the ZERO-attestation exact to be well below K=1 for every frame.)
        zero_is_residual = all(agg[h]["zero_exact"] <= 1.0 - MARGIN for h in FRAME_NAMES)
        zero_margin_ok = all((agg[h]["one_exact"] - agg[h]["zero_exact"]) >= MARGIN for h in FRAME_NAMES)
        #  (3) the PERMUTED-attestation control collapses -- the recovery needs the attestation's WORD ORDER.
        perm_collapses = all((agg[h]["one_exact"] - agg[h]["perm_one_exact"]) >= MARGIN for h in FRAME_NAMES)
        #  (4) the moat holds.
        moat_ok = (moat_calls == 0) and answer_ok

        go = bool(one_recovers_all and zero_is_residual and zero_margin_ok and perm_collapses and moat_ok)
        if go:
            verdict = (
                f"GO -- the EMERGE-63/64/65 HELD-OUT DISTINCTIVE-SLOT residual is a SINGLE-EXEMPLAR *DATA* residual, NOT "
                f"a mechanism wall: ONE attestation of a fully-held-out frame's OWN distinctive element recovers its "
                f"distinctive slot + order + exact spiking render (one-shot / fast-mapping). For each held-out frame the "
                f"corpus is the OTHER two frames' exemplars (the EMERGE-63/64 held-out baseline, where the distinctive "
                f"slot is NOT recoverable) + K attestations of the held frame's OWN canonical sentence; the miner + "
                f"order-learner recover the held frame's DISTINCTIVE slot (F_MODAL's `can` + position, F_NEGMOD's "
                f"`does`/`not` + does<not internal order, F_INTR's `3sg` inflection) from its now-attested exemplars, and "
                f"the EMERGE-59/63/64 spiking producer renders it EXACT on spikes (EMERGE-61 wash-out). At K=1 (a SINGLE "
                f"attestation) EVERY held-out frame recovers exactly "
                f"(F_MODAL one-exact {agg['F_MODAL']['one_exact']:.3f}, F_INTR {agg['F_INTR']['one_exact']:.3f}, "
                f"F_NEGMOD {agg['F_NEGMOD']['one_exact']:.3f}; order {agg['F_NEGMOD']['one_order_acc']:.3f} for the "
                f"does<not internal order). The controls COLLAPSE: ZERO-ATTESTATION (K=0, the held-out residual) does "
                f"NOT recover (exact F_MODAL {agg['F_MODAL']['zero_exact']:.3f} / F_INTR {agg['F_INTR']['zero_exact']:.3f}"
                f" / F_NEGMOD {agg['F_NEGMOD']['zero_exact']:.3f} -- proving the single attestation is LOAD-BEARING, the "
                f"recovery is not smuggled from the OTHER frames); PERMUTED-ATTESTATION (K=1 with the attestation's word "
                f"order shuffled) collapses (exact F_NEGMOD {agg['F_NEGMOD']['perm_one_exact']:.3f} -- the distinctive "
                f"slot's POSITION / the does<not internal order must come from the attestation's WORD ORDER, not merely "
                f"its token presence). The gate-first no-confab MOAT is intact (0 producer invocations on abstains). "
                f"{len(seeds)} seeds. ==> the residual EMERGE-63/64/65 honestly named -- 'ONE attestation of the "
                f"held-out frame's own function word / inflection / bigram suffices' -- is CONFIRMED: once the SHARED "
                f"backbone schema exists (from the other frames), a SINGLE well-formed exemplar slots the held frame's "
                f"distinctive element into that schema in one shot (Carey-Bartlett fast-mapping; McClelland-McNaughton-"
                f"O'Reilly CLS one-exposure schema-consistent encoding; catalog D.03/D.13 hippocampal one-shot; "
                f"Dominey-Hinaut grammar-as-order-statistics from few well-formed exemplars). This closes the held-out "
                f"residual for the BOUNDED EMERGE frame domain as a DATA residual; open prose (R4) is the separate "
                f"deferred wall. Reuse-by-import; NO sim/ edit; moat untouched.")
        else:
            miss = []
            if not one_recovers_all:
                bad = [h for h in FRAME_NAMES if not (agg[h]["one_recovered_frac"] >= 0.999
                                                      and agg[h]["one_exact"] >= 0.999)]
                miss.append("K=1 (single attestation) does NOT recover the distinctive slot+order+render for "
                            + ", ".join(f"{h} (one-exact {agg[h]['one_exact']:.3f}, rec-frac "
                                        f"{agg[h]['one_recovered_frac']:.2f}, min-K {agg[h]['min_k_all']})" for h in bad)
                            + " -- if a small K>1 recovers, the residual is a FEW-exemplar data residual (report the K); "
                              "if no K in {1,3,5} recovers, name the genuine mechanism gap")
            if not (zero_is_residual and zero_margin_ok):
                miss.append("ZERO-attestation control is NOT the residual (it recovered without the attestation, or the "
                            "K=1 margin over K=0 is < %.2f) -- BLOCKING: the single attestation must be LOAD-BEARING "
                            "(else the recovery is smuggled from the OTHER frames, not the attestation)" % MARGIN)
            if not perm_collapses:
                miss.append("PERMUTED-attestation did NOT collapse by >= %.2f (%s) -- BLOCKING: the recovery must come "
                            "from the attestation's WORD ORDER, not merely its token presence"
                            % (MARGIN, ", ".join(f"{h} one {agg[h]['one_exact']:.2f} vs perm "
                                                 f"{agg[h]['perm_one_exact']:.2f}" for h in FRAME_NAMES)))
            if not moat_ok:
                miss.append(f"MOAT: {moat_calls} producer-calls on abstains / answer-produced {answer_ok} -- BLOCKING, "
                            f"do NOT weaken the moat")
            verdict = ("BOUNDARY -- " + "; ".join(miss) + ". The precise residual is named above. If a SMALL K>1 "
                       "recovers each frame, the honest finding is a FEW-exemplar data residual (still not a wall -- "
                       "report the minimum K per frame). If NO K in {1,3,5} recovers a frame, name the genuine "
                       "mechanism gap (e.g. the distinctive slot's position is under-determined even by its own "
                       "exemplars). If the PERMUTED-attestation control did NOT collapse this is BLOCKING (the recovery "
                       "is not genuinely from the attestation's word order). If the MOAT was breached this is BLOCKING "
                       "-- do NOT weaken the moat. Do NOT force a GO.")
    else:
        verdict = f"ERROR -- {err}"
        agg = None
        moat_calls = None
        go = False

    lines = []
    try:
        lines, _ = _sample_transcript(seeds[0])
    except Exception:
        pass
    transcript = [{"tag": t, "question": q, "surface": s, "invocation": i} for (t, q, s, i) in lines]

    summary = {
        "probe": "emerge76_heldout_one_attestation", "verdict": verdict,
        "go": bool(go) if err is None else False,
        "mechanism": ("ONE attestation of a fully-held-out construction frame's OWN distinctive element (function word / "
                      "inflection / internal bigram order) recovers that frame's distinctive slot + order in ONE shot "
                      "(one-shot / fast-mapping). For each held-out frame the corpus = the OTHER two frames' exemplars "
                      "(the EMERGE-63/64 held-out baseline, where the distinctive slot is NOT recoverable) + K "
                      "attestations of the held frame's OWN canonical sentence; the EMERGE-64 miner + EMERGE-63 order-"
                      "learner recover the held frame's distinctive slot (F_MODAL can+position / F_NEGMOD does,not + "
                      "does<not order / F_INTR 3sg) from its now-attested exemplars, and the EMERGE-59/63/64 spiking "
                      "producer renders it EXACT on spikes (EMERGE-61 wash-out). The mining min_count == the attestation "
                      "regime K (transparent; the minimum K is the finding). The SHARED backbone schema (det<subj<verb) "
                      "is already learned from the other frames; the single attestation slots the distinctive element "
                      "into that schema (Carey-Bartlett fast-mapping; McClelland-McNaughton-O'Reilly CLS one-exposure "
                      "schema-consistent encoding; catalog D.03/D.13 hippocampal one-shot; Dominey-Hinaut grammar = the "
                      "statistics of element order, learnable from few well-formed exemplars once a schema exists). The "
                      "zero-attestation (residual) + permuted-attestation (word-order-destroyed) input-destruction "
                      "controls gate the result (project control-validity methodology). Reuse-by-import; NO sim/ edit."),
        "task": ("for each fully-held-out frame, add K attestations of its OWN canonical sentence to the OTHER-frames "
                 "corpus; recover its distinctive slot + order + exact spiking render at K=1 (single attestation) with "
                 "a margin over the zero-attestation residual (K=0) AND the permuted-attestation collapse; report the "
                 "minimum K per frame; gate-first moat (0 productions on abstains); >=6 seeds"),
        "frames_groundtruth": {f: [[t, p] for (t, p) in FRAMES[f]] for f in FRAME_NAMES},
        "attest_rungs": _ATTEST_RUNGS,
        "seeds": list(seeds), "elapsed_seconds": round(time.time() - t0, 1),
        "aggregate": agg,
        "moat_calls_on_abstain_total": moat_calls,
        "sample_transcript": transcript,
        "per_seed": per,
        "HONEST_NOTE": ("Closes the EMERGE-63/64/65 named HELD-OUT DISTINCTIVE-slot residual for the BOUNDED EMERGE "
                        "frame domain by showing it is a SINGLE-EXEMPLAR DATA residual, NOT a mechanism wall: once the "
                        "SHARED backbone schema exists (from the other frames), ONE well-formed attestation of the "
                        "held-out frame fixes its distinctive element (function word + position / inflection / does<not "
                        "internal order) in one shot. Load-bearing proof: the zero-attestation control (the held-out "
                        "residual) does NOT recover, and the permuted-attestation control (word order destroyed) "
                        "collapses -- the recovery genuinely comes from the single attestation's WORD ORDER. If a "
                        "frame needs a SMALL K>1 that is a FEW-exemplar data residual (reported, still not a wall). "
                        "This does NOT make the domain open-ended (open prose R4 is the separate deferred wall). The "
                        "corpus/attestation is offline syllabus prep (BRAIN-BASED-ONLY compliant); the recovered "
                        "structure is rendered on REAL spikes (EMERGE-61 wash-out); the gate-first moat is untouched "
                        "(0 productions on abstains). Cites EMERGE-63/64/65; one-shot/fast-mapping biology (Carey-"
                        "Bartlett, McClelland-McNaughton-O'Reilly CLS, catalog D.03/D.13); Dominey-Hinaut."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118, flush=True)
    print(f"[emerge76] VERDICT: {verdict}", flush=True)
    print(f"[emerge76] wrote {OUT}\n" + "=" * 118, flush=True)
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
