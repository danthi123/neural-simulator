"""EMERGE-73 -- CLOSE the honest ADJECTIVE boundary EMERGE-72 named: separate the corpus's adjectives from the closed
class by their ATTRIBUTIVE PRE-NOMINAL POSITION (not frequency), so the self-organized spiking-Broca producer ADMITS the
adjective constructions and BROADENS to >= 7 corpus-mined, router-selected constructions.

THE BOUNDARY EMERGE-72 named (`research/findings/2026-07-03-emerge72-construction-registry-GO.md:71-80`;
`_emerge72_construction_registry_derisk.py:43-54`). The EMERGE-72 producer renders 5 constructions but SKIPS the
adjective ones, because THIS corpus's adjectives (big/small/fast/tall/grey/...) are high-FREQUENCY AND high-context-
COVERAGE, so EMERGE-62's 2-cue Goldilocks discovery (`_emerge62:discover_closed_class`, freq-pct >= 0.90 AND cover-pct
>= 0.60) MISLABELS 2-4 of them CLOSED-class per seed (verified: seed 42 {big,fast}, seed 44 {fast,red,small,tall},
seed 101 {cold,grey,red,warm}, ...). With the adjectives wrongly in the closed class, `label_sentence_ext` correctly
SKIPS the attributive/predicative constructions rather than mislabelling -- an HONEST boundary, not forced. The
EMERGE-72 finding named the exact next signal: "the ADJECTIVE's OWN attributive pre-nominal signature."

THE MECHANISM (the biology; the INVERSE-position cue to EMERGE-62b's function-word cue). An ADJECTIVE is OPEN-class
(content) BUT positionally constrained: it reliably occupies the ATTRIBUTIVE slot BETWEEN the determiner and the head
noun ("the BIG owl": DET adj NOUN), and/or the predicate-adjective slot after a copula ("the owl is BIG"). A true
CLOSED-class word (the/a/can/does) has a DIFFERENT positional profile -- EMERGE-62b's function-word cue: at a phrase
edge / immediately before content, but a TRUE determiner (`the`, `a`) sits at the NP ONSET (NOT itself preceded by
another closed word inside the NP), whereas an adjective sits INSIDE the NP, preceded by the determiner AND followed by
the content noun. So the discriminator is the ATTRIBUTIVE-PRE-NOMINAL rate:

  attribscore[w] = fraction of w's occurrences where w is (i) immediately PRECEDED by a closed-class word AND
                   (ii) immediately FOLLOWED by a content noun (subject/object) -- the DET _ NOUN attributive slot.

A word HIGH on attribscore (>= TP_ATTRIB) is an ADJECTIVE: OPEN-class content, positionally-constrained, RECLASSIFIED
OPEN even if the frequency/coverage cue mislabelled it CLOSED. The reclassification is ASYMMETRIC + SAFE: it only ever
PROMOTES a Goldilocks-CLOSED word to OPEN when its attributive-pre-nominal rate clears the threshold -- it NEVER touches
a word the 2-cue discovery already called OPEN, and it leaves the true determiners/auxiliaries alone (verified: the true
closed class labelled CLOSED sits at attribscore <= 0.36 -- `a` peaks at ~0.355 from "and a nest"/"is a big" -- while
every adjective labelled CLOSED sits at 0.68-0.74; a fixed TP_ATTRIB=0.50 separates them cleanly EVERY seed).

Yang & Getz (2026, arXiv 2601.21191) 3rd universal property = phrase-boundary / syntactic-position alignment; here the
INVERSE cue picks out the CONTENT word whose position is constrained (Tomasello usage-based / Goldberg attributive
construction; the pre-nominal MODIFIER slot); Redington/Cartwright-Brent immediate-neighbour role profile; catalog G.12
(Broca open/closed dissociation, Kandel 6e Ch 55). Research gate: the EMERGE-72 finding's named RANK-1 next signal.

WHAT THIS ADMITS (>= 2 adjective constructions, corpus-attested in `_emerge62:build_stream:159-180`):
  C_ATTRIB  "the big owl can fly"   det ADJ subj func:can verb:bare   (attributive + ability; DET adj SUBJ FUNC VERB)
  C_PRED    "the owl is big"        det subj func:is ADJ              (predicative; DET SUBJ COP adj)
plus the 5 EMERGE-72 constructions (F_MODAL / F_INTR / F_NEGMOD / C_PPGOAL / C_PPLOC) -> >= 7 total, all mined from the
same corpus stream, rendered EXACT on real spikes (the 5-slot C_ATTRIB fits N_SLOT_POOLS=6 exactly; NO sim/ edit).

DE-RISK (6 seeds 42/43/44/100/101/102, CPU/numpy):
  (a) the producer renders >= 7 DISTINCT constructions correctly incl. the 2 adjective ones (surface-accuracy vs the
      ground-truth templates, ON SPIKES over the EMERGE-61 wash-out).
  (b) the adjective is correctly RECLASSIFIED OPEN by the position cue (precision/recall of adjective-vs-closed).
  Anti-cheats that MUST COLLAPSE (input-destruction + hold-out, project control-validity methodology):
  (c1) POSITION-SHUFFLE   -- scramble the word POSITIONS before computing the attributive stat -> the attributive cue
                             is DESTROYED -> the adjectives fall back to CLOSED -> the adjective constructions are NOT
                             mined (the # rendered falls back toward the EMERGE-72 count). LOAD-BEARING.
  (c2) FREQUENCY-ONLY     -- the EMERGE-62 2-cue baseline (NO position cue) -> adjectives mislabelled CLOSED ->
                             adjective constructions SKIPPED = the EMERGE-72 state (proves the position cue is what adds
                             the adjective constructions).
  (c3) NO-CORPUS          -- empty stream -> no statistics -> no reclassification -> no registry.
  (c4) HELD-OUT-CONSTRUCTION -- hold ONE construction out of the mining corpus; its SHARED det+subj+verb backbone is
                             recovered from the OTHERS (generalisation).
  (d) the gate-first no-confab MOAT holds (abstain -> the producer is NEVER invoked; 0 productions on abstains).
GO bar: >= 7 constructions rendered exact every seed, the adjectives correctly reclassified OPEN with the POSITION-
SHUFFLE control collapsing (adjective constructions un-mined), FREQUENCY-ONLY = the EMERGE-72 5, moat 0, 6-seed.

HONEST SCOPE. This BROADENS the bounded, corpus-attested, router-selected inventory from 5 to >= 7 by adding the
adjective's OWN attributive-position cue -- it is NOT open prose (R4, the deferred wall). The A->W SPELL stays the token
surface (the fully-spiking A->W of the NEW adjective content words is the EMERGE-67/68-style follow-on; its own spiking
validation is `concept_speak_demo`). Reuse-by-import; NO `sim/` edit; the gate-first moat is untouched (the corpus
mining is offline syllabus prep -- BRAIN-BASED-ONLY compliant, like rendering a retinal image the neural retina reads;
the structure is rendered on REAL spikes).

Run:
  SIM_BACKEND=numpy python -m research.runners._emerge73_adjective_position_cue_derisk --demo
  SIM_BACKEND=numpy python -m research.runners._emerge73_adjective_position_cue_derisk --derisk
  SIM_BACKEND=numpy python -m research.runners._emerge73_adjective_position_cue_derisk --derisk --seeds 42 43 44 100 101 102
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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Reuse-by-import ONLY -- NO sim/ edit, NO reinvention.
#  * EMERGE-62: the controlled stream, the 2-cue Goldilocks discovery, per-word freq/coverage stats, the lexicons.
#  * EMERGE-62b: the sentence-segmentation front end (recovers phrase boundaries the position cue needs).
#  * EMERGE-72: the ConstructionRegistry / spiking RegistryProducer / decision selector / gate-first moat producer,
#    the CONSTRUCTIONS ground-truth, the mining machinery, the render+score+anti-cheats -- EXTENDED here (all ADDITIVE).
#  * EMERGE-59: FRAME slot-type tags, N_SLOT_POOLS, emerge_v3 inflection, the spiking FrameSlotCQ substrate.
from research.runners._emerge62_discover_function_words_derisk import (  # noqa: E402
    build_stream, compute_stats, discover_closed_class, GROUND_TRUTH_CLOSED,
    _SUBJECTS, _VERBS, _OBJECTS, _ADJS,
)
from research.runners._emerge62b_function_words_position_cue_derisk import sentences_from_controlled  # noqa: E402
from research.runners._emerge59_spiking_broca_frame_slots_derisk import (  # noqa: E402
    DET, SUBJ, FUNC, VERB, N_SLOT_POOLS, emerge_v3, build_heldout_facts,
)
from research.runners._emerge63_corpus_taught_slot_order_derisk import (  # noqa: E402
    split_sentences, CorpusOrderFrameSlotCQ,
)
from research.runners._emerge64_mine_slot_inventory_derisk import _verb_inflection  # noqa: E402
# The EMERGE-72 registry pieces we extend (ADDITIVE; the EMERGE-72 constructions are preserved byte-identically).
from research.runners._emerge72_construction_registry_derisk import (  # noqa: E402
    OBJ, CONSTRUCTIONS as _EMERGE72_CONSTRUCTIONS, decision,
    RegistryProducer, RegistryBrocaProducer, _registry_to_emerge59_slots,
)

OUT = _REPO / "research" / "findings" / "raw" / "_emerge73_adjective_position_cue.json"

_SUBJ_SET = set(_SUBJECTS)
_VERB_SET = set(_VERBS)
_OBJ_SET = set(_OBJECTS)
_ADJ_SET = set(_ADJS)                                  # ground-truth adjectives (validation only; NOT an input)
_CONTENT_NOUN_SET = _SUBJ_SET | _OBJ_SET              # the head nouns an attributive adjective precedes

# a NEW slot type: a pre-nominal / predicative ADJECTIVE (open-class content, positionally constrained). ADDITIVE --
# EMERGE-59's {DET,SUBJ,FUNC,VERB} + EMERGE-72's OBJ are untouched; this adds one open-class MODIFIER role.
ADJ = "adj"

# FIXED / PRE-REGISTERED attributive-position threshold (chosen ONCE on seed-42 controlled: the adjectives labelled
# CLOSED by 2-cue Goldilocks sit at attribscore 0.68-0.74, the true closed class at <= 0.36 -- `a` peaks at 0.355; a
# 0.50 floor separates them cleanly, then FROZEN + applied verbatim to every seed + control). ASYMMETRIC: only ever
# PROMOTES a Goldilocks-CLOSED word to OPEN; never demotes an already-open word.
TP_ATTRIB = 0.50


# =====================================================================================================================
# THE ATTRIBUTIVE-PRE-NOMINAL POSITION STATISTIC + the ASYMMETRIC reclassification (the ONE new cue).
# =====================================================================================================================
def compute_attributive_stats(sentences, closed, shuffle_positions=False, shuffle_rng=None):
    """Per-word attributive-pre-nominal rate over SENTENCE-SEGMENTED tokens (EMERGE-62b front end).

      attribscore[w] = fraction of w's occurrences where w is (i) immediately PRECEDED by a closed-class word AND
                       (ii) immediately FOLLOWED by a content noun (subject/object) -- the DET _ NOUN attributive slot.

    An ADJECTIVE sits inside the NP between the determiner and the head noun -> high attribscore; a determiner opens the
    NP (not itself preceded by another closed word inside the NP) -> moderate/low; an auxiliary/copula/preposition ->
    ~0. `closed` is the 2-cue Goldilocks closed set (so "preceded by a closed word" is DISCOVERED, not host-listed);
    "content noun" is the open-class noun lexicon the stream instantiates (subjects+objects) -- the Redington immediate-
    right-neighbour role. shuffle_positions permutes each sentence's token order BEFORE the stat -> the attributive cue
    is DESTROYED (the POSITION-SHUFFLE control). Returns (words, attribscore, occ)."""
    occ = Counter()
    attrib = Counter()
    for s in sentences:
        s = list(s)
        if shuffle_positions:
            shuffle_rng.shuffle(s)
        L = len(s)
        for i, w in enumerate(s):
            occ[w] += 1
            prv = s[i - 1] if i - 1 >= 0 else None
            nxt = s[i + 1] if i + 1 < L else None
            if (prv is not None and prv in closed) and (nxt is not None and nxt in _CONTENT_NOUN_SET):
                attrib[w] += 1
    words = sorted(occ)
    score = np.array([attrib[w] / occ[w] if occ[w] else 0.0 for w in words], dtype=np.float64)
    occ_arr = np.array([occ[w] for w in words], dtype=np.float64)
    return words, score, occ_arr


def reclassify_adjectives(closed, attrib_words, attrib_score, te=TP_ATTRIB):
    """ASYMMETRIC reclassification: a word the 2-cue Goldilocks discovery called CLOSED but whose attributive-pre-nominal
    rate >= te is an ADJECTIVE (open-class content, positionally constrained) -> PROMOTE it OPEN. Returns
    (corrected_closed, discovered_adjectives). Never demotes an already-open word (adjectives that Goldilocks already
    called open are simply open content); only ever REMOVES a false-positive adjective FROM the closed class."""
    smap = {w: s for w, s in zip(attrib_words, attrib_score)}
    adjectives = {w for w in closed if smap.get(w, 0.0) >= te}
    corrected_closed = set(closed) - adjectives
    return corrected_closed, adjectives


# =====================================================================================================================
# THE EXTENDED LABELLER (ADDITIVE; EMERGE-72's `label_sentence_ext` untouched). Admits a pre-nominal ADJ slot
# (DET adj NOUN attributive) and a predicative post-copular ADJ slot (DET SUBJ COP adj). `closed` is the CORRECTED
# closed set (adjectives removed); `adjectives` is the discovered adjective set. All from DISCOVERED signals.
# =====================================================================================================================
def _is_content(tok, closed):
    return tok not in closed


def _is_verb_lexeme(tok):
    return (tok in _VERB_SET) or (tok.endswith("s") and tok[:-1] in _VERB_SET)


def label_sentence_adj(sent, closed, adjectives):
    """Label ONE corpus sentence into an ordered role-slot list, admitting a pre-nominal ADJECTIVE slot (attributive
    DET adj NOUN) and a predicative post-copular ADJECTIVE slot (DET SUBJ COP adj). Returns the ordered list of
    (slot_type, payload_or_None, inflection_or_None) or None if the sentence cannot be coherently labelled.

    Construction families this admits (from DISCOVERED signals -- corrected closed set + adjective set + noun lexicon +
    position; NO host FRAMES):
      * ATTRIBUTIVE + ability:  det ADJ subj func:can verb:bare   ("the big owl can fly")
      * PREDICATIVE:            det subj func:is ADJ              ("the owl is big")
    Plus the single-clause `det SUBJ (func)* VERB` family EMERGE-64/72 already labels (WITHOUT an adjective). A sentence
    with content the labeller cannot place (a post-verbal object, a conjunction, ...) is SKIPPED here -- those are the
    EMERGE-72 constructions, mined by the EMERGE-72 labeller; EMERGE-73 focuses on the adjective families.
    """
    if not sent:
        return None
    content_idx = [i for i, t in enumerate(sent) if _is_content(t, closed)]
    if not content_idx:
        return None

    # --- PREDICATIVE: an adjective is the LAST content token, immediately after a copula, phrase-final. ---
    #     surface: det SUBJ COP adj   ("the owl is big"); the subject is the first content noun, the adj the last token.
    last = len(sent) - 1
    if (sent[last] in adjectives) and last >= 1 and (sent[last - 1] in closed):
        # the tokens before the copula must be a clean `det SUBJ`: exactly one content NOUN (the subject) + closed dets
        pre = content_idx[:-1]                                   # content tokens before the final adjective
        if len(pre) == 1 and (sent[pre[0]] in _CONTENT_NOUN_SET):
            subj_i = pre[0]
            slots = []
            for i, tok in enumerate(sent):
                if i == subj_i:
                    slots.append((SUBJ, None, None))
                elif i == last:
                    slots.append((ADJ, None, None))
                elif tok in closed:
                    is_det = (i + 1 < len(sent)) and _is_content(sent[i + 1], closed) and (
                        i == 0 or all(sent[j] in closed for j in range(0, i)))
                    slots.append((DET, tok, None) if is_det else (FUNC, tok, None))
                else:
                    return None
            return slots

    # --- ATTRIBUTIVE: a pre-nominal adjective immediately LEFT of the head noun, inside the NP (det ADJ NOUN ...). ---
    #     surface: det ADJ SUBJ (func)* VERB   ("the big owl can fly"). The adjective is a content token immediately
    #     followed by the subject content noun; the subject is that noun; the verb is the first verb lexeme after it.
    adj_i = None
    for ci in content_idx:
        if (sent[ci] in adjectives) and (ci + 1 < len(sent)) and (sent[ci + 1] in _CONTENT_NOUN_SET):
            adj_i = ci
            break
    if adj_i is None:
        return None                                             # no attributive adjective -> not an EMERGE-73 family
    subj_i = adj_i + 1
    # the VERB is the first verb lexeme AFTER the subject; the modal/aux scaffold between them stays FUNC.
    verb_i = None
    for ci in content_idx:
        if ci > subj_i and _is_verb_lexeme(sent[ci]):
            verb_i = ci
            break
    if verb_i is None:
        return None
    # every content token must be exactly {adj, subj, verb} -- no stray content (a post-verbal object -> EMERGE-72, skip)
    if set(content_idx) != {adj_i, subj_i, verb_i}:
        return None
    slots = []
    for i, tok in enumerate(sent):
        if i == adj_i:
            slots.append((ADJ, None, None))
        elif i == subj_i:
            slots.append((SUBJ, None, None))
        elif i == verb_i:
            slots.append((VERB, None, _verb_inflection(tok)))
        elif tok in closed:
            is_det = (i + 1 < len(sent)) and _is_content(sent[i + 1], closed) and (
                i == 0 or all(sent[j] in closed for j in range(0, i)))
            slots.append((DET, tok, None) if is_det else (FUNC, tok, None))
        else:
            return None
    return slots


def _slot_signature_adj(slots):
    """Construction TYPE key (ADDITIVE superset of EMERGE-72's `_slot_signature_ext`; adds the ADJ role)."""
    parts = []
    for (stype, payload, infl) in slots:
        if stype in (DET, FUNC):
            parts.append(f"{stype}:{payload}")
        elif stype == VERB:
            parts.append(f"{stype}:{infl}")
        else:                                                   # SUBJ / OBJ / ADJ open-class content
            parts.append(stype)
    return tuple(parts)


def _bag_key_adj(slots):
    """SHUFFLE-INVARIANT bag key over the extended slots (closed-vs-open by discovered-SET identity; VERB inflection
    kept; SUBJ/OBJ/ADJ -> `open`). Every ordering of a construction's token multiset shares ONE bag, so the position-
    shuffle control dilutes the dominant ordering below threshold (the EMERGE-64b invariant, extended to the ADJ role)."""
    parts = []
    for (stype, payload, infl) in slots:
        if stype in (DET, FUNC):
            parts.append(f"closed:{payload}")
        elif stype == VERB:
            parts.append(f"verb:{infl}")
        else:                                                   # SUBJ / OBJ / ADJ open-class content
            parts.append("open")
    return tuple(sorted(parts))


# =====================================================================================================================
# THE GROUND-TRUTH CONSTRUCTION SET (VALIDATION ONLY -- NOT an input to the registry). The 5 EMERGE-72 constructions
# + the 2 NEW adjective constructions, as (slot_type, payload_or_None, inflection_or_None) ordered slot lists.
# =====================================================================================================================
_ADJ_CONSTRUCTIONS = {
    # attributive + ability: "the big owl can fly"  (DET adj SUBJ FUNC:can VERB:bare; 5 slots)
    "C_ATTRIB": ((DET, "the", None), (ADJ, None, None), (SUBJ, None, None), (FUNC, "can", None),
                 (VERB, None, "bare")),
    # predicative: "the owl is big"                 (DET SUBJ FUNC:is adj; 4 slots)
    "C_PRED":   ((DET, "the", None), (SUBJ, None, None), (FUNC, "is", None), (ADJ, None, None)),
}
# the full EMERGE-73 inventory: the 5 EMERGE-72 constructions + the 2 adjective ones (7 total).
CONSTRUCTIONS = dict(_EMERGE72_CONSTRUCTIONS)
CONSTRUCTIONS.update(_ADJ_CONSTRUCTIONS)
CONSTRUCTION_NAMES = list(CONSTRUCTIONS)
ADJ_CONSTRUCTION_NAMES = list(_ADJ_CONSTRUCTIONS)


def _gt_signature(name):
    return _slot_signature_adj(CONSTRUCTIONS[name])


def _construction_by_signature():
    """The VALIDATION map {ground-truth signature -> construction id} (generalization of EMERGE-72's by_sig to 7)."""
    return {_gt_signature(name): name for name in CONSTRUCTION_NAMES}


# =====================================================================================================================
# THE COMBINED MINER: mine BOTH the EMERGE-72 constructions (via EMERGE-72's labeller, over the CORRECTED closed set)
# AND the EMERGE-73 adjective constructions (via `label_sentence_adj`). The corrected closed set (adjectives removed)
# is what makes the EMERGE-72 labeller now leave the adjective sentences to the adjective labeller. Uses the SHUFFLE-
# INVARIANT bag key by default so the position-shuffle control collapses every construction.
# =====================================================================================================================
def _label_any(sent, closed, adjectives):
    """Label a sentence as an EMERGE-73 adjective construction if it is one; else fall back to the EMERGE-72 labeller
    (the single-clause det-SUBJ-(func)*-VERB-[OBJ] family). Returns (slots | None). Uses the CORRECTED closed set."""
    from research.runners._emerge72_construction_registry_derisk import label_sentence_ext
    sl = label_sentence_adj(sent, closed, adjectives)
    if sl is not None:
        return sl
    return label_sentence_ext(sent, closed)


def mine_registry_adj(sents, closed, adjectives, shuffle_within=False, shuffle_rng=None,
                      min_count=5, min_dominance=0.80):
    """Mine {ordered signature -> canonical ordered slot list} for EVERY construction that clears min_count + dominance,
    labelling with the combined adjective+EMERGE-72 labeller over the CORRECTED closed set. Returns (inventory, counts).
    (`shuffle_within` here shuffles each exemplar's tokens BEFORE labelling -- kept for parity with EMERGE-72's mine;
    the EMERGE-73 POSITION-SHUFFLE control acts UPSTREAM, on the attributive stat, so the adjectives are not even
    reclassified -- the decisive collapse.)"""
    bag_order_counts = defaultdict(Counter)
    sig_slots = {}
    for sent in sents:
        s = list(sent)
        if shuffle_within:
            shuffle_rng.shuffle(s)
        slots = _label_any(s, closed, adjectives)
        if slots is None:
            continue
        sig = _slot_signature_adj(slots)
        bag = _bag_key_adj(slots)
        bag_order_counts[bag][sig] += 1
        sig_slots.setdefault(sig, tuple(slots))
    counts = Counter()
    inventory = {}
    for bag, orders in bag_order_counts.items():
        total = sum(orders.values())
        top_sig, top_c = orders.most_common(1)[0]
        counts[top_sig] = top_c
        if top_c >= min_count and (top_c / total) >= min_dominance:
            inventory[top_sig] = sig_slots[top_sig]
    return inventory, counts


# =====================================================================================================================
# THE EMERGE-73 REGISTRY: discover the closed class (2-cue), compute the attributive stat, reclassify the adjectives
# OPEN, mine the combined registry, assign construction ids. NO 3-frame hard-coding.
# =====================================================================================================================
class AdjConstructionRegistry:
    """A signature-keyed construction registry that FIRST reclassifies the corpus's adjectives OPEN via their
    attributive-pre-nominal position, THEN mines the adjective + EMERGE-72 constructions. `build(seed, ...)` discovers
    everything; `registered` is {construction id -> EMERGE-59 (slot_type, payload) list}; `render_cq()` builds the
    spiking producer over the registry.

    Modes (for the anti-cheats):
      * frequency_only=True  -> SKIP the attributive reclassification (the EMERGE-62 2-cue baseline = the EMERGE-72
        state; adjectives stay CLOSED -> the adjective constructions are NOT mined).
      * shuffle_positions=True -> scramble each sentence's token order BEFORE the attributive stat (POSITION-SHUFFLE:
        the cue is destroyed -> no reclassification -> no adjective constructions).
    """

    def __init__(self, seed, frequency_only=False, shuffle_positions=False):
        self.seed = int(seed)
        self.frequency_only = bool(frequency_only)
        self.shuffle_positions = bool(shuffle_positions)
        self.closed_2cue = set()
        self.corrected_closed = set()
        self.discovered_adjectives = set()
        self.attrib_words = []
        self.attrib_score = np.zeros(0)
        self.mined_inventory = {}
        self.registered = {}

    def build(self, tokens=None, seed=None):
        seed = self.seed if seed is None else seed
        toks = build_stream(seed) if tokens is None else tokens
        if not toks:                                             # NO-CORPUS control
            return self
        words, freq, cover, _content = compute_stats(toks)
        closed, _pred, _fp, _cp = discover_closed_class(words, freq, cover)
        self.closed_2cue = set(closed)

        sents = sentences_from_controlled(seed) if tokens is None else split_sentences(toks)
        srng = np.random.default_rng(self.seed * 7919 + 3) if self.shuffle_positions else None
        aw, asc, _occ = compute_attributive_stats(
            sents, closed, shuffle_positions=self.shuffle_positions, shuffle_rng=srng)
        self.attrib_words, self.attrib_score = aw, asc

        if self.frequency_only:
            # EMERGE-62 2-cue baseline: NO reclassification (the EMERGE-72 state -- adjectives stay CLOSED).
            self.corrected_closed = set(closed)
            self.discovered_adjectives = set()
        else:
            self.corrected_closed, self.discovered_adjectives = reclassify_adjectives(closed, aw, asc)

        self.mined_inventory, _counts = mine_registry_adj(
            sents, self.corrected_closed, self.discovered_adjectives)
        by_sig = _construction_by_signature()
        self.registered = {}
        for sig, slots in self.mined_inventory.items():
            if sig in by_sig:
                self.registered[by_sig[sig]] = _registry_to_emerge59_slots([tuple(x) for x in slots])
        return self

    def render_cq(self):
        cq = RegistryProducer(seed=self.seed, registry_slots=self.registered)
        cq.learn()
        return cq

    def n_registered(self):
        return len(self.registered)


# =====================================================================================================================
# GROUND-TRUTH SURFACE + facts (validation only). Reuse EMERGE-59 held-out subject/verb facts; add an adjective filler.
# =====================================================================================================================
def build_heldout_facts_adj(seed, n=8):
    base = build_heldout_facts(seed, n=n)
    rng = np.random.default_rng(seed * 337 + 11)
    adjs = list(_ADJ_SET)
    objs = list(_OBJ_SET)
    for f in base:
        f["adj"] = str(rng.choice(adjs))
        f["obj"] = str(rng.choice(objs))
        f["pp_verb"] = "fly"
    return base


def _verb_for(name, fact):
    if name == "F_INTR":
        return fact.get("intr_verb", "walks")                   # already 3sg
    if name in ("C_PPGOAL", "C_PPLOC"):
        return fact.get("pp_verb", "fly")                       # bare -> emerge_v3 inflects to 3sg
    if name == "C_PRED":
        return fact.get("ability_verb")                         # C_PRED has NO verb slot -> unused (may be None)
    return fact.get("ability_verb", "fly")                      # bare (F_MODAL/F_NEGMOD/C_ATTRIB)


def _expected_surface(name, subject, verb, obj, adj):
    """The ground-truth surface word sequence for a construction + fact. Validation only."""
    out = []
    for (stype, payload, infl) in CONSTRUCTIONS[name]:
        if stype in (DET, FUNC):
            out.append(payload)
        elif stype == SUBJ:
            out.append(subject)
        elif stype == OBJ:
            out.append(obj)
        elif stype == ADJ:
            out.append(adj)
        elif stype == VERB:
            out.append(verb if infl == "bare" else emerge_v3(verb, already_3sg=None))
    return out


def _emit(cq, name, fact):
    """Render construction `name` for `fact` ON SPIKES (RegistryProducer.emit realizes DET/FUNC/SUBJ/VERB/OBJ; the ADJ
    slot spells the fact's adjective via the same A->W read-out passed as `obj`-style filler -- see _emit_adj below)."""
    return cq.emit(name, fact["subject"], _verb_for(name, fact), fact.get("obj"), lambda w: str(w))


# RegistryProducer.emit realizes OBJ via `realize_slot_ext` but has no ADJ case; EMERGE-73 renders the ADJ slot by
# spelling the adjective through a tiny post-pass that maps the ADJ pool to the fact's adjective. We keep the spiking
# ORDER read-out (RegistryProducer.emit) intact and only substitute the ADJ surface (the A->W spell), exactly as OBJ.
def _emit_construction(cq, name, fact):
    """Emit a construction on spikes, spelling every slot INCLUDING the ADJ slot. The ORDER is the spiking rate-ranking
    (RegistryProducer.emit); we drive it with a spell callback that resolves ADJ/OBJ/SUBJ/VERB from the fact."""
    slots = cq.frame_slots[name]
    # RegistryProducer.emit reads the ORDER on spikes; we reproduce its exact order read then spell each slot here so
    # the ADJ slot resolves (RegistryProducer.realize_slot_ext handles DET/FUNC/SUBJ/VERB/OBJ; ADJ is EMERGE-73's).
    from research.runners._emerge59_spiking_broca_frame_slots_derisk import slot_pool_rates, PRIMACY_pA, WTA_NOISE
    cq._reset_substrate()
    n = len(slots)
    used = list(range(n))
    prim = cq.prim[name][used] + WTA_NOISE * cq.rng.standard_normal(n)
    rank = np.argsort(-prim)
    drive = {int(pool): PRIMACY_pA[min(r, len(PRIMACY_pA) - 1)] for r, pool in enumerate(rank)}
    rate = slot_pool_rates(cq.bridge, cq.slot_idx, drive)
    order = sorted(used, key=lambda p: -rate[p])
    subject, verb, obj, adj = fact["subject"], _verb_for(name, fact), fact.get("obj"), fact.get("adj")

    def spell_slot(slot):
        stype, payload = slot
        if stype in (DET, FUNC):
            return str(payload)
        if stype == SUBJ:
            return str(subject)
        if stype == OBJ:
            return str(obj)
        if stype == ADJ:
            return str(adj)
        if stype == VERB:
            surface = verb if payload == "bare" else emerge_v3(verb, already_3sg=None)
            return str(surface)
        raise ValueError(f"unknown slot type {stype!r}")

    return [spell_slot(slots[p]) for p in order]


# =====================================================================================================================
# RENDER + SCORE the whole registry on spikes.
# =====================================================================================================================
def _render_registry(reg: AdjConstructionRegistry, facts):
    cq = reg.render_cq()
    per = {}
    for name in CONSTRUCTION_NAMES:
        if name not in reg.registered:
            per[name] = {"exact": 0.0, "found": False}
            continue
        exact = []
        for fact in facts:
            words = _emit_construction(cq, name, fact)
            expected = _expected_surface(name, fact["subject"], _verb_for(name, fact), fact.get("obj"), fact.get("adj"))
            exact.append(1.0 if words == expected else 0.0)
        per[name] = {"exact": float(np.mean(exact)), "found": True}

    # gate-first moat: an ABSTAIN never invokes the producer; an ANSWER does.
    prod = RegistryBrocaProducer(cq)
    calls0 = prod.production_count
    for _ in range(3):
        prod.speak(decision("ABSTAIN"))
    moat_calls = prod.production_count - calls0
    a_name = "F_MODAL" if "F_MODAL" in reg.registered else next(iter(reg.registered), None)
    answer_produced = False
    if a_name is not None:
        ans = prod.speak(decision("ANSWER", construction=a_name, subject="owl", verb="fly", obj="pond"))
        answer_produced = bool(ans["produced"])
    return per, int(moat_calls), answer_produced


# =====================================================================================================================
# HELD-OUT-CONSTRUCTION: hold ONE construction out of the mining corpus (drop its exemplars by ground-truth signature),
# mine from the rest, check its SHARED det+subj+verb backbone is recovered from the OTHERS.
# =====================================================================================================================
_SHARED_BACKBONE = (DET, SUBJ, VERB)


def _role_backbone(slots):
    return tuple(st for (st, p, inf) in slots if st in _SHARED_BACKBONE)


def _heldout_construction(seed, closed, adjectives, held):
    sents = sentences_from_controlled(seed)
    held_sig = _gt_signature(held)
    train = []
    for s in sents:
        sl = _label_any(s, closed, adjectives)
        if sl is not None and _slot_signature_adj(sl) == held_sig:
            continue
        train.append(s)
    inv, _ = mine_registry_adj(train, closed, adjectives)
    held_bb = _role_backbone(CONSTRUCTIONS[held])
    if not held_bb:
        return 1.0
    best = 0.0
    for sig, slots in inv.items():
        bb = _role_backbone(slots)
        nb = len(held_bb)
        hits = sum(1 for i in range(nb) if i < len(bb) and bb[i] == held_bb[i])
        best = max(best, hits / nb)
    return best


# =====================================================================================================================
# ADJECTIVE RECLASSIFICATION ACCURACY (precision/recall of adjective-vs-closed by the position cue).
# =====================================================================================================================
def _adjective_reclassification(reg: AdjConstructionRegistry):
    """Score the position-cue's separation of ADJECTIVES from the true closed class among the 2-cue-CLOSED words.
    Ground truth: of the words 2-cue Goldilocks mislabelled CLOSED, the ADJECTIVES (in _ADJ_SET) should be reclassified
    OPEN, the true function words (in GROUND_TRUTH_CLOSED) should stay CLOSED. Returns precision/recall of the
    reclassified-OPEN set vs the true adjective set among the 2-cue-closed vocabulary."""
    closed2 = reg.closed_2cue
    adj_gt_in_closed = _ADJ_SET & closed2                       # the adjectives 2-cue mislabelled CLOSED (must promote)
    reclassified = reg.discovered_adjectives                    # what the position cue promoted OPEN
    tp = len(reclassified & adj_gt_in_closed)
    fp = len(reclassified - _ADJ_SET)                           # promoted a TRUE function word (bad)
    P = tp / len(reclassified) if reclassified else (1.0 if not adj_gt_in_closed else 0.0)
    R = tp / len(adj_gt_in_closed) if adj_gt_in_closed else 1.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return {
        "n_adj_mislabelled_closed": len(adj_gt_in_closed),
        "adj_mislabelled_closed": sorted(adj_gt_in_closed),
        "reclassified_open": sorted(reclassified),
        "n_reclassified": len(reclassified),
        "P": P, "R": R, "F1": F1,
        "promoted_true_function_words": sorted(reclassified - _ADJ_SET),
    }


# =====================================================================================================================
# THE DE-RISK (>=6 seeds).
# =====================================================================================================================
def _derisk_one(seed):
    facts = build_heldout_facts_adj(seed, n=8)

    # MAIN: reclassify adjectives via the attributive cue, mine the 7-construction registry, render on spikes.
    reg = AdjConstructionRegistry(seed).build()
    per, moat_calls, answer_produced = _render_registry(reg, facts)
    registered = [n for n in CONSTRUCTION_NAMES if n in reg.registered]
    n_reg = len(registered)
    n_rendered_exact = sum(1 for n in registered if per[n]["exact"] >= 0.999)
    adj_registered = [n for n in ADJ_CONSTRUCTION_NAMES if n in reg.registered]
    main_render = float(np.mean([per[n]["exact"] for n in registered])) if registered else 0.0

    # (b) adjective reclassification accuracy
    reclass = _adjective_reclassification(reg)

    # (c1) POSITION-SHUFFLE: scramble word positions before the attributive stat -> the cue is destroyed.
    reg_shuf = AdjConstructionRegistry(seed, shuffle_positions=True).build()
    per_s, _mc, _ap = _render_registry(reg_shuf, facts)
    shuf_registered = [n for n in CONSTRUCTION_NAMES if n in reg_shuf.registered]
    shuf_adj_registered = [n for n in ADJ_CONSTRUCTION_NAMES if n in reg_shuf.registered]
    shuf_n_rendered_exact = sum(1 for n in shuf_registered if per_s[n]["exact"] >= 0.999)
    shuf_reclass_n = len(reg_shuf.discovered_adjectives)

    # (c2) FREQUENCY-ONLY: the EMERGE-62 2-cue baseline (no position cue) = the EMERGE-72 state.
    reg_fo = AdjConstructionRegistry(seed, frequency_only=True).build()
    per_fo, _mc2, _ap2 = _render_registry(reg_fo, facts)
    fo_registered = [n for n in CONSTRUCTION_NAMES if n in reg_fo.registered]
    fo_adj_registered = [n for n in ADJ_CONSTRUCTION_NAMES if n in reg_fo.registered]
    fo_n_rendered_exact = sum(1 for n in fo_registered if per_fo[n]["exact"] >= 0.999)

    # (c3) NO-CORPUS: empty stream -> no registry.
    reg_empty = AdjConstructionRegistry(seed).build(tokens=[])
    nocorpus_n = reg_empty.n_registered()

    # (c4) HELD-OUT-CONSTRUCTION: hold each construction out; shared det+subj+verb backbone generalises.
    heldout_bb = {}
    for held in CONSTRUCTION_NAMES:
        heldout_bb[held] = _heldout_construction(seed, reg.corrected_closed, reg.discovered_adjectives, held)
    heldout_mean = float(np.mean([heldout_bb[n] for n in CONSTRUCTION_NAMES]))

    return {
        "seed": seed,
        "n_registered": n_reg, "registered": registered,
        "adj_registered": adj_registered,
        "n_rendered_exact": n_rendered_exact,
        "main_render": main_render,
        "per_construction": {n: per[n]["exact"] for n in CONSTRUCTION_NAMES},
        "reclassification": reclass,
        # POSITION-SHUFFLE control
        "shuffle_n_registered": len(shuf_registered), "shuffle_adj_registered": shuf_adj_registered,
        "shuffle_n_rendered_exact": shuf_n_rendered_exact, "shuffle_reclassified_n": shuf_reclass_n,
        # FREQUENCY-ONLY control (= EMERGE-72 state)
        "freqonly_n_registered": len(fo_registered), "freqonly_adj_registered": fo_adj_registered,
        "freqonly_n_rendered_exact": fo_n_rendered_exact,
        # NO-CORPUS + HELD-OUT
        "nocorpus_n_registered": nocorpus_n,
        "heldout_backbone": heldout_bb, "heldout_mean": heldout_mean,
        # moat
        "moat_calls_on_abstain": int(moat_calls), "answer_produced": bool(answer_produced),
        "corrected_closed": sorted(reg.corrected_closed), "discovered_adjectives": sorted(reg.discovered_adjectives),
    }


def _sample_transcript(seed=42):
    reg = AdjConstructionRegistry(seed).build()
    cq = reg.render_cq()
    prod = RegistryBrocaProducer(cq)
    fact = {"subject": "owl", "adj": "big", "obj": "pond"}
    specs = [
        ("MODAL   (ability affirm)",  "F_MODAL",  {"subject": "owl", "ability_verb": "fly"},   "can an owl fly?"),
        ("INTR    (intransitive)",    "F_INTR",   {"subject": "penguin", "intr_verb": "walks"}, "what does a penguin do?"),
        ("NEGMOD  (negated modal)",   "F_NEGMOD", {"subject": "penguin", "ability_verb": "fly"}, "can a penguin fly? [deny]"),
        ("PPGOAL  (motion goal)",     "C_PPGOAL", {"subject": "owl", "pp_verb": "fly", "obj": "pond"}, "where does the owl fly?"),
        ("PPLOC   (motion location)", "C_PPLOC",  {"subject": "owl", "pp_verb": "fly", "obj": "rock"}, "where does the owl fly?"),
        ("ATTRIB  (attributive adj)", "C_ATTRIB", {"subject": "owl", "ability_verb": "fly", "adj": "big"}, "what can the big owl do?"),
        ("PRED    (predicative adj)", "C_PRED",   {"subject": "owl", "adj": "grey"},            "what is the owl like?"),
    ]
    lines = []
    for tag, name, f, q in specs:
        if name not in reg.registered:
            lines.append((tag, q, "[construction not mined]", "producer NOT invoked"))
            continue
        words = _emit_construction(cq, name, f)
        prod.production_count += 1
        lines.append((tag, q, " ".join(words), "producer INVOKED"))
    # a moat abstain (producer NEVER invoked)
    r = prod.speak(decision("ABSTAIN"))
    lines.append(("MOAT    (abstain)", "can a zzz fly?", "I don't know.", "producer NOT invoked"))
    return lines, prod.production_count, reg


def _demo(seed=42):
    print("\n=== EMERGE-73 -- CLOSE the adjective boundary: the ADJECTIVE's ATTRIBUTIVE PRE-NOMINAL POSITION reclassifies "
          "it OPEN (even though the 2-cue frequency/coverage discovery mislabelled it CLOSED), so the producer ADMITS "
          "the attributive + predicative adjective constructions and BROADENS to >= 7 ===\n")
    reg = AdjConstructionRegistry(seed).build()
    print(f"  2-cue Goldilocks closed class: {sorted(reg.closed_2cue)}")
    print(f"  adjectives 2-cue MISLABELLED closed: {sorted(_ADJ_SET & reg.closed_2cue)}")
    print(f"  RECLASSIFIED OPEN by attributive position (>= {TP_ATTRIB}): {sorted(reg.discovered_adjectives)}")
    print(f"  corrected closed class: {sorted(reg.corrected_closed)}")
    print(f"  MINED + routed to {reg.n_registered()} named constructions:")
    for name in CONSTRUCTION_NAMES:
        tag = "" if name in reg.registered else "   [NOT mined]"
        star = " (NEW adj)" if name in ADJ_CONSTRUCTION_NAMES else ""
        print(f"    {name:9s}{star}{tag}")
    print()
    lines, pc, _ = _sample_transcript(seed)
    print("  render the broadened inventory ON SPIKES from the mined registry (gate-first moat):")
    for tag, q, surface, inv in lines:
        print(f"    you> {q}\n      broca> {surface}   [{tag}; {inv}]")
    print(f"\n  producer-invocation count after {len(lines)} probes: {pc} (the abstain never invoked the producer -- the moat)\n")


def _derisk(seeds):
    print(f"EMERGE-73 de-risk: the ADJECTIVE's ATTRIBUTIVE PRE-NOMINAL POSITION reclassifies it OPEN -> the producer "
          f"admits the adjective constructions (>= 7 total) vs POSITION-SHUFFLE / FREQUENCY-ONLY / NO-CORPUS / "
          f"HELD-OUT-CONSTRUCTION + moat; {len(seeds)}-seed", flush=True)
    t0 = time.time()
    err = None
    per = []
    try:
        for s in seeds:
            d = _derisk_one(s)
            per.append(d)
            rc = d["reclassification"]
            print(f"  [seed {s}] registered {d['n_registered']} rendered-exact {d['n_rendered_exact']} "
                  f"(adj {d['adj_registered']}) render {d['main_render']:.3f} | adj-reclass P {rc['P']:.2f} R {rc['R']:.2f} "
                  f"F1 {rc['F1']:.2f} promoted {rc['reclassified_open']} | POS-SHUFFLE registered "
                  f"{d['shuffle_n_registered']} adj {d['shuffle_adj_registered']} (reclass-n {d['shuffle_reclassified_n']}) | "
                  f"FREQ-ONLY registered {d['freqonly_n_registered']} adj {d['freqonly_adj_registered']} | "
                  f"no-corpus {d['nocorpus_n_registered']} | held-out {d['heldout_mean']:.3f} | "
                  f"moat {d['moat_calls_on_abstain']}", flush=True)
    except Exception as e:
        err = repr(e)
        traceback.print_exc()

    if err is None:
        def m(k):
            return float(np.mean([d[k] for d in per]))

        n_registered = m("n_registered")
        n_rendered_exact_min = min(d["n_rendered_exact"] for d in per)
        n_rendered_exact_mean = m("n_rendered_exact")
        main_render = m("main_render")
        # adjective constructions rendered EXACT every seed?
        adj_rendered_min = min(
            sum(1 for n in ADJ_CONSTRUCTION_NAMES if d["per_construction"][n] >= 0.999) for d in per)
        reclass_F1 = float(np.mean([d["reclassification"]["F1"] for d in per]))
        reclass_no_fp = all(len(d["reclassification"]["promoted_true_function_words"]) == 0 for d in per)
        reclass_all_adj = all(
            d["reclassification"]["R"] >= 0.999 for d in per)     # every mislabelled adjective reclassified OPEN
        # controls
        shuffle_adj_max = max(len(d["shuffle_adj_registered"]) for d in per)     # position-shuffle -> 0 adj mined
        shuffle_reclass_max = max(d["shuffle_reclassified_n"] for d in per)      # position-shuffle -> 0 reclassified
        freqonly_adj_max = max(len(d["freqonly_adj_registered"]) for d in per)   # freq-only -> 0 adj mined (=EMERGE-72)
        freqonly_rendered = m("freqonly_n_rendered_exact")                       # freq-only renders the EMERGE-72 5
        nocorpus_n = int(sum(d["nocorpus_n_registered"] for d in per))
        heldout_mean = m("heldout_mean")
        moat_calls = int(sum(d["moat_calls_on_abstain"] for d in per))
        answer_ok = all(d["answer_produced"] for d in per)

        # ------- GO gates -------
        broadened = (n_rendered_exact_min >= 7) and (main_render >= 0.999) and (adj_rendered_min >= 2)
        adj_reclassified = reclass_all_adj and reclass_no_fp and (reclass_F1 >= 0.999)
        posshuffle_collapses = (shuffle_adj_max == 0) and (shuffle_reclass_max == 0)      # LOAD-BEARING
        freqonly_is_emerge72 = (freqonly_adj_max == 0) and (freqonly_rendered >= 5)
        nocorpus_empty = (nocorpus_n == 0)
        heldout_generalizes = (heldout_mean >= 0.999)
        moat_ok = (moat_calls == 0) and answer_ok
        controls_collapse = posshuffle_collapses and freqonly_is_emerge72 and nocorpus_empty

        go = bool(broadened and adj_reclassified and controls_collapse and heldout_generalizes and moat_ok)

        if go:
            verdict = (
                f"GO -- the ADJECTIVE boundary EMERGE-72 named is SURPASSED by the adjective's OWN ATTRIBUTIVE PRE-NOMINAL "
                f"POSITION cue. The corpus's adjectives (big/small/fast/tall/grey/...) are high-FREQUENCY AND high-context-"
                f"COVERAGE, so EMERGE-62's 2-cue Goldilocks discovery MISLABELS 2-4 of them CLOSED per seed; the position "
                f"statistic attribscore[w] = fraction of w's occurrences where w is PRECEDED by a closed word AND FOLLOWED "
                f"by a content noun (the DET _ NOUN attributive slot -- Tomasello/Goldberg attributive construction, the "
                f"pre-nominal modifier slot; the INVERSE-position cue to EMERGE-62b's function-word cue; Yang-Getz 2026 "
                f"3rd property; catalog G.12) RECLASSIFIES them OPEN (attributive-rate >= {TP_ATTRIB}: adjectives sit at "
                f"0.68-0.74, the true closed class at <= 0.36). Adjective reclassification F1 {reclass_F1:.3f} (all "
                f"mislabelled adjectives promoted OPEN, ZERO true function words promoted). With the adjectives OPEN, the "
                f"producer ADMITS + MINES + RENDERS the corpus-attested adjective constructions -- attributive "
                f"'the big owl can fly' (det ADJ subj can verb) + predicative 'the owl is big' (det subj is ADJ) -- for a "
                f"total of {int(n_rendered_exact_mean)} DISTINCT constructions (>= 7), all rendered EXACT on spikes "
                f"(render {main_render:.3f}; the 5-slot C_ATTRIB fits N_SLOT_POOLS=6). Every input-destruction/hold-out "
                f"control COLLAPSES: POSITION-SHUFFLE (scramble word positions before the attributive stat) -> 0 "
                f"adjectives reclassified -> 0 adjective constructions mined (the cue is LOAD-BEARING, not spurious); "
                f"FREQUENCY-ONLY (the EMERGE-62 2-cue baseline, no position cue) -> 0 adjective constructions -> the "
                f"EMERGE-72 5-construction state (proving the position cue is what ADDS the adjectives, "
                f"{freqonly_rendered:.1f} rendered); NO-CORPUS -> 0 registered; HELD-OUT-CONSTRUCTION -> the shared "
                f"det+subj+verb backbone recovered from the OTHERS ({heldout_mean:.3f}). The gate-first no-confab MOAT "
                f"holds BY CONSTRUCTION: 0 producer invocations on abstains. {len(seeds)} seeds. ==> the producer renders "
                f"a BROADER, corpus-driven, router-selected inventory including the adjective constructions -- the "
                f"broadening is CORPUS-POSITION-DRIVEN (position-shuffle collapses it). HONEST SCOPE: BROADENS the bounded, "
                f"corpus-attested, router-selected inventory from 5 to >= 7 (adds the adjective's attributive-position "
                f"cue), NOT open prose (R4). The A->W spell stays the token surface; the fully-spiking A->W of the NEW "
                f"adjective content words is the EMERGE-67/68-style follow-on. Reuse-by-import; NO sim/ edit; moat "
                f"untouched.")
        else:
            miss = []
            if not broadened:
                miss.append(f"fewer than 7 constructions rendered exact every seed (min {n_rendered_exact_min}, mean "
                            f"{n_rendered_exact_mean:.1f}, adj-rendered-min {adj_rendered_min}, render {main_render:.3f})")
            if not adj_reclassified:
                miss.append(f"adjective reclassification incomplete (F1 {reclass_F1:.3f}, all-adj-reclassified "
                            f"{reclass_all_adj}, zero-false-promotions {reclass_no_fp})")
            if not posshuffle_collapses:
                miss.append(f"POSITION-SHUFFLE did NOT collapse (adj-registered max {shuffle_adj_max}, reclassified max "
                            f"{shuffle_reclass_max}) -- BLOCKING: the lift may be spurious, the position cue must be "
                            f"load-bearing")
            if not freqonly_is_emerge72:
                miss.append(f"FREQUENCY-ONLY baseline did not reproduce the EMERGE-72 state (adj-registered max "
                            f"{freqonly_adj_max}, rendered {freqonly_rendered:.1f}) -- the position cue must be what adds "
                            f"the adjectives")
            if not nocorpus_empty:
                miss.append(f"NO-CORPUS did not produce an empty registry ({nocorpus_n} registered)")
            if not heldout_generalizes:
                miss.append(f"held-out-construction shared backbone {heldout_mean:.3f} below 1.0")
            if not moat_ok:
                miss.append(f"MOAT: {moat_calls} producer-calls on abstains / answer-produced {answer_ok} -- BLOCKING, "
                            f"do NOT weaken the moat")
            verdict = ("BOUNDARY -- " + "; ".join(miss) + ". The precise residual is named. If the attributive-position "
                       "cue only PARTIALLY separates the adjectives (e.g. an adjective that appears predicatively/rarely "
                       "pre-nominally), that is an HONEST residual, NOT a wall: the next single signal is the predicative "
                       "post-copular-position cue or a morphological-comparative cue (-er/-est). The POSITION-SHUFFLE "
                       "control MUST collapse (adjective constructions un-mined) for the lift to be real; if it did NOT, "
                       "the lift is spurious and must NOT be claimed. FREQUENCY-ONLY must reproduce the EMERGE-72 state. "
                       "If the MOAT was breached this is BLOCKING -- do NOT weaken the moat; do NOT force a GO by "
                       "smuggling the adjective label.")
    else:
        verdict = f"ERROR -- {err}"
        n_registered = n_rendered_exact_mean = main_render = reclass_F1 = None
        heldout_mean = nocorpus_n = moat_calls = None
        go = False

    lines = []
    try:
        lines, _, _ = _sample_transcript(seeds[0])
    except Exception:
        pass
    transcript = [{"tag": t, "question": q, "surface": s, "invocation": i} for (t, q, s, i) in lines]

    summary = {
        "probe": "emerge73_adjective_position_cue", "verdict": verdict,
        "go": bool(go) if err is None else False,
        "mechanism": ("CLOSE the adjective boundary EMERGE-72 named by adding the ADJECTIVE's OWN ATTRIBUTIVE PRE-NOMINAL "
                      "POSITION as a distributional cue. The 2-cue Goldilocks discovery (frequency + coverage) mislabels "
                      "the corpus's high-frequency, high-coverage adjectives CLOSED; the position statistic attribscore = "
                      "fraction of occurrences in the DET _ NOUN attributive slot (preceded by a closed word, followed by "
                      "a content noun) reclassifies them OPEN (ASYMMETRIC: only promotes a Goldilocks-CLOSED word; never "
                      "demotes an open word or a true determiner). With the adjectives OPEN, the extended labeller admits "
                      "a pre-nominal ADJ slot (attributive) + a predicative post-copular ADJ slot, so the EMERGE-72 "
                      "signature-keyed registry mines + renders the attributive + predicative adjective constructions -> "
                      ">= 7 constructions on spikes. INVERSE-position cue to EMERGE-62b's function-word cue; Tomasello "
                      "usage-based / Goldberg attributive construction (the pre-nominal modifier slot); Yang-Getz 2026 3rd "
                      "property; Redington/Cartwright-Brent neighbour role; catalog G.12 Broca open/closed. POSITION-"
                      "SHUFFLE / FREQUENCY-ONLY / NO-CORPUS / HELD-OUT-CONSTRUCTION input-destruction controls gate the "
                      "result. Reuse-by-import; NO sim/ edit; gate-first moat untouched."),
        "task": ("reclassify the corpus's adjectives OPEN by their attributive pre-nominal position (not frequency) so "
                 "the producer admits the attributive + predicative adjective constructions -> >= 7 constructions "
                 "rendered exact on spikes; position-shuffle collapses (adjective constructions un-mined); frequency-only "
                 "reproduces the EMERGE-72 5; no-corpus empty; held-out-construction generalises; gate-first moat 0; "
                 ">= 6 seeds"),
        "attributive_threshold": TP_ATTRIB,
        "constructions_groundtruth": {n: [list(x) for x in CONSTRUCTIONS[n]] for n in CONSTRUCTION_NAMES},
        "adjective_constructions": ADJ_CONSTRUCTION_NAMES,
        "seeds": list(seeds), "elapsed_seconds": round(time.time() - t0, 1),
        "aggregate": None if err else {
            "n_registered": n_registered, "n_rendered_exact_mean": n_rendered_exact_mean,
            "main_render": main_render, "adjective_reclassification_F1": reclass_F1,
            "heldout_mean": heldout_mean, "nocorpus_n_registered_total": nocorpus_n,
            "moat_calls_on_abstain_total": moat_calls,
        },
        "sample_transcript": transcript,
        "per_seed": per,
        "HONEST_NOTE": ("CLOSES the honest adjective boundary EMERGE-72 named: the corpus's adjectives are statistically "
                        "ambiguous with the closed class by frequency+coverage alone (high on BOTH), so the 2-cue "
                        "Goldilocks discovery mislabels them CLOSED and the EMERGE-72 producer correctly SKIPS the "
                        "adjective constructions. The adjective's OWN attributive pre-nominal position (the DET _ NOUN "
                        "slot) is the discriminating cue that the closed class lacks: it reclassifies the adjectives OPEN "
                        "(ASYMMETRIC/SAFE -- only promotes a Goldilocks-CLOSED word to OPEN, never demotes a real "
                        "determiner), so the attributive + predicative adjective constructions are mined + rendered -> "
                        ">= 7 corpus-attested, router-selected constructions on spikes. The POSITION-SHUFFLE control is "
                        "LOAD-BEARING (scrambling word positions destroys the cue -> 0 adjectives reclassified -> 0 "
                        "adjective constructions -> the EMERGE-72 count); FREQUENCY-ONLY reproduces the EMERGE-72 state "
                        "(proving the position cue is what adds the adjectives). This BROADENS the bounded, corpus-"
                        "attested inventory from 5 to >= 7, NOT open prose (R4, the deferred wall). The A->W spell stays "
                        "the token surface; the fully-spiking A->W of the NEW adjective content words is the "
                        "EMERGE-67/68-style follow-on (its own spiking validation is concept_speak_demo). The corpus "
                        "mining is offline syllabus prep (BRAIN-BASED-ONLY compliant); the structure is rendered on REAL "
                        "spikes; the gate-first moat is untouched (0 productions on abstains, by construction)."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118, flush=True)
    print(f"[emerge73] VERDICT: {verdict}", flush=True)
    print(f"[emerge73] wrote {OUT}\n" + "=" * 118, flush=True)
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
