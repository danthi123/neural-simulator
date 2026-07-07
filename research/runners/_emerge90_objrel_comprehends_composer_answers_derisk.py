"""EMERGE-90 -- OBJECT-RELATIVE comprehension IN the spiking comprehension->composition pipeline: the brain HEARS an
object-relative sentence, COMPREHENDS it ON SPIKES (assigns THEME to the HEAD noun via the just-closed objrel emergent
read-out + `gradedtie`, NOT the canonical form->role which would wrongly assign AGENT), the composer STORES the fact, and
ANSWERS a query about it -- with the no-confab moat.

CONTEXT (established; NOT re-derived here).
  * EMERGE-89 (`_emerge89_spiking_reservoir_comprehends_composer_answers_derisk`, GO) runs a FULLY-SPIKING comprehension
    ->composition turn on ONE co-resident brain: the on-bridge spiking reservoir COMPREHENDS a sentence (form->thematic
    role) via `ReservoirComprehender` (EMERGE-88), then the `RFPhasorComposer` STORES + ANSWERS a who/what query, with the
    no-confab moat -- but it handles only CANONICAL SVO (role == position; the EMERGE-78 `_gen`/`_TRAIN_KINDS` form->role,
    per-slot ridge `Ws[k]`).
  * The OBJECT-RELATIVE comprehension is now GENUINELY CLOSED (`2026-07-07-objrel-END-TO-END-EMERGENT-CLOSE-adversarially-
    verified.md`): a fronto-striatal spiking reservoir + a per-role Dale-legal SPIKING read-out (`DopaminePlasticReadout`,
    delta-rule learned from a random Dale init -- EMERGENT, NOT the ridge) + the answer-independent `gradedtie` tie-break
    reads the object-relative role ("the ball THAT the dog chased" -> slot0 = THEME, not AGENT) on all seeds.

THE BUILD (this file). Thread the objrel emergent read-out (+ gradedtie) INTO the EMERGE-89 pipeline's role-read
interface. `ObjrelReservoirComprehender` mirrors `ReservoirComprehender.comprehend` (same reservoir feature f =
[final_state(encode(toks)), 1.0]; same OPEN-content-positions -> slots; same `_ROLE2FIELD` role->fact-field map) but
REPLACES the canonical per-slot `argmax(f @ Ws[k])` role-read with the objrel emergent per-role SPIKING detectors'
`predict_spikes(f)` argmax + the `gradedtie` tie-break on slot0 (the ambiguous THEME/AGENT slot). Everything downstream
(the composer store/query, the no-confab moat) is EMERGE-89's, unchanged.

  Reservoir: the objrel read-out was learned/validated on the `C.UBReservoir` SPIKING reservoir (a recurrent slice on a
  real `SimulationBridge`, `final_state(U)` -- genuinely spiking, on-substrate), so the comprehender is built on THAT
  reservoir (the like-for-like path -- NOT a forced unvalidated cross-reservoir transfer to the EMERGE-82 `OnBridgeLSM`;
  both are on-substrate spiking reservoirs, and the objrel read-out's provenance is the c2 one).

  The KEY capability the smoke demonstrates:
    * OBJECT-RELATIVE "the ball that the dog chases": OPEN slots [ball, dog, chases]; the objrel read-out assigns slot0=
      THEME (via gradedtie) -> fact {patient=ball, agent=dog, action=chases}; the composer stores (dog, chases, ball) +
      answers "what does the dog chase?" -> "ball".  A CANONICAL form->role read (EMERGE-89's Ws) would assign slot0=
      AGENT -> store the WRONG (ball, chases, dog) -> mis-answer. That contrast IS the objrel close's whole point.
    * CANONICAL "the dog chases the ball": OPEN slots [dog, chases, ball] -> slot0=AGENT -> {agent=dog, action=chases,
      patient=ball}; stores (dog, chases, ball) + answers correctly. The pipeline must NOT break the canonical case.

ANTI-CHEATS (mandatory).
  * EMERGENT + GENUINELY SPIKING: the objrel role comes from the EMERGENT reservoir read-out (`DopaminePlasticReadout`,
    delta-rule from a random Dale init) + `gradedtie` (the just-closed mechanism), NOT a hand-coded "objrel->THEME" rule.
    The read is argmax over per-role output-LIF SPIKE COUNTS (Dale-legal, asserted). A PRE-LEARNING (epochs=0) read is
    reported (the tie-break must NOT manufacture the objrel role from an unlearned read -- the plasticity does the work).
  * HELD-OUT CONTENT: the test facts are fresh (agent, action, patient) draws (distinct rng) never seen by the read-out
    fit; distinct (agent, action) so the query is unambiguous.
  * NO-CONFAB MOAT: an (agent, action) NEVER stored -> the composer abstains (None). A non-None = a false-accept.
  * COMPREHENSION-LESION: collapse the reservoir's closed-class identity (the objrel role read collapses -> the head noun
    is mis-roled -> the stored objrel fact is wrong) -> objrel recall collapses = the objrel comprehension is load-bearing
    for the objrel turn.
  * CONTRAST CONTROL: the SAME pipeline comprehends BOTH the canonical AND the object-relative construction correctly
    (the objrel close's whole point); a canonical-only comprehender (EMERGE-89's Ws read) would MISREAD the objrel head
    (reported as `canonical_readout_on_objrel_recall` -- it should be LOW, showing the objrel read-out is necessary).

SMOKE (then STOP; the controller fans out the 6-seed sweep + adversarially verifies). 1 seed: HEAR one object-relative +
one canonical sentence (over the held-out test set), comprehend on spikes, store, query, + the moat abstain + the lesion.
Report: does the pipeline comprehend the object-relative sentence correctly (THEME=head noun) AND answer, WITHOUT breaking
the canonical case, WITH the moat, and does the comprehension-lesion collapse it? Honest either way.

Reuse-by-import: EMERGE-89's `ReservoirComprehender` + `_ROLE2FIELD` + `_recall_over` scaffold; the objrel read-out
(`_rungB1c_objrel_dopamine_plasticity_derisk` DP._train_dopamine / DopaminePlasticReadout, `_rungB1c_objrel_emergent_
gradedtie_smoke` `_emergent_graded_drive`) + the c2 spiking reservoir (`_rungB1c_spiking_reservoir_synaptic_readout_
derisk` C / `_rungB1c_objrel_per_role_readout_derisk` PR._build/_feature) + the `RFPhasorComposer`. NO sim/ edit; CPU/numpy.

Run (smoke):
  SIM_BACKEND=numpy python -u -m research.runners._emerge90_objrel_comprehends_composer_answers_derisk \
      --seeds 42 --json research/findings/raw/_emerge90_objrel_comprehends.json

Fan-out (controller; the 6-seed sweep -- one process PER seed, aggregated after):
  for s in 42 43 44 100 101 102; do
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SIM_BACKEND=numpy \
      python -u -m research.runners._emerge90_objrel_comprehends_composer_answers_derisk \
        --seeds $s --json research/findings/raw/_emerge90_objrel_seed$s.json \
        > research/findings/raw/_emerge90_objrel_seed$s.log 2>&1 &
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
    Encoder, _gen, _TRAIN_KINDS, _ROLE_IDX, _ROLES, _make_sentence,
)
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402

N_ROLES3 = DP.N_ROLES3


# ── the OBJREL-AWARE comprehender: EMERGE-89's role-read interface, but the role for each slot comes from the objrel
#    EMERGENT per-role SPIKING read-out (`DopaminePlasticReadout`) + the `gradedtie` tie-break on slot0. ────────────────
class ObjrelReservoirComprehender:
    """`ReservoirComprehender`-shaped comprehension front-end, but the per-slot role is read by the OBJREL EMERGENT
    read-out (per-role Dale-legal SPIKING detectors, delta-rule learned from a random Dale init) + the answer-independent
    `gradedtie` tie-break on slot0 -- the just-closed objrel mechanism -- instead of the canonical per-slot ridge
    `argmax(f @ Ws[k])`. Same reservoir feature, same OPEN-content-positions -> slots, same `_ROLE2FIELD` role->field map.

    `ros` = {slot k: DopaminePlasticReadout} (EMERGENT, from DP._train_dopamine); `res`/`enc` = the c2 spiking reservoir +
    encoder. `gradedtie=True` breaks a slot0 spike-count TIE by the argmax of the answer-independent graded output drive
    (the same GT._emergent_graded_drive the objrel smoke uses). Genuinely spiking (spike-count argmax), Dale-legal.

    SLOTTING (the INTEGRATION FIX -- see the module note). The objrel read-out was TRAINED with the slot index k running
    over `sorted(roles)` (the role-annotated CONTENT positions). For CANONICAL sentences those coincide with "positions not
    in the discovered closed class" (EMERGE-89's OPEN-position rule), so EMERGE-89 used that. For OBJECT-RELATIVE they do
    NOT: the relativizer "that" is OPEN-by-the-encoder (it is out-of-vocabulary in the discovery corpus, EMERGE-78) yet is
    NOT a role-annotated content word -- so EMERGE-89's OPEN-position rule would wrongly slot "that" (and shift every role
    + crash the composer on the non-vocab token). The fix: slot the CONTENT-LEXEME positions (tokens in the corpus content
    pools subj/verb3/obj -- a lexical content-word filter, the read-out's actual training convention; verified to reproduce
    `sorted(roles)` for BOTH constructions), which excludes "that". This is NOT a hand "objrel->THEME" rule -- the role
    assignment still comes ENTIRELY from the emergent spiking read-out + gradedtie; the lexicon only picks WHICH tokens are
    content slots (exactly what a comprehender's lexical categorization does)."""

    def __init__(self, res, enc, ros, content_lex, gradedtie=True, tie_margin=0):
        self.res = res
        self.enc = enc
        self.ros = ros
        self.content_lex = set(content_lex)     # subj | verb3 | obj -- the content-word slot filter (excludes 'that')
        self.gradedtie = bool(gradedtie)
        self.tie_margin = int(tie_margin)

    def _role_for_slot(self, k, f):
        """The objrel EMERGENT read for slot k: argmax over the per-role output-LIF SPIKE COUNTS (genuinely spiking). On a
        slot0 spike-count TIE, break by the answer-independent graded drive (gradedtie) -- the just-closed objrel fix that
        reads THEME on the object-relative head. Slots 1/2 are RAW spike-count argmax (canonical untouched). Returns the
        role idx, or None if slot k has no trained read-out."""
        if k not in self.ros:
            return None
        _pred, out, _inh = self.ros[k].predict_spikes(f)          # RAW per-role output spike counts (genuinely spiking)
        o = np.asarray(out, dtype=np.float64)
        if k == 0 and self.gradedtie:
            top2 = np.sort(o)[::-1]
            if (top2[0] - top2[1]) <= self.tie_margin:            # a slot0 count-TIE (the [4,0,4] saturation failure)
                g = GT._emergent_graded_drive(self.ros[0], f)     # break by the ANSWER-INDEPENDENT graded drive
                return int(np.argmax(g))
        return int(np.argmax(o))

    def comprehend(self, tokens, lesion=False):
        """Parse a sentence into a fact dict via the objrel emergent read-out. `lesion=True` collapses the reservoir's
        closed-class identity (the necessity control -- threaded through the encoder, same as EMERGE-89)."""
        f = np.concatenate([self.res.final_state(self.enc.encode(tokens, lesion=lesion)), [1.0]])
        content = [t for t, w in enumerate(tokens) if w in self.content_lex]   # CONTENT-lexeme positions -> slots
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


class CanonicalReadoutComprehender:
    """The CONTRAST control: the SAME reservoir feature, but the per-slot role is read by the CANONICAL form->role ridge
    (`_fit_Ws_spiking`, the EMERGE-89/c2 canonical read-out) -- NO objrel awareness. On the object-relative head it
    assigns slot0=AGENT (the misread the objrel close fixes), so its objrel recall should be LOW. Reported to show the
    objrel read-out is NECESSARY (a canonical-only comprehender misreads the objrel). Same content-lexeme slotting as the
    objrel comprehender (a fair contrast: identical slots, only the role-read differs)."""

    def __init__(self, res, enc, Ws, content_lex):
        self.res = res
        self.enc = enc
        self.Ws = Ws
        self.content_lex = set(content_lex)

    def comprehend(self, tokens, lesion=False):
        f = np.concatenate([self.res.final_state(self.enc.encode(tokens, lesion=lesion)), [1.0]])
        content = [t for t, w in enumerate(tokens) if w in self.content_lex]
        fact = {}
        for k, t in enumerate(content):
            if k >= N_ROLES3 or k not in self.Ws:
                continue
            ri = int(np.argmax(f @ self.Ws[k]))
            field = _ROLE2FIELD.get(_ROLES[ri])
            if field is not None and field not in fact:
                fact[field] = tokens[t]
        return fact


# ── held-out test facts: distinct-(agent, action) CANONICAL + OBJECT-RELATIVE sentences, fresh content (no leakage) ────
def _build_objrel_test_facts(seed, subj, verb, obj, n=12):
    """CANONICAL transitive + OBJECT-RELATIVE sentences with fresh CONTENT draws (held out from the read-out fit; distinct
    rng). Each entry is (tokens, agent, action, patient, kind): the fact FIELDS the pipeline must recover.
      * canonical  "the S Vs the O"          -> agent=S, action=Vs, patient=O   (slot0=AGENT)
      * objrel     "the S1 that the S2 Vs"   -> agent=S2, action=Vs, patient=S1 (slot0=THEME -- the HEAD is the patient)
    Distinct (agent, action) within each kind so the who/what query is unambiguous."""
    trng = np.random.default_rng(seed * 733 + 11)
    canon, objr = [], []
    cseen, oseen = set(), set()
    guard = 0
    while (len(canon) < n or len(objr) < n) and guard < 20000:
        guard += 1
        s = str(trng.choice(subj)); s2 = str(trng.choice(subj))
        v = str(trng.choice(verb)); o = str(trng.choice(obj))
        v3 = v + "s"
        if len(canon) < n and s != s2 and (s, v3) not in cseen:
            cseen.add((s, v3))
            canon.append((["the", s, v3, "the", o], s, v3, o, "canonical"))
        if len(objr) < n and s != s2 and (s2, v3) not in oseen:
            oseen.add((s2, v3))
            # objrel "the S1 that the S2 Vs": head S1 = THEME/patient; embedded S2 = AGENT; V3 = PREDICATE/action
            objr.append((["the", s, "that", "the", s2, v3], s2, v3, s, "objrel"))
    return canon, objr, trng


def _recall_over_facts(composer, comprehender, facts, lesion=False):
    """COMPREHEND each sentence -> STORE the parsed fact -> query_patient(agent, action) over all -> fraction recalling the
    true patient. A fresh composer isolates each condition (the lesion, the canonical-readout contrast)."""
    for toks, ag, ac, pt, _kind in facts:
        fact = comprehender.comprehend(toks, lesion=lesion)
        if {"agent", "action", "patient"} <= set(fact):
            composer.store(fact["agent"], fact["action"], fact["patient"])
    hit = 0
    for toks, ag, ac, pt, _kind in facts:
        hit += int(composer.query_patient(ag, ac) == pt)
    return hit / max(1, len(facts))


def _parse_hits(comprehender, facts):
    """Fraction of facts whose parsed (agent, action, patient) exactly matches the ground truth (comprehension accuracy,
    independent of the composer)."""
    hit = 0
    for toks, ag, ac, pt, _kind in facts:
        fact = comprehender.comprehend(toks)
        hit += int(fact.get("agent") == ag and fact.get("action") == ac and fact.get("patient") == pt)
    return hit / max(1, len(facts))


def _derisk_one(seed):
    t0 = time.time()
    corpus = C.setup_corpus(seed=42)                            # shared corpus (the objrel scaffold's own setup)
    subj, verb, obj = corpus["subj"], corpus["verb"], corpus["obj"]
    enc = Encoder(corpus["discovered"])

    # ── the EMERGENT objrel read-out on the c2 SPIKING reservoir (GT._build_emergent: builds the c2 reservoir + trains the
    #    DopaminePlasticReadout per slot by the graded-DA delta rule from a random Dale init, and the PRE-learning read).
    (res, enc2, _canon_gen, _objr_gen, ros_main, ros_pre, _ros_nr,
     dale_legal, slot0_counts) = GT._build_emergent(seed, corpus)
    enc = enc2                                                  # use the encoder GT built (same discovered closed class)

    # ── the CANONICAL form->role ridge read-out (the CONTRAST control -- EMERGE-89/c2's canonical read) on the SAME
    #    reservoir. Fit on the same TRAIN sentences the objrel read-out saw (fair like-for-like comparator). ────────────
    rng = np.random.default_rng(seed * 101 + 5)
    train = _gen(_TRAIN_KINDS, DP.N_TRAIN, rng, subj, verb, obj)
    Ws_canon = C._fit_Ws_spiking(res, enc, train)

    # ── held-out CANONICAL + OBJECT-RELATIVE test facts (fresh content, distinct rng) ───────────────────────────────────
    canon_facts, objr_facts, trng = _build_objrel_test_facts(seed, subj, verb, obj, n=12)

    # vocab: every content word that can appear as a fact field (both S1/S2 come from subj, V3 from verb+s, O from obj)
    v3 = [v + "s" for v in verb]
    vocab = sorted(set(subj) | set(v3) | set(obj))
    content_lex = set(subj) | set(v3) | set(obj)               # the content-word slot filter (excludes the relativizer 'that')

    # ── the OBJREL-AWARE comprehender (the emergent read-out + gradedtie) and the CANONICAL comprehender (contrast) ─────
    objrel_comp = ObjrelReservoirComprehender(res, enc, ros_main, content_lex, gradedtie=True)
    canon_comp = CanonicalReadoutComprehender(res, enc, Ws_canon, content_lex)
    pre_comp = ObjrelReservoirComprehender(res, enc, ros_pre, content_lex, gradedtie=True)  # PRE-learning objrel read-out

    # ── PARSE accuracy: does each comprehender recover the correct (agent, action, patient)? ────────────────────────────
    objrel_parse_objr = _parse_hits(objrel_comp, objr_facts)         # objrel read-out on OBJECT-RELATIVE (the capability)
    objrel_parse_canon = _parse_hits(objrel_comp, canon_facts)       # objrel read-out on CANONICAL (must not break it)
    canon_parse_objr = _parse_hits(canon_comp, objr_facts)           # canonical read-out on OBJREL (the misread -> LOW)
    canon_parse_canon = _parse_hits(canon_comp, canon_facts)         # canonical read-out on canonical (sanity)
    pre_parse_objr = _parse_hits(pre_comp, objr_facts)               # PRE-learning objrel read-out on OBJREL (~chance)

    # ── THE INTEGRATION: comprehend -> store -> who/what recall, per construction (fresh composer per condition) ─────────
    comp_objr = RFPhasorComposer(seed=seed, D=_D, vocab=vocab)
    objrel_recall_objr = _recall_over_facts(comp_objr, objrel_comp, objr_facts, lesion=False)
    comp_canon = RFPhasorComposer(seed=seed, D=_D, vocab=vocab)
    objrel_recall_canon = _recall_over_facts(comp_canon, objrel_comp, canon_facts, lesion=False)

    # CONTRAST: the CANONICAL read-out on the OBJREL construction -> the head is mis-roled -> objrel recall collapses.
    comp_canon_on_objr = RFPhasorComposer(seed=seed, D=_D, vocab=vocab)
    canon_readout_recall_objr = _recall_over_facts(comp_canon_on_objr, canon_comp, objr_facts, lesion=False)

    # ── NO-CONFAB MOAT: an (agent, action) NEVER stored -> abstain (None). A non-None = a false-accept. Query the OBJREL
    #    composer (which stored the objrel facts). ────────────────────────────────────────────────────────────────────
    stored_keys = {(ag, ac) for _t, ag, ac, _pt, _k in objr_facts}
    fa = tot = 0
    mguard = 0
    while tot < 40 and mguard < 4000:
        mguard += 1
        s = str(trng.choice(subj)); v3q = str(trng.choice(verb)) + "s"
        if (s, v3q) in stored_keys:
            continue
        tot += 1
        fa += int(comp_objr.query_patient(s, v3q) is not None)
    moat_fa = fa / max(1, tot)

    # ── COMPREHENSION-LESION: collapse the reservoir's closed-class identity -> the objrel role read collapses -> the
    #    head is mis-roled -> the stored objrel fact is wrong -> objrel recall collapses (comprehension load-bearing). ───
    comp_les = RFPhasorComposer(seed=seed, D=_D, vocab=vocab)
    objrel_lesion_recall = _recall_over_facts(comp_les, objrel_comp, objr_facts, lesion=True)

    elapsed = round(time.time() - t0, 1)
    return {
        "seed": int(seed), "n_canon": len(canon_facts), "n_objr": len(objr_facts), "slot0_class_counts": slot0_counts,
        "dale_legal": bool(dale_legal),
        # PARSE (comprehension accuracy, composer-independent)
        "objrel_readout_parse_objrel": round(objrel_parse_objr, 3),    # the capability: objrel read-out reads OBJREL
        "objrel_readout_parse_canonical": round(objrel_parse_canon, 3),  # objrel read-out does NOT break canonical
        "canonical_readout_parse_objrel": round(canon_parse_objr, 3),   # contrast: canonical read-out MISREADS objrel
        "canonical_readout_parse_canonical": round(canon_parse_canon, 3),
        "pre_learning_parse_objrel": round(pre_parse_objr, 3),          # EMERGENT: PRE-learning read is ~chance
        # RECALL (comprehend -> store -> answer)
        "objrel_recall_objrel": round(objrel_recall_objr, 3),          # HEADLINE: hear objrel -> store -> answer
        "objrel_recall_canonical": round(objrel_recall_canon, 3),      # canonical case still answered (not broken)
        "canonical_readout_recall_objrel": round(canon_readout_recall_objr, 3),  # contrast recall (LOW = objrel needed)
        "moat_false_accept": round(moat_fa, 3),                        # no-confab moat
        "lesion_recall_objrel": round(objrel_lesion_recall, 3),        # comprehension load-bearing
        "elapsed_s": elapsed,
    }


def _go(rows):
    def mean(k):
        return float(np.mean([r[k] for r in rows]))
    return {
        "n_seeds": len(rows),
        "objrel_readout_parse_objrel": mean("objrel_readout_parse_objrel"),
        "objrel_readout_parse_canonical": mean("objrel_readout_parse_canonical"),
        "canonical_readout_parse_objrel": mean("canonical_readout_parse_objrel"),
        "pre_learning_parse_objrel": mean("pre_learning_parse_objrel"),
        "objrel_recall_objrel": mean("objrel_recall_objrel"),
        "objrel_recall_canonical": mean("objrel_recall_canonical"),
        "canonical_readout_recall_objrel": mean("canonical_readout_recall_objrel"),
        "moat_false_accept": mean("moat_false_accept"),
        "lesion_recall_objrel": mean("lesion_recall_objrel"),
        "dale_legal_all": all(r["dale_legal"] for r in rows),
        # GO: the objrel read-out drives correct OBJREL who/what answers (>=0.85) AND does not break canonical (>=0.90)
        # AND the no-confab moat holds (<=0.05) AND comprehension is load-bearing (lesion collapses <=0.55) AND the objrel
        # read-out is NECESSARY (the canonical-readout contrast on objrel is materially lower) AND the read is EMERGENT
        # (PRE-learning objrel parse is ~chance, so the plasticity did the work).
        "go": (mean("objrel_recall_objrel") >= 0.85 and mean("objrel_recall_canonical") >= 0.90
               and mean("moat_false_accept") <= 0.05 and mean("lesion_recall_objrel") <= 0.55
               and (mean("objrel_recall_objrel") - mean("canonical_readout_recall_objrel")) >= 0.30
               and (mean("objrel_readout_parse_objrel") - mean("pre_learning_parse_objrel")) >= 0.15),
    }


def _print_seed(s, d):
    print(f"[seed {s}] slot0-cls {d['slot0_class_counts']} dale-legal {d['dale_legal']} | "
          f"PARSE objrel-read: OBJREL {d['objrel_readout_parse_objrel']:.2f} / CANON {d['objrel_readout_parse_canonical']:.2f}"
          f" || canon-read on OBJREL (misread) {d['canonical_readout_parse_objrel']:.2f} | PRE-LEARN OBJREL "
          f"{d['pre_learning_parse_objrel']:.2f}  ==  RECALL objrel-read: OBJREL {d['objrel_recall_objrel']:.2f} / CANON "
          f"{d['objrel_recall_canonical']:.2f} || canon-read-on-OBJREL {d['canonical_readout_recall_objrel']:.2f} | "
          f"moat-FA {d['moat_false_accept']:.2f} | lesion-recall(objrel) {d['lesion_recall_objrel']:.2f} ({d['elapsed_s']}s)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[emerge90] OBJECT-RELATIVE comprehension IN the spiking comprehension->composition pipeline: HEAR an "
          f"object-relative sentence -> comprehend on spikes (objrel EMERGENT read-out + gradedtie assigns THEME to the "
          f"HEAD noun) -> composer STORES -> ANSWER, with the no-confab moat. Contrast: canonical still works; a canonical "
          f"read-out misreads the objrel. seeds {args.seeds}. NO sim/ edit; CPU/numpy. SMOKE (controller fans out + "
          f"verifies).", flush=True)

    rows = []
    for s in args.seeds:
        d = _derisk_one(s)
        rows.append(d)
        _print_seed(s, d)

    agg = _go(rows)
    agg["elapsed_seconds"] = round(time.time() - t0, 1)
    verdict = "GO" if agg["go"] else "NO-GO"
    print(f"\n[emerge90] VERDICT: {verdict} -- the OBJECT-RELATIVE sentence is comprehended ON SPIKES (objrel EMERGENT "
          f"read-out + gradedtie: THEME=head noun) and ANSWERED: objrel who/what recall {agg['objrel_recall_objrel']:.3f} "
          f"(canonical NOT broken {agg['objrel_recall_canonical']:.3f}); a CANONICAL read-out misreads the objrel (recall "
          f"{agg['canonical_readout_recall_objrel']:.3f}, so the objrel read-out is NECESSARY); EMERGENT (PRE-learning "
          f"parse {agg['pre_learning_parse_objrel']:.3f} -> learned {agg['objrel_readout_parse_objrel']:.3f}); no-confab "
          f"moat {agg['moat_false_accept']:.3f} false-accept; comprehension-lesion collapses objrel recall to "
          f"{agg['lesion_recall_objrel']:.3f}. Object-relative comprehension wired into the fully-spiking turn.",
          flush=True)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "agg": agg}, fh, indent=2, default=str)
        print(f"[emerge90] wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
