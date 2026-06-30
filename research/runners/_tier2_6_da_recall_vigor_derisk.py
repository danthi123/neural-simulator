"""Tier-2 #6 DA-gated RECALL-VIGOR de-risk: does a value/salience PRIOR carried by the SHARED SPIKING DOPAMINE
determine WHICH of two familiarity-cleared stored facts is RETRIEVED from the conversational composer's cue-match scan
-- load-bearing for the read (DA-lesion kills it, equal-value is neutral, permuted follows DA), with the no-confab moat
intact?

THE MECHANISM (deep-research scoping research/findings/2026-06-30-tier2-6-limbic-to-composer-scoping.md, Option 1):
biology does NOT make a memory robust by a uniform stored-magnitude scalar (the CLOSED-but-MODEST Route-B encoding
gain). It scales the DRIFT RATE / VIGOR of the RETRIEVAL decision (Niv-2007 tonic-DA response vigor; Kandel/catalog
O.19 + G.16 "value scales the accumulator drift rate"; Lisman-Grace salience-gated retrieval). The composer's read is
exactly a decision over stored facts (the cue-match scan picks WHICH stored fact answers) -- so the biologically-
correct, behaviorally-SENSITIVE DA lever is on the RETRIEVAL decision, not the stored magnitude.

WHAT THE DA PRIOR BIASES (exactly): the cue-match scan ranks the facts whose cue role(s) DECODE-MATCH the cue (the
familiarity-gated candidate set). This de-risk adds, per candidate fact i, a VALUE-WEIGHTED retrieval score
    score'_i  =  match_score_i  +  beta * (DA - DA_baseline) * value_i
where  match_score_i = the cue-role's matched-filter cleanup confidence (mean-cos; the composer's OWN _cleanup_all_
scored, on the substrate unbind),  value_i = the fact's stored value/salience tag,  DA = the live shared dopamine
(get_concentration("dopamine") in deployment; a probe scalar / a real CPU SNc-bridge read in the de-risk), and beta =
the prior strength. The winner = argmax_i score'_i; its patient is recalled.

MOAT-SAFE BY CONSTRUCTION: the candidate set is the EXACT-MATCH (familiarity-gated) set -- a fact enters ONLY if its
cue role decode-matches the cue (the composer's own unbind+cleanup; this IS the no-confab moat on the RFPhasorComposer
fast path). An UNSTORED cue -> empty candidate set -> abstain (None), at EVERY DA level, regardless of beta/value. The
prior can only RE-RANK an already-matching set; it can NEVER manufacture a match for an unstored cue. The MOAT
anti-cheat below is therefore a HARD gate that holds structurally, not a tunable.

VALIDATE-BY-FUNCTION (the documented R4->R5 lesson, _navcloseout_R5_value_driven_choice.py): a VALUE-CONFLICTED cue --
two facts that BOTH clear familiarity for a single shared cue (here both share the agent) but carry DIFFERENT value.
Only the value/DA prior can disambiguate which is retrieved (the cue alone matches both; the content/first-match is a
fixed positional pick). The FOUR anti-cheats:
  (DA-LESION)  hold DA at baseline (the prior contribution -> 0) -> the value-driven pick COLLAPSES to the
               content/first-match pick (the EXTRA, value-driven recall vanishes -> the prior is the load-bearing
               signal). The decisive control.
  (EQUAL-value, the validate-by-function discriminator) both facts equal value -> the prior contributes EQUALLY -> it
               is NEUTRAL (no spurious bias); the recall equals the first-match pick even at high DA. Proves the
               lesion's effect is value-SPECIFIC, not a general lesion artifact.
  (PERMUTED)   swap which fact is high-value -> the recalled fact FOLLOWS the value (not a fixed positional/content
               bias).
  (MOAT, HARD) an UNSTORED cue returns None at BOTH DA levels (the prior re-ranks only within the gated set). Any
               breach = NEGATIVE, never a tunable.

REUSE-BY-IMPORT, NO sim/ edit: the wrapper composes RFPhasorComposer (the production test oracle + numpy-CPU path); the
inner composer holds the kb + does the on-substrate unbind/cleanup (the familiarity gate + the match scores); the
wrapper adds ONLY the read-layer value-prior re-rank over the gated candidate set. The DA SOURCE on the real claim is
the shared spiking SNc on the merged bridge (the Route-A/B `_settle_snc` recipe); the CPU smoke mocks DA with a scalar
(the same pattern the Route-B numpy de-risk used for the encoding gain) OR reads a real minimal CPU SNc bridge.

SCOPE / honesty: the prior is APPLIED by a host multiply at the READ layer (== the shipped Route-A read-layer pattern,
which reads the shared dopamine between ops; §1.5 of the scoping -- the RF ops run on the bypassed fast loop, so a live
per-op NM coupling would need a sim/ edit + risk the moat, explicitly rejected). The fully-spiking ideal (the prior
EMERGING in the cue-scan competition) is the follow-on, NOT claimed here. The DA magnitude itself is the SIGNED RPE of
the spiking SNc (`from_region_firing_signed`), SAME status as the shipped nav `dopamine->plasticity_rate` precedent.

CPU smoke (proves the WIRING + the controls compute):
    SIM_BACKEND=numpy python -m research.runners._tier2_6_da_recall_vigor_derisk --smoke
GPU 6-seed (the real claim; the controller runs this -- DO NOT run from the subagent):
    SIM_BACKEND=cupy python -m research.runners._tier2_6_da_recall_vigor_derisk --gpu --seeds 42 43 44 100 101 102
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from research.runners.rf_phasor_composer import RFPhasorComposer

# --- the value-conflicted geometry: two facts that SHARE the cue role (agent) but differ in (action, patient) and in
#     value. Cueing on the shared agent matches BOTH (both clear familiarity); only the value/DA prior disambiguates.
# EVERY word is drawn from the production composer's DEFAULT_VOCAB (rf_phasor_composer.DEFAULT_VOCAB, the 16-word probe
# set the merged bridge uses), so the SAME facts are in-codebook on BOTH the CPU oracle AND the deployed merged composer
# (the GPU path introspects agent.composer.words and asserts the picked words are present -- see _pick_conflict_facts).
VOCAB = ["dog", "cat", "go", "run", "come", "stop", "look", "north", "south", "east", "west",
         "apple", "river", "big", "small", "hot", "cold"]          # == rf_phasor_composer.DEFAULT_VOCAB
CUE_ROLE = "agent"
FACT_HI = {"agent": "dog", "action": "go", "patient": "north"}     # the (intended) HIGH-value memory
FACT_LO = {"agent": "dog", "action": "run", "patient": "south"}    # the (intended) LOW-value memory (shares agent=dog)
UNSTORED_CUE = "cat"                                               # an agent NEVER stored -> the moat probe (in-vocab)


def _pick_conflict_facts(vocab, cue_role=CUE_ROLE):
    """Derive the value-conflict facts from an ARBITRARY composer vocab (the GPU-path robustness the coordinator asked
    for): pick an agent + a shared action + two DISTINCT patients + an UNSTORED agent cue, ALL in `vocab`, so both
    facts share the cue-role word and differ only in (action, patient). PREFERS the module defaults (FACT_HI/FACT_LO/
    UNSTORED_CUE) when they are all in-vocab (the common case, since they are DEFAULT_VOCAB words); else picks the first
    suitable in-vocab words deterministically. Returns (fact_hi, fact_lo, unstored_cue). Asserts >=5 distinct words so
    a valid agent/2-actions/2-patients/unstored selection always exists."""
    vset = set(vocab)
    default_words = {FACT_HI["agent"], FACT_HI["action"], FACT_HI["patient"],
                     FACT_LO["action"], FACT_LO["patient"], UNSTORED_CUE}
    if default_words <= vset:
        return dict(FACT_HI), dict(FACT_LO), UNSTORED_CUE
    ws = sorted(vset)
    assert len(ws) >= 5, f"vocab too small ({len(ws)}) to build a value-conflict (need >= 5 distinct words)"
    agent, action_hi, action_lo, patient_hi, patient_lo = ws[0], ws[1], ws[2], ws[3], ws[4]
    unstored = ws[5] if len(ws) > 5 else ws[1]    # an in-vocab word never used as the stored agent (ws[0])
    fhi = {cue_role: agent, "action": action_hi, "patient": patient_hi}
    flo = {cue_role: agent, "action": action_lo, "patient": patient_lo}
    return fhi, flo, unstored

SEEDS = [42, 43, 44, 100, 101, 102]

# defaults (CPU smoke + GPU run share these where they apply)
DEFAULT_BETA = 8.0                 # value-prior strength: score'_i = match_i + beta*(DA-baseline)*value_i
DEFAULT_DA_BASELINE = 0.5          # the shared `dopamine` modulator baseline
DEFAULT_DA_LOW = 0.5               # baseline DA (the prior-off / lesion operating point)
DEFAULT_DA_HIGH = 0.84             # salient DA (the Route-A/B salient operating point)
DEFAULT_VALUE_HI = 1.0
DEFAULT_VALUE_LO = 0.0


class DARecallVigorComposer:
    """A thin READ-LAYER wrapper over RFPhasorComposer (reuse-by-import; NO sim/ edit) adding the DA-gated value prior
    on the cue-match scan. The inner composer owns the kb + the on-substrate unbind/cleanup (the familiarity gate + the
    matched-filter cue-role scores); this wrapper adds ONLY `score'_i = match_i + beta*(DA-baseline)*value_i` over the
    GATED candidate set and returns the value-winner's patient. `da_fn` () -> float reads the live shared dopamine in
    deployment (a probe scalar / a real CPU SNc-bridge read in the de-risk)."""

    def __init__(self, comp: RFPhasorComposer, da_fn, beta=DEFAULT_BETA, da_baseline=DEFAULT_DA_BASELINE):
        self.comp = comp
        self.da_fn = da_fn
        self.beta = float(beta)
        self.da_baseline = float(da_baseline)
        self.values = []            # per-kb-index stored value/salience (parallel to comp.kb)

    # --- store (records the per-fact value alongside the inner composer's kb) ---
    def store_valued(self, agent, action, patient, value, polarity=None):
        """Store a fact via the inner composer (unchanged on-substrate encode) + record its value/salience tag."""
        self.comp.store(agent, action, patient, polarity=polarity)
        self.values.append(float(value))

    # --- the familiarity-gated candidate set (the no-confab moat lives HERE) ---
    def _cue_scores(self, cue_value, cue_role=CUE_ROLE):
        """For EVERY stored fact, decode its `cue_role` (the inner composer's on-substrate batched unbind + matched-
        filter cleanup) and return (decoded_word[list], match_score[list]) -- the composer's OWN op result. A fact is
        a candidate iff its decoded cue word == `cue_value` (the EXACT-match familiarity gate = the moat)."""
        comps = [comp for _f, comp in self.comp.kb]
        if not comps:
            return [], []
        rec = self.comp._unbind_all_phases(comps, cue_role)       # (K, D) recovered phases on the substrate
        words, scores = self.comp._cleanup_all_scored(rec)        # decoded cue word + matched-filter confidence per fact
        return words, scores

    def candidate_indices(self, cue_value, cue_role=CUE_ROLE):
        """The familiarity-gated candidate set: indices of facts whose `cue_role` decode-matches `cue_value`. An
        unstored cue -> [] (the moat: nothing matches -> nothing to re-rank -> abstain)."""
        words, _scores = self._cue_scores(cue_value, cue_role)
        return [i for i, w in enumerate(words) if w == cue_value]

    def first_match_patient(self, cue_value, cue_role=CUE_ROLE):
        """The content / FIRST-match pick: the patient of the first (lowest kb-index) candidate fact, decoded from the
        substrate unbind -- the answer the prior-OFF read returns (the value-INDEPENDENT baseline)."""
        cand = self.candidate_indices(cue_value, cue_role)
        if not cand:
            return None
        return self.comp.unbind(self.comp.kb[cand[0]][1], "patient")

    # --- the DA-gated value-prior re-rank (the mechanism) ---
    def _ranked(self, cue_value, cue_role=CUE_ROLE):
        """Return (winner_idx, ranking_rows) where ranking_rows is a list of dicts per candidate with its match score,
        value, and value-weighted score'. The winner = argmax score'. None winner = abstain (empty gated set)."""
        words, scores = self._cue_scores(cue_value, cue_role)
        cand = [i for i, w in enumerate(words) if w == cue_value]
        if not cand:
            return None, []
        da = float(self.da_fn())
        da_term = self.beta * (da - self.da_baseline)             # the DA gate on the value prior (Niv vigor / drift)
        rows = []
        for i in cand:
            val = self.values[i] if i < len(self.values) else 0.0
            sprime = float(scores[i]) + da_term * float(val)
            rows.append({"idx": i, "match_score": float(scores[i]), "value": float(val), "score_prime": sprime})
        # argmax score'; ties (e.g. prior off, equal scores) resolve to the FIRST candidate (kb order) -> the value-
        # independent first-match baseline (so DA-lesion / equal-value deterministically yield the first-match pick).
        best = max(range(len(rows)), key=lambda r: (rows[r]["score_prime"], -rows[r]["idx"]))
        # but enforce first-match on an exact tie so the prior-off baseline is the *first* candidate, not any tie:
        top = max(row["score_prime"] for row in rows)
        tied = [row["idx"] for row in rows if row["score_prime"] >= top - 1e-12]
        winner_idx = min(tied) if len(tied) > 1 else rows[best]["idx"]
        return winner_idx, rows

    def valued_recall(self, cue_value, cue_role=CUE_ROLE):
        """The DA-gated recall: pick the candidate fact maximizing score'_i = match_i + beta*(DA-baseline)*value_i and
        return its patient (decoded from the substrate unbind). None = abstain (empty gated set -> the moat). When the
        prior is off (DA == baseline) or values are equal, this is the first-match pick (value-independent)."""
        winner_idx, _rows = self._ranked(cue_value, cue_role)
        if winner_idx is None:
            return None
        return self.comp.unbind(self.comp.kb[winner_idx][1], "patient")


def build_conflict_composer(seed=42, D=64, value_hi=DEFAULT_VALUE_HI, value_lo=DEFAULT_VALUE_LO,
                            beta=DEFAULT_BETA, da_fn=None, da_baseline=DEFAULT_DA_BASELINE,
                            vocab=VOCAB, fact_hi=None, fact_lo=None):
    """Build a DARecallVigorComposer with the two value-conflicted facts stored (HI then LO, so the FIRST-match /
    content pick is deterministically the HI fact's kb-index 0 -- making the DA-lesion 'collapse to first-match' a
    crisp, value-independent baseline). `da_fn` defaults to a baseline-DA constant (prior off); pass a closure that
    returns the operating DA for a high-DA run. Used by both the CPU smoke + the GPU runner's CPU-faithful path."""
    fact_hi = fact_hi or FACT_HI
    fact_lo = fact_lo or FACT_LO
    if da_fn is None:
        da_fn = (lambda: float(da_baseline))
    comp = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    w = DARecallVigorComposer(comp, da_fn=da_fn, beta=beta, da_baseline=da_baseline)
    # store LO first then HI so the value-INDEPENDENT first-match (kb-order) baseline is the LO fact -- then the
    # high-DA value prior must OVERRIDE first-match to recall HI (a strictly harder, unambiguous demonstration: the
    # prior has to beat the positional baseline, not merely agree with it).
    w.store_valued(fact_lo["agent"], fact_lo["action"], fact_lo["patient"], value_lo)
    w.store_valued(fact_hi["agent"], fact_hi["action"], fact_hi["patient"], value_hi)
    return w


# ======================================================================================================================
# The de-risk: the four anti-cheats, per seed, on the CPU oracle (the GPU path swaps the DA source -- see run_gpu).
# ======================================================================================================================
def _anticheats(mk_normal, mk_permuted, mk_lesion_normal, mk_lesion_permuted, mk_equal, cue, hi_patient, lo_patient):
    """The four anti-cheats, expressed as a CONTENT-CONTROLLED comparison (geometry-robust; no reliance on the two
    facts having equal intrinsic match scores). Each `mk_*` is a no-arg builder for a fresh DARecallVigorComposer at a
    given (DA, value-assignment). Returns the per-condition answers + the derived anti-cheat booleans.

    The decisive design (the R4->R5 lesson, content-controlled by the PERMUTED comparison):
      - HEADLINE: at high DA, NORMAL (HI fact = high value) recalls HI's patient AND PERMUTED (LO fact = high value)
        recalls LO's patient -- the recalled fact FOLLOWS THE VALUE (a content-driven read could not flip).
      - DA-LESION (the decisive control): at baseline DA the answer is INVARIANT to the value assignment
        (lesion-normal == lesion-permuted) -- the prior is the ONLY thing that made the read value-sensitive; lesioning
        it removes ALL value sensitivity. (Whatever fact content/match-score favors is irrelevant; what matters is the
        answer no longer depends on WHICH fact is high-value.)
      - EQUAL-value (validate-by-function): at high DA with equal value the answer == the lesion answer (the prior is
        neutral) -- proving the lesion's effect is value-SPECIFIC, not a general artifact.
      - (the headline already controls content via PERMUTED; we also record the lesion's value-invariance as the lesion
        gate, which is what 'the lesion kills the value-driven advantage' means precisely.)"""
    c_norm, c_perm = mk_normal(), mk_permuted()
    c_les_n, c_les_p = mk_lesion_normal(), mk_lesion_permuted()
    c_eq = mk_equal()

    a_norm = c_norm.valued_recall(cue)
    a_perm = c_perm.valued_recall(cue)
    a_les_n = c_les_n.valued_recall(cue)
    a_les_p = c_les_p.valued_recall(cue)
    a_eq = c_eq.valued_recall(cue)

    # HEADLINE: the recalled fact FOLLOWS the value (normal->HI, permuted->LO). This is the content-controlled headline.
    headline_follows_value = (a_norm == hi_patient) and (a_perm == lo_patient)
    headline_pick_hi = (a_norm == hi_patient)                # the normal high-DA pick is the HI-value fact
    # DA-LESION: the answer is INVARIANT to the value assignment (the value sensitivity is gone).
    lesion_value_invariant = (a_les_n == a_les_p)
    # EQUAL-value: the prior is neutral -> equal == the lesion answer (the value contributes nothing).
    equal_neutral = (a_eq == a_les_n)
    # PERMUTED (the explicit flip, recorded separately): normal picks HI, permuted picks LO.
    permuted_follows_value = headline_follows_value
    return {
        "answers": {"normal": a_norm, "permuted": a_perm, "lesion_normal": a_les_n,
                    "lesion_permuted": a_les_p, "equal": a_eq},
        "headline_follows_value": bool(headline_follows_value),
        "headline_pick_hi": bool(headline_pick_hi),
        "lesion_value_invariant": bool(lesion_value_invariant),
        "equal_neutral": bool(equal_neutral),
        "permuted_follows_value": bool(permuted_follows_value),
    }


def _seed_result_cpu(seed, D, beta, da_low, da_high, value_hi, value_lo, da_baseline):
    """One seed of the CPU-oracle de-risk. The DA source is a mocked scalar (the Route-B numpy de-risk pattern). On the
    CPU oracle the codes are deterministic-per-seed so the value-conflicted geometry is reproduced exactly. Returns a
    per-seed dict with the four anti-cheats (content-controlled via _anticheats)."""
    hi_patient = FACT_HI["patient"]
    lo_patient = FACT_LO["patient"]

    def _mk(da, vhi, vlo):
        return build_conflict_composer(seed=seed, D=D, value_hi=vhi, value_lo=vlo, beta=beta,
                                       da_fn=(lambda d=da: float(d)), da_baseline=da_baseline)

    ac = _anticheats(
        mk_normal=(lambda: _mk(da_high, value_hi, value_lo)),       # HI fact = high value, high DA
        mk_permuted=(lambda: _mk(da_high, value_lo, value_hi)),     # LO fact = high value, high DA
        mk_lesion_normal=(lambda: _mk(da_low, value_hi, value_lo)), # HI = high value, baseline DA (prior off)
        mk_lesion_permuted=(lambda: _mk(da_low, value_lo, value_hi)),  # LO = high value, baseline DA (prior off)
        mk_equal=(lambda: _mk(da_high, 0.5, 0.5)),                  # equal value, high DA
        cue=FACT_HI[CUE_ROLE], hi_patient=hi_patient, lo_patient=lo_patient)

    # gate (the no-confab moat lives in the candidate set) + the match scores (recorded; the value term must dominate
    # the intrinsic match-score gap for the winner to follow value -- the report shows the gap is small vs beta*dDA).
    c_hi = _mk(da_high, value_hi, value_lo)
    cand = c_hi.candidate_indices(FACT_HI[CUE_ROLE])
    gate_correct = (sorted(cand) == [0, 1])
    first_match = c_hi.first_match_patient(FACT_HI[CUE_ROLE])
    _words, scores = c_hi._cue_scores(FACT_HI[CUE_ROLE])
    cand_scores = [float(scores[i]) for i in sorted(cand)] if gate_correct else []
    match_gap = float(abs(scores[0] - scores[1])) if len(scores) >= 2 else None

    # (MOAT, HARD) an unstored cue abstains at BOTH DA levels (the prior re-ranks only within the gated set)
    c_les = _mk(da_low, value_hi, value_lo)
    moat_low = (c_les.candidate_indices(UNSTORED_CUE) == []) and (c_les.valued_recall(UNSTORED_CUE) is None)
    moat_high = (c_hi.candidate_indices(UNSTORED_CUE) == []) and (c_hi.valued_recall(UNSTORED_CUE) is None)
    moat_ok = bool(moat_low and moat_high)

    seed_go = bool(ac["headline_follows_value"] and gate_correct and ac["lesion_value_invariant"]
                   and ac["equal_neutral"] and ac["permuted_follows_value"] and moat_ok)
    return {
        "seed": seed,
        "headline_answer": ac["answers"]["normal"], "headline_pick_hi": ac["headline_pick_hi"],
        "headline_follows_value": ac["headline_follows_value"],
        "first_match_patient": first_match, "hi_patient": hi_patient, "lo_patient": lo_patient,
        "gate_correct": gate_correct, "candidate_indices": sorted(cand), "candidate_match_scores": cand_scores,
        "match_score_gap": match_gap,
        "lesion_normal_answer": ac["answers"]["lesion_normal"],
        "lesion_permuted_answer": ac["answers"]["lesion_permuted"],
        "lesion_value_invariant": ac["lesion_value_invariant"],
        "equal_value_answer": ac["answers"]["equal"], "equal_value_neutral": ac["equal_neutral"],
        "permuted_answer": ac["answers"]["permuted"], "permuted_follows_value": ac["permuted_follows_value"],
        "moat_low_abstains": bool(moat_low), "moat_high_abstains": bool(moat_high), "moat_ok": moat_ok,
        "seed_go": seed_go,
    }


def _aggregate(results, n_required):
    """Aggregate per-seed dicts -> the multi-seed verdict (mirrors the Route-A/R5 verdict logic; MOAT is a HARD gate)."""
    n = len(results)
    n_headline = sum(r["headline_follows_value"] for r in results)
    n_gate = sum(r["gate_correct"] for r in results)
    n_lesion = sum(r["lesion_value_invariant"] for r in results)
    n_equal = sum(r["equal_value_neutral"] for r in results)
    n_perm = sum(r["permuted_follows_value"] for r in results)
    n_moat = sum(r["moat_ok"] for r in results)
    n_go = sum(r["seed_go"] for r in results)
    moat_breach = (n_moat < n)                      # ANY moat breach -> HARD-gate failure

    if moat_breach:
        verdict = "NEGATIVE"                        # the no-confab moat broke on some seed (HARD gate)
    elif (n_go >= n_required and n_headline >= n_required and n_lesion >= n_required
          and n_equal >= n_required and n_perm >= n_required):
        verdict = "GO"
    else:
        verdict = "NEGATIVE"
    return {
        "n_seeds": n, "n_required": n_required,
        "n_headline_follows_value": n_headline, "n_gate_correct": n_gate,
        "n_lesion_value_invariant": n_lesion, "n_equal_value_neutral": n_equal,
        "n_permuted_follows_value": n_perm, "n_moat_ok": n_moat, "n_seed_go": n_go,
        "moat_breach": bool(moat_breach), "verdict": verdict,
    }


def run_cpu(seeds, D, beta, da_low, da_high, value_hi, value_lo, da_baseline, n_required):
    """The CPU-oracle de-risk over `seeds` (the smoke + a CPU multi-seed sanity). Returns (per_seed, summary)."""
    per_seed = [_seed_result_cpu(s, D, beta, da_low, da_high, value_hi, value_lo, da_baseline) for s in seeds]
    summary = _aggregate(per_seed, n_required)
    return per_seed, summary


# ----------------------------------------------------------------------------------------------------------------------
# GPU path: the REAL claim -- the DA source is the SHARED spiking SNc on the merged bridge (Route-A/B `_settle_snc`).
# The composer is still the deployed RF path (here the merged agent's composer); the value-prior wrapper is the same.
# The controller runs this (NOT the subagent). It is structurally identical to run_cpu; only the DA source changes.
# ----------------------------------------------------------------------------------------------------------------------
def _seed_result_gpu(seed, D, beta, i_low, i_high, value_hi, value_lo):
    """One seed on the merged bridge with the real shared dopamine. Builds MergedNavConvAgent(co_resident_limbic=True),
    drives the SNc to two operating points (the Route-A/B `_settle_snc`), and reads the LIVE dopamine into the value
    prior's da_fn. The composer is the merged agent's RF composer (the deployed path). The four anti-cheats mirror the
    CPU path exactly (only the DA source is the real SNc). Lazy GPU-only imports so the CPU smoke never touches them."""
    import numpy as _np
    from sim.backend import get_backend
    from research.runners.nav_conv_merged_bridge import MergedNavConvAgent
    from research.runners._tier2_routeB_deployment_smoke import _settle_snc

    xp, _ = get_backend()
    agent = MergedNavConvAgent(seed=seed, co_resident_limbic=True)
    nm = agent._merged_bridge.neuromodulator_manager
    assert nm is not None and "dopamine" in nm.modulator_names(), "the shared dopamine modulator must be present"
    da_base_cfg = float(nm._config_by_name("dopamine").baseline)
    snc_idx = _np.asarray(agent._merged_bridge.region_manager.indices("limbic_snc"), dtype=_np.int64)
    snc_idx_x = xp.asarray(snc_idx)

    # the value prior wraps the DEPLOYED merged composer; da_fn reads the LIVE shared dopamine. The facts are stored
    # ONCE (LO first, HI second); the value TAGS (w.values) are toggled per condition (the stored facts never change),
    # and the DA operating point is set by driving the shared SNc -- so the content-controlled comparison holds with
    # the REAL spiking dopamine. The DA is read live by da_fn at the moment of each valued_recall.
    comp = agent.composer
    # FIX (KeyError 'see'): derive the conflict facts from the DEPLOYED composer's ACTUAL vocab (agent.composer.words),
    # NOT the hardcoded module constants -- so every fact word is in the deployed codebook (comp.store -> _encode ->
    # comp.concepts[word]). The defaults (FACT_HI/FACT_LO/UNSTORED_CUE) ARE DEFAULT_VOCAB words so they pass through
    # unchanged on the standard 16-word merged build; the helper picks in-vocab words for any non-default vocab.
    fhi, flo, unstored = _pick_conflict_facts(list(comp.words), cue_role=CUE_ROLE)
    for wd in (fhi["agent"], fhi["action"], fhi["patient"], flo["action"], flo["patient"], unstored):
        assert wd in comp.concepts, f"derived word {wd!r} not in the deployed composer codebook"
    comp.kb = []
    w = DARecallVigorComposer(comp, da_fn=(lambda: float(nm.get_concentration("dopamine"))), beta=beta)
    w.store_valued(flo["agent"], flo["action"], flo["patient"], value_lo)   # LO fact -> kb index 0
    w.store_valued(fhi["agent"], fhi["action"], fhi["patient"], value_hi)   # HI fact -> kb index 1
    hi_patient, lo_patient = fhi["patient"], flo["patient"]
    cue = fhi[CUE_ROLE]
    V_NORMAL = [value_lo, value_hi]    # kb-index 0 = LO fact, 1 = HI fact -> HI fact carries value_hi
    V_PERMUT = [value_hi, value_lo]    # swap: the LO fact now carries value_hi
    V_EQUAL = [0.5, 0.5]

    # ORDER MATTERS on the real SNc (dopamine-EMA hysteresis): the lesion (tonic) reads run FIRST, on the fresh bridge,
    # so the "prior off" reference is the agent's ACTUAL tonic DA at THAT sequence point. The salient burst then runs
    # AFTER (raising the EMA above tonic). If the high-DA burst ran first, the EMA would not relax back to tonic within
    # one settle, leaving residual DA at the "lesion" read -> the prior would not be off. Biologically correct
    # (Niv-2007 / catalog C.20): tonic DA = the running reference; the PHASIC burst ABOVE it = the salience signal that
    # scales recall vigor (da_term = beta*(DA - DA_tonic)). da_baseline is SET to the measured tonic DA so the prior is
    # exactly off at the lesion reads.

    # --- (1) LESION / TONIC DA FIRST: drive the SNc to tonic, set the reference = this tonic DA -> the prior is OFF;
    #         the answer is INVARIANT to the value assignment (normal vs permuted give the SAME answer). ---
    da_low, _r2 = _settle_snc(agent._merged_bridge, snc_idx_x, I_snc=i_low)
    w.da_baseline = float(da_low)                    # reference = the agent's actual tonic DA (prior off at tonic)
    w.values = V_NORMAL; a_les_n = w.valued_recall(cue)
    w.values = V_PERMUT; a_les_p = w.valued_recall(cue)
    moat_low = (w.candidate_indices(unstored) == []) and (w.valued_recall(unstored) is None)

    # --- (2) HIGH (SALIENT) DA: drive the SNc to the salient burst (DA above tonic) -> the prior is ON; read normal vs
    #         permuted vs equal. ---
    da_high, _r = _settle_snc(agent._merged_bridge, snc_idx_x, I_snc=i_high)
    w.values = V_NORMAL; a_norm = w.valued_recall(cue)
    cand = w.candidate_indices(cue); gate_correct = (sorted(cand) == [0, 1])
    _words, scores = w._cue_scores(cue)
    cand_scores = [float(scores[i]) for i in sorted(cand)] if gate_correct else []
    match_gap = float(abs(scores[0] - scores[1])) if len(scores) >= 2 else None
    moat_high = (w.candidate_indices(unstored) == []) and (w.valued_recall(unstored) is None)
    w.values = V_PERMUT; a_perm = w.valued_recall(cue)
    w.values = V_EQUAL;  a_eq = w.valued_recall(cue)
    w.values = V_NORMAL  # restore
    da_tonic = float(da_low)                         # the reference used (== the tonic DA)

    headline_follows_value = (a_norm == hi_patient) and (a_perm == lo_patient)
    headline_pick_hi = (a_norm == hi_patient)
    lesion_value_invariant = (a_les_n == a_les_p)
    equal_neutral = (a_eq == a_les_n)
    permuted_follows_value = headline_follows_value
    moat_ok = bool(moat_low and moat_high)
    seed_go = bool(headline_follows_value and gate_correct and lesion_value_invariant
                   and equal_neutral and permuted_follows_value and moat_ok)
    return {
        "seed": seed, "da_high": da_high, "da_low": da_low,
        "da_baseline_used": da_tonic, "da_baseline_cfg": da_base_cfg,
        "facts": {"hi": fhi, "lo": flo, "unstored_cue": unstored},
        "headline_answer": a_norm, "headline_pick_hi": headline_pick_hi,
        "headline_follows_value": headline_follows_value,
        "first_match_patient": w.first_match_patient(cue) if gate_correct else None,
        "hi_patient": hi_patient, "lo_patient": lo_patient,
        "gate_correct": gate_correct, "candidate_indices": sorted(cand), "candidate_match_scores": cand_scores,
        "match_score_gap": match_gap,
        "lesion_normal_answer": a_les_n, "lesion_permuted_answer": a_les_p,
        "lesion_value_invariant": lesion_value_invariant,
        "equal_value_answer": a_eq, "equal_value_neutral": equal_neutral,
        "permuted_answer": a_perm, "permuted_follows_value": permuted_follows_value,
        "moat_low_abstains": bool(moat_low), "moat_high_abstains": bool(moat_high), "moat_ok": moat_ok,
        "seed_go": seed_go,
    }


def run_gpu(seeds, D, beta, i_low, i_high, value_hi, value_lo, n_required):
    per_seed = [_seed_result_gpu(s, D, beta, i_low, i_high, value_hi, value_lo) for s in seeds]
    summary = _aggregate(per_seed, n_required)
    return per_seed, summary


# ======================================================================================================================
def _print_report(per_seed, summary, mode):
    print("=" * 118)
    print(f"Tier-2 #6 DA-gated RECALL-VIGOR de-risk [{mode}] -- a value/salience prior carried by the shared dopamine")
    print("  RE-RANKS which familiarity-cleared stored fact is RETRIEVED (Niv-2007 vigor / O.19 drift-rate / Lisman-Grace);")
    print("  moat-safe by construction (re-ranks ONLY within the gated set). VALIDATE-BY-FUNCTION (R4->R5).")
    print("=" * 118)
    for r in per_seed:
        da_str = (f"DA_hi={r.get('da_high', float('nan')):.3f} DA_lo={r.get('da_low', float('nan')):.3f} "
                  f"DA_tonic_ref={r.get('da_baseline_used', float('nan')):.3f}"
                  if "da_high" in r else "DA=mock")
        gap = r.get("match_score_gap")
        print(f"  seed {r['seed']}: HEADLINE normal={r['headline_answer']!r} permuted={r['permuted_answer']!r} "
              f"follows_value={int(r['headline_follows_value'])}  |  {da_str}")
        print(f"           gate_correct={int(r['gate_correct'])} cand={r['candidate_indices']} "
              f"match_scores={['%.3f' % s for s in r['candidate_match_scores']]} gap={gap:.4f}"
              if gap is not None else
              f"           gate_correct={int(r['gate_correct'])} cand={r['candidate_indices']}")
        print(f"           LESION normal={r['lesion_normal_answer']!r} permuted={r['lesion_permuted_answer']!r} "
              f"value_invariant={int(r['lesion_value_invariant'])}  |  EQUAL ans={r['equal_value_answer']!r} "
              f"neutral={int(r['equal_value_neutral'])}")
        print(f"           MOAT lo={int(r['moat_low_abstains'])} hi={int(r['moat_high_abstains'])}  ->  "
              f"seed_GO={int(r['seed_go'])}")
    print("-" * 118)
    line = (f"VERDICT [{mode}]: {summary['verdict']}  | seeds {summary['n_seeds']} (>= {summary['n_required']} required): "
            f"headline_follows_value {summary['n_headline_follows_value']} | lesion_value_invariant "
            f"{summary['n_lesion_value_invariant']} | equal_neutral {summary['n_equal_value_neutral']} | "
            f"permuted_follows {summary['n_permuted_follows_value']} | moat_ok {summary['n_moat_ok']} | "
            f"seed_GO {summary['n_seed_go']} | moat_breach={summary['moat_breach']}")
    print(line)
    print("-" * 118)
    return line


def main():
    ap = argparse.ArgumentParser(description="Tier-2 #6 DA-gated recall-vigor de-risk (value prior on the cue-scan).")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU/numpy smoke: prove the WIRING + the four anti-cheats compute (3-seed CPU oracle). "
                         "The verdict need not be GO at toy scale; the wiring + controls are the deliverable.")
    ap.add_argument("--gpu", action="store_true",
                    help="the REAL claim: the shared spiking SNc on the merged bridge drives the prior (controller-run).")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="seeds (default: smoke=[42,43,44]; gpu/full=6)")
    ap.add_argument("--D", type=int, default=None, help="phasor dim (default: smoke=64; gpu=128)")
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA, help="value-prior strength")
    ap.add_argument("--da-baseline", type=float, default=DEFAULT_DA_BASELINE)
    ap.add_argument("--da-low", type=float, default=DEFAULT_DA_LOW, help="(CPU) prior-off DA (== baseline)")
    ap.add_argument("--da-high", type=float, default=DEFAULT_DA_HIGH, help="(CPU) salient DA operating point")
    ap.add_argument("--i-low", type=float, default=80.0, help="(GPU) tonic SNc drive pA -> DA~baseline")
    ap.add_argument("--i-high", type=float, default=600.0, help="(GPU) salient SNc drive pA -> DA~0.84")
    ap.add_argument("--value-hi", type=float, default=DEFAULT_VALUE_HI)
    ap.add_argument("--value-lo", type=float, default=DEFAULT_VALUE_LO)
    ap.add_argument("--n-required", type=int, default=None, help="seeds required to pass (default: smoke=3; full=5)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.gpu:
        from sim.backend import is_gpu_backend
        assert is_gpu_backend(), "the merged-bridge SNc is GPU-validated; run with SIM_BACKEND=cupy"
        seeds = args.seeds if args.seeds is not None else SEEDS
        D = args.D if args.D is not None else 128
        n_required = args.n_required if args.n_required is not None else 5
        per_seed, summary = run_gpu(seeds, D, args.beta, args.i_low, args.i_high, args.value_hi, args.value_lo,
                                    n_required)
        mode = "GPU-merged-real-SNc"
        out = args.out or "research/findings/raw/_tier2_6_da_recall_vigor_gpu.json"
    else:
        # CPU smoke (default) -- prove the wiring + the controls; verdict need not be GO at toy scale.
        seeds = args.seeds if args.seeds is not None else ([42, 43, 44] if args.smoke else SEEDS)
        D = args.D if args.D is not None else 64
        n_required = args.n_required if args.n_required is not None else (3 if args.smoke else 5)
        per_seed, summary = run_cpu(seeds, D, args.beta, args.da_low, args.da_high, args.value_hi, args.value_lo,
                                    args.da_baseline, n_required)
        mode = "CPU-oracle-mock-DA"
        out = args.out or "research/findings/raw/_tier2_6_da_recall_vigor_cpu_smoke.json"

    line = _print_report(per_seed, summary, mode)
    payload = {"mode": mode, "config": {"seeds": seeds, "D": D, "beta": args.beta, "da_baseline": args.da_baseline,
                                        "value_hi": args.value_hi, "value_lo": args.value_lo,
                                        "n_required": n_required,
                                        "cue_role": CUE_ROLE, "fact_hi": FACT_HI, "fact_lo": FACT_LO,
                                        "unstored_cue": UNSTORED_CUE},
               "per_seed": per_seed, "summary": summary, "verdict_line": line}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
