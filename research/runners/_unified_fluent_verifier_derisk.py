"""LANE 4 DE-RISK: does the fluency trilogy COMPOSE?

THE QUESTION. Three fluency-moat residuals were each closed in isolation, each adding a declared
host lexical ROUTING rule in front of the UNCHANGED extract (NPHeadBinder) + entail
(FactStore/classify_claim) pipeline:
  - HEDGE (`_hedged_assertion_verify_derisk.py`): `route_clause()` -- subjective->bypass,
    hedge->strip+verify, else->verify.
  - SYNONYM (`_synonym_expansion_verify_derisk.py`): `expand_action_candidates()` +
    `classify_claim_synonym_aware()` -- entailment tries a lemma/synonym-expanded candidate set,
    same polarity-reject semantics as the unmodified `classify_claim`.
  - REPORTING-CLAUSE (`_reporting_clause_strip_verify_derisk.py`): `strip_reporting_frame()`
    before the SVO/NP passes -- anchored on a REPORTING_VERBS token immediately before the
    clause's first "that".
Each is GO in isolation and regression-free vs the others' styles (they were never run TOGETHER
on the same clause). The prerequisite for wiring any of them into the LIVE verifier is proving
they COMPOSE -- all three routing rules applied together, in one principled order, on a COMBINED
item set that includes CROSS-STYLE items exercising more than one rule at once (a hedged
reporting clause, a reporting clause with a synonym verb, a hedged synonym, a hedged CONFAB
reporting clause, ...).

THE UNIFIED PIPELINE -- entirely NEW orchestration code in this file; NO edit to any existing
runner. `FactStore`, `Claim`, `classify_claim`, `split_clauses`, `is_opinion`, `HEDGES`,
`extract_svo_npbind`, `NPHeadBinder`, `BridgeParser`, `decide`, `is_subjective`, `strip_hedge`,
`route_clause`, `expand_action_candidates`, `classify_claim_synonym_aware`,
`strip_reporting_frame` are ALL imported UNCHANGED from the three de-risks + their shared
dependencies and reused verbatim -- this file only ever decides the ORDER they run in.

  `verify_clause_unified(clause, parser, np_binder)` applies the rules in this order, per clause:
    (i)   SUBJECTIVE -> BYPASS. `is_subjective(clause)` is checked on the clause AS GIVEN, before
          either strip runs -- a taste/aesthetic predicate has no truth-value to check regardless
          of hedge or reporting dressing, so subjectivity must win the routing race unconditionally
          (mirrors the hedge de-risk's own priority: is_subjective is checked before strip_hedge
          there too). On a match: kind="opinion", unchecked KEEP, extraction/entailment never run.
    (ii)  STRIP A REPORTING FRAME, if present (`strip_reporting_frame`, token-anchored: the token
          immediately before the clause's first "that" must be a REPORTING_VERBS member). Done
          BEFORE the hedge strip because the anchor check is a fixed TOKEN POSITION (the word right
          before "that") that a hedge phrase living outside that immediate span cannot disturb --
          stripping the frame first also happens to remove a hedge that PRECEDED the reporting verb
          "for free" (it lived inside the dropped span), and correctly LEAVES BEHIND a hedge that
          lives INSIDE the embedded clause for the next step to find. See CROSS_STYLE_ITEMS below
          for both sub-cases, verified separately.
    (iii) STRIP A HEDGE, if present (`strip_hedge`, an UNANCHORED substring search over the
          -- by now possibly reporting-stripped -- working clause). Runs after (ii) so a hedge that
          survived the reporting strip (because it was written INSIDE the embedded clause) is still
          caught.
    (iv)  EXTRACT via the unchanged `extract_svo_npbind` (NPHeadBinder spiking NP-binding +
          BridgeParser spiking position/voice read-out) on the final working clause.
    (v)   ENTAIL with `classify_claim_synonym_aware` (synonym/lemma expansion, same-polarity
          reject) -- reused verbatim from the synonym de-risk; this is always the LAST step, so
          synonym expansion never needs to know anything about hedge/reporting routing -- it only
          ever sees a fully-stripped (agent, action, patient, negated) claim, exactly the same
          shape classify_claim_synonym_aware already handles in isolation.
  `decide_unified(claim, store)` is a byte-for-byte copy of `decide()`'s KEEP/SUPPRESS routing
  (opinion->KEEP unchecked, unparsed->SUPPRESS abstain, assertion+grounded->KEEP,
  assertion+ungrounded->SUPPRESS) with its one `classify_claim(...)` call swapped for
  `classify_claim_synonym_aware(...)` -- identical shape to the synonym de-risk's own
  `decide_synonym_aware`, since only the entailment call changes here, never the KEEP/SUPPRESS map.

AN ORDER ABLATION (verified, not asserted). `verify_clause_alt_hedge_first` runs the SAME three
rules with hedge routed via the imported `route_clause()` FIRST (its own subjective+hedge
combined call, on the raw clause) and reporting-strip SECOND, on whatever `route_clause` hands
back. Every cross-style item in this file is run through BOTH orders and diffed in main(): if they
ever disagree, that disagreement -- not a hand-wave -- is the honest answer to "does order matter."

ITEM SET. The UNION of all three de-risks' item sets (BASE_ITEMS common to all three +
SUBJECTIVE_ITEMS + NEGATED_SYNONYM_ITEMS + NESTED_REPORT_ITEMS, all imported UNCHANGED, 34 items)
PLUS 11 NEW CROSS_STYLE_ITEMS defined in this file that combine two or three rules on one clause.
REGRESSION CHECK: for each of the three original de-risks' own item sets, this file re-runs BOTH
that de-risk's OWN isolated pipeline (its exact extractor+decide combo, its own store) AND the
unified pipeline (same items, the unified store) and diffs per-item correctness -- a regression is
an item the isolated fix got right that the unified pipeline gets wrong.

Run: python -m research.runners._unified_fluent_verifier_derisk
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")   # small nets; CPU is plenty, avoids GPU init overhead

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import get_backend  # noqa: E402

from research.runners._open_text_moat_verifier_derisk import (  # noqa: E402  (REUSE UNCHANGED)
    Claim, FactStore, classify_claim, split_clauses, is_opinion, HEDGES,
)
from research.runners._spiking_np_boundary_extraction_derisk import (  # noqa: E402  (REUSE UNCHANGED)
    NPHeadBinder, extract_svo_npbind, extract_claims_npbind,
    DETERMINERS, PASSIVE_AUX, COPULA_AUX, PARTICIPLES, VERB_LEXICON,
)
from research.runners._open_text_spiking_extraction_derisk import _find_claim, NEGATORS  # noqa: E402  (REUSE UNCHANGED)
from research.runners.brain_conversational_agent import BridgeParser  # noqa: E402  (REUSE UNCHANGED)
from research.runners._fluent_paraphrase_verify_suppress_derisk import (  # noqa: E402  (REUSE UNCHANGED)
    build_store, item, ITEMS as BASE_ITEMS, decide,
)
from research.runners._hedged_assertion_verify_derisk import (  # noqa: E402  (REUSE UNCHANGED)
    is_subjective, strip_hedge, route_clause, extract_claims_hedge_aware,
    SUBJECTIVE_ITEMS, SUBJECTIVE_VERBS, SUBJECTIVE_ADJS,
    ITEMS as HEDGE_FILE_ITEMS,
)
from research.runners._synonym_expansion_verify_derisk import (  # noqa: E402  (REUSE UNCHANGED)
    expand_action_candidates, classify_claim_synonym_aware, decide_synonym_aware,
    SYNONYM_LEMMA_MAP, NEGATED_SYNONYM_ITEMS, build_store_with_negation, lemmatize_verb,
    ITEMS as SYNONYM_FILE_ITEMS,
)
from research.runners._reporting_clause_strip_verify_derisk import (  # noqa: E402  (REUSE UNCHANGED)
    strip_reporting_frame, extract_svo_npbind_reporting_aware,
    extract_claims_npbind_reporting_aware, REPORTING_VERBS,
    NESTED_REPORT_ITEMS, RELATIVE_CLAUSE_GUARD_ITEMS,
    ITEMS as REPORTING_FILE_ITEMS,
)


# ============================================================================
# 1. The unified routing pipeline. Everything it calls (is_subjective,
#    strip_reporting_frame, strip_hedge, extract_svo_npbind,
#    classify_claim_synonym_aware) is imported unchanged; this is the ONLY
#    new decision logic in this file -- the ORDER, never a new rule.
# ============================================================================

def verify_clause_unified(clause, parser, np_binder):
    """Applies all three routing rules in the declared order (module docstring):
    (i) subjective bypass -> (ii) reporting-frame strip -> (iii) hedge strip -> (iv) extract ->
    [entailment, step (v), happens in decide_unified]. Returns (claim, trace) where trace records
    which steps fired, for diagnosing composition/interference honestly."""
    trace = {"clause": clause}

    # (i) SUBJECTIVE -> BYPASS, checked on the clause exactly as given.
    if is_subjective(clause):
        trace.update(route="subjective_bypass", reporting_stripped=False,
                      hedge_stripped=False, hedge=None)
        return Claim(text=clause, kind="opinion"), trace

    working = clause

    # (ii) STRIP A REPORTING FRAME, if present.
    tokens = re.findall(r"[a-zA-Z']+", working.lower())
    embedded_tokens = strip_reporting_frame(tokens)
    reporting_stripped = embedded_tokens is not None
    if reporting_stripped:
        working = " ".join(embedded_tokens)
    trace["reporting_stripped"] = reporting_stripped

    # (iii) STRIP A HEDGE, if present, on the (possibly reporting-stripped) working clause.
    stripped, hedge = strip_hedge(working)
    hedge_stripped = hedge is not None
    if hedge_stripped:
        working = stripped
    trace["hedge_stripped"] = hedge_stripped
    trace["hedge"] = hedge
    trace["working_clause"] = working
    trace["route"] = "assertion"

    # (iv) EXTRACT via the unchanged spiking pipeline.
    parsed, meta = extract_svo_npbind(working, parser, np_binder)
    if parsed is None:
        trace["extraction"] = "unparsed"
        return Claim(text=clause, kind="unparsed"), trace
    agent, action, patient, negated = parsed
    trace["extraction"] = "parsed"
    trace["extracted_triple"] = (agent, action, patient)
    trace["negated"] = negated
    return Claim(text=clause, kind="assertion", agent=agent, action=action,
                 patient=patient, negated=negated), trace


def extract_claims_unified(paragraph, parser, np_binder):
    claims, traces = [], []
    for clause in split_clauses(paragraph):
        claim, trace = verify_clause_unified(clause, parser, np_binder)
        claims.append(claim)
        traces.append(trace)
    return claims, traces


def decide_unified(claim, store):
    """(v) ENTAIL with synonym expansion + polarity reject -- byte-for-byte the same KEEP/SUPPRESS
    routing as the imported `decide()`, with its one classify_claim(...) call swapped for the
    synonym de-risk's classify_claim_synonym_aware(...) (same swap decide_synonym_aware makes)."""
    if claim is None:
        return "ERROR", "clause_split_miss"
    if claim.kind == "opinion":
        return "KEEP", "subjective_bypass_unchecked"
    if claim.kind == "unparsed":
        return "SUPPRESS", "unparsed_abstain"
    verdict = classify_claim_synonym_aware(claim, store)
    if verdict == "grounded":
        return "KEEP", "assertion_grounded"
    return "SUPPRESS", "assertion_ungrounded"


def _find_claim_with_trace(claims, traces, text_substr):
    tl = text_substr.lower()
    for c, t in zip(claims, traces):
        if tl in c.text.lower() or c.text.lower() in tl:
            return c, t
    return None, None


# ============================================================================
# 2. The order ablation -- hedge-before-reporting, using the imported
#    route_clause() literally for the combined subjective+hedge decision,
#    THEN reporting-strip on whatever it hands back. NOT the primary
#    pipeline; used only to VERIFY (module docstring) whether order matters.
# ============================================================================

def verify_clause_alt_hedge_first(clause, parser, np_binder):
    route, wc, hedge = route_clause(clause)
    trace = {"clause": clause, "alt_route": route, "alt_hedge": hedge}
    if route == "opinion":
        trace["alt_reporting_stripped"] = False
        return Claim(text=clause, kind="opinion"), trace
    tokens = re.findall(r"[a-zA-Z']+", wc.lower())
    embedded_tokens = strip_reporting_frame(tokens)
    reporting_stripped = embedded_tokens is not None
    working = " ".join(embedded_tokens) if reporting_stripped else wc
    trace["alt_reporting_stripped"] = reporting_stripped
    trace["alt_working_clause"] = working
    parsed, meta = extract_svo_npbind(working, parser, np_binder)
    if parsed is None:
        trace["alt_extraction"] = "unparsed"
        return Claim(text=clause, kind="unparsed"), trace
    agent, action, patient, negated = parsed
    trace["alt_extraction"] = "parsed"
    trace["alt_extracted_triple"] = (agent, action, patient)
    return Claim(text=clause, kind="assertion", agent=agent, action=action,
                 patient=patient, negated=negated), trace


# ============================================================================
# 3. Item set: the UNION of all three de-risks' item sets (34 items, all
#    imported unchanged) PLUS 11 new cross-style items combining two or
#    three rules on ONE clause.
# ============================================================================

CROSS_STYLE_ITEMS = [
    item("I believe scientists confirmed that Mercury orbits the sun.",
         "cross_hedge_reporting", "faithful", "KEEP",
         note="hedge PRECEDES the reporting verb -- reporting-strip removes 'I believe scientists "
              "confirmed that' as one span, so the hedge step is a structural no-op here (it lived "
              "inside the dropped span, not because hedge-detection ran and matched nothing)"),
    item("I believe scientists confirmed that Mercury orbits Neptune.",
         "cross_hedge_reporting", "confab", "SUPPRESS",
         note="same construction, false patient"),
    item("Scientists confirmed that Mercury circles the sun.",
         "cross_reporting_synonym", "faithful", "KEEP",
         note="reporting-strip exposes 'mercury circles the sun'; synonym expansion at "
              "entailment (circle~=orbit) grounds it"),
    item("Scientists confirmed that Mercury circles Neptune.",
         "cross_reporting_synonym", "confab", "SUPPRESS"),
    item("I think whales inhale air.",
         "cross_hedge_synonym", "faithful", "KEEP",
         note="no reporting frame present -- hedge-strip exposes 'whales inhale air'; synonym "
              "expansion (inhale~=breathe) grounds it"),
    item("I think whales inhale smoke.",
         "cross_hedge_synonym", "confab", "SUPPRESS"),
    item("Scientists confirmed that I believe whales inhale air.",
         "cross_triple_hedge_in_reporting", "faithful", "KEEP",
         note="hedge lives INSIDE the reporting frame -- reporting-strip fires first (drops "
              "'scientists confirmed that'), hedge-strip THEN genuinely fires on the remainder "
              "(drops 'i believe'), THEN synonym expansion grounds 'whales inhale air' -- all "
              "three rules exercised on one clause, in the declared order"),
    item("Scientists confirmed that I believe whales inhale smoke.",
         "cross_triple_hedge_in_reporting", "confab", "SUPPRESS"),
    item("Scientists confirmed that Mercury does not circle the sun.",
         "cross_reporting_synonym_negated", "confab", "SUPPRESS",
         note="false negation of the TRUE mercury/orbits/sun fact, wrapped in a reporting frame "
              "and worded with a synonym verb -- the polarity reject must survive both strips"),
    item("I think fish do not inhale air.",
         "cross_hedge_synonym_negated", "faithful", "KEEP",
         note="true negation (fish breathe via gills, not lungs -- a stored NEGATE fact this file "
              "inherits from the synonym de-risk's store) worded with a hedge AND a synonym verb"),
    item("Scientists reported that whales are beautiful.",
         "cross_reporting_subjective", "subjective", "KEEP",
         note="subjective bypass must win even under a reporting wrapper -- is_subjective is "
              "checked on the clause AS GIVEN, before strip_reporting_frame ever runs, so this "
              "item never reaches the reporting-strip step at all"),
]

ITEMS = (BASE_ITEMS + SUBJECTIVE_ITEMS + NEGATED_SYNONYM_ITEMS + NESTED_REPORT_ITEMS
          + CROSS_STYLE_ITEMS)


# ============================================================================
# 4. Harness.
# ============================================================================

def run_pass_unified(parser, np_binder, store, items):
    rows = []
    for it in items:
        claims, traces = extract_claims_unified(it["paragraph"], parser, np_binder)
        claim, trace = _find_claim_with_trace(claims, traces, it["clause_text"])
        decision, reason = decide_unified(claim, store)
        correct = (decision == it["gold_decision"])
        row = dict(it)
        row["extracted_kind"] = claim.kind if claim is not None else None
        row["extracted_triple"] = ((claim.agent, claim.action, claim.patient)
                                    if claim is not None and claim.kind == "assertion" else None)
        row["predicted_decision"] = decision
        row["predicted_reason"] = reason
        row["correct"] = correct
        row["trace"] = trace
        rows.append(row)
    return rows


def run_pass_isolated(extractor_fn, decide_fn, store, items):
    """extractor_fn(paragraph) -> (claims, metas); decide_fn(claim, store) -> (decision, reason).
    Mirrors each de-risk's own scoring loop exactly (byte-similar to their run_pass functions)."""
    rows = []
    for it in items:
        claims, metas = extractor_fn(it["paragraph"])
        claim = _find_claim(claims, it["clause_text"])
        decision, reason = decide_fn(claim, store)
        correct = (decision == it["gold_decision"])
        row = dict(it)
        row["predicted_decision"] = decision
        row["predicted_reason"] = reason
        row["correct"] = correct
        rows.append(row)
    return rows


def style_role_stats(rows):
    """Generalized over roles {faithful, confab, subjective} -- unlike the individual de-risks
    (which each only ever saw 2 of the 3 roles), the unified item set has all three."""
    stats = {}
    for r in rows:
        st = r["subtype"]
        d = stats.setdefault(st, {})
        b = d.setdefault(r["role"], {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += int(r["correct"])
    for st, d in stats.items():
        for role, b in d.items():
            b["accuracy"] = (b["correct"] / b["n"]) if b["n"] else None
    return stats


def role_frac(rows, role):
    sub = [r for r in rows if r["role"] == role]
    n = len(sub)
    c = sum(1 for r in sub if r["correct"])
    return {"n": n, "correct": c, "accuracy": (c / n) if n else None, "frac": f"{c}/{n}"}


def diff_regression(isolated_rows, unified_rows, label):
    """Per-item diff, matched by clause_text (both lists cover the SAME item set, same order).
    regression = isolated pipeline was correct, unified pipeline is wrong."""
    assert len(isolated_rows) == len(unified_rows)
    detail, regressions, newly_fixed = [], [], []
    for iso, uni in zip(isolated_rows, unified_rows):
        assert iso["clause_text"] == uni["clause_text"]
        entry = {"paragraph": iso["paragraph"], "subtype": iso["subtype"], "role": iso["role"],
                 "gold": iso["gold_decision"],
                 "isolated_decision": iso["predicted_decision"], "isolated_correct": iso["correct"],
                 "unified_decision": uni["predicted_decision"], "unified_correct": uni["correct"]}
        if iso["correct"] and not uni["correct"]:
            entry["regression"] = True
            regressions.append(entry)
        elif (not iso["correct"]) and uni["correct"]:
            entry["regression"] = False
            entry["newly_fixed"] = True
            newly_fixed.append(entry)
        else:
            entry["regression"] = False
        detail.append(entry)
    return {"label": label, "n_items": len(detail), "n_regressions": len(regressions),
            "regression_free": len(regressions) == 0,
            "regressions": regressions, "newly_fixed": newly_fixed, "detail": detail}


def run_order_ablation(parser, np_binder, store):
    """Runs every CROSS_STYLE item through BOTH orders (declared: reporting-before-hedge, and the
    alt: hedge-before-reporting via route_clause) and diffs the resulting decision. Answers "does
    order matter" empirically instead of by argument."""
    rows = []
    agree = True
    for it in CROSS_STYLE_ITEMS:
        claims_d, traces_d = extract_claims_unified(it["paragraph"], parser, np_binder)
        claim_d, trace_d = _find_claim_with_trace(claims_d, traces_d, it["clause_text"])
        dec_d, reason_d = decide_unified(claim_d, store)

        claim_a, trace_a = verify_clause_alt_hedge_first(it["clause_text"], parser, np_binder)
        dec_a, reason_a = decide_unified(claim_a, store)

        same = (dec_d == dec_a)
        agree = agree and same
        rows.append({
            "paragraph": it["paragraph"], "subtype": it["subtype"], "gold": it["gold_decision"],
            "declared_order_decision": dec_d, "declared_order_reason": reason_d,
            "declared_order_working_clause": trace_d.get("working_clause") if trace_d else None,
            "alt_order_decision": dec_a, "alt_order_reason": reason_a,
            "alt_order_working_clause": trace_a.get("alt_working_clause"),
            "orders_agree": same,
        })
    return rows, agree


def run_guard_checks(parser, np_binder):
    """Reproduces the reporting de-risk's relative-clause guard (3 adversarial 'X that Y' items
    where 'that' is a genuine relative pronoun, not a reporting frame) INSIDE the unified
    pipeline: verify_clause_unified must reach the same extracted triple as the plain unchanged
    extract_svo_npbind on the same clause (a true no-op across all three rules, not just the
    reporting one -- none of these items contain a hedge or subjective word either, so this also
    exercises "do the OTHER two rules stay silent here" honestly, not just the reporting guard)."""
    results = []
    all_ok = True
    for g in RELATIVE_CLAUSE_GUARD_ITEMS:
        before = extract_svo_npbind(g["clause_text"], parser, np_binder)
        claim, trace = verify_clause_unified(g["clause_text"], parser, np_binder)
        after = ((claim.agent, claim.action, claim.patient, claim.negated), trace) \
            if claim.kind == "assertion" else (None, trace)
        before_triple = before[0]
        after_triple = after[0]
        identical = (before_triple == after_triple)
        ok = identical and not trace.get("reporting_stripped") and not trace.get("hedge_stripped")
        all_ok = all_ok and ok
        results.append({"paragraph": g["paragraph"], "note": g["note"],
                         "before_triple": before_triple, "after_triple": after_triple,
                         "reporting_stripped": trace.get("reporting_stripped"),
                         "hedge_stripped": trace.get("hedge_stripped"),
                         "before_after_identical": identical, "ok": ok})
    return results, all_ok


def main():
    t0 = time.time()
    store = build_store_with_negation()   # superset store: fluent base's 10 facts + the fish/breathe/air NEGATE fact
    parser = BridgeParser(seed=42)
    np_binder = NPHeadBinder(seed=42)
    build_s = time.time() - t0
    xp, backend_name = get_backend()

    # -- primary pass: the unified pipeline over the FULL combined item set --
    rows = run_pass_unified(parser, np_binder, store, ITEMS)
    stats = style_role_stats(rows)

    overall_faithful = role_frac(rows, "faithful")
    overall_confab = role_frac(rows, "confab")
    overall_subjective = role_frac(rows, "subjective")

    cross_rows = [r for r in rows if r["subtype"].startswith("cross_")]
    cross_faithful = role_frac(cross_rows, "faithful")
    cross_confab = role_frac(cross_rows, "confab")
    cross_subjective = role_frac(cross_rows, "subjective")

    nested_rows = [r for r in rows if r["subtype"] == "nested_report"]
    nested_faithful = role_frac(nested_rows, "faithful")
    nested_confab = role_frac(nested_rows, "confab")

    # -- regression checks: each de-risk's OWN item set, isolated pipeline vs unified pipeline --
    hedge_isolated = run_pass_isolated(
        lambda p: extract_claims_hedge_aware(p, parser, np_binder), decide,
        build_store(), HEDGE_FILE_ITEMS)
    hedge_unified = run_pass_unified(parser, np_binder, store, HEDGE_FILE_ITEMS)
    hedge_regression = diff_regression(hedge_isolated, hedge_unified, "hedge_file_items")

    synonym_isolated = run_pass_isolated(
        lambda p: extract_claims_npbind(p, parser, np_binder), decide_synonym_aware,
        build_store_with_negation(), SYNONYM_FILE_ITEMS)
    synonym_unified = run_pass_unified(parser, np_binder, store, SYNONYM_FILE_ITEMS)
    synonym_regression = diff_regression(synonym_isolated, synonym_unified, "synonym_file_items")

    reporting_isolated = run_pass_isolated(
        lambda p: extract_claims_npbind_reporting_aware(p, parser, np_binder), decide,
        build_store(), REPORTING_FILE_ITEMS)
    reporting_unified = run_pass_unified(parser, np_binder, store, REPORTING_FILE_ITEMS)
    reporting_regression = diff_regression(reporting_isolated, reporting_unified, "reporting_file_items")

    overall_regression_free = (hedge_regression["regression_free"]
                                and synonym_regression["regression_free"]
                                and reporting_regression["regression_free"])

    # -- order ablation + relative-clause guard, run INSIDE the unified pipeline --
    ablation_rows, ablation_agree = run_order_ablation(parser, np_binder, store)
    guard_results, guard_all_ok = run_guard_checks(parser, np_binder)

    knobs = {
        "seed": 42, "backend": backend_name, "sim_backend_env": os.environ.get("SIM_BACKEND"),
        "neuron_model": "IZHIKEVICH",
        "parser_class": "BridgeParser (unchanged, reused, brain_conversational_agent.py)",
        "np_binder_class": "NPHeadBinder (unchanged, reused, _spiking_np_boundary_extraction_derisk.py)",
        "unified_extractor": "extract_claims_unified (NEW, this file -> verify_clause_unified per clause)",
        "unified_decide": "decide_unified (NEW, this file -> classify_claim_synonym_aware, same KEEP/SUPPRESS map as decide())",
        "declared_order": "(i) is_subjective->bypass  (ii) strip_reporting_frame  (iii) strip_hedge  "
                           "(iv) extract_svo_npbind  (v) classify_claim_synonym_aware",
        "entailment_module": "research.runners._open_text_moat_verifier_derisk",
        "extraction_module": "research.runners._spiking_np_boundary_extraction_derisk",
        "hedge_module": "research.runners._hedged_assertion_verify_derisk",
        "synonym_module": "research.runners._synonym_expansion_verify_derisk",
        "reporting_module": "research.runners._reporting_clause_strip_verify_derisk",
        "bridgeparser_num_neurons": parser.bridge.core_config.num_neurons,
        "npbinder_num_neurons": np_binder.bridge.core_config.num_neurons,
        "build_train_seconds": build_s,
        "n_grounded_facts": len(store.facts),
        "grounded_facts": [{"agent": a, "action": act, "patient": p, "polarity": pol}
                            for (a, act), (p, pol) in store.facts.items()],
        "hedge_phrases": list(HEDGES),
        "subjective_verbs": sorted(SUBJECTIVE_VERBS), "subjective_adjs": sorted(SUBJECTIVE_ADJS),
        "synonym_lemma_map": dict(SYNONYM_LEMMA_MAP),
        "reporting_verbs": sorted(REPORTING_VERBS),
        "determiners_stripped": sorted(DETERMINERS),
        "passive_aux": sorted(PASSIVE_AUX), "copula_aux": sorted(COPULA_AUX),
        "participles_lexicon": sorted(PARTICIPLES), "verb_lexicon_fallback": sorted(VERB_LEXICON),
        "n_items_total": len(ITEMS),
        "n_items_base": len(BASE_ITEMS), "n_items_subjective": len(SUBJECTIVE_ITEMS),
        "n_items_negated_synonym": len(NEGATED_SYNONYM_ITEMS), "n_items_nested_report": len(NESTED_REPORT_ITEMS),
        "n_items_cross_style": len(CROSS_STYLE_ITEMS),
        "n_items_hedge_file_regression_set": len(HEDGE_FILE_ITEMS),
        "n_items_synonym_file_regression_set": len(SYNONYM_FILE_ITEMS),
        "n_items_reporting_file_regression_set": len(REPORTING_FILE_ITEMS),
    }

    aggregate = {
        "overall_full_combined_set": {
            "recall_on_faithful": overall_faithful, "precision_on_confab": overall_confab,
            "subjective_bypass_correctness": overall_subjective,
        },
        "cross_style_items": {
            "recall_on_faithful": cross_faithful, "precision_on_confab": cross_confab,
            "subjective_bypass_correctness": cross_subjective,
            "n_items": len(cross_rows),
        },
        "nested_report_honest_negative": {
            "recall_on_faithful": nested_faithful, "precision_on_confab": nested_confab,
            "note": "KNOWN residual (reporting-clause de-risk): single-strip only removes the "
                    "OUTER reporting frame; a doubly-reported TRUE fact is still wrongly SUPPRESSED "
                    "via unparsed_abstain. Reproduced here unchanged (this file adds no new "
                    "reporting-frame fix) -- not counted as a regression, an inherited residual.",
        },
        "regression_vs_isolated": {
            "overall_regression_free": overall_regression_free,
            "hedge_file_items": hedge_regression,
            "synonym_file_items": synonym_regression,
            "reporting_file_items": reporting_regression,
        },
        "order_ablation": {
            "orders_agree_on_every_cross_style_item": ablation_agree,
            "items": ablation_rows,
        },
        "relative_clause_guard": {"all_ok": guard_all_ok, "items": guard_results},
        "by_style": stats,
    }

    print("=== Unified fluent verifier de-risk (does the fluency trilogy COMPOSE?) ===")
    for r in rows:
        flag = "OK" if r["correct"] else "MISS"
        print(f"  [{flag:<4}] style={r['subtype']:<30} role={r['role']:<10} "
              f"gold={r['gold_decision']:<8} pred={r['predicted_decision']:<8} "
              f"kind={r['extracted_kind']!s:<10} reason={r['predicted_reason']:<26} | {r['paragraph']}")

    print("\n=== OVERALL (full combined set, {} items) ===".format(len(ITEMS)))
    print("  recall_on_faithful:  ", json.dumps(overall_faithful))
    print("  precision_on_confab: ", json.dumps(overall_confab))
    print("  subjective_bypass:   ", json.dumps(overall_subjective))

    print("\n=== CROSS-STYLE ITEMS ({} items) ===".format(len(cross_rows)))
    for r in cross_rows:
        flag = "OK" if r["correct"] else "MISS"
        print(f"  [{flag:<4}] {r['subtype']:<30} gold={r['gold_decision']:<8} "
              f"pred={r['predicted_decision']:<8} reason={r['predicted_reason']:<26} | {r['paragraph']}")
    print("  recall_on_faithful:  ", json.dumps(cross_faithful))
    print("  precision_on_confab: ", json.dumps(cross_confab))
    print("  subjective_bypass:   ", json.dumps(cross_subjective))

    print("\n=== NESTED-REPORT HONEST NEGATIVE (inherited, not fixed here) ===")
    print("  recall_on_faithful:  ", json.dumps(nested_faithful))
    print("  precision_on_confab: ", json.dumps(nested_confab))

    print("\n=== REGRESSION vs EACH RULE IN ISOLATION ===")
    for label, reg in (("hedge", hedge_regression), ("synonym", synonym_regression),
                        ("reporting", reporting_regression)):
        print(f"  {label:<10} n_items={reg['n_items']:<3} n_regressions={reg['n_regressions']} "
              f"regression_free={reg['regression_free']}")
        for e in reg["regressions"]:
            print(f"      REGRESSION: {e['subtype']:<20} gold={e['gold']:<8} "
                  f"isolated={e['isolated_decision']:<8} unified={e['unified_decision']:<8} | {e['paragraph']}")
    print(f"\noverall_regression_free: {overall_regression_free}")

    print("\n=== ORDER ABLATION (declared reporting-then-hedge vs alt hedge-then-reporting) ===")
    for r in ablation_rows:
        flag = "AGREE" if r["orders_agree"] else "DISAGREE"
        print(f"  [{flag:<8}] {r['subtype']:<30} declared={r['declared_order_decision']:<8} "
              f"alt={r['alt_order_decision']:<8} | {r['paragraph']}")
    print(f"orders_agree_on_every_cross_style_item: {ablation_agree}")

    print("\n=== RELATIVE-CLAUSE GUARD (3 items, inside the unified pipeline) ===")
    for g in guard_results:
        print(f"  [{'OK' if g['ok'] else 'FAIL'}] reporting_stripped={g['reporting_stripped']} "
              f"hedge_stripped={g['hedge_stripped']} before==after={g['before_after_identical']} | {g['paragraph']}")
    print(f"guard all_ok: {guard_all_ok}")

    honest_verdict = {
        "rules_compose_cleanly": overall_regression_free and guard_all_ok,
        "cross_style_recall_on_faithful": cross_faithful["accuracy"],
        "cross_style_precision_on_confab": cross_confab["accuracy"],
        "nested_report_residual_inherited_unfixed": (nested_faithful["accuracy"] != 1.0),
        "order_matters": not ablation_agree,
    }
    print("\n=== HONEST VERDICT ===")
    print(json.dumps(honest_verdict, indent=2))

    out = {"knobs": knobs, "aggregate": aggregate, "honest_verdict": honest_verdict, "items": rows}
    out_path = os.path.join(_REPO, "research", "findings", "raw",
                             "_unified_fluent_verifier_derisk.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")
    return aggregate


if __name__ == "__main__":
    main()
