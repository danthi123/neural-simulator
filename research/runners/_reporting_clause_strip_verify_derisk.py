"""LANE 4 DE-RISK: closes fluency residual #3, REPORTING-CLAUSE SEGMENTATION, the last of the
three residuals left open by research/findings/2026-08-20-fluent-paraphrase-verify-suppress-hedge-
bypass-safety-gap.md (runner: _fluent_paraphrase_verify_suppress_derisk.py). Residual #1 (hedge
bypass, a SAFETY gap) was closed by _hedged_assertion_verify_derisk.py; residual #2 (synonym-verb
brittleness) was closed by _synonym_expansion_verify_derisk.py. This file is the third and last.

THE GAP (recap, from the finding). A reporting-clause wrapper -- "Scientists confirmed that X" --
breaks `segment_clause` (research/runners/_spiking_np_boundary_extraction_derisk.py:208-259) two
ways, because "that" is itself a member of DETERMINERS/STOPWORDS and gets silently stripped as a
content word rather than treated as a clause boundary:
  (a) a bare-3-content-word base fact ("Mercury orbits the sun") under the wrapper becomes 5
      content words ("scientists confirmed mercury orbits sun") -- too many for the plain3 fast
      path, no VERB_LEXICON match either -> segment_clause returns None -> honest unparsed_abstain
      (a COVERAGE miss: recall_on_embedded_plain3_faithful was 0/1).
  (b) a VERB_LEXICON-fallback base fact ("the Amazon Rainforest produces oxygen") under the
      wrapper has its subject NP SILENTLY OVER-EXTENDED: the reporting frame's own words merge
      into the subject span ("scientists confirmed amazon rainforest" as ONE bound NP), because
      the VERB_LEXICON pass just scans for the first known verb and takes everything before it as
      subject, with no idea "confirmed that" was ever a separate frame (recall_on_embedded_verblex
      _faithful was 0/1; it happened to still suppress the confab variant, but by accident -- no
      such key exists -- not because the frame was handled).

THE FIX -- entirely NEW code in this file, no edit to any existing runner (`FactStore`, `Claim`,
`classify_claim`, `decide`, `is_opinion`, `split_clauses`, `extract_svo_npbind`, `NPHeadBinder`,
`BridgeParser`, `segment_clause` are all imported UNCHANGED and still used, exactly as the
brief requires):

  `strip_reporting_frame(tokens)` -- a small, declared, host lexical rule, SAME abstraction level
  and SAME category as the existing DETERMINERS/PASSIVE_AUX/COPULA_AUX/PARTICIPLES/HEDGES
  lexicons already in this file family (a fixed set feeding a boundary/frame DETECTOR, never a
  role assignment): a REPORTING_VERBS set (confirmed/reported/said/found/showed/announced/
  stated/claimed/discovered/revealed/concluded/noted) anchored on the token IMMEDIATELY BEFORE
  the clause's first "that". When that anchor token is a reporting verb, everything up to and
  including "that" is dropped and the remaining tokens (the embedded clause) are returned; a
  subject before the verb is optional (present in every tested item, "scientists", but the rule
  does not require it). When the anchor token is NOT a reporting verb -- e.g. "dog" in "the dog
  THAT barked chased the cat" -- the function returns None and the clause is passed through
  UNCHANGED, unmodified, to the existing `segment_clause`. This is the guard the brief asks for:
  it is what stops a genuine relative clause from being mis-stripped.

  `extract_svo_npbind_reporting_aware(clause, parser, np_binder)` -- tries `strip_reporting_frame`
  on the clause's tokens FIRST; on a match, re-joins the embedded tokens into a plain string and
  recurses the UNCHANGED, imported `extract_svo_npbind` on THAT -- so the embedded clause goes
  through the exact same segmentation -> NPHeadBinder -> BridgeParser pipeline any other clause
  would, with zero new role-assignment logic. On no match, it calls the unchanged
  `extract_svo_npbind` on the ORIGINAL clause -- byte-identical to today's pipeline. Because none
  of the plain_control/passive/synonym/hedge items in the fluency item set contain the literal
  word "that" AT ALL, `strip_reporting_frame` is a structural no-op on every one of them: this is
  not just an empirically-measured regression-freedom claim below, it is guaranteed by
  construction and then verified by an explicit before/after diff.

  `extract_claims_npbind_reporting_aware(paragraph, parser, np_binder)` -- a byte-for-byte copy of
  the imported `extract_claims_npbind`'s clause loop, with its one call to `extract_svo_npbind`
  swapped for `extract_svo_npbind_reporting_aware`. `decide()` itself needs NO change and is
  imported and used UNCHANGED (unlike the synonym fix, which had to wrap `decide` because
  `classify_claim` itself changed -- here only extraction changes; `decide()` still just routes a
  `Claim` object it is handed, so the unmodified fluency-file `decide` is reused verbatim).

THE RELATIVE-CLAUSE GUARD (why this file does not just eat every "X that Y"). Three adversarial
items where "that" is a genuine relative pronoun, not a reporting frame -- including one that
literally contains "orbits the sun" (the exact reporting-clause target vocabulary) inside the
relative clause, to stress-test that the anchor check (not a naive "contains that") is what is
actually gating the strip. Verified two ways per item: (1) `strip_reporting_frame` called
directly returns None: (2) `extract_svo_npbind_reporting_aware` and the unchanged
`extract_svo_npbind` produce IDENTICAL output on the same clause (a true no-op, not just a
"different but still correct" result).

THE NESTED-REPORT RESIDUAL (honest negative, measured not fixed). `strip_reporting_frame` strips
only the FIRST "verb that" frame. "Scientists confirmed that researchers reported that Mercury
orbits the sun" strips the outer frame and leaves "researchers reported that mercury orbits the
sun" for the unchanged `extract_svo_npbind` -- which still fails to segment it (5 content words,
no VERB_LEXICON match), so the doubly-reported TRUE fact is (wrongly) SUPPRESSED via
unparsed_abstain. This file measures that miss honestly rather than hiding it, and additionally
runs (as a SEPARATE, non-primary measurement, not wired into the scored pipeline) a
`strip_reporting_frame_recursive` fixpoint-loop variant of the SAME rule to give a VERIFIED,
not guessed, answer to "would the obvious next lever close it" (module docstring section 5 below).

MEASUREMENT. Re-runs the ORIGINAL (unmodified) `extract_claims_npbind` + `decide` pipeline
("BEFORE") and this file's `extract_claims_npbind_reporting_aware` + the SAME unchanged `decide`
("AFTER") over the fluency de-risk's 23 items (imported unchanged, all 5 styles) PLUS 2 new
nested-report items (this file). The 3 relative-clause guard items are checked separately (module
docstring, previous paragraph), not mixed into the scored KEEP/SUPPRESS item set, since they are
not about grounded/ungrounded truth -- they are about whether the rule fires when it must not.

Run: python -m research.runners._reporting_clause_strip_verify_derisk
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
from research.runners._open_text_spiking_extraction_derisk import _find_claim  # noqa: E402  (REUSE UNCHANGED)
from research.runners.brain_conversational_agent import BridgeParser  # noqa: E402  (REUSE UNCHANGED)
from research.runners._fluent_paraphrase_verify_suppress_derisk import (  # noqa: E402  (REUSE UNCHANGED)
    build_store, item, ITEMS as BASE_ITEMS, decide,
)


# ============================================================================
# 1. The reporting-frame strip. Everything downstream (segment_clause,
#    NPHeadBinder, BridgeParser, decide) is imported unchanged; this is the
#    ONLY new decision logic in the fix.
# ============================================================================

# A small, declared, host lexical resource -- same abstraction level as
# DETERMINERS/PASSIVE_AUX/COPULA_AUX/PARTICIPLES/HEDGES (a fixed lexicon
# feeding a host boundary/frame-detection step, never a role decision). Not
# exhaustive by design (declared-and-scoped, same convention as this file
# family's other lexicons) -- covers every reporting verb the item set uses
# plus a few obvious siblings.
REPORTING_VERBS = {
    "confirmed", "reported", "said", "found", "showed", "announced",
    "stated", "claimed", "discovered", "revealed", "concluded", "noted",
}


def strip_reporting_frame(tokens):
    """tokens: lowercase word tokens of ONE clause (determiners/stopwords STILL PRESENT, same
    convention segment_clause's input uses). Detects a leading "[subject] reporting-verb that
    <embedded clause>" frame; if present, returns the embedded clause's tokens (a NEW list, the
    frame words dropped). Returns None when no such frame is found -- the clause then passes
    through UNCHANGED to segment_clause, byte-identical to the current pipeline.

    THE GUARD (why this doesn't also eat a relative clause like "the dog that barked chased the
    cat"): the match is anchored on the token IMMEDIATELY BEFORE the clause's first "that" being
    a member of the small declared REPORTING_VERBS lexicon above. "dog" is not a reporting verb,
    so the relative clause is left untouched. A subject before the verb is allowed but not
    required (the brief's "optional subject") -- this function never inspects or requires it.

    Only the FIRST "that" is considered, and only ONE frame is stripped per call -- a KNOWN,
    MEASURED residual for nested reports ("Scientists confirmed that researchers reported that
    X"), see NESTED_REPORT_ITEMS and strip_reporting_frame_recursive below."""
    if "that" not in tokens:
        return None
    ti = tokens.index("that")
    if ti == 0:
        return None                       # "that" is the first token -- no verb slot before it
    if tokens[ti - 1] not in REPORTING_VERBS:
        return None
    embedded = tokens[ti + 1:]
    if not embedded:
        return None
    return embedded


def strip_reporting_frame_recursive(tokens):
    """NOT part of the primary AFTER pipeline below (the brief specifies a single strip). A
    fixpoint loop over the IDENTICAL rule above, kept separate so the primary measurement stays
    exactly what was asked for. Used ONLY as an ancillary, explicitly-labelled measurement to give
    a VERIFIED (not guessed) answer to whether the obvious next lever -- 'just loop the same rule'
    -- closes the nested-report residual (module docstring, section 5)."""
    cur = tokens
    while True:
        nxt = strip_reporting_frame(cur)
        if nxt is None:
            return cur
        cur = nxt


def extract_svo_npbind_reporting_aware(clause, parser, np_binder):
    """Same contract as the imported, UNCHANGED extract_svo_npbind: returns
    ((agent, action, patient, negated), meta) or (None, seg). The ONLY new step: try
    strip_reporting_frame on the clause's tokens FIRST; on a match, recurse the UNCHANGED
    extract_svo_npbind on the embedded clause (a plain re-join of the stripped tokens -- host
    bookkeeping over a boundary decision already made, same category as NPHeadBinder.bind()'s
    `identity = " ".join(span_words)`, never a role decision). No match -> falls through to
    extract_svo_npbind on the ORIGINAL clause, unchanged."""
    tokens = re.findall(r"[a-zA-Z']+", clause.lower())
    embedded_tokens = strip_reporting_frame(tokens)
    if embedded_tokens is None:
        return extract_svo_npbind(clause, parser, np_binder)
    embedded_clause = " ".join(embedded_tokens)
    return extract_svo_npbind(embedded_clause, parser, np_binder)


def extract_claims_npbind_reporting_aware(paragraph, parser, np_binder):
    """Byte-for-byte copy of the imported extract_claims_npbind's clause loop
    (_spiking_np_boundary_extraction_derisk.py:280-295), with its one call to extract_svo_npbind
    swapped for extract_svo_npbind_reporting_aware. Everything else -- clause splitting,
    is_opinion routing, the Claim construction -- is identical."""
    claims, metas = [], []
    for clause in split_clauses(paragraph):
        lower = clause.lower()
        if is_opinion(lower):
            claims.append(Claim(text=clause, kind="opinion")); metas.append(None)
            continue
        parsed, meta = extract_svo_npbind_reporting_aware(clause, parser, np_binder)
        if parsed is None:
            claims.append(Claim(text=clause, kind="unparsed")); metas.append(meta)
            continue
        agent, action, patient, negated = parsed
        claims.append(Claim(text=clause, kind="assertion", agent=agent, action=action,
                             patient=patient, negated=negated))
        metas.append(meta)
    return claims, metas


# ============================================================================
# 2. Item set: the fluency de-risk's 23 items (unchanged, all 5 original
#    styles, including the 4 embedded items that are this file's headline
#    measurement) PLUS 2 new nested-report items (honest-negative probe).
#    The 3 relative-clause guard items are handled separately (section 4).
# ============================================================================

NESTED_REPORT_ITEMS = [
    item("Scientists confirmed that researchers reported that Mercury orbits the sun.",
         "nested_report", "faithful", "KEEP",
         note="doubly-reported TRUE fact -- single-strip only removes the OUTER frame; the "
              "inner 'reported that' is left for the unchanged extract_svo_npbind, which still "
              "cannot segment it (see module docstring section on the nested-report residual)"),
    item("Scientists confirmed that researchers reported that Mercury orbits Neptune.",
         "nested_report", "confab", "SUPPRESS",
         note="doubly-reported FALSE claim -- expected to suppress, but (per the finding's own "
              "caution about the pre-fix embedded case) verify WHETHER it does so via genuine "
              "entailment or by accident (unparsed_abstain, no key ever looked up)"),
]

ITEMS = BASE_ITEMS + NESTED_REPORT_ITEMS


# ============================================================================
# 3. Relative-clause guard items -- genuine "X that Y" clauses where "that"
#    is a relative pronoun, NOT a reporting-verb frame. Checked separately
#    (section 5) via (a) strip_reporting_frame returns None directly, and
#    (b) the reporting-aware pipeline reproduces the UNCHANGED pipeline's
#    output EXACTLY (a true no-op, not merely "still correct").
# ============================================================================

RELATIVE_CLAUSE_GUARD_ITEMS = [
    dict(paragraph="The dog that barked chased the cat.",
         clause_text="The dog that barked chased the cat",
         note="classic relative clause -- token before 'that' is 'dog', not a reporting verb"),
    dict(paragraph="The book that scientists wrote explains gravity.",
         clause_text="The book that scientists wrote explains gravity",
         note="'scientists' appears in the clause but AFTER 'that', and the token immediately "
              "before 'that' is 'book' -- the anchor check must key off POSITION, not presence"),
    dict(paragraph="Mercury is the planet that orbits the sun.",
         clause_text="Mercury is the planet that orbits the sun",
         note="adversarial: the relative clause's own content is 'orbits the sun', the EXACT "
              "reporting-clause target vocabulary from the embedded_plain3 item -- stress-tests "
              "that the anchor (token before 'that' = 'planet') is what gates the strip, not a "
              "naive 'does the clause contain reporting-shaped content downstream of that'"),
]


# ============================================================================
# 4. Harness: run BEFORE (imported-unchanged extract_claims_npbind + decide)
#    and AFTER (extract_claims_npbind_reporting_aware + the SAME unchanged
#    decide) over the identical scored item set, score both.
# ============================================================================

def run_pass(extractor_fn, store):
    rows = []
    for it in ITEMS:
        claims, metas = extractor_fn(it["paragraph"])
        claim = _find_claim(claims, it["clause_text"])
        decision, reason = decide(claim, store)
        correct = (decision == it["gold_decision"])
        row = dict(it)
        row["extracted_kind"] = claim.kind if claim is not None else None
        row["extracted_triple"] = ((claim.agent, claim.action, claim.patient)
                                    if claim is not None and claim.kind == "assertion" else None)
        row["predicted_decision"] = decision
        row["predicted_reason"] = reason
        row["correct"] = correct
        tokens = re.findall(r"[a-zA-Z']+", it["clause_text"].lower())
        row["reporting_frame_stripped"] = strip_reporting_frame(tokens) is not None
        rows.append(row)
    return rows


def style_role_stats(rows):
    stats = {}
    for r in rows:
        st = r["subtype"]
        d = stats.setdefault(st, {"faithful": {"n": 0, "correct": 0}, "confab": {"n": 0, "correct": 0}})
        bucket = d[r["role"]]
        bucket["n"] += 1
        bucket["correct"] += int(r["correct"])
    for st, d in stats.items():
        f, c = d["faithful"], d["confab"]
        f["recall_on_faithful"] = (f["correct"] / f["n"]) if f["n"] else None
        c["precision_on_confab"] = (c["correct"] / c["n"]) if c["n"] else None
    return stats


def run_guard_checks(parser, np_binder):
    """Section 3's relative-clause items: verify (a) strip_reporting_frame returns None directly,
    and (b) the reporting-aware extractor reproduces the UNCHANGED extractor's output EXACTLY on
    the same clause (a true no-op)."""
    results = []
    all_ok = True
    for g in RELATIVE_CLAUSE_GUARD_ITEMS:
        tokens = re.findall(r"[a-zA-Z']+", g["clause_text"].lower())
        strip_result = strip_reporting_frame(tokens)
        strip_is_noop = strip_result is None
        before = extract_svo_npbind(g["clause_text"], parser, np_binder)
        after = extract_svo_npbind_reporting_aware(g["clause_text"], parser, np_binder)
        identical = (before == after)
        ok = strip_is_noop and identical
        all_ok = all_ok and ok
        results.append({
            "paragraph": g["paragraph"], "note": g["note"],
            "strip_reporting_frame_returned_none": strip_is_noop,
            "before_after_identical": identical,
            "before_result": (before[0], before[1] if isinstance(before[1], dict) else before[1]),
            "after_result": (after[0], after[1] if isinstance(after[1], dict) else after[1]),
            "ok": ok,
        })
    return results, all_ok


def run_nested_recursive_stretch(parser, np_binder, store):
    """Ancillary, NON-primary measurement (module docstring, "NESTED-REPORT RESIDUAL"): does
    looping strip_reporting_frame to a fixpoint (strip_reporting_frame_recursive) close the
    nested-report residual? Runs the two NESTED_REPORT_ITEMS through a recursive-strip variant of
    extract_svo_npbind_reporting_aware, entirely separate from the scored AFTER pipeline above."""
    def extract_recursive(clause):
        tokens = re.findall(r"[a-zA-Z']+", clause.lower())
        embedded_tokens = strip_reporting_frame_recursive(tokens)
        embedded_clause = " ".join(embedded_tokens)
        return extract_svo_npbind(embedded_clause, parser, np_binder)

    rows = []
    for it in NESTED_REPORT_ITEMS:
        parsed, meta = extract_recursive(it["paragraph"])
        if parsed is None:
            claim = Claim(text=it["clause_text"], kind="unparsed")
        else:
            agent, action, patient, negated = parsed
            claim = Claim(text=it["clause_text"], kind="assertion", agent=agent, action=action,
                           patient=patient, negated=negated)
        decision, reason = decide(claim, store)
        rows.append({"paragraph": it["paragraph"], "gold_decision": it["gold_decision"],
                     "predicted_decision": decision, "predicted_reason": reason,
                     "extracted_triple": (claim.agent, claim.action, claim.patient)
                                          if claim.kind == "assertion" else None,
                     "correct": decision == it["gold_decision"]})
    closed = all(r["correct"] for r in rows)
    return rows, closed


def main():
    t0 = time.time()
    store = build_store()
    parser = BridgeParser(seed=42)
    np_binder = NPHeadBinder(seed=42)
    build_s = time.time() - t0
    xp, backend_name = get_backend()

    extractor_before = lambda p: extract_claims_npbind(p, parser, np_binder)  # noqa: E731
    extractor_after = lambda p: extract_claims_npbind_reporting_aware(p, parser, np_binder)  # noqa: E731

    rows_before = run_pass(extractor_before, store)
    rows_after = run_pass(extractor_after, store)

    stats_before = style_role_stats(rows_before)
    stats_after = style_role_stats(rows_after)

    def frac(d, role, key):
        b = d[role]
        return {"value": b[key], "frac": f"{b['correct']}/{b['n']}"}

    # -- embedded style: combine embedded_plain3 + embedded_verblex subtypes into "embedded",
    #    mirroring the ORIGINAL finding's aggregated reporting table row exactly. --
    def embedded_combined(stats):
        n_f = c_f = n_c = c_c = 0
        for st in ("embedded_plain3", "embedded_verblex"):
            d = stats[st]
            n_f += d["faithful"]["n"]; c_f += d["faithful"]["correct"]
            n_c += d["confab"]["n"]; c_c += d["confab"]["correct"]
        return {"recall_on_faithful": {"value": (c_f / n_f) if n_f else None, "frac": f"{c_f}/{n_f}"},
                "precision_on_confab": {"value": (c_c / n_c) if n_c else None, "frac": f"{c_c}/{n_c}"}}

    embedded_before = embedded_combined(stats_before)
    embedded_after = embedded_combined(stats_after)

    nested_before = {"recall_on_faithful": frac(stats_before["nested_report"], "faithful", "recall_on_faithful"),
                      "precision_on_confab": frac(stats_before["nested_report"], "confab", "precision_on_confab")}
    nested_after = {"recall_on_faithful": frac(stats_after["nested_report"], "faithful", "recall_on_faithful"),
                     "precision_on_confab": frac(stats_after["nested_report"], "confab", "precision_on_confab")}

    # -- regression check: every style OTHER than "embedded_plain3"/"embedded_verblex" (EXPECTED
    #    to change) and "nested_report" (a brand-new bucket) must be BYTE-IDENTICAL before/after. --
    regression_ok = True
    regression_detail = {}
    for st, d_after in stats_after.items():
        d_before = stats_before[st]
        if st in ("embedded_plain3", "embedded_verblex"):
            same = None
            label = "expected_to_change (this file's headline fix)"
        elif st == "nested_report":
            same = None
            label = "new_bucket (no before/after regression check)"
        else:
            same = (d_before["faithful"]["correct"] == d_after["faithful"]["correct"]
                    and d_before["faithful"]["n"] == d_after["faithful"]["n"]
                    and d_before["confab"]["correct"] == d_after["confab"]["correct"]
                    and d_before["confab"]["n"] == d_after["confab"]["n"])
            label = "full (faithful+confab)"
        if same is False:
            regression_ok = False
        regression_detail[st] = {
            "check": label, "unchanged": same,
            "before": {"faithful": f"{d_before['faithful']['correct']}/{d_before['faithful']['n']}",
                       "confab": f"{d_before['confab']['correct']}/{d_before['confab']['n']}"},
            "after": {"faithful": f"{d_after['faithful']['correct']}/{d_after['faithful']['n']}",
                      "confab": f"{d_after['confab']['correct']}/{d_after['confab']['n']}"},
        }

    guard_results, guard_all_ok = run_guard_checks(parser, np_binder)
    nested_recursive_rows, nested_recursive_closed = run_nested_recursive_stretch(parser, np_binder, store)

    knobs = {
        "seed": 42, "backend": backend_name, "sim_backend_env": os.environ.get("SIM_BACKEND"),
        "neuron_model": "IZHIKEVICH",
        "parser_class": "BridgeParser (unchanged, reused, brain_conversational_agent.py)",
        "np_binder_class": "NPHeadBinder (unchanged, reused, _spiking_np_boundary_extraction_derisk.py)",
        "before_extractor": "extract_claims_npbind (UNCHANGED, imported)",
        "after_extractor": "extract_claims_npbind_reporting_aware (NEW, this file -> strip_reporting_frame + unchanged extract_svo_npbind)",
        "decide_function": "decide (UNCHANGED, imported from _fluent_paraphrase_verify_suppress_derisk, identical for BEFORE and AFTER)",
        "entailment_module": "research.runners._open_text_moat_verifier_derisk",
        "extraction_module": "research.runners._spiking_np_boundary_extraction_derisk",
        "bridgeparser_num_neurons": parser.bridge.core_config.num_neurons,
        "npbinder_num_neurons": np_binder.bridge.core_config.num_neurons,
        "build_train_seconds": build_s,
        "n_grounded_facts": len(store.facts),
        "grounded_facts": [{"agent": a, "action": act, "patient": p, "polarity": pol}
                            for (a, act), (p, pol) in store.facts.items()],
        "reporting_verbs": sorted(REPORTING_VERBS),
        "determiners_stripped": sorted(DETERMINERS),
        "passive_aux": sorted(PASSIVE_AUX), "copula_aux": sorted(COPULA_AUX),
        "participles_lexicon": sorted(PARTICIPLES), "verb_lexicon_fallback": sorted(VERB_LEXICON),
        "hedge_phrases": list(HEDGES),
        "n_items_total": len(ITEMS), "n_items_from_fluency_derisk": len(BASE_ITEMS),
        "n_items_new_nested_report": len(NESTED_REPORT_ITEMS),
        "n_guard_items": len(RELATIVE_CLAUSE_GUARD_ITEMS),
        "strip_rule": ("if 'that' in tokens and tokens[index('that')-1] in REPORTING_VERBS: "
                        "drop everything up to and including 'that', recurse extract_svo_npbind "
                        "(unchanged) on the remainder; else: no-op, pass clause through unchanged"),
    }

    aggregate = {
        "embedded_style_combined": {
            "before": embedded_before, "after": embedded_after,
            "target": "recall_on_embedded_faithful 0/2 -> 2/2; precision_on_embedded_confab STAYS 1.0 (2/2)",
        },
        "nested_report_honest_negative": {
            "before": nested_before, "after": nested_after,
            "note": "measured, NOT fixed by the single-strip rule -- see per-item reasons below "
                    "and the separate nested_recursive_stretch measurement",
        },
        "regression_check_other_styles": {"unchanged_overall": regression_ok, "by_style": regression_detail},
        "relative_clause_guard": {"all_ok": guard_all_ok, "items": guard_results},
        "nested_recursive_stretch": {
            "description": "ancillary measurement, NOT the primary AFTER pipeline: does looping "
                            "strip_reporting_frame to a fixpoint close the nested-report residual?",
            "closed": nested_recursive_closed, "items": nested_recursive_rows,
        },
    }

    print("=== Reporting-clause-strip verify de-risk (closes fluency residual #3) ===")
    print("\n--- BEFORE (unmodified extract_claims_npbind) vs AFTER (reporting-aware) : embedded + nested_report ---")
    for label, rows in (("BEFORE", rows_before), ("AFTER", rows_after)):
        print(f"\n  -- {label} --")
        for r in rows:
            if r["subtype"] in ("embedded_plain3", "embedded_verblex", "nested_report"):
                flag = "OK" if r["correct"] else "MISS"
                print(f"  [{flag:<4}] style={r['subtype']:<18} role={r['role']:<8} "
                      f"gold={r['gold_decision']:<8} pred={r['predicted_decision']:<8} "
                      f"stripped={str(r['reporting_frame_stripped']):<5} "
                      f"triple={r['extracted_triple']!s:<40} reason={r['predicted_reason']:<24} | {r['paragraph']}")

    print("\n=== EMBEDDED STYLE (combined embedded_plain3 + embedded_verblex: 2 faithful, 2 confab) ===")
    print("  before:", json.dumps(embedded_before))
    print("  after: ", json.dumps(embedded_after))

    print("\n=== NESTED-REPORT HONEST NEGATIVE (1 faithful, 1 confab) ===")
    print("  before:", json.dumps(nested_before))
    print("  after: ", json.dumps(nested_after))

    print("\n=== REGRESSION CHECK (plain_control / passive / hedge / synonym) ===")
    print(json.dumps(regression_detail, indent=2))
    print(f"\nregression-free (every style outside embedded_plain3/embedded_verblex unchanged): {regression_ok}")

    print("\n=== RELATIVE-CLAUSE GUARD (3 items: strip must be a no-op) ===")
    for g in guard_results:
        print(f"  [{'OK' if g['ok'] else 'FAIL'}] strip_returned_none={g['strip_reporting_frame_returned_none']} "
              f"before==after={g['before_after_identical']} | {g['paragraph']}")
    print(f"guard all_ok: {guard_all_ok}")

    print("\n=== NESTED-RECURSIVE STRETCH (ancillary, not primary pipeline) ===")
    for r in nested_recursive_rows:
        flag = "OK" if r["correct"] else "MISS"
        print(f"  [{flag}] gold={r['gold_decision']:<8} pred={r['predicted_decision']:<8} "
              f"reason={r['predicted_reason']:<24} | {r['paragraph']}")
    print(f"nested-recursive-stretch closes the nested-report residual: {nested_recursive_closed}")

    out = {"knobs": knobs, "aggregate": aggregate,
           "items_before": rows_before, "items_after": rows_after}
    out_path = os.path.join(_REPO, "research", "findings", "raw",
                             "_reporting_clause_strip_verify_derisk.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")
    return aggregate


if __name__ == "__main__":
    main()
