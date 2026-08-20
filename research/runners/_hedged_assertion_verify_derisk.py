"""LANE 4 DE-RISK: closes the HEDGE-BYPASS SAFETY GAP found in
research/findings/2026-08-20-fluent-paraphrase-verify-suppress-hedge-bypass-safety-gap.md
(runner: _fluent_paraphrase_verify_suppress_derisk.py; underlying routing bug lives in
`is_opinion()`, research/runners/_open_text_moat_verifier_derisk.py:135-136).

THE GAP (recap). `is_opinion()` routes ANY clause containing a HEDGES phrase ("i believe" /
"i think" / "maybe" / ...) to `kind="opinion"`, and the moat's `decide()` KEEPS every opinion
clause UNCHECKED -- never runs extraction+entailment on it. A confabulated fact wrapped in a
hedge therefore sails straight through: `"I believe Darwin proposed gravity"` -> KEPT
(`reason=opinion_bypass_unchecked`) -- 0/2 precision on hedge-confab in the cited finding. A
fluent Qwen could evade the moat entirely just by hedging every claim, true or false.

THE PRINCIPLE. A hedge ("I believe X") is a STANCE MARKER on a FACTUAL assertion -- it softens
the SURFACE, it does not erase the claim's truth-value. X must still be extract+entail
VERIFIED under the hedge. The only claim that legitimately has NO truth-value to check against
the FactStore is genuinely SUBJECTIVE content -- taste / preference ("I like cats", "chocolate
is delicious") -- and THAT is the only case that should reach the unchecked opinion bypass.

THE FIX -- entirely NEW code in this file, no edit to any existing runner (`is_opinion`,
`extract_claims_npbind`, `classify_claim`, `FactStore` are all imported UNCHANGED and still used
downstream of routing):

  `route_clause(clause)` replaces the binary `is_opinion()` test with a 3-way routing rule,
  checked in this priority order:
    1. SUBJECTIVE predicate present (SUBJECTIVE_VERBS / SUBJECTIVE_ADJS, a small declared host
       lexicon -- same category of host preprocessing `segment_clause` / `is_opinion` already
       use; it never decides a grammatical ROLE or an entailment verdict, only whether the
       clause has a truth-value to check at all) -> OPINION, unchecked bypass. Checked FIRST so
       "I like cats" bypasses even with NO hedge phrase present, and "I think jazz is the best
       music" bypasses on its subjective predicate even though it also carries a hedge -- a
       taste claim has no truth-value to verify regardless of how it is hedged.
    2. else, a HEDGES phrase present -> STRIP the hedge substring (+ a trailing "that"), then
       run the SAME `extract_svo_npbind` (spiking NP-boundary extraction, unchanged) +
       `classify_claim` (FactStore entailment, unchanged) pipeline on the REMAINDER exactly as
       if it were a bare, unhedged assertion. `kind` stays "assertion"; only the surface hedge
       is discarded before parsing, never the verification.
    3. else -> unchanged: the existing `extract_svo_npbind` pipeline runs on the raw clause,
       byte-identical to what `extract_claims_npbind` already did.

Extraction (NPHeadBinder + BridgeParser, spiking) and entailment (FactStore.ask_yes_no via
classify_claim) are REUSED UNCHANGED throughout -- this file only ever replaces the ROUTING
decision (which path a clause takes BEFORE extraction), never the extraction or entailment
mechanism itself, and never touches the KEEP/SUPPRESS decision rule (`decide`, imported
unchanged from the fluency de-risk).

MEASUREMENT. Re-runs the ORIGINAL (unmodified) `extract_claims_npbind` + `decide` pipeline
("BEFORE") and this file's hedge-aware routing ("AFTER") over the SAME item set: the fluency
de-risk's 23 items (imported unchanged, all 5 original styles) PLUS 6 new genuine
subjective-opinion items (this file), so the fix can be checked against (a) the two named
hedge-confab items it must now catch, (b) the two hedge-faithful items it must keep catching,
and (c) genuinely subjective content it must NOT over-verify -- some of which also happens to
carry a hedge phrase, to prove subjectivity wins the routing race, not hedge-detection alone.

Run: python -m research.runners._hedged_assertion_verify_derisk
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
    NPHeadBinder, extract_claims_npbind, extract_svo_npbind,
    DETERMINERS, PASSIVE_AUX, COPULA_AUX, PARTICIPLES, VERB_LEXICON,
)
from research.runners._open_text_spiking_extraction_derisk import _find_claim  # noqa: E402  (REUSE UNCHANGED)
from research.runners.brain_conversational_agent import BridgeParser  # noqa: E402  (REUSE UNCHANGED)
from research.runners._fluent_paraphrase_verify_suppress_derisk import (  # noqa: E402  (REUSE UNCHANGED)
    build_store, item, ITEMS as BASE_ITEMS, decide,
)


# ============================================================================
# 1. The corrected routing rule. Everything it calls (extract_svo_npbind,
#    classify_claim, FactStore) is imported unchanged; this function is the
#    ONLY new decision logic in the whole fix.
# ============================================================================

# A small, declared, host lexical rule for "this predicate has no truth-value
# to check" -- taste / preference / aesthetic judgment. Same abstraction
# level as DETERMINERS/PASSIVE_AUX/HEDGES (a lexicon feeding a host
# preprocessing step), never used to decide a grammatical role or an
# entailment verdict, only whether a clause reaches entailment AT ALL.
SUBJECTIVE_VERBS = {"like", "likes", "love", "loves", "hate", "hates",
                     "prefer", "prefers", "enjoy", "enjoys", "adore", "adores",
                     "dislike", "dislikes", "admire", "admires"}
SUBJECTIVE_ADJS = {"delicious", "beautiful", "ugly", "boring", "fun",
                    "amazing", "best", "worst", "tasty", "gorgeous",
                    "wonderful", "terrible", "favorite", "favourite"}
SUBJECTIVE_LEXICON = SUBJECTIVE_VERBS | SUBJECTIVE_ADJS

_HEDGES_BY_LEN_DESC = sorted(HEDGES, key=len, reverse=True)  # longest-match first
_LEADING_THAT_RE = re.compile(r"^\s*,?\s*(that\s+)?", re.IGNORECASE)


def is_subjective(clause):
    """True iff the clause contains a taste/preference/aesthetic-judgment predicate --
    content with no truth-value to check against the FactStore, ever, hedge or no hedge."""
    words = set(re.findall(r"[a-zA-Z']+", clause.lower()))
    return bool(words & SUBJECTIVE_LEXICON)


def strip_hedge(clause):
    """Find the (longest-matching) HEDGES phrase in `clause` and remove it, plus a trailing
    connective 'that' ("I believe THAT X" -> "X"). Returns (stripped_clause, matched_hedge) or
    (clause, None) if no hedge phrase is present. Case-preserving on the remainder (downstream
    parsing lowercases anyway); pure substring surgery, no parsing decision made here."""
    lower = clause.lower()
    for h in _HEDGES_BY_LEN_DESC:
        idx = lower.find(h)
        if idx != -1:
            stripped = clause[:idx] + clause[idx + len(h):]
            stripped = _LEADING_THAT_RE.sub("", stripped, count=1).strip(" ,")
            return stripped, h
    return clause, None


def route_clause(clause):
    """Returns (route, working_clause, hedge) where route in {"opinion", "hedge_verify",
    "assertion"}. `working_clause` is what extraction should run on (the raw clause for
    "opinion"/"assertion", the hedge-stripped remainder for "hedge_verify")."""
    if is_subjective(clause):
        return "opinion", clause, None
    stripped, hedge = strip_hedge(clause)
    if hedge is not None:
        return "hedge_verify", stripped, hedge
    return "assertion", clause, None


def extract_claims_hedge_aware(paragraph, parser, np_binder):
    """Drop-in replacement for `extract_claims_npbind` that fixes the hedge-bypass routing.
    Identical extraction/entailment machinery; only the pre-extraction ROUTING decision differs
    from the imported `extract_claims_npbind` (which calls the unchanged `is_opinion()`)."""
    claims, metas = [], []
    for clause in split_clauses(paragraph):
        route, working_clause, hedge = route_clause(clause)
        if route == "opinion":
            claims.append(Claim(text=clause, kind="opinion"))
            metas.append({"route": "subjective_bypass", "hedge": hedge})
            continue
        parsed, meta = extract_svo_npbind(working_clause, parser, np_binder)
        meta = dict(meta) if isinstance(meta, dict) else {"segmentation_kind": meta}
        meta["route"] = "hedge_stripped_verified" if route == "hedge_verify" else "plain_assertion"
        if route == "hedge_verify":
            meta["hedge"] = hedge
            meta["stripped_clause"] = working_clause
        if parsed is None:
            claims.append(Claim(text=clause, kind="unparsed"))
            metas.append(meta)
            continue
        agent, action, patient, negated = parsed
        claims.append(Claim(text=clause, kind="assertion", agent=agent, action=action,
                             patient=patient, negated=negated))
        metas.append(meta)
    return claims, metas


# ============================================================================
# 2. Item set: the fluency de-risk's 23 items (unchanged, all 5 original
#    styles) + 6 new genuine subjective-opinion items proving the fix does
#    not over-verify taste. Two of the six ALSO carry a hedge phrase, to
#    prove subjectivity wins the routing race over hedge-verify.
# ============================================================================

SUBJECTIVE_ITEMS = [
    item("I love the ocean.", "subjective", "subjective", "KEEP",
         note="taste predicate 'love', NO hedge phrase -- must bypass on subjectivity alone"),
    item("I like cats.", "subjective", "subjective", "KEEP",
         note="taste predicate 'like', NO hedge phrase"),
    item("I hate mosquitoes.", "subjective", "subjective", "KEEP",
         note="taste predicate 'hate', NO hedge phrase"),
    item("The sunset is beautiful.", "subjective", "subjective", "KEEP",
         note="no first-person pronoun, no hedge -- pure predicate-adjective opinion"),
    item("I think jazz is the best music.", "subjective", "subjective", "KEEP",
         note="ALSO carries hedge 'i think' -- subjectivity must win over hedge-verify routing"),
    item("I believe chocolate is delicious.", "subjective", "subjective", "KEEP",
         note="ALSO carries hedge 'i believe' -- subjectivity must win over hedge-verify routing"),
]

ITEMS = BASE_ITEMS + SUBJECTIVE_ITEMS


# ============================================================================
# 3. Harness: run BEFORE (imported-unchanged extract_claims_npbind + decide)
#    and AFTER (this file's extract_claims_hedge_aware + decide, SAME decide)
#    over the identical item set, score both.
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
        rows.append(row)
    return rows


def style_role_stats(rows):
    stats = {}
    for r in rows:
        st = r["subtype"]
        d = stats.setdefault(st, {"faithful": {"n": 0, "correct": 0}, "confab": {"n": 0, "correct": 0},
                                   "subjective": {"n": 0, "correct": 0}})
        bucket = d[r["role"]]
        bucket["n"] += 1
        bucket["correct"] += int(r["correct"])
    for st, d in stats.items():
        for role in ("faithful", "confab", "subjective"):
            b = d[role]
            key = {"faithful": "recall_on_faithful", "confab": "precision_on_confab",
                   "subjective": "bypass_correctness"}[role]
            b[key] = (b["correct"] / b["n"]) if b["n"] else None
    return stats


def subjective_bypass_detail(rows):
    """Did each subjective item ACTUALLY reach the unchecked opinion bypass (kind=='opinion'),
    not merely land on decision==KEEP by coincidence (e.g. an accidental FactStore hit)?"""
    out = []
    for r in rows:
        if r["role"] != "subjective":
            continue
        genuinely_bypassed = (r["extracted_kind"] == "opinion")
        out.append({"paragraph": r["paragraph"], "predicted_decision": r["predicted_decision"],
                     "extracted_kind": r["extracted_kind"], "decision_correct": r["correct"],
                     "genuinely_bypassed_unchecked": genuinely_bypassed})
    return out


def main():
    t0 = time.time()
    store = build_store()
    parser = BridgeParser(seed=42)
    np_binder = NPHeadBinder(seed=42)
    build_s = time.time() - t0
    xp, backend_name = get_backend()

    rows_before = run_pass(lambda p: extract_claims_npbind(p, parser, np_binder), store)
    rows_after = run_pass(lambda p: extract_claims_hedge_aware(p, parser, np_binder), store)

    stats_before = style_role_stats(rows_before)
    stats_after = style_role_stats(rows_after)

    def hedge_numbers(stats):
        d = stats.get("hedge", {"faithful": {"n": 0, "correct": 0}, "confab": {"n": 0, "correct": 0}})
        return {"recall_on_faithful": d["faithful"].get("recall_on_faithful"),
                "recall_frac": f"{d['faithful']['correct']}/{d['faithful']['n']}",
                "precision_on_confab": d["confab"].get("precision_on_confab"),
                "precision_frac": f"{d['confab']['correct']}/{d['confab']['n']}"}

    def subjective_numbers(stats):
        d = stats.get("subjective", {"subjective": {"n": 0, "correct": 0}})
        b = d["subjective"]
        return {"bypass_correctness": b.get("bypass_correctness"),
                "frac": f"{b['correct']}/{b['n']}"}

    hedge_before, hedge_after = hedge_numbers(stats_before), hedge_numbers(stats_after)
    subj_before, subj_after = subjective_numbers(stats_before), subjective_numbers(stats_after)
    subj_detail_before = subjective_bypass_detail(rows_before)
    subj_detail_after = subjective_bypass_detail(rows_after)

    # regression check: styles the fix must NOT touch (plain_control, passive, synonym,
    # embedded_plain3, embedded_verblex) -- before/after must be byte-identical.
    regression_styles = [st for st in stats_before if st != "hedge" and st != "subjective"]
    regression_ok = True
    regression_detail = {}
    for st in regression_styles:
        b, a = stats_before[st], stats_after[st]
        same = (b["faithful"]["correct"] == a["faithful"]["correct"]
                and b["faithful"]["n"] == a["faithful"]["n"]
                and b["confab"]["correct"] == a["confab"]["correct"]
                and b["confab"]["n"] == a["confab"]["n"])
        regression_ok = regression_ok and same
        regression_detail[st] = {"before": {"faithful": f"{b['faithful']['correct']}/{b['faithful']['n']}",
                                             "confab": f"{b['confab']['correct']}/{b['confab']['n']}"},
                                  "after": {"faithful": f"{a['faithful']['correct']}/{a['faithful']['n']}",
                                            "confab": f"{a['confab']['correct']}/{a['confab']['n']}"},
                                  "unchanged": same}

    knobs = {
        "seed": 42, "backend": backend_name, "sim_backend_env": os.environ.get("SIM_BACKEND"),
        "neuron_model": "IZHIKEVICH",
        "parser_class": "BridgeParser (unchanged, reused, brain_conversational_agent.py)",
        "np_binder_class": "NPHeadBinder (unchanged, reused, _spiking_np_boundary_extraction_derisk.py)",
        "before_extractor": "extract_claims_npbind (UNCHANGED, imported)",
        "after_extractor": "extract_claims_hedge_aware (NEW, this file -- routing only)",
        "decide_function": "decide (UNCHANGED, imported from _fluent_paraphrase_verify_suppress_derisk)",
        "entailment_module": "research.runners._open_text_moat_verifier_derisk",
        "extraction_module": "research.runners._spiking_np_boundary_extraction_derisk",
        "bridgeparser_num_neurons": parser.bridge.core_config.num_neurons,
        "npbinder_num_neurons": np_binder.bridge.core_config.num_neurons,
        "build_train_seconds": build_s,
        "n_grounded_facts": len(store.facts),
        "grounded_facts": [{"agent": a, "action": act, "patient": p, "polarity": pol}
                            for (a, act), (p, pol) in store.facts.items()],
        "hedge_phrases": list(HEDGES),
        "subjective_verbs": sorted(SUBJECTIVE_VERBS), "subjective_adjs": sorted(SUBJECTIVE_ADJS),
        "routing_priority": "subjective(unchecked opinion) > hedge(strip+verify) > plain(verify)",
        "n_items_total": len(ITEMS), "n_items_from_fluency_derisk": len(BASE_ITEMS),
        "n_items_new_subjective": len(SUBJECTIVE_ITEMS),
    }

    aggregate = {
        "hedge_style": {"before": hedge_before, "after": hedge_after,
                         "target": "precision_on_confab 0/2 -> 2/2 (safety gap closed); "
                                   "recall_on_faithful stays 2/2 (true hedged facts still kept)"},
        "subjective_bypass": {"before": subj_before, "after": subj_after,
                               "before_detail": subj_detail_before, "after_detail": subj_detail_after,
                               "target": "6/6 genuinely bypassed unchecked (kind=='opinion'), "
                                         "not wrongly suppressed as false facts"},
        "regression_check_other_styles": {"unchanged_overall": regression_ok, "by_style": regression_detail},
    }

    print("=== Hedged-assertion verify de-risk (closes the hedge-bypass safety gap) ===")
    print("\n--- BEFORE (unmodified extract_claims_npbind + is_opinion) ---")
    for r in rows_before:
        if r["subtype"] in ("hedge", "subjective"):
            flag = "OK" if r["correct"] else "MISS"
            print(f"  [{flag:<4}] style={r['subtype']:<12} role={r['role']:<10} "
                  f"gold={r['gold_decision']:<8} pred={r['predicted_decision']:<8} "
                  f"kind={r['extracted_kind']!s:<10} reason={r['predicted_reason']:<24} | {r['paragraph']}")
    print("\n--- AFTER (route_clause + extract_claims_hedge_aware) ---")
    for r in rows_after:
        if r["subtype"] in ("hedge", "subjective"):
            flag = "OK" if r["correct"] else "MISS"
            print(f"  [{flag:<4}] style={r['subtype']:<12} role={r['role']:<10} "
                  f"gold={r['gold_decision']:<8} pred={r['predicted_decision']:<8} "
                  f"kind={r['extracted_kind']!s:<10} reason={r['predicted_reason']:<24} | {r['paragraph']}")

    print("\n=== HEDGE STYLE (2 faithful, 2 confab) ===")
    print("  before:", json.dumps(hedge_before))
    print("  after: ", json.dumps(hedge_after))
    print("\n=== SUBJECTIVE-OPINION BYPASS (6 items, gold=KEEP unchecked) ===")
    print("  before:", json.dumps(subj_before))
    print("  after: ", json.dumps(subj_after))
    print("\n=== REGRESSION CHECK (plain_control / passive / synonym / embedded_*) ===")
    print(json.dumps(regression_detail, indent=2))
    print(f"\nregression-free (all non-hedge/non-subjective styles unchanged): {regression_ok}")

    out = {"knobs": knobs, "aggregate": aggregate,
           "items_before": rows_before, "items_after": rows_after}
    out_path = os.path.join(_REPO, "research", "findings", "raw",
                             "_hedged_assertion_verify_derisk.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")
    return aggregate


if __name__ == "__main__":
    main()
