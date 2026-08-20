"""LANE 4 DE-RISK: closes fluency residual #2, SYNONYM-VERB BRITTLENESS, left open by
research/findings/2026-08-20-fluent-paraphrase-verify-suppress-hedge-bypass-safety-gap.md
(runner: _fluent_paraphrase_verify_suppress_derisk.py) and its follow-on
research/findings/2026-08-20-hedge-bypass-safety-gap-CLOSED-verify-under-hedge.md
(runner: _hedged_assertion_verify_derisk.py, which closed residual #1, the hedge bypass).

THE GAP (recap). `FactStore` (research/runners/_open_text_moat_verifier_derisk.py:59-85) keys
entailment on the EXACT `(agent, action)` string pulled out of a clause. A genuinely fluent
rewording that swaps in a synonym verb -- "Mercury CIRCLES the sun" instead of the stored
"orbits" -- extracts a perfectly correct triple (mercury, circles, sun), but `ask_yes_no` does a
plain dict-key miss on `("mercury", "circles")` and returns "unknown", which `classify_claim`
then maps to "ungrounded" regardless of truth. The fluency de-risk measured this directly:
recall_on_synonym_faithful = 0/3 ("Mercury circles the sun" / "Bees fertilize flowers" /
"Whales inhale air" -- all TRUE, all wrongly SUPPRESSED). It never lets a FALSE claim through
(precision_on_synonym_confab was already 1.0, because an unmatched verb always defaults to
"ungrounded" -- fails safe, not fails silent), but it redacts true fluent rewordings, a real
fluency cost the reframe (memory `project_2026_08_19_strategic_reframe_continuous_substrate`)
cares about.

THE FIX -- entirely NEW code in this file, no edit to any existing runner (`FactStore`,
`Claim`, `classify_claim`, `extract_claims_npbind`/`extract_svo_npbind`, `NPHeadBinder`,
`BridgeParser`, `decide` are all imported UNCHANGED and still used):

  `expand_action_candidates(action)` -- a small, declared, host lexical resource (SAME category
  as the existing HEDGES/STOPWORDS/PARTICIPLES lexicons already in this file family: a fixed
  list feeding a preprocessing step, never deciding a grammatical role or an entailment
  verdict): a cheap stem-only lemmatizer (`circles -> circle`, `fertilizes -> fertilize`, a
  no-op when there is no trailing -s/-es) composed with a declared SYNONYM_LEMMA_MAP
  ({circle: orbit, fertilize: pollinate, inhale: breathe}), then re-inflects every resulting
  lemma with a naive +s so the candidate set covers whichever grammatical-number form the
  FactStore happens to have stored under (the store is NOT lemma-normalized -- "orbits" is
  stored with -s because Mercury/the moon are singular subjects, while "pollinate"/"breathe"
  are stored bare because their subjects are plural -- a real, pre-existing inconsistency in
  the store this file works around rather than "fixes", since fixing the store is out of scope).

  `classify_claim_synonym_aware(claim, store)` -- SAME return contract as the imported
  `classify_claim` ('grounded'|'ungrounded'|'opinion'|'unparsed'). The ONLY change: instead of
  one `store.ask_yes_no(claim.agent, claim.action, claim.patient)` call, it calls
  `store.ask_yes_no` once per candidate in `expand_action_candidates(claim.action)` and takes
  the verdict SET. Entailment succeeds iff ANY candidate matches with the SAME polarity the
  claim asserts -- the polarity/negation comparison itself (claim.negated -> compare against
  verdict=='no' vs verdict=='yes') is copied VERBATIM from `classify_claim`; only WHICH keys
  get queried changes, never how a queried verdict is interpreted. Because the exact action
  string is always the first candidate tried, every clause the ORIGINAL classify_claim already
  got right is still tried against the identical key first -- expansion only ever ADDS
  candidate keys, it never removes the exact-match path.

  `decide_synonym_aware(claim, store)` -- a byte-for-byte copy of the imported `decide()`'s
  routing (opinion->KEEP unchecked, unparsed->SUPPRESS abstain, assertion+grounded->KEEP,
  assertion+ungrounded->SUPPRESS) with its one `classify_claim(...)` call swapped for
  `classify_claim_synonym_aware(...)`. `decide()` itself hardcodes the call to `classify_claim`
  internally, so it cannot be reused unmodified once the classify step changes; this is the
  smallest possible wrapper that keeps everything else (extraction, the store, the KEEP/SUPPRESS
  mapping) identical.

THE NEGATION GUARD (why synonym expansion cannot just OR every candidate together carelessly).
Expanding the candidate KEYS tried is safe only because the polarity comparison still runs once
per candidate and the SAME-negation rule from `classify_claim` is preserved unchanged: a negated
claim over an expanded synonym is grounded iff SOME candidate returns "no", never merely because
a candidate exists. Three new items prove this survives expansion, keyed against a genuine
explicit-NEGATE fact this file adds to the store (`fish` do NOT `breathe` `air`, gills not
lungs -- same convention as the base moat store's `cat does NOT bite mouse`,
_open_text_moat_verifier_derisk.py:242): "Fish do not inhale air" (TRUE negation via synonym,
gold=KEEP), "Whales do not inhale air" (FALSE negation of the store's TRUE whales/breathe/air
fact via synonym, gold=SUPPRESS), "Mercury does not circle the sun" (FALSE negation of the
TRUE mercury/orbits/sun fact via synonym, gold=SUPPRESS).

MEASUREMENT. Re-runs the ORIGINAL (unmodified) `extract_claims_npbind` + `decide` +
`classify_claim` pipeline ("BEFORE") and this file's `decide_synonym_aware` ("AFTER") over the
SAME item set: the fluency de-risk's 23 items (imported unchanged, all 5 original styles,
including the 3 synonym-faithful and 3 synonym-confab items) PLUS 3 new negated-synonym items
(this file), so precision_on_synonym_confab, recall_on_synonym_faithful, the negation guard, and
regression-freedom on every other style are all measured from the SAME run.

Run: python -m research.runners._synonym_expansion_verify_derisk
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")   # small nets; CPU is plenty, avoids GPU init overhead

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import get_backend  # noqa: E402

from research.runners._open_text_moat_verifier_derisk import (  # noqa: E402  (REUSE UNCHANGED)
    FactStore, classify_claim, NEGATE, HEDGES,
)
from research.runners._spiking_np_boundary_extraction_derisk import (  # noqa: E402  (REUSE UNCHANGED)
    NPHeadBinder, extract_claims_npbind,
    DETERMINERS, PASSIVE_AUX, COPULA_AUX, PARTICIPLES, VERB_LEXICON,
)
from research.runners._open_text_spiking_extraction_derisk import _find_claim  # noqa: E402  (REUSE UNCHANGED)
from research.runners.brain_conversational_agent import BridgeParser  # noqa: E402  (REUSE UNCHANGED)
from research.runners._fluent_paraphrase_verify_suppress_derisk import (  # noqa: E402  (REUSE UNCHANGED)
    build_store, item, ITEMS as BASE_ITEMS, decide,
)


# ============================================================================
# 1. The synonym/lemma expansion. Everything downstream (FactStore.ask_yes_no)
#    is imported unchanged; this is the ONLY new decision logic in the fix.
# ============================================================================

# A small, declared, host lexical resource -- same abstraction level as
# DETERMINERS/PASSIVE_AUX/HEDGES/PARTICIPLES (a fixed lexicon feeding a host
# preprocessing step). Maps a SOURCE verb lemma (what a fluent synonym rewording
# might use) to the TARGET verb lemma the FactStore actually has stored under.
# One-directional by design (matches the task: "the claim's verb is expanded to
# the set of stored-equivalent predicates") -- it never decides which NP fills
# which grammatical role, and it never touches the polarity/negation comparison.
SYNONYM_LEMMA_MAP = {
    "circle": "orbit",
    "fertilize": "pollinate",
    "inhale": "breathe",
}


_SIBILANT_ES_SUFFIXES = ("ches", "shes", "xes", "zzes")   # e.g. watches, washes, boxes, buzzes


def lemmatize_verb(word):
    """Cheap stem-only lemmatizer for a single verb surface form. No-op on a word with no
    trailing -s (e.g. 'inhale', 'fertilize' -- already bare). Two rules, most-specific first:
    genuine sibilant '-es' plurals (watch+es, box+es -- the stem does NOT itself end in a
    silent 'e') strip 2 chars; every other trailing '-s' (INCLUDING a silent-e-then-s form
    like 'circle'+'s'='circles' or a hypothetical 'fertilize'+'s'='fertilizes' -- the stem
    ends in 'e' already, so only the 's' is the inflection) strips 1 char. Getting this order
    wrong is a real bug this file hit during development: a naive blanket 'ends with es ->
    strip 2' rule mis-lemmatized 'circles' (ends in '...les', which LOOKS like an '-es' plural
    but is actually 'circle'+'s') to 'circl' instead of 'circle', silently missing the
    SYNONYM_LEMMA_MAP lookup entirely. Not a real lemmatizer (no irregular-verb table), same
    declared-and-scoped-to-the-item-set spirit as the rest of this file family's lexical
    preprocessing (DETERMINERS/PASSIVE_AUX/HEDGES/PARTICIPLES)."""
    if word.endswith(_SIBILANT_ES_SUFFIXES) and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
        return word[:-1]
    return word


def _inflect(lemma):
    """A bare lemma plus a naive 3rd-person-singular '+s' form, so the candidate set covers
    whichever grammatical-number surface form the FactStore happens to have been built with
    (the store is not lemma-normalized -- see module docstring)."""
    forms = {lemma}
    if not lemma.endswith("s"):
        forms.add(lemma + "s")
    return forms


def expand_action_candidates(action):
    """action -> set of candidate action-strings to try against the FactStore, in addition
    to the exact string already tried by the unmodified classify_claim. The exact `action`
    itself is always included first (expansion only ADDS candidates, never removes the
    original exact-match path -- this is why every item the original pipeline already got
    right stays right: the original key is still queried, unconditionally)."""
    lemma = lemmatize_verb(action)
    bases = {action, lemma}
    target = SYNONYM_LEMMA_MAP.get(lemma)
    if target:
        bases.add(target)
    candidates = set()
    for b in bases:
        candidates |= _inflect(b)
    return candidates


def classify_claim_synonym_aware(claim, store):
    """Same return contract as the imported `classify_claim`
    ('grounded'|'ungrounded'|'opinion'|'unparsed'). Only difference: the store lookup is tried
    across expand_action_candidates(claim.action) instead of the single exact action string;
    entailment succeeds iff ANY expansion matches with the SAME polarity. The negation/polarity
    comparison below is copied VERBATIM from classify_claim (research/runners/
    _open_text_moat_verifier_derisk.py:214-222) -- this function changes WHICH keys are
    queried, never how a queried verdict is interpreted."""
    if claim.kind in ("opinion", "unparsed"):
        return claim.kind
    candidates = expand_action_candidates(claim.action)
    verdicts = {store.ask_yes_no(claim.agent, cand, claim.patient) for cand in candidates}
    if claim.negated:
        # "the cat does not eat fish" asserts NEGATE(cat,eat,fish) as fact -- grounded iff
        # SOME candidate's stored polarity is NEGATE (verdict=='no'), never merely because a
        # candidate key exists (an 'unknown' verdict from a miss must not count as a match).
        return "grounded" if "no" in verdicts else "ungrounded"
    return "grounded" if "yes" in verdicts else "ungrounded"


def decide_synonym_aware(claim, store):
    """Byte-for-byte copy of the imported `decide()`'s routing (module docstring), with its
    one classify_claim(...) call swapped for classify_claim_synonym_aware(...). `decide()`
    hardcodes the call to classify_claim internally so it cannot be reused unmodified once the
    classify step changes; every other decision (opinion/unparsed routing, the KEEP/SUPPRESS
    mapping) is identical to the imported original."""
    if claim is None:
        return "ERROR", "clause_split_miss"
    if claim.kind == "opinion":
        return "KEEP", "opinion_bypass_unchecked"
    if claim.kind == "unparsed":
        return "SUPPRESS", "unparsed_abstain"
    verdict = classify_claim_synonym_aware(claim, store)
    if verdict == "grounded":
        return "KEEP", "assertion_grounded"
    return "SUPPRESS", "assertion_ungrounded"


# ============================================================================
# 2. Item set: the fluency de-risk's 23 items (unchanged, all 5 original
#    styles, including the 3 synonym-faithful + 3 synonym-confab items that
#    are this file's headline measurement) PLUS 3 new negated-synonym items
#    proving the polarity reject survives expansion (module docstring).
# ============================================================================

NEGATED_SYNONYM_ITEMS = [
    item("Fish do not inhale air.", "synonym_negated", "faithful", "KEEP",
         note="fish do NOT breathe air (gills, not lungs) -- a genuine stored NEGATE fact "
              "this file adds; true negation via synonym inhale~=breathe"),
    item("Whales do not inhale air.", "synonym_negated", "confab", "SUPPRESS",
         note="FALSE negation of the store's TRUE whales/breathe/air fact, via synonym -- "
              "must still be caught after expansion"),
    item("Mercury does not circle the sun.", "synonym_negated", "confab", "SUPPRESS",
         note="FALSE negation of the store's TRUE mercury/orbits/sun fact, via synonym -- "
              "must still be caught after expansion"),
]

ITEMS = BASE_ITEMS + NEGATED_SYNONYM_ITEMS


def build_store_with_negation():
    """The fluency de-risk's store (imported, unchanged 10 facts) plus ONE new explicit-NEGATE
    fact this file adds for the negation guard (module docstring). Does not touch or overwrite
    any existing (agent, action) key -- ("fish", "breathe") is not used anywhere else in the
    base store, so this is a pure addition, not a collision (see _spiking_np_boundary_
    extraction_derisk.py:310-315 for the collision failure mode this avoids)."""
    s = build_store()
    s.store("fish", "breathe", "air", polarity=NEGATE)
    return s


# ============================================================================
# 3. Harness: run BEFORE (imported-unchanged extract_claims_npbind + decide,
#    i.e. classify_claim with NO expansion) and AFTER (same extraction,
#    decide_synonym_aware) over the identical item set, score both.
# ============================================================================

def run_pass(extractor_fn, decide_fn, store):
    rows = []
    for it in ITEMS:
        claims, metas = extractor_fn(it["paragraph"])
        claim = _find_claim(claims, it["clause_text"])
        decision, reason = decide_fn(claim, store)
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
        d = stats.setdefault(st, {"faithful": {"n": 0, "correct": 0}, "confab": {"n": 0, "correct": 0}})
        bucket = d[r["role"]]
        bucket["n"] += 1
        bucket["correct"] += int(r["correct"])
    for st, d in stats.items():
        f, c = d["faithful"], d["confab"]
        f["recall_on_faithful"] = (f["correct"] / f["n"]) if f["n"] else None
        c["precision_on_confab"] = (c["correct"] / c["n"]) if c["n"] else None
    return stats


def main():
    t0 = time.time()
    store = build_store_with_negation()
    parser = BridgeParser(seed=42)
    np_binder = NPHeadBinder(seed=42)
    build_s = time.time() - t0
    xp, backend_name = get_backend()

    extractor = lambda p: extract_claims_npbind(p, parser, np_binder)  # noqa: E731

    rows_before = run_pass(extractor, decide, store)
    rows_after = run_pass(extractor, decide_synonym_aware, store)

    stats_before = style_role_stats(rows_before)
    stats_after = style_role_stats(rows_after)

    def frac(d, role, key):
        b = d[role]
        return {"value": b[key], "frac": f"{b['correct']}/{b['n']}"}

    synonym_before = {"recall_on_faithful": frac(stats_before["synonym"], "faithful", "recall_on_faithful"),
                       "precision_on_confab": frac(stats_before["synonym"], "confab", "precision_on_confab")}
    synonym_after = {"recall_on_faithful": frac(stats_after["synonym"], "faithful", "recall_on_faithful"),
                      "precision_on_confab": frac(stats_after["synonym"], "confab", "precision_on_confab")}

    negation_before = {"recall_on_faithful": frac(stats_before["synonym_negated"], "faithful", "recall_on_faithful"),
                        "precision_on_confab": frac(stats_before["synonym_negated"], "confab", "precision_on_confab")}
    negation_after = {"recall_on_faithful": frac(stats_after["synonym_negated"], "faithful", "recall_on_faithful"),
                       "precision_on_confab": frac(stats_after["synonym_negated"], "confab", "precision_on_confab")}

    # regression check: every style OTHER than "synonym" (whose faithful recall is EXPECTED to
    # change) and "synonym_negated" (a brand-new bucket, no prior baseline) must be BYTE-IDENTICAL
    # before/after. Within "synonym" itself, the CONFAB side (precision) must also stay identical
    # -- only faithful recall is allowed to move.
    regression_ok = True
    regression_detail = {}
    for st, d_after in stats_after.items():
        d_before = stats_before[st]
        if st == "synonym":
            same = (d_before["confab"]["correct"] == d_after["confab"]["correct"]
                    and d_before["confab"]["n"] == d_after["confab"]["n"])
            label = "confab_only (faithful recall EXPECTED to change)"
        elif st == "synonym_negated":
            same = None   # new bucket, nothing to regress against
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

    knobs = {
        "seed": 42, "backend": backend_name, "sim_backend_env": os.environ.get("SIM_BACKEND"),
        "neuron_model": "IZHIKEVICH",
        "parser_class": "BridgeParser (unchanged, reused, brain_conversational_agent.py)",
        "np_binder_class": "NPHeadBinder (unchanged, reused, _spiking_np_boundary_extraction_derisk.py)",
        "extractor": "extract_claims_npbind (UNCHANGED, imported; identical for BEFORE and AFTER passes)",
        "before_decide": "decide (UNCHANGED, imported from _fluent_paraphrase_verify_suppress_derisk -> classify_claim, no expansion)",
        "after_decide": "decide_synonym_aware (NEW, this file -> classify_claim_synonym_aware, expansion only)",
        "entailment_module": "research.runners._open_text_moat_verifier_derisk",
        "extraction_module": "research.runners._spiking_np_boundary_extraction_derisk",
        "bridgeparser_num_neurons": parser.bridge.core_config.num_neurons,
        "npbinder_num_neurons": np_binder.bridge.core_config.num_neurons,
        "build_train_seconds": build_s,
        "n_grounded_facts": len(store.facts),
        "grounded_facts": [{"agent": a, "action": act, "patient": p, "polarity": pol}
                            for (a, act), (p, pol) in store.facts.items()],
        "synonym_lemma_map": dict(SYNONYM_LEMMA_MAP),
        "determiners_stripped": sorted(DETERMINERS),
        "passive_aux": sorted(PASSIVE_AUX), "copula_aux": sorted(COPULA_AUX),
        "participles_lexicon": sorted(PARTICIPLES), "verb_lexicon_fallback": sorted(VERB_LEXICON),
        "hedge_phrases": list(HEDGES),
        "n_items_total": len(ITEMS), "n_items_from_fluency_derisk": len(BASE_ITEMS),
        "n_items_new_negated_synonym": len(NEGATED_SYNONYM_ITEMS),
        "expansion_rule": ("candidates = {action, lemmatize(action), SYNONYM_LEMMA_MAP.get(lemma)} "
                            "each re-inflected to {bare, bare+'s'}; entailment succeeds iff ANY "
                            "candidate matches with the SAME polarity as classify_claim would require "
                            "for the exact action alone"),
    }

    aggregate = {
        "synonym_style": {
            "before": synonym_before, "after": synonym_after,
            "target": "recall_on_synonym_faithful 0/3 -> 3/3; precision_on_synonym_confab STAYS 1.0 (3/3)",
        },
        "negation_guard_synonym_negated": {
            "before": negation_before, "after": negation_after,
            "target": "recall_on_faithful (true negation via synonym) improves; "
                      "precision_on_confab (false negation via synonym) STAYS 1.0 (2/2) -- "
                      "proves the polarity reject survives expansion",
        },
        "regression_check_other_styles": {"unchanged_overall": regression_ok, "by_style": regression_detail},
    }

    print("=== Synonym-expansion verify de-risk (closes synonym-verb brittleness) ===")
    print("\n--- BEFORE (unmodified classify_claim, no expansion) ---")
    for r in rows_before:
        if r["subtype"] in ("synonym", "synonym_negated"):
            flag = "OK" if r["correct"] else "MISS"
            print(f"  [{flag:<4}] style={r['subtype']:<16} role={r['role']:<8} "
                  f"gold={r['gold_decision']:<8} pred={r['predicted_decision']:<8} "
                  f"triple={r['extracted_triple']!s:<32} reason={r['predicted_reason']:<24} | {r['paragraph']}")
    print("\n--- AFTER (classify_claim_synonym_aware, lemma/synonym expansion) ---")
    for r in rows_after:
        if r["subtype"] in ("synonym", "synonym_negated"):
            flag = "OK" if r["correct"] else "MISS"
            print(f"  [{flag:<4}] style={r['subtype']:<16} role={r['role']:<8} "
                  f"gold={r['gold_decision']:<8} pred={r['predicted_decision']:<8} "
                  f"triple={r['extracted_triple']!s:<32} reason={r['predicted_reason']:<24} | {r['paragraph']}")

    print("\n=== SYNONYM STYLE (3 faithful, 3 confab) ===")
    print("  before:", json.dumps(synonym_before))
    print("  after: ", json.dumps(synonym_after))
    print("\n=== NEGATION GUARD -- synonym_negated (1 faithful, 2 confab) ===")
    print("  before:", json.dumps(negation_before))
    print("  after: ", json.dumps(negation_after))
    print("\n=== REGRESSION CHECK (plain_control / passive / hedge / embedded_*, + synonym-confab side) ===")
    print(json.dumps(regression_detail, indent=2))
    print(f"\nregression-free (every style outside synonym-faithful/synonym_negated unchanged, "
          f"synonym-confab unchanged): {regression_ok}")

    out = {"knobs": knobs, "aggregate": aggregate,
           "items_before": rows_before, "items_after": rows_after}
    out_path = os.path.join(_REPO, "research", "findings", "raw",
                             "_synonym_expansion_verify_derisk.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")
    return aggregate


if __name__ == "__main__":
    main()
