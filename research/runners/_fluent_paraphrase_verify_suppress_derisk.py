"""LANE 4 DE-RISK: does the verify-and-suppress moat survive FLUENT PARAPHRASE, not just canonical SVO?

THE QUESTION (memory `project_2026_08_19_strategic_reframe_continuous_substrate`; follow-on to
`_spiking_np_boundary_extraction_derisk.py`, which showed the extraction layer reaches passives, copulas
and multi-word NPs). The reframe wants Qwen to WORD the brain's content FLUENTLY (Qwen=FORM, brain=CONTENT)
with the open-text moat keeping it honest. Before widening Qwen to free prose we need to know: when Qwen
REWORDS a grounded fact fluently -- synonym verb, passive voice, a hedge, a subordinate ("reporting-clause")
construction -- does extract-then-entail correctly KEEP the true rewording, while still SUPPRESSING a
confabulated addition in the SAME surface style? Either verdict is the deliverable (CLAUDE.md: an honest
negative naming the residual is as valid a result as a GO).

PIPELINE UNDER TEST -- reused UNCHANGED, no new mechanism, this file only adds a harder labelled item set +
a KEEP/SUPPRESS scoring harness:
  - extraction: `extract_claims_npbind` (research/runners/_spiking_np_boundary_extraction_derisk.py) =
    host lexical clause/NP-boundary segmentation (`segment_clause`, declared, minimal) -> spiking NP-head
    binding (`NPHeadBinder`, Hebbian-trained population read-out) -> spiking position x voice role read-out
    (`BridgeParser.parse`, unchanged, brain_conversational_agent.py:163).
  - entailment: `classify_claim` / `FactStore.ask_yes_no` (research/runners/_open_text_moat_verifier_derisk.py)
    = the SAME abstain-on-unknown, same-key-opposite-polarity-reject semantics production's
    `composer.ask_yes_no` / `query_patient` implement (see that file's docstring for the production
    file:line citations) -- an exact (agent, action) dict-key lookup, NO lemmatization, NO synonym
    expansion, NO embedding similarity.
  - hedge routing: `is_opinion` (same file) -- ANY clause containing a HEDGES-listed phrase ("i believe",
    "i think", ...) is classified kind="opinion" and is NEVER checked against the store at all (bypasses
    entailment entirely, in both directions).

FINAL DECISION, defined by THIS file (mirrors what a production caller would actually show/hide):
    kind == "opinion"                          -> KEEP   (hedge pass-through, unchecked, by current design)
    kind == "unparsed"                         -> SUPPRESS (abstain -- never fabricate a triple)
    kind == "assertion", classify_claim=="grounded"   -> KEEP
    kind == "assertion", classify_claim=="ungrounded" -> SUPPRESS

ITEM SET. 10 grounded facts (a fresh FactStore, built here); for a subset of them, several FLUENT
PARAPHRASES in each of 5 styles the task named: `plain_control` (unmodified restatement, a sanity check
the pipeline works at all), `passive` (active->passive with a by-agent), `synonym` (a different verb,
same meaning, NOT in the store's lexicon), `hedge` ("I believe X" / "I think X"), `embedded` (a
reporting-verb "that"-clause wrapping the fact, e.g. "Scientists confirmed that X"). Each style has
faithful rewordings (gold=KEEP) of a TRUE fact and confabulated variants (gold=SUPPRESS, a changed
agent/object) in the SAME surface style. `embedded` additionally splits into `embedded_plain3` (base fact
is a bare 3-content-word SVO) vs `embedded_verblex` (base fact uses the VERB_LEXICON multi-word-subject
fallback pass) because -- as this run confirms -- the two hit genuinely different failure MODES in
`segment_clause`, not the same one.

Run: python -m research.runners._fluent_paraphrase_verify_suppress_derisk
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
    FactStore, classify_claim, HEDGES,
)
from research.runners._spiking_np_boundary_extraction_derisk import (  # noqa: E402  (REUSE UNCHANGED)
    NPHeadBinder, extract_claims_npbind, DETERMINERS, PASSIVE_AUX, COPULA_AUX, PARTICIPLES, VERB_LEXICON,
)
from research.runners._open_text_spiking_extraction_derisk import _find_claim  # noqa: E402  (REUSE UNCHANGED)
from research.runners.brain_conversational_agent import BridgeParser  # noqa: E402  (REUSE UNCHANGED)


# ============================================================================
# 1. The brain fact store -- fresh instance, this file's own 10 grounded facts
#    (chosen so each style category has a base fact whose verb sits in the
#    right lexical bucket: PARTICIPLES for passive, a plain content word for
#    synonym/hedge, VERB_LEXICON for the multi-word-subject embedded case).
# ============================================================================

def build_store():
    s = FactStore()
    s.store("mercury", "orbits", "sun")
    s.store("moon", "orbits", "earth")
    s.store("newton", "discovered", "gravity")
    s.store("einstein", "discovered", "relativity")
    s.store("gustave eiffel", "built", "eiffel tower")
    s.store("bees", "pollinate", "flowers")
    s.store("whales", "breathe", "air")
    s.store("darwin", "proposed", "evolution")
    s.store("amazon rainforest", "produces", "oxygen")
    s.store("great barrier reef", "is", "largest coral reef system in world")
    return s


# ============================================================================
# 2. Item set: (paragraph, style, subtype, role, gold_decision). `role` is
#    "faithful" (true reword of a stored fact, gold=KEEP) or "confab" (a
#    plausible but NOT entailed variant in the same surface style, gold=
#    SUPPRESS). Each paragraph is a single clause (no connectives), so
#    `clause_text` == the paragraph with trailing punctuation stripped --
#    unambiguous for `_find_claim`.
# ============================================================================

def item(paragraph, style, role, gold_decision, subtype=None, note=""):
    clause_text = paragraph.rstrip(".!?")
    return dict(paragraph=paragraph, clause_text=clause_text, style=style,
                subtype=subtype or style, role=role, gold_decision=gold_decision, note=note)


ITEMS = [
    # -- plain_control: unmodified restatement, sanity baseline (not really a "paraphrase") --
    item("Mercury orbits the sun.", "plain_control", "faithful", "KEEP"),
    item("The moon orbits the earth.", "plain_control", "faithful", "KEEP"),
    item("Mercury orbits Jupiter.", "plain_control", "confab", "SUPPRESS"),
    item("The moon orbits Mars.", "plain_control", "confab", "SUPPRESS"),

    # -- passive: active->passive with an explicit by-agent (same verb string; SHOULD survive) --
    item("Gravity was discovered by Newton.", "passive", "faithful", "KEEP"),
    item("Relativity was discovered by Einstein.", "passive", "faithful", "KEEP"),
    item("The Eiffel Tower was built by Gustave Eiffel.", "passive", "faithful", "KEEP"),
    item("Gravity was discovered by Darwin.", "passive", "confab", "SUPPRESS"),
    item("The Eiffel Tower was built by Isambard Kingdom Brunel.", "passive", "confab", "SUPPRESS"),

    # -- synonym: same meaning, a DIFFERENT verb string not in the store's exact lexicon --
    item("Mercury circles the sun.", "synonym", "faithful", "KEEP", note="circles = orbits"),
    item("Bees fertilize flowers.", "synonym", "faithful", "KEEP", note="fertilize ~= pollinate"),
    item("Whales inhale air.", "synonym", "faithful", "KEEP", note="inhale ~= breathe"),
    item("Mercury circles Neptune.", "synonym", "confab", "SUPPRESS"),
    item("Bees fertilize weeds.", "synonym", "confab", "SUPPRESS"),
    item("Whales inhale smoke.", "synonym", "confab", "SUPPRESS"),

    # -- hedge: "I believe X" / "I think X" -- is_opinion() routes these OUT of entailment entirely --
    item("I believe Darwin proposed evolution.", "hedge", "faithful", "KEEP"),
    item("I think whales breathe air.", "hedge", "faithful", "KEEP"),
    item("I believe Darwin proposed gravity.", "hedge", "confab", "SUPPRESS"),
    item("I think whales breathe fire.", "hedge", "confab", "SUPPRESS"),

    # -- embedded: a reporting-verb "that"-clause wrapping the fact. Two subtypes because they hit
    #    different segment_clause() failure modes (see the report): embedded_plain3 (base fact would be a
    #    bare 3-content-word SVO) vs embedded_verblex (base fact needs the VERB_LEXICON fallback pass). --
    item("Scientists confirmed that Mercury orbits the sun.", "embedded", "faithful", "KEEP",
         subtype="embedded_plain3"),
    item("Scientists confirmed that the Amazon Rainforest produces oxygen.", "embedded", "faithful", "KEEP",
         subtype="embedded_verblex"),
    item("Scientists confirmed that Mercury orbits Neptune.", "embedded", "confab", "SUPPRESS",
         subtype="embedded_plain3"),
    item("Scientists confirmed that the Amazon Rainforest produces methane.", "embedded", "confab", "SUPPRESS",
         subtype="embedded_verblex"),
]


# ============================================================================
# 3. Final decision: extraction (unchanged) -> entailment (unchanged) -> the
#    KEEP/SUPPRESS mapping this file defines (see module docstring).
# ============================================================================

def decide(claim, store):
    """claim: a Claim (kind in {opinion, unparsed, assertion}) or None (clause-split miss).
    Returns (decision, reason) where decision in {KEEP, SUPPRESS, ERROR} and reason documents why."""
    if claim is None:
        return "ERROR", "clause_split_miss"
    if claim.kind == "opinion":
        return "KEEP", "opinion_bypass_unchecked"
    if claim.kind == "unparsed":
        return "SUPPRESS", "unparsed_abstain"
    verdict = classify_claim(claim, store)   # 'grounded' | 'ungrounded'
    if verdict == "grounded":
        return "KEEP", "assertion_grounded"
    return "SUPPRESS", "assertion_ungrounded"


def main():
    t0 = time.time()
    store = build_store()
    parser = BridgeParser(seed=42)
    np_binder = NPHeadBinder(seed=42)
    build_s = time.time() - t0
    xp, backend_name = get_backend()

    rows = []
    for it in ITEMS:
        claims, metas = extract_claims_npbind(it["paragraph"], parser, np_binder)
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

    # -- aggregate BY STYLE (subtype) x role, per item 4 of the brief --
    style_stats = {}
    for r in rows:
        st = r["subtype"]
        d = style_stats.setdefault(st, {"faithful": {"n": 0, "correct": 0, "items": []},
                                         "confab": {"n": 0, "correct": 0, "items": []}})
        bucket = d[r["role"]]
        bucket["n"] += 1
        bucket["correct"] += int(r["correct"])
        bucket["items"].append({"paragraph": r["paragraph"], "gold": r["gold_decision"],
                                 "predicted": r["predicted_decision"], "reason": r["predicted_reason"],
                                 "correct": r["correct"]})
    for st, d in style_stats.items():
        f, c = d["faithful"], d["confab"]
        f["recall_on_faithful"] = (f["correct"] / f["n"]) if f["n"] else float("nan")
        c["precision_on_confab"] = (c["correct"] / c["n"]) if c["n"] else float("nan")

    n_faithful = sum(1 for r in rows if r["role"] == "faithful")
    n_confab = sum(1 for r in rows if r["role"] == "confab")
    overall_recall_on_faithful = (sum(1 for r in rows if r["role"] == "faithful" and r["correct"]) / n_faithful
                                   if n_faithful else float("nan"))
    overall_precision_on_confab = (sum(1 for r in rows if r["role"] == "confab" and r["correct"]) / n_confab
                                    if n_confab else float("nan"))
    n_errors = sum(1 for r in rows if r["predicted_decision"] == "ERROR")

    knobs = {
        "seed": 42, "backend": backend_name, "sim_backend_env": os.environ.get("SIM_BACKEND"),
        "neuron_model": "IZHIKEVICH",
        "parser_class": "BridgeParser (unchanged, reused, brain_conversational_agent.py)",
        "np_binder_class": "NPHeadBinder (unchanged, reused, _spiking_np_boundary_extraction_derisk.py)",
        "entailment_module": "research.runners._open_text_moat_verifier_derisk",
        "extraction_module": "research.runners._spiking_np_boundary_extraction_derisk",
        "bridgeparser_num_neurons": parser.bridge.core_config.num_neurons,
        "npbinder_num_neurons": np_binder.bridge.core_config.num_neurons,
        "build_train_seconds": build_s,
        "n_grounded_facts": len(store.facts),
        "grounded_facts": [{"agent": a, "action": act, "patient": p, "polarity": pol}
                            for (a, act), (p, pol) in store.facts.items()],
        "determiners_stripped": sorted(DETERMINERS),
        "passive_aux": sorted(PASSIVE_AUX), "copula_aux": sorted(COPULA_AUX),
        "participles_lexicon": sorted(PARTICIPLES), "verb_lexicon_fallback": sorted(VERB_LEXICON),
        "hedge_phrases": list(HEDGES),
        "n_items": len(ITEMS), "n_faithful": n_faithful, "n_confab": n_confab,
        "styles": sorted(style_stats.keys()),
        "final_decision_rule": ("opinion->KEEP(unchecked) | unparsed->SUPPRESS(abstain) | "
                                 "assertion+grounded->KEEP | assertion+ungrounded->SUPPRESS"),
    }

    aggregate = {
        "overall": {
            "n_faithful": n_faithful, "recall_on_faithful": overall_recall_on_faithful,
            "n_confab": n_confab, "precision_on_confab": overall_precision_on_confab,
            "n_errors_clause_split_miss": n_errors,
        },
        "by_style": {
            st: {
                "faithful": {"n": d["faithful"]["n"], "correct": d["faithful"]["correct"],
                             "recall_on_faithful": d["faithful"]["recall_on_faithful"]},
                "confab": {"n": d["confab"]["n"], "correct": d["confab"]["correct"],
                           "precision_on_confab": d["confab"]["precision_on_confab"]},
            } for st, d in style_stats.items()
        },
    }

    print("=== Fluent-paraphrase verify-and-suppress de-risk ===")
    for r in rows:
        flag = "OK" if r["correct"] else "MISS"
        print(f"  [{flag:<4}] style={r['subtype']:<18} role={r['role']:<8} "
              f"gold={r['gold_decision']:<8} pred={r['predicted_decision']:<8} "
              f"reason={r['predicted_reason']:<24} | {r['paragraph']}")

    print("\n=== BY STYLE ===")
    for st, d in aggregate["by_style"].items():
        print(f"  {st:<18} recall_on_faithful={d['faithful']['recall_on_faithful']:.3f} "
              f"({d['faithful']['correct']}/{d['faithful']['n']})   "
              f"precision_on_confab={d['confab']['precision_on_confab']:.3f} "
              f"({d['confab']['correct']}/{d['confab']['n']})")

    print("\n=== OVERALL ===")
    print(json.dumps(aggregate["overall"], indent=2))

    out = {"knobs": knobs, "aggregate": aggregate, "items": rows}
    out_path = os.path.join(_REPO, "research", "findings", "raw",
                             "_fluent_paraphrase_verify_suppress_derisk.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")
    return aggregate


if __name__ == "__main__":
    main()
