"""LANE 4 DE-RISK follow-on to #99 (_open_text_moat_verifier_derisk.py): can the claim
EXTRACTION layer -- today a host regex/verb-lexicon stand-in -- be done by the actual
ON-BRAIN SPIKING parser instead, per the brain-based-only standard (CLAUDE.md: anything
between sensation and action must be neurons/synapses; a host regex identifying which
word is the verb/agent/patient is a documented shortcut)?

THE PARSER UNDER TEST. `BridgeParser` (research/runners/brain_conversational_agent.py:28)
is the parser the PRODUCTION conversational entry point actually builds: `hear()` (same
file, :651-676) delegates to `self._ensure_parser().parse(...)` whenever the composer has
no `hear()` of its own -- true for the default `composer_kind='rf'` (RFPhasorComposer has
no `hear`; only `OneBrainComposer` carries an on-bridge parser). So for the shipped
'rf'-composer chat path, BridgeParser IS "the on-brain spiking parser" #43 refers to.
Mechanism: 6 conjunction units (word-position x voice) -> 3 Hebbian-trained role
ensembles on a SimulationBridge (Izhikevich, `enable_hebbian_learning`); `parse(words,
voice)` drives each position's conjunction unit ALONE and reads out which role ensemble
fires most. It is genuinely spiking (`_run_one_simulation_step()`), genuinely learned
(v16 Hebbian co-firing rule, not host-designed weights beyond the initial 0.5 seed), and
carries NO lexical/verb knowledge whatsoever -- role assignment is 100% POSITIONAL.

THE HONEST CONSTRAINT THIS IMPLIES (verified interactively before writing this set,
see the finding doc): `BridgeParser.parse` hard-`assert`s `len(words) == 3` and performs
NO clause segmentation, NO verb lookup, NO NP-boundary detection. Feeding it
`['the','big','apple']` returns `{'agent':'the','action':'big','patient':'apple'}` --
a confident, wrong answer, because "confident" here just means "position 1 fired the
action ensemble hardest," which it always does when driven alone. So THIS runner's host
preprocessing is deliberately narrow and reported as such: (1) clause splitting (reused
UNCHANGED from #99: `split_clauses`), (2) hedge/opinion detection (reused UNCHANGED:
`is_opinion`), (3) stopword + negator REMOVAL (a lexical FILTER, not a role decision --
removing "the"/"does"/"not" leaves the remaining content words in their original
left-to-right order). If, after that filter, a clause reduces to EXACTLY 3 content
words, the actual SUBJECT/VERB/OBJECT role assignment is handed to `BridgeParser.parse`
in spikes. If it does not reduce to 3, the clause is ABSTAIN-and-SUPPRESS: no triple is
fabricated, and it is counted honestly as "unparsed" (a coverage loss), never silently
treated as agreeing with anything.

ENTAILMENT LAYER: reused UNCHANGED by direct import from `_open_text_moat_verifier_derisk`
-- `Claim`, `FactStore`, `classify_claim`, `AFFIRM`, `NEGATE`. This file changes ONLY the
extraction step (`extract_svo` -> `extract_svo_spiking`); the verifier that turns a
resolved triple into grounded/ungrounded is byte-identical to #99 and, transitively, to
the production `routed_composer.ask_yes_no` / `query_patient` abstain-on-unknown /
same-SVO-opposite-polarity-reject semantics it mirrors (see #99's docstring for the
production file:line citations).

Run: python -m research.runners._open_text_spiking_extraction_derisk
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")   # small net; CPU is plenty and avoids GPU init overhead

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import get_backend  # noqa: E402

from research.runners._open_text_moat_verifier_derisk import (  # noqa: E402  (REUSE UNCHANGED)
    AFFIRM, NEGATE, Claim, FactStore, classify_claim, split_clauses, is_opinion,
    STOPWORDS,
)
from research.runners.brain_conversational_agent import BridgeParser  # noqa: E402  (THE SPIKING PARSER UNDER TEST)


# --------------------------------------------------------------------------
# 1. Host preprocessing kept to the bare minimum the spiking parser needs:
#    tokenize, strip determiners (STOPWORDS, imported unchanged from #99) and
#    negation markers (a polarity CUE, not a role decision), then hand the
#    surviving content words to BridgeParser IN ORDER. No verb lookup, no NP
#    boundary detection, no lemmatization -- if that is not enough to reach 3
#    tokens, it is an honest UNPARSED, not a fallback to regex.
# --------------------------------------------------------------------------

NEGATORS = {"not", "doesn't", "don't", "does", "do", "didn't", "never"}


def extract_svo_spiking(clause, parser, voice="active"):
    """Return (agent, action, patient, negated) via the ON-BRAIN BridgeParser, or None
    if the clause does not reduce to exactly 3 content words (BridgeParser's fixed
    3-slot positional read-out cannot segment anything else). `voice` is assumed
    'active' throughout this set -- no automatic passive-voice detector exists in this
    pipeline; that is a separately-documented gap (see the finding), not attempted
    here."""
    words = re.findall(r"[a-zA-Z']+", clause.lower())
    negated = any(w in NEGATORS for w in words)
    content = [w for w in words if w not in STOPWORDS and w not in NEGATORS]
    if len(content) != 3:
        return None
    roles = parser.parse(content, voice=voice)   # <-- the spiking role read-out, the whole point of this file
    return roles["agent"], roles["action"], roles["patient"], negated


def extract_claims_spiking(paragraph, parser):
    """Mirrors #99's `extract_claims`, but the assertion branch calls the spiking
    extractor instead of the regex verb-lexicon matcher. Same three claim kinds:
    'opinion' (hedge detected, never checked), 'assertion' (spiking-parsed, will be
    entailment-checked), 'unparsed' (abstain-and-suppress -- counted, not faked)."""
    claims = []
    for clause in split_clauses(paragraph):
        lower = clause.lower()
        if is_opinion(lower):
            claims.append(Claim(text=clause, kind="opinion"))
            continue
        parsed = extract_svo_spiking(clause, parser)
        if parsed is None:
            claims.append(Claim(text=clause, kind="unparsed"))
            continue
        agent, action, patient, negated = parsed
        claims.append(Claim(text=clause, kind="assertion", agent=agent,
                             action=action, patient=patient, negated=negated))
    return claims


# --------------------------------------------------------------------------
# 2. Brain knowledge store (same FactStore class as #99). Facts are keyed on
#    the EXACT surface verb form used in the corresponding sentence below,
#    because BridgeParser does zero lemmatization -- 'orbits' and 'orbit' are
#    different store keys to it, same as they would be to the real composer's
#    unbind-cleanup if the vocab codes them separately.
# --------------------------------------------------------------------------

def build_store():
    s = FactStore()
    s.store("mercury", "orbits", "sun")
    s.store("moon", "orbits", "earth")
    s.store("bees", "pollinate", "flowers")
    s.store("whales", "breathe", "air")
    s.store("fish", "breathe", "water")
    s.store("einstein", "discovered", "relativity")
    s.store("darwin", "proposed", "evolution")
    s.store("newton", "discovered", "gravity")
    s.store("wasps", "pollinate", "flowers", polarity=NEGATE)   # explicit stored NO
    return s


# --------------------------------------------------------------------------
# 3. Labelled set: 12 Qwen-STYLE prose items (single sentences and short
#    multi-clause paragraphs), each clause hand-labelled with a gold
#    (subject, relation, object) triple + a gold grounded/ungrounded label
#    (grounded == entailed by the FactStore above, i.e. the SAME notion of
#    "true" the moat itself checks -- not omniscient real-world truth), OR
#    kind='opinion' (hedge, gold_label=None, never checked), OR an explicit
#    `expect_parse=False` item deliberately written the way free Qwen prose
#    actually reads (multi-word subjects, copulas, passive voice, prepositions)
#    to measure how much of that BridgeParser's positional 3-slot reader can
#    reach -- the residual this file exists to quantify honestly.
# --------------------------------------------------------------------------

ITEMS = [
    dict(paragraph="Mercury orbits the sun.",
         clauses=[dict(text="Mercury orbits the sun", gold_triple=("mercury", "orbits", "sun"),
                       gold_label=True, expect_parse=True)]),
    dict(paragraph="The moon orbits the earth, and Newton discovered gravity.",
         clauses=[dict(text="moon orbits the earth", gold_triple=("moon", "orbits", "earth"),
                       gold_label=True, expect_parse=True),
                  dict(text="Newton discovered gravity", gold_triple=("newton", "discovered", "gravity"),
                       gold_label=True, expect_parse=True)]),
    dict(paragraph="The sun orbits the moon.",
         clauses=[dict(text="sun orbits the moon", gold_triple=("sun", "orbits", "moon"),
                       gold_label=False, expect_parse=True)]),
    dict(paragraph="Fish breathe air.",
         clauses=[dict(text="Fish breathe air", gold_triple=("fish", "breathe", "air"),
                       gold_label=False, expect_parse=True)]),
    dict(paragraph="Darwin discovered relativity.",
         clauses=[dict(text="Darwin discovered relativity", gold_triple=("darwin", "discovered", "relativity"),
                       gold_label=False, expect_parse=True)]),
    dict(paragraph="Flowers pollinate bees.",
         clauses=[dict(text="Flowers pollinate bees", gold_triple=("flowers", "pollinate", "bees"),
                       gold_label=False, expect_parse=True)]),
    dict(paragraph="Wasps pollinate flowers.",
         clauses=[dict(text="Wasps pollinate flowers", gold_triple=("wasps", "pollinate", "flowers"),
                       gold_label=False, expect_parse=True)]),
    dict(paragraph="Wasps do not pollinate flowers.",
         clauses=[dict(text="Wasps do not pollinate flowers", gold_triple=("wasps", "pollinate", "flowers"),
                       gold_label=True, expect_parse=True, negated=True)]),
    dict(paragraph="I think whales might breathe fire.",
         clauses=[dict(text="I think whales might breathe fire", gold_kind="opinion")]),
    dict(paragraph="The Great Barrier Reef is the largest coral reef system in the world.",
         clauses=[dict(text="The Great Barrier Reef is the largest coral reef system in the world",
                       gold_triple=None, gold_label=True, expect_parse=False)]),
    dict(paragraph="The Eiffel Tower was built in London.",
         clauses=[dict(text="The Eiffel Tower was built in London",
                       gold_triple=None, gold_label=False, expect_parse=False)]),
    dict(paragraph="Perhaps the ancient pyramids were built by extraterrestrial visitors.",
         clauses=[dict(text="Perhaps the ancient pyramids were built by extraterrestrial visitors",
                       gold_kind="opinion")]),
]


def _find_claim(claims, text_substr):
    tl = text_substr.lower()
    for c in claims:
        if tl in c.text.lower() or c.text.lower() in tl:
            return c
    return None


def main():
    t_build0 = time.time()
    store = build_store()
    parser = BridgeParser(seed=42)
    build_s = time.time() - t_build0

    xp, backend_name = get_backend()

    per_item = []
    scored_rows = []          # parsed-assertion subset: (gold_label, predicted_verdict)
    n_assertion_clauses = 0   # non-opinion clauses (the coverage denominator)
    n_parsed = 0
    n_unparsed = 0
    n_opinion = 0
    false_total = 0
    false_caught = 0
    false_slipped_unparsed = 0

    for item in ITEMS:
        paragraph = item["paragraph"]
        claims = extract_claims_spiking(paragraph, parser)
        clause_results = []
        for gold in item["clauses"]:
            claim = _find_claim(claims, gold["text"])
            row = {
                "gold_text": gold["text"],
                "gold_kind": gold.get("gold_kind", "assertion"),
                "gold_triple": gold.get("gold_triple"),
                "gold_label": gold.get("gold_label"),
                "gold_negated": gold.get("negated", False),
            }
            if claim is None:
                row["matched_claim"] = None
                row["outcome"] = "CLAUSE_SPLIT_MISS"   # shouldn't happen; flagged if it does
                clause_results.append(row)
                continue

            row["matched_claim_text"] = claim.text
            row["matched_kind"] = claim.kind

            if gold.get("gold_kind") == "opinion":
                n_opinion += 1
                row["outcome"] = "opinion_suppressed" if claim.kind == "opinion" else "MISLABELLED_NOT_OPINION"
                clause_results.append(row)
                continue

            n_assertion_clauses += 1
            is_false_gold = (gold["gold_label"] is False)
            if is_false_gold:
                false_total += 1

            if claim.kind == "unparsed":
                n_unparsed += 1
                row["extracted_triple"] = None
                row["extracted_negated"] = None
                row["predicted_verdict"] = "unparsed_suppressed"
                row["outcome"] = "unparsed_suppressed"
                if is_false_gold:
                    false_slipped_unparsed += 1
                clause_results.append(row)
                continue

            # claim.kind == "assertion" -> the spiking parser resolved a triple
            n_parsed += 1
            verdict = classify_claim(claim, store)   # 'grounded' | 'ungrounded' (entailment layer, UNCHANGED)
            predicted_label = (verdict == "grounded")
            row["extracted_triple"] = (claim.agent, claim.action, claim.patient)
            row["extracted_negated"] = claim.negated
            row["predicted_verdict"] = verdict
            row["predicted_label"] = predicted_label
            row["correct"] = (predicted_label == gold["gold_label"])
            row["outcome"] = "CORRECT" if row["correct"] else "WRONG"
            scored_rows.append({"gold_label": gold["gold_label"], "predicted_label": predicted_label})
            if is_false_gold and not predicted_label:
                false_caught += 1
            clause_results.append(row)

        per_item.append({"paragraph": paragraph,
                          "extracted_claims": [
                              {"text": c.text, "kind": c.kind, "agent": c.agent, "action": c.action,
                               "patient": c.patient, "negated": c.negated}
                              for c in claims],
                          "clause_results": clause_results})

    # Precision/recall/F1 on the PARSED subset, positive class = "caught as ungrounded/false"
    # (the moat's actual job: catching confabulation), matching #99's framing.
    tp = fp = fn = tn = 0
    for r in scored_rows:
        gold_false = (r["gold_label"] is False)
        pred_false = (r["predicted_label"] is False)
        if gold_false and pred_false:
            tp += 1
        elif (not gold_false) and pred_false:
            fp += 1
        elif gold_false and (not pred_false):
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (tp + fp) and (tp + fn) and (precision + recall) else float("nan"))
    accuracy_parsed = (tp + tn) / len(scored_rows) if scored_rows else float("nan")

    coverage = n_parsed / n_assertion_clauses if n_assertion_clauses else float("nan")

    aggregate = {
        "n_items": len(ITEMS),
        "n_assertion_clauses": n_assertion_clauses,
        "n_opinion_clauses": n_opinion,
        "n_parsed": n_parsed,
        "n_unparsed": n_unparsed,
        "extraction_coverage": coverage,
        "verifier_on_parsed_subset": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy_parsed,
        },
        "false_claim_catch": {
            "false_claims_total": false_total,
            "false_claims_caught_ungrounded": false_caught,
            "false_claims_slipped_unparsed_suppressed": false_slipped_unparsed,
            "catch_rate_over_all_false": (false_caught / false_total) if false_total else float("nan"),
        },
    }

    knobs = {
        "seed": 42,
        "parser_class": "BridgeParser",
        "parser_module": "research.runners.brain_conversational_agent",
        "parser_build_train_seconds": build_s,
        "parser_R": parser.R,
        "parser_note": "6 conjunction units (position x voice) -> 3 Hebbian role ensembles on a "
                        "SimulationBridge (Izhikevich); role assignment is 100% positional, no lexical "
                        "verb knowledge; hard assert len(words)==3.",
        "entailment_module": "research.runners._open_text_moat_verifier_derisk",
        "entailment_functions_reused_unchanged": ["Claim", "FactStore", "classify_claim", "split_clauses",
                                                    "is_opinion", "STOPWORDS", "AFFIRM", "NEGATE"],
        "voice_assumed": "active (no automatic passive-voice detector in this pipeline)",
        "negators_set": sorted(NEGATORS),
        "stopwords_set": sorted(STOPWORDS),
        "backend": backend_name,
        "sim_backend_env": os.environ.get("SIM_BACKEND"),
        "neuron_model": "IZHIKEVICH",
        "num_neurons": parser.bridge.core_config.num_neurons,
    }

    print("=== Open-text SPIKING extraction de-risk ===")
    for it in per_item:
        print("\nParagraph:", it["paragraph"])
        for c in it["extracted_claims"]:
            print("  extracted:", c)
        for r in it["clause_results"]:
            print(f"  [{r['outcome']:<22}] gold_label={str(r.get('gold_label')):<5} "
                  f"gold_triple={r.get('gold_triple')} "
                  f"extracted={r.get('extracted_triple')} verdict={r.get('predicted_verdict')}")

    print("\n=== AGGREGATE ===")
    print(json.dumps(aggregate, indent=2))
    print(f"\nextraction coverage (parsed/assertion-clauses): {coverage:.3f} "
          f"({n_parsed}/{n_assertion_clauses})")
    print(f"verifier on parsed subset: precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} "
          f"accuracy={accuracy_parsed:.3f}")
    print(f"false claims: {false_total} total, {false_caught} caught (ungrounded), "
          f"{false_slipped_unparsed} slipped through unparsed-and-suppressed")

    out = {"knobs": knobs, "aggregate": aggregate, "items": per_item}
    out_path = os.path.join(_REPO, "research", "findings", "raw",
                             "_open_text_spiking_extraction_derisk.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")
    return aggregate


if __name__ == "__main__":
    main()
