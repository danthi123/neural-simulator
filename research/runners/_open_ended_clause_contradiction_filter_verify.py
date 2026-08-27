"""WIRING + BEHAVIOR VERIFY: the CLAUSE-granularity contradiction repair is live inside
`webapp.open_ended_chat.post_filter`, closing the SAME-SENTENCE residual the 2026-08-27 wiring finding
disclosed as "Honest scope" (Vikunja #112). (2026-08-27)

CONTEXT. `_open_ended_clause_contradiction_filter_derisk.clause_filter_sentence` tries two safe, declared
repairs (an appositive/relative-clause strip, a coordinated relation-object-list strip) before ever falling
back to dropping a whole sentence, and re-verifies every repair against the UNCHANGED, imported
`sentence_contradicts` before returning edited text. This runner checks that mechanism THROUGH the real
`webapp.open_ended_chat.post_filter` entry point (not just the helper in isolation):

  (1) SAME-SENTENCE CORRECT+WRONG: on 4 sentences (2 from the ACTUAL saved canada reply, 2 synthetic on
      france/morocco built the same way as this arc's own de-risk items) carrying one correct detail and
      one wrong detail in the SAME sentence, the wired filter must KEEP the correct detail and STRIP the
      wrong one (not drop the whole sentence, which was the pre-existing behavior).
  (2) NO REGRESSION on the 10 known-wrong-details: re-runs the 2026-08-27 wiring verify's own MUST_DROP
      ground truth (mexico/35 million/1867/italy/germany/switzerland/algeria/tunisia/libya/egypt) through
      the NOW-clause-aware `webapp.open_ended_chat.post_filter` over the 3 saved known-topic replies --
      catch_rate must stay 1.0, leaks must stay 0, exactly as the prior wiring verify measured.
  (3) MOAT-SAFE: sentences with NO salvageable correct clause (every border-list item wrong: france's
      "Italy/Germany/Switzerland" sentence and morocco's "Algeria/Tunisia/Libya/Egypt" sentence; a bare
      unsupported number with no relative-clause boundary: canada's "35 million" sentence) are STILL fully
      removed by the wired filter -- the repair's own fallback, not a new hole.
  (4) BYTE-IDENTICAL OFF: `BRAIN_OPEN_ENDED` unset/0 -> `open_ended_enabled()` False, and
      `webapp/server.py`'s `open_ended_chat` import stays nested under the unchanged guard (structural,
      same check the 2026-08-27 wiring verify used) -- an off run never imports this changed module.

MEMORY-SAFE BY DESIGN: no GPU, no Qwen render -- reuses the SAME saved replies the prior de-risks used
(research/findings/raw/_open_ended_verify_postfilter_derisk.json). Deterministic filter-logic verification
over fixed text -- same seed-waiver as the 2026-08-21/08-27 findings: the evidence is catch/leak/keep counts
against a fixed ground truth, not a stochastic effect.

  python -m research.runners._open_ended_clause_contradiction_filter_verify
"""
from __future__ import annotations
import json, os, re, sys
os.environ.setdefault("SIM_BACKEND", "numpy")
import logging; logging.disable(logging.INFO)
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from webapp import open_ended_chat as OE  # noqa: E402 -- the module under test
from research.runners._open_ended_state_driven_generation_derisk import specificity  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_open_ended_clause_contradiction_filter_verify.json"
SAVED_REPLIES = _REPO / "research" / "findings" / "raw" / "_open_ended_verify_postfilter_derisk.json"

# ---------------------------------------------------------------------------------------------------------
# (2) the SAME MUST_DROP / FACTS ground truth the 2026-08-27 wiring verify used, unchanged -- no regression.
# ---------------------------------------------------------------------------------------------------------
FACTS = {
    "canada":  [("canada", "isa", "country"), ("canada", "capital", "ottawa"),
                ("canada", "continent", "north america"), ("canada", "borders", "united states")],
    "france":  [("france", "isa", "country"), ("france", "capital", "paris"),
                ("france", "continent", "europe"), ("france", "borders", "spain")],
    "morocco": [("morocco", "isa", "country"), ("morocco", "capital", "rabat"),
                ("morocco", "continent", "africa"), ("morocco", "borders", "spain")],
}
MUST_DROP = {"canada": {"mexico", "35 million", "1867"}, "france": {"italy", "germany", "switzerland"},
             "morocco": {"algeria", "tunisia", "libya", "egypt"}}

# ---------------------------------------------------------------------------------------------------------
# (1) SAME-SENTENCE correct+wrong items. 2 are the ACTUAL sentences from the saved canada reply (the exact
# residual the 2026-08-27 wiring finding named); 2 are synthetic france/morocco sentences built the SAME
# shape (a coordinated relation-object list with one correct + one wrong member) to check the repair
# generalizes across topics, not just the one saved reply that motivated it.
# ---------------------------------------------------------------------------------------------------------
SAME_SENTENCE_ITEMS = [
    dict(topic="canada", must_keep="united states", must_strip="mexico", source="saved-reply(canada)",
         sentence="Canada is bordered by the United States to the south and Mexico to the west"),
    dict(topic="canada", must_keep="ottawa", must_strip="1867", source="saved-reply(canada)",
         sentence="The capital city of Canada is Ottawa, which was founded in 1867"),
    dict(topic="france", must_keep="paris", must_strip="1523", source="synthetic(france)",
         sentence="France's capital is Paris, which was founded in the year 1523"),
    dict(topic="morocco", must_keep="spain", must_strip="algeria", source="synthetic(morocco)",
         sentence="Morocco borders Spain to the north and Algeria to the east"),
]

# ---------------------------------------------------------------------------------------------------------
# (3) MOAT-safe: no salvageable correct clause -- must stay a FULL drop (0 leaks, nothing invented to keep).
# ---------------------------------------------------------------------------------------------------------
NO_SALVAGE_ITEMS = [
    dict(topic="france", must_not_leak={"italy", "germany", "switzerland"}, source="saved-reply(france)",
         sentence="It's bordered by Italy to the west, Germany to the north, and Switzerland to the east"),
    dict(topic="morocco", must_not_leak={"algeria", "tunisia", "libya", "egypt"}, source="saved-reply(morocco)",
         sentence="Morocco has borders with several countries including Algeria, Tunisia, Libya, and Egypt"),
    dict(topic="canada", must_not_leak={"35 million"}, source="saved-reply(canada)",
         sentence="It has a population of around 35 million people, making it one of the largest countries "
                  "in the world"),
]


def _facts_for(topic):
    """The RAW (agent, action, patient) triples `webapp.open_ended_chat.post_filter` expects -- it runs
    `_facts_as_relation_pairs` on these ITSELF (the SAME adapter the 2026-08-27 wiring verify exercised);
    pre-adapting here would double-apply it and break the unpack."""
    return FACTS[topic]


def _check_off_path_gating():
    """Same structural check the 2026-08-27 wiring verify used (unaffected by this change -- server.py was not
    touched): webapp/server.py's open_ended_chat import stays nested under the unchanged BRAIN_OPEN_ENDED guard."""
    src = (_REPO / "webapp" / "server.py").read_text(encoding="utf-8")
    guard_re = re.compile(
        r'if os\.environ\.get\("BRAIN_OPEN_ENDED", "0"\)\.strip\(\)\.lower\(\) in \("1", "true", "on", "yes"\):'
        r'\s*\n\s*try:\s*\n\s*from webapp import open_ended_chat as _OE', re.M)
    gated = bool(guard_re.search(src))
    n_imports = len(re.findall(r'from webapp import open_ended_chat', src))
    return gated and n_imports == 1


def _check_env_flag_control():
    prior = os.environ.pop("BRAIN_OPEN_ENDED", None)
    try:
        off_is_false = OE.open_ended_enabled() is False
        os.environ["BRAIN_OPEN_ENDED"] = "1"
        on_is_true = OE.open_ended_enabled() is True
    finally:
        if prior is None:
            os.environ.pop("BRAIN_OPEN_ENDED", None)
        else:
            os.environ["BRAIN_OPEN_ENDED"] = prior
    return off_is_false, on_is_true


def main():
    saved = json.load(open(SAVED_REPLIES))
    by_topic = {r["topic"]: r["raw"] for r in saved["known_rows"]}

    # ---- (4) flag-off byte-identical -----------------------------------------------------------------------
    off_path_gated = _check_off_path_gating()
    off_is_false, on_is_true = _check_env_flag_control()

    # ---- (1) same-sentence correct+wrong, run through the wired post_filter on a ONE-sentence "reply" --------
    same_sentence_report = []
    for it in SAME_SENTENCE_ITEMS:
        topic, facts = it["topic"], _facts_for(it["topic"])
        filtered = OE.post_filter(it["sentence"], topic, True, facts)
        low = filtered.lower()
        kept_correct = it["must_keep"] in low
        stripped_wrong = it["must_strip"] not in low
        same_sentence_report.append({
            **it, "filtered": filtered, "kept_correct": kept_correct, "stripped_wrong": stripped_wrong,
            "ok": kept_correct and stripped_wrong,
        })
    same_sentence_all_ok = all(r["ok"] for r in same_sentence_report)
    n_kept_correct = sum(r["kept_correct"] for r in same_sentence_report)
    n_stripped_wrong = sum(r["stripped_wrong"] for r in same_sentence_report)

    # ---- (2) no regression: 10/10 known-wrong-details, through the SAME real post_filter --------------------
    known_report = []
    for topic, facts in FACTS.items():
        raw = by_topic[topic]
        wired = OE.post_filter(raw, topic, True, _facts_for(topic))
        low = wired.lower()
        caught = sorted(m for m in MUST_DROP[topic] if m not in low)
        leaked = sorted(m for m in MUST_DROP[topic] if m in low)
        spec = specificity(wired, [(a, v, p) for (a, v, p) in FACTS[topic]], topic=topic)
        known_report.append({"topic": topic, "wired_filtered": wired, "must_drop": sorted(MUST_DROP[topic]),
                              "caught": caught, "leaked": leaked, "nonempty": bool(wired.strip()),
                              "specificity_wired": spec})
    total_must = sum(len(MUST_DROP[t]) for t in FACTS)
    total_caught = sum(len(r["caught"]) for r in known_report)
    total_leaked = sum(len(r["leaked"]) for r in known_report)
    catch_rate = round(total_caught / (total_must or 1), 3)
    all_nonempty = all(r["nonempty"] for r in known_report)
    # prior (2026-08-27 wiring verify) canada specificity was 0 (its own disclosed same-sentence scope limit) --
    # this run's canada specificity is expected to be > 0 now that the correct co-located facts survive.
    prior_canada_specificity_wired = 0
    canada_specificity_improved = known_report[[r["topic"] for r in known_report].index("canada")][
        "specificity_wired"] > prior_canada_specificity_wired

    # ---- (3) moat-safe: no salvageable correct clause -> still a full, non-leaking drop ----------------------
    no_salvage_report = []
    for it in NO_SALVAGE_ITEMS:
        topic, facts = it["topic"], _facts_for(it["topic"])
        filtered = OE.post_filter(it["sentence"], topic, True, facts)
        low = filtered.lower()
        leaked = sorted(m for m in it["must_not_leak"] if m in low)
        no_salvage_report.append({
            "topic": it["topic"], "source": it["source"], "sentence": it["sentence"],
            "must_not_leak": sorted(it["must_not_leak"]), "filtered": filtered, "leaked": leaked,
            "used_honest_fallback": filtered == OE._empty_known_fallback(topic), "ok": len(leaked) == 0,
        })
    no_salvage_all_ok = all(r["ok"] for r in no_salvage_report)
    total_no_salvage_leaked = sum(len(r["leaked"]) for r in no_salvage_report)

    art = {
        "probe": "open_ended_clause_contradiction_filter_verify", "backend": "numpy",
        "source_replies": str(SAVED_REPLIES.relative_to(_REPO)),
        "off_path_import_gated": off_path_gated, "off_is_false": off_is_false, "on_is_true": on_is_true,
        "same_sentence_report": same_sentence_report,
        "same_sentence_n_items": len(SAME_SENTENCE_ITEMS),
        "same_sentence_n_kept_correct": n_kept_correct, "same_sentence_n_stripped_wrong": n_stripped_wrong,
        "same_sentence_all_ok": same_sentence_all_ok,
        "known_report": known_report, "wrong_supplements_total": total_must,
        "caught_wired": total_caught, "leaked_wired": total_leaked, "catch_rate_wired": catch_rate,
        "all_known_replies_nonempty": all_nonempty,
        "canada_specificity_wired": known_report[[r["topic"] for r in known_report].index("canada")]["specificity_wired"],
        "canada_specificity_improved_over_prior_wiring_verify": canada_specificity_improved,
        "no_salvage_report": no_salvage_report, "no_salvage_all_ok": no_salvage_all_ok,
        "no_salvage_leaked": total_no_salvage_leaked,
    }

    v = Verdict("clause-granularity contradiction repair keeps correct + strips wrong in the SAME sentence, "
                "through webapp.open_ended_chat.post_filter, with no regression and no new leak")
    v.require("(4) server.py's open_ended_chat import stays nested under the unchanged BRAIN_OPEN_ENDED guard",
              off_path_gated, expect=True)
    v.require("(4) BRAIN_OPEN_ENDED unset/0 -> open_ended_enabled() is False", off_is_false, expect=True)
    v.require("(4) BRAIN_OPEN_ENDED=1 -> open_ended_enabled() is True", on_is_true, expect=True)
    v.require("(1) every same-sentence correct+wrong item keeps the correct detail AND strips the wrong one",
              same_sentence_all_ok, expect=True)
    v.floor("(1) same-sentence correct-detail retention rate", measured=n_kept_correct / len(SAME_SENTENCE_ITEMS),
            floor=0.99)
    v.floor("(1) same-sentence wrong-detail strip rate", measured=n_stripped_wrong / len(SAME_SENTENCE_ITEMS),
            floor=0.99)
    v.floor("(2) NO-REGRESSION: known-topic wrong-supplement catch rate through the clause-aware post_filter",
            measured=catch_rate, floor=0.8)
    v.require("(2) NO-REGRESSION: no wrong supplement leaks (still 0, same as the 2026-08-27 wiring verify)",
              total_leaked, expect=0)
    v.require("(2) every known-topic filtered reply stays non-empty (still conversational)",
              all_nonempty, expect=True)
    v.require("(2) canada's specificity improves over the prior wiring verify's disclosed 0 "
              "(the same-sentence residual that verify itself named)", canada_specificity_improved, expect=True)
    v.require("(3) MOAT-SAFE: no-salvage sentences (every list item wrong / a bare unsupported number) "
              "still fully drop -- 0 leaks", no_salvage_all_ok, expect=True)
    v.require("(3) MOAT-SAFE: zero total leaks across the no-salvage set", total_no_salvage_leaked, expect=0)
    v.disabled("a general (non-gazetteer) entity check / NLI-based clause extraction",
               why="v1 (this file) makes exactly the TWO observed same-sentence residual shapes "
                   "(appositive/relative-clause date, coordinated relation-object list) clause-safe via "
                   "declared span-removal + a defense-in-depth re-check against sentence_contradicts; a bare "
                   "unsupported claim with no relative-clause/list boundary (canada's '35 million' sentence) "
                   "has no declared-safe repair and keeps falling back to whole-sentence removal, same as "
                   "before this file existed -- the SAME general-entity-check/NLI next rung the 2026-08-21 "
                   "and 2026-08-27 findings already name, now with a smaller residual left to close.")

    go = (off_path_gated and off_is_false and on_is_true and same_sentence_all_ok
          and catch_rate >= 0.8 and total_leaked == 0 and all_nonempty and canada_specificity_improved
          and no_salvage_all_ok and total_no_salvage_leaked == 0)
    decided = v.decide(go=go)
    art["verdict"] = decided
    art["preconditions"] = decided.get("preconditions", [])
    art["GO"] = bool(go)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(art, indent=1))
    print(json.dumps({k: art[k] for k in (
        "off_path_import_gated", "off_is_false", "on_is_true",
        "same_sentence_n_items", "same_sentence_n_kept_correct", "same_sentence_n_stripped_wrong",
        "same_sentence_all_ok", "wrong_supplements_total", "caught_wired", "leaked_wired", "catch_rate_wired",
        "all_known_replies_nonempty", "canada_specificity_wired",
        "canada_specificity_improved_over_prior_wiring_verify",
        "no_salvage_all_ok", "no_salvage_leaked", "GO")}, indent=1))
    print(f"wrote {OUT} -> {decided['status']}")
    return decided["status"]


if __name__ == "__main__":
    main()
