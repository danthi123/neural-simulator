"""WIRING VERIFY: the GO known-topic contradiction filter (2026-08-21) is now live inside
`webapp.open_ended_chat.post_filter`, closing the STUB `contradicts()` gap the wiring commit itself named. (2026-08-27)

CONTEXT. `webapp/open_ended_chat.py` (BRAIN_OPEN_ENDED, default-OFF) already wires the state-driven open-ended
generator + a VERIFY post-filter into `/api/brain-chat`. That post-filter's KNOWN-topic branch called
`contradicts()` -- a DECLARED STUB in `_open_ended_state_driven_generation_derisk.py` that always returns False
(its own comment: "too noisy to parse reliably; we return False ... kept as a named hook for the live wiring's
stronger verify"). The 2026-08-21 `_open_ended_known_supplement_filter_derisk.py` proved a real per-sentence
CONTRADICTION check (a stored relation asserted with a DIFFERENT object, or a bare number/year never in the store)
catches ALL 10 wrong supplements across the 3 saved known-topic replies with 0 leaks (GO,
research/findings/2026-08-21-contradiction-filter-catches-known-topic-wrong-supplements-GO.md). This runner
verifies that GO mechanism, now wired into `webapp.open_ended_chat.post_filter`
(commit-local change, see webapp/open_ended_chat.py), behaves IDENTICALLY through the REAL webapp entry point --
this is a WIRING verify, not a re-derivation of the filter's own logic (already GO and untouched here).

MEMORY-SAFE BY DESIGN: no GPU, no Qwen render. Reuses the SAME saved replies the two prior de-risks used
(research/findings/raw/_open_ended_verify_postfilter_derisk.json). Deterministic filter-logic verification over
fixed text -- same seed-waiver as the 2026-08-21 finding: the evidence is catch/leak counts against a fixed
ground truth, not a stochastic effect, so a seed sweep would not change the answer.

WHAT THIS CHECKS:
  (a) FLAG-OFF BYTE-IDENTICAL. `webapp/server.py` only imports `webapp.open_ended_chat` nested inside the
      `BRAIN_OPEN_ENDED` truthy branch -- verified structurally (the import line still sits directly under the
      unchanged guard) so an OFF run never imports the changed module at all. `open_ended_enabled()` is exercised
      directly as an env-flag control (unset/0 -> False, "1" -> True).
  (b) KNOWN-TOPIC WRONG SUPPLEMENT CAUGHT. `webapp.open_ended_chat.post_filter` (the WIRED version) over the 3
      saved known-topic replies (canada/france/morocco) with their store facts (the SAME (relation, object) ground
      truth the 2026-08-21 de-risk used, expressed as production-shaped (agent, action, patient) triples) must
      drop all 10 named wrong supplements (mexico, 35 million, 1867, italy, germany, switzerland, algeria,
      tunisia, libya, egypt).
  (c) KNOWN-TOPIC SUBSTANCE PRESERVED (reported honestly, not averaged away). Every filtered known-topic reply
      stays non-empty (still conversational). Per-topic grounded-content specificity is reported: france/morocco
      keep their capital + continent (specificity 3-4); canada's correct facts (Ottawa/North America/United
      States) sit in the SAME saved-reply sentences as its wrong supplements (Mexico, 1867), so the per-sentence
      filter drops both together (specificity 0, but the reply stays non-empty). This is an INHERITED,
      already-disclosed v1 scope limit (per-SENTENCE, not per-clause granularity -- the 2026-08-21 finding's own
      "Honest scope" names per-clause/NLI as the next rung), not a regression introduced by this wiring: the
      standalone de-risk measured the identical canada n_kept=1 before this wiring existed.
  (d) LOAD-BEARING LESION. The SAME raw replies through the UNWIRED `_base_post_filter` (its stub `contradicts()`
      intact) must LEAK the wrong supplements -- proving the catch in (b) is attributable to the newly-wired
      filter, not something else in the pipeline.
  (e) UNKNOWN-TOPIC MOAT UNCHANGED. The 8 saved hard (brain-unknown) replies through the WIRED `post_filter` are
      BYTE-IDENTICAL to `_base_post_filter`'s own output (the wrapper short-circuits to the base filter when
      `known=False`) and still signal uncertainty (fabrication suppressed) on every one.

  python -m research.runners._open_ended_chat_known_supplement_wiring_verify
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
from research.runners._open_ended_verify_postfilter_derisk import post_filter as _base_post_filter  # noqa: E402
from research.runners._open_ended_state_driven_generation_derisk import (  # noqa: E402
    specificity, n_sentences, uncertainty_signaled,
)
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_open_ended_chat_known_supplement_wiring_verify.json"
SAVED_REPLIES = _REPO / "research" / "findings" / "raw" / "_open_ended_verify_postfilter_derisk.json"

# The topic's store facts, in the SAME (relation, object) shape the 2026-08-21 de-risk's own ground-truth FACTS
# table used, expressed here as production-shaped (agent, action, patient) triples (what a real `retrieve()` call
# returns) -- (b) verifies the ADAPTER (`_facts_as_relation_pairs`) + the wired filter together, end to end.
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


def _check_off_path_gating():
    """(a) structural: webapp/server.py's `open_ended_chat` import is still nested directly under the unchanged
    BRAIN_OPEN_ENDED truthy guard -- so a default-off run never imports the module this wiring changed."""
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
    hard_rows = saved["hard_rows"]

    # ---- (a) flag-off byte-identical -------------------------------------------------------------------------
    off_path_gated = _check_off_path_gating()
    off_is_false, on_is_true = _check_env_flag_control()

    # ---- (b) + (c) known-topic: wired filter -------------------------------------------------------------------
    known_report = []
    for topic, facts in FACTS.items():
        raw = by_topic[topic]
        wired = OE.post_filter(raw, topic, True, facts)
        low = wired.lower()
        caught = sorted(m for m in MUST_DROP[topic] if m not in low)
        leaked = sorted(m for m in MUST_DROP[topic] if m in low)
        spec = specificity(wired, facts, topic=topic)
        known_report.append({
            "topic": topic, "wired_filtered": wired, "must_drop": sorted(MUST_DROP[topic]),
            "caught": caught, "leaked": leaked, "nonempty": bool(wired.strip()),
            "n_sentences_wired": n_sentences(wired), "specificity_wired": spec,
        })

    total_must = sum(len(MUST_DROP[t]) for t in FACTS)
    total_caught_wired = sum(len(r["caught"]) for r in known_report)
    total_leaked_wired = sum(len(r["leaked"]) for r in known_report)
    catch_rate_wired = round(total_caught_wired / (total_must or 1), 3)
    all_nonempty_known = all(r["nonempty"] for r in known_report)

    # ---- (d) load-bearing lesion: the SAME raw replies through the UNWIRED base filter (stub contradicts) -------
    lesion_report = []
    for topic, facts in FACTS.items():
        raw = by_topic[topic]
        lesioned = _base_post_filter(raw, topic, True, facts)
        low = lesioned.lower()
        caught = sorted(m for m in MUST_DROP[topic] if m not in low)
        leaked = sorted(m for m in MUST_DROP[topic] if m in low)
        lesion_report.append({"topic": topic, "lesioned_filtered": lesioned, "caught": caught, "leaked": leaked})
    total_caught_lesioned = sum(len(r["caught"]) for r in lesion_report)
    total_leaked_lesioned = sum(len(r["leaked"]) for r in lesion_report)
    catch_rate_lesioned = round(total_caught_lesioned / (total_must or 1), 3)
    # (d) attribution: what fraction of the wired catch rate is NOT present in the lesioned (stub) control --
    # the gap#5-shaped question, asked explicitly rather than leaving treatment/control sitting un-subtracted.
    attribution = attributable_to("known-topic contradiction catch: wiring vs lesioned stub",
                                   catch_rate_wired, catch_rate_lesioned)

    # ---- (e) unknown-topic moat unchanged ------------------------------------------------------------------------
    hard_report = []
    for r in hard_rows:
        topic, raw = r["topic"], r["raw"]
        wired = OE.post_filter(raw, topic, False, [])
        base = _base_post_filter(raw, topic, False, [])
        hard_report.append({
            "topic": topic, "byte_identical_to_base": (wired == base),
            "fabrication_suppressed_wired": bool(uncertainty_signaled(wired)),
        })
    hard_byte_identical = all(r["byte_identical_to_base"] for r in hard_report)
    hard_fab_suppressed = all(r["fabrication_suppressed_wired"] for r in hard_report)

    art = {
        "probe": "open_ended_chat_known_supplement_wiring_verify", "backend": "numpy",
        "source_replies": str(SAVED_REPLIES.relative_to(_REPO)),
        "off_path_import_gated": off_path_gated, "off_is_false": off_is_false, "on_is_true": on_is_true,
        "known_report": known_report, "lesion_report": lesion_report, "hard_report": hard_report,
        "wrong_supplements_total": total_must,
        "caught_wired": total_caught_wired, "leaked_wired": total_leaked_wired, "catch_rate_wired": catch_rate_wired,
        "caught_lesioned": total_caught_lesioned, "leaked_lesioned": total_leaked_lesioned,
        "catch_rate_lesioned": catch_rate_lesioned, "attributable_to_wiring": attribution,
        "all_known_replies_nonempty": all_nonempty_known,
        "hard_byte_identical_to_base": hard_byte_identical, "hard_fabrication_suppressed": hard_fab_suppressed,
        "specificity_by_topic": {r["topic"]: r["specificity_wired"] for r in known_report},
    }

    v = Verdict("the GO known-topic contradiction filter is correctly wired into webapp.open_ended_chat.post_filter")
    v.require("(a) server.py's open_ended_chat import stays nested under the unchanged BRAIN_OPEN_ENDED guard "
              "(off path never imports the changed module)", off_path_gated, expect=True)
    v.require("(a) BRAIN_OPEN_ENDED unset/0 -> open_ended_enabled() is False", off_is_false, expect=True)
    v.require("(a) BRAIN_OPEN_ENDED=1 -> open_ended_enabled() is True", on_is_true, expect=True)
    v.floor("(b) known-topic wrong-supplement catch rate through the WIRED webapp.open_ended_chat.post_filter",
            measured=catch_rate_wired, floor=0.8)
    v.require("(b) no wrong supplement leaks through the wired filter", total_leaked_wired, expect=0)
    v.require("(c) every known-topic filtered reply stays non-empty (still conversational)",
              all_nonempty_known, expect=True)
    v.control("(d) known-topic wrong-supplement catch rate: WIRED vs LESIONED (stub contradicts() restored)",
              treatment=catch_rate_wired, control=catch_rate_lesioned, min_separation=0.5,
              note="lesioning back to the stub must make the SAME wrong supplements leak -- attributes the catch "
                   "in (b) to this wiring, not something else in the pipeline")
    v.require("(d) the lesioned (unwired) path leaks the wrong supplements the wired path catches",
              total_leaked_lesioned, expect=lambda x: x >= 8)
    v.require("(e) unknown-topic filtered output is byte-identical to the base filter on every saved hard reply",
              hard_byte_identical, expect=True)
    v.require("(e) unknown-topic fabrication stays suppressed on every saved hard reply (moat unchanged)",
              hard_fab_suppressed, expect=True)
    v.disabled("per-topic specificity / correct-substance survival at per-CLAUSE granularity",
               why="v1 (this wiring, unmodified from the 2026-08-21 de-risk) drops entire SENTENCES, not clauses: "
                   "france/morocco keep their capital+continent (specificity 3-4) but canada's correct facts "
                   "co-occur, in this saved Qwen reply, with its wrong supplements in the SAME sentences (Ottawa + "
                   "'founded in 1867'; North America/United States + 'and Mexico'), so specificity_wired=0 for "
                   "canada though the reply stays non-empty. Inherited from the approved de-risk (its own canada "
                   "n_kept=1 predates this wiring); the general fix (per-clause split or a store-backed entity "
                   "check / NLI model) is the SAME next rung the 2026-08-21 finding already names.")
    v.disabled("live retrieval against the CURRENT on-disk LTM bundle for these exact topic strings",
               why="the shipped default bundle (~/Projects/sim-data/knowledge_bundles/wikidata_core_15k) keys "
                   "country entities like 'canada_portal', not the bare 'canada' a user types, so "
                   "retrieve('canada') against TODAY's bundle returns [] independent of this wiring -- a "
                   "pre-existing topic-routing/store-content gap, out of scope for this filter wiring. This "
                   "verify uses the topic's store facts in the (agent, action, patient) triple shape the "
                   "2026-08-21 de-risk itself documented as coming from the store, run through the REAL "
                   "production post_filter code path (webapp.open_ended_chat.post_filter), which is what proves "
                   "the WIRING (not the store's current content).")

    go = (off_path_gated and off_is_false and on_is_true and catch_rate_wired >= 0.8 and total_leaked_wired == 0
          and all_nonempty_known and catch_rate_lesioned <= catch_rate_wired - 0.5 and total_leaked_lesioned >= 8
          and hard_byte_identical and hard_fab_suppressed)
    decided = v.decide(go=go)
    art["verdict"] = decided
    art["preconditions"] = decided.get("preconditions", [])
    art["GO"] = bool(go)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(art, indent=1))
    print(json.dumps({k: art[k] for k in (
        "off_path_import_gated", "off_is_false", "on_is_true",
        "wrong_supplements_total", "caught_wired", "leaked_wired", "catch_rate_wired",
        "caught_lesioned", "leaked_lesioned", "catch_rate_lesioned",
        "all_known_replies_nonempty", "hard_byte_identical_to_base", "hard_fabrication_suppressed",
        "specificity_by_topic", "GO")}, indent=1))
    print(f"wrote {OUT} -> {decided['status']}")
    return decided["status"]


if __name__ == "__main__":
    main()
