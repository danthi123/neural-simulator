"""WIRING VERIFY: NPHeadBinder + entailment classification are LOAD-BEARING on the LIVE open-text
moat verifier (`webapp.open_ended_chat.post_filter`'s known-topic path), via the new
`webapp.np_entailment_moat_gate` module + `BRAIN_OPEN_ENDED_NP_ENTAILMENT` flag. (2026-09-01)

CONTEXT. The board frontier row asked to wire NPHeadBinder (spiking NP-boundary binding,
`_spiking_np_boundary_extraction_derisk.py`) + entailment (`FactStore`/`classify_claim`,
`_open_text_moat_verifier_derisk.py`) into the LIVE open-text moat verifier, and to prove the
wiring is LOAD-BEARING (a live-chat faculty is real only when it changes the output -- memory
`feedback_faculties_must_drive_not_observe`), not a hollow checkbox flip. The pre-existing live
known-topic filter (`_clause_filter_sentence` -> `sentence_contradicts`,
`_open_ended_known_supplement_filter_derisk.py`) is a HOST GAZETTEER limited to THREE relation
shapes (borders/continent/capital) + a bare number/year regex -- a wrong supplement on any OTHER
relation trips no branch and leaks. This runner proves (1) the new gate genuinely catches that
class the gazetteer cannot, through the REAL `webapp.open_ended_chat.post_filter` entry point,
(2) it does not regress real saved known-topic Qwen replies, and (3) the catch is attributable
SPECIFICALLY to the NPHeadBinder extraction and the entailment classification, each independently
lesioned.

WHAT THIS CHECKS:
  (a) FLAG CONTROL + OFF-PATH IMPORT GATING. `np_entailment_enabled()` env-flag semantics, and that
      `post_filter` never imports `webapp.np_entailment_moat_gate` while the flag is off (byte-
      identical + side-effect-free default).
  (b) LOAD-BEARING CATCH (the flag-level lesion). Two adversarial cases, each a wrong supplement on
      a relation OUTSIDE the gazetteer's 3-relation coverage (mercury/discovered/neptune,
      einstein/invented/telephone). FLAG ON must drop the false clause; FLAG OFF (identical
      `post_filter` call, only the flag differs) must LEAK it -- the SAME function, same inputs,
      verdict changes ONLY with the flag. This is the flag-level load-bearing proof + its lesion.
  (c) COMPONENT-LEVEL LESIONS (stronger than the flag alone -- attributes the catch to the two
      NAMED mechanisms, not "some code path"). With the flag ON:
        - ENTAILMENT LESION: `research.runners._open_text_moat_verifier_derisk.classify_claim` is
          monkeypatched to always return "grounded" (the entailment classifier can no longer say
          no). The catch must VANISH (the false clause leaks again) even though NPHeadBinder still
          extracts the triple correctly and the flag is still on.
        - EXTRACTION LESION: `research.runners._spiking_np_boundary_extraction_derisk.
          extract_svo_npbind` is monkeypatched to always return `(None, None)` (NPHeadBinder-based
          extraction can no longer parse anything). The catch must VANISH too, even though
          entailment classification is untouched and the flag is still on.
      Both lesions are restored (module attributes reset) before the next check runs.
  (d) FALSE-REJECT SAFETY (the gate's own declared monotonic-only scope, MEASURED not asserted).
      Three cases that must be KEPT UNCHANGED whether the flag is on or off: a grounded non-copula
      claim about the topic, an off-topic-agent clause (out of the retrieved fact store's scope),
      and a copula predicate-nominal clause (excluded by design -- see the gate module's docstring).
  (e) REAL-DATA REGRESSION. The 3 saved known-topic Qwen replies this arc's own prior wiring verify
      used (canada/france/morocco, `_open_ended_verify_postfilter_derisk.json`) through the WIRED
      `post_filter`, flag on vs off: must be BYTE-IDENTICAL (the new gate adds zero false rejects on
      real generated prose whose wrong content already falls inside the gazetteer's own coverage).

  python -m research.runners._np_entailment_moat_gate_wiring_verify
"""
from __future__ import annotations
import json, os, sys, time
os.environ.setdefault("SIM_BACKEND", "numpy")
import logging; logging.disable(logging.INFO)
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from webapp import open_ended_chat as OE  # noqa: E402 -- the module under test
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_np_entailment_moat_gate_wiring_verify.json"
SAVED_REPLIES = _REPO / "research" / "findings" / "raw" / "_open_ended_verify_postfilter_derisk.json"

FLAG = "BRAIN_OPEN_ENDED_NP_ENTAILMENT"

# (b) LOAD-BEARING CATCH cases: a wrong supplement on a relation the gazetteer does not recognize.
CATCH_CASES = [
    {"name": "mercury_discovered", "topic": "mercury", "facts": [("mercury", "orbits", "sun")],
     "raw": "Mercury orbits the sun. Mercury discovered Neptune.",
     "true_fragment": "mercury orbits the sun", "false_fragment": "discovered neptune"},
    {"name": "einstein_invented", "topic": "einstein", "facts": [("einstein", "developed", "relativity")],
     "raw": "Einstein developed relativity. Einstein invented the telephone.",
     "true_fragment": "developed relativity", "false_fragment": "invented"},
]

# (d) FALSE-REJECT SAFETY cases: must be unchanged flag on vs off (in/out of the gate's declared scope).
SAFETY_CASES = [
    {"name": "grounded_kept", "topic": "mercury", "facts": [("mercury", "orbits", "sun")],
     "raw": "Mercury orbits the sun.", "note": "grounded non-copula claim about the topic -> kept both ways"},
    {"name": "offtopic_agent_untouched", "topic": "einstein", "facts": [("einstein", "developed", "relativity")],
     "raw": "Newton discovered gravity.",
     "note": "subject != retrieved topic -> out of this gate's adjudicable scope, untouched both ways"},
    {"name": "copula_untouched", "topic": "canada",
     "facts": [("canada", "isa", "country"), ("canada", "capital", "ottawa"),
               ("canada", "continent", "north america"), ("canada", "borders", "united states")],
     "raw": "Canada is a vast country located in North America.",
     "note": "copula predicate nominal -> excluded by design (would false-reject elaborated true content)"},
]

# (e) real saved known-topic replies (byte-identical regression check)
FACTS_REAL = {
    "canada":  [("canada", "isa", "country"), ("canada", "capital", "ottawa"),
                ("canada", "continent", "north america"), ("canada", "borders", "united states")],
    "france":  [("france", "isa", "country"), ("france", "capital", "paris"),
                ("france", "continent", "europe"), ("france", "borders", "spain")],
    "morocco": [("morocco", "isa", "country"), ("morocco", "capital", "rabat"),
                ("morocco", "continent", "africa"), ("morocco", "borders", "spain")],
}


def _set_flag(on: bool):
    if on:
        os.environ[FLAG] = "1"
    else:
        os.environ.pop(FLAG, None)


def _check_flag_semantics():
    prior = os.environ.pop(FLAG, None)
    try:
        off_is_false = OE.np_entailment_enabled() is False
        os.environ[FLAG] = "1"
        on_is_true = OE.np_entailment_enabled() is True
    finally:
        if prior is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = prior
    return off_is_false, on_is_true


def _check_off_path_no_import():
    """(a) flag OFF -> post_filter must never import webapp.np_entailment_moat_gate."""
    sys.modules.pop("webapp.np_entailment_moat_gate", None)
    _set_flag(False)
    OE.post_filter("Mercury orbits the sun.", "mercury", True, [("mercury", "orbits", "sun")])
    not_imported = "webapp.np_entailment_moat_gate" not in sys.modules
    return not_imported


def run_catch_cases():
    rows = []
    for c in CATCH_CASES:
        _set_flag(False)
        off = OE.post_filter(c["raw"], c["topic"], True, c["facts"])
        _set_flag(True)
        on = OE.post_filter(c["raw"], c["topic"], True, c["facts"])
        off_low, on_low = off.lower(), on.lower()
        leaked_off = c["false_fragment"] in off_low
        leaked_on = c["false_fragment"] in on_low
        true_kept_off = c["true_fragment"] in off_low
        true_kept_on = c["true_fragment"] in on_low
        rows.append({
            "name": c["name"], "topic": c["topic"], "raw": c["raw"],
            "flag_off_filtered": off, "flag_on_filtered": on,
            "false_fragment": c["false_fragment"],
            "leaked_flag_off": leaked_off, "leaked_flag_on": leaked_on,
            "true_fragment_kept_flag_off": true_kept_off, "true_fragment_kept_flag_on": true_kept_on,
            # the load-bearing signature: leak WITHOUT the gate, no leak WITH it, true content survives both ways
            "load_bearing": bool(leaked_off and not leaked_on and true_kept_off and true_kept_on),
        })
    _set_flag(False)
    return rows


def run_safety_cases():
    rows = []
    for c in SAFETY_CASES:
        _set_flag(False)
        off = OE.post_filter(c["raw"], c["topic"], True, c["facts"])
        _set_flag(True)
        on = OE.post_filter(c["raw"], c["topic"], True, c["facts"])
        rows.append({"name": c["name"], "note": c["note"], "flag_off_filtered": off, "flag_on_filtered": on,
                     "unchanged": (off == on)})
    _set_flag(False)
    return rows


def run_component_lesions():
    """(c) With the flag ON, lesion ENTAILMENT then EXTRACTION independently; the catch must vanish
    each time. Monkeypatches the SOURCE modules' attributes (gate_sentence re-imports them by name
    on every call, so patching the source is visible immediately; restored in a finally block)."""
    import research.runners._open_text_moat_verifier_derisk as _moat_mod
    import research.runners._spiking_np_boundary_extraction_derisk as _npb_mod

    real_classify_claim = _moat_mod.classify_claim
    real_extract_svo_npbind = _npb_mod.extract_svo_npbind

    rows = []
    _set_flag(True)

    # -- entailment lesion: classify_claim always says "grounded" --------------------------------------------
    try:
        _moat_mod.classify_claim = lambda claim, store: "grounded"
        for c in CATCH_CASES:
            filtered = OE.post_filter(c["raw"], c["topic"], True, c["facts"])
            leaked = c["false_fragment"] in filtered.lower()
            rows.append({"lesion": "entailment_classify_claim_always_grounded", "name": c["name"],
                         "filtered": filtered, "leaked": leaked,
                         "catch_vanished": leaked})   # catch vanishing == leak reappears
    finally:
        _moat_mod.classify_claim = real_classify_claim

    # -- extraction lesion: extract_svo_npbind always returns (None, None) (nothing parses) -----------------
    try:
        _npb_mod.extract_svo_npbind = lambda clause, parser, np_binder: (None, None)
        for c in CATCH_CASES:
            filtered = OE.post_filter(c["raw"], c["topic"], True, c["facts"])
            leaked = c["false_fragment"] in filtered.lower()
            rows.append({"lesion": "extraction_extract_svo_npbind_always_none", "name": c["name"],
                         "filtered": filtered, "leaked": leaked,
                         "catch_vanished": leaked})
    finally:
        _npb_mod.extract_svo_npbind = real_extract_svo_npbind

    # -- sanity: with BOTH restored + flag still on, the catch must be back (proves the finally blocks worked) --
    post_restore = []
    for c in CATCH_CASES:
        filtered = OE.post_filter(c["raw"], c["topic"], True, c["facts"])
        leaked = c["false_fragment"] in filtered.lower()
        post_restore.append({"name": c["name"], "filtered": filtered, "leaked": leaked})

    _set_flag(False)
    return rows, post_restore


def run_real_data_regression():
    saved = json.load(open(SAVED_REPLIES))
    by_topic = {r["topic"]: r["raw"] for r in saved["known_rows"]}
    rows = []
    for topic, facts in FACTS_REAL.items():
        raw = by_topic[topic]
        _set_flag(False)
        off = OE.post_filter(raw, topic, True, facts)
        _set_flag(True)
        on = OE.post_filter(raw, topic, True, facts)
        rows.append({"topic": topic, "byte_identical": (off == on), "flag_off_filtered": off, "flag_on_filtered": on})
    _set_flag(False)
    return rows


def main():
    t0 = time.time()

    off_path_no_import = _check_off_path_no_import()
    off_is_false, on_is_true = _check_flag_semantics()

    catch_rows = run_catch_cases()
    safety_rows = run_safety_cases()
    lesion_rows, post_restore_rows = run_component_lesions()
    real_rows = run_real_data_regression()

    # (c) ATTRIBUTION: measuring both arms (flag-on-unlesioned catch rate vs each lesioned catch rate) is
    # not the same as asking whose the difference was (tools/gates/attribution_required.py) -- subtract
    # them explicitly, the same call shape the prior known-supplement wiring verify used.
    n_catch = len(CATCH_CASES)
    catch_rate_unlesioned = sum(1 for r in catch_rows if not r["leaked_flag_on"]) / n_catch
    entail_lesion_rows = [r for r in lesion_rows if r["lesion"] == "entailment_classify_claim_always_grounded"]
    extract_lesion_rows = [r for r in lesion_rows if r["lesion"] == "extraction_extract_svo_npbind_always_none"]
    catch_rate_entailment_lesioned = sum(1 for r in entail_lesion_rows if not r["leaked"]) / len(entail_lesion_rows)
    catch_rate_extraction_lesioned = sum(1 for r in extract_lesion_rows if not r["leaked"]) / len(extract_lesion_rows)
    attribution_entailment = attributable_to(
        "known-topic wrong-relation catch: flag ON unlesioned vs ENTAILMENT-lesioned (classify_claim always "
        "'grounded')", catch_rate_unlesioned, catch_rate_entailment_lesioned)
    attribution_extraction = attributable_to(
        "known-topic wrong-relation catch: flag ON unlesioned vs EXTRACTION-lesioned (extract_svo_npbind "
        "always None)", catch_rate_unlesioned, catch_rate_extraction_lesioned)

    all_load_bearing = all(r["load_bearing"] for r in catch_rows)
    all_safety_unchanged = all(r["unchanged"] for r in safety_rows)
    all_lesions_vanish = all(r["catch_vanished"] for r in lesion_rows)
    all_real_byte_identical = all(r["byte_identical"] for r in real_rows)
    all_post_restore_leaked_false = all(not r["leaked"] for r in post_restore_rows)  # catch is BACK after restore

    art = {
        "probe": "np_entailment_moat_gate_wiring_verify",
        "flag": FLAG,
        "off_path_no_import": off_path_no_import, "off_is_false": off_is_false, "on_is_true": on_is_true,
        "catch_cases": catch_rows, "safety_cases": safety_rows,
        "component_lesions": lesion_rows, "post_restore_sanity": post_restore_rows,
        "real_data_regression": real_rows,
        "catch_rate_unlesioned": catch_rate_unlesioned,
        "catch_rate_entailment_lesioned": catch_rate_entailment_lesioned,
        "catch_rate_extraction_lesioned": catch_rate_extraction_lesioned,
        "attributable_to_entailment": attribution_entailment,
        "attributable_to_extraction": attribution_extraction,
    }

    v = Verdict("NPHeadBinder + entailment are load-bearing on webapp.open_ended_chat.post_filter's "
                "live moat verdict, and the coupling is attributable to each named mechanism")
    v.require("(a) flag OFF -> post_filter never imports webapp.np_entailment_moat_gate", off_path_no_import,
              expect=True)
    v.require("(a) BRAIN_OPEN_ENDED_NP_ENTAILMENT unset/0 -> np_entailment_enabled() is False", off_is_false,
              expect=True)
    v.require("(a) BRAIN_OPEN_ENDED_NP_ENTAILMENT=1 -> np_entailment_enabled() is True", on_is_true, expect=True)
    v.require("(b) every catch case is load-bearing: leaks flag-OFF, caught flag-ON, true content survives "
              "both ways", all_load_bearing, expect=True)
    v.require("(c) lesioning entailment (classify_claim forced to always 'grounded') makes the SAME catches "
              "vanish even with the flag ON", all_lesions_vanish, expect=True)
    v.require("(c) after restoring both lesions the catch reappears (flag ON, nothing monkeypatched) -- proves "
              "the lesions above were real and reversible, not a broken harness", all_post_restore_leaked_false,
              expect=True)
    v.require("(d) false-reject safety: grounded / off-topic / copula cases unchanged flag on vs off",
              all_safety_unchanged, expect=True)
    v.require("(e) real saved known-topic Qwen replies: byte-identical flag on vs off (no new false rejects "
              "on real generated prose)", all_real_byte_identical, expect=True)
    v.control("(b) mercury_discovered catch: flag ON vs flag OFF (same post_filter call, only the flag differs)",
              treatment=int(not catch_rows[0]["leaked_flag_on"]), control=int(not catch_rows[0]["leaked_flag_off"]),
              min_separation=0.5, note="flag ON must catch (1); flag OFF (the lesion) must leak (0)")
    v.require("(c) attribution: the entailment-lesion catch drop is attributable to the manipulation, not "
              "vacuous", attribution_entailment, expect=lambda x: x is not None and x >= 0.99)
    v.require("(c) attribution: the extraction-lesion catch drop is attributable to the manipulation, not "
              "vacuous", attribution_extraction, expect=lambda x: x is not None and x >= 0.99)

    go = (off_path_no_import and off_is_false and on_is_true and all_load_bearing and all_lesions_vanish
          and all_post_restore_leaked_false and all_safety_unchanged and all_real_byte_identical
          and attribution_entailment is not None and attribution_entailment >= 0.99
          and attribution_extraction is not None and attribution_extraction >= 0.99)
    decided = v.decide(go=go)
    art["verdict"] = decided
    art["preconditions"] = decided.get("preconditions", [])
    art["GO"] = bool(go)
    art["elapsed_seconds"] = round(time.time() - t0, 1)

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(art, indent=1))
    print(json.dumps({k: art[k] for k in (
        "off_path_no_import", "off_is_false", "on_is_true",
        "catch_cases", "safety_cases", "component_lesions", "post_restore_sanity",
        "real_data_regression", "GO")}, indent=1, default=str))
    print(f"wrote {OUT} -> {decided['status']}")
    return decided["status"]


if __name__ == "__main__":
    main()
