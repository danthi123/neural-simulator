"""PARSING-LEVEL VERIFY for the participial + pronoun-referent widening in `webapp.
np_entailment_moat_gate` (`BRAIN_OPEN_ENDED_NP_ENTAILMENT_PARTICIPIAL_PRONOUN_COVERAGE`, 2026-09-01
same-day follow-on to the copula-coverage widening,
`research/findings/2026-09-01-np-entailment-copula-coverage-widening.md`).

THE GAP THIS CHECKS. That finding's own "Honest limits" named two real-traffic construction
classes the copula-coverage widening left untouched (also visible in the moat-safety soak's own
before/after examples #2 and #4): PARTICIPIAL phrases set off by a comma ("City, bordering
Virginia, ..."; "the club, founded in 1892, ...") and PRONOUN-REFERENT sentences whose subject is
a pronoun standing for the known topic ("It's often associated with ..."). This runner is
PARSING-LEVEL ONLY (`_get_spiking_pair()` builds the same tiny (126 + 82)-neuron BridgeParser +
NPHeadBinder pair the live gate itself builds -- RAM-light, no 15k-LTM brain, no cupy/GPU needed;
`SIM_BACKEND=numpy` set below): it calls `webapp.np_entailment_moat_gate.gate_sentence` directly
with hand-built (sentence, topic, facts) inputs, exactly the shape `post_filter` calls it with.

WHAT THIS MEASURES:
  (1) NEW-CATCH RATE on a fabrication battery -- 3 participial-relation-conflict cases (borders /
      founded-year / discovered) and 3 pronoun-referent category-conflict cases (sport x2,
      nationality x1 -- the last one exercises the SAME-DAY category-lexicon widening) -- flag ON
      must catch (gate_sentence -> None), flag OFF must leak (gate_sentence -> sent unchanged).
  (2) FALSE-POSITIVE RATE on a true-sentence battery -- correct participial claims, a
      no-matching-relation-fact participial (must NOT trip -- no store opinion to conflict with),
      a negated / unrecognized-participle / comma-less participial (all conservative backoffs),
      a correct pronoun claim, a no-category-word pronoun, a negated / present-participle /
      passive pronoun predicate, a non-sentence-initial pronoun (out of scope by construction),
      plus the parent gate's own saved SAFETY_CASES (offtopic_agent, grounded_kept) -- flag ON
      must NOT change any of these (byte-identical to flag OFF).
  (3) REGRESSION CHECK on the copula-coverage widening's OWN saved battery -- the category-lexicon
      widening (sport -> +nationality/profession/religion) is a SHARED edit, so this runner also
      re-runs every case from `_np_entailment_copula_coverage_verify.py` (with ONLY the copula
      flag on, participial/pronoun flag off, exactly its own original test conditions) and
      confirms every result is IDENTICAL to that finding's own saved verdict artifact.
  (4) BYTE-IDENTICAL-OFF, measured against the ACTUAL pre-widening file content (`git show
      HEAD:webapp/np_entailment_moat_gate.py`, loaded as an isolated module), not assumed from
      reading the diff: with the NEW flag OFF (the default), every case in both new batteries
      above must produce the IDENTICAL `gate_sentence` output the original, unmodified file
      produces (the copula flag, independently, may be on or off in either module -- this checks
      the NEW flag's off-state specifically, mirroring the copula verify's own methodology).

Run: python -m research.runners._np_entailment_participial_pronoun_coverage_verify
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")   # tiny parsing nets; CPU is plenty, no GPU init

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from webapp import np_entailment_moat_gate as GATE  # noqa: E402 -- module under test
from tools.verdict import Verdict  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_np_entailment_participial_pronoun_coverage_verify.json"
FLAG = GATE._FLAG_PARTICIPIAL_PRONOUN_COVERAGE
COPULA_FLAG = GATE._FLAG_COPULA_COVERAGE


def _set_flag(on: bool):
    if on:
        os.environ[FLAG] = "1"
    else:
        os.environ.pop(FLAG, None)


def _set_copula_flag(on: bool):
    if on:
        os.environ[COPULA_FLAG] = "1"
    else:
        os.environ.pop(COPULA_FLAG, None)


# =================================================================================================
# (1) FABRICATION battery -- participial-relation-conflict + pronoun-referent category-conflict
#     claims that must be CAUGHT (flag ON -> None).
# =================================================================================================

CASTLEFORD_FACTS = [("castleford_f_c", "country", "united_kingom"), ("castleford_f_c", "sport", "rugby_leauge"),
                     ("castleford_f_c", "borders", "normanton")]

FABRICATION_CASES = [
    # -- participial: borders --
    {"name": "castleford_borders_conflict", "topic": "castleford_f_c", "facts": CASTLEFORD_FACTS,
     "sent": "Castleford FC, bordering Wakefield to the south, is a rugby club.",
     "note": "present-participle 'bordering' vs stored borders=normanton -- wakefield not in the store patient"},
    # -- participial: founded year --
    {"name": "deutsche_arbeiter_partei_founded_year_conflict", "topic": "deutsche_arbeiter_partei",
     "facts": [("deutsche_arbeiter_partei", "founded", "1919")],
     "sent": "Deutsche Arbeiter Partei, founded in 1920, was a political party.",
     "note": "past-participle 'founded' + leading-preposition-stripped year object vs stored founded=1919"},
    # -- participial: discovered (present-participle form, not the map's bare past-participle) --
    {"name": "fleming_discovered_conflict", "topic": "alexander_fleming",
     "facts": [("alexander_fleming", "discovered", "penicillin")],
     "sent": "Alexander Fleming, discovering radium in 1898, changed medicine.",
     "note": "'-ing' participle form of a lexicon entry vs stored discovered=penicillin"},
    # -- pronoun-referent: sport category conflict (the exact class the soak's #1 example took) --
    {"name": "castleford_pronoun_football_conflict", "topic": "castleford_f_c", "facts": CASTLEFORD_FACTS,
     "sent": "It's a well-known football team in the area.",
     "note": "leading pronoun 'It's' standing for the topic, category conflict vs stored sport=rugby_leauge"},
    {"name": "chicago_bulls_pronoun_baseball_conflict", "topic": "chicago_bulls",
     "facts": [("chicago_bulls", "sport", "basketball")],
     "sent": "They're a professional baseball team.",
     "note": "second pronoun form ('They're'), second sport pair (basketball vs baseball)"},
    # -- pronoun-referent: nationality family (exercises the SAME-DAY category-lexicon widening) --
    {"name": "band_pronoun_nationality_conflict", "topic": "some_uk_band",
     "facts": [("some_uk_band", "nationality", "british")],
     "sent": "It's an American rock band.",
     "note": "nationality family (new, this widening) -- 'American' predicate vs stored nationality=british"},
]


# =================================================================================================
# (2) TRUE battery -- must NOT be caught (flag ON output == flag OFF output) on either arm.
# =================================================================================================

TRUE_CASES = [
    # -- participial: correct value on each of the 3 fabrication shapes --
    {"name": "castleford_borders_correct", "topic": "castleford_f_c", "facts": CASTLEFORD_FACTS,
     "sent": "Castleford FC, bordering Normanton to the south, is a rugby club.",
     "note": "SAME shape as the caught case, correct border town -- false-positive guard"},
    {"name": "deutsche_arbeiter_partei_founded_year_correct", "topic": "deutsche_arbeiter_partei",
     "facts": [("deutsche_arbeiter_partei", "founded", "1919")],
     "sent": "Deutsche Arbeiter Partei, founded in 1919, was a political party.",
     "note": "correct founding year -- must not be flagged"},
    {"name": "fleming_discovered_correct", "topic": "alexander_fleming",
     "facts": [("alexander_fleming", "discovered", "penicillin")],
     "sent": "Alexander Fleming, discovering penicillin in 1928, changed medicine.",
     "note": "correct discovery -- must not be flagged"},
    {"name": "no_matching_relation_fact", "topic": "some_topic", "facts": [("some_topic", "country", "canada")],
     "sent": "Some Topic, founded in 1990, is well known.",
     "note": "store has NO founded fact for this topic at all -- must never trip (no opinion to conflict with)"},
    {"name": "participial_negated", "topic": "castleford_f_c", "facts": CASTLEFORD_FACTS,
     "sent": "Castleford FC, not bordering Wakefield, is a rugby club.",
     "note": "negated participial -- widened path must back off, never assert the negation itself"},
    {"name": "participial_unrecognized_verb", "topic": "castleford_f_c", "facts": CASTLEFORD_FACTS,
     "sent": "Castleford FC, sitting near the river, is a rugby club.",
     "note": "'sitting' is not in the recognized relational-participle lexicon -- honest coverage limit"},
    {"name": "participial_no_comma", "topic": "castleford_f_c", "facts": CASTLEFORD_FACTS,
     "sent": "Castleford FC bordering Wakefield is a rugby club.",
     "note": "no comma at all -- the whole-sentence extraction requires >=2 comma-segments, conservative no-op"},
    # -- pronoun-referent: correct / no-category / guarded shapes --
    {"name": "pronoun_correct_sport", "topic": "castleford_f_c", "facts": CASTLEFORD_FACTS,
     "sent": "It's a well-known rugby team in the area.",
     "note": "SAME word as the store's sport family (rugby) -- must not be flagged"},
    {"name": "pronoun_no_category_word", "topic": "castleford_f_c", "facts": CASTLEFORD_FACTS,
     "sent": "It's often associated with local sports culture.",
     "note": "no recognized category word ('sports' plural is not a lexicon entry) -- must not be flagged"},
    {"name": "pronoun_negated", "topic": "castleford_f_c", "facts": CASTLEFORD_FACTS,
     "sent": "It's not a football club.",
     "note": "negated pronoun predicate -- widened path must back off"},
    {"name": "pronoun_present_participle", "topic": "castleford_f_c", "facts": CASTLEFORD_FACTS,
     "sent": "It's playing football this weekend.",
     "note": "progressive aspect, not an identity predicate -- widened path must back off"},
    {"name": "pronoun_passive", "topic": "castleford_f_c", "facts": CASTLEFORD_FACTS,
     "sent": "It's built by unknown architects.",
     "note": "passive construction -- widened path must back off, not misread 'built' as a predicate nominal"},
    {"name": "pronoun_not_sentence_initial", "topic": "castleford_f_c", "facts": CASTLEFORD_FACTS,
     "sent": "Meanwhile, it is a football club.",
     "note": "pronoun is not the sentence's FIRST word -- out of scope by construction, an honest residual, "
             "not a false positive (identical on/off)"},
    # -- parent gate's own saved SAFETY_CASES, reused for extra regression assurance --
    {"name": "offtopic_agent_untouched", "topic": "einstein",
     "facts": [("einstein", "developed", "relativity")],
     "sent": "Newton discovered gravity.",
     "note": "parent gate's own SAFETY_CASE: subject != retrieved topic -- out of scope either way"},
    {"name": "grounded_kept", "topic": "mercury", "facts": [("mercury", "orbits", "sun")],
     "sent": "Mercury orbits the sun.",
     "note": "parent gate's own SAFETY_CASE: grounded non-copula claim -- unaffected by this widening"},
]


def _run_battery(cases):
    rows = []
    for c in cases:
        _set_flag(False)
        off = GATE.gate_sentence(c["sent"], c["topic"], c["facts"])
        _set_flag(True)
        on = GATE.gate_sentence(c["sent"], c["topic"], c["facts"])
        _set_flag(False)
        rows.append({"name": c["name"], "note": c["note"], "topic": c["topic"], "sent": c["sent"],
                     "flag_off_result": off, "flag_on_result": on,
                     "caught_flag_on": (on is None), "leaked_flag_off": (off is not None)})
    return rows


def _load_original_module():
    """Load the pre-this-widening `webapp/np_entailment_moat_gate.py` from git HEAD (NOT the
    working tree) as an isolated module -- the same committed original the copula-coverage verify
    already compares against (this branch was cut from origin/main AFTER that widening landed, so
    HEAD here IS the copula-coverage version, exactly the module this new flag is additive on top
    of)."""
    src = subprocess.check_output(["git", "show", "HEAD:webapp/np_entailment_moat_gate.py"],
                                   cwd=str(_REPO)).decode("utf-8")
    tmp_path = _REPO / "research" / "findings" / "raw" / "_np_entailment_moat_gate_ORIGINAL_scratch2.py"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(src)
    spec = importlib.util.spec_from_file_location("_np_entailment_moat_gate_original2", str(tmp_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tmp_path.unlink()   # scratch file only -- not a persisted artifact
    return mod


def _byte_identical_off_check(cases, original_mod):
    """For every case, NEW flag OFF (copula flag also OFF, matching the default state), confirm
    the CURRENT (widened) module's output equals the ORIGINAL (pre-this-widening) module's output
    on the identical input."""
    rows = []
    _set_flag(False)
    _set_copula_flag(False)
    for c in cases:
        current = GATE.gate_sentence(c["sent"], c["topic"], c["facts"])
        original = original_mod.gate_sentence(c["sent"], c["topic"], c["facts"])
        rows.append({"name": c["name"], "current_flag_off": current, "original": original,
                     "byte_identical": (current == original)})
    return rows


# =================================================================================================
# (3) Regression check -- re-run the copula-coverage widening's OWN saved battery (copula flag ON,
#     this new flag OFF) against the CURRENT module, and diff against its saved verdict artifact.
# =================================================================================================

def _copula_regression_check():
    copula_verify_path = _REPO / "research" / "findings" / "raw" / "_np_entailment_copula_coverage_verify.json"
    if not copula_verify_path.exists():
        return {"ran": False, "reason": "saved copula-coverage verify artifact not found"}
    saved = json.loads(copula_verify_path.read_text())
    _set_flag(False)   # this new flag OFF -- isolates the copula flag's own behavior
    mismatches = []
    for bucket in ("fabrication_battery", "true_copula_battery"):
        for saved_row in saved[bucket]:
            _set_copula_flag(False)
            off = GATE.gate_sentence(saved_row["sent"], saved_row["topic"],
                                      # facts are not embedded in the saved rows; re-derive from the
                                      # known battery definitions imported below
                                      _COPULA_CASES_BY_NAME[saved_row["name"]]["facts"])
            _set_copula_flag(True)
            on = GATE.gate_sentence(saved_row["sent"], saved_row["topic"],
                                     _COPULA_CASES_BY_NAME[saved_row["name"]]["facts"])
            _set_copula_flag(False)
            if off != saved_row["flag_off_result"] or on != saved_row["flag_on_result"]:
                mismatches.append({"name": saved_row["name"], "bucket": bucket,
                                    "saved_off": saved_row["flag_off_result"], "current_off": off,
                                    "saved_on": saved_row["flag_on_result"], "current_on": on})
    return {"ran": True, "n_checked": sum(len(saved[b]) for b in ("fabrication_battery", "true_copula_battery")),
            "n_mismatches": len(mismatches), "mismatches": mismatches, "no_regression": (len(mismatches) == 0)}


# Re-declared (not imported, to keep this runner import-independent of the other verify module's
# top-level side effects) EXACTLY matching `_np_entailment_copula_coverage_verify.py`'s own
# FABRICATION_CASES + TRUE_CASES facts, keyed by name, for the regression re-check above.
_COPULA_CASTLEFORD_FACTS = [("castleford_f_c", "country", "united_kingom"), ("castleford_f_c", "sport", "rugby_leauge")]
_COPULA_CASTLEFORD_REAL_SENT = (
    "Castleford FC, commonly known as Castleford F , is a professional football club based in "
    "Castleford, West Yorkshire, England"
)
_COPULA_CASES_BY_NAME = {
    "castleford_real_traffic_appositive": {"facts": _COPULA_CASTLEFORD_FACTS},
    "castleford_no_appositive": {"facts": _COPULA_CASTLEFORD_FACTS},
    "castleford_soccer_synonym": {"facts": _COPULA_CASTLEFORD_FACTS},
    "chicago_bulls_baseball": {"facts": [("chicago_bulls", "sport", "basketball")]},
    "underscore_slug_no_appositive": {"facts": [("leeds_rhinos", "sport", "rugby_leauge")]},
    "castleford_correct_sport": {"facts": _COPULA_CASTLEFORD_FACTS},
    "canada_elaborated_copula": {"facts": [("canada", "isa", "country"), ("canada", "capital", "ottawa"),
                                            ("canada", "continent", "north america"),
                                            ("canada", "borders", "united states")]},
    "eiffel_tower_landmark": {"facts": [("eiffel tower", "is", "famous landmark")]},
    "castleford_negated": {"facts": _COPULA_CASTLEFORD_FACTS},
    "castleford_present_participle": {"facts": _COPULA_CASTLEFORD_FACTS},
    "eiffel_tower_passive_by_agent": {"facts": [("gustave eiffel", "built", "eiffel tower")]},
    "offtopic_agent_untouched": {"facts": [("einstein", "developed", "relativity")]},
    "grounded_kept": {"facts": [("mercury", "orbits", "sun")]},
}


def main():
    t0 = time.time()
    _set_flag(False)
    _set_copula_flag(False)   # clean starting env -- neither flag ambient-set before this runner starts

    fab_rows = _run_battery(FABRICATION_CASES)
    true_rows = _run_battery(TRUE_CASES)

    original_mod = _load_original_module()
    byte_id_rows = _byte_identical_off_check(FABRICATION_CASES + TRUE_CASES, original_mod)

    regression = _copula_regression_check()

    n_fab = len(fab_rows)
    n_caught = sum(1 for r in fab_rows if r["caught_flag_on"])
    n_leaked_off = sum(1 for r in fab_rows if r["leaked_flag_off"])
    new_catch_rate = n_caught / n_fab if n_fab else float("nan")

    n_true = len(true_rows)
    n_false_positive = sum(1 for r in true_rows if r["flag_on_result"] != r["flag_off_result"])
    false_positive_rate = n_false_positive / n_true if n_true else float("nan")

    all_byte_identical_off = all(r["byte_identical"] for r in byte_id_rows)

    art = {
        "probe": "np_entailment_participial_pronoun_coverage_verify",
        "flag": FLAG,
        "fabrication_battery": fab_rows,
        "true_battery": true_rows,
        "byte_identical_off_check": byte_id_rows,
        "copula_coverage_regression_check": regression,
        "n_fabrication_cases": n_fab, "n_caught_flag_on": n_caught, "n_leaked_flag_off": n_leaked_off,
        "new_catch_rate": new_catch_rate,
        "n_true_cases": n_true, "n_false_positive": n_false_positive,
        "false_positive_rate": false_positive_rate,
        "all_byte_identical_off": all_byte_identical_off,
    }

    v = Verdict("BRAIN_OPEN_ENDED_NP_ENTAILMENT_PARTICIPIAL_PRONOUN_COVERAGE widens the gate to catch "
                "participial-relation-conflict and pronoun-referent category-conflict fabrications "
                "without regressing true content (participial/pronoun OR the parent copula-coverage "
                "widening), and is byte-identical to the pre-widening module when the flag is off")
    v.require("every fabrication case leaks with the flag OFF (the pre-existing gap, reproduced)",
              n_leaked_off == n_fab, expect=True)
    v.require("every fabrication case is caught with the flag ON (new_catch_rate == 1.0)",
              new_catch_rate, expect=lambda x: x == 1.0)
    v.require("zero false positives on the true battery (flag ON == flag OFF on every case)",
              false_positive_rate, expect=lambda x: x == 0.0)
    v.require("flag OFF is byte-identical to the actual pre-widening (git HEAD) module on every case",
              all_byte_identical_off, expect=True)
    v.require("no regression to the copula-coverage widening's own saved battery",
              regression.get("no_regression"), expect=True)

    go = (n_leaked_off == n_fab and new_catch_rate == 1.0 and false_positive_rate == 0.0
          and all_byte_identical_off and regression.get("no_regression") is True)
    decided = v.decide(go=go)
    art["verdict"] = decided
    art["preconditions"] = decided.get("preconditions", [])
    art["GO"] = bool(go)
    art["elapsed_seconds"] = round(time.time() - t0, 1)

    print("=== FABRICATION battery (must catch, flag ON) ===")
    for r in fab_rows:
        flag = "OK" if r["caught_flag_on"] and r["leaked_flag_off"] else "MISS"
        print(f"  [{flag}] {r['name']}: flag_off={'LEAKED' if r['leaked_flag_off'] else 'caught?!'} "
              f"flag_on={'CAUGHT' if r['caught_flag_on'] else 'LEAKED'}")
    print("\n=== TRUE battery (must be unchanged) ===")
    for r in true_rows:
        flag = "OK" if r["flag_on_result"] == r["flag_off_result"] else "FALSE_POSITIVE"
        print(f"  [{flag}] {r['name']}")
    print("\n=== BYTE-IDENTICAL-OFF (vs actual git-HEAD original module) ===")
    for r in byte_id_rows:
        print(f"  [{'OK' if r['byte_identical'] else 'DIFF'}] {r['name']}")
    print("\n=== COPULA-COVERAGE REGRESSION CHECK (shared _CATEGORY_WORDS edit) ===")
    print(json.dumps(regression, indent=1))

    print(f"\nnew_catch_rate={new_catch_rate:.3f}  false_positive_rate={false_positive_rate:.3f}  "
          f"all_byte_identical_off={all_byte_identical_off}  "
          f"copula_no_regression={regression.get('no_regression')}")
    print(json.dumps({k: art[k] for k in (
        "n_fabrication_cases", "n_caught_flag_on", "n_leaked_flag_off", "new_catch_rate",
        "n_true_cases", "n_false_positive", "false_positive_rate", "all_byte_identical_off", "GO")}, indent=1))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(art, indent=1))
    print(f"\nwrote {OUT} -> {decided['status']}")
    return decided["status"]


if __name__ == "__main__":
    main()
