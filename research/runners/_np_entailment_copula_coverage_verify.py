"""PARSING-LEVEL VERIFY for the copula-coverage widening in `webapp.np_entailment_moat_gate`
(`BRAIN_OPEN_ENDED_NP_ENTAILMENT_COPULA_COVERAGE`, 2026-09-02 follow-on to the 2026-09-01 moat-
safety soak, `research/findings/2026-09-01-open-ended-bundle-moat-safety-soak-fabrication-delta.md`).

THE GAP THIS CHECKS. That soak measured `NP_ENTAILMENT` changing ZERO of 12 real known-topic
Qwen replies because scope (d) excludes copula ("is a ...") outright, and real Qwen prose is
dominated by it -- the concrete miss: `castleford_f_c` called a "professional football club" when
the store's only sport fact is `rugby_leauge`, surviving the gate untouched. This runner is
PARSING-LEVEL ONLY (`_get_spiking_pair()` builds the same tiny (126 + 82)-neuron BridgeParser +
NPHeadBinder pair the live gate itself builds -- RAM-light, no 15k-LTM brain, no cupy/GPU needed;
`SIM_BACKEND=numpy` set below): it calls `webapp.np_entailment_moat_gate.gate_sentence` directly
with hand-built (sentence, topic, facts) inputs, exactly the shape `post_filter` calls it with.

WHAT THIS MEASURES:
  (1) NEW-CATCH RATE on a fabrication battery -- sentences asserting a category (sport) that
      CONFLICTS with a store fact for the SAME topic, including the exact real-traffic castleford
      shape (leading appositive comma) -- flag ON must catch (gate_sentence -> None), flag OFF
      must leak (gate_sentence -> sent unchanged), the same load-bearing signature the parent
      gate's own wiring verify uses.
  (2) FALSE-POSITIVE RATE on a true-copula battery -- correct category claims, category-free
      copula, negated / present-participle / passive predicates, and the parent gate's own saved
      SAFETY_CASES (canada, off-topic-agent, grounded-non-copula) -- flag ON must NOT change any
      of these (byte-identical to flag OFF); a mismatch here is a genuine false-reject regression.
  (3) BYTE-IDENTICAL-OFF, measured against the ACTUAL pre-widening file content (`git show
      HEAD:webapp/np_entailment_moat_gate.py`, loaded as an isolated module), not assumed from
      reading the diff: with the flag OFF (the default), every case in both batteries above must
      produce the IDENTICAL `gate_sentence` output the original, unmodified file produces.

Run: python -m research.runners._np_entailment_copula_coverage_verify
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

OUT = _REPO / "research" / "findings" / "raw" / "_np_entailment_copula_coverage_verify.json"
FLAG = GATE._FLAG_COPULA_COVERAGE
GATE_MODULE_PATH = _REPO / "webapp" / "np_entailment_moat_gate.py"


def _set_flag(on: bool):
    if on:
        os.environ[FLAG] = "1"
    else:
        os.environ.pop(FLAG, None)


# =================================================================================================
# (1) FABRICATION battery -- category-conflict copula claims that must be CAUGHT (flag ON -> None).
# =================================================================================================

CASTLEFORD_FACTS = [("castleford_f_c", "country", "united_kingom"), ("castleford_f_c", "sport", "rugby_leauge")]
# the exact real-traffic sentence shape from the soak artifact (leading appositive comma between
# subject and copula -- research/findings/raw/_open_ended_bundle_moat_soak_full.json, arm A, known,
# topic=castleford_f_c), lightly truncated to the one clause under test.
CASTLEFORD_REAL_SENT = (
    "Castleford FC, commonly known as Castleford F , is a professional football club based in "
    "Castleford, West Yorkshire, England"
)

FABRICATION_CASES = [
    {"name": "castleford_real_traffic_appositive", "topic": "castleford_f_c", "facts": CASTLEFORD_FACTS,
     "sent": CASTLEFORD_REAL_SENT,
     "note": "exact real-traffic shape (soak artifact): appositive between subject and copula, "
             "underscored-slug topic vs human-readable subject text"},
    {"name": "castleford_no_appositive", "topic": "castleford_f_c", "facts": CASTLEFORD_FACTS,
     "sent": "Castleford FC is a professional football club.",
     "note": "same conflict, no appositive -- isolates the category-conflict check from the "
             "appositive-comma fix"},
    {"name": "castleford_soccer_synonym", "topic": "castleford_f_c", "facts": CASTLEFORD_FACTS,
     "sent": "Castleford FC is a well-known soccer team.",
     "note": "synonym coverage: 'soccer' canonicalizes to the same family/word as 'football'"},
    {"name": "chicago_bulls_baseball", "topic": "chicago_bulls", "facts": [("chicago_bulls", "sport", "basketball")],
     "sent": "The Chicago Bulls is a professional baseball team.",
     "note": "a second sport pair (basketball vs baseball), no underscore-slug quirk, no appositive"},
    {"name": "underscore_slug_no_appositive", "topic": "leeds_rhinos",
     "facts": [("leeds_rhinos", "sport", "rugby_leauge")],
     "sent": "Leeds Rhinos is a professional cricket club.",
     "note": "another underscore-topic / space-subject pair, different sport conflict, no appositive"},
]


# =================================================================================================
# (2) TRUE-COPULA battery -- must NOT be caught (flag ON output == flag OFF output) on either arm.
# =================================================================================================

TRUE_CASES = [
    {"name": "castleford_correct_sport", "topic": "castleford_f_c", "facts": CASTLEFORD_FACTS,
     "sent": "Castleford FC, commonly known as Castleford F , is a professional rugby club based in "
             "Castleford, West Yorkshire, England",
     "note": "SAME sentence shape as the caught case, but the predicate's category MATCHES the "
             "store's sport fact -- must not be flagged (false-positive guard)"},
    {"name": "canada_elaborated_copula", "topic": "canada",
     "facts": [("canada", "isa", "country"), ("canada", "capital", "ottawa"),
               ("canada", "continent", "north america"), ("canada", "borders", "united states")],
     "sent": "Canada is a vast country located in North America.",
     "note": "the parent gate's own SAFETY_CASE (copula_untouched) -- no category word at all"},
    {"name": "eiffel_tower_landmark", "topic": "eiffel tower",
     "facts": [("eiffel tower", "is", "famous landmark")],
     "sent": "The Eiffel Tower is a famous landmark.",
     "note": "true copula, no category word -- must not be flagged"},
    {"name": "castleford_negated", "topic": "castleford_f_c", "facts": CASTLEFORD_FACTS,
     "sent": "Castleford FC is not a football club.",
     "note": "negated predicate -- widened path must back off (guarded), never assert the negation "
             "is itself the fabrication"},
    {"name": "castleford_present_participle", "topic": "castleford_f_c", "facts": CASTLEFORD_FACTS,
     "sent": "Castleford FC is playing football this weekend.",
     "note": "progressive aspect, not an identity predicate -- widened path must back off"},
    {"name": "eiffel_tower_passive_by_agent", "topic": "gustave eiffel",
     "facts": [("gustave eiffel", "built", "eiffel tower")],
     "sent": "The Eiffel Tower was built by Gustave Eiffel.",
     "note": "passive construction (already handled by segment_clause's own passive pass) -- "
             "widened path must back off, not misread 'built' as a copula predicate nominal"},
    {"name": "offtopic_agent_untouched", "topic": "einstein",
     "facts": [("einstein", "developed", "relativity")],
     "sent": "Newton discovered gravity.",
     "note": "parent gate's own SAFETY_CASE: subject != retrieved topic -- out of scope either way"},
    {"name": "grounded_kept", "topic": "mercury", "facts": [("mercury", "orbits", "sun")],
     "sent": "Mercury orbits the sun.",
     "note": "parent gate's own SAFETY_CASE: grounded non-copula claim -- unaffected by copula widening"},
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
    """Load the pre-widening `webapp/np_entailment_moat_gate.py` from git HEAD (NOT the working
    tree) as an isolated module, so the byte-identical-off check compares against the ACTUAL
    committed original, not a hand-reconstructed guess of what it used to do."""
    src = subprocess.check_output(["git", "show", "HEAD:webapp/np_entailment_moat_gate.py"],
                                   cwd=str(_REPO)).decode("utf-8")
    tmp_path = _REPO / "research" / "findings" / "raw" / "_np_entailment_moat_gate_ORIGINAL_scratch.py"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(src)
    spec = importlib.util.spec_from_file_location("_np_entailment_moat_gate_original", str(tmp_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tmp_path.unlink()   # scratch file only -- not a persisted artifact
    return mod


def _byte_identical_off_check(cases, original_mod):
    """For every case, flag OFF, confirm the CURRENT (widened) module's output equals the
    ORIGINAL (pre-widening) module's output on the identical input. This is the strongest form of
    'byte-identical when off': not a diff of the source, an equality of the actual return value
    against the actual prior implementation."""
    rows = []
    _set_flag(False)
    for c in cases:
        current = GATE.gate_sentence(c["sent"], c["topic"], c["facts"])
        original = original_mod.gate_sentence(c["sent"], c["topic"], c["facts"])
        rows.append({"name": c["name"], "current_flag_off": current, "original": original,
                     "byte_identical": (current == original)})
    return rows


def main():
    t0 = time.time()

    fab_rows = _run_battery(FABRICATION_CASES)
    true_rows = _run_battery(TRUE_CASES)

    original_mod = _load_original_module()
    byte_id_rows = _byte_identical_off_check(FABRICATION_CASES + TRUE_CASES, original_mod)

    n_fab = len(fab_rows)
    n_caught = sum(1 for r in fab_rows if r["caught_flag_on"])
    n_leaked_off = sum(1 for r in fab_rows if r["leaked_flag_off"])
    new_catch_rate = n_caught / n_fab if n_fab else float("nan")

    n_true = len(true_rows)
    n_false_positive = sum(1 for r in true_rows if r["flag_on_result"] != r["flag_off_result"])
    false_positive_rate = n_false_positive / n_true if n_true else float("nan")

    all_byte_identical_off = all(r["byte_identical"] for r in byte_id_rows)

    art = {
        "probe": "np_entailment_copula_coverage_verify",
        "flag": FLAG,
        "fabrication_battery": fab_rows,
        "true_copula_battery": true_rows,
        "byte_identical_off_check": byte_id_rows,
        "n_fabrication_cases": n_fab, "n_caught_flag_on": n_caught, "n_leaked_flag_off": n_leaked_off,
        "new_catch_rate": new_catch_rate,
        "n_true_cases": n_true, "n_false_positive": n_false_positive,
        "false_positive_rate": false_positive_rate,
        "all_byte_identical_off": all_byte_identical_off,
    }

    v = Verdict("BRAIN_OPEN_ENDED_NP_ENTAILMENT_COPULA_COVERAGE widens the gate to catch copula "
                "category-conflict fabrications without regressing true copula content, and is "
                "byte-identical to the pre-widening module when the flag is off")
    v.require("every fabrication case leaks with the flag OFF (the pre-existing gap, reproduced)",
              n_leaked_off == n_fab, expect=True)
    v.require("every fabrication case is caught with the flag ON (new_catch_rate == 1.0)",
              new_catch_rate, expect=lambda x: x == 1.0)
    v.require("zero false positives on the true-copula battery (flag ON == flag OFF on every case)",
              false_positive_rate, expect=lambda x: x == 0.0)
    v.require("flag OFF is byte-identical to the actual pre-widening (git HEAD) module on every case",
              all_byte_identical_off, expect=True)

    go = (n_leaked_off == n_fab and new_catch_rate == 1.0 and false_positive_rate == 0.0
          and all_byte_identical_off)
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
    print("\n=== TRUE-COPULA battery (must be unchanged) ===")
    for r in true_rows:
        flag = "OK" if r["flag_on_result"] == r["flag_off_result"] else "FALSE_POSITIVE"
        print(f"  [{flag}] {r['name']}")
    print("\n=== BYTE-IDENTICAL-OFF (vs actual git-HEAD original module) ===")
    for r in byte_id_rows:
        print(f"  [{'OK' if r['byte_identical'] else 'DIFF'}] {r['name']}")

    print(f"\nnew_catch_rate={new_catch_rate:.3f}  false_positive_rate={false_positive_rate:.3f}  "
          f"all_byte_identical_off={all_byte_identical_off}")
    print(json.dumps({k: art[k] for k in (
        "n_fabrication_cases", "n_caught_flag_on", "n_leaked_flag_off", "new_catch_rate",
        "n_true_cases", "n_false_positive", "false_positive_rate", "all_byte_identical_off", "GO")}, indent=1))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(art, indent=1))
    print(f"\nwrote {OUT} -> {decided['status']}")
    return decided["status"]


if __name__ == "__main__":
    main()
