"""CI guard for Tier 0.3 (wh-questions as a filler-gap dependency).

Guards the production research/runners/wh_question_parser.py: the fronted wh-word = the FILLER, the verb's
Tier-0.1 frame = which role is the GAP, the wh-word selects which role to query. Covers the full filler-gap path
on an ArgStructureComposer (typed roles), the graceful fallback on a plain RFPhasorComposer (agent/action/patient
only), the no-confab moat, and the LOAD-BEARING permuted-mapping anti-cheat. All CPU/numpy. See
research/findings/2026-06-27-tier0.3-wh-questions-GO.md.
"""
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.argstructure_composer import ArgStructureComposer, reparse_to_fact  # noqa: E402
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402
from research.runners.wh_question_parser import (  # noqa: E402
    parse_wh_question, answer_wh, bare_answer, is_wh_question, WH_ROLE_CANDIDATES)


# A deranged wh->role table for the permuted-mapping anti-cheat (where->patient, what->GOAL, who->THEME, ...).
PERMUTED = {
    "who": ["THEME", "patient"], "who_to": ["THEME", "patient"], "what": ["GOAL", "LOCATION"],
    "where": ["patient", "THEME"], "when": ["agent"], "whom": ["agent"], "with": ["agent"],
}


@pytest.fixture(scope="module")
def comp():
    vocab = ["boy", "girl", "dog", "cat", "go", "give", "put", "chase",
             "park", "ball", "bone", "table", "river"]
    c = ArgStructureComposer(seed=42, D=64, vocab=vocab)
    c.store_fact({"agent": "boy", "action": "go", "GOAL": "park"})
    c.store_fact({"agent": "girl", "action": "give", "THEME": "ball", "RECIPIENT": "dog"})
    c.store_fact({"agent": "dog", "action": "put", "THEME": "bone", "LOCATION": "table"})
    c.store_fact({"agent": "cat", "action": "chase", "patient": "river"})
    return c


# --- parsing: the wh-word -> the frame-gapped role -------------------------------------------------------------
def test_parse_where_maps_to_frame_role():
    """where -> GOAL for `go` (frame licenses GOAL), LOCATION for `put` (frame licenses LOCATION)."""
    assert parse_wh_question("where does the boy go?")["role"] == "GOAL"
    assert parse_wh_question("where does the dog put?")["role"] == "LOCATION"


def test_parse_who_subject_vs_recipient():
    """who -> agent (bare subject question); who+trailing-'to' -> RECIPIENT (the to-PP gap)."""
    assert parse_wh_question("who chase river?")["role"] == "agent"
    assert parse_wh_question("who does the girl give to?")["role"] == "RECIPIENT"


def test_parse_unlicensed_role_abstains():
    """A wh whose role the verb frame doesn't license -> __UNLICENSED__ (e.g. when->TIME, but `go` has no TIME)."""
    assert parse_wh_question("when does the boy go?")["role"] == "__UNLICENSED__"
    assert parse_wh_question("where does the cat chase?")["role"] == "__UNLICENSED__"  # chase: no GOAL/LOCATION


def test_non_wh_returns_none():
    assert parse_wh_question("the dog chased the cat") is None
    assert parse_wh_question("hi there") is None
    assert is_wh_question("where does the boy go?")
    assert not is_wh_question("tell me about dogs")


# --- the headline answer + render ------------------------------------------------------------------------------
def test_where_does_the_boy_go(comp):
    """The headline: 'where does the boy go?' -> GOAL -> 'park'; bare answer 'to the park'."""
    filler, role, parse = answer_wh(comp, "where does the boy go?")
    assert (filler, role) == ("park", "GOAL")
    assert bare_answer(role, filler) == "to the park"


def test_full_wh_coverage(comp):
    """Every wh-form answers correctly over the stored arg-structure facts."""
    cases = [
        ("where does the boy go?", "GOAL", "park"),
        ("what does the girl give?", "THEME", "ball"),
        ("who does the girl give to?", "RECIPIENT", "dog"),
        ("where does the dog put?", "LOCATION", "table"),
        ("what does the dog put?", "THEME", "bone"),
        ("what does the cat chase?", "patient", "river"),
        ("who chase river?", "agent", "cat"),
    ]
    for q, exp_role, exp_filler in cases:
        filler, role, _ = answer_wh(comp, q)
        assert (role, filler) == (exp_role, exp_filler), q


# --- the no-confab moat ----------------------------------------------------------------------------------------
def test_moat_unanswerable_abstains(comp):
    """An unanswerable / unstored / frame-unlicensed wh -> None (0 false-accepts)."""
    assert answer_wh(comp, "where does the boy give?")[0] is None     # boy+give has no GOAL stored
    assert answer_wh(comp, "where does the cat go?")[0] is None        # cat+go not stored
    assert answer_wh(comp, "what does the boy give?")[0] is None       # boy+give not stored
    assert answer_wh(comp, "who does the dog give to?")[0] is None     # dog+give not stored
    assert answer_wh(comp, "when does the boy go?")[0] is None         # go's frame: no TIME slot
    assert answer_wh(comp, "where does the cat chase?")[0] is None     # chase: no GOAL/LOCATION


def test_verify_reparse_of_rendered_answer(comp):
    """The rendered full answer re-parses to the stored typed fact (content-mismatch would reject)."""
    filler, role, parse = answer_wh(comp, "where does the boy go?")
    fact = dict(parse["cue"]); fact[role] = filler
    rendered = comp.render(fact)
    assert rendered == "the boy goes to the park"
    assert reparse_to_fact(rendered, fact)


# --- the LOAD-BEARING permuted-mapping anti-cheat --------------------------------------------------------------
def test_permuted_mapping_does_not_reproduce_true_answers(comp):
    """A WRONG wh->role table must NOT reproduce the true answers -- proving the mapping carries the meaning."""
    cases = ["where does the boy go?", "what does the girl give?", "who does the girl give to?",
             "where does the dog put?", "what does the dog put?", "what does the cat chase?", "who chase river?"]
    for q in cases:
        true_filler, _, _ = answer_wh(comp, q)
        wrong_filler, _, _ = answer_wh(comp, q, role_map=PERMUTED)
        # the wrong mapping must not reproduce a correct (non-None) answer.
        assert not (true_filler is not None and wrong_filler == true_filler), q


# --- the graceful fallback on a PLAIN RFPhasorComposer (the deployed first-chat console; no typed roles) -------
@pytest.fixture(scope="module")
def plain_comp():
    vocab = ["boy", "girl", "dog", "cat", "go", "give", "chase", "park", "ball", "bone", "river"]
    c = RFPhasorComposer(seed=42, D=64, vocab=vocab)
    c.store("dog", "chase", "cat", polarity="AFFIRM")
    c.store("boy", "go", "park", polarity="AFFIRM")     # 'park' lands in the plain patient slot (no GOAL role)
    return c


def test_plain_composer_fallback_what_who(plain_comp):
    """On a plain RFPhasorComposer the wh route falls back to query_patient/query_agent (who/what still answer)."""
    # 'what does the dog chase?' -> patient -> cat
    assert answer_wh(plain_comp, "what does the dog chase?")[0] == "cat"
    # 'who chase cat?' -> agent (subject) -> dog
    assert answer_wh(plain_comp, "who chase cat?")[0] == "dog"


def test_plain_composer_typed_oblique_abstains(plain_comp):
    """A typed oblique (where->GOAL) on a plain composer with no GOAL role -> abstain (no fabrication)."""
    # 'where does the boy go?' -> GOAL, but a plain composer has no GOAL role -> None (graceful).
    assert answer_wh(plain_comp, "where does the boy go?")[0] is None


def test_plain_composer_moat(plain_comp):
    """The moat holds on the fallback path too: an unstored cue abstains."""
    assert answer_wh(plain_comp, "what does the cat chase?")[0] is None     # cat+chase not stored
    assert answer_wh(plain_comp, "who chase river?")[0] is None             # nobody chases river
