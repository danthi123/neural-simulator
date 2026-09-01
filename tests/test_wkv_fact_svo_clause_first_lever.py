"""Cheap regression pin for `_wkv_fact_svo_clause_first_lever.py`'s independent structural parser and the
fact->slot mapping convention -- the parts of the board #112 rung-2 first-lever investigation that do NOT need
a `SimulationBridge` build, so this stays fast (no spiking, no checkpoint, no store).

The full 6-seed spiking measurement (genuinely-spiking + well-formed + faithful + permuted-control +
attribution) lives in `research/runners/_wkv_fact_svo_clause_first_lever.py` /
`research/findings/raw/_wkv_fact_svo_clause_first_lever.json` -- this file pins the parser's OWN correctness
(it must accept a genuinely well-formed clause and reject malformed ones) so a future edit cannot silently
loosen it.
"""
from __future__ import annotations

from research.runners._wkv_fact_svo_clause_first_lever import parse_plain_transitive
from research.runners._emerge57_ra_refinetune_emerge_frames_derisk import emerge_v3


def test_well_formed_clause_parses_and_recovers_svo():
    surface = "the asimov_isaac employers the university_of_boston"
    pr = parse_plain_transitive(surface)
    assert pr["well_formed"] is True
    assert pr["svo"] == ("asimov_isaac", "employers", "university_of_boston")


def test_wrong_token_count_rejected():
    assert parse_plain_transitive("the asimov_isaac employers university_of_boston")["well_formed"] is False
    assert parse_plain_transitive("the a b the c the d")["well_formed"] is False


def test_wrong_determiner_position_rejected():
    assert parse_plain_transitive("a asimov_isaac employers the university_of_boston")["well_formed"] is False
    assert parse_plain_transitive("the asimov_isaac employers a university_of_boston")["well_formed"] is False


def test_emerge_v3_is_deterministic_and_vocabulary_agnostic():
    """The morphology function used to build the VERB slot must accept ARBITRARY relation-label strings (not
    just a closed verb lexicon) and be a pure deterministic function of its input -- both load-bearing for the
    lever's vocabulary-agnostic claim."""
    for action in ("sport", "employer", "follows", "contains_administrative_territorial_enti", "country"):
        v1 = emerge_v3(action)
        v2 = emerge_v3(action)
        assert v1 == v2
        assert isinstance(v1, str) and len(v1) > 0
