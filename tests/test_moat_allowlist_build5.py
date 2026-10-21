"""Pins for build 5 of the strict allowlist claim checker (`BRAIN_MOAT_ALLOWLIST`, default OFF).

Round 6 (research/findings/raw/_moat_allowlist_hunt4/r6/) kept 5 of 17 adjudicated-U cases at all seven seeds on build 4, in two classes:
  - a clause word fused into a spaced name: "Solomon is located in time zone Kosrae time." spells solomon_is (Solomon Is., the islands)
    plus the telegraphic relation label, and is read "Solomon is located in" (R6-CF01, R6-CF02);
  - a coordinated list naming one thing twice: "Take That has part Robbie Williams and a Robbie Williams." (R6-S201, R6-S203, R6-S210).
Build 5 rejects a SUBJ/OBJ span whose displayed text, split on spaces, holds a finite auxiliary or a NEG_MARKERS word
(`_structure_word_violation`), and a parse that lists one proposition twice (`_repeated_proposition`).

These tests pin: the five leaks rejected with the build-5 reason (pure function, the production claim gate and the composer entry at
all seven parser seeds); the exact word set; every word of it rejected inside a spaced name and kept in the slug written verbatim; the
stub mouth's own rendering still kept; the non-finite words that real names hold ("am", "do", "be") kept; a list of distinct gathered
objects kept and every repeated form rejected.

Run: CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m pytest tests/test_moat_allowlist_build5.py -q
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest

from research.runners import _moat_allowlist as A
from research.runners._moat_claim_entailment_derisk import NEG_MARKERS
from research.runners.brain_conversational_agent import BridgeParser

pytestmark = pytest.mark.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
R6 = ROOT / "research" / "findings" / "raw" / "_moat_allowlist_hunt4" / "r6"
SEEDS = (42, 43, 44, 100, 101, 102, 1)
FLAG = "BRAIN_MOAT_ALLOWLIST"
R6_LEAKS = ("R6-CF01", "R6-CF02", "R6-S201", "R6-S203", "R6-S210")
CASES = {c["id"]: c for c in json.loads((R6 / "cases.json").read_text(encoding="utf-8"))}
CLAUSE = "holds the clause word"
REPEAT = "a coordinated list asserts distinct things"


@pytest.fixture(autouse=True)
def _clean_flags(monkeypatch):
    for k in (FLAG, "BRAIN_MOAT_SLUG_NORM", "BRAIN_CLAIM_MOAT"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def seed_chats():
    from research.runners.brain_chat_tui import ChatBrain
    out = {}
    for sd in SEEDS:
        chat = ChatBrain.__new__(ChatBrain)
        chat.inner = BridgeParser(seed=sd)
        out[sd] = chat
    return out


def _gate_all_seeds(seed_chats, surface, gathered):
    os.environ[FLAG] = "1"
    try:
        out = []
        for sd in SEEDS:
            seed_chats[sd]._claim_verifier_cache = {}
            acc, res = seed_chats[sd]._verify_claim_set(surface, gathered)
            out.append((acc, res["reject_reason"] if isinstance(res, dict) else res))
        return out
    finally:
        os.environ.pop(FLAG, None)


def _composer_entry(chat, surface, gathered):
    from research.runners.rich_answer_composer import RichAnswerComposer
    comp = RichAnswerComposer.__new__(RichAnswerComposer)
    comp.chat = chat
    os.environ[FLAG] = "1"
    try:
        chat._claim_verifier_cache = {}
        return comp._verify_rendered(surface, None, tuple(gathered[0]), gathered)
    finally:
        os.environ.pop(FLAG, None)


def _keep(sentence, gathered):
    return A.allowlist_check(sentence, gathered)


# ============================================================================================================ round-6 leaks
def test_the_leak_list_is_build4s_kept_u_set():
    doc = json.loads((R6 / "results_build4.json").read_text(encoding="utf-8"))
    ent = doc["summary"]["allowlist"]["entries"]
    kept = set(ent["gate"]["U"]["kept_ids"]) | set(ent["composer"]["U"]["kept_ids"])
    assert sorted(kept) == sorted(R6_LEAKS)


@pytest.mark.parametrize("cid", R6_LEAKS)
def test_every_round6_leak_is_rejected_at_all_seven_seeds_through_both_entries(cid, seed_chats):
    c = CASES[cid]
    r = _keep(c["sentence"], c["gathered"])
    assert not r["keep"], (cid, r["reason"])
    assert (CLAUSE in r["reason"]) if cid.startswith("R6-CF") else (REPEAT in r["reason"]), (cid, r["reason"])
    got = _gate_all_seeds(seed_chats, c["sentence"], c["gathered"])
    assert all(acc is False and r["reason"] in why for acc, why in got), (cid, got)
    assert [_composer_entry(seed_chats[sd], c["sentence"], c["gathered"]) for sd in SEEDS] == [False] * len(SEEDS)


# ================================================================================ no clause word inside a spaced name
def test_the_word_set_is_the_finite_auxiliaries_and_the_negation_markers():
    assert A.STRUCTURE_WORDS == (frozenset(("is", "are", "was", "were", "be", "has", "have", "had", "does", "did"))
                                 | frozenset(NEG_MARKERS))
    assert A.CONTRACTION_RESIDUE == frozenset(("s", "t"))
    for w in ("am", "been", "being", "do", "will", "can", "may", "of", "the", "and", "a"):
        assert w not in A.STRUCTURE_WORDS, w


@pytest.mark.parametrize("w", sorted(A.STRUCTURE_WORDS))
def test_every_clause_word_inside_a_spaced_name_rejects_and_the_slug_verbatim_is_kept(w):
    g = [["east_%s_west" % w, "country", "freedonia"]]
    r = _keep("East %s west country Freedonia." % w, g)
    assert not r["keep"] and CLAUSE in r["reason"] and ("%r" % w) in r["reason"], r["reason"]
    g2 = [["freedonia", "country", "east_%s_west" % w]]                       # the object span too
    r2 = _keep("Freedonia country East %s west." % w, g2)
    assert not r2["keep"] and CLAUSE in r2["reason"], r2["reason"]
    assert _keep("east_%s_west country freedonia." % w, g)["keep"]            # the slug as written: one word


@pytest.mark.parametrize("space", [" ", " ", " ", " "])
def test_a_display_identical_space_still_splits_the_word(space):
    g = [["solomon_is", "located_in_time_zone", "kosrae_time"]]
    r = _keep("Solomon%sis located in time zone Kosrae time." % space, g)
    assert not r["keep"] and CLAUSE in r["reason"], r["reason"]


@pytest.mark.parametrize("svo", [("solomon_is", "located_in_time_zone", "kosrae_time"), ("religion_is_dead", "genre", "pop_rock"),
                                 ("x_not_y", "country", "freedonia")])
def test_the_stub_mouths_own_rendering_is_kept(svo):
    from research.runners._grounded_lang_p3_derisk import TemplateStubFaculty
    surface, asserted = TemplateStubFaculty().render_svo(*svo)
    assert list(asserted) == list(svo)
    assert _keep(surface, [list(svo)])["keep"], surface


@pytest.mark.parametrize("sentence,svo", [
    ("Offenbach am Main country Germany.", ["offenbach_am_main", "country", "germany"]),
    ("Rio Grande do Sul country Brazil.", ["rio_grande_do_sul", "country", "brazil"]),
    ("Divine being instance of deity.", ["divine_being", "instance_of", "deity"]),
])
def test_non_finite_words_inside_real_names_are_kept(sentence, svo):
    r = _keep(sentence, [svo])
    assert r["keep"], r["reason"]


def test_the_words_outside_a_name_are_unchanged():
    g = [["solomon_is", "located_in_time_zone", "kosrae_time"]]
    assert _keep("solomon_is located_in_time_zone kosrae_time.", g)["keep"]
    g3 = [["paris", "instance_of", "city"]]
    assert _keep("Paris is an instance of city.", g3)["keep"]               # shape 3's copula is not inside a span


# ============================================================================================== a list names distinct things
TT = [["take_that", "has_part", "robbie_williams"], ["take_that", "has_part", "gary_barlow"], ["take_that", "has_part", "mark_owen"]]


@pytest.mark.parametrize("sentence", [
    "Take That has part Robbie Williams and Robbie Williams.",
    "Take That has part Robbie Williams and a Robbie Williams.",
    "Take That has part a Robbie Williams and the Robbie Williams.",
    "Take That has part Robbie Williams, Gary Barlow and Robbie Williams.",
    "Take That has part Robbie Williams, Gary Barlow, and robbie_williams.",
    "Take That has part robbie_williams and Robbie Williams.",
])
def test_every_repeated_list_rejects(sentence):
    r = _keep(sentence, TT)
    assert not r["keep"] and REPEAT in r["reason"], r["reason"]


@pytest.mark.parametrize("sentence", [
    "Take That has part Robbie Williams and Gary Barlow.",
    "Take That has part Robbie Williams, Gary Barlow and Mark Owen.",
    "Take That has part Robbie Williams, Gary Barlow, and Mark Owen.",
])
def test_a_list_of_distinct_gathered_objects_is_kept(sentence):
    r = _keep(sentence, TT)
    assert r["keep"], r["reason"]


# ============================================================================================ review fixes (2026-10-20, before measurement)
HP = [["nine_inch_nails", "has_part", "trent_reznor"]]


@pytest.mark.parametrize("sentence,gathered", [
    ("Nine Inch Nails has_parted Trent Reznor.", HP),                           # reads "has parted" (separated from): the U leak
    ("Nine Inch Nails has_parting Trent Reznor.", HP),
    ("Nine Inch Nails has_partes Trent Reznor.", HP),
    ("Master P record_labeled No_Limit_label.", [["master_p", "record_label", "no_limit_label"]]),
    ("X member_of_political_parties Y.", [["x", "member_of_political_party", "y"]]),
    ("X followed_bies Y.", [["x", "followed_by", "y"]]),
])
def test_a_multi_word_relation_keeps_no_inflection_but_the_stubs(sentence, gathered, seed_chats):
    r = _keep(sentence, gathered)
    assert not r["keep"], r["reason"]
    got = _gate_all_seeds(seed_chats, sentence, gathered)
    assert all(acc is False for acc, _why in got), got


@pytest.mark.parametrize("sentence", ["Nine Inch Nails has_parts Trent Reznor.", "The nine_inch_nails has_parts trent_reznor.",
                                      "Nine Inch Nails has part Trent Reznor.", "nine_inch_nails has_part trent_reznor."])
def test_the_stub_token_the_slug_and_the_label_are_kept(sentence):
    assert _keep(sentence, HP)["keep"], sentence


def test_the_kept_token_is_the_stubs_own_inflection_for_every_multi_word_relation():
    from research.runners._grounded_lang_p3_derisk import _inflect
    from research.runners._moat_allowlist_relations import RELATIONS, reading_dropped, stub_third_person
    multi = [r for r in RELATIONS if "_" in r]
    assert len(multi) >= 40
    for rel in multi:
        assert stub_third_person(rel) == _inflect(rel), rel
        kept = [t for t in RELATIONS[rel].readings if not reading_dropped(rel, t)]
        assert sorted(kept) == sorted({rel, _inflect(rel)} & set(RELATIONS[rel].readings)), (rel, kept)


TT1 = [["take_that", "has_part", "robbie_williams"]]


@pytest.mark.parametrize("surface", [
    "Take That has part Robbie Williams. Take That has part a Robbie Williams.",
    "Take That has part Robbie Williams.\nTake That has part the Robbie Williams.",
    "Take That has part Robbie Williams. Take That has part Robbie Williams.",
])
def test_a_reply_stating_one_fact_twice_rejects(surface, seed_chats):
    v = A.allowlist_verify(surface, TT1)
    assert not v["accepted"] and "both state (take_that, has_part, robbie_williams)" in v["reject_reason"], v["reject_reason"]
    got = _gate_all_seeds(seed_chats, surface, TT1)
    assert all(acc is False for acc, _why in got), got


def test_two_different_facts_in_two_sentences_are_kept():
    g = TT1 + [["robbie_williams", "member_of", "take_that"]]
    assert A.allowlist_verify("Take That has part Robbie Williams. Robbie Williams is a member of Take That.", g)["accepted"]


@pytest.mark.parametrize("gap,ok", [("\r\n", True), ("\n", True), (" ", True), ("\r", False), ("\n\r", False), ("\r\r\n", False)])
def test_a_carriage_return_between_sentences_only_before_a_line_feed(gap, ok):
    g = TT1 + [["ringo_star", "part_of", "johnny_and_the_moondogs"]]
    v = A.allowlist_verify("Ringo Star is part of Johnny and the Moondogs.%sTake That has part Robbie Williams." % gap, g)
    assert v["accepted"] is ok, (repr(gap), v["reject_reason"])


@pytest.mark.parametrize("sentence", ["Marshall_Is located in time zone Fiji standard time.",
                                      "Marshall_is located in time zone Fiji standard time.",
                                      "MARSHALL_IS located in time zone Fiji standard time.",
                                      "Fiji standard time is the time zone of Solomon_Is."])
def test_a_prose_cased_underscore_slug_splits_into_words(sentence):
    g = [["marshall_is", "located_in_time_zone", "fiji_standard_time"]]
    r = _keep(sentence, g)
    assert not r["keep"], r["reason"]
    assert _keep("marshall_is located in time zone Fiji standard time.", g)["keep"]          # lower-case verbatim: one identifier
    assert _keep("The marshall_is located_in_time_zones fiji_standard_time.", g)["keep"]     # the stub's own sentence
    for prose, svo in (("No_Saints member of National Football League.", ["no_saints", "member_of", "national_football_league"]),
                       ("Friendship_Is_Magic genre fantasy fiction.", ["friendship_is_magic", "genre", "fantasy_fiction"]),
                       ("Saint_Alban_s located in time zone UTC 00 00.", ["saint_alban_s", "located_in_time_zone", "utc_00_00"])):
        assert not _keep(prose, [svo])["keep"], prose


def test_a_capitalised_underscore_slug_without_a_clause_word_is_kept():
    """Real Qwen F sentences of the bar-4 must-keep set capitalise the sentence's first letter on a verbatim slug."""
    assert _keep("Luk_cs_p_l is an instance of human_specie.", [["luk_cs_p_l", "instance_of", "human_specie"]])["keep"]
    assert _keep("The S_Te_France instance_of communes_in_france.", [["s_te_france", "instance_of", "communes_in_france"]])["keep"]


@pytest.mark.parametrize("sentence,svo", [
    ("Thun be located in the administrative territoria Canton de Berne.", ["thun_be", "located_in_the_administrative_territoria", "canton_de_berne"]),
    ("Biel be country Switzerland.", ["biel_be", "country", "switzerland"]),
    ("Saint Alban s located in time zone UTC 00 00.", ["saint_alban_s", "located_in_time_zone", "utc_00_00"]),
    ("You don t arrest Voltaire genre essay.", ["you_don_t_arrest_voltaire", "genre", "essay"]),
])
def test_be_and_the_residue_of_a_dropped_apostrophe_reject(sentence, svo):
    r = _keep(sentence, [svo])
    assert not r["keep"] and ("clause word" in r["reason"] or "dropped apostrophe" in r["reason"]), r["reason"]


def test_a_capital_initial_and_a_fold_residue_are_not_contraction_residue():
    assert _keep("Harry S Truman country United States.", [["harry_s_truman", "country", "united_states"]])["keep"]
    assert _keep("Bill country u s of a.", [["bill", "country", "u_s_of_a"]])["keep"]                        # a folded period
    assert _keep("Hu Berlin country humboldt universit t zu berlin.", [["hu_berlin", "country", "humboldt_universit_t_zu_berlin"]])["keep"]
    assert not _keep("Mount St Mary s country u s of a.", [["mount_st_mary_s", "country", "u_s_of_a"]])["keep"]
    assert _keep("saint_alban_s located_in_time_zone utc_00_00.", [["saint_alban_s", "located_in_time_zone", "utc_00_00"]])["keep"]


def test_a_signed_numbers_de_slugged_form_is_not_accepted():
    g = [["puebla_city", "located_in_time_zone", "utc_06"]]
    rows = A.names_table().entities["utc_06"]
    assert rows and all(A._SIGNED_NUMBER.search(r) for r in rows), rows
    r = _keep("Puebla City located in time zone UTC 06.", g)
    assert not r["keep"] and "utc_06" in r["reason"], r["reason"]
    assert "utc-06" in rows                                                              # the ASCII store spelling is kept
    assert _keep("Puebla City located in time zone UTC-06.", g)["keep"]
    assert _keep("puebla_city located_in_time_zone utc_06.", g)["keep"]                 # the slug verbatim
    assert not A._SIGNED_NUMBER.search("1971-72 cypriot cup") and not A._SIGNED_NUMBER.search("0-6-0dh")   # ranges and joiners
