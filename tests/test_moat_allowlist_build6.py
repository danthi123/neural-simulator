"""Pins for build 6 of the strict allowlist claim checker (`BRAIN_MOAT_ALLOWLIST`, default OFF; brief
/home/dant123/Projects/sim-rev/preflip/BUILD6_BRIEF.md, Claude rulings 2026-10-21; spec docs/drafts/moat-allowlist/SPEC.md).

  B. SHAPE 4, one reviewed wording per no-reading relation (research/runners/_moat_allowlist_relations.py PHRASE_READINGS): the nine
     ROLE-REVERSING / MEANING-DRIFT relations keep no verb reading, but "[Det] A PHRASE [Det] B" voices each in its row's orientation; every
     row's role-reversed sentence rejects; the stub writes the phrase under the flag only.
  A. THE FLAG OWNS THE WHOLE MOUTH PATH (see the section header below).
  C. THE TWO LEAK CLASSES (see the section header below).

The matcher is pure: its verdict cannot depend on the parser seed. Where a test goes through the production `ChatBrain` (a seeded
BridgeParser) it runs at all seven parser seeds 42, 43, 44, 100, 101, 102, 1.

Run: CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m pytest tests/test_moat_allowlist_build6.py -q
"""
from __future__ import annotations

import json
import os
import random

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest

from research.runners import _moat_allowlist as A
from research.runners import _moat_allowlist_relations as T
from research.runners.brain_conversational_agent import BridgeParser

pytestmark = pytest.mark.filterwarnings("ignore")

SEEDS = (42, 43, 44, 100, 101, 102, 1)
FLAG = "BRAIN_MOAT_ALLOWLIST"


@pytest.fixture(autouse=True)
def _clean_flags(monkeypatch):
    for k in (FLAG, "BRAIN_MOAT_SLUG_NORM", "BRAIN_CLAIM_MOAT", "BRAIN_RICH_BATCH_RENDER", "BRAIN_TOUCHPOINT_A_FACT_CLAUSE",
              "BRAIN_ONE_BRAIN", "BRAIN_OPEN_ENDED"):
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


def _on(chat, surface, gathered):
    os.environ[FLAG] = "1"
    try:
        chat._claim_verifier_cache = {}
        return chat._verify_claim_set(surface, gathered)
    finally:
        os.environ.pop(FLAG, None)


def keep(sentence, gathered):
    return A.allowlist_verify(sentence, [list(g) for g in gathered])["accepted"]


# =================================================================================================================== B. shape 4
G = {
    'sport': ('boca_juniors', 'sport', 'association_football_club'),
    'instrument': ('sir_cliff_richard', 'instrument', 'history_of_the_guitar'),
    'conflict': ('charles_degaulle', 'conflict', 'war_world_two'),
    'spouse': ('stephane_audran', 'spouse', 'jean_louis_trintignant'),
    'parent_taxon': ('pothyne', 'parent_taxon', 'longhorn_beetles'),
    'father': ('carl_laemmle_jr', 'father', 'carl_laemmle'),
    'child': ('john_huston', 'child', 'huston_anjelica'),
    'author': ('lord_of_the_rings', 'author', 'j_r_r_tolkien'),
    'platform': ('street_fighter', 'platform', 'windows_applications'),
}
NINE = sorted(G)


def test_the_phrase_table_is_the_ruled_one():
    """Claude rulings 2026-10-21: one wording per relation; 'plays' (instrument) and 'participated in' (conflict) are the Wikidata
    labels' own verbs; 'the' only for the functional parent_taxon and father; spouse orientation-exact; child and author as designed.
    A change here is a change of the reviewed table, not of a test."""
    got = {r: (row.text, row.orientation) for r, row in T.PHRASE_READINGS.items()}
    assert got == {'sport': ('is a sport of', T.INV), 'instrument': ('plays', T.FWD), 'conflict': ('participated in', T.FWD),
                   'spouse': ('is a spouse of', T.INV), 'parent_taxon': ('is the parent taxon of', T.INV),
                   'father': ('is the father of', T.INV), 'child': ('is a child of', T.INV), 'author': ('is an author of', T.INV),
                   'platform': ('is a platform for', T.INV)}


def test_every_phrase_row_is_well_formed_and_cannot_collide_with_shape_3():
    of_rels = {r for r, row in T.RELATIONS.items() if r.endswith("_of")}
    for rel, row in T.PHRASE_READINGS.items():
        assert T.relation_class(rel) in T.NO_READING_CLASSES and not T.has_verb_reading(rel), rel   # shapes 1-3 still reject
        assert row.orientation in (T.FWD, T.INV) and row.why.strip(), rel
        assert row.text == row.text.lower() and row.text.isascii() and row.text == " ".join(row.text.split()), rel
        words = row.text.split()
        if len(words) >= 4 and words[:2] in (["is", "a"], ["is", "an"], ["is", "the"]) and words[-1] == "of":
            assert "_".join(words[2:]) not in of_rels, rel                                    # no shape-3 collision
        if words[1:2] == ["the"]:
            assert rel in ("father", "parent_taxon"), rel                                    # "the" asserts uniqueness
    assert T.phrase_reading("no_such_relation") is None and T.phrase_reading("country") is None
    assert sorted(T.PHRASE_READINGS) == sorted(r for r, row in T.RELATIONS.items() if row.cls in T.NO_READING_CLASSES)


@pytest.mark.parametrize("sentence,rel", [
    ("Association football club is a sport of Boca Juniors.", 'sport'),
    ("Sir Cliff Richard plays history of the guitar.", 'instrument'),
    ("Charles Degaulle participated in war world two.", 'conflict'),
    ("Jean Louis Trintignant is a spouse of Stephane Audran.", 'spouse'),
    ("Longhorn beetles is the parent taxon of Pothyne.", 'parent_taxon'),
    ("Carl Laemmle is the father of Carl Laemmle Jr.", 'father'),           # a prefix-sharing name: the longest slug wins
    ("Huston Anjelica is a child of John Huston.", 'child'),
    ("J R R Tolkien is an author of the Lord of the Rings.", 'author'),     # 'of the' inside a name
    ("Windows applications is a platform for Street Fighter.", 'platform'),
    ("The carl_laemmle is the father of carl_laemmle_jr.", 'father'),      # the stub's own slug-verbatim form
    ("The association_football_club is a sport of boca_juniors.", 'sport'),
    ("The sir_cliff_richard plays history_of_the_guitar.", 'instrument'),
    ("Carl Laemmle is the Father of Carl Laemmle Jr.", 'father'),          # a capitalised (not all-caps) phrase word
    ("The Carl Laemmle is the father of the Carl Laemmle Jr.", 'father'),  # [Det] A ... [Det] B
])
def test_shape4_keeps_the_faithful_wording(seed_chats, sentence, rel):
    assert keep(sentence, [G[rel]]), A.allowlist_verify(sentence, [G[rel]])["reject_reason"]
    r = A.allowlist_check(sentence, [G[rel]])
    assert r["shape"] == "shape4" and r["props"] == [list(G[rel])]
    for sd in SEEDS:
        assert _on(seed_chats[sd], sentence, [list(G[rel])])[0] is True, (sd, sentence)


ROLE_REVERSED = {
    'sport': ["Boca Juniors is a sport of association football club.", "The boca_juniors is a sport of association_football_club."],
    'instrument': ["History of the guitar plays Sir Cliff Richard.", "The history_of_the_guitar plays sir_cliff_richard."],
    'conflict': ["War world two participated in Charles Degaulle.", "The war_world_two participated in charles_degaulle."],
    # true by symmetry in the world, but not the gathered orientation: a conservative reject (ruling: orientation-exact)
    'spouse': ["Stephane Audran is a spouse of Jean Louis Trintignant.", "The stephane_audran is a spouse of jean_louis_trintignant."],
    'parent_taxon': ["Pothyne is the parent taxon of longhorn beetles.", "The pothyne is the parent taxon of longhorn_beetles."],
    'father': ["Carl Laemmle Jr is the father of Carl Laemmle.", "The carl_laemmle_jr is the father of carl_laemmle."],
    'child': ["John Huston is a child of Huston Anjelica.", "The john_huston is a child of huston_anjelica."],
    'author': ["Lord of the Rings is an author of J R R Tolkien.", "The lord_of_the_rings is an author of j_r_r_tolkien."],
    'platform': ["Street Fighter is a platform for windows applications.", "The street_fighter is a platform for windows_applications."],
}


@pytest.mark.parametrize("rel", NINE)
def test_every_rows_role_reversed_sentence_rejects_at_every_seed(seed_chats, rel):
    """The brief: every row gets pinned role-reversal tests (the reversed sentence must reject). Also when the reversed triple is NOT
    gathered but the row's own is, and through the production claim gate at every parser seed."""
    for s in ROLE_REVERSED[rel]:
        res = A.allowlist_verify(s, [list(G[rel])])
        assert not res["accepted"] and "not gathered" in res["reject_reason"], (s, res["reject_reason"])
        for sd in SEEDS:
            assert _on(seed_chats[sd], s, [list(G[rel])])[0] is False, (sd, s)


@pytest.mark.parametrize("sentence,rel", [
    ("Association football club is the sport of Boca Juniors.", 'sport'),
    ("Jean Louis Trintignant is the spouse of Stephane Audran.", 'spouse'),
    ("Huston Anjelica is the child of John Huston.", 'child'),
    ("J R R Tolkien is the author of Lord of the Rings.", 'author'),        # (new_testiment, author, peter_the_disciple) is the witness
    ("Windows applications is the platform for Street Fighter.", 'platform'),
    ("Longhorn beetles is a parent taxon of Pothyne.", 'parent_taxon'),
    ("Carl Laemmle is a father of Carl Laemmle Jr.", 'father'),
    ("Carl Laemmle is father of Carl Laemmle Jr.", 'father'),
    ("Huston Anjelica is child of John Huston.", 'child'),
])
def test_the_wrong_determiner_in_a_phrase_rejects(sentence, rel):
    assert not keep(sentence, [G[rel]]), sentence


@pytest.mark.parametrize("sentence,rel", [
    ("Boca Juniors sports association football club.", 'sport'),
    ("Boca Juniors plays association football club.", 'sport'),            # 'plays' is instrument's row, not sport's
    ("Sir Cliff Richard played history of the guitar.", 'instrument'),     # no tense variants of a phrase
    ("Sir Cliff Richard play history of the guitar.", 'instrument'),
    ("Sir Cliff Richard invented history of the guitar.", 'instrument'),
    ("History of the guitar is an instrument of Sir Cliff Richard.", 'instrument'),
    ("Charles Degaulle fought in war world two.", 'conflict'),
    ("Charles Degaulle conflicts war world two.", 'conflict'),
    ("Charles Degaulle participates in war world two.", 'conflict'),
    ("Charles Degaulle participated war world two.", 'conflict'),
    ("Jean Louis Trintignant is a husband of Stephane Audran.", 'spouse'),
    ("The carl_laemmle_jr fathers carl_laemmle.", 'father'),
    ("The father of Carl Laemmle Jr is Carl Laemmle.", 'father'),           # the noun-of form is not in the family
    ("Huston Anjelica is a daughter of John Huston.", 'child'),
    ("John Huston is a parent of Huston Anjelica.", 'child'),
    ("The lord_of_the_rings authors j_r_r_tolkien.", 'author'),
    ("Street Fighter runs on windows applications.", 'platform'),
    ("Pothyne parent taxon longhorn beetles.", 'parent_taxon'),             # the bare telegraphic label stays unread
    ("Sir Cliff Richard instrument history of the guitar.", 'instrument'),
])
def test_other_wordings_stay_rejected(sentence, rel):
    assert not keep(sentence, [G[rel]]), sentence


def test_an_inverse_pair_is_not_cross_inferred():
    assert not keep("Carl Laemmle Jr is a child of Carl Laemmle.", [G['father']])
    assert not keep("Carl Laemmle is the father of Carl Laemmle Jr.", [('carl_laemmle', 'child', 'carl_laemmle_jr')])


def test_a_row_is_read_only_for_its_own_gathered_relation():
    assert not keep("Carl Laemmle is the father of Carl Laemmle Jr.", [('carl_laemmle_jr', 'spouse', 'carl_laemmle')])
    assert not keep("Carl Laemmle is a spouse of Carl Laemmle Jr.", [G['father']])
    assert not keep("Carl Laemmle is the father of Carl Laemmle Jr.", [('carl_laemmle_jr', 'country', 'carl_laemmle')])


@pytest.mark.parametrize("sentence,gathered", [
    ("Carl Laemmle is not the father of Carl Laemmle Jr.", [G['father']]),
    ("Carl Laemmle is also the father of Carl Laemmle Jr.", [G['father']]),
    ("Carl Laemmle is the father of Carl Laemmle Jr too.", [G['father']]),
    ("Huston Anjelica is a child of John Huston and Evelyn Keyes.", [G['child'], ('evelyn_keyes', 'child', 'huston_anjelica')]),
    ("Huston Anjelica and Danny Huston is a child of John Huston.", [G['child'], ('john_huston', 'child', 'danny_huston')]),
    ("Sir Cliff Richard plays history of the guitar and keyboard hammer.",
     [G['instrument'], ('sir_cliff_richard', 'instrument', 'keyboard_hammer')]),
    ("Sir Cliff Richard plays history of the guitar, keyboard hammer.",
     [G['instrument'], ('sir_cliff_richard', 'instrument', 'keyboard_hammer')]),
    ("CARL LAEMMLE is the father of Carl Laemmle Jr.", [G['father']]),
    ("Carl Laemmle is the FATHER of Carl Laemmle Jr.", [G['father']]),
    ("Carl Laemmle IS the father of Carl Laemmle Jr.", [G['father']]),
    ("Carl Laemmle is the father-of Carl Laemmle Jr.", [G['father']]),
    ("Carl Laemmle is the father of Carl Laemmle Jr!", [G['father']]),
    ("Perhaps Carl Laemmle is the father of Carl Laemmle Jr.", [G['father']]),
    ("Carl Laemmle is the father of Carl Laemmle Jr. Carl Laemmle is the father of Carl Laemmle Jr.", [G['father']]),
])
def test_additions_lists_case_and_repeats_reject(sentence, gathered):
    assert not keep(sentence, gathered), sentence


def test_existing_shapes_are_unaffected():
    assert keep("Rick Wakeman is a member of Yes.", [('rick_wakeman', 'member_of', 'yes')])
    assert keep("Spain shares border with France.", [('spain', 'shares_border_with', 'france')])
    # a phrase relation gathered beside a verb-reading one: each sentence on its own shape
    g = [G['father'], ('carl_laemmle_jr', 'country', 'united_states')]
    assert keep("Carl Laemmle is the father of Carl Laemmle Jr. Carl Laemmle Jr country United States.", g)


def test_a_reject_still_names_the_class_of_a_blocked_verb_reading():
    """A sentence that never started a phrase keeps its build-5 reason; one that started it names the phrase."""
    why = A.allowlist_verify("The carl_laemmle_jr fathers carl_laemmle.", [G['father']])["reject_reason"]
    assert T.ROLE_REVERSING in why and "shape 4" not in why, why
    why = A.allowlist_verify("Carl Laemmle is the father Carl Laemmle Jr.", [G["father"]])["reject_reason"]
    assert "the reviewed relation phrase 'is the father of' of father (shape 4)" in why, why


# ---- the stub renderer under the flag
def test_the_stub_writes_the_phrase_under_the_flag_only(monkeypatch):
    from research.runners._grounded_lang_p3_derisk import TemplateStubFaculty
    from research.runners.brain_chat_tui import StubRenderer
    stub, fac = StubRenderer(), TemplateStubFaculty()
    for rel, svo in G.items():
        monkeypatch.delenv(FLAG, raising=False)
        assert stub.render_svo(*svo) == fac.render_svo(*svo)                     # flag off: byte-identical
        for val in ("0", "off", "garbage"):
            monkeypatch.setenv(FLAG, val)
            assert stub.render_svo(*svo) == fac.render_svo(*svo)
        monkeypatch.setenv(FLAG, "1")
        surface, asserted = stub.render_svo(*svo)
        row = T.PHRASE_READINGS[rel]
        a, _v, p = svo
        assert surface == "The %s %s %s." % ((a, row.text, p) if row.orientation == T.FWD else (p, row.text, a))
        assert asserted == fac.render_svo(*svo)[1]
        assert keep(surface, [svo]), surface
    monkeypatch.setenv(FLAG, "1")
    assert stub.render_svo("spain", "country", "europe") == fac.render_svo("spain", "country", "europe")   # no row: unchanged


def _store_facts():
    if not os.path.exists(T.STORE_FACTS):
        pytest.skip("production store not on this machine")
    return [(r["fact"]["agent"], r["fact"]["action"], r["fact"]["patient"]) for r in json.load(open(T.STORE_FACTS))
            if r["fact"].get("polarity") == "AFFIRM"]


def test_stub_and_matcher_agree_on_every_store_fact_of_the_nine(monkeypatch):
    """Under the flag the stub sentence of every store fact of the nine relations is kept (preflip prototype: 5,444 of 5,444), and its
    role-reversed stub form is rejected; flag off, the stub output equals TemplateStubFaculty for every one of them and a seeded
    sample of the rest."""
    from research.runners._grounded_lang_p3_derisk import TemplateStubFaculty
    from research.runners.brain_chat_tui import StubRenderer
    facts = _store_facts()
    nine = [f for f in facts if f[1] in T.PHRASE_READINGS]
    assert len(nine) == 5444
    stub, fac = StubRenderer(), TemplateStubFaculty()
    rest = random.Random(6).sample([f for f in facts if f[1] not in T.PHRASE_READINGS], 2000)
    for f in nine + rest:
        assert stub.render_svo(*f) == fac.render_svo(*f), f
    monkeypatch.setenv(FLAG, "1")
    bad = [f for f in nine if not keep(stub.render_svo(*f)[0], [f])]
    assert not bad, bad[:5]
    rev = [f for f in nine[::7] if f[0] != f[2] and keep(stub.render_svo(f[2], f[1], f[0])[0], [f])]
    assert not rev, rev[:5]


# =================================================================================================================== C. the two leak classes
# Claude rulings 2026-10-21 on preflip topic 3: (C1) first- AND second-person subject names in spaced form reject in any casing (the
# honesty boundary); the verbatim slug stays sayable. (C2) in a multi-sentence reply, a later entity span led by "the" whose words are a
# subset of an earlier span's words (a different slug) rejects that later sentence. (C3) lower-case names with a lexical verb stay open.
# The design's position-independent rules (a), (b1'), (b2) are built as designed.
G_WAKE = [("rick_wakeman", "member_of", "yes_band"), ("the_band", "award_received", "grammy_lifetime_achievement_award")]
WAKE1 = "Rick Wakeman is a member of Yes band."


def _r6():
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "research" / "findings" / "raw" / "_moat_allowlist_hunt4" / "r6" / "cases.json"
    return {c["id"]: c for c in json.loads(p.read_text(encoding="utf-8"))}


@pytest.mark.parametrize("sentence,gathered,why", [
    # (a) a definite description
    (WAKE1 + " The band award received Grammy lifetime achievement award.", G_WAKE, "opens with 'the' and a lower-case word"),
    ("The band award received Grammy lifetime achievement award.", G_WAKE, "opens with 'the' and a lower-case word"),
    ("the band award received grammy lifetime achievement award.", G_WAKE, "opens with 'the' and a lower-case word"),
    ("The_band award received Grammy lifetime achievement award.", G_WAKE, "opens with 'the' and a lower-case word"),
    ("Trabzon Province country the republic of turkey.", [("trabzon_province", "country", "the_republic_of_turkey")], "description"),
    ("The weather in Ottawa shares border with Gatineau Quebec.", [("the_weather_in_ottawa", "shares_border_with", "gatineau_quebec")],
     "description"),
    ("The u country u_s_of_a.", [("the_u", "country", "u_s_of_a")], "description"),
    # (b1) first / second person, any case
    ("I need a beat genre hip hop music.", [("i_need_a_beat", "genre", "hip_hop_music")], "first- or second-person subject"),
    ("I Need a Beat genre hip hop music.", [("i_need_a_beat", "genre", "hip_hop_music")], "first- or second-person subject"),
    ("Louis the alligator composer I heart tuesdays.", [("louis_the_alligator", "composer", "i_heart_tuesdays")], "subject"),
    ("Louis the Alligator composer I Heart Tuesdays.", [("louis_the_alligator", "composer", "i_heart_tuesdays")], "subject"),
    ("You Know Who genre rock.", [("you_know_who", "genre", "rock")], "first- or second-person subject"),
    ("you know who genre rock.", [("you_know_who", "genre", "rock")], "first- or second-person subject"),
    ("YOU Know Who genre rock.", [("you_know_who", "genre", "rock")], ""),                                   # all-caps: caps table
    ("We Are Scientists genre indie rock.", [("we_are_scientists", "genre", "indie_rock")], ""),
    # (b1') a one-word pronoun span not after a determiner
    ("i language of work or name French language.", [("i", "language_of_work_or_name", "french_language")], "reads as the pronoun"),
    ("I language of work or name French language.", [("i", "language_of_work_or_name", "french_language")], "reads as the pronoun"),
    ("Abc genre you.", [("abc", "genre", "you")], "reads as the pronoun"),
    # (b2) a modal / pronoun / focus word written as prose
    ("will hurt award received theatre world awards.", [("will_hurt", "award_received", "theatre_world_awards")], "written as prose"),
    ("Will hurt award received Theatre World Awards.", [("will_hurt", "award_received", "theatre_world_awards")], "written as prose"),
    ("brian may country of citizenship united kingom.", [("brian_may", "country_of_citizenship", "united_kingom")], "written as prose"),
    ("John Swithen place of birth Portland me.", [("john_swithen", "place_of_birth", "portland_me")], "written as prose"),
    ("Fire walk with me country of origin u_s_of_a.", [("fire_walk_with_me", "country_of_origin", "u_s_of_a")], "written as prose"),
    ("lifting you country of citizenship u_s_of_a.", [("lifting_you", "country_of_citizenship", "u_s_of_a")], "written as prose"),
    ("just keep walking country of origin commonwealth_australia.",
     [("just_keep_walking", "country_of_origin", "commonwealth_australia")], "written as prose"),
    ("Yule Brenner place of birth Port may city.", [("yule_brenner", "place_of_birth", "port_may_city")], "written as prose"),
])
def test_names_that_read_as_a_description_or_a_clause_reject(seed_chats, sentence, gathered, why):
    res = A.allowlist_verify(sentence, [list(g) for g in gathered])
    assert not res["accepted"], sentence
    assert why in res["reject_reason"], res["reject_reason"]
    for sd in (42, 1):
        assert _on(seed_chats[sd], sentence, [list(g) for g in gathered])[0] is False, (sd, sentence)


@pytest.mark.parametrize("case_id", ["R6-MC02", "R6-LC03"])
def test_round6_b_cases_named_by_the_design_now_reject(case_id):
    """Topic 3 measured 3 round-6 B cases newly rejected (R6-DP06 'the cruellest month', R6-MC02 'the wHO', R6-LC03 'will hurt');
    the two whose span the rule reads directly are pinned here (R6-DP06's description is in the object)."""
    c = _r6()[case_id]
    assert not A.allowlist_verify(c["sentence"], c["gathered"])["accepted"], c["sentence"]


@pytest.mark.parametrize("sentence,gathered", [
    # the slug written verbatim (the stub's form): one identifier
    ("The the_band award_receiveds grammy_lifetime_achievement_award.", G_WAKE),
    ("the_band award received Grammy lifetime achievement award.", G_WAKE),
    ("i_need_a_beat genre hip hop music.", [("i_need_a_beat", "genre", "hip_hop_music")]),
    ("Louis the Alligator composer i_heart_tuesdays.", [("louis_the_alligator", "composer", "i_heart_tuesdays")]),
    ("will_hurt award received theatre world awards.", [("will_hurt", "award_received", "theatre_world_awards")]),
    ("The i language_of_work_or_names french_language.", [("i", "language_of_work_or_name", "french_language")]),   # Det before i
    # real names: a capital after 'the', 'the' not first, a capital mid-sentence, a digit, an abbreviation, a Roman numeral
    ("The Who genre symphonic hard rock.", [("the_who", "genre", "symphonic_hard_rock")]),
    ("the Who genre symphonic hard rock.", [("the_who", "genre", "symphonic_hard_rock")]),
    ("The Band award received Grammy lifetime achievement award.", G_WAKE),          # a one-sentence 'The Band' reply
    ("The Rolling Stones country of origin United Kingom.", [("the_rolling_stones", "country_of_origin", "united_kingom")]),
    ("Sheila Marie Ryan place of birth the Bronx.", [("sheila_marie_ryan", "place_of_birth", "the_bronx")]),
    ("The U country u_s_of_a.", [("the_u", "country", "u_s_of_a")]),
    ("Le X located in the administrative territoria The 7th arrondissement of Paris.",
     [("le_x", "located_in_the_administrative_territoria", "the_7th_arrondissement_of_paris")]),
    ("Real Valladolid CF country Kingdom of the Spains.", [("real_valladolid_cf", "country", "kingdom_of_the_spains")]),
    ("Peter the Disciple occupation assistant priest.", [("peter_the_disciple", "occupation", "assistant_priest")]),
    ("Louis the alligator country of origin u_s_of_a.", [("louis_the_alligator", "country_of_origin", "u_s_of_a")]),
    ("Brian May country of citizenship united kingom.", [("brian_may", "country_of_citizenship", "united_kingom")]),
    ("June follows May sickness.", [("june", "follows", "may_sickness")]),                                       # R6-F48
    ("Will Munny country of origin u_s_of_a.", [("will_munny", "country_of_origin", "u_s_of_a")]),
    ("Will Hurt award received Theatre World Awards.", [("will_hurt", "award_received", "theatre_world_awards")]),
    ("John Swithen place of birth Portland ME.", [("john_swithen", "place_of_birth", "portland_me")]),
    ("Clinton Rodham member of political party Republican Party US.",
     [("clinton_rodham", "member_of_political_party", "republican_party_us")]),
    ("Charles I of England occupation political figure.", [("charles_i_of_england", "occupation", "political_figure")]),
    ("Cold War I has part united soviets socialist republic.", [("cold_war_i", "has_part", "united_soviets_socialist_republic")]),
    ("Lom Norway shares border with Vang i Valdres.", [("lom_norway", "shares_border_with", "vang_i_valdres")]),
    ("Fire Walk with Me country of origin u_s_of_a.", [("fire_walk_with_me", "country_of_origin", "u_s_of_a")]),
    ("Only Fools and Horses country of origin united kingom.", [("only_fools_and_horses", "country_of_origin", "united_kingom")]),
    ("Lifting You country of citizenship u_s_of_a.", [("lifting_you", "country_of_citizenship", "u_s_of_a")]),
    ("Feel Your Heart Beat genre pop starlets.", [("feel_your_heart_beat", "genre", "pop_starlets")]),
    ("If Chins Could Kill occupation television actor.", [("if_chins_could_kill", "occupation", "television_actor")]),
    ("Yule Brenner place of birth Port May City.", [("yule_brenner", "place_of_birth", "port_may_city")]),
    ("Trabzon Province country the Republic of Turkey.", [("trabzon_province", "country", "the_republic_of_turkey")]),
    ("Steve Threw Up record label XL Recordings.", [("steve_threw_up", "record_label", "xl_recordings")]),
])
def test_real_names_and_verbatim_slugs_stay_kept(sentence, gathered):
    res = A.allowlist_verify(sentence, [list(g) for g in gathered])
    assert res["accepted"], (sentence, res["reject_reason"])


@pytest.mark.parametrize("sentence,gathered", [
    # C3: lower-case names with a lexical verb stay open (known B residue; no verb lexicon, no proper-name bit in the store)
    ("steve threw up record label xl recordings.", [("steve_threw_up", "record_label", "xl_recordings")]),
    # build-6 review (doc): a first-person "I" that is not the name's first word; open because every such store name is a Roman numeral
    # or a non-English "i" (synthetic slugs here: no store witness; round 8 probes conversation-taught names)
    ("Baby I Love You performer the Ramones.", [("baby_i_love_you", "performer", "the_ramones")]),
    ("And i love her performer the Beatles.", [("and_i_love_her", "performer", "the_beatles")]),
    ("Vang i Valdres country Norway.", [("vang_i_valdres", "country", "norway")]),          # a real name: a deliberate keep
])
def test_known_b_residue_stays_kept(sentence, gathered):
    """Pinned so a later change is visible: the next blind round adjudicates it."""
    assert A.allowlist_verify(sentence, [list(g) for g in gathered])["accepted"]


# ---- C2: a definite description that points back (cross-sentence, in allowlist_verify)
@pytest.mark.parametrize("reply,gathered,bad_sentence", [
    (WAKE1 + " The Band award received Grammy lifetime achievement award.", G_WAKE, 2),           # Title case: (a) keeps, C2 rejects
    (WAKE1 + " the Band award received Grammy lifetime achievement award.", G_WAKE, 2),
    # a 'the' determiner before a different, shorter slug
    (WAKE1 + " The band genre rock.", [G_WAKE[0], ("band", "genre", "rock")], 2),
    # the earlier span may be an object or a subject; the later span may be an object
    ("Yes band record label Atlantic jazz. Rick Wakeman is a member of the Band.",
     [("yes_band", "record_label", "atlantic_jazz"), ("rick_wakeman", "member_of", "the_band")], 2),
    # a real different entity whose name is among the earlier name's words: the rule is conservative (it cannot tell)
    ("Yes band record label Atlantic jazz. The Band genre rock musician.",
     [("yes_band", "record_label", "atlantic_jazz"), ("the_band", "genre", "rock_musician")], 2),
])
def test_a_later_the_span_pointing_back_rejects_that_sentence(reply, gathered, bad_sentence):
    res = A.allowlist_verify(reply, [list(g) for g in gathered])
    assert not res["accepted"], reply
    assert "points back" in res["reject_reason"] and ("[sentence %d of" % bad_sentence) in res["reject_reason"], res["reject_reason"]
    # each sentence alone is kept: the rule is cross-sentence
    assert all(A.allowlist_verify(s, [list(g) for g in gathered])["accepted"] for s in A.split_sentences(reply))


@pytest.mark.parametrize("reply,gathered", [
    ("The Band award received Grammy lifetime achievement award.", G_WAKE),                          # one sentence: nothing earlier
    ("The Band award received Grammy lifetime achievement award. " + WAKE1, G_WAKE),                  # the 'the' span comes FIRST
    (WAKE1 + " The Who genre symphonic hard rock.", [G_WAKE[0], ("the_who", "genre", "symphonic_hard_rock")]),   # unrelated 'the X'
    (WAKE1 + " Sheila Marie Ryan place of birth the Bronx.", [G_WAKE[0], ("sheila_marie_ryan", "place_of_birth", "the_bronx")]),
    # the same slug named again is the same thing
    ("The Who genre symphonic hard rock. The Who record label Decca music group.",
     [("the_who", "genre", "symphonic_hard_rock"), ("the_who", "record_label", "decca_music_group")]),
    ("The dog chases cat. The cat eats fish.", [("dog", "chase", "cat"), ("cat", "eat", "fish")]),
    ("The dog chases the cat. The cat eats the fish.", [("dog", "chase", "cat"), ("cat", "eat", "fish")]),
    # the later span holds a word the earlier one does not
    ("Rick Wakeman is a member of the Band. The Yes band record label Atlantic jazz.",
     [("rick_wakeman", "member_of", "the_band"), ("yes_band", "record_label", "atlantic_jazz")]),
])
def test_one_sentence_unrelated_and_repeated_names_stay_kept(reply, gathered):
    res = A.allowlist_verify(reply, [list(g) for g in gathered])
    assert res["accepted"], (reply, res["reject_reason"])


# ---- C2 review fix (build-6 review, leak-B): a hyphenated names-table surface of the antecedent ("boy-band", "norway u-21") split on
# spaces and underscores only stayed one word, so the later 'the' span was not a subset; the word split is now every non-alphanumeric run
TT_BOY = ("take_that", "instance_of", "boy_band")
BAND_CO = ("the_band", "country_of_origin", "canada_portal")
NOR21 = ("norway_u_21", "sport", "association_football_club")
THE_U = ("the_u", "country", "u_s_of_a")


@pytest.mark.parametrize("reply,gathered", [
    ("Take That is an instance of boy-band. The Band country of origin Canada portal.", [TT_BOY, BAND_CO]),
    ("Take That is an instance of boy-band. the_band country of origin canada_portal.", [TT_BOY, BAND_CO]),
    ("Take That is an instance of boy-band. The the_band country_of_origins canada_portal.", [TT_BOY, BAND_CO]),
    ("Association football club is a sport of Norway u-21. The U country U S of A.", [NOR21, THE_U]),
    # the space-spelled twins (already rejected before the fix)
    ("Take That is an instance of boy band. The Band country of origin Canada portal.", [TT_BOY, BAND_CO]),
    ("Association football club is a sport of Norway u 21. The U country U S of A.", [NOR21, THE_U]),
])
def test_a_hyphen_surface_of_the_antecedent_does_not_hide_it(seed_chats, reply, gathered):
    g = [list(x) for x in gathered]
    res = A.allowlist_verify(reply, g)
    assert not res["accepted"] and "points back" in (res["reject_reason"] or ""), (reply, res["reject_reason"])
    assert all(A.allowlist_verify(s, g)["accepted"] for s in A.split_sentences(reply)), "each sentence alone is kept"
    for sd, chat in seed_chats.items():
        assert _on(chat, reply, g)[0] is False, (sd, reply)


def test_the_span_word_split_breaks_on_every_non_alphanumeric_run():
    toks = ["x"]
    for orig, words in (("boy-band", ("boy", "band")), ("Norway u-21", ("norway", "u", "21")), ("the_band", ("band",)),
                        ("The\u00a0Band", ("band",)), ("Bod\u00f8", ("bod\u00f8",)), ("a.b/c", ("a", "b", "c"))):
        got = A._ent_span_words([(1, [], [("ent", 0, "s")])], toks, [orig])
        assert got[0][1] == words, (orig, got)


# ---- known B residue (build-6 review): a hypernym definite description whose noun is NOT a word of the antecedent's name. The subset
# rule cannot see it ("The Band" after "the Rolling Stones" reads as the Stones). Pinned KEPT so a change is visible; round 8 targets
# hypernym and question-supplied antecedents.
KR = ("keith_richards", "member_of", "the_rolling_stones")


@pytest.mark.parametrize("reply", [
    "Keith Richards is a member of the Rolling Stones. The Band country of origin Canada portal.",
    "Keith Richards is a member of the Rolling Stones. The BAnd country of origin Canada portal.",
    "Keith Richards is a member of the Rolling Stones. the_band country of origin canada_portal.",
])
def test_known_b_residue_hypernym_definite_description_stays_kept(reply):
    res = A.allowlist_verify(reply, [list(KR), list(BAND_CO)])
    assert res["accepted"], res["reject_reason"]


def test_the_build6_name_rules_only_add_rejections():
    """Monotone: a sentence build 6 keeps parses as before and passes every build-5 check (the new checks run after them)."""
    import inspect
    src = inspect.getsource(A.allowlist_check)
    assert src.index("_structure_word_violation(parses") < src.index("_name_reading_violation(parses")
    assert "_points_back(trace)" in inspect.getsource(A.allowlist_verify)


# =================================================================================================================== A. the whole mouth path
# Under the flag a mouth sentence reaches the user only if the allowlist kept it against the triple(s) it was generated for, AND the joined
# mouth text of the reply passed `allowlist_verify` against the union of the turn's gathered triples. Declared non-claims: the spiking
# mood marker {"!", ".", ""}, the leading "perhaps " of a spiking hypothesis, the host guess frames and abstain strings (pinned verbatim
# below). Flag OFF: every path equals the golden captured at fde5e739e before any build-6 edit.
from pathlib import Path  # noqa: E402

from tests import _moat_mouthpath_harness as H  # noqa: E402

GOLDEN6 = json.loads((Path(__file__).parent / "fixtures" / "moat_mouthpath_flagoff_golden.json").read_text())
DOG, CAT, BRAIN = ["dog", "chase", "cat"], ["cat", "eat", "fish"], ["brain", "use", "spikes"]


@pytest.mark.parametrize("val", [None, "0", "off", "garbage"])
def test_flag_off_mouth_paths_equal_the_base_golden(monkeypatch, val):
    """render_paragraph / _render_one_verified / _render_paragraph_batched (scripted matrix: call sequences included) and the real
    ChatBrain render / hypothesis / spiking recall / mood tone / _verify paths, with the flag unset or falsy, equal the golden."""
    if val is not None:
        monkeypatch.setenv(FLAG, val)
    got = json.loads(json.dumps(H.run_all()))
    assert set(got["composer"]) == set(GOLDEN6["composer"])
    bad = [k for k in GOLDEN6["composer"] if got["composer"][k] != GOLDEN6["composer"][k]]
    bad += [k for k in GOLDEN6["chatbrain"] if got["chatbrain"].get(k) != GOLDEN6["chatbrain"][k]]
    assert not bad, bad[:5]


class FakeMouth:
    """A stand-in SpikingRecallMouth (the cached `_spk_recall_mouth`): scripted surface per svo, else the faithful spiking frame."""

    def __init__(self, table=None):
        self.table = table or {}

    def render(self, a, v, p):
        return self.table.get((a, v, p), "the %s %ss the %s" % (a, v, p))


_BASE = {}


def _chat(renderer="stub", mouth=None, level=0):
    """A fresh ChatBrain on the tiny-demo facts (one shared BridgeParser, seed 42: under the flag it is never consulted)."""
    from research.runners.brain_chat_tui import ChatBrain, StubRenderer
    if "chat" not in _BASE:
        _BASE["chat"] = H.tiny_chat("stub")
    chat = ChatBrain.__new__(ChatBrain)
    chat.__dict__.update({k: v for k, v in _BASE["chat"].__dict__.items()
                          if k in ("inner", "stored_facts", "agents_set", "actions_set", "patients_set", "inflect", "raw_mode")})
    chat.renderer = StubRenderer() if renderer == "stub" else renderer
    chat._spk_recall_mouth = mouth or FakeMouth()
    chat._mood_tone_level = level
    chat._claim_verifier_cache = {}
    return chat


def _comp(chat):
    from research.runners.rich_answer_composer import RichAnswerComposer
    comp = RichAnswerComposer.__new__(RichAnswerComposer)
    comp.chat = chat
    return comp


class Prose:
    """A Qwen-like renderer (asserted None) with optional regen and batch methods, scripted per svo; records its calls."""

    name = "scripted"

    def __init__(self, table, regen=None, batch=False):
        self.table, self.regen, self.calls = table, regen, []
        if batch:
            self.render_svo_batch = lambda svos: ([self.render_svo(*s) for s in svos], 0.0)
            self.render_svo_regen_batch = lambda svos: ([self.render_svo_regen(*s) for s in svos], 0.0)

    def render_svo(self, a, v, p):
        self.calls.append(("render_svo", a, v, p))
        return self.table.get((a, v, p), "The %s %ss the %s." % (a, v, p)), None

    def render_svo_regen(self, a, v, p):
        self.calls.append(("render_svo_regen", a, v, p))
        return (self.regen or {}).get((a, v, p), "The %s %ss the %s." % (a, v, p)), None


HOSTILE = ["The dog chases the cat and is famous for its vodka.", "The cat chases the dog.", "It chases the cat.",
           "The dog chases the cat!", "The dog chases the cat, perhaps.", "The dog chases the cat. The dog chases the cat."]


def _no_hostile(text):
    t = text or ""
    return not any(w in t for w in ("vodka", "cat chases the dog", "It chases", "cat!", "perhaps.")) and t.count("chases") <= 1


@pytest.mark.parametrize("hostile", HOSTILE)
@pytest.mark.parametrize("path", ["sequential", "batched", "render", "tpa"])
def test_hostile_mouth_text_never_reaches_the_reply(monkeypatch, path, hostile):
    """The key end-to-end pin, flag ON: a hostile render on each mouth path never appears in the output, and the faithful render on
    the same path IS emitted (counter-case, so the test cannot pass by rejecting everything)."""
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("BRAIN_SPIKING_MOUTH_RECALL", "0")                  # the renderer path, not the spiking surface
    for text, faithful in ((hostile, False), ("The dog chases the cat.", True)):
        ren = Prose({tuple(DOG): text}, regen={tuple(DOG): text}, batch=(path == "batched"))
        chat = _chat(ren)
        if path == "render":
            out = chat.render(list(DOG))
            assert (out == "The dog chases the cat.") is faithful, out
            assert _no_hostile(out) or faithful, out
            continue
        if path == "tpa":
            monkeypatch.setenv("BRAIN_TOUCHPOINT_A_FACT_CLAUSE", "1")
            with H.fake_touchpoint_a({tuple(DOG): text}):
                bird = "The dog chases the cat and the bird."
                para, kept, _d = _comp(_chat(Prose({tuple(DOG): bird}, regen={tuple(DOG): bird}))).render_paragraph([list(DOG)])
            assert (para == "The dog chases the cat.") is faithful and (kept == [DOG]) is faithful, para
            continue
        if path == "batched":
            monkeypatch.setenv("BRAIN_RICH_BATCH_RENDER", "1")
        para, kept, dropped = _comp(chat).render_paragraph([list(DOG), list(CAT)])
        assert _no_hostile(para), para
        assert para.startswith("The dog chases the cat.") is faithful, para
        assert "The cat eats the fish." in para and (DOG in kept) is faithful, para        # the other fact's sentence stays


def test_the_spiking_recall_surface_is_checked_by_the_allowlist(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    for bad in ("the cat chases the dog", "the dog chases the cat and the fish", "the dog chases the cat the", "it chases the cat"):
        chat = _chat(mouth=FakeMouth({tuple(DOG): bad}))
        assert chat.spiking_recall_surface(*DOG) is None, bad
        assert chat.spiking_recall_parts(*DOG) is None, bad
    chat = _chat()
    assert chat.spiking_recall_surface(*DOG) == "the dog chases the cat."
    assert chat.spiking_recall_parts(*DOG) == ("the dog chases the cat.", False)


def test_two_spiking_facts_join_with_a_terminal_period(monkeypatch):
    """Flag ON: 'the dog chases the cat. the cat eats the fish.'; flag OFF (golden) the unterminated run-on, which the matcher
    rejects as a joined string (the defect is real)."""
    para, kept, dropped = _comp(_chat()).render_paragraph([list(DOG), list(CAT)])
    assert para == "the dog chases the cat the cat eats the fish"                          # flag off: today's run-on
    assert GOLDEN6["composer"]["spiking_hit"]["render_paragraph"]["result"][0] == "the dog chases the cat the cat eats the fish"
    assert not A.allowlist_verify(para, [DOG, CAT])["accepted"]
    monkeypatch.setenv(FLAG, "1")
    para, kept, dropped = _comp(_chat()).render_paragraph([list(DOG), list(CAT)])
    assert para == "the dog chases the cat. the cat eats the fish." and kept == [DOG, CAT] and dropped == []
    assert A.allowlist_verify(para, [DOG, CAT])["accepted"]


# ---- the joined reply check
class Spy:
    def __init__(self, chat):
        self.chat, self.sets = chat, []
        orig = chat._verify_claim_set

        def spy(surface, gated):
            self.sets.append((surface, [list(g) for g in gated]))
            return orig(surface, gated)
        chat._verify_claim_set = spy


def test_joined_check_keeps_clean_replies_byte_equal(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("BRAIN_SPIKING_MOUTH_RECALL", "0")
    facts = [list(DOG), list(CAT), list(BRAIN), ["brain", "learn", "words"]]
    for n in range(1, 5):
        chat = _chat(Prose({}))
        para, kept, dropped = _comp(chat).render_paragraph(facts[:n])
        want = " ".join("The %s %ss the %s." % tuple(f) for f in facts[:n])
        assert para == want and kept == facts[:n] and dropped == [], (n, para)
    # an unterminated Qwen candidate followed by another sentence: terminated, both kept
    chat = _chat(Prose({tuple(DOG): "The dog chases the cat"}))
    para, kept, _d = _comp(chat).render_paragraph([list(DOG), list(CAT)])
    assert para == "The dog chases the cat. The cat eats the fish." and kept == [DOG, CAT]


def test_joined_check_cuts_at_the_first_later_sentence_that_fails_together(monkeypatch):
    """A later sentence that states an earlier proposition again (here a shape-2 list then one of its objects) or points back is cut,
    and with it every sentence after it (the LONGEST PASSING PREFIX, ruling A1); the cut facts go to `dropped`, not `kept`."""
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("BRAIN_SPIKING_MOUTH_RECALL", "0")
    f1, f2, f3, f4 = ["spain", "shares_border_with", "france"], ["spain", "shares_border_with", "portugal"], \
        ["spain", "member_of", "european_union"], ["spain", "country", "europe"]
    ren = Prose({tuple(f1): "Spain shares border with France and Portugal.", tuple(f2): "Spain shares border with Portugal.",
                 tuple(f3): "Spain is a member of the European Union.", tuple(f4): "Spain country Europe."})
    chat = _chat(ren)
    comp = _comp(chat)
    para, kept, dropped = comp.render_paragraph([f1, f2, f3, f4])
    assert para == "Spain shares border with France and Portugal." and kept == [f1] and dropped == [f2, f3, f4], para
    assert comp._last_reply_rejects and comp._last_reply_rejects[0]["svo"] == f2
    assert "both state" in comp._last_reply_rejects[0]["reject_reason"]
    # points back (C2) through the composer
    g = [list(x) for x in G_WAKE]
    ren = Prose({tuple(g[0]): WAKE1, tuple(g[1]): "The Band award received Grammy lifetime achievement award."})
    para, kept, dropped = _comp(_chat(ren)).render_paragraph(g)
    assert para == WAKE1 and kept == [g[0]] and dropped == [g[1]], para


def test_joined_check_invariants(monkeypatch):
    """(i) the emitted paragraph ('!' restored to '.') is accepted by allowlist_verify against the gathered set; (ii) every sentence
    prefix of it is accepted too; (iii) kept keeps gathered order; (iv) every joint check is against exactly the gathered set."""
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("BRAIN_SPIKING_MOUTH_RECALL", "0")
    rnd = random.Random(6)
    pool = [list(DOG), list(CAT), list(BRAIN), ["brain", "learn", "words"], ["brain", "store", "memory"]]
    kinds = ["The %s %ss the %s.", "The %s %ss the %s", "The %s %ss the %s and is famous.", "The %s %ss the %s. The %s %ss the %s."]
    for _ in range(40):
        facts = rnd.sample(pool, rnd.randint(1, 4))
        table = {}
        for f in facts:
            k = rnd.choice(kinds)
            table[tuple(f)] = k % (tuple(f) * (k.count("%s") // 3))
        chat = _chat(Prose(table))
        spy = Spy(chat)
        para, kept, dropped = _comp(chat).render_paragraph(facts)
        gated = [list(f) for f in facts]
        assert all(s == gated for _t, s in spy.sets)
        if kept:
            assert A.allowlist_verify(para.replace("!", "."), gated)["accepted"], para
            sents = A.split_sentences(para)
            for k in range(1, len(sents) + 1):
                assert A.allowlist_verify(" ".join(sents[:k]), gated)["accepted"]
        assert kept == [f for f in facts if f in kept] and sorted(map(tuple, kept + dropped)) == sorted(map(tuple, facts))


def test_every_sentence_failing_abstains_through_the_existing_branch(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("BRAIN_SPIKING_MOUTH_RECALL", "0")
    chat = _chat(Prose({tuple(DOG): HOSTILE[0], tuple(CAT): "The fish eats the cat."}, regen={tuple(DOG): HOSTILE[0],
                                                                                              tuple(CAT): "The fish eats the cat."}))
    comp = _comp(chat)
    comp.gather = lambda q, followup=False: ("dog", [list(DOG), list(CAT)])
    out = comp.answer("tell me about the dog")
    assert out["abstained"] is True and out["answer"] == "I don't know about that." and out["facts"] == []


def test_raw_mode_and_no_renderer_are_unchanged_by_the_flag(monkeypatch):
    for attr in ("raw_mode", "renderer"):
        outs = []
        for val in (None, "1"):
            if val:
                monkeypatch.setenv(FLAG, val)
            chat = _chat()
            if attr == "raw_mode":
                chat.raw_mode = True
            else:
                chat.renderer = None
            outs.append(_comp(chat).render_paragraph([list(DOG), list(CAT)]))
        assert outs[0] == outs[1] == ("dog chase cat cat eat fish", [DOG, CAT], [])


# ---- the mood marker
def test_the_mood_marker_is_a_declared_non_claim_added_after_the_check(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    for level, want in ((2, "the dog chases the cat!"), (-2, "the dog chases the cat."), (0, "the dog chases the cat.")):
        chat = _chat(level=level)
        spy = Spy(chat)
        assert chat.spiking_recall_surface(*DOG) == want, level
        assert spy.sets and all(s == "the dog chases the cat." for s, _g in spy.sets)       # the checker saw the body with "."
        para, kept, _d = _comp(_chat(level=level)).render_paragraph([list(DOG)])
        assert para == want and kept == [DOG]
    assert _chat(level=2).spiking_recall_parts(*DOG)[1] is True
    assert _chat(level=-2).spiking_recall_parts(*DOG)[1] is False                           # curt collapses to neutral (declared)
    monkeypatch.setenv("BRAIN_SPIKING_MOUTH_MOOD", "0")
    assert _chat(level=2).spiking_recall_surface(*DOG) == "the dog chases the cat."          # mood lesion: '.' only
    monkeypatch.delenv("BRAIN_SPIKING_MOUTH_MOOD")
    # the affect lesion still changes this surface (level 0 vs +2)
    assert _chat(level=0).spiking_recall_surface(*DOG) != _chat(level=2).spiking_recall_surface(*DOG)
    # the exemption is structural, not textual: a Qwen sentence ending '!' rejects
    monkeypatch.setenv("BRAIN_SPIKING_MOUTH_RECALL", "0")
    chat = _chat(Prose({tuple(DOG): "The dog chases the cat!"}, regen={tuple(DOG): "The dog chases the cat!"}))
    assert chat.render(list(DOG)).endswith("[unverified render -> spoke the brain's raw fact]")
    assert not chat._verify("The dog chases the cat!", None, DOG)


# ---- build-6 review (test-gap): the joined check on the BATCHED path (BRAIN_RICH_BATCH_RENDER, renderer with batch methods)
PROG_ = ["progressive_rock", "has_part", "band"]


def _batched(facts, table, level=0):
    chat = _chat(Prose(table, regen=table, batch=True), level=level)
    comp = _comp(chat)
    para, kept, dropped = comp.render_paragraph([list(f) for f in facts])
    return para, kept, dropped


def _whole_ok(para, facts):
    return A.allowlist_verify(para.replace("!", "."), [list(f) for f in facts])["accepted"]


def test_batched_path_joins_terminates_and_cuts(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("BRAIN_RICH_BATCH_RENDER", "1")
    monkeypatch.setenv("BRAIN_SPIKING_MOUTH_RECALL", "0")
    # unterminated Qwen sentences: no run-on, 'A. B.'
    para, kept, dropped = _batched([DOG, CAT], {tuple(DOG): "The dog chases the cat", tuple(CAT): "The cat eats the fish"})
    assert para == "The dog chases the cat. The cat eats the fish." and kept == [DOG, CAT] and dropped == [], para
    assert _whole_ok(para, [DOG, CAT])
    # a duplicate proposition: the later sentence is cut and listed in dropped
    f1, f2 = ["spain", "shares_border_with", "france"], ["spain", "shares_border_with", "portugal"]
    para, kept, dropped = _batched([f1, f2], {tuple(f1): "Spain shares border with France and Portugal.",
                                              tuple(f2): "Spain shares border with Portugal."})
    assert para == "Spain shares border with France and Portugal." and kept == [f1] and dropped == [f2], para
    # points back: the later sentence is cut
    g = [list(x) for x in G_WAKE]
    para, kept, dropped = _batched(g, {tuple(g[0]): WAKE1, tuple(g[1]): "The Band award received Grammy lifetime achievement award."})
    assert para == WAKE1 and kept == [g[0]] and dropped == [g[1]], para
    assert _whole_ok(para, g)


def test_batched_path_warm_spiking_marker_is_added_after_the_check(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("BRAIN_RICH_BATCH_RENDER", "1")
    para, kept, dropped = _batched([DOG, PROG_], {tuple(PROG_): "Progressive rock has part band"}, level=2)
    assert para == "the dog chases the cat! Progressive rock has part band." and kept == [DOG, PROG_] and dropped == [], para
    assert not A.allowlist_verify(para, [DOG, PROG_])["accepted"]          # the '!' is a declared non-claim, added after the check
    assert _whole_ok(para, [DOG, PROG_])


@pytest.mark.parametrize("seed", range(40))
def test_batched_and_sequential_paragraphs_pass_whole(monkeypatch, seed):
    """Property (the reviewer's join_e2e.py, compact): random fact orders, terminated or not, hostile or faithful prose, warm or not;
    every emitted paragraph (with '!' read back as '.') passes allowlist_verify against the gathered union, on both paths."""
    rnd = random.Random(seed)
    pool = [list(DOG), list(CAT), list(BRAIN), PROG_, list(G_WAKE[0]), list(G_WAKE[1]),
            ["spain", "shares_border_with", "france"], ["spain", "shares_border_with", "portugal"]]
    faithful = {tuple(DOG): "The dog chases the cat", tuple(CAT): "The cat eats the fish.", tuple(BRAIN): "The brain uses spikes",
                tuple(PROG_): "Progressive rock has part the band.", G_WAKE[0]: WAKE1,
                G_WAKE[1]: "The Band award received Grammy lifetime achievement award",
                ("spain", "shares_border_with", "france"): "Spain shares border with France and Portugal.",
                ("spain", "shares_border_with", "portugal"): "Spain shares border with Portugal"}
    facts = rnd.sample(pool, rnd.randint(2, 5))
    table = {k: (rnd.choice(HOSTILE) if rnd.random() < 0.2 else v) for k, v in faithful.items()}
    monkeypatch.setenv(FLAG, "1")
    if rnd.random() < 0.5:
        monkeypatch.setenv("BRAIN_SPIKING_MOUTH_RECALL", "0")
    level = rnd.choice((0, 2, -2))
    for batch in ("1", "0"):
        monkeypatch.setenv("BRAIN_RICH_BATCH_RENDER", batch)
        para, kept, dropped = _batched(facts, table, level=level)
        assert sorted(map(tuple, kept + dropped)) == sorted(map(tuple, facts))
        if para:
            assert _whole_ok(para, facts), (batch, para)
            assert para.endswith((".", "!")) and "  " not in para


# ---- the hypothesis path
FRAME = " -- that's a guess from what I've learned, not something I was taught."
SPK_SUFFIX = "  [a guess from what I've learned -- not something I was taught]"


def test_the_declared_frame_and_abstain_strings_are_pinned_verbatim():
    """A future edit that adds a claim to a declared non-claim string trips this test."""
    from research.runners.brain_chat_tui import ChatBrain
    assert ChatBrain._frame_guess("The dog eats the fish.") == "Maybe the dog eats the fish" + FRAME
    assert ChatBrain._frame_guess_spiking("perhaps the dog eats the fish") == "perhaps the dog eats the fish" + SPK_SUFFIX
    assert ChatBrain._hypothesis_template("dog", "eat", "fish") == "perhaps dog eat fish" + SPK_SUFFIX
    import inspect
    from research.runners import brain_chat_tui, rich_answer_composer
    src = inspect.getsource(brain_chat_tui) + inspect.getsource(rich_answer_composer)
    for s in ('"I don\'t know about that."', '"I don\'t have anything more to add on that."',
              '"   [unverified render -> spoke the brain\'s raw fact]"'):
        assert s in src, s


def test_hypothesis_qwen_branch_checks_the_spoken_body(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("BRAIN_SPIKING_MOUTH", "0")
    hyp = ["dog", "eat", "fish"]
    for prose, ok in (("The dog eats the fish.", True), ("the dog eats the fish", True),
                      ("The dog eats the fish and is famous for its vodka.", False), ("The fish eats the dog.", False),
                      ("The dog chases the cat.", False), ("Perhaps the dog eats the fish.", False)):
        chat = _chat(Prose({tuple(hyp): prose}, regen={tuple(hyp): prose}))
        surface, fluent = chat.render_hypothesis_verified(hyp)
        if ok:
            assert (surface, fluent) == ("Maybe the dog eats the fish" + FRAME, True), prose
        else:
            assert (surface, fluent) == ("perhaps dog eat fish" + SPK_SUFFIX, False), prose


def test_hypothesis_qwen_spoken_body_is_exactly_the_checked_body(monkeypatch):
    """Build-6 review (test-gap): the first-word case can flip the matcher verdict. 'Will Hurt award received Theatre World Awards.' is
    kept, but `_frame_guess` would speak 'will Hurt ...' (a modal-as-prose name, rejected). So the checked body must be the spoken body,
    and this surface falls to the raw flagged template."""
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("BRAIN_SPIKING_MOUTH", "0")
    hyp = ["will_hurt", "award_received", "theatre_world_awards"]
    prose = "Will Hurt award received Theatre World Awards."
    assert A.allowlist_verify(prose, [hyp])["accepted"]
    assert not A.allowlist_verify("will Hurt award received Theatre World Awards.", [hyp])["accepted"]
    chat = _chat(Prose({tuple(hyp): prose}, regen={tuple(hyp): prose}))
    seen = []
    orig = chat._verify
    chat._verify = lambda surface, asserted, svo: seen.append(surface) or orig(surface, asserted, svo)
    surface, fluent = chat.render_hypothesis_verified(hyp)
    assert (surface, fluent) == ("perhaps will_hurt award_received theatre_world_awards" + SPK_SUFFIX, False), surface
    assert seen and all(x == "will Hurt award received Theatre World Awards" for x in seen), seen
    # the checked body is the spoken body for every accepted candidate: capitalised first letter and a trailing '.' are equivalent
    hyp2 = ["dog", "eat", "fish"]
    for p2 in ("The dog eats the fish.", "The dog eats the fish", "the dog eats the fish.", "  The dog eats the fish.  "):
        chat = _chat(Prose({tuple(hyp2): p2}, regen={tuple(hyp2): p2}))
        seen = []
        orig = chat._verify
        chat._verify = lambda surface, asserted, svo, _o=orig: seen.append(surface) or _o(surface, asserted, svo)
        surface, fluent = chat.render_hypothesis_verified(hyp2)
        assert fluent and surface == "Maybe " + seen[-1] + FRAME == "Maybe the dog eats the fish" + FRAME, (p2, surface, seen)


class FakeBroca:
    def __init__(self, words):
        self.words = words

    def emit(self, frame, dctx):
        return list(self.words)


def test_hypothesis_spiking_branch_exempts_only_the_leading_hedge(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    hyp = ["dog", "eat", "fish"]
    for words, want in ((["perhaps", "the", "dog", "eats", "the", "fish"], "perhaps the dog eats the fish" + SPK_SUFFIX),
                        (["the", "dog", "perhaps", "eats", "the", "fish"], None),
                        (["perhaps", "the", "fish", "eats", "the", "dog"], None),
                        (["perhaps", "the", "dog", "eats", "the", "fish", "and", "the", "cat"], None)):
        chat = _chat()
        chat._spk_producer = FakeBroca(words)
        assert chat._render_hypothesis_spiking(hyp) == want, words
        surface, fluent = chat.render_hypothesis_verified(hyp)
        assert fluent is (want is not None) and (surface == want if want else surface == "perhaps dog eat fish" + SPK_SUFFIX)


# ---- Touchpoint A, ChatBrain.render, the one-brain articulate, the open-ended warning
def test_touchpoint_a_gets_the_allowlist_under_the_flag(monkeypatch):
    monkeypatch.setenv("BRAIN_TOUCHPOINT_A_FACT_CLAUSE", "1")
    monkeypatch.setenv("BRAIN_SPIKING_MOUTH_RECALL", "0")
    f = ["rick_wakeman", "member_of", "yes_band"]
    for val, fc, want_para, qwen_called in (
            ("1", "the rick_wakeman is a member of the yes_band", "the rick_wakeman is a member of the yes_band.", False),
            ("1", "the rick_wakeman is located in the yes_band.", "Rick Wakeman is a member of Yes band.", True),
            (None, "the rick_wakeman is located in the yes_band.", "the rick_wakeman is located in the yes_band.", False)):
        if val:
            monkeypatch.setenv(FLAG, val)
        else:
            monkeypatch.delenv(FLAG, raising=False)
        ren = Prose({tuple(f): "Rick Wakeman is a member of Yes band."})
        with H.fake_touchpoint_a({tuple(f): fc}):
            para, kept, _d = _comp(_chat(ren)).render_paragraph([f])
        assert para == want_para and kept == [f], (val, para)
        assert bool(ren.calls) is qwen_called, (val, ren.calls)


def test_chatbrain_render_under_the_flag(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("BRAIN_SPIKING_MOUTH_RECALL", "0")
    assert _chat(Prose({})).render(list(DOG)) == "The dog chases the cat."
    out = _chat(Prose({tuple(DOG): HOSTILE[0]}, regen={tuple(DOG): HOSTILE[0]})).render(list(DOG))
    assert out == "dog chase cat   [unverified render -> spoke the brain's raw fact]"
    # the stub's asserted SVO is never trusted over its prose
    from research.runners.brain_chat_tui import StubRenderer

    class Liar(StubRenderer):
        def render_svo(self, a, v, p):
            return "The %s %ss %s." % (p, v, a), [a, v, p]
    assert _chat(Liar()).render(list(DOG)).endswith("[unverified render -> spoke the brain's raw fact]")
    monkeypatch.delenv(FLAG)
    assert _chat(Liar()).render(list(DOG)) == "The cat chases dog."                         # flag off: the stock hole, as before


def test_one_brain_articulate_checks_the_substituted_reply(monkeypatch):
    from webapp import one_brain as OB
    import contextlib

    class Mouth:
        def __init__(self, table):
            self.table = table

        def generate(self, c):
            return self.table.get(tuple(c["svo"]))
    monkeypatch.setenv(FLAG, "1")
    chat = _chat()
    t1, t2 = "The dog chases cat.", "The cat eats fish."
    calls = [{"svo": DOG, "template": t1, "method": "render_svo", "manner": ""},
             {"svo": CAT, "template": t2, "method": "render_svo", "manner": ""}]
    verify = lambda prose, svo: bool(chat._verify(prose, None, svo))
    reply = lambda texts, svos: bool(chat._verify_claim_set(" ".join(texts), svos)[0])
    relock = contextlib.nullcontext
    # a second clause: the template stays
    ans, rec = OB.articulate(t1 + " " + t2, calls, Mouth({tuple(DOG): HOSTILE[0]}), verify, relock, verify_reply=reply)
    assert ans == t1 + " " + t2 and rec["n_verified"] == 0
    # a faithful sentence (unterminated: terminated): substituted
    ans, rec = OB.articulate(t1 + " " + t2, calls, Mouth({tuple(DOG): "The dog chases the cat", tuple(CAT): "The cat eats the fish."}),
                             verify, relock, verify_reply=reply)
    assert ans == "The dog chases the cat. The cat eats the fish." and rec["n_verified"] == 2
    # each prose alone faithful, but the joined reply repeats a proposition: the later substitution keeps its template
    calls2 = [{"svo": ["spain", "shares_border_with", "france"], "template": "The spain shares_border_withs france.",
               "method": "render_svo", "manner": ""},
              {"svo": ["spain", "shares_border_with", "portugal"], "template": "The spain shares_border_withs portugal.",
               "method": "render_svo", "manner": ""}]
    m = Mouth({("spain", "shares_border_with", "france"): "Spain shares border with France.",
               ("spain", "shares_border_with", "portugal"): "Spain shares border with France."})
    ans, rec = OB.articulate("The spain shares_border_withs france. The spain shares_border_withs portugal.", calls2, m,
                             lambda prose, svo: True, relock, verify_reply=reply)
    assert ans == "Spain shares border with France. The spain shares_border_withs portugal." and rec["n_reply_rejected"] == 1
    # without verify_reply: exactly as before
    ans, rec = OB.articulate(t1 + " " + t2, calls, Mouth({tuple(DOG): HOSTILE[0]}), lambda prose, svo: True, relock)
    assert ans.startswith(HOSTILE[0]) and "n_reply_rejected" not in rec


# build-6 review fix (test-gap): the one-brain substitution re-checks the WHOLE mouth paragraph, not only the deferred-mouth calls
W_YES = ["rick_wakeman", "member_of", "yes_band"]
PROG = ["progressive_rock", "has_part", "band"]


class _ScriptedMouth:
    def __init__(self, table):
        self.table = table

    def generate(self, c):
        return self.table.get(tuple(c["svo"]))


def test_one_brain_articulate_checks_the_whole_mouth_paragraph(monkeypatch):
    """A non-call mouth sentence (the spiking recall surface, Touchpoint A) before a call: a substitution that adds a 'the' pointing back
    to it ('Progressive rock has part the band.' after '... Yes band.') passes alone and against the calls alone, but not whole."""
    from webapp import one_brain as OB
    from research.runners.brain_chat_tui import StubRenderer
    import contextlib
    monkeypatch.setenv(FLAG, "1")
    chat = _chat()
    tmpl = StubRenderer().render_svo(*PROG)[0]
    assert tmpl == "The progressive_rock has_parts band."
    call = {"svo": PROG, "template": tmpl, "method": "render_svo", "manner": ""}
    verify = lambda prose, svo: bool(chat._verify(prose, None, svo))
    reply = lambda texts, svos: bool(chat._verify_claim_set(" ".join(texts), svos)[0])
    relock = contextlib.nullcontext
    bad, good = "Progressive rock has part the band.", "Progressive rock has part band."
    assert verify(bad, PROG) and verify(good, PROG)                                   # each prose alone is faithful
    assert A.allowlist_verify(WAKE1 + " " + tmpl, [W_YES, PROG])["accepted"]       # the template paragraph passes whole
    assert not A.allowlist_verify(WAKE1 + " " + bad, [W_YES, PROG])["accepted"]    # the substituted one does not
    for lead, tail in (("", ""), ("Someone told me: ", " (Someone told me part of this.)")):
        answer = lead + WAKE1 + " " + tmpl + tail
        replies = [{"spoken": [WAKE1, tmpl], "checked": [WAKE1, tmpl], "gated": [W_YES, PROG]}]
        ans, rec = OB.articulate(answer, [dict(call)], _ScriptedMouth({tuple(PROG): bad}), verify, relock, verify_reply=reply,
                                 replies=replies)
        assert ans == answer and rec["n_verified"] == 0 and rec["n_reply_rejected"] == 1, ans
        ans, rec = OB.articulate(answer, [dict(call)], _ScriptedMouth({tuple(PROG): good}), verify, relock, verify_reply=reply,
                                 replies=replies)
        assert ans == lead + WAKE1 + " " + good + tail and rec["n_verified"] == 1, ans
        # what is emitted passes whole
        assert A.allowlist_verify(ans[len(lead):len(ans) - len(tail)], [W_YES, PROG])["accepted"]
    # without the record only the call texts are checked: the substitution the record catches goes through (why the record exists)
    ans, _rec = OB.articulate(WAKE1 + " " + tmpl, [dict(call)], _ScriptedMouth({tuple(PROG): bad}), verify, relock, verify_reply=reply)
    assert ans == WAKE1 + " " + bad


def test_composer_records_its_paragraph_for_the_one_brain_mouth(monkeypatch):
    """End to end on the real composer: a warm spiking sentence (non-call, spoken '!', checked '.') before a deferred-mouth call. The
    record carries both forms and the gathered union; the substitution lands at the call's position and the '!' stays."""
    from webapp import one_brain as OB
    import contextlib
    monkeypatch.setenv(FLAG, "1")
    mouth = OB.DeferredMouth(Prose({}))
    chat = _chat(mouth, level=2)
    comp = _comp(chat)
    para, kept, dropped = comp.render_paragraph([list(DOG), list(PROG)])
    assert not hasattr(chat, "_moat_joined_replies")                                   # no record open: nothing stored
    mouth.take_calls()
    OB.begin_joined_record(chat)
    para, kept, dropped = comp.render_paragraph([list(DOG), list(PROG)])
    rec_ = OB.take_joined_record(chat)
    assert not hasattr(chat, "_moat_joined_replies") and OB.take_joined_record(chat) == []
    tmpl = "The progressive_rock has_parts band."
    assert para == "the dog chases the cat! " + tmpl and kept == [DOG, PROG] and dropped == [], para
    assert rec_ == [{"spoken": ["the dog chases the cat!", tmpl], "checked": ["the dog chases the cat.", tmpl],
                     "gated": [DOG, PROG, DOG, PROG]}]
    calls = mouth.take_calls()
    assert [c["svo"] for c in calls] == [PROG]                                        # the spiking sentence is not a call
    verify = lambda prose, svo: bool(chat._verify(prose, None, svo))
    reply = lambda texts, svos: bool(chat._verify_claim_set(" ".join(texts), svos)[0])
    seen = []
    ans, rec = OB.articulate(para, calls, _ScriptedMouth({tuple(PROG): "Progressive rock has part band"}), verify,
                             contextlib.nullcontext, verify_reply=lambda t, s: seen.append(list(t)) or reply(t, s), replies=rec_)
    assert ans == "the dog chases the cat! Progressive rock has part band." and rec["n_verified"] == 1, ans
    assert seen == [["the dog chases the cat.", "Progressive rock has part band."]]    # the whole paragraph, in its checked form


def test_open_ended_is_warned_out_of_scope_at_startup():
    from webapp.server import moat_allowlist_scope_warning
    assert moat_allowlist_scope_warning({}) is None
    assert moat_allowlist_scope_warning({FLAG: "1"}) is None
    assert moat_allowlist_scope_warning({"BRAIN_OPEN_ENDED": "1"}) is None
    assert "BRAIN_OPEN_ENDED" in moat_allowlist_scope_warning({FLAG: "1", "BRAIN_OPEN_ENDED": "yes"})
    # build-6 review (doc): BRAIN_OPEN_ENDED_GATED replaces the composer's answer after the allowlist too
    assert moat_allowlist_scope_warning({"BRAIN_OPEN_ENDED_GATED": "1"}) is None
    w = moat_allowlist_scope_warning({FLAG: "1", "BRAIN_OPEN_ENDED_GATED": "on"})
    assert w and "BRAIN_OPEN_ENDED_GATED" in w and "BRAIN_OPEN_ENDED and" not in w
    w = moat_allowlist_scope_warning({FLAG: "1", "BRAIN_OPEN_ENDED": "1", "BRAIN_OPEN_ENDED_GATED": "1"})
    assert "BRAIN_OPEN_ENDED and BRAIN_OPEN_ENDED_GATED" in w


# ---- the census: every mouth call in the two modules sits in a registered function
CENSUS = {
    # brain_chat_tui.py
    ("ChatBrain", "render"): "C10: spiking (C2) then render_svo / regen through the _verify chokepoint; raw fallback is the brain's triple",
    ("ChatBrain", "render_hypothesis_verified"): "C7: Qwen branch, the spoken body through the chokepoint",
    ("ChatBrain", "_render_hypothesis_spiking"): "C7: spiking branch, the body after the declared hedge through the chokepoint",
    ("ChatBrain", "_spiking_broca_producer"): "builds the hypothesis producer (no text)",
    ("ChatBrain", "spiking_recall_surface"): "C2: flag ON routes through spiking_recall_parts",
    ("ChatBrain", "_spiking_recall_checked"): "C2: the allowlist over the terminated sentence",
    ("ChatBrain", "spiking_recall_parts"): "C2: checked text + structural mood marker",
    ("ChatBrain", "_spiking_recall_mouth"): "builds the recall mouth (no text)",
    ("StubRenderer", "render_svo"): "the stub mouth itself",
    ("QwenRenderer", "render_svo"): "the Qwen mouth itself",
    ("QwenRenderer", "render_svo_regen"): "the Qwen mouth itself",
    ("QwenRenderer", "render_svo_batch"): "the Qwen mouth itself",
    ("QwenRenderer", "render_svo_regen_batch"): "the Qwen mouth itself",
    # rich_answer_composer.py
    ("RichAnswerComposer", "render_paragraph"): "C5/C6: per-fact allowlist then the joined check",
    ("RichAnswerComposer", "_render_paragraph_batched"): "C3: spiking parts, batch + regen via _verify_rendered, joined check",
    ("RichAnswerComposer", "_render_one_verified_ex"): "C2/C4/C5: spiking parts, Touchpoint A and Qwen/stub via _verify_rendered",
    ("_ConfabOneRenderer", "render_svo"): "the composer smoke's adversarial confabulating mouth (a test renderer, not production)",
}
MOUTH_CALLS = {"render_svo", "render_svo_batch", "render_svo_regen", "render_svo_regen_batch", "render_fact_sentence",
               "_spiking_recall_mouth", "_spiking_broca_producer", "emit", "spiking_recall_surface", "spiking_recall_parts", "render"}


# the webapp side (build-6 review): every mouth / generator call in webapp/*.py sits in a registered function with its status
WEBAPP_MOUTH_CALLS = {"render_svo", "render_svo_batch", "render_svo_regen", "render_svo_regen_batch", "render_svo_manner",
                      "render_fact_sentence", "get_generator", "get_warm_faculty"}
WEBAPP_CENSUS = {
    ("one_brain.py", "DeferredMouth._record"): "the template articulation (StubRenderer), verified inside the turn like any stub",
    ("one_brain.py", "DeferredMouth.generate"): "BRAIN_ONE_BRAIN deferred mouth: substituted only through articulate's verify + joined check",
    ("open_ended_chat.py", "answer_turn"): "OUT OF SCOPE: BRAIN_OPEN_ENDED (startup warning)",
    ("wkv_mouth_generator.py", "generate._run"): "OUT OF SCOPE: reached only from BRAIN_OPEN_ENDED's answer_turn",
    ("open_ended_gated_turn.py", "ltm_render"): "OUT OF SCOPE: BRAIN_OPEN_ENDED_GATED (startup warning)",
    ("open_ended_gated_turn.py", "offkb_render"): "OUT OF SCOPE: BRAIN_OPEN_ENDED_GATED (startup warning)",
}


def test_every_webapp_mouth_call_is_in_the_census():
    import ast
    from webapp.server import MOAT_ALLOWLIST_OUT_OF_SCOPE
    root = Path(__file__).resolve().parents[1] / "webapp"
    found = set()

    def walk(fname, node, stack):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(fname, ch, stack + [ch.name])
                continue
            if isinstance(ch, ast.Call):
                f = ch.func
                name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
                if name in WEBAPP_MOUTH_CALLS:
                    found.add((fname, ".".join(stack)))
            walk(fname, ch, stack)
    for path in sorted(root.glob("*.py")):
        walk(path.name, ast.parse(path.read_text(encoding="utf-8")), [])
    unregistered = sorted(found - set(WEBAPP_CENSUS))
    assert not unregistered, "webapp mouth calls outside the census (register them with their flag-on status): %s" % unregistered
    for (_f, _fn), status in WEBAPP_CENSUS.items():
        if status.startswith("OUT OF SCOPE"):
            assert any(k in status for k in MOAT_ALLOWLIST_OUT_OF_SCOPE), status


def test_every_mouth_call_is_in_the_census():
    """Fails loudly when a new mouth call appears outside a registered function (a new path the flag would not own)."""
    import ast
    import inspect
    from research.runners import brain_chat_tui, rich_answer_composer
    found = set()
    for mod in (brain_chat_tui, rich_answer_composer):
        tree = ast.parse(inspect.getsource(mod))
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
                for node in ast.walk(fn):
                    if isinstance(node, ast.Call):
                        f = node.func
                        name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
                        if name in MOUTH_CALLS and not (name == "render" and cls.name in ("ChatBrain",) and fn.name == "answer"):
                            found.add((cls.name, fn.name))
    unregistered = sorted(found - set(CENSUS) - {("ChatBrain", "answer"), ("ChatBrain", "render_hypothesis")})
    assert not unregistered, "mouth calls outside the census (register them with their flag-on check): %s" % unregistered
