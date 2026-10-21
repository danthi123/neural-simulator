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


def test_the_build6_name_rules_only_add_rejections():
    """Monotone: a sentence build 6 keeps parses as before and passes every build-5 check (the new checks run after them)."""
    import inspect
    src = inspect.getsource(A.allowlist_check)
    assert src.index("_structure_word_violation(parses") < src.index("_name_reading_violation(parses")
    assert "_points_back(trace)" in inspect.getsource(A.allowlist_verify)
