"""Behavioural pins from the mutation test of the strict allowlist claim matcher (`BRAIN_MOAT_ALLOWLIST`, default OFF).

The mutation test of build 1 (/home/dant123/Projects/sim-rev/moat-allow/mut/REPORT.md, mutation_report.json: 159 hand-written mutants of
research/runners/_moat_allowlist.py and _moat_allowlist_relations.py) found 40 survivors of tests/test_moat_allowlist.py that are real gaps:
an input exists on which the mutant's verdict, output field or exception differs from the original. Each test below pins the ORIGINAL
(correct) behaviour on the report's witness input, so a regression of that kind now fails. The INPUTS are ported, not the mutants: they are
behavioural and hold on build 3's code. One verdict changed legitimately since build 1 and is pinned with its reason: "A B CC borders Z."
(m32's witness) is rejected by build 3's case rule (CC is all-caps and b_cc is not a store entity), so the slug-order pin uses "A B Cc".
One behaviour was a crash in the original and is now a clean reject (m96: a malformed gathered frozenset).
Build 4 (fold to find, exact to accept): four witnesses keep an accented span of a slug that is no store entity (m23, m16, m111). They pin
how the fold FINDS a span, so they run with those names added to the names table as if they were store names (`WITNESS_NAMES`); without
them, build 4 rejects each with the span named (test_build4_rejects_the_fold_witnesses_without_a_store_name).

Run: CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m pytest tests/test_moat_allowlist_witnesses.py -q
"""
from __future__ import annotations

import json
import os
from collections import Counter

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest

from research.runners import _moat_allowlist as A
from research.runners import _moat_allowlist_relations as T
from research.runners.brain_conversational_agent import BridgeParser

pytestmark = pytest.mark.filterwarnings("ignore")

SEEDS = (42, 43, 44, 100, 101, 102, 1)
FLAG = "BRAIN_MOAT_ALLOWLIST"
ES = [["spain", "shares_border_with", "france"], ["spain", "shares_border_with", "portugal"],
      ["spain", "member_of", "european_union"], ["portugal", "member_of", "european_union"]]
HN = [["ha_noi_vietnam", "twinned_administrative_body", "montr_al_canada"]]
INYO = [["county_of_inyo", "country", "u_s_of_a"], ["spain", "country", "france"]]
S = "Spain shares border with France."


@pytest.fixture(autouse=True)
def _clean_flags(monkeypatch):
    for k in (FLAG, "BRAIN_MOAT_SLUG_NORM", "BRAIN_CLAIM_MOAT"):
        monkeypatch.delenv(k, raising=False)


# build 4: hypothetical store names of the fictional slugs some witnesses keep (see the module docstring)
WITNESS_NAMES = {"rue_mile": ["Rue Émile"], "montr_cole": ["Montr École"], "mo_x": ["Moﬁ X"], "x1_y": ["X1é Y"]}


@pytest.fixture(autouse=True)
def _witness_names(request, monkeypatch):
    if request.node.get_closest_marker("no_witness_names"):
        return
    t = A._NamesTable(A.NAMES_TABLE_PATH)
    assert t.error is None, t.error
    for slug, names in WITNESS_NAMES.items():
        t.entities[slug] = t.entities.get(slug, frozenset()) | frozenset(A.surface_form(n) for n in names)
    monkeypatch.setattr(A, "names_table", lambda: t)


@pytest.mark.no_witness_names
@pytest.mark.parametrize("sentence,slug", [("Rue Émile borders Canada.", "rue_mile"), ("Montr École borders Canada.", "montr_cole"),
                                           ("Moﬁ X borders Y.", "mo_x"), ("X1é Y borders Canada.", "x1_y")])
def test_build4_rejects_the_fold_witnesses_without_a_store_name(sentence, slug):
    g = [[slug, "borders", "canada" if "Canada" in sentence else "y"]]
    r = A.allowlist_check(sentence, g)
    assert not r["keep"] and ("entity span %r" % sentence.split(" borders")[0]) in r["reason"] and slug in r["reason"], r["reason"]


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
            out.append(seed_chats[sd]._verify_claim_set(surface, gathered)[0])
        return out
    finally:
        os.environ.pop(FLAG, None)


# (witness id(s), sentence, gathered, keep, shape of the result or None, props of a keep or None). keep is the verdict of allowlist_check
# on the sentence AND of allowlist_verify on it as a surface.
WITNESSES = [
    # ---- high: a leak if the code regressed
    ("m130", "Spain county of Inyo France.", INYO, False, None, None),                          # board #266 3-letter prefix collision
    ("m61", "Spain shares border with France, or Portugal.", ES, False, None, None),           # ', or' is not ', and'
    ("m62", "Spain shares border with France or Portugal.", ES, False, None, None),            # 'or' is not 'and'
    ("m78", "Spain shares border with the European Union and France.", ES, False, "shape2", None),   # membership on EVERY conjunct
    ("m78", "Spain shares border with European Union, Portugal, and France.", ES, False, "shape2", None),
    ("m91 m06 m07 m90 m112", "Ha Noi Vietnam's twinned administrative body Montréal Canada.", HN, False, None, None),
    ("m91 m07", "Ha Noi Vietnam’s twinned administrative body Montréal Canada.", HN, False, None, None),
    ("m08", "Ha Noi Vietnam`s twinned administrative body Montréal Canada.", HN, False, None, None),
    # ---- medium
    ("m13", "Spain shares border with France.Spain shares border with Portugal.", ES, False, None, None),   # no space after '.'
    ("m60", "Spain shares border with the the France.", ES, False, None, None),                # double determiner
    ("m60", "The the Spain shares border with France.", ES, False, None, None),
    ("m93", "Spain shares border with France..", ES, False, None, None),                       # a second period
    ("m93", "Spain shares border with France...", ES, False, None, None),
    ("m11", "France founded in 1904.", [["france", "founded_in", "1904"], ["france", "instance_of", "national_team"],
                                        ["france", "participant_of", "world_championship_football_2010"]], True, "shape1",
     [["france", "founded_in", "1904"]]),                                                      # a digit-only slug is a noun
    ("m23", "Rue Émile borders Canada.", [["rue_mile", "borders", "canada"]], True, "shape1", [["rue_mile", "borders", "canada"]]),
    ("m23", "Montr École borders Canada.", [["montr_cole", "borders", "canada"]], True, "shape1",
     [["montr_cole", "borders", "canada"]]),                                                   # a word-initial accented letter
    ("m32", "A B Cc borders Z.", [["a_b", "borders", "z"], ["b_cc", "borders", "z"]], True, "shape1",
     [["b_cc", "borders", "z"]]),                                                              # longest slug first: Det + b_cc
    ("t_drop_chase", "The cat chases the dog.", [["cat", "chase", "dog"]], True, "shape1", [["cat", "chase", "dog"]]),
    ("t_drop_uses", "The brain uses spikes.", [["brain", "uses", "spikes"]], True, "shape1", [["brain", "uses", "spikes"]]),
    ("m57", "Spain shares border with the.", ES, False, None, None),                           # crashes: a clean reject instead
    ("m57", "The.", ES, False, None, None),
    ("m72", "Spain is.", ES, False, None, None),
    ("m109", "Spain shares border with Canadá", ES, False, None, None),
    # ---- low
    ("m16", "ſpain shares border with France.", ES, False, None, None),                    # long s is not s (no NFKC)
    ("m16", "Moﬁ X borders Y.", [["mo_x", "borders", "y"]], True, "shape1", [["mo_x", "borders", "y"]]),
    ("m16", "Moﬁ X borders Y.", [["mofi_x", "borders", "y"]], False, None, None),
    ("m17", "Montré不al borders Canada.", [["montr_al", "borders", "canada"]], False, None, None),   # mixed Latin + CJK run
    ("m25", "Montr\U00017000al borders Canada.", [["montr_al", "borders", "canada"]], False, None, None),    # a nameless code point
    ("m64", "Spain shares border with France and Portugal and Andorra.", ES + [["spain", "shares_border_with", "andorra"]], False,
     None, None),
    ("m83", "The is member of x.", [["the", "member_of", "x"], ["is", "borders", "x"]], False, "shape3", None),   # every parse
    ("m85", "The is member of x.", [["the", "member_of", "x"], ["is", "member_of", "x"]], True, "shape3",
     [["the", "member_of", "x"]]),                                                              # the FIRST parse is reported
    ("m48", S, [["spain", "shares_border_with", "france"], ["x", "y", "z", "spain_shares_border_with_france"]], True, "shape1",
     [["spain", "shares_border_with", "france"]]),                                              # a 4-element entry is ignored
    ("m48", S, [["spain", "shares_border_with", "france", "x"]], False, None, None),
    ("m49", "A borders B.", [["a", "borders", "b"], ["c", "borders", None]], True, "shape1", [["a", "borders", "b"]]),
    ("m50", "A borders B.", [["a", "borders", "b"], 5], True, "shape1", [["a", "borders", "b"]]),
    ("m111", "X1é Y borders Canada.", [["x1_y", "borders", "canada"]], True, "shape1", [["x1_y", "borders", "canada"]]),
    ("m115", "Portugal shares border with France.", ES, False, "shape1", None),                 # the shape of a not-gathered reject
    ("m76", S, ES, True, "shape1", [["spain", "shares_border_with", "france"]]),
    ("m76", "Spain shares border with France and Portugal.", ES, True, "shape2",
     [["spain", "shares_border_with", "france"], ["spain", "shares_border_with", "portugal"]]),
]


@pytest.mark.parametrize("wid,sentence,gathered,keep,shape,props", WITNESSES, ids=["%s:%s" % (w[0].split()[0], w[1][:40]) for w in WITNESSES])
def test_the_original_verdict_on_each_mutation_witness(wid, sentence, gathered, keep, shape, props):
    r = A.allowlist_check(sentence, gathered)                     # never an exception
    assert r["keep"] is keep, (wid, r["reason"])
    assert r["shape"] == shape, (wid, r["shape"], r["reason"])
    if keep:
        assert r["props"] == props and r["reason"] == "ok", (wid, r)
    else:
        assert r["props"] == [] and r["reason"].startswith("allowlist:"), (wid, r)
    v = A.allowlist_verify(sentence, gathered)
    assert v["accepted"] is keep, (wid, v["reject_reason"])
    assert A.allowlist_keep(sentence, gathered) == (keep, r["reason"])


@pytest.mark.parametrize("sentence,gathered", [
    ("Spain county of Inyo France.", INYO), ("Spain shares border with France, or Portugal.", ES),
    ("Spain shares border with France or Portugal.", ES), ("Spain shares border with the European Union and France.", ES),
    ("Ha Noi Vietnam's twinned administrative body Montréal Canada.", HN),
    ("Ha Noi Vietnam’s twinned administrative body Montréal Canada.", HN),
    ("Spain shares border with France.Spain shares border with Portugal.", ES)])
def test_the_high_severity_witnesses_are_rejected_through_the_production_gate_at_all_seven_seeds(seed_chats, sentence, gathered):
    assert _gate_all_seeds(seed_chats, sentence, gathered) == [False] * len(SEEDS)


def test_the_slug_order_witness_is_now_held_to_the_case_rule():
    """m32's witness "A B CC borders Z." was KEPT by build 1 (Det + b_cc). Build 3 rejects it: "CC" is written in capitals and b_cc is
    not a store entity, so no store name allows it (controller ruling 2026-10-20). The slug-order coverage moved to "A B Cc" above."""
    keep, why = A.allowlist_keep("A B CC borders Z.", [["a_b", "borders", "z"], ["b_cc", "borders", "z"]])
    assert not keep and "'CC'" in why and "b_cc" in why, why


# =========================================================================================================== malformed input: clean rejects
@pytest.mark.parametrize("sentence", [None, 5, "", "   ", b"Spain shares border with France."])
def test_a_non_string_or_empty_sentence_is_a_clean_reject(sentence):
    r = A.allowlist_check(sentence, ES)                            # m89 m123 m88: never an exception, never a keep
    assert r["keep"] is False and r["reason"] == "allowlist: empty sentence", r
    assert A.allowlist_keep(sentence, ES) == (False, "allowlist: empty sentence")


@pytest.mark.parametrize("surface", [None, 5, "", "   "])
def test_a_non_string_or_empty_surface_is_a_clean_reject(surface):
    v = A.allowlist_verify(surface, ES)                            # m53
    assert v["accepted"] is False and v["reject_reason"] == "allowlist: empty surface" and v["grounded"] == []


@pytest.mark.parametrize("gathered", [None, [], (), [["spain", "shares_border_with"]], [5], [None], "abc",
                                      frozenset({("a", "r")}), frozenset({("a", "r", "b", "c")}), frozenset({(1, 2, 3)}),
                                      frozenset({"abc"})])
def test_malformed_gathered_input_is_a_clean_reject(gathered):
    """m113 (None), and m96: a malformed frozenset passed directly used to raise IndexError; it is filtered like any other input."""
    r = A.allowlist_check(S, gathered)
    assert r["keep"] is False and r["reason"] == "allowlist: nothing gathered", r
    assert A.allowlist_verify(S, gathered)["accepted"] is False
    assert A.AllowlistClaimVerifier(gathered).verify(S)["accepted"] is False


def test_a_frozenset_mixing_good_and_malformed_entries_keeps_only_the_good_ones():
    g = frozenset({("spain", "shares_border_with", "france"), ("a", "r")})
    assert A.allowlist_keep(S, g) == (True, "ok")
    assert A.allowlist_keep("Spain shares border with Portugal.", g)[0] is False


def test_a_one_shot_iterable_of_facts_is_read_once():
    """m102 / m114: allowlist_verify and AllowlistClaimVerifier filter the gathered input ONCE into a frozenset key."""
    assert A.allowlist_verify(S + " Spain is a member of the European Union.", iter(ES))["accepted"] is True
    v = A.AllowlistClaimVerifier(iter(ES))
    assert isinstance(v.key, frozenset) and v.key == A._key_of(ES)
    assert [v.verify(S)["accepted"] for _ in range(3)] == [True, True, True]


def test_the_verifier_class_markers():
    """m103 / m104: the class attributes ChatBrain and the composer read to tell the verifiers apart."""
    assert A.AllowlistClaimVerifier.allowlist is True and A.AllowlistClaimVerifier.slug_norm is False
    v = A.AllowlistClaimVerifier(ES)
    assert v.allowlist is True and v.slug_norm is False and v.verify(S)["allowlist"] is True


# =========================================================================================================== the relation table
# every row of the reviewed table, by source (deleting a row deleted its own parametrised test before: t_drop_uses, t_drop_chase)
ROWS = {
    "self_knowledge": ["abstains", "adds", "admits", "answers", "assists", "binds", "carry", "checks", "comes", "consolidates", "controls",
                       "develops", "fires", "form", "grows", "has", "hold", "learns", "persists", "phrases", "prevents", "reads",
                       "refuses", "remembers", "replays", "retrieves", "runs", "send", "simulates", "stores", "strengthens", "teaches",
                       "uses"],
    "leak_hunt": ["border", "borders", "capital", "founded_in", "industry", "located_in"],
    "tiny_demo": ["chase", "eat", "learn", "store", "use"],
}
# n_store of every store relation in wikidata_100k (sums to its 78,857 facts): pinned here so a wrong n_store fails on a node WITHOUT the
# store file too (t_nstore_nostore); where the store is present it is re-counted from it
N_STORE = {"author": 27, "award_received": 1556, "cast_member": 200, "child": 81, "composer": 140, "conflict": 280,
           "contains_administrative_territorial_enti": 410, "country": 12743, "country_of_citizenship": 3151, "country_of_origin": 538,
           "described_by_source": 2024, "director": 41, "educated_at": 1156, "employer": 154, "family_name": 428, "father": 125,
           "followed_by": 413, "follows": 417, "genre": 1250, "given_name": 2032, "has_part": 256, "headquarters_location": 2649,
           "instance_of": 13562, "instrument": 410, "is_a_list_of": 58, "iucn_conservation_status": 5, "language_of_work_or_name": 1128,
           "languages_spoken_written_or_signed": 1548, "located_in_the_administrative_territoria": 4442, "located_in_time_zone": 3647,
           "manner_of_death": 929, "member_of": 1168, "member_of_political_party": 270, "member_of_sports_team": 14,
           "military_branch": 206, "named_after": 399, "occupation": 2981, "operator": 89, "original_language_of_film_or_tv_show": 281,
           "parent_taxon": 182, "part_of": 610, "participant_of": 195, "performer": 7, "place_of_birth": 2198, "place_of_death": 1506,
           "platform": 21, "position_held": 105, "position_played_on_team_speciality": 10, "producer": 73, "publisher": 14,
           "record_label": 548, "screenwriter": 17, "shares_border_with": 2351, "sibling": 138, "sport": 4116, "spouse": 202,
           "subclass_of": 2004, "taxon_rank": 721, "twinned_administrative_body": 2437, "work_location": 194}
STORE_FACTS = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json"


def test_every_row_of_the_reviewed_table_exists_with_its_source():
    got = {}
    for rel, row in T.RELATIONS.items():
        for src in row.sources:
            got.setdefault(src, set()).add(rel)
    for src, rels in ROWS.items():
        assert got.get(src, set()) == set(rels), (src, sorted(set(rels) ^ got.get(src, set())))
    assert got.get("store", set()) == set(N_STORE)
    assert len(T.RELATIONS) == sum(len(v) for v in ROWS.values()) + len(N_STORE)


def test_the_store_path_and_every_n_store_are_pinned():
    """t_storepath: the store path the inventory check reads is the real one (a wrong path used to make the check skip silently);
    t_nstore_nostore: every row's n_store is pinned here, so a wrong one fails with or without the store file."""
    assert T.STORE_FACTS == STORE_FACTS
    assert {r: row.n_store for r, row in T.RELATIONS.items() if row.n_store} == N_STORE
    assert sum(N_STORE.values()) == 78857
    assert all(T.RELATIONS[r].n_store == 0 for rels in ROWS.values() for r in rels if "store" not in T.RELATIONS[r].sources)


def test_the_pinned_n_store_matches_the_store_file():
    if not os.path.exists(STORE_FACTS):
        pytest.skip("LOUD SKIP: the wikidata_100k store file %s is absent on this node, so the pinned N_STORE cannot be re-counted "
                    "from it (the pin itself still ran in test_the_store_path_and_every_n_store_are_pinned)" % STORE_FACTS)
    with open(STORE_FACTS, "r", encoding="utf-8") as fh:
        n = Counter(r["fact"]["action"] for r in json.load(fh))
    assert dict(n) == N_STORE
