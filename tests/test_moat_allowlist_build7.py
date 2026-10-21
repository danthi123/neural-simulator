"""Pins for build 7 of the strict allowlist claim checker (`BRAIN_MOAT_ALLOWLIST`, default OFF): two narrow name rules that close the remaining
first-person (honesty-boundary) leaks of the round-8 blind hunt (Claude rulings 2026-10-21; spec docs/drafts/moat-allowlist/SPEC.md, the
build-7 bullet; rules in the comment block above `PRONOUN_NAME_CLASS` in research/runners/_moat_allowlist.py).

  R7a. A ONE-word SUBJ/OBJ span whose word, case-folded, is in a closed pronoun class rejects when it is written in lower case OR is the
       sentence's first word, unless a determiner stands directly before it or it is written all-caps with 2+ letters. It runs before the
       slug-verbatim skip.
  R7b. In a spaced span (2+ words, not the slug verbatim) a word "i" / "I" at position k > 0 reads as the first-person pronoun when
       (1) it is lower-case "i" and no word of the span starts with an upper-case letter; (2) it is "i" OR "I" (either case, widened by the
       coordinator's ruling of 2026-10-21), the next word starts with a lower-case letter and is neither "of" nor a determiner; (3) the word
       directly before it, case-folded, is and / or / nor / but / "&".

The matcher is pure: its verdict cannot depend on the parser seed. Where a test goes through the production `ChatBrain` (a seeded
BridgeParser) it runs at all seven parser seeds 42, 43, 44, 100, 101, 102, 1, as the build-6 tests do. The ten round-8 cases are copied
from research hunt round 8 (`_moat_allowlist_hunt4/r8/cases.json`, class `taught`: invented slugs that follow the store's `sanitize` rule),
all kept by build 6 at every seed.

Run: CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m pytest tests/test_moat_allowlist_build7.py -q
"""
from __future__ import annotations

import inspect
import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest

from research.runners import _moat_allowlist as A
from research.runners.brain_conversational_agent import BridgeParser

pytestmark = pytest.mark.filterwarnings("ignore")

SEEDS = (42, 43, 44, 100, 101, 102, 1)
FLAG = "BRAIN_MOAT_ALLOWLIST"
STORE_FACTS = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json"


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
        return chat._verify_claim_set(surface, [list(g) for g in gathered])
    finally:
        os.environ.pop(FLAG, None)


def verdict(sentence, gathered):
    return A.allowlist_verify(sentence, [list(g) for g in gathered])


def keep(sentence, gathered):
    return verdict(sentence, gathered)["accepted"]


def _noop(*_a, **_k):
    return None


# =============================================================================================================== the ten round-8 leaks
# id -> (reply, gathered, the rule that rejects it, a fragment of its reason). R8-TA15 / TA16 / TA17 are two-sentence replies.
LEAKS = {
    "R8-TA01": ("Reginald K Dwight is an author of me.", [("me", "author", "reginald_k_dwight")], "R7a", "reads as the pronoun 'me'"),
    "R8-TA04": ("Taylor Alison Swift is an author of mine.", [("mine", "author", "taylor_alison_swift")], "R7a",
                "reads as the pronoun 'mine'"),
    "R8-TA07": ("Canada portal shares border with us.", [("canada_portal", "shares_border_with", "us")], "R7a", "reads as the pronoun 'us'"),
    "R8-TA15": ("Stoker Bram is an author of Carfax Abbey. It is an instance of theatrical film.",
                [("carfax_abbey", "author", "stoker_bram"), ("it", "instance_of", "theatrical_film")], "R7a",
                "reads as the pronoun 'It' (the first word of the sentence"),
    "R8-TA16": ("Scarlett Johansson place of birth New York New York. Her original language of film or TV show number of English words.",
                [("scarlett_johansson", "place_of_birth", "new_york_new_york"),
                 ("her", "original_language_of_film_or_tv_show", "number_of_english_words")], "R7a",
                "reads as the pronoun 'Her' (the first word of the sentence"),
    "R8-TA09": ("honey i shrunk the kids is an instance of theatrical film.",
                [("honey_i_shrunk_the_kids", "instance_of", "theatrical_film")], "R7b", "a lower-case 'i' (word 2)"),
    "R8-TA10": ("So I married an axe murderer genre comedy drama.", [("so_i_married_an_axe_murderer", "genre", "comedy_drama")], "R7b",
                "'I' (word 2) before the lower-case word 'married'"),
    "R8-TA11": ("Alice Feeney is an author of sometimes i lie.", [("sometimes_i_lie", "author", "alice_feeney")], "R7b",
                "a lower-case 'i' (word 2)"),
    "R8-TA17": ("Trevor Horn is a member of Yes band. Trevor and I participated in Korean police action.",
                [("trevor_horn", "member_of", "yes_band"), ("trevor_and_i", "conflict", "korean_police_action")], "R7b",
                "a conjunction before 'I' ('and' 'I')"),
    "R8-TA20": ("and i love her follows if i fell.", [("and_i_love_her", "follows", "if_i_fell")], "R7b", "a lower-case 'i' (word 2)"),
}
R7A_IDS = [k for k, v in LEAKS.items() if v[2] == "R7a"]
R7B_IDS = [k for k, v in LEAKS.items() if v[2] == "R7b"]


def test_there_are_ten_round8_cases_five_per_rule():
    assert len(LEAKS) == 10 and len(R7A_IDS) == 5 and len(R7B_IDS) == 5


@pytest.mark.parametrize("cid", sorted(LEAKS))
def test_each_round8_leak_rejects_at_every_parser_seed(seed_chats, cid):
    reply, gathered, _rule, why = LEAKS[cid]
    res = verdict(reply, gathered)
    assert not res["accepted"] and why in res["reject_reason"], (cid, res["reject_reason"])
    for sd in SEEDS:
        acc, r = _on(seed_chats[sd], reply, gathered)
        assert acc is False and why in r["reject_reason"], (sd, cid, r["reject_reason"])


# The same proposition with the offending span written so that it stays sayable (a determiner before the pronoun word, a title-case word
# that is not the sentence's first word, or the slug verbatim): kept. So the rejection above is the new rule's and nothing else's.
CONTROLS = {
    "R8-TA01": "Reginald K Dwight is an author of Me.",
    "R8-TA04": "Taylor Alison Swift is an author of Mine.",
    "R8-TA07": "Canada portal shares border with the us.",
    "R8-TA15": "Stoker Bram is an author of Carfax Abbey. The It is an instance of theatrical film.",
    "R8-TA16": "Scarlett Johansson place of birth New York New York. The Her original language of film or TV show number of English words.",
    "R8-TA09": "honey_i_shrunk_the_kids is an instance of theatrical film.",
    "R8-TA10": "The so_i_married_an_axe_murderer genre comedy drama.",
    "R8-TA11": "Alice Feeney is an author of sometimes_i_lie.",
    "R8-TA17": "Trevor Horn is a member of Yes band. trevor_and_i participated in Korean police action.",
    "R8-TA20": "and_i_love_her follows if_i_fell.",
}


@pytest.mark.parametrize("cid", sorted(LEAKS))
def test_each_leaks_proposition_is_kept_when_the_name_is_written_so_it_stays_sayable(cid):
    gathered = LEAKS[cid][1]
    res = verdict(CONTROLS[cid], gathered)
    assert res["accepted"], (cid, CONTROLS[cid], res["reject_reason"])


# ------------------------------------------------------------------------------------------------------------ mutation checks
@pytest.mark.parametrize("cid", sorted(LEAKS))
def test_mutation_the_ruling_rule_off_makes_the_leak_kept_and_the_other_rule_does_not_hold_it(monkeypatch, cid):
    """Disable the rule that rejects the case (monkeypatch its helper to a no-op): the case is KEPT again (the build-6 verdict), so the
    test guards the rule; disable only the OTHER rule: the case is still rejected."""
    reply, gathered, rule, why = LEAKS[cid]
    own, other = ("_pronoun_name_reading", "_inner_first_person_reading") if rule == "R7a" else \
                 ("_inner_first_person_reading", "_pronoun_name_reading")
    with monkeypatch.context() as m:
        m.setattr(A, other, _noop)
        assert not keep(reply, gathered), cid
    with monkeypatch.context() as m:
        m.setattr(A, own, _noop)
        res = verdict(reply, gathered)
        assert res["accepted"], (cid, res["reject_reason"])


def test_mutation_both_rules_off_all_ten_are_kept_as_build6_kept_them(monkeypatch):
    monkeypatch.setattr(A, "_pronoun_name_reading", _noop)
    monkeypatch.setattr(A, "_inner_first_person_reading", _noop)
    bad = [cid for cid, (reply, gathered, _r, _w) in LEAKS.items() if not keep(reply, gathered)]
    assert not bad, bad


# Each branch of R7b has a sole-trigger example (no other branch, no build-6 rule fires on it): the branch tags in the reason are distinct.
R7B_BRANCH = [
    (["honey", "i", "shrunk", "the", "kids"], "a lower-case 'i' (word 2) in a name with no capital letter"),                    # (1)
    (["So", "I", "married", "an", "axe", "murderer"], "'I' (word 2) before the lower-case word 'married'"),                      # (2)
    (["Trevor", "and", "I"], "a conjunction before 'I' ('and' 'I')"),                                                           # (3), I last
    (["Trevor", "and", "i"], "a conjunction before 'I' ('and' 'i')"),                                                           # (3), "i" with a capital
    (["Trevor", "or", "I", "Horn"], "a conjunction before 'I' ('or' 'I')"),
    (["Trevor", "nor", "I"], "a conjunction before 'I' ('nor' 'I')"),
    (["Trevor", "but", "I"], "a conjunction before 'I' ('but' 'I')"),
    (["Trevor", "&", "I"], "a conjunction before 'I' ('&' 'I')"),
    (["Trevor", "And", "I"], "a conjunction before 'I' ('And' 'I')"),                                                          # case-folded
    (["Trevor", "And", "i"], "a conjunction before 'I' ('And' 'i')"),                                                           # case-folded
    (["Sometimes", "I", "lie"], "'I' (word 2) before the lower-case word 'lie'"),                                               # (2), "I"
    (["Sometimes", "i", "lie"], "'i' (word 2) before the lower-case word 'lie'"),                                               # (2), "i": widened
    (["Fatih", "i", "stanbul"], "'i' (word 2) before the lower-case word 'stanbul'"),
    (["Vang", "i", "valdres"], "'i' (word 2) before the lower-case word 'valdres'"),
    (["And", "i", "love", "her"], "'i' (word 2) before the lower-case word 'love'"),            # (2) is checked before (3): a lower-case word follows
    (["x", "y", "i", "z"], "a lower-case 'i' (word 3) in a name with no capital letter"),
]


@pytest.mark.parametrize("words,why", R7B_BRANCH)
def test_r7b_branches_name_themselves(words, why):
    assert why in (A._inner_first_person_reading(words) or ""), words


@pytest.mark.parametrize("words", [
    ["Charles", "I", "of", "England"],                # (2): "of" follows
    ["Nicholas", "I", "the", "Conqueror"],            # (2): a determiner follows
    ["Alexander", "I", "a", "Great"],                 # (2): a determiner follows
    ["Dom", "Pedro", "I"],                            # nothing follows
    ["World", "War", "I"],
    ["Division", "I", "Bowl"],                        # a capital follows
    ["Fuad", "I", "University"],
    ["V", "I", "Lenin"],
    ["Vang", "i", "Valdres"],                         # a lower-case "i" between capitalised words
    ["Baby", "I", "Love", "You"],                     # title case reads as a title: a DELIBERATE keep
    ["Honey", "I", "Shrunk", "The", "Kids"],
    ["Charles", "i", "of", "England"],                # (2): "of" follows, in either case of the "i"
    ["Charles", "i", "the", "Conqueror"],             # (2): a determiner follows
    ["Kunsthøgskolen", "i", "Oslo"],                  # a lower-case "i" before a capitalised word: the preposition
    ["Dom", "Pedro", "i"],                            # a lower-case "i" last, after capitalised words: nothing follows
    ["Charles", "i", "2nd"],                          # the next word starts with a digit, not a lower-case letter
    ["I", "need", "a", "beat"],                       # position 0 is rule (b1)'s, not R7b's
    ["Charles", "I", "2nd"],                          # next word starts with a digit, not a lower-case letter
    ["Android", "ia", "I", "Dom"],
])
def test_r7b_leaves_these_alone(words):
    assert A._inner_first_person_reading(words) is None, words


def test_r7b_reports_the_first_hit_and_skips_position_zero():
    assert "word 3" in A._inner_first_person_reading(["w", "x", "i", "y", "i"])           # the first hit is reported
    assert A._inner_first_person_reading(["i", "x"]) is None and A._inner_first_person_reading(["I", "x"]) is None
    assert A._inner_first_person_reading(["I"]) is None and A._inner_first_person_reading([]) is None
    # a capital anywhere in the span (even a leading "I" at position 0) means rule (1) has no casing-free display to read
    assert A._inner_first_person_reading(["I", "x", "i"]) is None


# ------------------------------------------------------------------------------------------------------------ R7a, the helper
CLASS = ("me my mine myself us our ours ourselves your yours yourself yourselves he him his himself she her hers herself it its itself "
         "they them their theirs themselves this that these those").split()


def test_the_pronoun_class_is_the_ruled_one():
    assert A.PRONOUN_NAME_CLASS == frozenset(CLASS) and len(CLASS) == 32
    assert not (A.PRONOUN_NAME_CLASS & {"i", "we", "you"})                      # those stay under (b1'): any case, anywhere
    assert A.NAME_CONJUNCTIONS == frozenset(("and", "or", "nor", "but", "&"))


@pytest.mark.parametrize("w", CLASS)
def test_r7a_helper_matrix_for_every_class_word(w):
    title, caps = w.capitalize(), w.upper()
    assert A._pronoun_name_reading(w, False, False)                               # lower case, mid-sentence: rejects
    assert A._pronoun_name_reading(w, True, False)                                # lower case, sentence-initial: rejects
    assert A._pronoun_name_reading(title, True, False)                            # Title case as the sentence's first word: rejects
    assert A._pronoun_name_reading(title, False, False) is None                   # Title case elsewhere: reads as a title
    if len(w) >= 2:
        assert A._pronoun_name_reading(caps, True, False) is None                 # all-caps 2+ letters: an abbreviation
        assert A._pronoun_name_reading(caps, False, False) is None
    for word in (w, title):
        assert A._pronoun_name_reading(word, True, True) is None                  # a determiner directly before it
        assert A._pronoun_name_reading(word, False, True) is None


@pytest.mark.parametrize("w", ["i", "we", "you", "Taylor", "mesa", "ithe", "ours2", ""])
def test_r7a_helper_ignores_words_outside_the_class(w):
    assert A._pronoun_name_reading(w, True, False) is None and A._pronoun_name_reading(w, False, False) is None


# =========================================================================================================== what stays sayable
# store triples (a fact of the slug in wikidata_100k, checked below when the store is on this machine) or taught triples (marked)
STORE_KEEPS = [
    ("Charles I of England is an instance of human specie.", [("charles_i_of_england", "instance_of", "human_specie")]),
    ("Charles I of England occupation political figure.", [("charles_i_of_england", "occupation", "political_figure")]),
    ("Dom Pedro I is an instance of human specie.", [("dom_pedro_i", "instance_of", "human_specie")]),
    ("Dom Pedro I occupation political figure.", [("dom_pedro_i", "occupation", "political_figure")]),
    ("Nicholas I the Conqueror is an instance of human specie.", [("nicholas_i_the_conqueror", "instance_of", "human_specie")]),
    ("Division I Bowl is an instance of sports leagues.", [("division_i_bowl", "instance_of", "sports_leagues")]),
    ("Fuad I University is an instance of higher general education.",
     [("fuad_i_university", "instance_of", "higher_general_education")]),
    ("V I Lenin is an instance of human specie.", [("v_i_lenin", "instance_of", "human_specie")]),
    ("V I Lenin occupation political figure.", [("v_i_lenin", "occupation", "political_figure")]),
    ("Lom Norway shares border with Vang i Valdres.", [("lom_norway", "shares_border_with", "vang_i_valdres")]),
    ("Vang i Valdres shares border with Lom Norway.", [("vang_i_valdres", "shares_border_with", "lom_norway")]),
    # the coordinator's ruling of 2026-10-21: a lower-case "i" before a capitalised word, and a lower-case "i" last after capitalised words
    ("Kunsth\u00f8gskolen i Oslo is an instance of studio art.", [("kunsth_gskolen_i_oslo", "instance_of", "studio_art")]),
    ("Kunsth\u00f8gskolen i Oslo country Royal Kingdom of Norway.", [("kunsth_gskolen_i_oslo", "country", "royal_kingdom_of_norway")]),
    ("Dom Pedro i is an instance of human specie.", [("dom_pedro_i", "instance_of", "human_specie")]),
    ("Dom Pedro i occupation political figure.", [("dom_pedro_i", "occupation", "political_figure")]),
]
TAUGHT_KEEPS = [
    # a Roman numeral with nothing after it; no store slug (taught triple)
    ("World War I is an instance of war conflict.", [("world_war_i", "instance_of", "war_conflict")]),
    # title case reads as a title: a DELIBERATE keep (the clause "Baby I Love You" is open, documented)
    ("Baby I Love You performer the Ramones.", [("baby_i_love_you", "performer", "the_ramones")]),
    # a capitalised one-word span that is not the sentence's first word reads as a title
    ("Taylor Swift is an author of Mine.", [("mine", "author", "taylor_swift")]),
    ("Reginald K Dwight is an author of Me.", [("me", "author", "reginald_k_dwight")]),
    ("Taylor Alison Swift is an author of Mine.", [("mine", "author", "taylor_alison_swift")]),
    # (the brief's "Spike Jonze directed Her.": no verb reading for `director`, so the author phrase stands in)
    ("Spike Jonze is an author of Her.", [("her", "author", "spike_jonze")]),
    ("Stoker Bram is an author of It.", [("it", "author", "stoker_bram")]),
    # a determiner directly before the pronoun word, in any case, mid-sentence or opening the sentence
    ("Canada portal shares border with the us.", [("canada_portal", "shares_border_with", "us")]),
    ("Taylor Swift is an author of the mine.", [("mine", "author", "taylor_swift")]),
    ("The It is an instance of theatrical film.", [("it", "instance_of", "theatrical_film")]),
    ("The her is an instance of theatrical film.", [("her", "instance_of", "theatrical_film")]),
    # a lower-case "i" between capitalised words is the preposition; a Roman numeral followed by a capital
    ("Charles i of England is an instance of human specie.", [("charles_i_of_england", "instance_of", "human_specie")]),
    ("Charles I Of England is an instance of human specie.", [("charles_i_of_england", "instance_of", "human_specie")]),
    # the slug written verbatim (the stub's form) stays one identifier for R7b
    ("The honey_i_shrunk_the_kids is an instance of theatrical_film.", [("honey_i_shrunk_the_kids", "instance_of", "theatrical_film")]),
    ("honey_i_shrunk_the_kids is an instance of theatrical film.", [("honey_i_shrunk_the_kids", "instance_of", "theatrical_film")]),
    ("The charles_i_of_england occupation political_figure.", [("charles_i_of_england", "occupation", "political_figure")]),
    ("Alice Feeney is an author of sometimes_i_lie.", [("sometimes_i_lie", "author", "alice_feeney")]),
]
KEEPS = STORE_KEEPS + TAUGHT_KEEPS


@pytest.mark.parametrize("sentence,gathered", KEEPS)
def test_what_stays_sayable_stays_kept_at_every_parser_seed(seed_chats, sentence, gathered):
    res = verdict(sentence, gathered)
    assert res["accepted"], (sentence, res["reject_reason"])
    for sd in SEEDS:
        assert _on(seed_chats[sd], sentence, gathered)[0] is True, (sd, sentence)


def test_the_store_keeps_use_real_store_facts():
    if not os.path.exists(STORE_FACTS):
        pytest.skip("production store not on this machine")
    have = {(r["fact"]["agent"], r["fact"]["action"], r["fact"]["patient"]) for r in json.load(open(STORE_FACTS))
            if r["fact"].get("polarity") == "AFFIRM"}
    missing = [g for _s, gs in STORE_KEEPS for g in gs if tuple(g) not in have]
    assert not missing, missing


class _FakeCaps:
    """A caps table that lets the all-caps abbreviation through for the pinned one-word slugs (no store slug is one: the real table has
    no row, so the case check would reject an all-caps token first)."""
    error = None

    def allowed(self, kind, slug):
        return frozenset(("US", "ME", "IT")) if kind == "ent" and slug in ("us", "me", "it") else frozenset()


@pytest.mark.parametrize("sentence,gathered", [
    ("Canada portal shares border with US.", [("canada_portal", "shares_border_with", "us")]),
    ("Taylor Swift is an author of ME.", [("me", "author", "taylor_swift")]),
    ("IT is an instance of theatrical film.", [("it", "instance_of", "theatrical_film")]),          # all-caps AND the sentence's first word
])
def test_an_all_caps_abbreviation_of_2_letters_is_not_the_pronoun(monkeypatch, sentence, gathered):
    monkeypatch.setattr(A, "caps_table", lambda: _FakeCaps())
    assert keep(sentence, gathered), verdict(sentence, gathered)["reject_reason"]
    # without the exemption the same sentence in title case at the start, or lower case, rejects (the fake table is not what keeps it)
    low = sentence.replace(" US.", " us.").replace(" ME.", " me.")
    if low != sentence:
        assert not keep(low, gathered)


def test_mutation_the_all_caps_exemption_is_what_keeps_an_abbreviation_that_opens_a_sentence(monkeypatch):
    """The exemption matters where the capitals would otherwise read as the sentence's first word: "IT is an instance of ..." (a mid-sentence
    "US" is a capitalised word that is not the sentence's first word and stays sayable anyway). Without the `caps_tokens` test in
    `_pronoun_name_reading` the opening abbreviation would reject."""
    monkeypatch.setattr(A, "caps_table", lambda: _FakeCaps())
    sentence, gathered = "IT is an instance of theatrical film.", [("it", "instance_of", "theatrical_film")]
    assert keep(sentence, gathered)
    monkeypatch.setattr(A, "caps_tokens", lambda text: [])                       # no word counts as an all-caps token any more
    res = verdict(sentence, gathered)
    assert not res["accepted"] and "reads as the pronoun 'IT' (the first word of the sentence" in res["reject_reason"], res["reject_reason"]


@pytest.mark.parametrize("sentence,gathered,why", [
    ("Me is an instance of theatrical film.", [("me", "instance_of", "theatrical_film")], "the first word of the sentence"),
    ("Mine is an instance of theatrical film.", [("mine", "instance_of", "theatrical_film")], "the first word of the sentence"),
    ("them is an instance of theatrical film.", [("them", "instance_of", "theatrical_film")], "written in lower case"),
    ("Canada portal shares border with them.", [("canada_portal", "shares_border_with", "them")], "written in lower case"),
    ("Canada portal shares border with those.", [("canada_portal", "shares_border_with", "those")], "written in lower case"),
    ("Taylor Swift is an author of itself.", [("itself", "author", "taylor_swift")], "written in lower case"),
])
def test_other_class_words_reject_as_the_rule_reads_them(sentence, gathered, why):
    res = verdict(sentence, gathered)
    assert not res["accepted"] and "reads as the pronoun" in res["reject_reason"] and why in res["reject_reason"], res["reject_reason"]


@pytest.mark.parametrize("sentence,gathered", [
    ("Trevor And I participated in Korean police action.", [("trevor_and_i", "conflict", "korean_police_action")]),
    ("Trevor and i participated in Korean police action.", [("trevor_and_i", "conflict", "korean_police_action")]),
    ("Pride and I is an instance of theatrical film.", [("pride_and_i", "instance_of", "theatrical_film")]),
    ("Tom but I is an instance of theatrical film.", [("tom_but_i", "instance_of", "theatrical_film")]),
    ("And i love her follows If I Fell.", [("and_i_love_her", "follows", "if_i_fell")]),                 # the build-6 residue pin, now closed
    ("charles i of england is an instance of human specie.", [("charles_i_of_england", "instance_of", "human_specie")]),   # COST: no signal
    ("Sometimes I lie is an instance of theatrical film.", [("sometimes_i_lie", "instance_of", "theatrical_film")]),
])
def test_further_first_person_forms_reject(sentence, gathered):
    res = verdict(sentence, gathered)
    assert not res["accepted"], sentence
    assert any(t in res["reject_reason"] for t in ("'I' (word", "'i' (word", "a conjunction before 'I'")), res["reject_reason"]


# ---- the widened rule R7b(2): "i" as well as "I" before a lower-case word (coordinator's ruling of 2026-10-21; closes "Sometimes i lie")
WIDENED = [
    ("Alice Feeney is an author of Sometimes i lie.", [("sometimes_i_lie", "author", "alice_feeney")], "'i' (word 2) before the lower-case word 'lie'"),
    ("Fatih i stanbul country the Republic of Turkey.", [("fatih_i_stanbul", "country", "the_republic_of_turkey")],
     "'i' (word 2) before the lower-case word 'stanbul'"),
    ("Lom Norway shares border with Vang i valdres.", [("lom_norway", "shares_border_with", "vang_i_valdres")],
     "'i' (word 2) before the lower-case word 'valdres'"),
    ("And i love her follows If I Fell.", [("and_i_love_her", "follows", "if_i_fell")], "'i' (word 2) before the lower-case word 'love'"),
]


@pytest.mark.parametrize("sentence,gathered,why", WIDENED)
def test_the_widened_rule_rejects_a_lower_case_i_before_a_lower_case_word_at_every_parser_seed(seed_chats, sentence, gathered, why):
    res = verdict(sentence, gathered)
    assert not res["accepted"] and why in res["reject_reason"], (sentence, res["reject_reason"])
    for sd in SEEDS:
        acc, r = _on(seed_chats[sd], sentence, gathered)
        assert acc is False and why in r["reject_reason"], (sd, sentence)


@pytest.mark.parametrize("sentence,gathered,why", WIDENED[:3])
def test_mutation_restricting_rule_7b2_back_to_upper_case_I_makes_the_widened_cases_kept(monkeypatch, sentence, gathered, why):
    """The first build 7 read rule (2) on an upper-case "I" only. Restrict `CLAUSE_I_FORMS` back to it (monkeypatch): the widened cases are
    KEPT again (the first build 7's verdict), while the "I" half of rule (2) is untouched (R8-TA10 "So I married an axe murderer" still
    rejects), so these tests guard the widening."""
    assert not keep(sentence, gathered)
    ta10 = LEAKS["R8-TA10"]
    with monkeypatch.context() as m:
        m.setattr(A, "CLAUSE_I_FORMS", frozenset(("I",)))
        res = verdict(sentence, gathered)
        assert res["accepted"], (sentence, res["reject_reason"])
        assert not keep(ta10[0], ta10[1])


def test_mutation_and_i_love_her_stays_rejected_without_the_widening_but_by_rule_3(monkeypatch):
    """"And i love her" has a capitalised "And", so rule (1) does not read it; with (2) restricted to "I" it is still rejected, by rule (3)
    (a conjunction before the "i", case-folded) -- the first build 7's reason for it."""
    sentence, gathered, why = WIDENED[3]
    monkeypatch.setattr(A, "CLAUSE_I_FORMS", frozenset(("I",)))
    res = verdict(sentence, gathered)
    assert not res["accepted"] and "a conjunction before 'I' ('And' 'i')" in res["reject_reason"], res["reject_reason"]


def test_mutation_the_widening_is_the_only_difference_for_sometimes_i_lie(monkeypatch):
    sentence, gathered = "Alice Feeney is an author of Sometimes i lie.", [("sometimes_i_lie", "author", "alice_feeney")]
    assert A.CLAUSE_I_FORMS == frozenset(("i", "I"))
    assert not keep(sentence, gathered)
    monkeypatch.setattr(A, "CLAUSE_I_FORMS", frozenset(("I",)))
    assert keep(sentence, gathered)                                  # the first build 7's verdict: the documented residue
    for words in (["Sometimes", "i", "lie"], ["Fatih", "i", "stanbul"], ["Vang", "i", "valdres"]):
        assert A._inner_first_person_reading(words) is None, words
    assert A._inner_first_person_reading(["Sometimes", "I", "lie"])  # the upper-case half is unchanged


# ---- KNOWN ANSWER COSTS (accepted by the coordinator's rulings of 2026-10-21; recorded in the module comment)
def test_known_answer_cost_the_all_lower_case_deslugged_name_rejects_but_the_slug_and_title_case_stay():
    for slug, lower, title, fact in [("v_i_lenin", "v i lenin", "V I Lenin", ("v_i_lenin", "instance_of", "human_specie")),
                                     ("charles_i_of_england", "charles i of england", "Charles I of England",
                                      ("charles_i_of_england", "instance_of", "human_specie")),
                                     ("dom_pedro_i", "dom pedro i", "Dom Pedro I", ("dom_pedro_i", "instance_of", "human_specie"))]:
        rej = verdict("%s is an instance of human specie." % lower, [fact])
        assert not rej["accepted"] and "a lower-case 'i' (word" in rej["reject_reason"], (slug, rej["reject_reason"])
        assert keep("%s is an instance of human specie." % title, [fact]), title
        assert keep("The %s instance_ofs human_specie." % slug, [fact]), slug


def test_known_answer_cost_uk_of_gb_and_i_rejects_in_every_spaced_spelling_and_only_the_slug_stays():
    r6 = ("UK of GB and I follows Kingdom of Great Britian.", [("uk_of_gb_and_i", "follows", "kingdom_of_great_britian")])     # R6-CN05 (B)
    chuck = ("Chuck Darwin country of citizenship UK of GB and I.", [("chuck_darwin", "country_of_citizenship", "uk_of_gb_and_i")])
    for sentence, gathered in (r6, chuck):
        res = verdict(sentence, gathered)
        assert not res["accepted"] and "a conjunction before 'I' ('and' 'I')" in res["reject_reason"], (sentence, res["reject_reason"])
    for spelling in ("UK of GB and I", "Uk Of Gb And I", "uk of gb and i", "UK Of GB And I"):
        assert not keep("%s follows Kingdom of Great Britian." % spelling, r6[1]), spelling
    assert keep("uk_of_gb_and_i follows Kingdom of Great Britian.", r6[1])                       # the slug written verbatim
    assert keep("Chuck Darwin country of citizenship uk_of_gb_and_i.", chuck[1])
    # the slug has no names-table row: its canonical surfaces are the slug and its de-slugged form only
    table = A.names_table()
    assert not table.entities.get("uk_of_gb_and_i") and table.surfaces("uk_of_gb_and_i") == frozenset(("uk_of_gb_and_i", "uk of gb and i"))


# ---- the stub renders every slug verbatim: the consequence of "a one-word slug written verbatim is the pronoun" is pinned both ways
def test_the_stubs_verbatim_one_word_slug_is_the_pronoun_only_where_no_determiner_stands_before_it(monkeypatch):
    from research.runners.brain_chat_tui import StubRenderer
    monkeypatch.setenv(FLAG, "1")
    stub = StubRenderer()
    # the stub writes "The <subject> <verb> <object>.": the subject has a determiner, the object has none
    subj = stub.render_svo("me", "instance_of", "theatrical_film")[0]
    obj = stub.render_svo("canada_portal", "shares_border_with", "us")[0]
    assert subj.startswith("The me ") and keep(subj, [("me", "instance_of", "theatrical_film")]), subj
    res = verdict(obj, [("canada_portal", "shares_border_with", "us")])
    assert not res["accepted"] and "reads as the pronoun 'us'" in res["reject_reason"], (obj, res["reject_reason"])
    # R7b never fires on a stub sentence: the stub's slugs are verbatim
    for svo in [("honey_i_shrunk_the_kids", "instance_of", "theatrical_film"), ("trevor_and_i", "conflict", "korean_police_action"),
                ("and_i_love_her", "follows", "if_i_fell"), ("so_i_married_an_axe_murderer", "genre", "comedy_drama"),
                ("charles_i_of_england", "occupation", "political_figure"), ("v_i_lenin", "instance_of", "human_specie")]:
        surf = stub.render_svo(*svo)[0]
        assert keep(surf, [svo]), (surf, verdict(surf, [svo])["reject_reason"])


# =========================================================================================================== structure
def test_the_build7_rules_only_add_rejections_and_run_after_every_build6_check():
    """Monotone: the build-7 pass sits after the whole build-6 pass in `_name_reading_violation`, which sits after the build-5 checks in
    `allowlist_check`; with both helpers disabled the function is the build-6 one."""
    src = inspect.getsource(A._name_reading_violation)
    assert src.index("CASED_CLAUSE_WORDS") < src.index("# build 7: after every build-6 check") < src.index("_pronoun_name_reading(")
    assert src.index("_pronoun_name_reading(") < src.index("_inner_first_person_reading(")
    check = inspect.getsource(A.allowlist_check)
    assert check.index("_structure_word_violation(parses") < check.index("_name_reading_violation(parses")


@pytest.mark.parametrize("sentence,gathered", KEEPS + [(r, g) for r, g, _a, _b in LEAKS.values()] + [
    ("The Band award received Grammy lifetime achievement award.", [("the_band", "award_received", "grammy_lifetime_achievement_award")]),
    ("I need a beat genre hip hop music.", [("i_need_a_beat", "genre", "hip_hop_music")]),
    ("Brian May country of citizenship united kingom.", [("brian_may", "country_of_citizenship", "united_kingom")]),
])
def test_with_both_rules_disabled_a_kept_sentence_is_the_same_kept_sentence(monkeypatch, sentence, gathered):
    """kept(build 7) implies kept(build 6 behaviour: both helpers off): the rules cannot turn a rejection into a keep."""
    now = keep(sentence, gathered)
    monkeypatch.setattr(A, "_pronoun_name_reading", _noop)
    monkeypatch.setattr(A, "_inner_first_person_reading", _noop)
    before = keep(sentence, gathered)
    assert before or not now, sentence


def test_a_build6_reject_keeps_its_build6_reason(monkeypatch):
    """A sentence the build-6 rules reject names a build-6 reason, not a build-7 one (the build-7 pass runs after)."""
    for s, g, frag in [("I need a beat genre hip hop music.", [("i_need_a_beat", "genre", "hip_hop_music")], "first- or second-person subject"),
                       ("brian may country of citizenship united kingom.", [("brian_may", "country_of_citizenship", "united_kingom")],
                        "written as prose"),
                       ("i language of work or name French language.", [("i", "language_of_work_or_name", "french_language")],
                        "reads as the pronoun 'i'")]:
        assert frag in verdict(s, g)["reject_reason"], s


def test_flag_off_the_module_is_not_consulted_and_the_flag_default_is_off(monkeypatch):
    """The change sits inside the allowlist module, reached only through the flag: with the flag unset or falsy `allowlist_enabled` is
    False (the call sites are pinned in tests/test_moat_allowlist_build6.py::test_flag_off_mouth_paths_equal_the_base_golden)."""
    assert A.allowlist_enabled() is False
    for val in ("0", "off", "false", "garbage"):
        monkeypatch.setenv(FLAG, val)
        assert A.allowlist_enabled() is False
    monkeypatch.setenv(FLAG, "1")
    assert A.allowlist_enabled() is True
