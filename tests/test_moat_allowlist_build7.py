"""Pins for build 7 of the strict allowlist claim checker (`BRAIN_MOAT_ALLOWLIST`, default OFF): two narrow name rules that close the remaining
first-person (honesty-boundary) leaks of the round-8 blind hunt (Claude rulings 2026-10-21; spec docs/drafts/moat-allowlist/SPEC.md, the
build-7 bullet; rules in the comment block above `PRONOUN_NAME_CLASS` in research/runners/_moat_allowlist.py).

  R7a. A ONE-word SUBJ/OBJ span whose word, case-folded, is in a closed pronoun class rejects when it is written in lower case OR is the
       sentence's first word, unless a determiner stands directly before it or it is written all-caps with 2+ letters. It runs before the
       slug-verbatim skip.
  R7b. In a spaced span (2+ words, not the slug verbatim) a word "i" / "I" at position k > 0 reads as the first-person pronoun when
       (1) it is lower-case "i" and no word of the span starts with an upper-case letter; (2) it is "I", the next word starts with a
       lower-case letter and is neither "of" nor a determiner; (3) the word directly before it, case-folded, is and / or / nor / but / "&".

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
    (["And", "i", "love", "her"], "a conjunction before 'I' ('And' 'i')"),                                                      # case-folded
    (["Sometimes", "I", "lie"], "'I' (word 2) before the lower-case word 'lie'"),
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
    ["Sometimes", "i", "lie"],                        # DOCUMENTED RESIDUE: a capital somewhere, so rule (1) does not apply
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
    assert A._inner_first_person_reading(["I", "x", "i", "y"]) is None


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
    # DOCUMENTED RESIDUE (rule (1) needs no capital anywhere in the name): kept, pinned so a change is visible
    ("Alice Feeney is an author of Sometimes i lie.", [("sometimes_i_lie", "author", "alice_feeney")]),
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
    assert any(t in res["reject_reason"] for t in ("'I' (word", "a lower-case 'i' (word", "a conjunction before 'I'")), res["reject_reason"]


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
