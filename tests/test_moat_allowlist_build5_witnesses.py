"""Pins for the 70 NON-equivalent survivors of the build-5 mutation test of the strict allowlist claim checker (`BRAIN_MOAT_ALLOWLIST`).

The build-5 mutation test (/home/dant123/Projects/sim-rev/moat-allow5/mut/summary_b5.json: 347 mutants of commit ef51d6243 over
research/runners/_moat_allowlist.py and _moat_allowlist_relations.py, run against the five allowlist test files) left 118 survivors. 48 are
equivalent (no input separates the mutant from the original). The other 70 are real gaps: an input exists on which the original and the mutant
differ. Each test below pins the ORIGINAL behaviour on that input, so the mutant now fails. The ids in each docstring are the mutant ids of
summary_b5.json (cat. in mutants_b5.py).

Two kinds of pin:
  - exact-set pins for the dropped-apostrophe word sets (PRONOUNS_S, PRONOUNS_AUX, AUX_RESIDUE, NEGATIVE_STEMS) and DROPPED_READINGS['eat'],
    plus a behavioural check of EVERY member written from a literal list in this file (never from the constant under test, which the
    mutant edits: a parametrisation read from the constant loses the dropped member with it);
  - behavioural witnesses for the loop edges of the residue rule, the non-adjacent reply-level repeat, a relation span that mixes a space
    and an underscore, the non-ASCII edge characters of the pure sentence function, the accent letter before a determiner slug, and
    `deslug` on a run of underscores.

RECORDED DESIGN CHOICES, not fixes (the current behaviour is pinned so a change is a deliberate decision): a hyphen is not a word break
(sw12: the two store names that hold a clause word joined by hyphens are kept); a lone 's' or 't' that is a whole one-word name, the first
word, or the last word after a word that is not a negative stem is not apostrophe residue (sw21, sw27, sw29); a name that merely starts
with a pronoun-like word outside the reviewed pairs is kept ("My s ...", "Who ll ..."); a gathered slug written verbatim with an edge
underscore is one identifier (sw52). Where a pin is a boundary rather than an endorsement the docstring says so.

Run: CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m pytest tests/test_moat_allowlist_build5_witnesses.py -q
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest

from research.runners import _moat_allowlist as A
from research.runners import _moat_allowlist_relations as T

pytestmark = pytest.mark.filterwarnings("ignore")

FLAG = "BRAIN_MOAT_ALLOWLIST"
SPAIN = [["spain", "borders", "france"], ["spain", "borders", "portugal"]]
SBW = [["spain", "shares_border_with", "france"]]
RESIDUE = "the residue of a dropped apostrophe"


@pytest.fixture(autouse=True)
def _clean_flags(monkeypatch):
    for k in (FLAG, "BRAIN_MOAT_SLUG_NORM", "BRAIN_CLAIM_MOAT"):
        monkeypatch.delenv(k, raising=False)


def _u(slug):
    """One gathered fact whose subject is `slug`: the sentence "<Name> uses Baz." is a shape-1 sentence on it."""
    return [[slug, "uses", "baz"]]


def _kept(sentence, gathered):
    """The verdict of the pure function AND of the production reply check agree: -> the check result when both keep, asserting that."""
    r = A.allowlist_check(sentence, gathered)
    assert r["keep"] is True and r["reason"] == "ok", (sentence, r["reason"])
    v = A.allowlist_verify(sentence, gathered)
    assert v["accepted"] is True, (sentence, v["reject_reason"])
    return r


def _rejected(sentence, gathered, *fragments):
    """Both entries reject; every fragment is in the reason of both; -> the check reason."""
    r = A.allowlist_check(sentence, gathered)
    assert r["keep"] is False and r["reason"].startswith("allowlist:"), (sentence, r)
    v = A.allowlist_verify(sentence, gathered)
    assert v["accepted"] is False, (sentence, v)
    for f in fragments:
        assert f in r["reason"], (sentence, f, r["reason"])
        assert f in v["reject_reason"], (sentence, f, v["reject_reason"])
    return r["reason"]


# ====================================================================================================================== deslug
@pytest.mark.parametrize("slug,expected", [
    ("foo__bar", "foo bar"), ("foo___bar", "foo bar"), ("foo_", "foo"), ("_foo", "foo"), ("_foo__bar_", "foo bar"),
    ("a_b_c", "a b c"), ("foo", "foo"), ("_", ""), ("__", ""), ("", ""),
])
def test_deslug_gives_one_space_between_words_and_none_at_the_edges(slug, expected):
    """ds01 ds02: the docstring of `deslug` says a run of underscores still gives one space; an edge underscore gives no space. Neither
    `slug.replace("_", " ")` nor a join that keeps the empty pieces of the split does that."""
    assert A.deslug(slug) == expected


@pytest.mark.parametrize("sentence,slug", [("Foo bar uses Baz.", "foo__bar"), ("Foo uses Baz.", "foo_"), ("Bar uses Baz.", "_bar")])
def test_a_slug_holding_an_underscore_run_or_edge_is_kept_in_its_deslugged_form(sentence, slug):
    """ds01 ds02: the same contract through the matcher. These slugs are not store-shaped (the store's `sanitize` makes neither a run nor an
    edge underscore); the pin is on `deslug`'s documented contract, so a change to it is a deliberate one."""
    r = _kept(sentence, _u(slug))
    assert r["props"] == [[slug, "uses", "baz"]]


# ============================================================================================ the non-ASCII scan (build 4 exact check)
@pytest.mark.parametrize("det", ["a", "an", "the"])
@pytest.mark.parametrize("letter", ["ĳ", "é", "ø"])
def test_an_accent_letter_in_front_of_a_determiner_slug_is_outside_every_span(det, letter):
    """ex21: when the determiner ("a", "an", "the") is also a gathered noun slug, the folded accent letter in front of it ("ĳa") is
    absorbed into that slug's match; in the parse where the word is the DETERMINER it belongs to no span, so the non-ASCII scan must
    reject it (a letter is not exempt from the scan). The sentence without the letter is kept."""
    g = [[det, "uses", "baz"], ["x", "uses", "baz"]]
    _kept("%s x uses baz." % det, g)
    why = _rejected("%s%s x uses baz." % (letter, det), g, "non-ASCII character %r (U+%04X) at 0 outside every accepted SUBJ/OBJ span"
                    % (letter, ord(letter)))
    assert "entity span" not in why


@pytest.mark.parametrize("space", [" ", " ", " ", " "])
def test_the_pure_function_scans_the_first_and_the_last_character_for_non_ascii(space):
    """ex22 ex23: a display-identical space is legal INSIDE an entity name and between tokens' span, but at the edge of a sentence handed
    straight to `allowlist_check` it is a non-ASCII character outside every accepted SUBJ/OBJ span. `allowlist_verify` never lets such an
    edge reach the sentence check (`split_sentences` strips it, `_bad_gap` rejects it first): only the public pure function can see it,
    and the scan must start at index 0 and end at the last index."""
    base = "Spain borders France."
    assert A.allowlist_check(base, SPAIN)["keep"] is True
    for sentence, k in ((space + base, 0), (base + space, len(base))):
        r = A.allowlist_check(sentence, SPAIN)
        assert r["keep"] is False and r["shape"] == "shape1", (sentence, r)
        assert r["reason"] == ("allowlist: non-ASCII character %r (U+%04X) at %d outside every accepted SUBJ/OBJ span"
                               % (space, ord(space), k)), r["reason"]
        v = A.allowlist_verify(sentence, SPAIN)             # the production path: rejected before the sentence check, same verdict
        assert v["accepted"] is False and "between sentences or at an edge of the reply" in v["reject_reason"], v["reject_reason"]


def test_a_span_space_inside_a_name_is_still_legal_so_the_edge_pin_is_about_position_only():
    """ex22 ex23 control: the four display-identical spaces INSIDE an entity span stay accepted (they are covered by the span)."""
    for space in (" ", " ", " ", " "):
        _kept("Costa%sRica uses Baz." % space, _u("costa_rica"))


# =================================================================================================================== relation span
@pytest.mark.parametrize("relation", ["shares border_with", "shares_border with", "shares-border-with", "shares border-with"])
def test_a_relation_span_mixing_a_space_and_an_underscore_or_a_hyphen_is_rejected(relation):
    """ex29: a relation span is written exactly as the slug (underscores) or exactly as its words (single spaces); a spelling that mixes a
    space with an underscore, or uses a hyphen, is neither, and no fold is applied to relation words."""
    why = _rejected("Spain %s France." % relation, SBW, "relation span %r is not written exactly as the relation reading "
                    "'shares border with' of shares_border_with" % relation, "no fold on relation words")
    assert "entity span" not in why


@pytest.mark.parametrize("relation", ["shares border with", "shares_border_with", "Shares Border With", "shares  border   with"])
def test_the_two_exact_spellings_of_a_relation_span_are_kept(relation):
    """ex29 control: the verbatim slug and the words with spaces (any casing, runs of spaces collapse) are the accepted spellings."""
    _kept("Spain %s France." % relation, SBW)


# ========================================================================================== a hyphen is not a word break (sw12)
@pytest.mark.parametrize("sentence,slug,row", [
    ("Canton of is-sur-tille uses Baz.", "canton_of_is_sur_tille", "canton of is-sur-tille"),
    ("Not-for-profit organization uses Baz.", "not_for_profit_organization", "not-for-profit organization"),
])
def test_a_store_name_holding_a_clause_word_joined_by_hyphens_is_kept(sentence, slug, row):
    """sw12 (RECORDED DESIGN CHOICE, not a fix): `_WORD_BREAK` splits on spaces and underscores only, so "is-sur-tille" and
    "not-for-profit" are one word each and the clause-word rule does not see "is" / "not" inside them. These are the two exact store
    surfaces (committed names table) that hold a clause word joined only by hyphens; a name with the same word after a SPACE is rejected."""
    assert row in A.names_table().entities[slug], "the pinned store row is gone: re-pin sw12 (%s)" % slug
    r = _kept(sentence, _u(slug))
    assert r["props"] == [[slug, "uses", "baz"]]


def test_the_same_clause_word_after_a_space_in_a_name_is_still_rejected():
    """sw12 control: only the hyphen joins; "Canton of is sur tille" (spaces) holds the clause word "is". The exact check is passed by
    the de-slugged form, so the structure rule is what rejects it."""
    _rejected("Canton of is sur tille uses Baz.", _u("canton_of_is_sur_tille"), "holds the clause word 'is'")


# ================================================================================================ residue rule: positions and edges
@pytest.mark.parametrize("sentence,slug", [
    ("s uses Baz.", "s"),                                  # sw21: a one-word name that is the lone letter itself
    ("Saint Alban t uses Baz.", "saint_alban_t"),          # sw27: a final lone 't' after a word that is not a negative stem
    ("Alban t uses Baz.", "alban_t"),
    ("T Don uses Baz.", "t_don"),                          # sw29: a FIRST word 't' has no previous word, it does not wrap round
    ("T Can uses Baz.", "t_can"),
    ("Communaut de communes de la c te d alb t uses Baz.", "communaut_de_communes_de_la_c_te_d_alb_t"),     # a real store slug (3 facts)
    ("S He uses Baz.", "s_he"),
])
def test_a_lone_s_or_t_outside_the_reviewed_positions_is_kept(sentence, slug):
    """sw21 sw27 sw29 (RECORDED DESIGN CHOICE, not a fix). The final-"s" rule needs a SPACED name (more than one word); the "t" rule needs
    a negative stem right BEFORE it; the pair rules look at a word and the one before it, so the first word never pairs with the last.
    A one-word name 's', a name ending in a lone 't' after an ordinary word, and 'T Don' are therefore kept (the store slug
    communaut_de_communes_de_la_c_te_d_alb_t ends in the accent fold of a 't')."""
    r = _kept(sentence, _u(slug))
    assert r["props"] == [[slug, "uses", "baz"]]


@pytest.mark.parametrize("sentence,slug,lone", [
    ("Don T uses Baz.", "don_t", "T"),                     # sw30: the pair is the WHOLE name, so the last word is the one that matters
    ("Don t uses Baz.", "don_t", "t"),
    ("Won T uses Baz.", "won_t", "T"),
    ("Isn t uses Baz.", "isn_t", "t"),
    ("He s uses Baz.", "he_s", "s"),                       # the pair rules also reach the last word of a two-word name
    ("I ll uses Baz.", "i_ll", "ll"),
    ("I m uses Baz.", "i_m", "m"),
])
def test_a_residue_pair_that_is_the_whole_two_word_name_is_rejected(sentence, slug, lone):
    """sw30: the pair loop runs to the LAST word. With only two words the pair is the entire name and the last word is the residue, so a
    loop that stops one word early (range(1, n - 1)) keeps it."""
    _rejected(sentence, _u(slug), "holds a lone lower-case %r" % lone, RESIDUE)


@pytest.mark.parametrize("sentence,slug", [("Foo Don T uses Baz.", "foo_don_t"), ("Foo He s uses Baz.", "foo_he_s"),
                                           ("Don T Foo uses Baz.", "don_t_foo"), ("Foo I m Bar uses Baz.", "foo_i_m_bar")])
def test_a_residue_pair_anywhere_in_a_longer_name_is_rejected(sentence, slug):
    """sw30 (with sw29): at the end, at the start and in the middle of a longer name."""
    _rejected(sentence, _u(slug), RESIDUE)


# =================================================================================================== the dropped-apostrophe word sets
PRONOUNS_S = ("he", "she", "it", "that", "there", "here", "what", "who", "where", "someone", "somebody", "everyone", "everybody",
              "everything", "something", "nobody", "nothing", "let")
PRONOUNS_AUX = ("i", "you", "we", "they", "he", "she", "it")
AUX_RESIDUE = ("ll", "re", "ve", "d", "m")
NEGATIVE_STEMS = ("don", "doesn", "didn", "isn", "aren", "wasn", "weren", "won", "can", "couldn", "wouldn", "shouldn", "hasn", "haven",
                  "hadn", "ain", "mustn", "needn")


def test_the_dropped_apostrophe_word_sets_are_exactly_the_reviewed_ones():
    """pr_s_* (15 dropped members), pr_s_add_* (my, the), pr_aux_* (you we they he she it), pr_aux_add_who, ar_* (re ve d m), ar_add_t,
    ns_* (17 stems), ns_add_* (do, will): every member of every set, and no other word. The behavioural tests below take their members
    from the literals above, so a member dropped from the constant still fails them."""
    assert A.PRONOUNS_S == frozenset(PRONOUNS_S)
    assert A.PRONOUNS_AUX == frozenset(PRONOUNS_AUX)
    assert A.AUX_RESIDUE == frozenset(AUX_RESIDUE)
    assert A.NEGATIVE_STEMS == frozenset(NEGATIVE_STEMS)
    assert (len(PRONOUNS_S), len(PRONOUNS_AUX), len(AUX_RESIDUE), len(NEGATIVE_STEMS)) == (18, 7, 5, 18)


@pytest.mark.parametrize("pron", PRONOUNS_S)
def test_every_pronoun_followed_by_a_lone_s_is_apostrophe_residue(pron):
    """pr_s_*: "She s Dream" reads "She's Dream" (= she is). Each of the 18 reviewed pronouns, capitalised and lower-case, rejects."""
    for sentence in ("%s s Dream uses Baz." % pron.capitalize(), "%s s Dream uses Baz." % pron):
        _rejected(sentence, _u("%s_s_dream" % pron), "holds a lone lower-case 's'", RESIDUE)


@pytest.mark.parametrize("pron", PRONOUNS_AUX)
@pytest.mark.parametrize("res", AUX_RESIDUE)
def test_every_pronoun_followed_by_an_auxiliary_residue_is_apostrophe_residue(pron, res):
    """pr_aux_* ar_* sw43: "You ll Dream" reads "You'll", "I m Dream" reads "I'm". All 7 x 5 reviewed pairs reject (re / ve / d / m
    were unpinned for all but 'i' / 'll'), capitalised and lower-case."""
    for sentence in ("%s %s Dream uses Baz." % (pron.capitalize(), res), "%s %s Dream uses Baz." % (pron, res)):
        _rejected(sentence, _u("%s_%s_dream" % (pron, res)), "holds a lone lower-case %r" % res, RESIDUE)


@pytest.mark.parametrize("stem", NEGATIVE_STEMS)
def test_every_negative_stem_followed_by_a_lone_t_is_apostrophe_residue(stem):
    """ns_*: "Doesn t Dream" is "Doesn't" with the apostrophe dropped. Each of the 18 reviewed stems rejects, either case of the 't'."""
    for t in ("t", "T"):
        _rejected("%s %s Dream uses Baz." % (stem.capitalize(), t), _u("%s_t_dream" % stem), "holds a lone lower-case %r" % t, RESIDUE)


@pytest.mark.parametrize("sentence,slug", [
    ("My s Dream uses Baz.", "my_s_dream"),                 # pr_s_add_my   : 'my' is not a pronoun of the 's set
    ("The s Dream uses Baz.", "the_s_dream"),               # pr_s_add_the
    ("I t Dream uses Baz.", "i_t_dream"),                   # ar_add_t      : 't' is not an auxiliary residue (it follows a negative stem)
    ("Do t Dream uses Baz.", "do_t_dream"),                 # ns_add_do     : 'do' is not a negative stem (only 'don')
    ("Will t Dream uses Baz.", "will_t_dream"),             # ns_add_will   : 'will' is not a negative stem ('won')
    ("Who ll Dream uses Baz.", "who_ll_dream"),             # pr_aux_add_who: 'who' takes 's' but is not in the 'll set
    ("We s Dream uses Baz.", "we_s_dream"),                 # 's' follows the 's set only ('we' is in the aux set)
    ("Don d Dream uses Baz.", "don_d_dream"),               # a negative stem takes 't' only
])
def test_a_pair_outside_the_reviewed_sets_is_kept(sentence, slug):
    """pr_s_add_* pr_aux_add_who ar_add_t ns_add_* (RECORDED BOUNDARY, not an endorsement: "Who ll" is "Who'll" in English, but the
    reviewed 'll set is I / you / we / they / he / she / it). A name that is not one of the reviewed pronoun + residue pairs is kept; the
    exact-set test above is what makes a change to a set a deliberate one."""
    r = _kept(sentence, _u(slug))
    assert r["props"] == [[slug, "uses", "baz"]]


# ========================================================================================================== the verbatim slug (sw52)
def test_the_slug_written_verbatim_is_one_identifier_edge_underscore_included():
    """sw52 (RECORDED DESIGN CHOICE, not a fix): the verbatim test compares the span as written with the slug, so a gathered slug with
    an edge underscore written exactly is one identifier and is not split into the clause word "is". Only `SPAN_SPACES` are stripped from
    the span first; stripping underscores too would turn the verbatim "_is" into the word "is". No store slug has an edge underscore (the
    store's `sanitize` strips it), so this is about the helper's contract, which the pin keeps explicit."""
    r = _kept("_is uses Baz.", _u("_is"))
    assert r["props"] == [["_is", "uses", "baz"]]


# ============================================================================================ reply level: a fact stated twice
R1 = "Spain borders France."
R2 = "Spain borders Portugal."


@pytest.mark.parametrize("reply,pair", [
    ("%s %s %s" % (R1, R2, R1), (1, 3)),                   # rv02: the repeat is NOT adjacent
    ("%s\n%s\r\n%s" % (R1, R2, R1), (1, 3)),
    ("%s %s %s %s" % (R1, R2, R2, R1), (2, 3)),            # the first repeat found, scanning in order
    ("%s %s %s" % (R2, R1, R2), (1, 3)),
])
def test_a_fact_stated_twice_in_one_reply_rejects_even_when_the_repeat_is_not_adjacent(reply, pair):
    """rv02: the list rule across sentences compares every sentence with ALL earlier ones, not only the one before. Each sentence is
    kept on its own (the pure check has no reply context); the reply is rejected."""
    for sentence in (R1, R2):
        assert A.allowlist_check(sentence, SPAIN)["keep"] is True
    v = A.allowlist_verify(reply, SPAIN)
    assert v["accepted"] is False and "a reply that states one fact twice can read as two distinct things" in v["reject_reason"], \
        v["reject_reason"]
    assert "sentences %d and %d both state (spain, borders, " % pair in v["reject_reason"], v["reject_reason"]
    assert all(t["keep"] for t in v["trace"])


def test_the_first_repeated_fact_of_a_longer_reply_is_named_with_its_two_positions():
    reply = "%s %s Portugal borders Spain. %s" % (R1, R2, R2)
    g = SPAIN + [["portugal", "borders", "spain"]]
    v = A.allowlist_verify(reply, g)
    assert v["accepted"] is False and "sentences 2 and 4 both state (spain, borders, portugal)" in v["reject_reason"], v["reject_reason"]


def test_distinct_facts_in_any_number_of_sentences_are_kept():
    """rv02 control: no fact twice (adjacent or not) -> accepted, and every proposition is grounded once."""
    g = SPAIN + [["portugal", "borders", "spain"], ["france", "borders", "spain"]]
    reply = "%s %s Portugal borders Spain. France borders Spain." % (R1, R2)
    v = A.allowlist_verify(reply, g)
    assert v["accepted"] is True and v["reject_reason"] is None
    assert [t[:3] for t in v["grounded"]] == [["spain", "borders", "france"], ["spain", "borders", "portugal"],
                                              ["portugal", "borders", "spain"], ["france", "borders", "spain"]]


# =========================================================================================================== DROPPED_READINGS['eat']
EAT_DROPPED = ("consume", "consumed", "consumes", "consuming", "devour", "devoured", "devouring", "devours")


def test_the_dropped_readings_table_is_exactly_the_eight_synonyms_of_eat():
    """dr_*: the existing per-reading test iterates the table it checks, so a synonym removed from the table is never tried. The table
    is pinned here from a literal."""
    assert T.DROPPED_READINGS == {"eat": frozenset(EAT_DROPPED)}
    assert set(EAT_DROPPED) <= set(T.RELATIONS["eat"].readings)
    for tok in EAT_DROPPED:
        assert T.reading_dropped("eat", tok) is True
    for tok in ("eat", "eats", "ate", "eaten", "eating"):
        assert T.reading_dropped("eat", tok) is False, tok


@pytest.mark.parametrize("tok", EAT_DROPPED)
def test_each_synonym_of_eat_is_rejected_as_meaning_drift_and_eats_is_kept(tok):
    """dr_consume .. dr_devours: "Cat consumes Mouse." (also: uses up) and "Cat devours Mouse." (eats greedily) do not paraphrase back to
    exactly (cat, eat, mouse): rejected with MEANING-DRIFT in the reason, in all four spellings of the sentence."""
    g = [["cat", "eat", "mouse"]]
    _kept("Cat eats Mouse.", g)
    for sentence in ("Cat %s Mouse." % tok, "The cat %s the mouse." % tok, "A cat %s a mouse." % tok, "cat %s mouse." % tok):
        _rejected(sentence, g, "MEANING-DRIFT", "%r reads eat" % tok)
