"""Pins for build 3 of the strict allowlist claim checker (`BRAIN_MOAT_ALLOWLIST`, default OFF): NO INVENTED ACRONYMS, the case fidelity
of entity spans (controller ruling 2026-10-20, research/runners/_moat_allowlist.py docstring, docs/drafts/moat-allowlist/SPEC.md).

The store's slug rule lowercases, so the span "the WHO" spells the band's slug the_who while it reads as the World Health Organization
(round-4 case R4-DT02, kept by build 2 at all seven seeds through both entries). Inside every SUBJ/OBJ span an all-caps token of 2+ letters
is kept only if that same token is written in capitals in one of the entity's own store names (the committed table
research/findings/raw/_moat_allowlist_caps/caps_table.json, made by research/runners/_moat_allowlist_caps_build.py from the source
wikidata_100k was built from). These tests pin:
  - R4-DT02 rejected through the production claim gate AND the composer entry at all seven parser seeds, the reason naming 'WHO' and
    the_who; "The Who" / "the who" kept; "THE WHO" / "The WHO" rejected;
  - a programmatic sample (>= 20) of store entities whose canonical label holds an all-caps token: the label as written is kept, an
    all-caps rendering of a label word that no name of the entity writes in capitals is rejected naming that token and the entity;
  - the relation span: an all-caps relation word rejects (no store relation label is written in capitals); tokens outside the spans
    (a determiner, the copula, "and") are not read;
  - a slug outside the table allows no all-caps token, and a missing table fails closed;
  - the table covers every entity and relation of the store (fails if the table misses one), and is small and read once.
Run: CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m pytest tests/test_moat_allowlist_caps.py -q
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest

from research.runners import _moat_allowlist as A
from research.runners.brain_conversational_agent import BridgeParser

pytestmark = pytest.mark.filterwarnings("ignore")

SEEDS = (42, 43, 44, 100, 101, 102, 1)        # parser seeds of one deterministic build: one replicate per seed, not draws
FLAG = "BRAIN_MOAT_ALLOWLIST"
TABLE = json.loads(Path(A.CAPS_TABLE_PATH).read_text(encoding="utf-8"))
STORE_FACTS = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json"
DT02 = "Peter Dennis Blandford Townshend is a member of the WHO."
DT02_G = [["peter_dennis_blandford_townshend", "member_of", "the_who"]]


@pytest.fixture(autouse=True)
def _clean_flags(monkeypatch):
    for k in (FLAG, "BRAIN_MOAT_SLUG_NORM", "BRAIN_CLAIM_MOAT"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def seed_chats():
    """One production ChatBrain method set per parser seed (real BridgeParser, without the heavy ChatBrain.__init__)."""
    from research.runners.brain_chat_tui import ChatBrain
    out = {}
    for sd in SEEDS:
        chat = ChatBrain.__new__(ChatBrain)
        chat.inner = BridgeParser(seed=sd)
        out[sd] = chat
    return out


def _gate(chat, surface, gathered):
    os.environ[FLAG] = "1"
    try:
        chat._claim_verifier_cache = {}
        return chat._verify_claim_set(surface, gathered)
    finally:
        os.environ.pop(FLAG, None)


def _composer_entry(chat, surface, gathered):
    from research.runners.rich_answer_composer import RichAnswerComposer
    comp = RichAnswerComposer.__new__(RichAnswerComposer)
    comp.chat = chat
    os.environ[FLAG] = "1"
    try:
        chat._claim_verifier_cache = {}
        return comp._verify_rendered(surface, None, gathered[0], gathered)
    finally:
        os.environ.pop(FLAG, None)


# =========================================================================================================== R4-DT02 and The Who
def test_r4_dt02_is_rejected_through_the_production_gate_and_the_composer_at_all_seven_seeds(seed_chats):
    assert TABLE["entities"]["the_who"] == [], "the band's store names write no all-caps token"
    for sd in SEEDS:
        acc, res = _gate(seed_chats[sd], DT02, DT02_G)
        assert acc is False, (sd, res)
        why = res["reject_reason"]
        assert "'WHO'" in why and "the_who" in why and "all-caps" in why, (sd, why)
        assert _composer_entry(seed_chats[sd], DT02, DT02_G) is False, sd


@pytest.mark.parametrize("obj", ["The Who", "the who", "the Who", "The who"])
def test_the_band_written_in_its_own_case_is_kept_at_all_seven_seeds(seed_chats, obj):
    s = "Peter Dennis Blandford Townshend is a member of %s." % obj
    assert A.allowlist_keep(s, DT02_G) == (True, "ok")
    assert [_gate(seed_chats[sd], s, DT02_G)[0] for sd in SEEDS] == [True] * len(SEEDS)
    assert [_composer_entry(seed_chats[sd], s, DT02_G) for sd in SEEDS] == [True] * len(SEEDS)


@pytest.mark.parametrize("obj,tok", [("THE WHO", "THE"), ("The WHO", "WHO"), ("the WHO", "WHO"), ("THE who", "THE"), ("the wHO", None)])
def test_an_all_caps_word_inside_the_band_span_is_rejected_naming_it(obj, tok):
    s = "Peter Dennis Blandford Townshend is a member of %s." % obj
    keep, why = A.allowlist_keep(s, DT02_G)
    if tok is None:                                        # "wHO" is not written entirely in capitals: untouched by the rule
        assert keep, why
    else:
        assert not keep and "%r" % tok in why and "the_who" in why, (s, why)


def test_the_subject_span_is_held_to_the_same_rule():
    keep, why = A.allowlist_keep("PETER Dennis Blandford Townshend is a member of The Who.", DT02_G)
    assert not keep and "'PETER'" in why and "peter_dennis_blandford_townshend" in why, why


# =========================================================================================================== caps_tokens
@pytest.mark.parametrize("text,toks", [("WHO", ["WHO"]), ("the WHO", ["WHO"]), ("The Who", []), ("FC Barcelona", ["FC"]),
                                       ("R&B", []), ("U S of A", []), ("MP3", ["MP3"]), ("McDONALD", []), ("ÉCOLE", ["ÉCOLE"]),
                                       ("Linzer ASK", ["ASK"]), ("IX Commonwealth Games", ["IX"]), ("wHO", []), ("X", [])])
def test_caps_tokens_reads_words_of_two_or_more_letters_all_in_capitals(text, toks):
    assert A.caps_tokens(text) == toks


# =========================================================================================================== a sample of store entities
_SPELLABLE = re.compile(r"^[^\W_]+(?:[ \-][^\W_]+)*$")       # letters/digits words joined by one space or hyphen


def _invented(lab, allowed):
    """The first word of `lab` (2+ letters, not written in capitals) whose all-caps form no store name of the entity writes."""
    for w in re.findall(r"[^\W_]+", lab):
        up = w.upper()
        if (sum(c.isalpha() for c in w) >= 2 and not A.caps_tokens(w) and up not in allowed and up.lower() == w.lower()):
            return w
    return None


def _sample():
    """Store entities whose canonical label (as written) holds an all-caps token, picked by a fixed stride over the sorted candidates so
    the sample spreads over the alphabet. A candidate must be structurally checkable: its label spells the slug as one span (the
    lowercased label is kept) and has a word to render in capitals that the entity's names never write so."""
    cands = []
    for slug in sorted(TABLE["labels"]):
        allowed = set(TABLE["entities"][slug])
        for lab in TABLE["labels"][slug]:
            if not (_SPELLABLE.match(lab) and A.caps_tokens(lab)):
                continue
            w = _invented(lab, allowed)
            if w and A.allowlist_keep("%s has part Zeta Q." % lab.lower(), [[slug, "has_part", "zeta_q"]])[0]:
                cands.append((slug, lab, w))
                break
    stride = max(1, len(cands) // 40)
    return cands[::stride][:40]


SAMPLE = _sample()


def test_the_sample_is_at_least_twenty_store_entities():
    assert len(SAMPLE) >= 20, len(SAMPLE)


@pytest.mark.parametrize("slug,lab,word", SAMPLE, ids=[s for s, _l, _w in SAMPLE])
def test_a_store_label_with_its_own_capitals_is_kept_and_an_invented_capital_word_is_rejected(slug, lab, word):
    allowed = set(TABLE["entities"][slug])
    assert set(A.caps_tokens(lab)) <= allowed, (slug, lab)
    g = [[slug, "has_part", "zeta_q"]]
    assert A.allowlist_keep("%s has part Zeta Q." % lab, g) == (True, "ok"), (slug, lab)
    assert A.allowlist_keep("Zeta Q has part %s." % lab, [["zeta_q", "has_part", slug]]) == (True, "ok"), (slug, lab)
    bad = re.sub(r"(?<![^\W_])%s(?![^\W_])" % re.escape(word), word.upper(), lab, count=1)
    for p, gg in (("%s has part Zeta Q." % bad, g), ("Zeta Q has part %s." % bad, [["zeta_q", "has_part", slug]])):
        keep, why = A.allowlist_keep(p, gg)
        assert not keep and "%r" % word.upper() in why and slug in why, (p, why)


def test_fc_is_kept_where_the_store_writes_it():
    """The ruling's example ("FC Barcelona" passes if the store's names for that entity write "FC"); fc_barcelona is not a store entity,
    FC Aberdeen is."""
    slug = "fc_aberdeen"
    assert "fc_barcelona" not in TABLE["entities"]
    assert "FC" in TABLE["entities"][slug] and "ABERDEEN" not in TABLE["entities"][slug]
    g = [[slug, "has_part", "zeta_q"]]
    assert A.allowlist_keep("FC Aberdeen has part Zeta Q.", g) == (True, "ok")
    keep, why = A.allowlist_keep("FC ABERDEEN has part Zeta Q.", g)
    assert not keep and "'ABERDEEN'" in why and slug in why, why
    keep, why = A.allowlist_keep("FC Barcelona has part Zeta Q.", [["fc_barcelona", "has_part", "zeta_q"]])
    assert not keep and "'FC'" in why, why                 # outside the store: no row, no all-caps token allowed


# =========================================================================================================== relation span, scope
_ES = [["spain", "shares_border_with", "france"], ["spain", "member_of", "european_union"]]


@pytest.mark.parametrize("prose,tok", [("Spain SHARES border with France.", "SHARES"), ("Spain shares BORDER WITH France.", "BORDER"),
                                       ("Spain is a MEMBER of the European Union.", "MEMBER"),
                                       ("Spain is a member OF the European Union.", "OF")])
def test_an_all_caps_relation_word_rejects(prose, tok):
    assert TABLE["relations"]["shares_border_with"] == [] and TABLE["relations"]["member_of"] == []
    keep, why = A.allowlist_keep(prose, _ES)
    assert not keep and "%r" % tok in why and "relation span" in why, (prose, why)


def test_a_relation_label_written_with_capitals_keeps_them_and_only_them():
    """Two store relation labels write an all-caps token ("IUCN conservation status", "original language of film or TV show"): those
    tokens are kept in the relation span, any other all-caps relation word still rejects."""
    assert {k: v for k, v in TABLE["relations"].items() if v} == {"iucn_conservation_status": ["IUCN"],
                                                                  "original_language_of_film_or_tv_show": ["TV"]}
    g = [["red_fox", "iucn_conservation_status", "least_concern"], ["iron_man", "original_language_of_film_or_tv_show", "english"]]
    for p in ("Red Fox IUCN conservation status Least Concern.", "Red Fox iucn conservation status Least Concern.",
              "Iron Man original language of film or TV show English.", "Iron Man original language of film or tv show English."):
        assert A.allowlist_keep(p, g) == (True, "ok"), p
    for p, tok in (("Red Fox IUCN CONSERVATION status Least Concern.", "CONSERVATION"),
                   ("Iron Man ORIGINAL language of film or TV show English.", "ORIGINAL")):
        keep, why = A.allowlist_keep(p, g)
        assert not keep and "%r" % tok in why and "relation span" in why, (p, why)


@pytest.mark.parametrize("prose", ["Spain IS a member of the European Union.", "Spain is A member of THE European Union.",
                                   "THE Spain shares border with France.", "Spain shares border with France AND Portugal."])
def test_tokens_outside_the_spans_are_not_read(prose):
    """The ruling covers the SUBJ/OBJ spans and the relation span; a determiner, the copula and "and" are outside them. (Build 5: the
    list names two distinct gathered objects; "France AND France" now rejects as a repeated list.)"""
    assert A.allowlist_keep(prose, _ES + [["spain", "shares_border_with", "portugal"]]) == (True, "ok"), prose


def test_every_parse_of_an_ambiguous_sentence_is_case_checked():
    """With gathered nouns spelled 'the' and 'is', "The IS member of x." has two parses: shape 3 (the, member_of, x), where "IS" is the
    copula (outside every span), and shape 1 (is, member_of, x), where "IS" is the SUBJ span of the slug is (no row: no all-caps token
    allowed). The sentence is kept only if EVERY parse passes, so it is rejected naming 'IS' (a check of the first parse only keeps it)."""
    g = [["the", "member_of", "x"], ["is", "member_of", "x"]]
    keep, why = A.allowlist_keep("The IS member of x.", g)
    assert not keep and "'IS'" in why and " of is:" in why, why
    # build 5: the lower-case sentence is rejected too, by the clause-word rule on the shape-1 parse's SUBJ span "is" (the case check
    # runs first, so the all-caps sentence above still names 'IS')
    keep, why = A.allowlist_keep("The is member of x.", g)
    assert not keep and "holds the clause word 'is'" in why, why


def test_a_slug_outside_the_table_allows_no_all_caps_token():
    g = [["nato_x", "has_part", "zeta_q"]]
    assert "nato_x" not in TABLE["entities"]
    assert A.allowlist_keep("Nato X has part Zeta Q.", g)[0]
    keep, why = A.allowlist_keep("NATO X has part Zeta Q.", g)
    assert not keep and "'NATO'" in why and "nato_x" in why


def test_an_accented_out_of_store_name_with_capitals_is_rejected(monkeypatch):
    """São Paulo FC spells s_o_paulo_fc by the store rule, but that slug is not a store entity, so its "FC" is not allowed (the
    accent-placement pin in tests/test_moat_allowlist.py uses "São Paulo Futebol Clube" since build 3)."""
    from research.runners._knowledge_core_curate import sanitize
    slug = sanitize("S\u00e3o Paulo FC")
    assert slug not in TABLE["entities"]
    keep, why = A.allowlist_keep("S\u00e3o Paulo FC has part Zeta Q.", [[slug, "has_part", "zeta_q"]])
    assert not keep and "'FC'" in why and slug in why, why
    # build 4 (fold to find, exact to accept): the mixed-case form passes the case rule but is no exact surface of an out-of-store
    # slug (its only surfaces are the slug and its de-slugged form), so it is rejected by the exact check, not by the case rule; as if
    # "São Paulo Fc" were one of its store names, it is kept
    keep, why = A.allowlist_keep("São Paulo Fc has part Zeta Q.", [[slug, "has_part", "zeta_q"]])
    assert not keep and "'FC'" not in why and "entity span 'São Paulo Fc'" in why and slug in why, why
    t = A._NamesTable(A.NAMES_TABLE_PATH)
    t.entities[slug] = frozenset({"são paulo fc"})
    monkeypatch.setattr(A, "names_table", lambda: t)
    assert A.allowlist_keep("São Paulo Fc has part Zeta Q.", [[slug, "has_part", "zeta_q"]]) == (True, "ok")
    keep, why = A.allowlist_keep("São Paulo FC has part Zeta Q.", [[slug, "has_part", "zeta_q"]])
    assert not keep and "'FC'" in why, why                 # the case rule still applies on top


def test_a_missing_table_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(A, "CAPS_TABLE_PATH", str(tmp_path / "absent.json"))
    A.caps_table.cache_clear()
    try:
        keep, why = A.allowlist_keep("FC Barcelona has part Zeta Q.", [["fc_barcelona", "has_part", "zeta_q"]])
        assert not keep and "caps table unreadable" in why, why
        assert A.allowlist_keep("Peter Dennis Blandford Townshend is a member of The Who.", DT02_G) == (True, "ok")
    finally:
        A.caps_table.cache_clear()


def test_the_rule_only_turns_a_keep_into_a_reject_the_other_reasons_are_unchanged():
    """The case check runs after the shape and the gathered check, so an earlier reason is reported as before."""
    g = [["spain", "member_of", "european_union"], ["portugal", "member_of", "nato_x"]]
    assert A.allowlist_keep("SPAIN is the member of the European Union.", g)[1].startswith("allowlist: no shape")
    assert A.allowlist_keep("SPAIN is a member of Nato X.", g)[1].startswith("allowlist: not gathered")
    assert A.allowlist_keep("SPAIN is a member of the European Union.", g)[1].startswith("allowlist: all-caps token 'SPAIN'")


# =========================================================================================================== the table
def test_the_table_covers_every_entity_and_relation_of_the_store():
    """Fails if the table misses an entity (or relation) of the store the matcher is used with."""
    if not os.path.exists(STORE_FACTS):
        pytest.skip("the wikidata_100k store is not on this machine")
    rows = json.loads(Path(STORE_FACTS).read_text(encoding="utf-8"))
    ents = {r["fact"][k] for r in rows for k in ("agent", "patient")}
    rels = {r["fact"]["action"] for r in rows}
    missing = sorted(ents - set(TABLE["entities"]))
    assert not missing, (len(missing), missing[:10])
    assert not sorted(rels - set(TABLE["relations"]))
    assert TABLE["n_entities"] == len(TABLE["entities"]) == len(ents)
    assert TABLE["reproduction"]["identical"] is True


def test_the_table_is_compact_and_read_once_per_process():
    assert os.path.getsize(A.CAPS_TABLE_PATH) < 1_200_000
    assert A.caps_table() is A.caps_table() and A.caps_table().error is None
    assert all(t == t.upper() and len([c for c in t if c.isalpha()]) >= 2 for v in TABLE["entities"].values() for t in v)
    assert set(TABLE["labels"]) <= {s for s, v in TABLE["entities"].items() if v}
    assert TABLE["no_names"] == [] and TABLE["picker_drift"] == []
