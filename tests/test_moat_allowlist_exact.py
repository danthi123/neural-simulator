"""Pins for build 4 of the strict allowlist claim checker (`BRAIN_MOAT_ALLOWLIST`, default OFF): FOLD TO FIND, EXACT TO ACCEPT.

Round 5 (research/findings/raw/_moat_allowlist_hunt4/r5/) kept 24 of 89 adjudicated-U cases at all seven seeds on build 3. Every one went
through ONE step: a SUBJ/OBJ span was accepted because the store's lossy slug fold mapped it onto a gathered slug ("George V" + U+013A
reads as George VI but folds to george_v; "1-500 m" folds to 1_500_m). Build 4 keeps the fold only to FIND which gathered slug a token
run could be. A span is ACCEPTED only if its displayed text, by `surface_form` (NFC, lowercase, whitespace runs -> one space,
stripped), equals one of the slug's canonical surfaces: the slug verbatim, its de-slugged form, or one of that entity's own store names
whose sanitize() is the slug (research/findings/raw/_moat_allowlist_names/names_table.json, made by
research/runners/_moat_allowlist_names_build.py). Relation words are matched with no fold, and a non-ASCII character outside the accepted
entity spans rejects. Build 3's case rule and the per-reading relation rules are unchanged and apply on top.

These tests pin: the 24 round-5 leaks rejected with the span and slug named (pure function and the production claim gate at all seven
parser seeds); real accented / hyphenated store names accepted when written exactly as a store name and rejected when one character
differs; the de-slugged form and the slug verbatim accepted; hyphen-for-space rejected unless a store name has the hyphen; non-ASCII text
in the relation words or between words rejected; a missing table fails closed; the table's shape.
Review rulings (2026-10-20): F1 rows only from the ONE Wikidata id carrying a slug's store facts (the demonstrated j_n / saint_cloud leaks
rejected at seven seeds through the claim gate and the composer entry; the table's own id data checked, and against wikidata5m where
present); F2 the ", " -> " " rows (Z39 kept); F3 the whitespace rule, every listed character inside and outside a span; F4 no Cf/Cc surface.

Run: CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m pytest tests/test_moat_allowlist_exact.py -q
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest

from research.runners import _moat_allowlist as A
from research.runners._knowledge_core_curate import sanitize
from research.runners.brain_conversational_agent import BridgeParser

pytestmark = pytest.mark.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
R5 = ROOT / "research" / "findings" / "raw" / "_moat_allowlist_hunt4" / "r5"
SEEDS = (42, 43, 44, 100, 101, 102, 1)
FLAG = "BRAIN_MOAT_ALLOWLIST"
ES = [["spain", "shares_border_with", "france"], ["spain", "member_of", "european_union"]]
_SPAN = re.compile(r"entity span '(.+?)' is not an exact store surface of (\S+) ")

# the U cases build 3 kept at all seven seeds in round 5 (results_build3.json; the list is checked against that file below)
R5_LEAKS = ("R5-DL01 R5-DL02 R5-DQ01 R5-IN01 R5-NX01 R5-NX02 R5-NX03 R5-NX04 R5-NX05 R5-NX07 R5-NX08 R5-NX09 R5-NX10 R5-NX11 R5-NX12 "
            "R5-NX13 R5-NX14 R5-NX15 R5-NX16 R5-NX18 R5-NX19 R5-NX20 R5-NX21 R5-NX23").split()
CASES = {c["id"]: c for c in json.loads((R5 / "cases.json").read_text(encoding="utf-8"))}


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


# ============================================================================================================ round-5 leaks
def test_the_leak_list_is_build3s_kept_u_set():
    doc = json.loads((R5 / "results_build3.json").read_text(encoding="utf-8"))
    ent = doc["summary"]["allowlist"]["entries"]
    kept = set(ent["gate"]["U"]["kept_ids"]) | set(ent["composer"]["U"]["kept_ids"])
    assert sorted(kept) == sorted(R5_LEAKS) and len(R5_LEAKS) == 24


@pytest.mark.parametrize("cid", R5_LEAKS)
def test_every_round5_leak_is_rejected_naming_the_span_and_slug(cid, seed_chats):
    c = CASES[cid]
    r = A.allowlist_check(c["sentence"], c["gathered"])
    assert not r["keep"], (cid, r["reason"])
    m = _SPAN.search(r["reason"])
    assert m, (cid, r["reason"])
    span, slug = m.group(1), m.group(2)
    assert span in unicodedata.normalize("NFC", c["sentence"]), (cid, span)
    assert slug in {t for f in c["gathered"] for t in (f[0], f[2])}, (cid, slug)
    assert A.surface_form(span) not in A.names_table().surfaces(slug)
    # the production claim gate, flag ON, every parser seed: rejected with the same reason
    got = _gate_all_seeds(seed_chats, c["sentence"], c["gathered"])
    assert all(acc is False and r["reason"] in why for acc, why in got), (cid, got)


# ==================================================================================== real store names: exact, or rejected
# (slug, a store name of it as written in the names table, the same name with ONE character changed so that it still folds to the slug)
REAL = [
    ("conf_d_ration_suisse", "Confédération suisse", "Confèdération suisse"),               # accent inside a word
    ("koninkrijk_belgi", "Koninkrijk België", "Koninkrijk Belgiê"),                          # accent at the end (strip('_'))
    ("magyar_k_zt_rsas_g", "Magyar Köztársaság", "Magyar Köztàrsaság"),
    ("k_br_s_futbol_federasyonu", "Kıbrıs Futbol Federasyonu", "Kıbrís Futbol Federasyonu"),   # dotless i
    ("auvergne_rh_ne_alpes", "Auvergne-Rhône-Alpes", "Auvergne Rhône-Alpes"),               # hyphenated region with an accent
    ("humboldt_universit_t_zu_berlin", "Humboldt-Universität zu Berlin", "Humboldt-Universitát zu Berlin"),
    ("universit_paris_i_panth_on_sorbonne", "Université Paris I Panthéon-Sorbonne", "Université Paris Iĺ Panthéon-Sorbonne"),
    ("danubian_carpathian_area", "Danubian-Carpathian area", "Danubian-Carpathian-area"),    # hyphenated, ASCII
    ("gabriel_garc_a_m_rquez", "Gabriel García Márquez", "Gabriel Garcïa Márquez"),
]


@pytest.mark.parametrize("slug,name,variant", REAL, ids=[r[0] for r in REAL])
def test_a_real_store_name_is_kept_exactly_and_rejected_when_one_character_differs(slug, name, variant):
    table = A.names_table()
    assert table.error is None, table.error
    assert sanitize(name) == slug and A.surface_form(name) in table.entities[slug], (slug, name)
    assert sanitize(variant) == slug or "-" in variant, (slug, variant)        # the variant still spells the slug by the fold
    assert A.surface_form(variant) not in table.surfaces(slug)
    for fmt, g in (("%s has part Zeta Q.", [[slug, "has_part", "zeta_q"]]), ("Zeta Q has part %s.", [["zeta_q", "has_part", slug]])):
        assert A.allowlist_keep(fmt % name, g) == (True, "ok"), (fmt % name)
        keep, why = A.allowlist_keep(fmt % variant, g)
        assert not keep and ("entity span %r" % variant) in why and slug in why, (fmt % variant, why)
    # case and Unicode normal form are not differences: NFD and upper/lower case of the same name are the same text
    nfd = unicodedata.normalize("NFD", name)
    assert A.allowlist_keep("%s has part Zeta Q." % nfd, [[slug, "has_part", "zeta_q"]]) == (True, "ok")


def test_the_real_names_cover_accent_hyphen_and_both():
    names = [n for _s, n, _v in REAL]
    assert sum(1 for n in names if not n.isascii()) >= 5
    assert sum(1 for n in names if "-" in n) >= 3
    assert sum(1 for n in names if "-" in n and not n.isascii()) >= 2


@pytest.mark.parametrize("slug,name,variant", REAL, ids=[r[0] for r in REAL])
def test_the_de_slugged_form_and_the_slug_verbatim_are_kept(slug, name, variant):
    for text in (A.deslug(slug), A.deslug(slug).title(), slug):
        assert A.allowlist_keep("%s has part Zeta Q." % text, [[slug, "has_part", "zeta_q"]]) == (True, "ok"), text


def test_a_slug_with_no_row_accepts_only_its_de_slugged_form_and_itself():
    assert "george_v" not in A.names_table().entities
    g = [["george_v", "member_of", "british_royal_society"]]
    for p in ("George V is a member of British Royal Society.", "george_v is a member of british_royal_society."):
        assert A.allowlist_keep(p, g) == (True, "ok"), p
    for p in ("George-V is a member of British Royal Society.", "George V is a member of British-Royal Society.",
              "George  V is a member of British Royal-Society.",
              "George Vĺ is a member of British Royal Society.", "George ᐯ is a member of British Royal Society.",
              "George Ⅴ is a member of British Royal Society."):
        assert not A.allowlist_keep(p, g)[0], p


# ============================================================================================================ hyphen / space
def test_hyphen_for_space_is_rejected_unless_a_store_name_has_the_hyphen():
    # a store name has the hyphen: kept; the de-slugged (space) form: kept
    for p, slug in (("for-profit has part Zeta Q.", "for_profit"), ("for profit has part Zeta Q.", "for_profit"),
                    ("Danubian-Carpathian area has part Zeta Q.", "danubian_carpathian_area"),
                    ("Danubian Carpathian area has part Zeta Q.", "danubian_carpathian_area")):
        assert A.allowlist_keep(p, [[slug, "has_part", "zeta_q"]]) == (True, "ok"), p
    # no store name has this hyphen
    for p, slug, span in (("Spain is a member of European-Union.", "european_union", "European-Union"),
                          ("Danubian Carpathian-area has part Zeta Q.", "danubian_carpathian_area", "Danubian Carpathian-area"),
                          ("Non metropolitan-districts has part Zeta Q.", "non_metropolitan_districts", "Non metropolitan-districts")):
        g = ES if slug == "european_union" else [[slug, "has_part", "zeta_q"]]
        keep, why = A.allowlist_keep(p, g)
        assert not keep and ("entity span %r" % span) in why and slug in why, (p, why)


def test_only_the_four_display_identical_spaces_collapse_inside_an_entity_span():
    """F3: inside an entity span U+0020 and exactly U+00A0, U+2007, U+2009, U+202F count as one space; outside the spans only U+0020."""
    g = [["george_v", "member_of", "british_royal_society"]]
    assert A.allowlist_keep("George  V is a member of British Royal Society.", g) == (True, "ok")
    assert A.allowlist_keep("George V  is  a member of British Royal Society.", g) == (True, "ok")       # U+0020 runs outside
    for ws in SPAN_SPACES:
        assert A.allowlist_keep("George%sV is a member of British%sRoyal Society." % (ws, ws), g) == (True, "ok"), hex(ord(ws))
        for p in ("George V is a member%sof British Royal Society." % ws, "George V is%sa member of British Royal Society." % ws,
                  "George V%sis a member of British Royal Society." % ws):
            keep, why = A.allowlist_keep(p, g)
            assert not keep and "non-ASCII character" in why, (hex(ord(ws)), why)
    for zw in (chr(0x200B), chr(0x200C), chr(0x2060), chr(0xFEFF)):           # zero-width: not whitespace
        assert not A.allowlist_keep("George%s V is a member of British Royal Society." % zw, g)[0], hex(ord(zw))


# F3: every listed character, and more of "the rest" (other Zs spaces, CR, LF, NUL, DEL, C1 controls)
SPAN_SPACES = (chr(0xA0), chr(0x2007), chr(0x2009), chr(0x202F))
F3_LISTED = ["\t", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x1f", "\x85", "\u1680", "\u2028", "\u2029", "\u3000"]
F3_REST = ["\r", "\n", "\x00", "\x7f", "\x9f", "\u2002", "\u2003", "\u200a", "\u205f", "\u2000", "\u2001", "\u2008"]


@pytest.mark.parametrize("ch", F3_LISTED + F3_REST, ids=["U+%04X" % ord(c) for c in F3_LISTED + F3_REST])
def test_any_other_whitespace_or_control_character_rejects_inside_or_outside_a_span(ch):
    g = [["george_v", "member_of", "british_royal_society"]]
    for p in ("George%sV is a member of British Royal Society." % ch,          # inside the SUBJ span
              "George V is a member of British%sRoyal Society." % ch,          # inside the OBJ span
              "George V is a member%sof British Royal Society." % ch,          # inside the relation words
              "George V%sis a member of British Royal Society." % ch,          # between a name and the copula
              "The%sGeorge V is a member of British Royal Society." % ch):     # between the determiner and a name
        r = A.allowlist_check(p, g)
        assert not r["keep"] and r["reason"].startswith("allowlist: whitespace or control character"), (hex(ord(ch)), p, r["reason"])
        assert ("U+%04X" % ord(ch)) in r["reason"]
        v = A.allowlist_verify(p, g)                                          # the production entry: same reject (mid-sentence)
        assert v["accepted"] is False and "whitespace or control character" in v["reject_reason"], (hex(ord(ch)), v["reject_reason"])


def test_sentence_splitting_is_unchanged_by_f3():
    """Whitespace after a sentence-final '.' is consumed by the splitter (unchanged), and each piece is stripped."""
    assert A.split_sentences("Spain shares border with France.\tSpain is a member of the European Union.") == [
        "Spain shares border with France.", "Spain is a member of the European Union."]


_TWO = ("Spain shares border with France.", "Spain is a member of the European Union.")


@pytest.mark.parametrize("gap", [" ", "\n", "\r\n", " \n ", "\n\n"])
def test_plain_space_and_line_breaks_between_sentences_are_kept(gap):
    """F3 (Claude ruling after the correctness re-review): between sentences and at the reply's edges, U+0020, LF and CR are
    allowed (real mouth output breaks lines between sentences)."""
    assert A.allowlist_verify(_TWO[0] + gap + _TWO[1], ES)["accepted"] is True
    assert A.allowlist_verify(gap + _TWO[0] + gap, ES)["accepted"] is True


@pytest.mark.parametrize("ch", ["\t", "\x0b", "\x0c", "\x1c", "\x1f", "\u0085", "\u1680", "\u2028", "\u2029", "\u3000",
                                "\u00a0", "\u2009"])
def test_other_whitespace_between_sentences_or_at_an_edge_rejects(ch):
    """F3: any other whitespace or control character the splitter would consume rejects the reply, wherever it sits."""
    for prose in (_TWO[0] + ch + _TWO[1], _TWO[0] + " " + ch + _TWO[1], ch + _TWO[0], _TWO[0] + ch):
        r = A.allowlist_verify(prose, ES)
        assert r["accepted"] is False and "between sentences or at an edge" in r["reject_reason"], (repr(prose), r)


# ========================================================================================= no fold on relation words, ASCII outside
@pytest.mark.parametrize("prose,gathered,needle", [
    ("Spain shares border withĺ France.", ES, "relation span 'shares border withĺ'"),
    ("Spain sharesĺ border with France.", ES, "relation span 'sharesĺ border with'"),
    ("Spain is a member ofĺ European Union.", ES, "relation span 'member ofĺ'"),
    ("VI Olympic Winter Games followsĺ X Games.", [["vi_olympic_winter_games", "follows", "x_games"]], "relation span 'followsĺ'"),
    ("Spain shares-border-with France.", ES, "relation span 'shares-border-with'"),
    ("Spain" + chr(0xA0) + "shares border with France.", ES, "non-ASCII character"),
    ("Spain shares border with the" + chr(0xA0) + "France.", ES, "non-ASCII character"),
    ("Spain shares border" + chr(0xA0) + "with France.", ES, "non-ASCII character"),
    ("Spain shares border with France." + chr(0x3000), ES, "whitespace or control character"),        # F3
    ("Spain shаres border with France.", ES, "non-Latin"),                    # Cyrillic a
    ("Spain shares border with thé France.", ES, "no shape"),
    ("Spain shares border with France．", ES, "no shape"),                    # fullwidth full stop
])
def test_non_ascii_in_the_relation_words_or_between_words_rejects(prose, gathered, needle):
    keep, why = A.allowlist_keep(prose, gathered)
    assert not keep and needle in why, (prose, why)


def test_the_plain_relation_words_are_kept():
    assert A.allowlist_keep("Spain shares border with France.", ES) == (True, "ok")
    assert A.allowlist_keep("Spain shares_border_with France.", ES) == (True, "ok")       # the relation slug verbatim
    assert A.allowlist_keep("Spain is a member of the European Union.", ES) == (True, "ok")
    assert A.allowlist_keep("VI Olympic Winter Games follows X Games.", [["vi_olympic_winter_games", "follows", "x_games"]]) == (True, "ok")


# ============================================================================================================ fail closed
def test_a_missing_names_table_accepts_only_the_slug_and_its_de_slugged_form(monkeypatch, tmp_path):
    monkeypatch.setattr(A, "NAMES_TABLE_PATH", str(tmp_path / "absent.json"))
    A.names_table.cache_clear()
    try:
        g = [["koninkrijk_belgi", "has_part", "zeta_q"]]
        keep, why = A.allowlist_keep("Koninkrijk België has part Zeta Q.", g)
        assert not keep and "names table unreadable" in why and "entity span 'Koninkrijk België'" in why, why
        assert A.allowlist_keep("Koninkrijk belgi has part Zeta Q.", g) == (True, "ok")
        assert A.allowlist_keep("koninkrijk_belgi has part Zeta Q.", g) == (True, "ok")
        assert A.allowlist_keep("Danubian-Carpathian area has part Zeta Q.", [["danubian_carpathian_area", "has_part", "zeta_q"]])[0] is False
    finally:
        A.names_table.cache_clear()


def test_an_unreadable_names_table_fails_closed(monkeypatch, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"entities": [1, 2', encoding="utf-8")
    monkeypatch.setattr(A, "NAMES_TABLE_PATH", str(bad))
    A.names_table.cache_clear()
    try:
        assert A.names_table().error and A.names_table().entities == {}
        keep, why = A.allowlist_keep("Koninkrijk België has part Zeta Q.", [["koninkrijk_belgi", "has_part", "zeta_q"]])
        assert not keep and "names table unreadable" in why, why
    finally:
        A.names_table.cache_clear()


# ============================================================================================================ the table
def test_the_names_table_is_small_exceptions_only_and_spells_its_slugs():
    path = Path(A.NAMES_TABLE_PATH)
    assert path.stat().st_size < 800_000                 # with row_ids and multi_id_slugs (F1) and the comma rows (F2)
    assert (path.parent / (path.name + ".prov.json")).exists()
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["reproduction"]["identical"] is True and doc["reproduction"]["store_facts"] == 78857
    ents = doc["entities"]
    assert doc["n_rows"] == len(ents) and doc["n_surfaces"] == sum(len(v) for v in ents.values())
    store_slugs = set(json.loads(Path(A.CAPS_TABLE_PATH).read_text(encoding="utf-8"))["entities"])
    assert set(ents) <= store_slugs
    for slug, forms in ents.items():
        assert forms and forms == sorted(set(forms)), slug
        for f in forms:
            assert f == A.surface_form(f) and f != A.deslug(slug), (slug, f)
            assert sanitize(f) == slug, (slug, f)
    assert A.names_table() is A.names_table()


def test_surface_form_is_only_nfc_lower_and_whitespace():
    assert A.surface_form("  Koninkrijk" + chr(0xA0) + "Belgie" + chr(0x308) + "  ") == "koninkrijk belgië"
    assert A.surface_form("A" + chr(0xA0) + chr(0x2009) + chr(0x2007) + chr(0x202F) + "B") == "a b"
    assert A.surface_form("A\tB") == "a\tb" and A.surface_form("A" + chr(0x3000) + "B") == "a" + chr(0x3000) + "b"   # F3: not collapsed
    assert A.surface_form("A" + chr(0x200B) + "B") == "a" + chr(0x200B) + "b"      # zero-width is not whitespace
    assert A.surface_form("Auvergne-Rhône") == "auvergne-rhône"
    assert A.surface_form("ﬁ") == "ﬁ"                      # no NFKC
    assert A.deslug("1_500_m") == "1 500 m"


# ============================================================================== F1: one carrier id per row (no re-imported fold)
J = ["j_n", "language_of_work_or_name", "slenska_language"]
SC = ["saint_cloud", "country", "u_s_of_a"]
F1_LEAKS = [   # (sentence, gathered, the span that must be named, its slug): review 2026-10-20, kept by the first build-4 table
    ("Ján language of work or name Íslenska language.", [J], "Ján", "j_n"),
    ("Ján language of work or name slenska_language.", [J], "Ján", "j_n"),
    ("Saint-Cloud country u s of a.", [SC], "Saint-Cloud", "saint_cloud"),
    ("Ján language of work or name Íslenska language.", [J, ["jean_marie", "language_of_work_or_name", "french_language"]], "Ján", "j_n"),
    ("Ján language of work or name slenska_language.", [J, ["jean_marie", "language_of_work_or_name", "french_language"]], "Ján", "j_n"),
    ("Saint-Cloud country u s of a.", [SC, ["bernard_blier", "place_of_death", "saint_cloud"]], "Saint-Cloud", "saint_cloud"),
]


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


@pytest.mark.parametrize("sentence,gathered,span,slug", F1_LEAKS, ids=["L%d" % i for i in range(len(F1_LEAKS))])
def test_every_f1_leak_is_rejected_at_all_seven_seeds_through_both_entries(seed_chats, sentence, gathered, span, slug):
    r = A.allowlist_check(sentence, gathered)
    assert not r["keep"] and ("entity span %r" % span) in r["reason"] and (" %s " % slug) in r["reason"], r["reason"]
    got = _gate_all_seeds(seed_chats, sentence, gathered)
    assert all(acc is False and r["reason"] in why for acc, why in got), got
    assert [_composer_entry(seed_chats[sd], sentence, gathered) for sd in SEEDS] == [False] * len(SEEDS)


def test_the_demonstrated_merges_follow_the_carrier_rule():
    """F1. saint_cloud: the store facts under it come from two ids (the French commune Q189153, St. Cloud, Minnesota Q736834), so it has
    no row and accepts only its slug and de-slugged form. j_n: the alias map gave it two ids (Jan with an acute, Q6319579; Jon with an
    acute, Q1715043) but only Q1715043 carries store facts, so its one row is that id's name and the other id's spelling rejects."""
    t = A.names_table()
    assert "saint_cloud" not in t.entities and t.surfaces("saint_cloud") == frozenset(("saint_cloud", "saint cloud"))
    assert A.allowlist_keep("Saint Cloud country u s of a.", [SC]) == (True, "ok")
    assert sorted(t.entities["j_n"]) == ["jón"]
    assert A.allowlist_keep("Jón language of work or name Íslenska language.", [J]) == (True, "ok")
    assert A.allowlist_keep("j n language of work or name slenska_language.", [J]) == (True, "ok")
    assert not A.allowlist_keep("Jân language of work or name slenska_language.", [J])[0]


def test_the_table_records_its_carrier_ids_and_no_row_comes_from_a_multi_id_slug():
    """F1, from the generator's own id data: every row names exactly one carrier id, no row slug is a multi-id slug, and the two
    demonstrated merges are recorded with both ids."""
    doc = json.loads(Path(A.NAMES_TABLE_PATH).read_text(encoding="utf-8"))
    ents, row_ids, multi = doc["entities"], doc["row_ids"], doc["multi_id_slugs"]
    assert set(row_ids) == set(ents)
    assert all(isinstance(v, str) and re.fullmatch(r"Q\d+", v) for v in row_ids.values())
    assert not set(ents) & set(multi)
    assert all(len(v) >= 2 and v == sorted(set(v)) for v in multi.values())
    assert multi["saint_cloud"] == ["Q189153", "Q736834"] and "j_n" not in multi
    assert row_ids["j_n"] == "Q1715043"                    # Q6319579 (Jan with an acute) carries no store fact under j_n
    rep = doc["reproduction"]
    assert rep["identical"] and rep["carriers_cover_every_store_slug"] and rep["carrier_winners_rebuild_the_store"]
    assert doc["steps"]["n_multi_id_slugs"] == len(multi) and doc["steps"]["no_carrier"] == []
    assert doc["steps"]["f1_f4_f2"] == {"rows": doc["n_rows"], "surfaces": doc["n_surfaces"]}


ENTITY_TXT = "/home/dant123/Projects/sim-data/wikidata5m/wikidata5m_entity.txt"


def test_every_row_surface_is_a_name_of_its_one_carrier_id():
    """F1 + F2 + F4 against the source: each surface is the surface_form of a name of row_ids[slug] (sanitize() == slug, no Cf/Cc
    character), or that form with ', ' written ' '."""
    if not os.path.exists(ENTITY_TXT):
        pytest.skip("LOUD SKIP: %s is absent on this node; the id data was still checked in "
                    "test_the_table_records_its_carrier_ids_and_no_row_comes_from_a_multi_id_slug" % ENTITY_TXT)
    from research.runners._moat_allowlist_caps_build import names_of
    from research.runners._moat_allowlist_names_build import comma_derived, has_format_or_control
    doc = json.loads(Path(A.NAMES_TABLE_PATH).read_text(encoding="utf-8"))
    names = names_of(ENTITY_TXT, set(doc["row_ids"].values()))
    for slug, forms in doc["entities"].items():
        own = {A.surface_form(n) for n in names[doc["row_ids"][slug]] if sanitize(n) == slug and not has_format_or_control(n)}
        own |= {comma_derived(f) for f in own if ", " in f}
        assert set(forms) <= own, (slug, sorted(set(forms) - own))


def test_build_rows_takes_names_only_from_a_single_carrier():
    """The generator's row rule on a synthetic merge (no data needed): two ids under one slug -> no row; one carrier among two ids
    named by the alias map -> only that id's names."""
    from research.runners._moat_allowlist_names_build import build_rows
    names = {"Q1": ["Ján", "Jan"], "Q2": ["Jón", "Jon"], "Q3": ["Saint-Cloud"], "Q4": ["St. Cloud", "Saint Cloud"],
             "Q5": ["Sète, France", "Sete"], "Q6": ["Tier" + chr(0x200E) + "park", "Tier park"]}
    ents = {"j_n", "saint_cloud", "s_te_france", "tier_park"}
    carriers = {"j_n": {"Q1", "Q2"}, "saint_cloud": {"Q4"}, "s_te_france": {"Q5"}, "tier_park": {"Q6"}}
    all_ids = {"j_n": {"Q1", "Q2"}, "saint_cloud": {"Q3", "Q4"}, "s_te_france": {"Q5"}, "tier_park": {"Q6"}}
    entities, row_ids, multi, steps = build_rows(ents, carriers, names, all_ids)
    assert multi == {"j_n": ["Q1", "Q2"]} and "j_n" not in entities
    assert "saint_cloud" not in entities                       # Q3's "Saint-Cloud" carries no fact; Q4 has no exception name
    assert entities["s_te_france"] == ["sète france", "sète, france"] and row_ids["s_te_france"] == "Q5"   # F2
    assert "tier_park" not in entities and steps["n_f4_names_excluded"] == 1                                 # F4
    assert steps["before_f1"] == {"rows": 4, "surfaces": 5} and steps["n_f2_surfaces_added"] == 1


# ============================================================================== F2: ", " -> " " rows; F4: no Cf/Cc surfaces
def test_z39_the_accented_name_without_its_comma_is_kept():
    g = [["s_te_france", "located_in_time_zone", "rome_time"], ["s_te_france", "country", "franzoesische_republik"],
         ["s_te_france", "instance_of", "communes_in_france"], ["s_te_france", "located_in_the_administrative_territoria",
                                                                "department_of_herault"]]
    assert "sète france" in A.names_table().entities["s_te_france"]
    assert A.allowlist_keep("Sète France is an instance of communes in France.", g) == (True, "ok")
    assert not A.allowlist_keep("Sète, France is an instance of communes in France.", g)[0]      # the comma stays structural
    assert not A.allowlist_keep("Séte France is an instance of communes in France.", g)[0]        # still exact


def test_no_table_surface_holds_a_format_or_control_character():
    doc = json.loads(Path(A.NAMES_TABLE_PATH).read_text(encoding="utf-8"))
    bad = [(slug, f) for slug, forms in doc["entities"].items() for f in forms
           if any(unicodedata.category(c) in ("Cf", "Cc") for c in f)]
    assert bad == []
    assert doc["steps"]["n_f4_names_excluded"] == len(doc["steps"]["f4_names_excluded"])
