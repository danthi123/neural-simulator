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


def test_whitespace_runs_collapse_inside_an_entity_span_but_not_outside():
    """'every run of whitespace collapsed to one ASCII space' reads whitespace as Python does (str.isspace): a no-break, thin or
    ideographic space inside a name is a space. A zero-width character is not whitespace, so it stays and rejects. Outside the entity
    spans every character must be ASCII, so the same spaces between the relation words, or before a name, reject."""
    g = [["george_v", "member_of", "british_royal_society"]]
    assert A.allowlist_keep("George  V is a member of British Royal Society.", g) == (True, "ok")
    for ws in (chr(0xA0), chr(0x2009), chr(0x202F), chr(0x3000), " \t"):
        assert A.allowlist_keep("George%sV is a member of British%sRoyal Society." % (ws, ws), g) == (True, "ok"), repr(ws)
        if ws.isascii():                                                       # ASCII whitespace between words is just a space
            assert A.allowlist_keep("George V is a member%sof British Royal Society." % ws, g) == (True, "ok")
            continue
        for p in ("George V is a member%sof British Royal Society." % ws, "George V is%sa member of British Royal Society." % ws):
            keep, why = A.allowlist_keep(p, g)
            assert not keep and "non-ASCII character" in why, (repr(ws), why)
    for zw in (chr(0x200B), chr(0x200C), chr(0x2060), chr(0xFEFF)):           # zero-width: not whitespace
        assert not A.allowlist_keep("George%s V is a member of British Royal Society." % zw, g)[0], hex(ord(zw))


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
    ("Spain shares border with France." + chr(0x3000), ES, "non-ASCII character"),
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
    assert path.stat().st_size < 600_000
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
    assert A.surface_form("  Koninkrijk\tBelgie" + chr(0x308) + "  ") == "koninkrijk belgië"
    assert A.surface_form("A" + chr(0xA0) + chr(0x2009) + "B") == "a b"
    assert A.surface_form("A" + chr(0x200B) + "B") == "a" + chr(0x200B) + "b"      # zero-width is not whitespace
    assert A.surface_form("Auvergne-Rhône") == "auvergne-rhône"
    assert A.surface_form("ﬁ") == "ﬁ"                      # no NFKC
    assert A.deslug("1_500_m") == "1 500 m"
