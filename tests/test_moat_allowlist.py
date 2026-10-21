"""Pins for `BRAIN_MOAT_ALLOWLIST`, the default-OFF STRICT allowlist claim checker on the scaffold mouth
(research/runners/_moat_allowlist.py; spec docs/drafts/moat-allowlist/SPEC.md; bars in
research/findings/2026-10-20-moat-allowlist-claim-checker-PREREGISTRATION.md).

A sentence is kept only if ONE exact proposition shape consumes every token and every proposition it encodes is a gathered
triple of this turn; everything else is rejected. These tests pin:
  - flag OFF: the production claim gate is byte-identical to the unmodified verifier (golden), the call sites route as before,
    and the three flag readers share one truthy contract;
  - flag ON: the matcher owns the whole path (the stock clause parser, `_noun_is_before`, the role-permutation guard and the
    substrate role parse are never consulted; no fallback to the single-triple `_verify`);
  - MUST REJECT: every expect-U case of the B2h verify-go leak hunts and the sweep_real templates over the 90 held-out gathered
    sets (tests/fixtures/moat_allowlist_bank.json), plus the board holes #265 (a mutual shares_border_with pair made the stock
    verifier unbuildable) and #266 (a subject slug that starts with the relation's first three letters);
  - MUST KEEP: the supported twins of cases_b.py (the twin clause as its own sentence; the literal ", and" twin is two
    subjects in one sentence, a multi-clause shape the spec excludes, pinned as an expected loss) and every F-labelled S1/S3
    sentence B2h kept that grounds exactly one triple (tests/fixtures/moat_allowlist_must_keep.json), except the listed losses;
  - shape 3 cannot reverse roles; nothing is dropped; one sentence, one shape;
  - the reviewed per-relation verb-reading table (research/runners/_moat_allowlist_relations.py, prereg AMENDMENT 1 item 2): every
    store relation has a row, the rows' readings are the live verb-form table's, and a ROLE-REVERSING relation (author, father,
    child, parent_taxon, platform), a MEANING-DRIFT relation (build 2: sport, conflict, instrument, spouse, eat) or an unclassified
    one takes no verb reading and no shape 3;
  - build 2: shape 3 takes only a / an or no determiner before the relation noun ("X is the member of Y" asserts uniqueness), and
    SUBJ / OBJ are spelled by the store's own slug rule (`sanitize` in research/runners/_knowledge_core_curate.py), including an
    accented letter at the start or end of a name;
  - build 3 (no invented acronyms, the case fidelity of entity spans) is pinned in tests/test_moat_allowlist_caps.py; three
    assertions here that kept an all-caps span of a name whose store names do not write it so were moved to that rule.
The matcher is a pure host function: its verdict cannot depend on the parser seed. Where a test goes through the production
`ChatBrain` (which carries a seeded BridgeParser) it runs at all seven parser seeds 42, 43, 44, 100, 101, 102, 1.

Run: CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m pytest tests/test_moat_allowlist.py -q
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest

from research.runners import _moat_allowlist as A
from research.runners import _moat_allowlist_relations as T
from research.runners import _moat_claim_entailment_derisk as M
from research.runners.brain_conversational_agent import BridgeParser

pytestmark = pytest.mark.filterwarnings("ignore")

FIX = Path(__file__).parent / "fixtures"
GOLDEN = json.loads((FIX / "moat_claim_entailment_golden.json").read_text())
BANK = json.loads((FIX / "moat_allowlist_bank.json").read_text())
KEEP = json.loads((FIX / "moat_allowlist_must_keep.json").read_text())
SEEDS = (42, 43, 44, 100, 101, 102, 1)        # parser seeds of one deterministic build: one replicate per seed, not draws
FLAG = "BRAIN_MOAT_ALLOWLIST"
SLUG = "BRAIN_MOAT_SLUG_NORM"


@pytest.fixture(autouse=True)
def _clean_flags(monkeypatch):
    for k in (FLAG, SLUG, "BRAIN_CLAIM_MOAT"):
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


@pytest.fixture(scope="module")
def golden_chat():
    from research.runners.brain_chat_tui import ChatBrain
    chat = ChatBrain.__new__(ChatBrain)
    chat.inner = BridgeParser(seed=GOLDEN["seed"])
    return chat


def _on(chat, surface, gathered):
    """The production claim gate with the flag ON -> (accepted, result)."""
    os.environ[FLAG] = "1"
    try:
        return chat._verify_claim_set(surface, gathered)
    finally:
        os.environ.pop(FLAG, None)


def _on_all_seeds(seed_chats, surface, gathered):
    return [_on(seed_chats[sd], surface, gathered)[0] for sd in SEEDS]


def _digest(res):
    return hashlib.sha256(json.dumps(res, sort_keys=True, default=str).encode()).hexdigest()[:20]


# =========================================================================================================== flag OFF
@pytest.mark.parametrize("val", [None, "0", "", "off", "false", "no", "garbage"])
def test_flag_off_production_claim_gate_matches_the_unmodified_verifier_golden(golden_chat, monkeypatch, val):
    """ChatBrain._verify_claim_set (the production gate) with the allowlist flag unset/falsy reproduces the golden captured from
    the UNMODIFIED verifier at 9cf4ec708 (accepted, reject_reason, hash of the full result incl. trace) on all 318 real-world
    sentences (178 real-Qwen + stub + smuggling mutations)."""
    if val is not None:
        monkeypatch.setenv(FLAG, val)
    golden_chat._claim_verifier_cache = {}
    bad = []
    for e in GOLDEN["world"]:
        acc, res = golden_chat._verify_claim_set(e["s"], GOLDEN["gated_sets"][e["g"]])
        if acc is None:
            if e["a"] is not None:
                bad.append((e["s"], "unbuildable now, golden", e["a"]))
            continue
        if (acc, res["reject_reason"], _digest(res)) != (e["a"], e["r"], e["h"]):
            bad.append((e["src"], e["s"], acc, res["reject_reason"]))
    assert not bad, bad[:3]
    assert all(not getattr(v, "allowlist", False) for v in golden_chat._claim_verifier_cache.values() if v is not None)


def test_the_three_flag_readers_agree_and_default_off(monkeypatch):
    from research.runners.brain_chat_tui import ChatBrain
    from research.runners.rich_answer_composer import _moat_allowlist_enabled
    for val in (None, "", "0", "1", "true", "True", " yes ", "on", "ON", "off", "false", "no", "2", "enable", "y"):
        if val is None:
            monkeypatch.delenv(FLAG, raising=False)
        else:
            monkeypatch.setenv(FLAG, val)
        got = {A.allowlist_enabled(), ChatBrain._moat_allowlist_enabled(), _moat_allowlist_enabled()}
        assert len(got) == 1, (val, got)
        assert got == {val is not None and val.strip().lower() in ("1", "true", "on", "yes")}, val
    monkeypatch.delenv(FLAG, raising=False)
    assert A.allowlist_enabled() is False
    assert A.ALLOWLIST_ENV == FLAG


class _RecordingChat:
    """What RichAnswerComposer._verify_rendered touches: records which verifier it is sent to and with which set."""

    def __init__(self, claim=(True, {})):
        self.calls, self.sets = [], []
        self._claim = claim

    def _verify(self, surface, asserted, svo):
        self.calls.append("single")
        return True

    def _verify_claim_set(self, surface, gated):
        self.calls.append("claim")
        self.sets.append(gated)
        return self._claim


def _composer(chat):
    from research.runners.rich_answer_composer import RichAnswerComposer
    comp = RichAnswerComposer.__new__(RichAnswerComposer)
    comp.chat = chat
    return comp


_SVO = ["angora_turkey", "instance_of", "city_work"]
_ANGORA = [["angora_turkey", "located_in_time_zone", "kaliningrad_time"], _SVO,
           ["angora_turkey", "country", "the_republic_of_turkey"], ["angora_turkey", "twinned_administrative_body", "washington_dc"]]


@pytest.mark.parametrize("val", [None, "0", "off", "garbage"])
def test_flag_off_composer_routes_exactly_as_before(monkeypatch, val):
    if val is not None:
        monkeypatch.setenv(FLAG, val)
    chat = _RecordingChat()
    assert _composer(chat)._verify_rendered("s", None, _SVO, [_SVO]) is True
    assert chat.calls == ["single"]                                           # one-fact turn: the single-triple _verify
    chat = _RecordingChat()
    _composer(chat)._verify_rendered("s", None, _SVO, _ANGORA)
    assert chat.calls == ["claim"]                                            # multi-fact turn: the claim gate
    chat = _RecordingChat(claim=(None, None))
    assert _composer(chat)._verify_rendered("s", None, _SVO, _ANGORA) is True
    assert chat.calls == ["claim", "single"]                                  # unbuildable -> the old fallback
    chat = _RecordingChat()
    assert _composer(chat)._verify_rendered("s", None, _SVO, None) is True
    assert chat.calls == ["single"]                                           # no gathered set -> _verify, as before


def test_flag_off_builds_the_stock_verifier_with_the_old_cache_key(golden_chat):
    golden_chat._claim_verifier_cache = {}
    ver = golden_chat._build_claim_verifier(_ANGORA)
    assert isinstance(ver, M.ClaimEntailmentVerifier) and ver.slug_norm is False
    assert list(golden_chat._claim_verifier_cache) == [frozenset(tuple(f) for f in _ANGORA)]


# =========================================================================================================== flag ON wiring
@pytest.mark.parametrize("val", ["1", "true", "TRUE", " on ", "yes"])
def test_flag_on_routes_every_turn_to_the_allowlist_with_no_fallback(monkeypatch, val):
    monkeypatch.setenv(FLAG, val)
    for gated in ([_SVO], _ANGORA):
        chat = _RecordingChat(claim=(False, {"reject_reason": "x"}))
        assert _composer(chat)._verify_rendered("s", None, _SVO, gated) is False          # its verdict is final
        assert chat.calls == ["claim"]
        chat = _RecordingChat(claim=(None, None))                                          # nothing buildable -> REJECT
        assert _composer(chat)._verify_rendered("s", None, _SVO, gated) is False
        assert chat.calls == ["claim"]                                                     # never the single-triple re-parse
    chat = _RecordingChat()
    assert _composer(chat)._verify_rendered("s", None, _SVO, None) is True
    assert chat.calls == ["claim"] and chat.sets == [[_SVO]]                               # no set given -> [svo]


def test_flag_on_builds_an_always_buildable_allowlist_verifier_under_its_own_cache_key(golden_chat, monkeypatch):
    golden_chat._claim_verifier_cache = {}
    off = golden_chat._build_claim_verifier(_ANGORA)
    monkeypatch.setenv(FLAG, "1")
    on = golden_chat._build_claim_verifier(_ANGORA)
    assert isinstance(on, A.AllowlistClaimVerifier) and on is not off
    assert (frozenset(tuple(f) for f in _ANGORA), "allowlist") in golden_chat._claim_verifier_cache
    assert golden_chat._build_claim_verifier(_ANGORA) is on
    monkeypatch.setenv(SLUG, "1")                                    # the allowlist takes precedence over B2h
    assert golden_chat._build_claim_verifier(_ANGORA) is on
    monkeypatch.delenv(FLAG)
    assert golden_chat._build_claim_verifier(_ANGORA).slug_norm is True
    monkeypatch.delenv(SLUG)
    assert golden_chat._build_claim_verifier(_ANGORA) is off        # the OFF cache entry is reused


def test_flag_on_ignores_the_claim_moat_escape_but_flag_off_keeps_it(golden_chat, monkeypatch):
    monkeypatch.setenv("BRAIN_CLAIM_MOAT", "0")
    assert golden_chat._verify_claim_set("Angora turkey is an instance of city work.", _ANGORA) == (None, None)
    monkeypatch.setenv(FLAG, "1")
    acc, res = golden_chat._verify_claim_set("Angora turkey is an instance of city work.", _ANGORA)
    assert acc is True and res["allowlist"] is True


def test_flag_on_consults_neither_the_stock_parser_nor_the_substrate(seed_chats, monkeypatch):
    """The flag owns the whole path: break every stock/B2h parser entry point and the substrate role parse, and the
    allowlist verdicts are unchanged (so they cannot come from those)."""
    cases = [("Angora turkey is an instance of city work.", True), ("City work is an instance of angora turkey.", False),
             ("The county_of_inyo countrys u_s_of_a.", True), ("Angora turkey country washington dc.", False)]
    gathered = _ANGORA + [["county_of_inyo", "country", "u_s_of_a"]]
    before = [[_on(seed_chats[sd], s, gathered)[0] for s, _ in cases] for sd in SEEDS]

    def boom(*a, **k):
        raise AssertionError("consulted under BRAIN_MOAT_ALLOWLIST")
    for name in ("verify", "_clauses", "_clauses_slug_norm", "_noun_is_before", "_noun_before_verb", "_svo", "_content",
                 "_article_relation_guard", "__init__"):
        monkeypatch.setattr(M.ClaimEntailmentVerifier, name, boom)
    monkeypatch.setattr(BridgeParser, "parse", boom)
    for sd in SEEDS:
        seed_chats[sd]._claim_verifier_cache = {}
        monkeypatch.setattr(seed_chats[sd], "_verify", boom)
    after = [[_on(seed_chats[sd], s, gathered)[0] for s, _ in cases] for sd in SEEDS]
    assert before == after == [[exp for _, exp in cases]] * len(SEEDS)


# =========================================================================================================== MUST REJECT: the bank
def test_bank_is_the_full_port_and_nontrivial():
    """Anti-vacuity + port completeness: every source is present with its sha256, and the expect-U counts are the ones the
    builder ported (any later edit of a source file shows up as a sha mismatch when the builder is re-run)."""
    by = {}
    for c in BANK["cases"]:
        by.setdefault(c["source"], {}).setdefault(c["expect"], 0)
        by[c["source"]][c["expect"]] += 1
    assert by == BANK["counts_by_source"]
    assert {s: by[s].get("U", 0) for s in by} == {"leak_hunt.py": 57, "leak_hunt2.py": 19, "cases_a.py": 105, "cases_b.py": 68,
                                                   "sweep_real.py": 1329}
    assert set(BANK["sources"]) == {"leak_hunt.py", "leak_hunt2.py", "cases_a.py", "cases_b.py", "sweep_real.py"}
    assert len({c["gathered"] for c in BANK["cases"] if c["source"] == "sweep_real.py"}) == 90


@pytest.mark.parametrize("source", ["leak_hunt.py", "leak_hunt2.py", "cases_a.py", "cases_b.py", "sweep_real.py"])
def test_every_expect_u_bank_case_is_rejected_at_every_seed(seed_chats, source):
    leaks = []
    for c in BANK["cases"]:
        if c["source"] != source or c["expect"] != "U":
            continue
        got = _on_all_seeds(seed_chats, c["prose"], BANK["gathered_sets"][c["gathered"]])
        if any(a is not False for a in got):
            leaks.append((c["id"], c["prose"], got))
    assert not leaks, leaks[:5]


def test_the_bank_can_see_a_leak_the_b2h_arm_accepts_some_of_it(seed_chats, monkeypatch):
    """Power control: the same expect-U inputs DO get through the refuted slug-aware arm (B2h, flag BRAIN_MOAT_SLUG_NORM), so a
    0 under the allowlist is the matcher's doing, not a bank that nothing could pass."""
    monkeypatch.setenv(SLUG, "1")
    chat = seed_chats[42]
    chat._claim_verifier_cache = {}
    passed = [c["id"] for c in BANK["cases"] if c["expect"] == "U" and c["source"] in ("cases_a.py", "cases_b.py")
              and chat._verify_claim_set(c["prose"], BANK["gathered_sets"][c["gathered"]])[0]]
    assert len(passed) >= 5, passed


# ---- board hole #265: a mutual shares_border_with pair
_MUTUAL = [["shawano_county", "shares_border_with", "menominee_county"], ["menominee_county", "shares_border_with", "shawano_county"],
           ["shawano_county", "instance_of", "county_of_wisconsin"]]


def test_board_265_mutual_pair_is_unbuildable_for_the_stock_verifier_control(golden_chat):
    golden_chat._claim_verifier_cache = {}
    assert golden_chat._build_claim_verifier(_MUTUAL) is None
    assert golden_chat._verify_claim_set("Shawano County shares border with Menominee County.", _MUTUAL) == (None, None)


@pytest.mark.parametrize("prose,keep", [
    ("Shawano County shares border with Menominee County.", True),
    ("Menominee County shares border with Shawano County.", True),
    ("Shawano County is an instance of county of Wisconsin.", True),
    ("Menominee County is an instance of county of Wisconsin.", False),
    ("Shawano County shares border with county of Wisconsin.", False),
    ("Shawano County shares border with Menominee County and county of Wisconsin.", False),
    ("Menominee County shares border with Shawano County, which is an instance of county of Wisconsin.", False),
])
def test_board_265_mutual_pair_under_the_flag(seed_chats, prose, keep):
    assert _on_all_seeds(seed_chats, prose, _MUTUAL) == [keep] * len(SEEDS), prose


def test_board_265_composer_decides_the_mutual_pair_without_the_single_triple_fallback(seed_chats, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    for sd in SEEDS:
        chat = seed_chats[sd]
        chat._claim_verifier_cache = {}
        monkeypatch.setattr(chat, "_verify", lambda *a, **k: (_ for _ in ()).throw(AssertionError("fallback")))
        comp = _composer(chat)
        assert comp._verify_rendered("Shawano County shares border with Menominee County.", None, _MUTUAL[0], _MUTUAL) is True
        assert comp._verify_rendered("Shawano County shares border with Wausau.", None, _MUTUAL[0], _MUTUAL) is False


# ---- board hole #266: a subject slug that starts with the relation's first three letters
_PREFIX = [("county_of_inyo", "country", "u_s_of_a"), ("sporting_benevento", "sport", "association_football_club"),
           ("gene_wolf", "genre", "sciene_fiction"), ("shawano_county", "shares_border_with", "wausau_wi_msa"),
           ("suburban_train", "subclass_of", "public_transportation"),
           ("insulin_dependent_diabetes_mellitus", "instance_of", "organic_cause"),
           ("headington_united_fc", "headquarters_location", "city_of_dreaming_spires")]


def _stub(svo):
    from research.runners.brain_chat_tui import StubRenderer
    return StubRenderer().render_svo(*svo)[0]


def test_board_266_the_stock_three_letter_match_loses_the_subject_control(golden_chat):
    """Control: the stock verifier's `_noun_is_before` reads 'county_of_inyo' as the verb 'country' (same first 3 letters), so
    the subject is lost and the one-fact stub sentence is rejected ('no resolvable subject')."""
    golden_chat._claim_verifier_cache = {}
    acc, res = golden_chat._verify_claim_set(_stub(_PREFIX[0]), [list(_PREFIX[0])])
    assert acc is False and "no resolvable subject" in res["reject_reason"]


@pytest.mark.parametrize("fact", _PREFIX, ids=lambda f: f[1])
def test_board_266_prefix_subject_under_the_flag(seed_chats, fact):
    s, r, o = fact
    title = lambda x: " ".join(w.capitalize() for w in x.split("_"))
    rel = r.replace("_", " ")
    gathered = [list(fact), ["alpha", "shares_border_with" if r != "shares_border_with" else "instance_of", "beta"],
                [s, "shares_border_with" if r != "shares_border_with" else "instance_of", "beta"]]
    first = "Alpha shares border with Beta" if r != "shares_border_with" else "Alpha instance of beta"
    keep = [_stub(fact), "%s %s %s." % (title(s), rel, title(o)), "%s. %s %s %s." % (first, title(s), rel, title(o))]
    reject = ["%s %s %s." % (title(o), rel, title(s)),                                  # role swap
              "%s %s %s." % (title(s), rel, "Mexico"),                                   # object not gathered
              "%s, and %s %s Beta." % (first, title(s), rel),                            # borrowed-subject form, (s, r, beta) not gathered
              "%s. %s %s Beta." % (first, title(s), rel)]
    # a relation the table gives no verb reading (build 2: sport is MEANING-DRIFT) takes none, so its sentences are rejected and the
    # hole cannot open; the stub sentence's reason names the class
    want = T.has_verb_reading(r)
    for p in keep:
        assert _on_all_seeds(seed_chats, p, gathered) == [want] * len(SEEDS), p
    if not want:
        assert T.relation_class(r) in A.allowlist_keep(_stub(fact), gathered)[1]
    for p in reject:
        assert _on_all_seeds(seed_chats, p, gathered) == [False] * len(SEEDS), p


# =========================================================================================================== MUST KEEP
_TWINS = [c for c in BANK["cases"] if c["source"] == "cases_b.py" and c["expect"] == "S" and c["id"].startswith("pc")]


def test_the_twins_are_all_there():
    assert len(_TWINS) == 26 and all(c["id"].endswith("c") for c in _TWINS)


@pytest.mark.parametrize("case", _TWINS, ids=lambda c: c["id"])
def test_cases_b_supported_twin_is_kept_as_its_own_sentence(seed_chats, case):
    """The twin's purpose (cases_b.py): show the clause about the prefix-colliding subject is not simply rejected. Its clause, as
    a sentence of its own or after the first clause's own sentence, is KEPT at every seed. The literal twin ("<first>, and <S R
    o1>.") puts two subjects in one sentence -- a multi-clause shape the spec excludes -- so it is rejected (an expected loss,
    never loosened for)."""
    g = BANK["gathered_sets"][case["gathered"]]
    first, second = case["prose"].rstrip(".").split(", and ", 1)
    # build 2: a twin whose relation the table denies a verb reading (MEANING-DRIFT / ROLE-REVERSING) is rejected by that rule
    # over the whole table, not per case; which twins those are is pinned in test_twins_lost_to_the_relation_table
    want = case["id"] not in _TWINS_LOST_TO_THE_TABLE
    for p in (second + ".", first + ". " + second + "."):
        assert _on_all_seeds(seed_chats, p, g) == [want] * len(SEEDS), p
    if not want:
        why = A.allowlist_keep(second + ".", g)[1]
        assert any(c in why for c in T.NO_READING_CLASSES), why
    assert _on_all_seeds(seed_chats, case["prose"], g) == [False] * len(SEEDS), case["prose"]


_TWINS_LOST_TO_THE_TABLE = {"pc12c": "sport", "pc13c": "sport"}


def test_twins_lost_to_the_relation_table():
    """Exactly the twins whose clause reads a MEANING-DRIFT / ROLE-REVERSING relation (sport is MEANING-DRIFT in build 2: 'X sport Y'
    shares its row with 'X sports Y' = X wears Y). 24 of 26 twins are kept."""
    lost = {}
    for c in _TWINS:
        second = c["prose"].rstrip(".").split(", and ", 1)[1] + "."
        res = A.allowlist_check(second, BANK["gathered_sets"][c["gathered"]])
        if not res["keep"]:
            lost[c["id"]] = next(r for r in T.RELATIONS if r in res["reason"].split("reads ", 1)[-1].split(",")[0].split())
    assert lost == _TWINS_LOST_TO_THE_TABLE
    assert all(T.relation_class(r) in T.NO_READING_CLASSES for r in lost.values())


def test_must_keep_set_is_nontrivial_and_from_the_held_out_strata():
    assert KEEP["parser_seed"] == 42
    assert KEEP["n_single_triple"] == len(KEEP["single_triple"]) >= 30
    assert {r["stratum"] for r in KEEP["single_triple"]} == {"S1", "S3"}
    assert all(len(r["b2h_grounded"]) == 1 for r in KEEP["single_triple"])


def test_every_single_triple_f_sentence_b2h_kept_is_kept_except_the_listed_losses(seed_chats):
    """Prereg bar 4 at the matcher level: every F-labelled S1/S3 sentence B2h kept at seed 42 that grounds exactly one triple is
    kept by the allowlist at every seed, except `allowlist_losses` (single-clause sentences outside shapes 1-3: a past-tense
    copula, 'is located in time zone', 'X sport is Y'). The losses are pinned exactly, so the list cannot grow silently."""
    losses = {(x["turn"], x["surface"]): x["reason"] for x in KEEP["allowlist_losses"]}
    kept, rejected = [], {}
    for r in KEEP["single_triple"]:
        g = KEEP["gathered_sets"][r["gathered"]]
        got = _on_all_seeds(seed_chats, r["surface"], g)
        assert len(set(got)) == 1, (r["surface"], got)                      # seed-independent
        if got[0]:
            kept.append(r)
            # the kept sentence encodes exactly the triple B2h grounded
            res = A.allowlist_check(r["surface"], g)
            assert sorted(map(tuple, res["props"])) == sorted(map(tuple, r["b2h_grounded"])), r["surface"]
        else:
            rejected[(r["turn"], r["surface"])] = A.allowlist_keep(r["surface"], g)[1]
    assert rejected == losses
    assert len(kept) == len(KEEP["single_triple"]) - len(losses)


def test_no_multi_triple_f_sentence_was_kept_by_b2h():
    """Recorded so a change in the strata shows up: B2h kept no F sentence that grounds more than one triple at seed 42."""
    assert KEEP["n_multi_triple"] == 0 and KEEP["multi_triple"] == []


@pytest.mark.parametrize("fact", KEEP["stub_one_per_relation"], ids=lambda f: f[1])
def test_the_stub_mouth_sentence_of_every_store_relation_is_kept_iff_its_relation_has_a_verb_reading(fact):
    """One real fact per relation of the production LTM store (60 relations). The template stub's one-fact sentence is kept exactly
    when the relation's table row allows a verb reading (VERB-SAFE / NOT-A-VERB); for a ROLE-REVERSING relation it is rejected with
    the class in the reason (stock keeps it: a stock leak, prereg AMENDMENT 1, not a HURTS). The roles-swapped sentence is always
    rejected."""
    s, r, o = fact
    res = A.allowlist_verify(_stub(fact), [fact])
    assert res["accepted"] == T.has_verb_reading(r), (_stub(fact), res["reject_reason"])
    if T.relation_class(r) in T.NO_READING_CLASSES:
        assert T.relation_class(r) in res["reject_reason"], res["reject_reason"]
    assert not A.allowlist_verify(_stub([o, r, s]), [fact])["accepted"], _stub([o, r, s])


# =========================================================================================================== the relation table
_STORE_ROWS = sorted(r for r, row in T.RELATIONS.items() if "store" in row.sources)
_RR = sorted(r for r, row in T.RELATIONS.items() if row.cls == T.ROLE_REVERSING)


def test_every_store_relation_has_a_row_in_the_table():
    """AMENDMENT 1: a store relation missing from the table fails here. The fixture's one-fact-per-relation list was drawn from the
    whole store (tests/fixtures/build_moat_allowlist_fixtures.py); when the store file is on this machine its full inventory and
    fact counts are checked too."""
    store_rels = sorted({f[1] for f in KEEP["stub_one_per_relation"]})
    assert len(store_rels) == 60
    missing = [r for r in store_rels if r not in T.RELATIONS]
    assert not missing, "store relations with no row in research/runners/_moat_allowlist_relations.py: %s" % missing
    assert _STORE_ROWS == store_rels
    if os.path.exists(T.STORE_FACTS):
        from collections import Counter
        n = Counter(r["fact"]["action"] for r in json.load(open(T.STORE_FACTS)))
        assert sorted(n) == store_rels, sorted(set(n) ^ set(store_rels))
        assert {r: T.RELATIONS[r].n_store for r in n} == dict(n)


def test_every_row_is_well_formed():
    for rel, row in T.RELATIONS.items():
        assert row.cls in T.CLASSES, rel
        assert row.sources and set(row.sources) <= {"store", "tiny_demo", "self_knowledge", "leak_hunt"}, rel
        assert row.why.strip() and row.readings and rel in row.readings, rel
        assert (row.n_store > 0) == ("store" in row.sources), rel
    assert T.relation_class("no_such_relation_x") == T.UNCLASSIFIED and not T.has_verb_reading("no_such_relation_x")


@pytest.mark.parametrize("rel", sorted(T.RELATIONS))
def test_a_rows_readings_are_exactly_what_the_verb_form_table_accepts(rel):
    """The `readings` column is the live table's: every token the matcher's lexicon (B2h `_classify`: inflections, synonyms, guarded
    -es / -ies) maps to the relation when it alone is gathered, over every inflection / synonym candidate."""
    lex = A._Lexicon(frozenset({("subj_x", rel, "obj_y")}))
    cand = set(M._build_inflection_map([rel])) | set(M.VERB_SYNONYMS) | {rel + "es", rel[:-1] + "ies"}
    live = sorted(t for t in cand if lex._classify(t) == ("verb", rel))
    assert tuple(live) == T.RELATIONS[rel].readings


def test_the_role_reversing_relations():
    """The reviewed ROLE-REVERSING rows (a change here is a change of the reviewed table, not of a test)."""
    assert _RR == ["author", "child", "father", "parent_taxon", "platform"]
    assert all("store" in T.RELATIONS[r].sources for r in _RR)


_MD = sorted(r for r, row in T.RELATIONS.items() if row.cls == T.MEANING_DRIFT)


def test_the_meaning_drift_relations():
    """Build 2's MEANING test: the reviewed MEANING-DRIFT rows (a change here is a change of the reviewed table)."""
    assert _MD == ["conflict", "instrument", "sport", "spouse"]
    assert T.MEANING_DRIFT in T.NO_READING_CLASSES and T.MEANING_DRIFT not in T.VERB_READING_CLASSES


@pytest.mark.parametrize("rel", _MD)
def test_no_reading_of_a_meaning_drift_relation_is_ever_kept(rel):
    """Every reading in the row, shapes 1 and 2, any spelling; reason names the class."""
    g = [["red_fox", rel, "canid_group"], ["red_fox", rel, "blue_jay"]]
    for tok in T.RELATIONS[rel].readings:
        for p in ("Red Fox %s Canid Group." % tok, "The red_fox %s the canid_group." % tok,
                  "Red Fox %s Canid Group and Blue Jay." % tok, "Red Fox %s Canid Group, and Blue Jay." % tok,
                  "Red Fox %s Canid Group." % tok.replace("_", " ")):
            keep, why = A.allowlist_keep(p, g)
            assert not keep and T.MEANING_DRIFT in why, (p, why)


@pytest.mark.parametrize("fact,prose", [
    (["sporting_benevento", "sport", "association_football_club"], "Sporting Benevento sports association football club."),
    (["sporting_benevento", "sport", "association_football_club"], "Sporting Benevento sported association football club."),
    (["james_whitmore", "conflict", "war_world_two"], "James Whitmore conflicts War World Two."),
    (["merle_haggard", "instrument", "electric_guitar"], "Merle Haggard instruments electric guitar."),
    (["victoria_of_england", "spouse", "albert_prince_consort"], "Victoria of England spoused Albert Prince Consort."),
])
def test_role_safe_but_meaning_drifting_verb_readings_are_rejected_at_every_seed(seed_chats, fact, prose):
    """'X sports Y' (wears), 'X conflicts Y' (contradicts), 'X instruments Y' (equips), 'X spoused Y' (archaic: married) keep X as
    subject but assert another relation: rejected through the production gate at every parser seed."""
    for sd in SEEDS:
        acc, res = _on(seed_chats[sd], prose, [fact])
        assert acc is False and T.MEANING_DRIFT in res["reject_reason"], (sd, res["reject_reason"])


_FATHER = ["albert_prince_of_wales", "father", "albert_prince_consort"]
_AUTHOR = ["book_of_gen", "author", "moshe_rabbenu"]


@pytest.mark.parametrize("fact,prose", [
    (_AUTHOR, "The book_of_gen authors moshe_rabbenu."),                       # the stub mouth's own sentence
    (_AUTHOR, "The book_of_gen authors the moshe_rabbenu."),
    (_AUTHOR, "Book of Gen authors Moshe Rabbenu."),
    (_AUTHOR, "Book of Gen authored Moshe Rabbenu."),
    (_AUTHOR, "The Book of Gen author Moshe Rabbenu."),
    (_AUTHOR, "Book of Gen authoring Moshe Rabbenu."),
    (_FATHER, "The albert_prince_of_wales fathers albert_prince_consort."),   # the stub mouth's own sentence
    (_FATHER, "Albert Prince of Wales fathers Albert Prince Consort."),
    (_FATHER, "Albert Prince of Wales fathered Albert Prince Consort."),
    (_FATHER, "Albert Prince of Wales father Albert Prince Consort."),
    (_FATHER, "The Albert Prince of Wales fathering the Albert Prince Consort."),
])
def test_father_and_author_sentences_are_rejected_at_every_seed(seed_chats, fact, prose):
    """'X authors Y' says X wrote Y and 'X fathers Y' says X is Y's father: the reverse of (X, author, Y) / (X, father, Y). Rejected
    through the production claim gate at every parser seed, with the class in the reason."""
    for sd in SEEDS:
        acc, res = _on(seed_chats[sd], prose, [fact])
        assert acc is False, (sd, prose)
        assert "ROLE-REVERSING" in res["reject_reason"], (sd, res["reject_reason"])


def test_the_english_true_reading_of_a_father_or_author_fact_is_not_gathered_either():
    """'Moshe Rabbenu authored Book of Gen.' is what (book_of_gen, author, moshe_rabbenu) means in English, but it encodes the
    ungathered (moshe_rabbenu, author, book_of_gen): no verb reading of a ROLE-REVERSING relation is ever kept, in either order."""
    for fact, prose in ((_AUTHOR, "Moshe Rabbenu authored Book of Gen."), (_FATHER, "Albert Prince Consort fathered Albert Prince of Wales.")):
        assert not A.allowlist_keep(prose, [fact])[0]
        assert not A.allowlist_keep(prose, [fact, [fact[2], fact[1], fact[0]]])[0]


@pytest.mark.parametrize("rel", _RR)
def test_no_reading_of_a_role_reversing_relation_is_ever_kept(rel):
    """Every reading in the row, in shapes 1 and 2, with and without determiners, slug or spaced spelling of the nouns."""
    g = [["red_fox", rel, "canid_group"], ["red_fox", rel, "blue_jay"]]
    for tok in T.RELATIONS[rel].readings:
        for p in ("Red Fox %s Canid Group." % tok, "The red_fox %s the canid_group." % tok,
                  "Red Fox %s Canid Group and Blue Jay." % tok, "Red Fox %s Canid Group, and Blue Jay." % tok,
                  "Red Fox %s Canid Group." % tok.replace("_", " ")):
            assert not A.allowlist_keep(p, g)[0], p


def test_a_role_reversing_sentence_fails_its_candidate_but_not_the_other_facts_sentence():
    g = [_FATHER, ["albert_prince_of_wales", "country", "united_kingdom"]]
    assert A.allowlist_verify("The albert_prince_of_wales countrys united_kingdom.", g)["accepted"]
    both = A.allowlist_verify("The albert_prince_of_wales countrys united_kingdom. The albert_prince_of_wales fathers "
                              "albert_prince_consort.", g)
    assert not both["accepted"] and "[sentence 2 of 2]" in both["reject_reason"]


@pytest.mark.parametrize("rel", ["founded_by", "owner_of", "capital_of", "student_of", "mother"])
def test_an_unclassified_relation_gets_no_verb_reading_and_no_shape_3(rel):
    """Relations of the larger bundles (wikidata_500k_fast / 1M_fast) that have no row: fail closed."""
    assert rel not in T.RELATIONS
    g = [["acme_co", rel, "john_q"]]
    noun = rel.replace("_", " ")
    for p in ("Acme Co %s John Q." % rel, "Acme Co %s John Q." % noun, "Acme Co %ss John Q." % rel, "Acme Co is the %s John Q." % noun,
              "Acme Co is %s John Q." % noun):
        keep, why = A.allowlist_keep(p, g)
        assert not keep, p
    assert "UNCLASSIFIED" in A.allowlist_keep("Acme Co %s John Q." % rel, g)[1]


# =========================================================================================================== shape 3
_OF_RELATIONS = sorted({f[1] for f in KEEP["stub_one_per_relation"] if f[1].endswith("_of")})
_OTHER_RELATIONS = sorted({f[1] for f in KEEP["stub_one_per_relation"] if not f[1].endswith("_of")})


def test_the_store_relations_that_take_shape_3():
    assert _OF_RELATIONS == ["instance_of", "is_a_list_of", "member_of", "part_of", "participant_of", "subclass_of"]
    assert len(_OTHER_RELATIONS) == 54
    assert all(T.has_verb_reading(r) for r in _OF_RELATIONS)


@pytest.mark.parametrize("rel", _OF_RELATIONS)
def test_shape3_reads_subject_first_and_cannot_reverse_roles(rel):
    g = [["red_fox", rel, "canid_group"]]
    noun = rel[:-3].replace("_", " ")
    for det in ("", "a ", "an "):                          # before the relation noun: a / an / nothing (build 2)
        for odet in ("", "the ", "a "):
            for sdet in ("", "The "):
                ok = "%sRed Fox is %s%s of %sCanid Group." % (sdet, det, noun, odet)
                rev = "%sCanid Group is %s%s of %sRed Fox." % (sdet, det, noun, odet)
                res = A.allowlist_check(ok, g)             # (is_a_list_of reads "Red Fox is is a list of Canid Group")
                assert res["keep"] and res["shape"] == "shape3" and res["props"] == [["red_fox", rel, "canid_group"]], ok
                assert not A.allowlist_keep(rev, g)[0], rev
    # the reversed triple is the only thing that could license the reversed sentence -- and then it is the true reading
    both = g + [["canid_group", rel, "red_fox"]]
    assert A.allowlist_keep("Canid Group is a %s of Red Fox." % noun, both)[0]


@pytest.mark.parametrize("rel", _OTHER_RELATIONS)
def test_shape3_never_applies_to_a_relation_without_of(rel):
    """'Ankara is the capital of Turkey' means (turkey, capital, ankara): a relation whose slug does not end in _of must never
    take the 'X is the R of Y' shape, in either direction, whichever way the triple is gathered."""
    noun = rel.replace("_", " ")
    for g in ([["turkey_x", rel, "ankara_y"]], [["ankara_y", rel, "turkey_x"]]):
        for p in ("Ankara Y is the %s of Turkey X." % noun, "Turkey X is the %s of Ankara Y." % noun,
                  "Ankara Y is %s of Turkey X." % noun, "Turkey X is a %s of Ankara Y." % noun,
                  # without a literal "of" that no shape consumes (the four above reject even if shape 3 took any relation:
                  # mutation report m35 / m74); "Ankara Y is capital Turkey X." is the report's witness
                  "Ankara Y is %s Turkey X." % noun, "Turkey X is %s Ankara Y." % noun, "Ankara Y is a %s Turkey X." % noun):
            assert not A.allowlist_keep(p, g)[0], (g, p)


@pytest.mark.parametrize("rel", _OF_RELATIONS)
def test_shape3_the_before_the_relation_noun_is_rejected(rel):
    """'X is THE member of Y' asserts that X is Y's only member: the triple does not say so (build 2, controller ruling)."""
    g = [["red_fox", rel, "canid_group"]]
    noun = rel[:-3].replace("_", " ")
    for odet in ("", "the "):
        for sdet in ("", "The "):
            for the in ("the", "The", "THE"):
                p = "%sRed Fox is %s %s of %sCanid Group." % (sdet, the, noun, odet)
                keep, why = A.allowlist_keep(p, g)
                assert not keep and "uniqueness" in why, (p, why)


def test_shape3_the_rule_holds_through_the_production_gate(seed_chats):
    g = [["rep_blica_del_paraguay", "member_of", "united_nations"]]
    assert _on_all_seeds(seed_chats, "Rep Blica del Paraguay is the member of the United Nations.", g) == [False] * len(SEEDS)
    assert _on_all_seeds(seed_chats, "Rep Blica del Paraguay is a member of the United Nations.", g) == [True] * len(SEEDS)


def test_shape3_needs_the_copula_is_and_only_one_object():
    g = [["spain", "member_of", "european_union"], ["spain", "member_of", "nato_x"]]
    assert A.allowlist_keep("Spain is a member of the European Union.", g)[0]
    for p in ("Spain was a member of the European Union.", "Spain are a member of the European Union.",
              "Spain is a member of the European Union and Nato X.", "Spain is a member of European Union, Nato X.",
              "Spain is also a member of the European Union.", "Spain is a member of the European Union too."):
        assert not A.allowlist_keep(p, g)[0], p


# =========================================================================================================== nothing dropped
_ES = [["spain", "shares_border_with", "france"], ["spain", "shares_border_with", "portugal"],
       ["spain", "member_of", "european_union"], ["portugal", "member_of", "european_union"]]
_BASE = ["Spain", "shares", "border", "with", "France"]


@pytest.mark.parametrize("word", ["just", "also", "only", "not", "never", "no", "perhaps", "maybe", "might", "it", "its", "they",
                                  "which", "that", "because", "although", "and", "or", "but", "1998", "42", "very", "too", ",",
                                  ":", ";", "-", "(", ")", "!", "?", "—", "安", "this", "those", "some", "every",
                                  "a", "the", "is", "was", "in", "of", "by"])
def test_any_extra_token_anywhere_rejects_the_sentence(word):
    assert A.allowlist_keep(" ".join(_BASE) + ".", _ES)[0]
    for i in range(len(_BASE) + 1):
        toks = _BASE[:i] + [word] + _BASE[i:]
        p = " ".join(toks) + "."
        keep, why = A.allowlist_keep(p, _ES)
        # a determiner directly before SUBJ / OBJ. Build 4 (no fold on relation words): a hyphen inside the relation's words
        # ("shares - border with") is no longer its spelling, so it rejects too
        legal = word in ("a", "the") and i in (0, 4)
        assert keep is legal, (p, why)


@pytest.mark.parametrize("prose", ["Spain's shares border with France.", "Spain shares border with France's.",
                                   "Spain shares border with France’s.", "Spain shares border with Fr'ance.",
                                   "Spain doesn't share border with France.", "Spain shares border with `France`."])
def test_any_apostrophe_rejects(prose):
    keep, why = A.allowlist_keep(prose, _ES)
    assert not keep and "apostrophe" in why, prose


@pytest.mark.parametrize("prose,keep", [
    ("Spain shares border with France and Portugal.", True),
    ("Spain shares border with France, and Portugal.", True),
    ("Spain shares border with the France and the Portugal.", True),
    ("Spain shares border with France, Portugal, and France.", False),       # build 5: a list names distinct things
    ("Spain shares border with France and Italy.", False),                    # every object must be gathered
    ("Spain shares border with France, Portugal.", False),                    # a list must end with "and OBJn"
    ("Spain shares border with France and Portugal and.", False),
    ("Spain and Portugal shares border with France.", False),                 # no subject coordination
    ("Spain shares border with France and is a member of the European Union.", False),
    ("Spain shares border with France and Portugal is a member of the European Union.", False),
])
def test_shape2_object_lists(prose, keep):
    assert A.allowlist_keep(prose, _ES)[0] is keep, prose


def test_one_sentence_one_shape_and_the_turn_keeps_its_other_sentences():
    good, bad = "Spain shares border with France.", "It is a member of the European Union."
    res = A.allowlist_verify(good + " " + bad, _ES)
    assert not res["accepted"] and "[sentence 2 of 2]" in res["reject_reason"]
    assert [t["keep"] for t in res["trace"]] == [True, False]
    assert A.allowlist_verify(good + " Portugal is a member of the European Union.", _ES)["accepted"]
    for p in ("Spain shares border with France. Portugal.", "Spain shares border with France.France.", "", "   ", ".",
              "Spain shares border with France!", "Does Spain share border with France?"):
        assert not A.allowlist_verify(p, _ES)["accepted"], p


def test_composer_paragraph_keeps_the_other_facts_sentences(seed_chats, monkeypatch):
    """Production composer path (sequential render): one gathered fact renders to a kept sentence, the other to a rejected one;
    the turn keeps the first and drops only the second."""
    monkeypatch.setenv(FLAG, "1")
    for k in ("BRAIN_RICH_BATCH_RENDER", "BRAIN_TOUCHPOINT_A_FACT_CLAUSE"):
        monkeypatch.delenv(k, raising=False)
    surfaces = {("spain", "shares_border_with", "france"): "Spain shares border with France.",
                ("spain", "member_of", "european_union"): "Spain is a member of the European Union, just like Portugal."}

    class R:
        def render_svo(self, a, v, p):
            return surfaces[(a, v, p)], None
    for sd in SEEDS:
        chat = seed_chats[sd]
        chat._claim_verifier_cache = {}
        monkeypatch.setattr(chat, "raw_mode", False, raising=False)
        monkeypatch.setattr(chat, "renderer", R(), raising=False)
        monkeypatch.setattr(chat, "spiking_recall_surface", lambda a, v, p: None, raising=False)
        facts = [list(k) for k in surfaces]
        para, kept, dropped = _composer(chat).render_paragraph(facts)
        assert para == "Spain shares border with France." and kept == [facts[0]] and dropped == [facts[1]], (sd, para)


# =========================================================================================================== normalisation
def test_slug_spellings_and_the_store_accent_form():
    g = [["ha_noi_vietnam", "twinned_administrative_body", "montr_al_canada"], ["united_states_virgin_islands", "capital", "charlotte_amalie"],
         ["united_states", "shares_border_with", "canada"]]
    for p in ("Ha noi vietnam twinned administrative body Montr al Canada.",
              "ha_noi_vietnam twinned_administrative_body montr_al_canada.",
              "United States Virgin Islands capital Charlotte Amalie.", "United States shares border with Canada."):
        assert A.allowlist_keep(p, g)[0], p
    for p in ("Ha noi vietnam twinned administrative body Montreal Canada.", "United States capital Charlotte Amalie.",
              "United States Virgin Islands shares border with Canada.", "Ha noi vietnam twinned administrative body 安 Canada."):
        assert not A.allowlist_keep(p, g)[0], p
    # build 4 F2: montr_al_canada's store name is "Montréal, Canada" (with a comma), so the table also holds "montréal canada"; NFC
    # makes the decomposed spelling the same text
    for p in ("Ha noi vietnam twinned administrative body Montréal Canada.",
              "Ha noi vietnam twinned administrative body Montre\u0301al Canada."):
        assert A.allowlist_keep(p, g) == (True, "ok"), p
    # build 4 (fold to find, exact to accept): the fold still FINDS these spans, but each is rejected because its text is not an
    # exact canonical surface (another accent; no hyphen/space equivalence; no fold on relation words)
    for p, span in (("Ha noi vietnam twinned administrative body Montrèal Canada.", "'Montrèal Canada'"),
                    ("Ha-Noi-Vietnam twinned administrative body montr_al_canada.", "'Ha-Noi-Vietnam'"),
                    ("Ha noi vietnam twinned-administrative-body montr_al_canada.", "'twinned-administrative-body'")):
        keep, why = A.allowlist_keep(p, g)
        assert not keep and span in why, (p, why)
    # build 3 (no invented acronyms): HA / NOI / VIETNAM are not written in capitals in any store name of ha_noi_vietnam
    keep, why = A.allowlist_keep("HA-NOI-VIETNAM twinned_administrative_body montr_al_canada.", g)
    assert not keep and "'HA'" in why and "ha_noi_vietnam" in why, why


@pytest.mark.parametrize("prose", ["Spain shares \u4e0d border with France.", "Spain shares\u4e0dborder with France.",
                                   "Spain shares \u00e0 border with France.", "Spain shares border with \u0444 France.",
                                   "Spain shares border with France \u0663.", "Spain shares border with France\u00b2.",
                                   "Spain shares border \u00e9 with France.", "\u4e0d Spain shares border with France.",
                                   # literal-slug spellings: the STOCK claim gate (flag off, seed 42) accepts each of these (the
                                   # stock tokenizer drops the non-ASCII word; reported for the board, stock unchanged)
                                   "spain shares_border_with \u4e0d france.", "spain \u4e0d shares_border_with france.",
                                   "spain member_of \u4e0d european_union.", "spain \u6ca1\u6709 shares_border_with france.",
                                   "spain shares_border_with \u043d\u0435 france.", "spain shares_border_with \u0644\u064a\u0633 france."])
def test_non_latin_or_free_standing_non_ascii_text_cannot_hide_between_slug_words(prose):
    """B2h's slug spelling joins slug words with [\\s\\-_]+ AFTER mapping non-ASCII runs to "_", so a free-standing non-Latin word
    between two slug words (\u4e0d = 'not') would vanish into the relation's spelling. Rejected before normalisation."""
    keep, why = A.allowlist_keep(prose, _ES)
    assert not keep and "non-ASCII" in why, (prose, why)


def test_accented_latin_inside_a_word_still_reads_as_the_store_slug():
    g = [["jos_mar_a_aznar", "member_of", "partido_popular"], ["rep_blica_del_paraguay", "member_of", "united_nations"]]
    # rep_blica_del_paraguay is a store entity whose store name "Rep\u00fablica del Paraguay" is in the build-4 names table
    for p in ("Rep\u00fablica del Paraguay is a member of the United Nations.", "Rep\u00dablica del Paraguay member of united nations."):
        assert A.allowlist_keep(p, g)[0], p
    # build 4: jos_mar_a_aznar is no store entity, so the accented spelling is found by the fold but is no exact surface of it
    keep, why = A.allowlist_keep("Jos\u00e9 Mar\u00eda Aznar is a member of Partido Popular.", g)
    assert not keep and "'Jos\u00e9 Mar\u00eda Aznar'" in why and "jos_mar_a_aznar" in why, why
    assert A.allowlist_keep("Jos mar a Aznar is a member of Partido Popular.", g)[0]
    # build 3 (no invented acronyms): an all-caps rendering is held to the store names of rep_blica_del_paraguay (ISO, PAR, PY only)
    keep, why = A.allowlist_keep("REP\u00daBLICA DEL PARAGUAY member of united nations.", g)
    assert not keep and "'REP\u00daBLICA'" in why and "rep_blica_del_paraguay" in why, why


# names chosen to cover each place an accented letter can sit (middle, start, end, whole first word, a combining mark after
# lowercasing, a ligature); the slug each one must spell is computed by the STORE's own function, not written here
_STORE_RULE_NAMES = ["Montr\u00e9al Canada", "Rep\u00fablica del Paraguay", "\u00c9cole Sup\u00e9rieur des Beaux-Arts",
                     "The weather in Bod\u00f8", "Koninkrijk Belgi\u00eb", "\u0130stanbul Ba\u015fak\u015fehir", "Historia Regum Britanni\u00e6",
                     "Universit\u00e4t des Saarlandes", "Canon Yaound\u00e9", "\u00c5l", "\u00cele de France", "S\u00e3o Paulo Futebol Clube",
                     "Jos\u00e9 Mar\u00eda Aznar", "Stra\u00dfe des 17 Juni"]


# of these, the build-4 names table holds exactly these as store names spelling their slug (the rest are not store entities);
# "Montréal Canada" is the F2 row of the store name "Montréal, Canada"
_STORE_RULE_NAMES_IN_TABLE = {"Montr\u00e9al Canada", "Rep\u00fablica del Paraguay", "\u00c9cole Sup\u00e9rieur des Beaux-Arts", "The weather in Bod\u00f8",
                              "Koninkrijk Belgi\u00eb", "\u0130stanbul Ba\u015fak\u015fehir", "Historia Regum Britanni\u00e6", "Canon Yaound\u00e9"}


def _with_names(monkeypatch, extra):
    """Make `names_table()` the committed table plus `extra` ({slug: [name, ...]}): a hypothetical store name, so a test can pin that
    the fold FINDS the span and only the exact check decides."""
    base = A._NamesTable(A.NAMES_TABLE_PATH)
    assert base.error is None, base.error
    for slug, names in extra.items():
        base.entities[slug] = base.entities.get(slug, frozenset()) | frozenset(A.surface_form(n) for n in names)
    monkeypatch.setattr(A, "names_table", lambda: base)


@pytest.mark.parametrize("name", _STORE_RULE_NAMES)
def test_an_accented_name_spells_the_slug_the_store_rule_makes_of_it(name, monkeypatch):
    """Spec: SUBJ / OBJ in 'the store's slug form for accented names'. The store's slugs were made by `sanitize`
    (research/runners/_knowledge_core_curate.py, which built wikidata_100k): lowercase, every run outside [a-z0-9] -> '_', strip '_'.
    An accented letter is a separator, and at the start or end of a name it vanishes. Build 4 (fold to find, exact to accept): the
    fold FINDS the slug; the span is ACCEPTED only when the name is one of that slug's own store names (the names table) or its
    de-slugged form, so an accented name of a slug outside the store is found and rejected, with the span and slug named."""
    from research.runners._knowledge_core_curate import sanitize
    slug = sanitize(name)
    in_table = A.surface_form(name) in A.names_table().surfaces(slug)
    assert in_table == (name in _STORE_RULE_NAMES_IN_TABLE), (name, slug)
    for p, g in (("%s has part Zeta Q." % name, [[slug, "has_part", "zeta_q"]]), ("Zeta Q has part %s." % name, [["zeta_q", "has_part", slug]])):
        keep, why = A.allowlist_keep(p, g)
        assert keep is in_table, (p, why)
        if not in_table:
            assert ("entity span %r" % name) in why and slug in why, why
    other = slug + "_x"                                    # not the store's slug of this name: never spelled by it
    assert not A.allowlist_keep("%s has part Zeta Q." % name, [[other, "has_part", "zeta_q"]])[0]
    _with_names(monkeypatch, {slug: [name]})                # as if it were a store name: the fold finds it, the exact check keeps it
    assert A.allowlist_keep("%s has part Zeta Q." % name, [[slug, "has_part", "zeta_q"]]) == (True, "ok"), (name, slug)
    assert A.allowlist_keep("Zeta Q has part %s." % name, [["zeta_q", "has_part", slug]]) == (True, "ok"), (name, slug)


def test_universitat_des_saarlandes_follows_the_store_rule_not_a_transliteration():
    """The store's universitat_des_saarlandes was made from the ASCII alias 'Universitat des Saarlandes' (Q700758, the alias
    `pick_clean_alias` chose); by the store's rule 'Universit\u00e4t des Saarlandes' is universit_t_des_saarlandes. Folding \u00e4 -> a
    is not the store's rule, so the accented spelling does not spell that slug."""
    from research.runners._knowledge_core_curate import sanitize
    assert sanitize("Universit\u00e4t des Saarlandes") == "universit_t_des_saarlandes"
    g = [["universitat_des_saarlandes", "member_of", "german_research_network"]]
    assert not A.allowlist_keep("Universit\u00e4t des Saarlandes is a member of German Research Network.", g)[0]
    assert A.allowlist_keep("Universitat des Saarlandes is a member of German Research Network.", g)[0]


@pytest.mark.parametrize("prose", ["\u00e0 Spain shares border with France.", "Spain shares border with France \u00e0.",
                                   "Spain shares border \u00e9 with France.", "Spain shares border with-\u00e9 France.",
                                   "Spain shares\u0301 border with France.", "Spain shares \u0301 border with France.",
                                   "Spain shares border with Fran\u00e7e.", "Sp\u00e4in shares border with France."])
def test_the_store_fold_never_hides_a_free_standing_accent_or_changes_a_spelled_name(prose):
    """Edge absorption is only the store's strip('_') of a NAME: a free-standing accented word, a combining mark between words, or
    an accent inside a gathered name's word (a different name by the store's rule) is still rejected."""
    assert not A.allowlist_keep(prose, _ES)[0], prose


@pytest.mark.parametrize("prose", ["Imperial\u2013Royal Institute has part Zeta Q.", "Imperial \u00d7 Royal Institute has part Zeta Q."])
def test_non_letter_separators_the_store_also_folded_are_not_folded(prose):
    """Out of scope (spec: only accented names): the store's rule also folds an en dash or a times sign, but the matcher keeps every
    non-letter character structural, so such a spelling is rejected (a recorded divergence on the reject side)."""
    assert not A.allowlist_keep(prose, [["imperial_royal_institute", "has_part", "zeta_q"]])[0]


def test_a_letterless_slug_never_matches_anything():
    g = [["alpha", "borders", "_"], ["alpha", "instance_of", "region"], ["alpha", "_", "beta"]]
    assert A.allowlist_keep("Alpha instance of region.", g)[0]
    for p in ("Alpha borders 安.", "Alpha 安 beta.", "Alpha borders _.", "Alpha _ beta."):
        assert not A.allowlist_keep(p, g)[0], p


def test_a_negation_word_is_never_a_noun_even_when_gathered():
    g = [["alpha", "genre", "none"], ["may", "instance_of", "month"]]
    assert not A.allowlist_keep("Alpha genre none.", g)[0]
    assert A.allowlist_keep("May instance of month.", g)[0]                  # a hedge-looking gathered noun is a noun


def test_pure_function_is_seed_free():
    assert A.allowlist_keep("Spain shares border with France.", _ES) == (True, "ok")
    assert A.allowlist_keep("Spain shares border with Italy.", _ES)[0] is False
    assert A.allowlist_keep("Spain shares border with France.", []) == (False, "allowlist: nothing gathered")


def test_the_meaning_test_is_per_reading_eat_keeps_eats_and_drops_its_synonyms():
    """Controller ruling 2026-10-20: a reading that fails the meaning test is dropped, the row's passing readings stay. (cat, eat, fish):
    "eats"/"ate" kept; the table synonyms "devours"/"consumes" (eats greedily / uses up) rejected with MEANING-DRIFT in the reason."""
    assert T.RELATIONS["eat"].cls == T.VERB_SAFE and set(T.DROPPED_READINGS) == {"eat"}
    g = [["cat", "eat", "fish"]]
    for p in ("The cat eats fish.", "The cat ate fish.", "Cat eats fish."):
        keep, why = A.allowlist_keep(p, g)
        assert keep, (p, why)
    for tok in sorted(T.DROPPED_READINGS["eat"]):
        for p in ("The cat %s fish." % tok, "Cat %s fish." % tok):
            keep, why = A.allowlist_keep(p, g)
            assert not keep and T.MEANING_DRIFT in why, (p, why)
    for tok in ("pursues", "pursued"):                                  # chase's synonym passes the meaning test and stays
        keep, why = A.allowlist_keep("The fox %s the rabbit." % tok, [["fox", "chase", "rabbit"]])
        assert keep, (tok, why)
