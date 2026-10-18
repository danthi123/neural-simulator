"""Pins for `BRAIN_MOAT_SLUG_NORM` ("B2h"), the default-OFF slug-normalising mode of the claim-entailment verifier
("the moat"), research/runners/_moat_claim_entailment_derisk.py (+ its two call sites in brain_chat_tui.py and
rich_answer_composer.py).

PART 1 -- BYTE-IDENTITY with the flag unset. The verifier's output (accepted, reject_reason and a hash of the FULL result
dict, trace included) equals the UNMODIFIED verifier's on 30 toy-world suite prose + 178 real-Qwen sentences (the
2026-10-18 memo corpus, each with its turn's gathered set) + 20 template-stub literal-slug sentences + 120 smuggling
mutations of those. tests/fixtures/moat_claim_entailment_golden.json was captured from the unmodified verifier (see its
`generated_from`); regenerate it only from an unmodified verifier. Also: the flag-off call sites route exactly as before.

PART 2 -- flag ON: slug names match, digits are kept, the article/relation-name guard stops the "X is a country in Y" leak,
a hedge word never licenses an ungrounded proposition, a faithful paraphrase is kept, a one-fact turn is verified by the
claim gate, and the module's own 6-seed adversarial suite still has 0 leaks. Each guard has a control that shows the
sentence WOULD pass without it (a test that cannot see the leak proves nothing).

Run: CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m pytest tests/test_moat_slug_norm.py -q
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest

from research.runners import _moat_claim_entailment_derisk as M
from research.runners.brain_conversational_agent import BridgeParser

pytestmark = pytest.mark.filterwarnings("ignore")

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "moat_claim_entailment_golden.json").read_text())


@pytest.fixture(scope="module")
def parser():
    return BridgeParser(seed=GOLDEN["seed"])


def _digest(res):
    return hashlib.sha256(json.dumps(res, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _build_like_production(parser, gated, **kw):
    """Mirror of ChatBrain._build_claim_verifier's construction (None on a role-permutation collision)."""
    g = [list(f) for f in {tuple(f) for f in gated}]
    nouns = {t for f in g for t in (f[0], f[2])}
    verbs = {f[1] for f in g}
    try:
        return M.ClaimEntailmentVerifier(parser, g, nouns, verbs, M.VERB_SYNONYMS,
                                         M._build_inflection_map(sorted(verbs)), **kw)
    except AssertionError:
        return None


def test_golden_is_nontrivial():
    """Anti-vacuity: the golden holds both accepts and rejects, and the unmodified-verifier provenance."""
    assert GOLDEN["generated_from"]
    assert sum(1 for t in GOLDEN["toy"] if t["a"]) >= 10 and sum(1 for t in GOLDEN["toy"] if not t["a"]) >= 10
    w = GOLDEN["world"]
    assert len(w) >= 300
    assert sum(1 for e in w if e["a"]) >= 50 and sum(1 for e in w if e["a"] is False) >= 100


@pytest.mark.parametrize("kw", [{}, {"slug_norm": False}], ids=["default", "explicit-off"])
def test_toy_suite_matches_unmodified_verifier(parser, kw):
    ver = M.ClaimEntailmentVerifier(parser, M.GATED, M.NOUNS, M.VERBS, M.VERB_SYNONYMS, M.INFLECT, **kw)
    bad = []
    for t in GOLDEN["toy"]:
        r = ver.verify(t["p"])
        if (r["accepted"], r["reject_reason"], _digest(r)) != (t["a"], t["r"], t["h"]):
            bad.append((t["p"], r["accepted"], r["reject_reason"]))
    assert not bad, bad[:3]


@pytest.mark.parametrize("kw", [{}, {"slug_norm": False}], ids=["default", "explicit-off"])
def test_real_world_sentences_match_unmodified_verifier(parser, kw):
    bad = []
    for e in GOLDEN["world"]:
        ver = _build_like_production(parser, GOLDEN["gated_sets"][e["g"]], **kw)
        if ver is None:
            assert e["a"] is None, e
            continue
        r = ver.verify(e["s"])
        if (r["accepted"], r["reject_reason"], _digest(r)) != (e["a"], e["r"], e["h"]):
            bad.append((e["src"], e["s"], r["accepted"], r["reject_reason"]))
    assert not bad, bad[:3]


# ------------------------------------------------------------------------------------------------ flag-off call sites
_SVO = ["angora_turkey", "instance_of", "city_work"]
_ANGORA = [["angora_turkey", "located_in_time_zone", "kaliningrad_time"], ["angora_turkey", "instance_of", "city_work"],
           ["angora_turkey", "country", "the_republic_of_turkey"], ["angora_turkey", "twinned_administrative_body", "washington_dc"]]
_LEQUIPE = [["l_quipe_de_france", "sport", "association_football_club"],
            ["association_football_club", "described_by_source", "armenian_soviet_encyclopedia"],
            ["armenian_soviet_encyclopedia", "country", "hayastani_hanrapetut_yun"],
            ["l_quipe_de_france", "country", "franzoesische_republik"],
            ["l_quipe_de_france", "participant_of", "world_championship_football_2010"]]
_UKRAINE = [["ukraine", "instance_of", "sovereign_states"]]
_FLAG = "BRAIN_MOAT_SLUG_NORM"


def _chat(parser):
    """A real ChatBrain method set over the real BridgeParser, without the heavy ChatBrain.__init__."""
    from research.runners.brain_chat_tui import ChatBrain
    chat = ChatBrain.__new__(ChatBrain)
    chat.inner = parser
    return chat


def test_flag_off_builds_the_stock_verifier_with_the_old_cache_key(parser, monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    chat = _chat(parser)
    ver = chat._build_claim_verifier(_ANGORA)
    assert ver.slug_norm is False
    assert list(chat._claim_verifier_cache) == [frozenset(tuple(f) for f in _ANGORA)]    # bare frozenset key, as before


def test_flag_off_rejects_what_it_always_rejected(parser, monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    chat = _chat(parser)
    acc, res = chat._verify_claim_set("Angora turkey is an instance of city work.", _ANGORA)
    assert acc is False and "unrepresentable" in res["reject_reason"]


class _RecordingChat:
    """What RichAnswerComposer._verify_rendered touches: records which verifier it is sent to."""

    def __init__(self, claim=(True, {})):
        self.calls = []
        self._claim = claim

    def _verify(self, surface, asserted, svo):
        self.calls.append("single")
        return True

    def _verify_claim_set(self, surface, gated):
        self.calls.append("claim")
        return self._claim


def _composer(chat):
    from research.runners.rich_answer_composer import RichAnswerComposer
    comp = RichAnswerComposer.__new__(RichAnswerComposer)
    comp.chat = chat
    return comp


@pytest.mark.parametrize("val", [None, "0", "", "off", "false", "no", "garbage"])
def test_flag_off_one_fact_turn_uses_the_single_triple_verify(monkeypatch, val):
    if val is None:
        monkeypatch.delenv(_FLAG, raising=False)
    else:
        monkeypatch.setenv(_FLAG, val)
    chat = _RecordingChat()
    assert _composer(chat)._verify_rendered("s", None, _SVO, [_SVO]) is True
    assert chat.calls == ["single"]


def test_multi_fact_turn_uses_the_claim_gate_regardless_of_the_flag(monkeypatch):
    for val in (None, "1"):
        if val is None:
            monkeypatch.delenv(_FLAG, raising=False)
        else:
            monkeypatch.setenv(_FLAG, val)
        chat = _RecordingChat()
        _composer(chat)._verify_rendered("s", None, _SVO, _ANGORA)
        assert chat.calls == ["claim"]


@pytest.mark.parametrize("val", ["1", "true", "TRUE", " on ", "yes"])
def test_flag_on_routes_a_one_fact_turn_through_the_claim_gate(monkeypatch, val):
    monkeypatch.setenv(_FLAG, val)
    chat = _RecordingChat(claim=(False, {"reject_reason": "x"}))
    assert _composer(chat)._verify_rendered("s", None, _SVO, [_SVO]) is False      # the claim gate's verdict is authoritative
    assert chat.calls == ["claim"]


def test_flag_on_but_claim_moat_disabled_falls_back_to_the_single_triple_verify(monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    chat = _RecordingChat(claim=(None, None))          # BRAIN_CLAIM_MOAT=0 / unbuildable verifier
    assert _composer(chat)._verify_rendered("s", None, _SVO, [_SVO]) is True
    assert chat.calls == ["claim", "single"]


def test_the_three_flag_readers_agree(monkeypatch):
    from research.runners.brain_chat_tui import ChatBrain
    from research.runners.rich_answer_composer import _moat_slug_norm_enabled
    for val in (None, "", "0", "1", "true", "True", " yes ", "on", "off", "false", "no", "2", "enable"):
        if val is None:
            monkeypatch.delenv(_FLAG, raising=False)
        else:
            monkeypatch.setenv(_FLAG, val)
        got = {M.slug_norm_enabled(), ChatBrain._moat_slug_norm_enabled(), _moat_slug_norm_enabled()}
        assert len(got) == 1, (val, got)
    monkeypatch.delenv(_FLAG, raising=False)
    assert M.slug_norm_enabled() is False                # default OFF


# ------------------------------------------------------------------------------------------------ flag ON
def _verify_on(parser, gated, prose, **kw):
    ver = _build_like_production(parser, gated, slug_norm=True, **kw)
    return ver.verify(prose)


def _verify_off(parser, gated, prose):
    return _build_like_production(parser, gated).verify(prose)


@pytest.mark.parametrize("prose", [
    "Angora turkey is an instance of city work.",             # the slug name, spelled with spaces, any case
    "Angora Turkey is an instance of City Work.",
    "The angora_turkey is an instance of city_work.",         # the literal slug still passes
    "Angora-Turkey is an instance of city-work.",
    "Angora Turkey's country is The Republic of Turkey.",     # possessive
    "Angora Turkey twinned_administrative_body Washington DC.",
])
def test_slug_names_match_when_flag_on(parser, prose):
    on = _verify_on(parser, _ANGORA, prose)
    assert on["accepted"], on["reject_reason"]
    assert not _verify_off(parser, _ANGORA, prose)["accepted"] or "_" in prose     # control: the stock tokenizer cannot read it


def test_a_slug_name_does_not_make_an_ungrounded_proposition_grounded(parser):
    for prose in ("Angora turkey is an instance of washington dc.", "Washington DC is an instance of angora turkey.",
                  "Angora turkey country kaliningrad time."):
        assert not _verify_on(parser, _ANGORA, prose)["accepted"], prose


def test_digits_are_kept_an_ungrounded_year_is_rejected(parser):
    literal = "The angora_turkey instance_of city_work."
    assert _verify_on(parser, _ANGORA, literal)["accepted"]
    for tail in (" in 1998", " since 1923", " 42"):
        prose = literal.rstrip(".") + tail + "."
        assert not _verify_on(parser, _ANGORA, prose)["accepted"], prose
        assert _verify_off(parser, _ANGORA, prose)["accepted"], "control: the stock tokenizer drops digits, so it passes " + prose


def test_a_digit_slug_is_matched_and_a_wrong_year_is_not(parser):
    ok = "L quipe de France participant of World Championship Football 2010."
    assert _verify_on(parser, _LEQUIPE, ok)["accepted"]
    for bad in ("L quipe de France participant of World Championship Football 2011.",
                "L quipe de France participant of World Championship Football."):
        assert not _verify_on(parser, _LEQUIPE, bad)["accepted"], bad


_COUNTRY_LEAK = "Armenian Soviet Encyclopedia is a country in Hayastani Hanrapetut Yun."


def test_the_country_leak_is_rejected(parser, monkeypatch):
    on = _verify_on(parser, _LEQUIPE, _COUNTRY_LEAK)
    assert not on["accepted"] and "common noun after an article" in on["reject_reason"]
    # control (anti-vacuity): with the guard removed the slug-normalising verifier ACCEPTS this sentence, so it is the
    # guard, not some other stage, that stops the leak -- and the true relation statement still passes.
    monkeypatch.setattr(M.ClaimEntailmentVerifier, "_article_relation_guard", lambda self, prose: None)
    assert _verify_on(parser, _LEQUIPE, _COUNTRY_LEAK)["accepted"]
    monkeypatch.undo()
    assert _verify_on(parser, _LEQUIPE, "Armenian Soviet Encyclopedia country Hayastani Hanrapetut Yun.")["accepted"]


def test_a_hedge_does_not_license_an_ungrounded_triple(parser):
    ungrounded = "Perhaps angora turkey is an instance of washington dc."
    on = _verify_on(parser, _ANGORA, ungrounded)
    assert not on["accepted"] and "not entailed" in on["reject_reason"]
    # control: the stock verifier ADMITS a hedged in-vocabulary ungrounded triple as a flagged hypothesis -- the rich path
    # drops the flag, so the guess reaches the reply as plain text (the latent hole B2h closes).
    lit = "Perhaps angora_turkey instance_of washington_dc."
    off = _verify_off(parser, _ANGORA, lit)
    assert off["accepted"] and off["hypotheses"] == [["angora_turkey", "instance_of", "washington_dc"]]
    assert not _verify_on(parser, _ANGORA, lit)["accepted"]
    # a hedge over a GROUNDED fact is simply that fact
    assert _verify_on(parser, _ANGORA, "Perhaps angora turkey is an instance of city work.")["accepted"]


def test_a_faithful_paraphrase_of_a_gathered_fact_is_kept(parser):
    for prose in ("Angora turkey is an instance of city work.", "Angora Turkey is twinned administrative body washington dc",
                  "Angora turkey country the republic of turkey."):
        res = _verify_on(parser, _ANGORA, prose)
        assert res["accepted"] and res["grounded"], prose


@pytest.mark.parametrize("prose", [
    "Angora turkey is an instance of city work and is famous for its knots.",     # extra clause
    "Angora turkey is an instance of city work, which is old.",
    "Angora turkey is not an instance of city work.",                              # negation
    "Angora turkey is an instance of city work and the dog chases the cat.",       # foreign entity/proposition
    "City work is an instance of angora turkey.",                                  # role swap
    "Angora turkey is located in the time zone of Kaliningrad.",                   # relation wording outside the slug
])
def test_smuggling_probes_are_still_rejected(parser, prose):
    assert not _verify_on(parser, _ANGORA, prose)["accepted"], prose


def test_one_fact_turn_prose_is_checked_by_the_claim_gate(parser):
    ok = "Ukraine is an instance of sovereign states."
    assert _verify_on(parser, _UKRAINE, ok)["accepted"]
    assert not _verify_on(parser, _UKRAINE, ok.rstrip(".") + " and is famous for its vodka.")["accepted"]
    assert not _verify_on(parser, _UKRAINE, ok.rstrip(".") + " in 1991.")["accepted"]


def test_chatbrain_builds_a_slug_norm_verifier_only_when_flagged(parser, monkeypatch):
    chat = _chat(parser)
    monkeypatch.delenv(_FLAG, raising=False)
    off = chat._build_claim_verifier(_ANGORA)
    monkeypatch.setenv(_FLAG, "1")
    on = chat._build_claim_verifier(_ANGORA)
    assert off.slug_norm is False and on.slug_norm is True and off is not on        # separate cache entries
    acc, _ = chat._verify_claim_set("Angora turkey is an instance of city work.", _ANGORA)
    assert acc is True
    monkeypatch.delenv(_FLAG, raising=False)
    assert chat._verify_claim_set("Angora turkey is an instance of city work.", _ANGORA)[0] is False
    assert chat._build_claim_verifier(_ANGORA) is off                               # the OFF cache entry is reused


def test_toy_suite_has_no_leak_and_all_core_pass_with_flag_on_six_seeds():
    """The module's own adversarial suite (de-risk GO: 0 confab leaks x 6 seeds), run with the option ON. The three
    flagged-HYPOTHESIS cases now reject by design (a hedge never licenses an ungrounded proposition; the rich path never
    uses the flagged-hypothesis carve-out), so they are excluded from the core-pass count."""
    suite = M.build_suite()
    for seed in (42, 43, 44, 100, 101, 102):
        ver = M.ClaimEntailmentVerifier(BridgeParser(seed=seed), M.GATED, M.NOUNS, M.VERBS, M.VERB_SYNONYMS, M.INFLECT,
                                        slug_norm=True)
        for c in suite:
            acc = ver.verify(c["prose"])["accepted"]
            assert not (acc and c["has_false_assertion"]), (seed, c["name"])
            if c["category"] == "grounded_core":
                assert acc, (seed, c["name"])
            if c["category"] == "hypothesis":
                assert not acc, (seed, c["name"])


# ------------------------------------------------------------------------------------------------ leak hunt (verify-go panel)
# Every case of the panel's adversarial scripts (moat-verify/flagoff-leak/leak_hunt.py and leak_hunt2.py, 2026-10-18), pinned.
# expect: U = unsupported by the gathered set -> MUST be rejected with the flag ON at every parser seed; S = supported ->
# accepted at every seed, except the KNOWN_FALSE_REJECT cases (rejected on purpose, the safe side); "S?" = ambiguous reading,
# recorded in the script but not asserted. Fixes under slug_norm only: (a) non-Latin text, (b) relative-clause carry, (c) the
# article/relation guard's determiners and bare copula, (d) adjacent gathered nouns, (e) a negation-only clause and a hedge-word
# noun. The flag-off verifier is untouched (the golden above), so the same inputs still leak there: asserted below.
USVI = [["united_states", "borders", "canada"],
        ["united_states_virgin_islands", "capital", "charlotte_amalie"],
        ["united_states_virgin_islands", "borders", "british_virgin_islands"]]
USVI_SPLIT = [["united_states", "borders", "canada"], ["virgin_islands", "borders", "canada"],
              ["canada", "capital", "ottawa"]]
GEO = [["georgia", "capital", "tbilisi"], ["georgia", "borders", "alabama"], ["georgia", "borders", "armenia"]]
GEO2 = [["georgia", "capital", "tbilisi"], ["georgia_us_state", "capital", "atlanta"], ["georgia", "borders", "armenia"]]
YEAR = [["france", "participant_of", "world_championship_football_2010"], ["france", "instance_of", "national_team"],
        ["france", "founded_in", "1904"]]
TURK = [["turkey", "borders", "greece"], ["turkey", "capital", "ankara"], ["greece", "capital", "athens"],
        ["turkey", "instance_of", "country_nation"]]
GRK = [["greece", "borders", "turkey"], ["greece", "capital", "athens"], ["turkey", "capital", "ankara"]]
MAY = [["turkey", "borders", "greece"], ["june", "follows", "may"], ["may", "instance_of", "month"]]
CTY = [["ukraine", "country", "poland"], ["ukraine", "instance_of", "sovereign_states"], ["poland", "borders", "ukraine"]]
ONE = [["ukraine", "instance_of", "sovereign_states"]]
LEQ = [["armenian_soviet_encyclopedia", "country", "hayastani_hanrapetut_yun"], ["armenian_soviet_encyclopedia", "instance_of", "encyclopedia"]]
NOT = [["x_marks_the_spot", "instance_of", "film"], ["not_a_number", "instance_of", "value"], ["film", "subclass_of", "work"]]

ANG = [["angora_turkey", "borders", "greece"], ["angora_turkey", "instance_of", "city_work"], ["greece", "capital", "athens"],
       ["angora_turkey", "capital", "ankara"]]
ANG2 = [["canada", "borders", "united_states"], ["canada", "borders", "virgin_islands"], ["united_states", "capital", "washington_dc"]]
ASE = [["armenian_soviet_encyclopedia", "country", "hayastani_hanrapetut_yun"], ["armenian_soviet_encyclopedia", "instance_of", "encyclopedia"]]
DEG = [["alpha", "borders", "_"], ["alpha", "instance_of", "region"]]
HUNT1_CASES = [
    # --- slug collisions
    ("usvi: true compound slug", USVI, "United States Virgin Islands capital Charlotte Amalie.", "S"),
    ("usvi: true short slug", USVI, "United States borders Canada.", "S"),
    ("usvi: short entity given the long entity's capital", USVI, "United States capital Charlotte Amalie.", "U"),
    ("usvi: long entity given the short entity's neighbour", USVI, "United States Virgin Islands borders Canada.", "U"),
    ("usvi: short entity borders the long entity's neighbour", USVI, "United States borders British Virgin Islands.", "U"),
    ("usvi: compound with trailing partial", USVI, "United States borders Canada and Virgin Islands.", "U"),
    ("usvi-split: compound name spelled from two gathered slugs (Canada borders 'United States Virgin Islands')", USVI_SPLIT,
     "Ottawa is the capital of United States Virgin Islands.", "U"),
    ("usvi-split: adjacent-noun run as one entity", USVI_SPLIT, "Canada capital United States Virgin Islands.", "U"),
    ("usvi-split: the same, hyphenated", USVI_SPLIT, "United States borders Canada and Virgin Islands borders Canada.", "S"),
    ("georgia: country and state share one slug (upstream collision) - mixed claims", GEO,
     "Georgia borders Alabama and its capital is Tbilisi.", "S?"),
    ("georgia: wrong capital", GEO, "Georgia capital Atlanta.", "U"),
    ("georgia2: us-state slug read for the country", GEO2, "Georgia US State capital Tbilisi.", "U"),
    ("georgia2: us-state slug right", GEO2, "Georgia US State capital Atlanta.", "S"),
    ("georgia2: country slug given the state's capital", GEO2, "Georgia capital Atlanta.", "U"),
    ("georgia2: parenthesised qualifier", GEO2, "Georgia (US State) capital Atlanta.", "S?"),
    # --- digits
    ("year: right year", YEAR, "France participant of World Championship Football 2010.", "S"),
    ("year: wrong year sharing digits", YEAR, "France participant of World Championship Football 2011.", "U"),
    ("year: transposed", YEAR, "France participant of World Championship Football 2001.", "U"),
    ("year: extra year", YEAR, "France participant of World Championship Football 2010 and 2014.", "U"),
    ("year: founded right", YEAR, "France founded in 1904.", "S"),
    ("year: founded wrong", YEAR, "France founded in 1940.", "U"),
    ("year: founded wrong, 5 digits containing the right one", YEAR, "France founded in 19040.", "U"),
    ("year: right year + decade s", YEAR, "France founded in 1904s.", "U"),
    ("year: right year inside a range", YEAR, "France founded in 1904-1905.", "U"),
    ("year: right year, then a fresh digit clause", YEAR, "France founded in 1904 since 1999.", "U"),
    ("year: spelled-out year", YEAR, "France founded in nineteen hundred four.", "U"),
    ("year: trailing number", YEAR, "France instance of national team 7.", "U"),
    ("year: unicode digits", YEAR, "France founded in ١٩٠٤.", "U"),
    # --- relation swap / role swap / negation / hedge (supported set TURK)
    ("rel-swap: capital for borders", TURK, "Turkey capital Greece.", "U"),
    ("rel-swap: borders for capital", TURK, "Turkey borders Ankara.", "U"),
    ("role-swap: Greece capital Turkey", TURK, "Greece capital Turkey.", "U"),
    ("role-swap: Ankara capital Turkey", TURK, "Ankara capital Turkey.", "U"),
    ("role-swap passive: Turkey is borders by Greece", TURK, "Turkey is bordered by Greece.", "S?"),
    ("role-swap passive, reversed: Ankara is capitalled by Turkey", TURK, "Ankara is capital by Turkey.", "S?"),
    ("natural reverse: Ankara is the capital of Turkey", TURK, "Ankara is the capital of Turkey.", "S?"),
    ("natural reverse w/o article: Ankara is capital of Turkey", TURK, "Ankara is capital of Turkey.", "U"),
    ("natural reverse: Greece is the capital of Athens", TURK, "Athens is the capital of Greece.", "S?"),
    ("negation: is not", TURK, "Turkey does not border Greece.", "U"),
    ("negation: isn't", TURK, "Turkey isn't a country nation.", "U"),
    ("negation: is not instance_of", TURK, "Turkey is not an instance of country nation.", "U"),
    ("negation: never", TURK, "Turkey never borders Greece.", "U"),
    ("negation: true fact then negation-only clause", TURK, "Turkey borders Greece, although it doesn't.", "U"),
    ("negation: true fact then 'It is not.'", TURK, "Turkey borders Greece. It is not.", "U"),
    ("negation: 'no'", TURK, "No, Turkey borders Greece.", "U"),
    ("negation: n't glued", TURK, "Turkey doesn't border Greece.", "U"),
    ("negation: curly apostrophe", TURK, "Turkey doesn’t border Greece.", "U"),
    ("negation: lexical (fails to)", TURK, "Turkey fails to border Greece.", "U"),
    ("hedge: perhaps ungrounded", TURK, "Perhaps Turkey borders Ankara.", "U"),
    ("hedge: may ungrounded", TURK, "Turkey may border Ankara.", "U"),
    ("hedge: trailing perhaps", TURK, "Turkey borders Greece, perhaps Ankara.", "U"),
    ("hedge: I think", TURK, "I think Turkey capital Athens.", "U"),
    ("hedge: grounded", TURK, "Perhaps Turkey borders Greece.", "S"),
    ("hedge-word entity dropped: 'and May'", MAY, "Turkey borders Greece and May.", "U"),
    ("hedge-word entity as object", MAY, "Turkey borders May.", "U"),
    ("hedge-word entity as subject", MAY, "May borders Greece.", "U"),
    ("hedge-word entity true", MAY, "May instance of month.", "S"),
    # --- antecedent carry through relative clauses (not slug-specific; tests the structure real Qwen emits)
    ("carry: 'which has capital' binds to the WRONG noun", GRK, "Greece borders Turkey, which has capital Athens.", "U"),
    ("carry: 'whose capital is'", TURK, "Turkey borders Greece, whose capital is Ankara.", "U"),
    ("carry: true relative", TURK, "Turkey borders Greece, whose capital is Athens.", "S?"),
    ("carry: via 'and its capital'", GRK, "Greece borders Turkey and its capital is Athens.", "S"),
    # --- relation word as a common noun (guard coverage)
    ("guard: 'a country in'", LEQ, "Armenian Soviet Encyclopedia is a country in Hayastani Hanrapetut Yun.", "U"),
    ("guard: 'the country of'", LEQ, "Armenian Soviet Encyclopedia is the country of Hayastani Hanrapetut Yun.", "U"),
    ("guard bypass: 'that country in'", CTY, "Ukraine is that country in Poland.", "U"),
    ("guard bypass: 'this country'", CTY, "Ukraine is this country Poland.", "U"),
    ("guard bypass: 'its country'", CTY, "Ukraine is its country Poland.", "U"),
    ("guard bypass: 'your country'", CTY, "Ukraine is your country Poland.", "U"),
    ("guard: 'a countries'", CTY, "Ukraine is a countries Poland.", "U"),
    ("guard: 'one country'", CTY, "Ukraine is one country Poland.", "U"),
    ("guard: possessive (intended reading)", CTY, "Ukraine's country is Poland.", "S"),
    # --- one-fact turn (the new routing)
    ("one-fact: exact", ONE, "Ukraine is an instance of sovereign states.", "S"),
    ("one-fact: extra clause", ONE, "Ukraine is an instance of sovereign states and is famous for its vodka.", "U"),
    ("one-fact: extra year", ONE, "Ukraine is an instance of sovereign states in 1991.", "U"),
    ("one-fact: hedge", ONE, "Perhaps Ukraine is an instance of sovereign states.", "S"),
    ("one-fact: wrong object", ONE, "Ukraine is an instance of sovereign.", "U"),
    ("one-fact: object of a different relation", ONE, "Ukraine is sovereign states.", "S?"),
    ("one-fact: restated twice", ONE, "Ukraine is an instance of sovereign states. Ukraine is an instance of sovereign states.", "S"),
    # --- slug words that are function/negation words
    ("slug has 'not': subject slug", NOT, "Not a number is an instance of value.", "S"),
    ("slug has 'not': plain negation of a similar", NOT, "X marks the spot is not an instance of film.", "U"),
    ("slug has 'not': negating the slug content", NOT, "X is not marks the spot.", "U"),
]

HUNT2_CASES = [
    ("guard (held-out leak form, article a)", ASE, "Armenian Soviet Encyclopedia is a country in Hayastani Hanrapetut Yun.", "U"),
    ("guard bypass: that", ASE, "Armenian Soviet Encyclopedia is that country in Hayastani Hanrapetut Yun.", "U"),
    ("guard bypass: this", ASE, "Armenian Soviet Encyclopedia is this country in Hayastani Hanrapetut Yun.", "U"),
    ("guard bypass: no article", ASE, "Armenian Soviet Encyclopedia is country in Hayastani Hanrapetut Yun.", "U"),
    ("guard bypass: its", ASE, "Armenian Soviet Encyclopedia is its country, Hayastani Hanrapetut Yun.", "U"),
    ("guard bypass: these", ASE, "Armenian Soviet Encyclopedia is these country Hayastani Hanrapetut Yun.", "U"),
    ("adjacency: two gathered slugs read as one unseen entity", ANG2, "Canada borders United States Virgin Islands.", "U"),
    ("adjacency: same, with article", ANG2, "Canada borders the United States Virgin Islands.", "U"),
    ("carry multiword: wrong-noun relative clause", ANG, "Angora Turkey borders Greece, which has capital Ankara.", "U"),
    ("carry multiword: 'whose capital is' (Greece's)", ANG, "Angora Turkey borders Greece, whose capital is Ankara.", "U"),
    ("carry multiword: 'that has capital'", ANG, "Angora Turkey borders Greece that has capital Ankara.", "U"),
    ("carry multiword: 'and has capital' (subject = Angora Turkey, true)", ANG, "Angora Turkey borders Greece and has capital Ankara.", "S"),
    ("carry multiword: 'which is an instance of city work' (wrong noun)", ANG, "Angora Turkey borders Greece, which is an instance of city work.", "U"),
    ("nonlatin: CJK clause after a grounded sentence", ANG, "Angora Turkey borders Greece. \u5b89\u5361\u62c9\u662f\u5e0c\u814a\u7684\u9996\u90fd\u3002", "U"),
    ("nonlatin: CJK word as an extra object", ANG, "Angora Turkey borders Greece and \u96c5\u5178.", "U"),
    ("nonlatin: Cyrillic extra object", ANG, "Angora Turkey borders Greece \u0438 \u0410\u043d\u043a\u0430\u0440\u0430.", "U"),
    ("nonlatin: fullwidth digits (a wrong year)", ANG, "Angora Turkey borders Greece \uff12\uff10\uff11\uff11.", "U"),
    ("nonlatin: emoji", ANG, "Angora Turkey borders Greece \U0001F600 and capital Athens.", "U"),
    ("nonlatin: CJK negation 'not' after grounded", ANG, "Angora Turkey borders Greece, \u4f46\u662f\u4e0d\u662f.", "U"),
    ("nonlatin: accented near-miss of a slug (resume vs resumé)", ANG, "Angora Turkey borders Gr\u00e9ce.", "U"),
    ("degenerate slug '_' in the gathered set", DEG, "Alpha instance of region.", "S"),
]


SEEDS = (42, 43, 44, 100, 101, 102, 1)          # the parser is seeded; a deterministic replicate per seed, not independent draws
ALL_HUNT_CASES = HUNT1_CASES + HUNT2_CASES
KNOWN_FALSE_REJECT = {
    # supported statements that slug_norm rejects ON PURPOSE (the safe side): the possessive 's is not read, and "its capital"
    # is a determiner + relation word whose antecedent ("Greece or Turkey?") is exactly the ambiguity the guard exists for.
    "guard: possessive (intended reading)",
    "carry: via 'and its capital'",
    "degenerate slug '_' in the gathered set",
}


@pytest.fixture(scope="module")
def seed_parsers():
    return {sd: BridgeParser(seed=sd) for sd in SEEDS}


def _verdicts(seed_parsers, gated, prose, slug_norm):
    out = []
    for sd in SEEDS:
        ver = _build_like_production(seed_parsers[sd], gated, slug_norm=slug_norm)
        out.append(None if ver is None else bool(ver.verify(prose)["accepted"]))
    return out


@pytest.mark.parametrize("case", [c for c in ALL_HUNT_CASES if c[3] == "U"], ids=lambda c: c[0])
def test_hunt_unsupported_clause_is_rejected_at_every_seed(seed_parsers, case):
    name, gated, prose, _ = case
    assert _verdicts(seed_parsers, gated, prose, True) == [False] * len(SEEDS), (name, prose)


@pytest.mark.parametrize("case", [c for c in ALL_HUNT_CASES if c[3] == "S" and c[0] not in KNOWN_FALSE_REJECT],
                         ids=lambda c: c[0])
def test_hunt_supported_clause_is_accepted_at_every_seed(seed_parsers, case):
    name, gated, prose, _ = case
    assert _verdicts(seed_parsers, gated, prose, True) == [True] * len(SEEDS), (name, prose)


@pytest.mark.parametrize("case", [c for c in ALL_HUNT_CASES if c[0] in KNOWN_FALSE_REJECT], ids=lambda c: c[0])
def test_hunt_known_false_rejects_stay_on_the_safe_side(seed_parsers, case):
    name, gated, prose, _ = case
    assert not any(_verdicts(seed_parsers, gated, prose, True)), (name, prose)


def test_hunt_cases_the_flag_off_verifier_still_leaks_stay_unchanged_when_off(parser):
    """Flag OFF is byte-identical to the stock verifier (the golden above), so the leak classes fixed under slug_norm are
    still open there. Pinned so nobody 'fixes' them in the stock path by accident (that would break byte-identity)."""
    still_open = [(TURK, "Turkey borders Greece, although it doesn't."), (TURK, "Turkey borders Greece. It is not."),
                  (TURK, "No, Turkey borders Greece."), (TURK, "Perhaps Turkey borders Ankara."),
                  (MAY, "Turkey borders Greece and May."), (GRK, "Greece borders Turkey, which has capital Athens."),
                  (TURK, "Turkey borders Greece, whose capital is Ankara."), (CTY, "Ukraine is that country in Poland."),
                  (CTY, "Ukraine is its country Poland.")]
    for gated, prose in still_open:
        assert _build_like_production(parser, gated).verify(prose)["accepted"], prose


# ---- (a) non-Latin text, accents
_TWIN = [["ha_noi_vietnam", "twinned_administrative_body", "montr_al_canada"], ["ha_noi_vietnam", "instance_of", "city_work"]]
_PARA = [["rep_blica_del_paraguay", "member_of", "united_nations"], ["rep_blica_del_paraguay", "instance_of", "country_nation"]]


@pytest.mark.parametrize("prose", [
    "Ha noi vietnam twinned administrative body Montr\u00e9al Canada.",          # accent read as the store's "_"
    "Ha noi vietnam twinned administrative body Montre\u0301al Canada.",         # decomposed accent (NFD)
    "Ha noi vietnam twinned administrative body MONTR\u00c9AL CANADA.",
])
def test_accented_latin_names_match_the_stores_underscore_slugs(parser, prose):
    assert _verify_on(parser, _TWIN, prose)["accepted"], prose


def test_an_accented_word_that_is_not_a_gathered_slug_is_unknown(parser):
    assert not _verify_on(parser, _TWIN, "Ha noi vietnam twinned administrative body Montreal Canada.")["accepted"]
    assert not _verify_on(parser, _TWIN, "Ha noi vietnam twinned administrative body Montr\u00e9al.")["accepted"]
    # the store's own slug for "Rep\u00fablica del Paraguay" is rep_blica_del_paraguay: the accented spelling matches it ...
    assert _verify_on(parser, _PARA, "Rep\u00fablica del Paraguay is a member of the United Nations.")["accepted"]
    # ... a different accented word does not
    assert not _verify_on(parser, _PARA, "Rep\u00fablica del Paragu\u00e1y is a member of the United Nations.")["accepted"]


@pytest.mark.parametrize("tail", [
    " and \u5b89\u5361\u62c9", " \u0438 \u0410\u043d\u043a\u0430\u0440\u0430", " \uff12\uff10\uff11\u0031", " \U0001F600",
    " \u0663", " \u00b2", " \u00a9", " $", " +", " \u5b89\u5361\u62c9\u662f\u5e0c\u814a\u7684\u9996\u90fd\u3002",
])
def test_nothing_non_latin_can_vanish_silently(parser, tail):
    ok = "Angora Turkey borders Greece"
    assert _verify_on(parser, ANG, ok + ".")["accepted"]
    assert not _verify_on(parser, ANG, ok + tail + ".")["accepted"], tail
    assert not _verify_on(parser, ANG, ok + ". " + tail.strip() + ".")["accepted"], tail


def test_non_ascii_punctuation_is_still_just_punctuation(parser):
    for prose in ("Angora Turkey borders \u2014 Greece.", "Angora Turkey borders \u201cGreece\u201d.",
                  "Angora Turkey borders Greece\u2026", "\u00abAngora Turkey borders Greece\u00bb."):
        assert _verify_on(parser, ANG, prose)["accepted"], prose


# ---- (b) relative clauses: no carry of the previous agent; the stranded-subject frame survives
def test_relative_clause_takes_its_subject_from_the_nearest_noun_phrase_or_is_rejected(parser):
    for gated, prose in ((GRK, "Greece borders Turkey, which has capital Athens."),
                         (TURK, "Turkey borders Greece, whose capital is Ankara."),
                         (TURK, "Turkey borders Greece, who has capital Ankara."),
                         (GRK, "Greece borders Turkey, which capital Ankara.")):
        res = _verify_on(parser, gated, prose)
        assert not res["accepted"] and "relative clause" in res["reject_reason"], (prose, res["reject_reason"])
    # that + a second verb in one clause was already rejected (no boundary word); still is
    assert not _verify_on(parser, ANG, "Angora Turkey borders Greece that has capital Ankara.")["accepted"]
    # controls: an explicit subject, a coordinated verb phrase and the stranded-subject frame keep working
    assert _verify_on(parser, ANG, "Angora Turkey borders Greece and has capital Ankara.")["accepted"]
    assert _verify_on(parser, ANG, "Angora Turkey borders Greece, which Angora Turkey has capital Ankara.")["accepted"]
    ver = M.ClaimEntailmentVerifier(parser, M.GATED, M.NOUNS, M.VERBS, M.VERB_SYNONYMS, M.INFLECT, slug_norm=True)
    assert ver.verify("The dog, which chases the cat, eats meat.")["accepted"]


# ---- (c) guard determiners and bare copula
@pytest.mark.parametrize("lead", ["a", "an", "the", "this", "that", "these", "those", "its", "their", "your", "one", "is", "are",
                                  "was", "were", "is that", "is this"])
def test_relation_word_after_any_determiner_or_copula_is_a_common_noun(parser, lead):
    prose = "Armenian Soviet Encyclopedia is %s country in Hayastani Hanrapetut Yun." % lead
    if lead in ("is", "are", "was", "were"):
        prose = "Armenian Soviet Encyclopedia %s country in Hayastani Hanrapetut Yun." % lead
    res = _verify_on(parser, ASE, prose)
    assert not res["accepted"], prose


def test_the_relation_word_alone_is_still_the_relation(parser):
    assert _verify_on(parser, ASE, "Armenian Soviet Encyclopedia country Hayastani Hanrapetut Yun.")["accepted"]


# ---- (d) adjacent gathered nouns
def test_two_adjacent_gathered_slugs_are_not_one_unseen_entity(parser):
    for prose in ("Canada borders United States Virgin Islands.", "Canada borders the United States Virgin Islands."):
        res = _verify_on(parser, ANG2, prose)
        assert not res["accepted"] and "side by side" in res["reject_reason"], prose
    assert _verify_on(parser, ANG2, "Canada borders United States and Canada borders Virgin Islands.")["accepted"]


# ---- (e) a negation-only clause; a hedge word that is itself a gathered noun
@pytest.mark.parametrize("prose", ["Turkey borders Greece, although it doesn't.", "Turkey borders Greece. It is not.",
                                   "No, Turkey borders Greece.", "Turkey borders Greece, but not.", "Turkey borders Greece, never."])
def test_a_negation_only_clause_is_rejected(parser, prose):
    assert not _verify_on(parser, TURK, prose)["accepted"], prose
    assert _verify_on(parser, TURK, "Turkey borders Greece.")["accepted"]


def test_a_hedge_looking_gathered_noun_is_kept_as_a_noun(parser):
    assert _verify_on(parser, MAY, "May instance of month.")["accepted"]
    assert _verify_on(parser, MAY, "June follows May.")["accepted"]
    for bad in ("Turkey borders Greece and May.", "Turkey borders May.", "May borders Greece."):
        assert not _verify_on(parser, MAY, bad)["accepted"], bad
    # a hedge word that is NOT a noun is still dropped (an ungrounded hedged triple is rejected, a grounded one is kept)
    assert not _verify_on(parser, TURK, "Perhaps Turkey borders Ankara.")["accepted"]
    assert _verify_on(parser, TURK, "Perhaps Turkey borders Greece.")["accepted"]
