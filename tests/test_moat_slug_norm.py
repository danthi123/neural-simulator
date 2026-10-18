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
