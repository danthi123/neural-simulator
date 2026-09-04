"""Regression pin for the generator-trace mislabel found 2026-09-03 during the linattn live verification, fixed
2026-09-04 (research/findings/2026-09-04-generator-trace-mislabel-fix.md).

THE BUG. `webapp/open_ended_chat.py::answer_turn` tries the WKV mouth (`webapp.wkv_mouth_generator.generate()`)
FIRST, inside its own try-block. That function has TWO internal ways to produce a reply: (1) the genuine
WKV/linattn free-gen few-spike spiking decode, or (2) -- when `sentence_facts` names a known topic whose
relation `RELATION_LEXICON` covers -- a call to `render_fact_sentence` (the SAME already-6-seed-GO
`SpikingClauseProducer` mechanism the SEPARATE, outer "fact-clause fallback" branch below also wires in).
`answer_turn` used to infer WHICH mechanism produced `raw` purely from WHICH OF ITS OWN TRY-BLOCKS called
`generate()`, not from what `generate()` itself did -- so whenever branch (2) fired INSIDE the WKV try-block
(the common case once `BRAIN_WKV_MOUTH_SCOPE=broad` makes `in_vocab_scope` admit nearly every prompt, so the
WKV try-block is entered -- and succeeds -- before the outer fact-clause-fallback branch is ever reached), the
trace read `generator="wkv_mouth"` / `wkv_mouth_used=True` / `fact_clause_used=False` even though the reply was
genuinely written by `render_fact_sentence`. This corrupted the per-touchpoint Qwen-vs-substrate provenance the
one-brain roadmap's de-risk #2 depends on (research/findings/2026-09-03-one-brain-mouth-integration-ROADMAP.md).

THE FIX (additive/guarded, reply CONTENT unchanged -- see `TestByteIdenticalReplyContent` below). A `trace: dict
| None = None` out-parameter was added to `webapp.wkv_mouth_generator.generate()`; when passed, `_run()` records
`trace["sentence_fact_used"]` = True/False depending on which of ITS OWN two branches actually produced the
text. `answer_turn` now reads this back and sets `generator`/`fact_clause_used`/`wkv_mouth_used` from the
ACTUAL producer rather than from which try-block reached it -- and gates the separate fact-clause-fallback
block on a new `wkv_attempted` flag (not `wkv_used`) so a `sentence_facts` hit inside the WKV try-block does not
ALSO get re-rendered by the outer fallback (a second-order bug the naive fix would otherwise introduce).

WHAT THIS FILE PINS, end-to-end through the REAL `answer_turn`/`generate()` entry points (no mocked mechanism):
  1. The bug scenario itself: a known topic whose relation is lexicon-covered traces "spiking_clause" under
     BOTH `BRAIN_WKV_MOUTH_SCOPE=broad` (where the bug was found) AND the default `scope=vocab` (a rarer but
     real ~3% pre-existing case, `wkv_fact_sentence_enabled`'s own docstring) -- 6-seed on the broad-scope arm.
  2. The pre-existing-correct cell (protected against regression): scope=vocab, an out-of-vocab known topic --
     already routed through the SEPARATE outer fact-clause-fallback branch, already "spiking_clause".
  3. A GENUINE free-gen turn (known topic, but the relation is NOT lexicon-covered, or the topic is unknown)
     still traces "wkv_mouth" -- the fix must not over-correct into labelling every known-topic reply
     "spiking_clause" regardless of what actually produced it.
  4. An out-of-vocab + unknown-topic turn still degrades to "qwen" (a deterministic stub replaces the real
     off-bridge Qwen-0.5B load, mirroring `research/runners/_open_ended_qwen_fact_clause_fallback_verify.py`'s
     own isolation strategy) -- unaffected by this fix.
  5. `wkv_mouth_generator.generate()`'s own `trace` parameter, isolated: records the correct boolean, never
     changes the returned text (`trace=None` / `trace={}` / omitted are byte-identical), and is a `dict.get`-safe
     no-op contract for every pre-existing call site (`trace=None`, the default).

Run: CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m pytest tests/test_generator_trace_matches_producer.py -q
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest

from webapp import open_ended_chat as OEC
from webapp import wkv_mouth_generator as WKV
from research.runners._wkv_fact_to_sentence_lexicon_lever import RELATION_LEXICON, expected_surface

pytestmark = pytest.mark.filterwarnings("ignore")

SEEDS = (42, 43, 44, 100, 101, 102)   # the repo's 6-seed non-negotiable (CLAUDE.md)

# A relation RELATION_LEXICON covers (see research/runners/_wkv_fact_to_sentence_lexicon_lever.py) -- render_fact_
# sentence renders this deterministically and moat-safely regardless of which branch reaches it.
_COVERED_ACTION = "employer"
assert _COVERED_ACTION in RELATION_LEXICON
_UNCOVERED_ACTION = "zzz_totally_uncovered_relation_xyz"
assert _UNCOVERED_ACTION not in RELATION_LEXICON

# An OUT-OF-VOCAB agent slug for the shipped V=1000 TinyStories checkpoint (a Wikidata-style entity name) --
# `in_vocab_scope` fails on it under the default scope='vocab' (verified below), so a message about it only
# ever reaches the WKV mouth's own free-gen decode when scope='broad' forces admission.
_OOV_TOPIC = "isaac_asimov"
_OOV_KNOWN_MSG = f"tell me about {_OOV_TOPIC}"

# An IN-VOCAB message/topic (TinyStories-domain content words "dog"/"cat") that passes in_vocab_scope under
# BOTH scope='vocab' and scope='broad' -- used for the cells that must reach the WKV try-block either way.
_IN_VOCAB_TOPIC = "dog and the cat"
_IN_VOCAB_KNOWN_MSG = "tell me about the dog and the cat"
_UNKNOWN_IN_VOCAB_MSG = "the dog and the cat had a story"

# A message with no TinyStories-domain content overlap at all -- fails in_vocab_scope even under scope='broad'
# is NOT true (scope='broad' admits everything unconditionally), but under the default scope='vocab' it fails,
# which is what the qwen-fallback test below needs.
_OOV_UNKNOWN_MSG = "what do you think about quantum chromodynamics superconductivity"


@pytest.fixture()
def clean_env(monkeypatch):
    """Every env knob this fix touches, stripped -- so each test starts from the module's own documented
    defaults (scope='vocab', WKV mouth ON, fact-clause-fallback ON, sentence-fact rendering ON) regardless of
    what a prior test or the outer shell left set. `monkeypatch` auto-reverts at teardown regardless, but an
    explicit clean start matches this repo's existing test convention (see test_wkv_mouth_bpe_decode_wiring.py)."""
    for k in ("BRAIN_WKV_MOUTH_SCOPE", "BRAIN_OPEN_ENDED_WKV_MOUTH", "BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK",
              "BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE", "BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_GROUND",
              "BRAIN_WKV_MOUTH_RECURRENCE", "BRAIN_OPEN_ENDED_GEN_TIME_HONESTY"):
        monkeypatch.delenv(k, raising=False)
    yield monkeypatch


def _make_bundle(tmp_path, facts: list[dict]) -> str:
    """A minimal on-disk LTM-bundle-shaped facts.json (`webapp.open_ended_chat._read_facts_json`'s
    "developed-brain bundle" schema) -- no dependency on the real wikidata_core_15k data lake."""
    d = tmp_path / "bundle"
    d.mkdir(exist_ok=True)
    (d / "facts.json").write_text(json.dumps({"schema_version": 1, "facts": facts}), encoding="utf-8")
    return str(d)


class _FakeQwenGenerator:
    """Drop-in replacement for `OpenEndedGenerator` -- records whether/how often `.generate()` fired, without
    loading the real off-bridge Qwen-0.5B (heavy torch/CUDA model), mirroring the isolation strategy
    `research/runners/_open_ended_qwen_fact_clause_fallback_verify.py::_FakeQwenGenerator` already uses."""

    def __init__(self):
        self.calls = 0

    def generate(self, system, user, seed=42, max_new_tokens=None):
        self.calls += 1
        return "FAKE QWEN REPLY -- this generator must never be reached by the fact-clause/WKV paths", 0.01


# ---------------------------------------------------------------------------------------------------------------
# Sanity: the routing preconditions every scenario below depends on (fails loudly, first, if these ever drift).
# ---------------------------------------------------------------------------------------------------------------
def test_routing_preconditions(clean_env):
    assert WKV.in_vocab_scope(_OOV_KNOWN_MSG, seed=42) is False, (
        f"{_OOV_KNOWN_MSG!r} must fail in_vocab_scope under scope='vocab' -- if this checkpoint's vocabulary "
        f"changed and it now passes, the OOV-routing tests below need a different topic."
    )
    assert WKV.in_vocab_scope(_IN_VOCAB_KNOWN_MSG, seed=42) is True
    assert WKV.in_vocab_scope(_OOV_UNKNOWN_MSG, seed=42) is False


# ---------------------------------------------------------------------------------------------------------------
# 1-3: end-to-end through the REAL answer_turn -- the label/booleans must follow the ACTUAL producer.
# ---------------------------------------------------------------------------------------------------------------
class TestGeneratorLabelMatchesProducerViaAnswerTurn:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_broad_scope_covered_relation_traces_spiking_clause(self, clean_env, tmp_path, seed):
        """THE BUG ITSELF, 6-seed: BRAIN_WKV_MOUTH_SCOPE=broad + a known topic whose relation is lexicon-covered
        must trace generator='spiking_clause' (not 'wkv_mouth'), because render_fact_sentence -- reached from
        INSIDE _WKV.generate()'s own sentence_facts branch -- is what actually wrote `raw`."""
        clean_env.setenv("BRAIN_WKV_MOUTH_SCOPE", "broad")
        bundle = _make_bundle(tmp_path, [{"agent": _OOV_TOPIC, "action": _COVERED_ACTION,
                                          "patient": "university_of_boston", "polarity": "AFFIRM"}])
        res = OEC.answer_turn(_OOV_KNOWN_MSG, None, valence=0.0, arousal=0.5, ltm_bundle=bundle,
                              brain_bundle=None, seed=seed, max_new_tokens=25)
        assert res["known"] is True
        assert res["generator"] == "spiking_clause", res
        assert res["fact_clause_used"] is True
        assert res["wkv_mouth_used"] is False
        exp, covered, _n = expected_surface(_OOV_TOPIC, _COVERED_ACTION, "university_of_boston")
        assert covered is True
        assert res["raw"] == exp
        assert res["answer"] == res["raw"], "a moat-safe single fact clause must survive post_filter whole"

    def test_vocab_scope_in_vocab_covered_relation_traces_spiking_clause(self, clean_env, tmp_path):
        """The SAME bug, under the DEFAULT scope='vocab': when the message ALSO happens to pass the narrow
        word-overlap in_vocab_scope gate (the ~3% real-traffic case `wkv_fact_sentence_enabled`'s own docstring
        names), the WKV try-block is still entered first and must still be labelled by what it produced."""
        assert WKV.in_vocab_scope(_IN_VOCAB_KNOWN_MSG, seed=42) is True
        bundle = _make_bundle(tmp_path, [{"agent": _IN_VOCAB_TOPIC, "action": _COVERED_ACTION,
                                          "patient": "university_of_boston", "polarity": "AFFIRM"}])
        res = OEC.answer_turn(_IN_VOCAB_KNOWN_MSG, None, valence=0.0, arousal=0.5, ltm_bundle=bundle,
                              brain_bundle=None, seed=42, max_new_tokens=25)
        assert res["known"] is True
        assert res["generator"] == "spiking_clause", res
        assert res["fact_clause_used"] is True
        assert res["wkv_mouth_used"] is False

    def test_vocab_scope_out_of_vocab_covered_relation_still_traces_spiking_clause(self, clean_env, tmp_path):
        """The PRE-EXISTING-CORRECT cell, protected against regression: scope='vocab' + an out-of-vocab known
        topic never enters the WKV try-block at all (in_vocab_scope fails) -- it was already, and must remain,
        answered by the SEPARATE outer fact-clause-fallback block, `generator='spiking_clause'`."""
        assert WKV.in_vocab_scope(_OOV_KNOWN_MSG, seed=42) is False
        bundle = _make_bundle(tmp_path, [{"agent": _OOV_TOPIC, "action": _COVERED_ACTION,
                                          "patient": "university_of_boston", "polarity": "AFFIRM"}])
        res = OEC.answer_turn(_OOV_KNOWN_MSG, None, valence=0.0, arousal=0.5, ltm_bundle=bundle,
                              brain_bundle=None, seed=42, max_new_tokens=25)
        assert res["generator"] == "spiking_clause"
        assert res["fact_clause_used"] is True
        assert res["wkv_mouth_used"] is False

    @pytest.mark.parametrize("scope", ["broad", "vocab"])
    def test_uncovered_relation_known_topic_traces_wkv_mouth(self, clean_env, tmp_path, scope):
        """NOT an over-correction: a known topic whose relation is NOT lexicon-covered must still be produced
        (and traced) by the genuine WKV free-gen spiking decode -- render_fact_sentence honestly declines
        (returns None) and generate() falls through to _free_gen, under EITHER scope setting."""
        clean_env.setenv("BRAIN_WKV_MOUTH_SCOPE", scope)
        bundle = _make_bundle(tmp_path, [{"agent": _IN_VOCAB_TOPIC, "action": _UNCOVERED_ACTION,
                                          "patient": "something", "polarity": "AFFIRM"}])
        res = OEC.answer_turn(_IN_VOCAB_KNOWN_MSG, None, valence=0.0, arousal=0.5, ltm_bundle=bundle,
                              brain_bundle=None, seed=42, max_new_tokens=25)
        assert res["known"] is True
        assert res["generator"] == "wkv_mouth", res
        assert res["wkv_mouth_used"] is True
        assert res["fact_clause_used"] is False

    def test_unknown_topic_in_vocab_traces_wkv_mouth(self, clean_env):
        """No facts retrieved at all (known=False) -> sentence_facts is never even passed to generate() -> the
        genuine free-gen decode is the only possible producer, traced 'wkv_mouth'."""
        res = OEC.answer_turn(_UNKNOWN_IN_VOCAB_MSG, None, valence=0.0, arousal=0.5, ltm_bundle=None,
                              brain_bundle=None, seed=42, max_new_tokens=25)
        assert res["known"] is False
        assert res["generator"] == "wkv_mouth"
        assert res["wkv_mouth_used"] is True
        assert res["fact_clause_used"] is False

    def test_out_of_vocab_and_unknown_falls_back_to_qwen(self, clean_env):
        """Neither the WKV mouth (out-of-vocab) nor the fact-clause fallback (known=False) can fire -- the turn
        must still degrade to the pre-existing Qwen path, traced 'qwen', unaffected by this fix."""
        assert WKV.in_vocab_scope(_OOV_UNKNOWN_MSG, seed=42) is False
        fake = _FakeQwenGenerator()
        old = OEC.get_generator
        OEC.get_generator = lambda warm_faculty: fake
        try:
            res = OEC.answer_turn(_OOV_UNKNOWN_MSG, None, valence=0.0, arousal=0.5, ltm_bundle=None,
                                  brain_bundle=None, seed=42, max_new_tokens=25)
        finally:
            OEC.get_generator = old
        assert res["known"] is False
        assert res["generator"] == "qwen"
        assert res["wkv_mouth_used"] is False
        assert res["fact_clause_used"] is False
        assert fake.calls == 1
        assert res["raw"].startswith("FAKE QWEN REPLY")


# ---------------------------------------------------------------------------------------------------------------
# 4: byte-identical reply content -- this fix must only ever change trace metadata.
# ---------------------------------------------------------------------------------------------------------------
class TestByteIdenticalReplyContent:
    def test_answer_and_raw_identical_across_scope_for_the_same_covered_fact(self, clean_env, tmp_path):
        """The SAME known+covered-relation turn (broad vs vocab scope, both of which now correctly reach
        render_fact_sentence -- one from inside the WKV try-block, one from the outer fallback) must produce
        the IDENTICAL `raw`/`answer` text either way: only the internal ROUTE and the trace label differ."""
        bundle = _make_bundle(tmp_path, [{"agent": _OOV_TOPIC, "action": _COVERED_ACTION,
                                          "patient": "university_of_boston", "polarity": "AFFIRM"}])
        # arm 1: scope='vocab' (clean_env already leaves BRAIN_WKV_MOUTH_SCOPE unset -> scope_mode()'s
        # documented default) -- _OOV_KNOWN_MSG fails in_vocab_scope, so this reaches the OUTER fact-clause-
        # fallback branch (the WKV try-block is never even entered).
        assert os.environ.get("BRAIN_WKV_MOUTH_SCOPE") is None
        res_vocab = OEC.answer_turn(_OOV_KNOWN_MSG, None, valence=0.0, arousal=0.5, ltm_bundle=bundle,
                                    brain_bundle=None, seed=42, max_new_tokens=25)
        # arm 2: scope='broad' -- in_vocab_scope now admits the SAME message unconditionally, so this reaches
        # render_fact_sentence FROM INSIDE the WKV try-block's own sentence_facts branch instead -- a genuinely
        # DIFFERENT internal route to the SAME producer.
        clean_env.setenv("BRAIN_WKV_MOUTH_SCOPE", "broad")
        res_broad = OEC.answer_turn(_OOV_KNOWN_MSG, None, valence=0.0, arousal=0.5, ltm_bundle=bundle,
                                    brain_bundle=None, seed=42, max_new_tokens=25)
        assert res_broad["raw"] == res_vocab["raw"]
        assert res_broad["answer"] == res_vocab["answer"]
        # ...and the labels genuinely differ in ROUTE (WKV try-block vs the outer fallback) while agreeing on
        # the PRODUCER label -- both correctly "spiking_clause" post-fix.
        assert res_broad["generator"] == res_vocab["generator"] == "spiking_clause"


# ---------------------------------------------------------------------------------------------------------------
# 5: the mechanism the fix depends on, isolated -- webapp.wkv_mouth_generator.generate()'s own `trace` parameter.
# ---------------------------------------------------------------------------------------------------------------
class TestGenerateTraceParameterDirect:
    def test_trace_none_is_backward_compatible(self):
        """Every pre-existing call site (trace=None, the default): generate() must behave exactly as before
        this parameter existed -- a plain (text, seconds) 2-tuple, no exception."""
        text, secs = WKV.generate("once upon a time", seed=42, max_new_tokens=10, topk=32, read_window=10, pop=4)
        assert isinstance(text, str) and len(text) > 0
        assert isinstance(secs, float)

    def test_trace_records_true_when_sentence_facts_covered(self):
        triple = (_OOV_TOPIC, _COVERED_ACTION, "university_of_boston")
        trace: dict = {}
        text, _secs = WKV.generate(_OOV_KNOWN_MSG, seed=42, max_new_tokens=25, sentence_facts=[triple],
                                    trace=trace)
        assert trace == {"sentence_fact_used": True}
        exp, covered, _n = expected_surface(*triple)
        assert covered is True
        assert text == exp

    def test_trace_records_false_when_sentence_facts_uncovered_or_absent(self):
        uncovered_triple = (_IN_VOCAB_TOPIC, _UNCOVERED_ACTION, "something")
        trace_uncovered: dict = {}
        text_uncovered, _s = WKV.generate("once upon a time", seed=42, max_new_tokens=10, topk=32,
                                          read_window=10, pop=4, sentence_facts=[uncovered_triple],
                                          trace=trace_uncovered)
        assert trace_uncovered == {"sentence_fact_used": False}

        trace_absent: dict = {}
        text_absent, _s = WKV.generate("once upon a time", seed=42, max_new_tokens=10, topk=32, read_window=10,
                                       pop=4, trace=trace_absent)
        assert trace_absent == {"sentence_fact_used": False}
        # an uncovered sentence_facts and no sentence_facts at all must fall through to the IDENTICAL free-gen
        # call (sentence_facts plays no further role once render_fact_sentence declines) -- same seed/prompt.
        assert text_uncovered == text_absent

    def test_trace_parameter_never_changes_returned_text(self):
        """The core additive/guarded contract: trace=None vs trace={} vs omitted must never change WHAT is
        generated (covered-relation case, where the bug actually lived) -- only whether the caller can observe
        which mechanism produced it."""
        triple = (_OOV_TOPIC, _COVERED_ACTION, "university_of_boston")
        text_none, _s1 = WKV.generate(_OOV_KNOWN_MSG, seed=42, max_new_tokens=25, sentence_facts=[triple],
                                      trace=None)
        text_dict, _s2 = WKV.generate(_OOV_KNOWN_MSG, seed=42, max_new_tokens=25, sentence_facts=[triple],
                                      trace={})
        text_omitted, _s3 = WKV.generate(_OOV_KNOWN_MSG, seed=42, max_new_tokens=25, sentence_facts=[triple])
        assert text_none == text_dict == text_omitted

    def test_malformed_sentence_facts_never_crashes_and_reports_false(self):
        """Defensive: a caller that passes non-triple records (e.g. raw dict-shaped fact records instead of
        (agent, action, patient) tuples) must degrade honestly -- `pick_covered_fact` already guards `len(triple)
        != 3` -- not raise, and `trace` must still report the honest outcome (no covered relation found)."""
        bad_facts = [{"agent": _OOV_TOPIC, "action": _COVERED_ACTION, "patient": "university_of_boston",
                     "polarity": "AFFIRM"}]
        trace: dict = {}
        text, _s = WKV.generate("once upon a time", seed=42, max_new_tokens=10, topk=32, read_window=10, pop=4,
                                sentence_facts=bad_facts, trace=trace)
        assert trace == {"sentence_fact_used": False}
        assert isinstance(text, str) and len(text) > 0
