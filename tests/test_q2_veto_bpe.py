"""Fix A regression: the BPE-aware faithful per-token grounded veto.

The HOLE: the old word-level isolated-token mask STRUCTURALLY VETOED
grounded multi-subword content words (the model literally could not
emit "max" -> ['ma','x</w>'] or "friendly" -> ['friend','ly</w>'])
AND leaked ungrounded content via short subword-fragment collisions.

This test loads the REAL trained Generator-F BPE tokenizer and proves
the REQUIRED PROPERTY of Fix A on the actual artifact:
  (i)  a multi-subword grounded word ("max", "friendly" from the KB
       prop "max is a big friendly dog") IS emittable -- its full BPE
       id-sequence is traversable/allowed by the constructed automaton.
  (ii) a clearly-ungrounded word ("castle") is NOT completable -- its
       full BPE id-sequence cannot be traversed/completed.

torch is needed only to construct the model object; this test does NOT
run generation (CPU-only, fast). It exercises the precomputed automaton
directly via _GroundedConstrainedLM helpers.
"""
import os

import pytest

_GEN_F = "research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real"


def _need_artifact():
    if not (os.path.exists(_GEN_F + ".pt")
            and os.path.exists(_GEN_F + ".bpe.json")):
        pytest.skip("Generator-F artifact absent")
    pytest.importorskip("torch")


def _build_lm():
    _need_artifact()
    from research.runners.constrained_decode_gate import (
        _GroundedConstrainedLM)
    return _GroundedConstrainedLM(_GEN_F, mode="constrained")


def _word_emittable(lm, automaton, word):
    """A content word is emittable iff every step of its BPE id-seq is
    traversable in the automaton: each proper prefix is in PREFIXES (or
    a FULL), and the complete id-seq is in FULLS."""
    enc = tuple(lm.tok.encode(word))
    if not enc:
        return False
    pref, fulls = automaton["prefixes"], automaton["fulls"]
    for k in range(len(enc)):
        if enc[:k] not in pref and enc[:k] not in fulls:
            return False
    return enc in fulls


def test_multi_subword_grounded_word_is_emittable():
    lm = _build_lm()
    prop = "max is a big friendly dog"
    automaton = lm._allowed_automaton(prop)
    # Sanity: these ARE multi-subword on the real tokenizer (the HOLE).
    assert len(lm.tok.encode("max")) >= 2
    assert len(lm.tok.encode("friendly")) >= 2
    # REQUIRED PROPERTY (i): grounded multi-subword words emittable.
    assert _word_emittable(lm, automaton, "max") is True
    assert _word_emittable(lm, automaton, "friendly") is True
    assert _word_emittable(lm, automaton, "dog") is True
    assert _word_emittable(lm, automaton, "big") is True


def test_ungrounded_word_not_completable():
    lm = _build_lm()
    prop = "max is a big friendly dog"
    automaton = lm._allowed_automaton(prop)
    # REQUIRED PROPERTY (ii): a clearly-ungrounded content word can NOT
    # be completed by the faithful veto.
    assert _word_emittable(lm, automaton, "castle") is False
    assert _word_emittable(lm, automaton, "dragon") is False


def test_function_words_emittable_even_when_multi_subword():
    lm = _build_lm()
    prop = "max is a big friendly dog"
    automaton = lm._allowed_automaton(prop)
    # FUNCTION_WORDS join ALLOW; multi-subword function words (e.g.
    # "this" -> 2 ids on this tokenizer) must still be emittable.
    from research.runners.generator_g_core import FUNCTION_WORDS
    multi_fn = [w for w in sorted(FUNCTION_WORDS)
                if len(lm.tok.encode(w)) > 1]
    assert multi_fn, "expected >=1 multi-subword function word"
    for w in multi_fn[:5]:
        assert _word_emittable(lm, automaton, w) is True


def test_mask_steps_allow_grounded_seq_forbid_ungrounded():
    """Drive the per-token allow check across a grounded word's full id
    sequence (must stay allowed every step) and across an ungrounded
    word's sequence (must be FORBIDDEN before completion)."""
    lm = _build_lm()
    prop = "max is a big friendly dog"
    automaton = lm._allowed_automaton(prop)
    # grounded "friendly": every prefix step allowed; completes a FULL.
    enc = list(lm.tok.encode("friendly"))
    cur = []
    for t in enc:
        assert lm._token_allowed(automaton, cur, t) is True
        cur, _done = lm._advance(automaton, cur, t)
    # ungrounded "castle": at least one step must be FORBIDDEN.
    enc_u = list(lm.tok.encode("castle"))
    cur = []
    forbidden_somewhere = False
    for t in enc_u:
        if not lm._token_allowed(automaton, cur, t):
            forbidden_somewhere = True
            break
        cur, _done = lm._advance(automaton, cur, t)
    assert forbidden_somewhere is True


def test_emittable_rate_helper_high_on_grounded_props():
    """The Fix-B per-seed metric helper: fraction of KB props whose ALL
    content words are emittable under the faithful mask should be high
    (the whole point of Fix A) -- definitely above the frozen floor."""
    lm = _build_lm()
    from research.runners.constrained_decode_gate import _GROUNDED
    from research.runners.constrained_decode_core import (
        _CDC_MIN_MULTITOKEN_EMITTABLE)
    props = list(_GROUNDED.values())[:6]
    rate = lm._props_fully_emittable_rate(props)
    assert 0.0 <= rate <= 1.0
    assert rate >= _CDC_MIN_MULTITOKEN_EMITTABLE
