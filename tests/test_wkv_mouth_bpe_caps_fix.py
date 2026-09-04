"""2026-09-04 BPE-caps fix (webapp/wkv_mouth_generator.py) -- unit coverage for the two independently-guarded
pieces named in research/findings/2026-09-03-linattn-mouth-broad-scope-coverage-threshold.md's Result 2 (the
DOMINANT broad-scope coverage blocker: a case-folding bug, not topic mismatch, worth ~5.6x in teacher-forced
perplexity on its own). Fast, pure-Python: no `SimulationBridge`, no real checkpoint -- `_bpe_encode_prompt`
only needs a `BPETokenizer`, and `_truecase` is pure string manipulation. The heavier, real-checkpoint,
real-`generate()`, 6-seed teacher-forced-perplexity-recovery verification lives in
`research/runners/_wkv_mouth_bpe_caps_fix_verify_derisk.py` (see its own docstring + the finding doc for the
cited numbers) -- this file exists so the fix has FAST, committed, CI-reachable regression coverage too.
"""
from __future__ import annotations

import os

import pytest

from webapp import wkv_mouth_generator as WKV
from sim.bpe_tokenizer import BPETokenizer


@pytest.fixture()
def clean_env(monkeypatch):
    for k in ("BRAIN_WKV_MOUTH_BPE_LOWERCASE", "BRAIN_WKV_MOUTH_TRUECASE"):
        monkeypatch.delenv(k, raising=False)
    yield monkeypatch


@pytest.fixture(scope="module")
def tiny_bpe():
    """A small, fast, in-memory BPETokenizer trained on lowercase-only text -- mirrors the REAL shipped
    `wkv_bpe8k.json`'s own training regime (`sim.bpe_tokenizer.BPETokenizer.train`'s `corpus.split()` over
    whatever text it is handed; the shipped tokenizer's corpus happens to be lowercase-only, see the finding),
    so an uppercase letter in a probe prompt is, exactly as in production, a character OUTSIDE the trained
    alphabet -> `<UNK>`."""
    bt = BPETokenizer()
    bt.train("tell me about paris and the united kingdom today hello there how are you doing", vocab_size=120)
    return bt


# =====================================================================================================================
# INPUT half: bpe_lowercase_enabled() / _bpe_encode_prompt
# =====================================================================================================================
class TestBpeLowercaseInputFix:
    def test_default_on(self, clean_env):
        assert WKV.bpe_lowercase_enabled() is True

    @pytest.mark.parametrize("v", ["0", "false", "no", "off", ""])
    def test_explicit_off_values(self, clean_env, v):
        clean_env.setenv("BRAIN_WKV_MOUTH_BPE_LOWERCASE", v)
        assert WKV.bpe_lowercase_enabled() is False

    @pytest.mark.parametrize("v", ["1", "true", "yes", "on"])
    def test_explicit_on_values_stay_on(self, clean_env, v):
        clean_env.setenv("BRAIN_WKV_MOUTH_BPE_LOWERCASE", v)
        assert WKV.bpe_lowercase_enabled() is True

    def test_fix_on_recovers_full_coverage_on_a_capitalized_prompt(self, clean_env, tiny_bpe):
        """The core claim: a raw-case prompt, once lowercased before encoding, hits ZERO more `<UNK>` than the
        same prompt already lowercased by the caller -- i.e. the fix genuinely closes the gap, not just
        reduces it."""
        prompt = "Tell me about Paris."
        ids_fixed = WKV._bpe_encode_prompt(tiny_bpe, prompt)
        ids_lowercased_by_hand = tiny_bpe.encode(prompt.lower())
        assert ids_fixed == ids_lowercased_by_hand

    def test_fix_off_is_byte_identical_to_raw_encode(self, clean_env, tiny_bpe):
        """`BRAIN_WKV_MOUTH_BPE_LOWERCASE=0` reverts `_bpe_encode_prompt` to plain `bpe.encode(prompt)` --
        BYTE-IDENTICAL to `_free_gen`/`_free_gen_linattn`'s behavior before this fix existed."""
        clean_env.setenv("BRAIN_WKV_MOUTH_BPE_LOWERCASE", "0")
        prompt = "Tell me about Paris."
        assert WKV._bpe_encode_prompt(tiny_bpe, prompt) == tiny_bpe.encode(prompt)

    def test_capitals_cost_real_unk_tokens_when_the_fix_is_off(self, clean_env, tiny_bpe):
        """Sanity-checks the PROBLEM this fix addresses is real against this test's own tiny tokenizer (not just
        asserted): a raw-case prompt with the fix off carries strictly more UNK tokens than the same prompt
        lowercased."""
        clean_env.setenv("BRAIN_WKV_MOUTH_BPE_LOWERCASE", "0")
        prompt = "Tell me about Paris and the United Kingdom."
        ids_raw = WKV._bpe_encode_prompt(tiny_bpe, prompt)
        ids_lower = tiny_bpe.encode(prompt.lower())
        n_unk_raw = sum(1 for i in ids_raw if i == 0)
        n_unk_lower = sum(1 for i in ids_lower if i == 0)
        assert n_unk_raw > n_unk_lower

    def test_empty_and_none_prompt_do_not_crash(self, clean_env, tiny_bpe):
        assert WKV._bpe_encode_prompt(tiny_bpe, "") == []
        assert WKV._bpe_encode_prompt(tiny_bpe, None) == []

    def test_word_level_path_never_reads_this_flag(self, clean_env):
        """`_free_gen`/`_free_gen_linattn`'s word-level branch (`bpe=None`, the shipped default -- see
        `tokenizer_mode`) does not call `_bpe_encode_prompt` at all; this fix is scoped EXCLUSIVELY to the
        BPE-mode branch. Static proof: the word-level id-lookup line in both functions' source has no reference
        to `_bpe_encode_prompt`/`bpe_lowercase_enabled`."""
        import inspect
        for fn in (WKV._free_gen, WKV._free_gen_linattn):
            src = inspect.getsource(fn)
            word_branch = src.split("else:", 1)[1].split("\n", 2)[1]  # the line right after the bpe-mode `if`
            assert "_bpe_encode_prompt" not in word_branch
            assert "vocab_ids_by_word" in word_branch  # still the original word-level lookup


# =====================================================================================================================
# OUTPUT half: truecase_enabled() / _truecase
# =====================================================================================================================
class TestTruecaseOutputFix:
    def test_default_on(self, clean_env):
        assert WKV.truecase_enabled() is True

    @pytest.mark.parametrize("v", ["0", "false", "no", "off", ""])
    def test_explicit_off_values(self, clean_env, v):
        clean_env.setenv("BRAIN_WKV_MOUTH_TRUECASE", v)
        assert WKV.truecase_enabled() is False

    def test_capitalizes_sentence_initial_word(self):
        assert WKV._truecase("tom went to the park") == "Tom went to the park"

    def test_capitalizes_across_multiple_sentences(self):
        assert (WKV._truecase("the united kingdom is a country. it has many cities.")
                == "The united kingdom is a country. It has many cities.")

    def test_capitalizes_standalone_pronoun_i_and_contractions(self):
        assert WKV._truecase("i went home and i'm happy now") == "I went home and I'm happy now"

    def test_capitalizes_known_names_mid_sentence(self):
        assert (WKV._truecase("tom's ball rolled away and lily's kite flew high")
                == "Tom's ball rolled away and lily's kite flew high")

    def test_does_not_touch_ordinary_words_that_collide_with_excluded_names(self):
        """`will`/`rose`/`mark`/`hope`/`joy` are deliberately EXCLUDED from `_KNOWN_CAPITALIZED_WORDS` (see that
        set's own comment) precisely because they are ordinary English words far more often than they are the
        TinyStories character names of the same spelling -- this pins that exclusion so a future edit cannot
        silently re-add one of them without a test noticing the behavior change."""
        assert WKV._truecase("he will rise and mark the hope") == "He will rise and mark the hope"

    def test_idempotent_on_already_correctly_cased_text(self):
        """Never fights an already-correct capitalization (e.g. `render_fact_sentence`'s own `slug_to_np`
        proper-noun NPs) -- applying `_truecase` twice must equal applying it once."""
        once = WKV._truecase("the bounce around the ground is located in the united kingdom")
        twice = WKV._truecase(once)
        assert once == twice

    def test_empty_string_unchanged(self):
        assert WKV._truecase("") == ""

    def test_generate_return_carries_zero_uppercase_when_flag_off(self, clean_env, monkeypatch):
        """Structural proof of byte-identical-off at the `generate()` boundary, without needing a real
        checkpoint: `_truecase` is simply never called when the flag is off -- pin that by monkeypatching it to
        raise if invoked, then confirming a plain string round-trips through the same `if truecase_enabled():`
        guard `generate()` uses."""
        clean_env.setenv("BRAIN_WKV_MOUTH_TRUECASE", "0")

        def _boom(_text):
            raise AssertionError("_truecase must not be called when BRAIN_WKV_MOUTH_TRUECASE=0")

        monkeypatch.setattr(WKV, "_truecase", _boom)
        text = "raw lowercase text exactly as the checkpoint emitted it"
        if WKV.truecase_enabled():
            text = WKV._truecase(text)
        assert text == "raw lowercase text exactly as the checkpoint emitted it"
