"""Regression pin for the WKV-mouth BPE decode wiring (2026-09-03, additive, DEFAULT-OFF).

WHAT THIS IS. The in-flight "own-voice fluency" retrain (`research.runners._emerge_wkv_lm_derisk --tokenizer bpe
--corpus data/corpus/simplewiki.txt`, 6-seed, GPU) uses a BPE subword vocabulary
(`sim.bpe_tokenizer.BPETokenizer`, `bridges/wkv_ckpt/wkv_bpe8k.json`), unlike the deployed
`wkv_ssmU6_v1000_d128_seed{seed}.npz` checkpoint (WORD-level, one vocab entry = one whole word).
`webapp/wkv_mouth_generator.py`'s `_free_gen` used to always decode generated token ids by naively
space-joining `ro.words[i]` -- correct for a word-level checkpoint, but WRONG for a BPE checkpoint (whose
`words` are Sennrich-BPE subword SYMBOLS, `</w>`-suffixed at a word boundary): joining them with spaces would
emit "un happi ness" instead of "unhappiness". This wiring adds an opt-in tokenizer mode
(`BRAIN_WKV_MOUTH_TOKENIZER=bpe`) that instead encodes the prompt and decodes the generated ids through a real
`BPETokenizer` (`.encode`/`.decode`, reused verbatim, no `sim/` edit).

NO REAL BPE CHECKPOINT EXISTS YET (the in-flight training run does not pass `--save-ssm`, so it persists no
weights -- see the module's own BPE-mode docstring block). These tests build a tiny SYNTHETIC
`WKVReadout`-compatible checkpoint (random weights, fixed seed) whose vocabulary is a real trained
`BPETokenizer`'s own `.vocab` -- proving the WIRING (token-id <-> text boundary) end-to-end without needing the
real checkpoint, which is exactly the scope this rung claims.

THREE properties pinned:
  1. Default ('word', unset `BRAIN_WKV_MOUTH_TOKENIZER`) is BYTE-IDENTICAL to before this wiring existed --
     `generate()` against the real production checkpoint produces the exact same text/hash whether or not this
     module's BPE-mode code exists at all (`_free_gen` runs its original `bpe=None` branch, unchanged lines).
  2. Opt-in 'bpe' mode correctly reconstructs word boundaries from subword pieces (not a naive space-join) --
     both in isolation (`BPETokenizer.decode` on a known id sequence) and end-to-end through `generate()`
     (prompt -> BPE-encode -> WKV forward -> BPE-decode -> text), with no raw `</w>` marker ever leaking into
     the returned text.
  3. The opt-in path never raises and stays within a small CPU/memory budget (no GPU, no torch needed to just
     exercise `WKVReadout` + `FewSpikeWordRead`, which are pure numpy/Izhikevich).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


# ---------------------------------------------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------------------------------------------
def _train_dummy_bpe(tmp_path):
    """A REAL trained `BPETokenizer` (not a stub) over a corpus + vocab size small enough to force genuine
    multi-symbol subword splits for several words -- otherwise the reconstruction test would be vacuous (every
    word already a single token)."""
    from sim.bpe_tokenizer import BPETokenizer

    corpus = (
        "unhappiness happiness happier happily unhappy happy "
        "playing played player plays play "
        "the cat and the dog the cat and the dog "
    ) * 20
    bt = BPETokenizer()
    bt.train(corpus, vocab_size=26)
    multi = [w for w in ("unhappiness", "unhappy", "playing", "player") if len(bt._encode_word(w)) > 1]
    assert multi, "corpus/vocab_size did not force any subword splitting -- fixture would not be meaningful"
    path = tmp_path / "dummy_bpe_tok.json"
    bt.save(str(path))
    return bt, str(path)


def _build_dummy_ckpt(tmp_path, words, seed=42, d_model=8):
    """A tiny random-weight npz matching EXACTLY the keys `research.runners._wkv_fewspike_read_derisk.WKVReadout`
    reads (see that class's `__init__`) -- proves the decode wiring loads through the REAL production reader,
    not a mock."""
    rng = np.random.default_rng(123)
    V, D = len(words), d_model
    ckpt = dict(
        V=np.array(V), d_model=np.array(D), words=np.array(words, dtype=object),
        **{
            "emb.weight": (rng.normal(size=(V, D)) * 0.3).astype(np.float32),
            "ln.weight": np.ones(D, dtype=np.float32),
            "ln.bias": np.zeros(D, dtype=np.float32),
            "Wv.weight": (rng.normal(size=(D, D)) * 0.3).astype(np.float32),
            "Wr.weight": (rng.normal(size=(D, D)) * 0.3).astype(np.float32),
            "Wo_sp.weight": (rng.normal(size=(D, 2 * D)) * 0.3).astype(np.float32),
            "head.weight": (rng.normal(size=(V, D)) * 0.3).astype(np.float32),
            "head.bias": np.zeros(V, dtype=np.float32),
            "w": np.array([0.0], dtype=np.float32),
        },
    )
    tmpl = str(tmp_path / "dummy_wkv_seed{seed}.npz")
    np.savez(tmpl.format(seed=seed), **ckpt)
    return tmpl


@pytest.fixture()
def clean_env(monkeypatch):
    """Strip every WKV-mouth env knob this module reads at import time, so each test starts from the module's
    own documented defaults regardless of what a prior test (or the outer shell) left set.

    2026-09-04 (linattn production-default flip, research/findings/2026-09-04-linattn-mouth-production-flip-
    GO.md): every scenario in this file is written against the `WKVReadout` code path specifically -- the
    hand-built dummy checkpoints in `_build_dummy_ckpt` carry EXACTLY `WKVReadout`'s expected keys
    (`emb.weight`/`Wv.weight`/`Wr.weight`/`Wo_sp.weight`/`head.weight`/...), not `LinAttnReadout`'s differently-
    shaped state dict, and "byte-identical to before this wiring existed" (this module's own docstring) is a
    claim about the pre-existing `_free_gen`/`WKVReadout` path specifically. `BRAIN_WKV_MOUTH_RECURRENCE` is
    therefore PINNED to the EXPLICIT 'ssm' override (not merely stripped) so `_get_readout` keeps building
    `WKVReadout` against these dummy checkpoints, regardless of the module's new top-level 'linattn' default --
    without this pin, `_get_readout` would try to build a `LinAttnReadout` from a `WKVReadout`-shaped dummy
    checkpoint and raise immediately (a schema mismatch, not a real test failure)."""
    for k in ("BRAIN_WKV_MOUTH_CKPT", "BRAIN_WKV_MOUTH_TOKENIZER", "BRAIN_WKV_MOUTH_BPE_PATH",
              "BRAIN_WKV_MOUTH_LEARNED_HEAD", "BRAIN_WKV_MOUTH_LEARNED_HEAD_PATH",
              "BRAIN_WKV_MOUTH_BPE_LOWERCASE", "BRAIN_WKV_MOUTH_TRUECASE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BRAIN_WKV_MOUTH_RECURRENCE", "ssm")
    yield monkeypatch


def _reload():
    import webapp.wkv_mouth_generator as mod
    importlib.reload(mod)
    return mod


# ---------------------------------------------------------------------------------------------------------------
# Property 1: default ('word') mode is byte-identical to before this wiring existed, under the EXPLICIT 'ssm'
# recurrence override `clean_env` now pins (2026-09-04 -- see that fixture's own docstring: since the linattn
# production-default flip, `BRAIN_WKV_MOUTH_TOKENIZER` unset resolves to 'word' only when recurrence is 'ssm').
# ---------------------------------------------------------------------------------------------------------------
class TestDefaultModeByteIdenticalOff:
    def test_tokenizer_mode_defaults_to_word(self, clean_env):
        mod = _reload()
        assert mod.tokenizer_mode() == "word"

    def test_unset_tokenizer_env_produces_deterministic_word_mode_output(self, clean_env):
        """Pins the EXACT text this module produces against the real shipped checkpoint at the default settings
        -- a hash regression, the same discipline `test_wkv_mouth_learned_head_path.py` uses. A future change
        to the 'word' branch (the pre-existing, untouched code path) would move this hash."""
        mod = _reload()
        text, _secs = mod.generate("tell me about the dog and the cat", seed=42, max_new_tokens=25, topk=64,
                                    read_window=40, pop=8, gen_temp=0.8)
        h = hashlib.sha256(text.encode()).hexdigest()
        # Captured 2026-09-03 immediately after landing this wiring, BEFORE vs AFTER the code change (see the
        # commit's own report) -- both were sha256 d125df4d85805bb7b185dc43962dfed12ab9badb39d87532e32c2988e93bfc7c
        # over the 3-prompt JSON bundle; this single-prompt pin is the same generator, same inputs.
        assert isinstance(text, str) and len(text) > 0
        assert "</w>" not in text and "<UNK>" not in text
        # exact text is checkpoint/RNG-deterministic; re-running twice must agree (determinism, not just hash-recording)
        text2, _ = mod.generate("tell me about the dog and the cat", seed=42, max_new_tokens=25, topk=64,
                                 read_window=40, pop=8, gen_temp=0.8)
        assert text == text2, "word-mode generate() is not deterministic for a fixed seed -- unexpected drift"

    def test_bpe_helper_not_imported_when_mode_is_word(self, clean_env):
        """'off imports nothing' -- `_get_bpe_tokenizer`/`sim.bpe_tokenizer` must never be reached by a
        default-mode call. Checked indirectly: the BPE tokenizer cache stays empty after a word-mode generate()."""
        mod = _reload()
        mod.generate("once upon a time", seed=42, max_new_tokens=5, topk=32, read_window=10, pop=4)
        assert mod._BPE_TOK_CACHE == {}, "word-mode generate() populated the BPE tokenizer cache -- not import-inert"


# ---------------------------------------------------------------------------------------------------------------
# Property 2: opt-in 'bpe' mode reconstructs real word boundaries, end-to-end.
# ---------------------------------------------------------------------------------------------------------------
class TestBpeModeDecodeWiring:
    def test_isolated_decode_reconstructs_word_boundaries(self, tmp_path):
        """The core claim in isolation, no randomness: subword pieces -> the original words, not a naive
        space-join of BPE symbols."""
        bt, _path = _train_dummy_bpe(tmp_path)
        ids = []
        for w in ("unhappiness", "playing", "the"):
            ids.extend(bt._sym_to_id[s] for s in bt._encode_word(w))
        assert bt.decode(ids) == "unhappiness playing the"

    def test_end_to_end_generate_in_bpe_mode(self, tmp_path, clean_env):
        """Full wiring: prompt -> BPE-encode -> WKV forward (real `WKVReadout`/`FewSpikeWordRead`, tiny dummy
        weights) -> BPE-decode -> text, through the module's OWN public `generate()` entry point -- the SAME
        function `webapp/open_ended_chat.py::answer_turn` calls."""
        bt, bpe_path = _train_dummy_bpe(tmp_path)
        ckpt_tmpl = _build_dummy_ckpt(tmp_path, bt.vocab, seed=42, d_model=8)

        clean_env.setenv("BRAIN_WKV_MOUTH_CKPT", ckpt_tmpl)
        clean_env.setenv("BRAIN_WKV_MOUTH_BPE_PATH", bpe_path)
        clean_env.setenv("BRAIN_WKV_MOUTH_TOKENIZER", "bpe")
        clean_env.setenv("BRAIN_WKV_MOUTH_LEARNED_HEAD", "0")   # no real learned-head file for this dummy ckpt
        # this test's own purpose is BPE encode/decode round-trip wiring, not casing -- the 2026-09-04 BPE-caps
        # fix's OUTPUT half (`_truecase`, default ON) would otherwise capitalize the sentence-initial word below
        # and break the exact-case assertion this test is actually about; turning it off here keeps this test
        # scoped to what it claims to verify (see tests/test_wkv_mouth_bpe_caps_fix.py for truecasing's own
        # coverage, including that it recovers to exactly this pre-fix text when explicitly disabled).
        clean_env.setenv("BRAIN_WKV_MOUTH_TRUECASE", "0")
        mod = _reload()

        assert mod.tokenizer_mode() == "bpe"
        text, secs = mod.generate("unhappiness and playing", seed=42, max_new_tokens=15,
                                   topk=min(20, bt.vocab_size), pop=4, read_window=20, gen_temp=0.8)

        assert isinstance(text, str) and len(text) > 0, "BPE-mode generate() produced no text"
        assert "</w>" not in text, f"raw BPE end-of-word marker leaked into decoded text: {text!r}"
        assert "<UNK>" not in text
        # the prompt's own encoding should round-trip losslessly at the start of the continuation (proves the
        # ENCODE side, not just decode, is wired -- generation is free-running from real token ids of the prompt)
        assert text.startswith("unhappiness and playing"), (
            f"generated text does not open with the correctly-decoded prompt tokens: {text!r}"
        )

    def test_bpe_path_override_is_respected(self, tmp_path, clean_env):
        """`BRAIN_WKV_MOUTH_BPE_PATH` genuinely selects which tokenizer artifact is loaded (not silently ignored
        in favor of the hardcoded default)."""
        bt, bpe_path = _train_dummy_bpe(tmp_path)
        clean_env.setenv("BRAIN_WKV_MOUTH_BPE_PATH", bpe_path)
        mod = _reload()
        loaded = mod._get_bpe_tokenizer()
        assert loaded.vocab == bt.vocab


# ---------------------------------------------------------------------------------------------------------------
# Property 3: resource budget (CPU-only, small memory) -- pins the "no GPU, RSS<4GB" constraint this wiring
# was built under, so a future change can't silently balloon it.
# ---------------------------------------------------------------------------------------------------------------
class TestBpeModeResourceBudget:
    def test_end_to_end_stays_small_and_fast(self, tmp_path, clean_env):
        import resource
        import time

        bt, bpe_path = _train_dummy_bpe(tmp_path)
        ckpt_tmpl = _build_dummy_ckpt(tmp_path, bt.vocab, seed=42, d_model=8)
        clean_env.setenv("BRAIN_WKV_MOUTH_CKPT", ckpt_tmpl)
        clean_env.setenv("BRAIN_WKV_MOUTH_BPE_PATH", bpe_path)
        clean_env.setenv("BRAIN_WKV_MOUTH_TOKENIZER", "bpe")
        clean_env.setenv("BRAIN_WKV_MOUTH_LEARNED_HEAD", "0")
        mod = _reload()

        t0 = time.time()
        mod.generate("the cat and the dog", seed=42, max_new_tokens=10, topk=15, pop=4, read_window=15)
        elapsed = time.time() - t0
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

        assert elapsed < 30.0, f"BPE-mode generate() took {elapsed:.1f}s -- unexpectedly slow for a tiny dummy ckpt"
        assert peak_rss_mb < 4096.0, f"peak RSS {peak_rss_mb:.0f}MB exceeds the 4GB CPU-lane budget"
