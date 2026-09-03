"""BRAIN_OPEN_ENDED_WKV_MOUTH — the from-scratch WKV/SSM spiking mouth as an alternate, IN-VOCAB-ONLY generator
for the `BRAIN_OPEN_ENDED` free-generation channel (default-OFF; see `webapp/open_ended_chat.py::answer_turn`).

WHAT THIS IS (per research/findings/2026-08-28-mouth-crutch-burndown-scope.md §4, the scoping pass that named this
exact rung). `webapp/open_ended_chat.py`'s `BRAIN_OPEN_ENDED` channel is the ONE touchpoint in the live
`/api/brain-chat` pipeline where the LITERAL Qwen2.5-0.5B model (`SpikingQwenFaculty`) is the SOLE generator, with
no fallback or competitor. This module wires in a genuinely DIFFERENT, from-scratch, home-grown recurrent SSM/RWKV-
style spiking cortex (`bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz` — vocab V=1000, hidden D=128, its OWN
embeddings/Wv/Wr/Wo_sp/head, trained on TinyStories, architecturally unrelated to Qwen) as an alternate generator,
reusing VERBATIM the GO-verified few-spike Izhikevich soft-WTA word-decode mechanism
(`research.runners._wkv_fewspike_read_derisk.WKVReadout` / `FewSpikeWordRead` — genuine spiking population-coded
winner-take-all read off `cp_firing_states`, NOT a host argmax/softmax-sample; see that module's own anti-cheats).

SCOPE (honest, load-bearing — do not silently widen). The checkpoint's vocabulary is V=1000 TinyStories-domain
words. This module is scoped EXCLUSIVELY to prompts whose content words substantially overlap that vocabulary
(`in_vocab_scope`); out-of-vocab / broad-topic prompts return False from that gate and the CALLER (`answer_turn`)
falls back to the existing `SpikingQwenFaculty` path, unchanged. This module NEVER attempts an out-of-scope prompt
itself, and never silently pads/guesses past what the checkpoint's own 1000-word vocabulary supports.

TWO HONEST RESIDUALS CARRIED FORWARD FROM THE SCOPING PASS (not resolved here — see the wiring finding doc):
  1. The specific e-prop LOCALLY-LEARNED read-out head (`W_hat`, `sub_recov_ratio_mean=0.8686`,
     research/findings/2026-08-28-mouth-stale-coo-training-fix-fullscale-confirmation-GO.md) was never persisted
     to disk by the runner that produced it (`_wkv_mouth_readout_eprop_batched_substrate_derisk.py` trains `W_hat`
     in-memory each run and writes only summary metrics to its JSON artifact — verified by reading that runner:
     no `np.save`/`savez` call on the learned matrix anywhere in it). This module therefore reads the checkpoint's
     OWN existing `head.weight` (the run's "copied" reference, `sub_copied_recov_mean=0.9785` — itself WKV-native,
     architecturally unrelated to Qwen, and empirically coherent, see the scope doc's own re-confirmed smoke and
     `verify_in_vocab_generation` below) rather than the specific 0.8686-ratio matrix. Wiring in the actual
     e-prop-learned weights requires the eprop runner to be extended to persist `W_hat` and a fresh training pass
     (GPU-bound per that runner's own docstring) — named here as the concrete next step, not done in this rung.
  2. Whether the checkpoint's OWN recurrent-store weights (Wv/Wr/Wo_sp/embeddings) — as opposed to the read-out
     head specifically — were produced by the diagonal e-prop local rule (2026-08-12 GO) or an earlier host-BPTT
     pass is UNVERIFIED (flagged, not re-derived, in the scope doc's §4 residual 2). A quick provenance sanity
     check run for this rung (see the wiring finding doc) confirms the checkpoint is a LEGITIMATELY TRAINED model
     (frequency-rank-ordered vocabulary, non-degenerate weight statistics, coherent free generation) and NOT a
     placeholder — but does not resolve WHICH training method produced it.

RNG DISCIPLINE (the #77 footgun, replicated from `webapp/affect_drives_chat.py::_isolated`). Building the read
mechanism's `SimulationBridge` reseeds the process-global `numpy`/`cupy`/`random` RNGs (`sim/bridge.py:1625-1627`,
`cfg.seed`) — an UNGUARDED call would perturb every RNG-dependent organ that runs later in THIS process. Every
entry point below runs on a PRIVATE, per-seed RNG timeline: the host RNG is snapshotted, the private timeline is
swapped in, the mechanism runs, the advanced private timeline is captured, and the host RNG is restored — so this
module never alters host-process RNG state as an observable side effect, on OR off the new flag.

DEFAULT-OFF. `BRAIN_OPEN_ENDED_WKV_MOUTH` unset/0 (see `wkv_mouth_enabled()` in `open_ended_chat.py`) -> this
module is never imported by `answer_turn` (lazy import, mirrors the `BRAIN_OPEN_ENDED` top-level gate's own
"off imports nothing" discipline) -> zero behavioral or import-time change to the existing byte-identical path.

NO `sim/` edit. Reuse-by-import of `WKVReadout` / `FewSpikeWordRead` / `_softmax`; only the free-generation DRIVING
LOOP is lifted out of `_wkv_fewspike_read_derisk.run_seed`'s nested `_free_gen` closure into a standalone function
(the closure is not itself importable) — the spiking-WTA MECHANISM those classes implement is unchanged.

FACT->SENTENCE WIRE-IN (`sentence_facts` on `generate()`, 2026-09-01, board #112 rung-3 "clean unlock" gate).
`research/findings/2026-09-01-wkv-fact-to-sentence-lexicon-and-np-lever.md` built a curated relation->predicate
lexicon + slug->NP surfacer driving the already-6-seed-GO `SpikingClauseProducer` (`research.runners.
_spiking_fluent_surface_derisk`) to render a real recalled fact as a coherent English clause — but left it a
PARALLEL renderer, never reachable from this module's own `generate()`. `render_fact_sentence` (below) closes
that: called from `generate()`'s `_run()` closure, it is now genuinely part of the mouth's own generation path.
Gated by `webapp.open_ended_chat.wkv_fact_sentence_enabled()` (`BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE`,
default-OFF) — see that module's docstring for the live call-site wiring and honest scope (only fires for a
known topic whose relation the lexicon covers; falls through to the pre-existing free-gen/fact-boost path
otherwise).
"""
from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

import numpy as np

from research.runners._wkv_fewspike_read_derisk import WKVReadout, FewSpikeWordRead, _softmax

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CKPT_TEMPLATE = os.environ.get(
    "BRAIN_WKV_MOUTH_CKPT", str(_REPO_ROOT / "bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
)
_WORD_RE = re.compile(r"[a-zA-Z']+")

# ── e-prop LEARNED read-out head (honest residual #1, see the module docstring). FLIPPED DEFAULT-ON 2026-09-02
# (board #191, see `learned_head_enabled` below for the GO evidence); `BRAIN_WKV_MOUTH_LEARNED_HEAD=0` reverts
# to the native checkpoint head.weight, byte-identical to before this flag existed. `_wkv_mouth_readout_eprop_
# batched_substrate_derisk.py --save-w-hat <path>` is the ONLY producer of this file's shape/basis; see
# `_apply_learned_head` for the compatibility check performed before it is ever substituted in.
_LEARNED_HEAD_ENV = "BRAIN_WKV_MOUTH_LEARNED_HEAD"
_LEARNED_HEAD_PATH_TEMPLATE = os.environ.get(
    "BRAIN_WKV_MOUTH_LEARNED_HEAD_PATH",
    # Default points at the 6/6-GO persisted heads (ratio 0.9273, min 0.8906; finding
    # 2026-08-28-mouth-better-head-persist-6seed-GO-plus-wander-production-partial). Now default-ON
    # (BRAIN_WKV_MOUTH_LEARNED_HEAD, 2026-09-02); override the path with BRAIN_WKV_MOUTH_LEARNED_HEAD_PATH.
    str(_REPO_ROOT / "research/findings/raw/_persist_eprop_head_scope/wkv_eprop_learned_head_0p94_s{seed}.npz"),
)


# 2026-09-02 FLIPPED DEFAULT-ON (board #191, 6-seed A/B GO reproduced fresh against this exact code:
# research/findings/raw/_wkv_learned_vs_native_head_ab_6seed.json / research/findings/2026-09-01-wkv-mouth-
# learned-head-6seed-ab-through-fixed-default-path-GO.md): the fixed default per-seed learned-head path
# resolves and APPLIES cleanly on every one of the 6 non-negotiable seeds (no silent fail-safe fallback), and
# generates a lower (better) mean self-NLL than the native head on all 6 seeds (45/48 individual prompt wins).
# ZERO PRODUCTION RISK today: this module is only ever imported when `BRAIN_OPEN_ENDED` is truthy (still
# default-OFF, see webapp/open_ended_chat.py) -- with that top gate off, this flag never executes.
_LEARNED_HEAD_DEFAULT_ON = True


def learned_head_enabled() -> bool:
    """DEFAULT-ON (flipped 2026-09-02, `_LEARNED_HEAD_DEFAULT_ON`). `BRAIN_WKV_MOUTH_LEARNED_HEAD` in
    {0,false,no,off,""} -> an explicit OFF, reverting to the checkpoint's NATIVE head, byte-identical to before
    this flag existed. Only ever reached when `BRAIN_OPEN_ENDED` (still default-OFF) is also truthy."""
    v = os.environ.get(_LEARNED_HEAD_ENV)
    if _LEARNED_HEAD_DEFAULT_ON:
        return not (v is not None and v.strip().lower() in ("0", "false", "no", "off", ""))
    return v is not None and v.strip().lower() in ("1", "true", "on", "yes")


def _learned_head_path(seed: int) -> str:
    if "{seed}" in _LEARNED_HEAD_PATH_TEMPLATE:
        return _LEARNED_HEAD_PATH_TEMPLATE.format(seed=seed)
    return _LEARNED_HEAD_PATH_TEMPLATE


# ── BPE tokenizer mode (2026-09-03, "own-voice fluency retrain" decode wiring, DEFAULT-OFF). The checkpoint this
# module has shipped against so far (`wkv_ssmU6_v1000_d128_seed{seed}.npz`) is WORD-level: each vocabulary entry
# in the checkpoint's own `words` array is one whole TinyStories word, so `_free_gen`'s final join
# (`" ".join(ro.words[i] for i in gen ...)`) is a correct detokenizer for it. The in-flight "own-voice fluency"
# retrain (`research.runners._emerge_wkv_lm_derisk --tokenizer bpe --corpus data/corpus/simplewiki.txt`, 6-seed,
# GPU, running as of this comment -- see research/findings/raw/_emerge_wkv_lm_simplewiki_6seed.log) instead uses
# `_BPEVocabAdapter` (that module) to tokenize with `sim.bpe_tokenizer.BPETokenizer`, so ITS checkpoint's `words`
# array holds Sennrich-BPE SUBWORD SYMBOLS (`</w>`-suffixed at a word boundary, e.g. "un", "happi", "ness</w>"),
# not whole words -- naively space-joining them would emit "un happi ness" instead of "unhappiness". This block
# wires the fix: an opt-in tokenizer mode that, when set, decodes (and encodes the prompt) via the SAME
# `BPETokenizer` class the trainer used, reusing its `.decode`/`.encode` VERBATIM (no `sim/` edit; see
# `_get_bpe_tokenizer`). DEFAULT is "word" -- byte-identical to every call site and every behavior that existed
# before this block (see `_free_gen`'s own `bpe=None` branch, which is the UNMODIFIED original code path).
#
# HONEST SCOPE -- what this rung does NOT do (named here so the gap is visible, not silently assumed away):
#   1. No real BPE checkpoint exists yet. The in-flight training run above does not pass `--save-ssm`, so it
#      persists no weights at all -- only NLL metrics to its `--json` output. A follow-up `--save-ssm`-enabled
#      run is needed regardless of the NLL verdict before this mode has anything real to load.
#   2. `WKVReadout` (`research.runners._wkv_fewspike_read_derisk`, reused unmodified here) only reads a SINGLE
#      WKV block's weights (`Wv.weight`/`Wr.weight`/`Wo_sp.weight`/`head.*`) -- it has no code path for the
#      `extra.*` residual layers `_emerge_wkv_lm_derisk --n-layers 2` (the in-flight run's own config) would add
#      to a `--save-ssm` checkpoint's state dict. Loading an n-layers>1 checkpoint here would silently ignore the
#      second layer's weights, not error -- `WKVReadout` needs an `--n-layers`-aware extension before a
#      multi-layer checkpoint can be read correctly. Named, not fixed, in this rung.
#   3. `in_vocab_scope`'s function-word/content-word heuristic below is untouched and still assumes a WORD-level
#      vocabulary; it is not meaningful over a BPE subword vocabulary. Routing a NEW BPE checkpoint into the live
#      `answer_turn` dispatch (`webapp/open_ended_chat.py`) needs its own scoping decision -- out of scope here.
# This rung's OWN scope is exactly the token-id<->text boundary: given a BPE-vocabulary checkpoint already
# loaded by `WKVReadout`, decode its generated ids into real detokenized text (and encode a prompt into ids),
# proven end-to-end against a tiny synthetic checkpoint (see the module's own test companion).
_TOKENIZER_ENV = "BRAIN_WKV_MOUTH_TOKENIZER"                 # "word" (default) | "bpe"
_BPE_PATH_ENV = "BRAIN_WKV_MOUTH_BPE_PATH"
_DEFAULT_BPE_PATH = os.environ.get(_BPE_PATH_ENV, str(_REPO_ROOT / "bridges/wkv_ckpt/wkv_bpe8k.json"))


def tokenizer_mode() -> str:
    """'word' (default, unset or anything other than 'bpe') -> `generate()`'s pre-existing word-level prompt-encode
    + space-join decode, BYTE-IDENTICAL to before this function existed. 'bpe' (`BRAIN_WKV_MOUTH_TOKENIZER=bpe`,
    opt-in) -> `generate()` instead encodes the prompt and decodes the generated ids through a real
    `sim.bpe_tokenizer.BPETokenizer` (see `_get_bpe_tokenizer`), matching a BPE-vocabulary checkpoint."""
    v = os.environ.get(_TOKENIZER_ENV, "word").strip().lower()
    return "bpe" if v == "bpe" else "word"


_BPE_TOK_LOCK = threading.Lock()
_BPE_TOK_CACHE: dict[str, object] = {}


def _get_bpe_tokenizer(path: str | None = None):
    """A cached `sim.bpe_tokenizer.BPETokenizer` loaded from `path` (default `_DEFAULT_BPE_PATH`, the SAME
    `bridges/wkv_ckpt/wkv_bpe8k.json` artifact `research.runners._emerge_wkv_lm_derisk.DEFAULT_BPE_PATH` trains
    against -- so a BPE checkpoint's own vocabulary ids are, BY CONSTRUCTION, the SAME id space as this loaded
    tokenizer's `.vocab`/`.merges`, see `_BPEVocabAdapter` in that module). Lazy-imported (only when a caller
    actually requests 'bpe' mode) so the default 'word' path never even imports `sim.bpe_tokenizer` -- matching
    this module's existing "off imports nothing" discipline (see the module docstring's DEFAULT-OFF paragraph)."""
    p = path or _DEFAULT_BPE_PATH
    hit = _BPE_TOK_CACHE.get(p)
    if hit is not None:
        return hit
    with _BPE_TOK_LOCK:
        hit = _BPE_TOK_CACHE.get(p)
        if hit is not None:
            return hit
        from sim.bpe_tokenizer import BPETokenizer          # lazy: only 'bpe' mode ever imports this
        bt = BPETokenizer.load(p)
        _BPE_TOK_CACHE[p] = bt
        return bt


# ── the checkpoint (pure np.load — no RNG effect, cached) ─────────────────────────────────────────────────────────
_CKPT_LOCK = threading.Lock()
_CKPT_CACHE: dict[tuple, tuple] = {}
_HEAD_INFO: dict[tuple, dict] = {}


def _ckpt_path(seed: int) -> str:
    if "{seed}" in _CKPT_TEMPLATE:
        p = _CKPT_TEMPLATE.format(seed=seed)
        if Path(p).exists():
            return p
        return _CKPT_TEMPLATE.format(seed=42)          # a per-seed ckpt may be missing; seed42 always ships
    return _CKPT_TEMPLATE


def _apply_learned_head(ro, seed: int) -> dict:
    """Opt-in (`learned_head_enabled()`): overrides `ro.head_w` IN PLACE with the e-prop LOCALLY-LEARNED read-out
    matrix `W_hat[V,D]` persisted by `_wkv_mouth_readout_eprop_batched_substrate_derisk.py --save-w-hat` (or the
    host-linear-forward `_wkv_mouth_readout_eprop_learn_derisk.py`, same [V,D] shape/basis if ever extended to
    save). `head_b` (the base-rate prior) is left UNTOUCHED -- both eprop runners explicitly keep `head_b` COPIED,
    only `W_hat` is locally-learned (see their docstrings). Fails SAFE: on a missing file or a shape mismatch, the
    native checkpoint head is left in place and the reason is recorded (never raised) -- this opt-in path must
    never be able to break the default-off caller. Returns a provenance dict (also cached in `_HEAD_INFO`)."""
    info = {"enabled": True, "path": _learned_head_path(seed), "applied": False, "reason": None}
    p = Path(info["path"])
    if not p.exists():
        info["reason"] = "file_missing"
        return info
    try:
        d = np.load(p, allow_pickle=True)
        W_hat = np.asarray(d["W_hat"], dtype=np.float64)
    except Exception as exc:                                    # noqa: BLE001 — fail-safe by design, see docstring
        info["reason"] = f"load_failed:{exc}"
        return info
    if W_hat.shape != ro.head_w.shape:
        info["reason"] = f"shape_mismatch:{W_hat.shape}_vs_{ro.head_w.shape}"
        return info
    ro.head_w = W_hat                                            # the swap: identical [V,D] linear-map basis
    info["applied"] = True
    for k in ("sub_recov_ratio", "sub_learned_recov", "sub_copied_recov", "integrated_go"):
        if k in d:
            info[k] = d[k].item() if hasattr(d[k], "item") else d[k]
    return info


def _get_readout(seed: int):
    learned = learned_head_enabled()
    key = (seed, learned)
    hit = _CKPT_CACHE.get(key)
    if hit is not None:
        return hit
    with _CKPT_LOCK:
        hit = _CKPT_CACHE.get(key)
        if hit is not None:
            return hit
        ro = WKVReadout(_ckpt_path(seed))
        if learned:
            _HEAD_INFO[key] = _apply_learned_head(ro, seed)      # native head kept on any failure (fail-safe)
        vocab = set(w.lower() for w in ro.words)
        word_to_id = {w.lower(): i for i, w in enumerate(ro.words)}
        hit = (ro, vocab, word_to_id)
        _CKPT_CACHE[key] = hit
        return hit


def learned_head_status(seed: int = 42) -> dict | None:
    """Diagnostic: provenance of the last `_get_readout(seed)` call's learned-head load attempt under the
    CURRENT `learned_head_enabled()` state (None if that state was never off/on-attempted for this seed yet)."""
    return _HEAD_INFO.get((seed, learned_head_enabled()))


# ── RNG isolation (the #77 fix, replicated from affect_drives_chat.ObservedAffectLadder._isolated) ────────────────
class _RngIsolation:
    """Run `fn()` on a PRIVATE, per-seed, continuous RNG timeline; leave the host process-global numpy/cupy/random
    RNG byte-untouched as an observable side effect. `FewSpikeWordRead._build_bank` reseeds those globals via
    `cfg.seed` (`sim/bridge.py`) -- without this wrapper, enabling this module would perturb every OTHER
    RNG-dependent organ that runs later in the SAME process."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state: dict[int, dict] = {}

    def run(self, seed: int, fn):
        xp = None
        try:
            from sim.backend import get_backend
            xp, _ = get_backend()
        except Exception:
            xp = None
        with self._lock:
            import random as _random
            host_np = np.random.get_state()
            host_py = _random.getstate()
            host_xp = None
            if xp is not None and xp is not np:
                try:
                    host_xp = xp.random.get_random_state().get_state()
                except Exception:
                    host_xp = None
            priv = self._state.get(seed)
            if priv is None:
                np.random.seed(seed)
                _random.seed(seed)
                if xp is not None and xp is not np:
                    try:
                        xp.random.seed(seed)
                    except Exception:
                        pass
            else:
                try:
                    np.random.set_state(priv["np"])
                    _random.setstate(priv["py"])
                except Exception:
                    pass
                if xp is not None and xp is not np and priv.get("xp") is not None:
                    try:
                        xp.random.get_random_state().set_state(priv["xp"])
                    except Exception:
                        pass
            try:
                return fn()
            finally:
                st = {"np": np.random.get_state(), "py": _random.getstate(), "xp": None}
                if xp is not None and xp is not np:
                    try:
                        st["xp"] = xp.random.get_random_state().get_state()
                    except Exception:
                        st["xp"] = None
                self._state[seed] = st
                try:
                    np.random.set_state(host_np)
                    _random.setstate(host_py)
                except Exception:
                    pass
                if host_xp is not None:
                    try:
                        xp.random.get_random_state().set_state(host_xp)
                    except Exception:
                        pass


_RNG = _RngIsolation()


# ── scope gate (honest residual #1 — V=1000 TinyStories vocab, not a general-purpose vocabulary) ──────────────────
# A pure function-word overlap check is GAMEABLE: "the a to and" or "what is your opinion on the stock market
# today" both score >=60% word-overlap against the checkpoint's vocab purely off ultra-common English function
# words, despite carrying ZERO TinyStories-domain CONTENT (found by an independent verify-go skeptic, 2026-08-28,
# adversarially testing this exact gate). The checkpoint's own top-40-by-frequency words are dominated by such
# function words ("the/and/a/to/was/they/he/it/she/her/with/in/his/you/but/not/on/i/of/there/so/for/that" —
# `bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed42.npz::words[:40]`), so they cannot be excluded from the checkpoint's
# OWN vocabulary set; instead this gate additionally requires `min_content_hits` NON-function-word matches, so a
# prompt must carry genuine TinyStories-domain CONTENT (a noun/verb the checkpoint actually knows), not just
# grammatical scaffolding, before it is routed to the WKV mouth.
_FUNCTION_WORDS = frozenset("""
the and a to was they he it she her with in his you but not on i of there so for that is are am
this these those of at as by from into onto up down over under again further then once here there
all any both each few more most other some such no nor only own same so than too very s t can will
just don should now had has have do does did doing having been being am were what which who whom
your my our its today now yesterday tomorrow
""".split())


# ── lead-in-phrase loophole fix (2026-09-01, closes the `research/FAILURE_LOG.md` entry logged the same day by
# `research/findings/2026-09-01-wkv-mouth-fact-grounding-lever.md`'s Part 2 measurement). `in_vocab_scope`'s
# `min_content_hits` check previously credited ANY non-`_FUNCTION_WORDS` vocab hit as genuine domain content --
# including the caller's OWN conversational lead-in template ("tell me about", "what do you know about", ...,
# the SAME phrases `webapp.open_ended_chat._LEADINS` strips before topic extraction; word set duplicated here so
# this module stays import-independent of `open_ended_chat` -- keep in sync if `_LEADINS` changes). A handful of
# those template words ("tell", "me", "about", "know", "think", "an", "describe", "explain", "what's", "who's")
# sit in this checkpoint's V=1000 vocabulary AND are not `_FUNCTION_WORDS` (grammatical scaffolding), so a FIXED
# lead-in phrase alone could satisfy `min_content_hits=2` regardless of the actual topic that followed --
# `in_vocab_scope("tell me about " + <anything>)` measured True 68.17% of the time on a random real-store sample,
# only 25.9-40.5% of which the topic's own facts could genuinely express (2026-09-01 fact-grounding finding,
# Part 1/2). These are meta-conversational REQUEST verbs ("tell me", "describe", "explain") or grammatical
# fragments of a question template ("what's", "who's", "an") -- never themselves the story-domain CONTENT a
# TinyStories-trained checkpoint should be credited for recognizing, the same reasoning `_FUNCTION_WORDS` already
# applies to plain grammatical scaffolding. (Superset of every word appearing across `_LEADINS`'s phrases --
# several, e.g. "what"/"can"/"you"/"do"/"is"/"a"/"the"/"are"/"who"/"was", are already in `_FUNCTION_WORDS` and
# listed again here only for legibility against that source list; the union with `_FUNCTION_WORDS` makes the
# duplication harmless.)
_LEADIN_WORDS = frozenset("""
what can you tell me about do know think is a an the are what's who was who's describe explain
""".split())


# ── fact-grounding lever (board #112 "clean unlock" — the moat-soak's named next action, see
# research/findings/2026-09-01-open-ended-bundle-moat-safety-soak-fabrication-delta.md and
# research/findings/2026-09-01-wkv-mouth-fact-grounding-lever.md). HONEST SCOPE, stated up front: this checkpoint's
# V=1000 vocabulary is closed (word-level, no subword fallback) and was trained on TinyStories, not Wikidata --
# research/findings/2026-08-31-wkv-mouth-rung4-vocab-coverage.md already measured only 9.55% raw corpus-word
# coverage for THIS checkpoint. A direct token-level measurement against the real shipped `wikidata_core_15k`
# store's facts.json (15000 AFFIRM triples) finds the SAME structural ceiling from the fact side: only 25.9% of
# facts have >=1 real CONTENT word (i.e. excluding `_FUNCTION_WORDS`) whose literal string sits in this
# checkpoint's vocabulary at all (most of the 74.1% majority is Wikidata slugs -- `rugby_leauge`,
# `castleford_f_c`, `deutsche_arbeiter_partei` -- that never had an embedding trained for them and structurally
# CANNOT be produced by this fixed-vocabulary decoder, full stop; there is no way to inject an unseen word's
# meaning into a fixed [V,D] embedding table without retraining). This lever does NOT close that structural gap
# -- that would need a wider-vocab / subword checkpoint (the V=4000 checkpoint already measured, not yet wired,
# per the rung-4 finding above) or new grounded fine-tuning. What THIS lever does: for the ~26% of facts whose
# content word DOES already exist in this checkpoint's vocabulary, increase (not force) the odds the genuine
# few-spike spiking WTA actually SELECTS that TRUE word during free generation, instead of leaving it to chance
# alongside whatever TinyStories-plausible-but-unsupported word the model would otherwise favor.
def fact_grounding_ids(facts, seed: int = 42, max_ids: int = 6) -> list:
    """Decompose a list of (agent, action, patient) triples (the SAME shape `webapp.open_ended_chat.retrieve`
    returns) into their real CONTENT words (excluding `_FUNCTION_WORDS`) and map each to this checkpoint's
    vocabulary id where one exists. Returns a de-duplicated, ORDER-PRESERVING list of up to `max_ids` token ids
    (patient words first, then action, then agent -- the patient is normally the informative slot: "sport" ->
    "basketball", not the entity's own name). Empty facts, or facts with zero in-vocab content words, return `[]`
    -- the honest majority case for this store (only fed a `facts=None`/falsy list from the caller in that case
    anyway, but this function is itself pure/side-effect-free so it is safe to call regardless)."""
    if not facts:
        return []
    _, _vocab, word_to_id = _get_readout(seed)
    ids = []
    seen = set()
    for triple in facts:
        if len(triple) != 3:
            continue
        _agent, action, patient = triple
        for field in (patient, action, _agent):
            for w in _WORD_RE.findall(str(field)):
                wl = w.lower()
                if wl in _FUNCTION_WORDS:
                    continue
                tid = word_to_id.get(wl)
                if tid is None or tid in seen:
                    continue
                seen.add(tid)
                ids.append(tid)
                if len(ids) >= max_ids:
                    return ids
    return ids


# ── fact->SENTENCE rendering (board #112 rung 3 wire-in, 2026-09-01 — the "clean unlock" gate). Reuse-by-import
# of the JUST-MERGED lexicon lever (`research.runners._wkv_fact_to_sentence_lexicon_lever`, 6-seed GO:
# readable=1.0, faithful=1.0, moat_safe=1.0 on every one of 48 real sampled facts, 34/34 live-relation
# coverage) driving the already-6-seed-GO `SpikingClauseProducer` (`research.runners._spiking_fluent_surface_
# derisk`, EMERGE-59/60/61) UNMODIFIED. THIS is the actual wire-in the lever's own finding named as the open
# residual: unlike that lever's own runner (a PARALLEL renderer never reachable from `answer_turn`),
# `render_fact_sentence` is called FROM `generate()`'s own `_run()` closure below via the new `sentence_facts`
# parameter -- a known-topic reply is genuinely written by the mouth's fact->sentence path, not a side-channel.
# Lazy-imported (only when actually invoked) so the default-OFF path (`sentence_facts=None`, the parameter's
# own default and every pre-existing call site) never even imports the lever module -- this module's own
# import-time behavior is unaffected either way, matching the "off imports nothing" discipline the module
# docstring already states for `BRAIN_OPEN_ENDED_WKV_MOUTH` itself.
_CLAUSE_PROD_LOCK = threading.Lock()
_CLAUSE_PROD_CACHE: dict[tuple, object] = {}


def _get_clause_producer(seed: int, n_slots: int):
    """A `SpikingClauseProducer` built + taught ONCE per (seed, slot-count), then reused across calls/turns --
    the SAME reuse pattern the lexicon lever's own `_render_facts` already uses within one seed's fact list
    (its `by_len` dict there); this just extends the cache lifetime process-wide, exactly like `_get_readout`'s
    own WKV-checkpoint cache above. Every `emit()` call still runs the EMERGE-61 inter-utterance wash-out
    (`_restore_state(..., neuron_idx="slots")`) before reading rates, so reusing the producer object across
    turns/facts is safe and does not leak state between emissions -- verified by the lexicon lever's own
    6-seed GO, which reuses producers across all 8 facts of a seed this exact same way."""
    key = (seed, n_slots)
    hit = _CLAUSE_PROD_CACHE.get(key)
    if hit is not None:
        return hit
    with _CLAUSE_PROD_LOCK:
        hit = _CLAUSE_PROD_CACHE.get(key)
        if hit is not None:
            return hit
        from research.runners._spiking_fluent_surface_derisk import SpikingClauseProducer
        prod = SpikingClauseProducer(seed)
        prod.learn(n_slots)
        _CLAUSE_PROD_CACHE[key] = prod
        return prod


def pick_covered_fact(facts):
    """The first (agent, action, patient) triple in `facts` whose relation exists in the lexicon lever's
    `RELATION_LEXICON` (34/34 live-store coverage) -- i.e. the first fact this rung can render as a genuinely
    COHERENT sentence, not the naive-morphology/raw-slug fallback. `None` when `facts` is empty/falsy or no
    triple's relation is covered (the honest degrade case for a hypothetical future 35th relation type; none
    exist in the live store today per the lever's own `_check_lexicon_coverage`)."""
    if not facts:
        return None
    from research.runners._wkv_fact_to_sentence_lexicon_lever import RELATION_LEXICON
    for triple in facts:
        if len(triple) != 3:
            continue
        _agent, action, _patient = triple
        if action in RELATION_LEXICON:
            return triple
    return None


def render_fact_sentence(facts, seed: int = 42) -> str | None:
    """Render ONE recalled fact from `facts` (the SAME (agent, action, patient) triples
    `webapp.open_ended_chat.retrieve` returns) as a coherent factual clause, via the merged board #112 rung-3
    lexicon (`RELATION_LEXICON` + `slug_to_np`) driving the UNMODIFIED, already-6-seed-GO
    `SpikingClauseProducer` -- e.g. `("bounce_around_the_ground", "country", "united_kingom")` -> "the Bounce
    Around the Ground is located in the United Kingom". Returns `None` (NEVER a fabricated or naive-morphology
    guess) when `facts` carries no relation this lexicon covers -- the caller (`generate()`) then falls
    through to the pre-existing free-generation path, unchanged.

    MOAT-SAFE by construction (the same property the lever's own `parse_and_score` verified 1.0/6-seed): every
    token in the returned surface is either the fact's own subject/object NP or a fixed closed-class
    predicate/determiner word -- nothing else can appear.

    Must be called from inside `_RngIsolation.run` (see `generate()`'s `_run()` below) -- `SpikingClauseProducer.
    __init__` (reached on a cache miss, see `_get_clause_producer`) builds a real `SimulationBridge`
    (`cfg.seed=seed`), which reseeds the process-global RNGs exactly like `WKVReadout` does (the #77 footgun
    this module's own `_RngIsolation` class exists to guard)."""
    triple = pick_covered_fact(facts)
    if triple is None:
        return None
    from research.runners._wkv_fact_to_sentence_lexicon_lever import _dctx_and_slots
    agent, action, patient = triple
    slots, dctx, covered = _dctx_and_slots(agent, action, patient)
    if not covered:                      # pick_covered_fact already guarantees this; belt-and-braces
        return None
    prod = _get_clause_producer(seed, len(slots))
    words = prod.emit(slots, dctx)
    if not prod.spiked:
        # honesty: never claim a spiking-produced sentence the bridge did not genuinely spike for
        return None
    return " ".join(words)


def _apply_fact_boost(lg: np.ndarray, fact_ids, boost: float) -> np.ndarray:
    """Additive decode-time logit boost for `fact_ids` (see `fact_grounding_ids`) -- legitimately HOST territory,
    the SAME category as the pre-existing `_apply_repetition_controls` (decode control over WHICH candidates
    reach the spiking WTA read; the read mechanism itself, `reader.read(p)`, is never touched). Returns `lg`
    UNCHANGED (same object, no allocation) when `fact_ids` is empty or `boost == 0.0` -- the exact no-op default
    both the caller's own default (`fact_boost_ids=None`) and every pre-existing call site hit before this
    function existed."""
    if not fact_ids or boost == 0.0:
        return lg
    lg = lg.copy()
    v = lg.shape[0]
    for t in fact_ids:
        if 0 <= t < v:
            lg[t] += boost
    return lg


def in_vocab_scope(text: str, seed: int = 42, min_frac: float = 0.6, min_hits: int = 2,
                    min_content_hits: int = 2) -> bool:
    """True only when `text` carries substantial genuine CONTENT overlap with the checkpoint's V=1000 TinyStories
    vocabulary -- not merely common English function words the checkpoint's vocab (frequency-sorted) is itself
    dominated by, AND NOT merely the caller's own conversational lead-in template ("tell me about", "what do you
    know about", ...). The caller (`answer_turn`) MUST fall back to Qwen when this is False -- this module never
    attempts an out-of-scope prompt itself. Three conditions, all required: (1) `min_hits` total vocab-word
    matches (guards very short prompts), (2) `min_frac` overall word-overlap fraction, (3) `min_content_hits`
    matches that are NOT in `_FUNCTION_WORDS` OR `_LEADIN_WORDS` -- closes both the stopword-only /
    function-word-dominated false-positive an adversarial verify-go pass found in the first version of this gate
    (2026-08-28), AND the lead-in-phrase-alone false-positive the 2026-09-01 fact-grounding measurement found
    (`in_vocab_scope("tell me about " + <any nonsense>)` was True 68.17% of the time on a real-topic sample --
    see `_LEADIN_WORDS`'s own docstring above and `research/FAILURE_LOG.md` 2026-09-01). `min_hits`/`min_frac`
    are still scored over the FULL original `text` (unchanged) -- only which hits count as CONTENT changes, so a
    genuinely content-bearing message is not penalized for also carrying a lead-in phrase."""
    _, vocab, _ = _get_readout(seed)
    words = [w.lower() for w in _WORD_RE.findall(text or "")]
    if not words:
        return False
    hits = [w for w in words if w in vocab]
    content_hits = [w for w in hits if w not in _FUNCTION_WORDS and w not in _LEADIN_WORDS]
    return (len(hits) >= min_hits and (len(hits) / len(words)) >= min_frac
            and len(content_hits) >= min_content_hits)


# ── decode-time repetition guard (the A/B GO's named next lever, per research/findings/2026-08-28-wkv-learned-
# vs-native-head-AB-worth-keeping-opt-in.md SS5: "a repetition penalty / no-repeat n-gram constraint at
# generation time (cheap, host-side, does not touch the learned weights)"). Two default-off knobs, applied to
# the FULL-vocab logits `lg` -- NOT to `p` after the top-k cut -- so a banned/penalized token cannot re-enter
# the top-`topk` candidate set the spiking reader samples over. Defaults (1.0, 0) are an EXACT no-op: this is
# legitimately HOST territory (decode control, same category as the pre-existing `topk`/`gen_temp` knobs) --
# the spiking population-coded WTA read (`reader.read(p)`) is never touched, so the brain-based read mechanism
# stays exactly what it was; only WHICH candidates reach it changes.
def _apply_repetition_controls(lg: np.ndarray, gen: list, repetition_penalty: float, no_repeat_ngram_size: int):
    """Returns a (possibly new) logits array with decode-time repetition controls applied. Byte-identical
    no-op (`lg` returned UNCHANGED, same object) when `repetition_penalty == 1.0` and
    `no_repeat_ngram_size <= 0` -- the default-off path never allocates or mutates.

    (a) `repetition_penalty` (CTRL-style, Keskar et al. 2019 / the HF `RepetitionPenaltyLogitsProcessor`
        convention): for every token id already in `gen`, `lg[t] = lg[t]/rp if lg[t]>0 else lg[t]*rp` --
        rp>1 SUPPRESSES re-selection of anything already generated, symmetric about the sign of the logit
        so a penalty never flips a strongly-negative logit positive.
    (b) `no_repeat_ngram_size=n>0` (Fan et al. 2018 / the HF `NoRepeatNGramLogitsProcessor` convention):
        for the current (n-1)-token suffix of `gen`, hard-ban (`-1e30`) every token that would complete an
        n-gram already seen earlier in `gen` -- an exact constraint, not a soft nudge."""
    if repetition_penalty == 1.0 and no_repeat_ngram_size <= 0:
        return lg
    lg = lg.copy()
    if repetition_penalty != 1.0:
        for t in set(gen):
            lg[t] = lg[t] / repetition_penalty if lg[t] > 0 else lg[t] * repetition_penalty
    if no_repeat_ngram_size > 0:
        n = no_repeat_ngram_size
        prefix_len = n - 1
        banned_prefix = tuple(gen[len(gen) - prefix_len:]) if prefix_len > 0 else ()
        for i in range(len(gen) - n + 1):
            if tuple(gen[i:i + prefix_len]) == banned_prefix:
                lg[gen[i + prefix_len]] = -1e30
    return lg


# ── generation (reuses WKVReadout + FewSpikeWordRead verbatim; only the driving loop is new) ───────────────────────
def _free_gen(ro, vocab_ids_by_word, reader, prompt: str, seed: int, max_new_tokens: int, topk: int, gen_temp: float,
              repetition_penalty: float = 1.0, no_repeat_ngram_size: int = 0,
              fact_boost_ids=None, fact_boost: float = 0.0, bpe=None):
    """`bpe` (default `None`, added 2026-09-03): a `sim.bpe_tokenizer.BPETokenizer` instance (see
    `_get_bpe_tokenizer`) for BPE-vocabulary checkpoints. `bpe=None` (the default, and every pre-existing call
    site) runs the EXACT SAME two lines that existed before this parameter did -- the word-level prompt-id
    lookup and the final space-joined decode -- so the default path is byte-identical BY CONSTRUCTION, not
    merely by equivalent behavior. When `bpe` is given, both the prompt encode and the final decode instead go
    through that tokenizer's own `.encode`/`.decode` (Sennrich-BPE subword id space, matching what produced the
    checkpoint's `words` vocabulary in the first place -- see the module's BPE-mode block above)."""
    if bpe is not None:
        pid = bpe.encode(prompt or "")
    else:
        pid = [vocab_ids_by_word[w] for w in (t.lower() for t in _WORD_RE.findall(prompt or "")) if w in vocab_ids_by_word]
    if not pid:
        pid = [0]
    ap = np.zeros(ro.D)
    an = np.zeros(ro.D)
    for t in pid:
        ap, an = ro.advance(ap, an, t)
    gen = list(pid)
    self_nll = 0.0
    steps = 0
    for _ in range(max_new_tokens):
        lg = ro.logits(ap, an, gen[-1])
        if ro.unk_idx >= 0:
            lg = lg.copy()
            lg[ro.unk_idx] = -1e30
        lg = _apply_repetition_controls(lg, gen, repetition_penalty, no_repeat_ngram_size)
        # fact-grounding boost (board #112 clean-unlock lever, default off): applied AFTER the repetition guard,
        # on the SAME full-vocab logits the repetition controls already touched -- so a boosted fact token can
        # still be repetition-suppressed/n-gram-banned like any other candidate, and a repetition-banned token
        # cannot be un-banned by the boost (an additive `+boost` cannot overcome a `-1e30` hard ban). No-op
        # (same object, no allocation) when `fact_boost_ids` is empty or `fact_boost == 0.0`.
        lg = _apply_fact_boost(lg, fact_boost_ids, fact_boost)
        cand = np.argpartition(-lg, topk - 1)[:topk]
        cand = cand[np.argsort(-lg[cand])]
        p = _softmax(lg[cand] / gen_temp)
        win, _, _ = reader.read(p)
        nxt = int(cand[win]) if win >= 0 else int(cand[0])
        pfull = _softmax(lg)
        self_nll += -float(np.log(max(pfull[nxt], 1e-12)))
        steps += 1
        gen.append(nxt)
        ap, an = ro.advance(ap, an, nxt)
        if ro.words[nxt] == "endoftext":
            break
    if bpe is not None:
        kept = [i for i in gen if 0 <= i < len(ro.words) and ro.words[i] != "endoftext"]
        text = bpe.decode(kept)
    else:
        text = " ".join(ro.words[i] if 0 <= i < len(ro.words) else "<unk>" for i in gen if ro.words[i] != "endoftext")
    return text, (self_nll / max(1, steps)), gen


def generate(prompt: str, seed: int = 42, max_new_tokens: int = 60, topk: int = 64, read_window: int = 40,
             pop: int = 8, gen_temp: float = 0.8, repetition_penalty: float = 1.0,
             no_repeat_ngram_size: int = 0, facts=None, fact_boost: float = 6.0,
             sentence_facts=None) -> tuple[str, float]:
    """Free-generate a continuation of `prompt` via the GENUINE few-spike Izhikevich spiking soft-WTA word decode
    (`FewSpikeWordRead.read`, population-coded winner read off `cp_firing_states` over `read_window` Izhikevich
    steps -- NOT a host argmax/softmax-sample; reused verbatim from the GO-verified
    `research.runners._wkv_fewspike_read_derisk` module). Mirrors `OpenEndedGenerator.generate`'s `(text, seconds)`
    return shape so it drops into `webapp/open_ended_chat.py::answer_turn`'s generator slot unchanged. Runs on a
    private RNG timeline (see `_RngIsolation`) -- never perturbs host process-global RNG state.

    `repetition_penalty` (default 1.0) and `no_repeat_ngram_size` (default 0) are DEFAULT-OFF decode-time
    repetition guards -- see `_apply_repetition_controls` -- addressing the repetition/looping residual named
    as the next lever by research/findings/2026-08-28-wkv-learned-vs-native-head-AB-worth-keeping-opt-in.md
    SS5. Left at their defaults, `generate()` is byte-identical to before these kwargs existed.

    `facts` (default None) and `fact_boost` (default 6.0, inert while `facts` is None/empty): the board #112
    fact-grounding lever. When `facts` (the SAME (agent, action, patient) triple list `open_ended_chat.retrieve`
    returns) is truthy, its in-vocab CONTENT-word ids (`fact_grounding_ids`) get an additive decode-time logit
    boost every generation step (`_apply_fact_boost`) -- byte-identical to before this parameter existed when
    `facts` is None (the pre-existing default and every pre-existing call site).

    `sentence_facts` (default None, INDEPENDENT of `facts`/`fact_boost` above -- board #112 rung-3 wire-in,
    2026-09-01): when truthy, `_run()` tries `render_fact_sentence(sentence_facts, seed=seed)` FIRST, before
    any free generation. If it returns a coherent clause (a covered relation was found -- see that function),
    THAT clause is `generate()`'s entire return text, and `_free_gen`/the WKV checkpoint's own free decode
    never runs for this turn -- a known-topic reply is genuinely written by the mouth's fact->sentence path,
    not a parallel renderer. If it returns `None` (no covered relation in `sentence_facts`), `_run()` falls
    straight through to the SAME free-generation path as before this parameter existed (still honoring `facts`/
    `fact_boost` if those were also passed) -- so this parameter can only ADD a generator choice, never remove
    the pre-existing fallback. `sentence_facts=None` (the default and every pre-existing call site) is
    BYTE-IDENTICAL to before this parameter existed: `render_fact_sentence` is never even imported.

    TOKENIZER MODE (`BRAIN_WKV_MOUTH_TOKENIZER`, default 'word', see `tokenizer_mode`/the module's BPE-mode
    block above): 'word' (the default) is BYTE-IDENTICAL to before this env var existed -- `_free_gen` runs
    `bpe=None`, its original two lines unchanged. 'bpe' encodes/decodes through a real `BPETokenizer` instead,
    for a BPE-vocabulary checkpoint (e.g. `BRAIN_WKV_MOUTH_CKPT` pointed at the in-flight own-voice retrain's
    eventual `--save-ssm` output, once one exists -- see the honest-scope note above for what is NOT yet true
    of that checkpoint)."""
    t0 = time.time()

    def _run():
        if sentence_facts:
            sentence = render_fact_sentence(sentence_facts, seed=seed)
            if sentence is not None:
                return sentence
        ro, _vocab, word_to_id = _get_readout(seed)
        reader = FewSpikeWordRead(topk, pop, seed, read_window=read_window)
        fact_ids = fact_grounding_ids(facts, seed=seed) if facts else None
        bpe = _get_bpe_tokenizer() if tokenizer_mode() == "bpe" else None
        text, _self_nll, _gen = _free_gen(ro, word_to_id, reader, prompt, seed, max_new_tokens, topk, gen_temp,
                                          repetition_penalty, no_repeat_ngram_size,
                                          fact_boost_ids=fact_ids, fact_boost=fact_boost, bpe=bpe)
        return text

    text = _RNG.run(seed, _run)
    return text, round(time.time() - t0, 3)
