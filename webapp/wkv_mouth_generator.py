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

BPE-CAPS FIX (2026-09-04, closes the DOMINANT broad-scope coverage blocker research/findings/2026-09-03-linattn-
mouth-broad-scope-coverage-threshold.md's Result 2 measured: a case-folding bug, not topic mismatch, worth ~5.6x
in teacher-forced perplexity on its own, 12827 -> 2284). Both checkpoint families' vocabularies are trained
EXCLUSIVELY on lowercase text, so a raw-case prompt's capitals BPE-encode to `<UNK>` on the way IN, and the
model structurally cannot emit a capital letter on the way OUT. Two independent, independently-guarded fixes,
both default-ON: INPUT — `_bpe_encode_prompt`/`bpe_lowercase_enabled()` lowercase the prompt before BPE-encoding
it (`_free_gen`/`_free_gen_linattn`'s only two `bpe.encode` call sites); OUTPUT — `_truecase`/`truecase_enabled()`
restore sentence-initial + pronoun-"I" + a small known-name-allowlist capitalization on `generate()`'s final
text. See `generate()`'s own docstring for the full mechanism and the two flags' own docstrings for the
byte-identical-off guard.
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


# ── RECURRENCE family (2026-09-03, `LinAttnReadout` production wiring, DEFAULT-OFF -- see research/findings/
# 2026-09-03-linattn-production-mouth-wiring-DESIGN.md Sec 3e P2). `_get_readout` below is the ONLY place this
# is read: 'ssm' (unset, the default) builds `WKVReadout` exactly as before this function existed -- BYTE-
# IDENTICAL default path; 'linattn' builds `research.runners._wkv_fewspike_read_derisk.LinAttnReadout` against a
# `--recurrence linattn --save-ssm` checkpoint instead (a DIFFERENT, mutually-exclusive recurrence family --
# never silently mixed with 'ssm', matching the "never share a read path" discipline `WKVReadout`'s own
# multi-layer 'wkv' extension states for the identical reason: a checkpoint read through the wrong family's math
# loads WITHOUT error and produces near-uniform garbage, research/findings/2026-07-20-gap1-ROOT-CAUSE-...).
_RECUR_ENV = "BRAIN_WKV_MOUTH_RECURRENCE"                    # "ssm" (default) | "linattn"


def recurrence_mode() -> str:
    """'ssm' (default, unset or anything other than 'linattn') -> `_get_readout` builds `WKVReadout` and
    `generate()` drives it through the pre-existing `_free_gen` -- BYTE-IDENTICAL to before this function
    existed. 'linattn' (`BRAIN_WKV_MOUTH_RECURRENCE=linattn`, opt-in) -> `_get_readout` builds `LinAttnReadout`
    (point `BRAIN_WKV_MOUTH_CKPT` at a `--recurrence linattn --save-ssm` checkpoint) and `generate()` drives it
    through `_free_gen_linattn` instead -- a separate driving-loop function (not a branch inside `_free_gen`)
    because `LinAttnReadout`'s state is one opaque object (`ro.advance(state, tid)`/`ro.logits(state, tid)`),
    not `WKVReadout`'s `(ap, an)` array pair; every decode control (top-K cut, repetition guard, fact-boost, the
    genuine few-spike `reader.read(p)` WTA) is copied verbatim between the two, so linattn composes with the
    exact same spiking read-out for free."""
    v = os.environ.get(_RECUR_ENV, "ssm").strip().lower()
    return "linattn" if v == "linattn" else "ssm"


# ── SCOPE mode (2026-09-03, design doc Sec 3e P3). `in_vocab_scope`'s function-word/content-word overlap check
# below assumes a CLOSED word-level vocabulary (the shipped V=1000 TinyStories checkpoint) -- not meaningful over
# a general-vocabulary BPE checkpoint, which tokenizes any input with no OOV at all (see the module's BPE-mode
# block, residual #3). "broad" mode names that this gate should become a coverage/confidence decision instead of
# a hard vocab-overlap gate -- but the exact threshold is an HONEST, NOT-YET-MEASURED de-risk knob (the design
# doc's own words: "set from the 6-seed's own held-out coverage, NOT guessed here"), so "broad" currently just
# ADMITS every prompt to this gate (the caller's post_filter VERIFY moat still runs unconditionally afterward --
# this is scope-ROUTING, not a safety boundary) rather than fabricating an ungrounded number. Tightening this
# into a real coverage threshold is the named next step once the deployable linattn checkpoint's held-out
# coverage is measured (design doc Sec 6-iii names the fabrication-surface risk this leaves open until then).
_SCOPE_ENV = "BRAIN_WKV_MOUTH_SCOPE"                         # "vocab" (default) | "broad"


def scope_mode() -> str:
    """'vocab' (default, unset or anything other than 'broad') -> `in_vocab_scope`'s pre-existing TinyStories
    word-overlap gate, BYTE-IDENTICAL to before this function existed. 'broad' (`BRAIN_WKV_MOUTH_SCOPE=broad`,
    opt-in) -> `in_vocab_scope` admits every prompt (see this function's own docstring above for the honest
    reason a real coverage threshold is not implemented yet)."""
    v = os.environ.get(_SCOPE_ENV, "vocab").strip().lower()
    return "broad" if v == "broad" else "vocab"


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


# ── BPE-CAPS FIX, INPUT half (2026-09-04, research/findings/2026-09-03-linattn-mouth-broad-scope-coverage-
# threshold.md Result 2 -- the "dominant driver" finding). `sim.bpe_tokenizer.BPETokenizer.train`'s merge table
# is built EXCLUSIVELY over lowercase `[a-z']+` corpus words (`_train_bpe_bounded`'s `raw.lower()`), and the
# shipped `bridges/wkv_ckpt/wkv_bpe8k.json` was itself trained against `data/corpus/simplewiki.txt`, which is
# PRE-LOWERCASED ON DISK (verified by reading the raw file) -- so the checkpoint never saw a single capital
# letter during training. A raw, un-lowercased chat prompt's every capital -- every sentence-initial word, every
# proper noun -- therefore falls outside the trained character alphabet and BPE-encodes to `<UNK>` id 0, one per
# capital letter (`BPETokenizer._encode_word`'s `self._sym_to_id.get(sym, 0)` fallback). Measured directly: mean
# 6-seed teacher-forced perplexity over the 124-utterance realistic-chat probe is 12827.32 AS FED vs 2283.64 once
# capitals are lowercased -- a ~5.6x hit, and (per that finding's Result 3) larger than any plausible topic/
# vocabulary-coverage effect. `_bpe_encode_prompt` below is the ONE place `_free_gen`/`_free_gen_linattn` turn a
# raw prompt into BPE ids -- fixing it here fixes both call sites at once and keeps the coverage/perplexity
# measurement runner (`research.runners._wkv_mouth_linattn_broad_scope_coverage_derisk`) production-faithful by
# CONSTRUCTION if it is ever pointed at this same helper, rather than a hand-copied duplicate that could drift.
_BPE_LOWERCASE_ENV = "BRAIN_WKV_MOUTH_BPE_LOWERCASE"


def bpe_lowercase_enabled() -> bool:
    """Default-ON (this closes a MEASURED ~5.6x production-perplexity gap, not an opt-in trial -- see the module
    comment block above). `BRAIN_WKV_MOUTH_BPE_LOWERCASE` in {0,false,no,off} -> an explicit OFF, reverting
    `_bpe_encode_prompt` to the pre-fix raw-case `bpe.encode(prompt)`, BYTE-IDENTICAL to before this function
    existed. Only ever reached when `bpe is not None` in `_free_gen`/`_free_gen_linattn`, i.e. `tokenizer_mode()
    == 'bpe'` (itself default-OFF, see that function's docstring) -- the shipped word-level default path never
    calls this at all, so this fix has ZERO effect on anything unless BOTH the WKV mouth AND its BPE tokenizer
    mode are already opted into."""
    v = os.environ.get(_BPE_LOWERCASE_ENV)
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def _bpe_encode_prompt(bpe, prompt: str) -> list:
    """The SINGLE source of truth for "what does the BPE-mode mouth path feed the tokenizer" -- called from both
    `_free_gen` and `_free_gen_linattn`'s prompt-encode step (replacing their prior direct `bpe.encode(prompt or
    "")` call), and reusable by a measurement runner that wants to stay production-faithful rather than
    reimplementing this logic. `bpe_lowercase_enabled()` (default ON, see above) lowercases `prompt` before
    encoding it -- matching the training distribution and recovering the ~5.6x perplexity measured in the module
    comment block above. Flag OFF reverts to `bpe.encode(prompt or "")` verbatim, byte-identical to every call
    site's behavior before this function existed."""
    text = prompt or ""
    if bpe_lowercase_enabled():
        text = text.lower()
    return bpe.encode(text)


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
    """Byte-identical default path: `recurrence_mode()` unset -> 'ssm' -> builds `WKVReadout` exactly as before
    that function existed (the cache key's 3rd element is a constant "ssm" in that case, so the shipped
    single-seed / single-recurrence cache behavior is unaffected). `recurrence_mode()=="linattn"` builds
    `LinAttnReadout` instead, under its OWN cache key so a seed's ssm and linattn readouts never collide."""
    learned = learned_head_enabled()
    recur = recurrence_mode()
    key = (seed, learned, recur)
    hit = _CKPT_CACHE.get(key)
    if hit is not None:
        return hit
    with _CKPT_LOCK:
        hit = _CKPT_CACHE.get(key)
        if hit is not None:
            return hit
        if recur == "linattn":
            from research.runners._wkv_fewspike_read_derisk import LinAttnReadout
            ro = LinAttnReadout(_ckpt_path(seed))
            # The e-prop learned-head matrices `_apply_learned_head` loads are trained against the ssm/dual-
            # nonneg checkpoint's OWN hidden basis (`bridges/wkv_ckpt/wkv_ssmU6_*`) -- never apply them to a
            # linattn readout's differently-shaped/differently-trained hidden representation, regardless of
            # `BRAIN_WKV_MOUTH_LEARNED_HEAD` (default-ON). `_apply_learned_head`'s own shape check would usually
            # catch a real mismatch anyway (its fail-safe docstring), but this skip is an explicit guard, not a
            # reliance on that shape happening to differ.
        else:
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
    CURRENT `learned_head_enabled()`/`recurrence_mode()` state (None if that state was never off/on-attempted
    for this seed yet, OR if the current recurrence mode is 'linattn' -- the learned-head lever never applies
    there, see `_get_readout`)."""
    return _HEAD_INFO.get((seed, learned_head_enabled(), recurrence_mode()))


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
    genuinely content-bearing message is not penalized for also carrying a lead-in phrase.

    `scope_mode()=="broad"` (`BRAIN_WKV_MOUTH_SCOPE=broad`, opt-in, see that function's docstring) bypasses all
    of the above and returns True unconditionally -- see `scope_mode`'s own docstring for why a real coverage
    threshold is not implemented yet. Default ('vocab') runs the exact check below, byte-identical to before
    this bypass existed."""
    if scope_mode() == "broad":
        return True
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


# ── AFFECT coupling (2026-09-03, closes the affect-hollow gap named by research/findings/2026-09-03-linattn-
# mouth-live-brain-grounded-honest-verification-PARTIAL-affect-gap.md (ii-c): `answer_turn` already assembles a
# LIVE valence/arousal read off the real spiking affect organ (`research.runners.affect_production_organ.
# AffectProductionOrgan.read_differential`, `_valence_from_differential`) into `state`/`system`/`user` for the
# Qwen/gen-time-veto paths, but never passed it to `_WKV.generate()` at all -- `_free_gen`/`_free_gen_linattn`
# took no affect parameter, full stop (a NEVER-IMPLEMENTED gap for the WKV-mouth family, not a dropped wire).
# THE MECHANISM (the SAME decode-control category as `_apply_fact_boost`/`_apply_repetition_controls` above --
# an additive bias on the FULL-vocab logits, applied BEFORE the top-k cut, so the genuine few-spike Izhikevich
# soft-WTA `reader.read(p)` still makes the actual selection among the biased candidates; the read mechanism
# itself is untouched, only which candidates reach it shifts): every checkpoint-vocabulary word that is a
# strongly affect-bearing word in the SAME Warriner-norm-gated, DR-2-learned-value lexicon `research.runners.
# affect_production_organ.appraise_text` already uses to appraise the LIVE conversational turn (reused BY
# IMPORT, not re-derived) gets a signed per-word valence in [-1,1]. The bias added to that word's logit is
# `affect_boost * valence * clip(arousal,0,1) * word_valence` -- POSITIVE (favored) when the turn's mood and the
# word's own valence AGREE in sign (mood-CONGRUENT production, the same direction Bower 1981's mood-congruent
# recall/production effect describes), scaled by arousal as a GAIN term (arousal amplifies an existing
# directional signal; it supplies no direction of its own -- the same shape as LC-noradrenergic arousal-
# dependent gain modulation, Aston-Jones & Cohen 2005, "An integrative theory of locus coeruleus-norepinephrine
# function," Annu. Rev. Neurosci. 28:403-450). `valence == 0.0` (the parameter's own default, and EXACTLY what
# `AffectProductionOrgan.read_differential(..., lesion=True)` clamps the organ's differential -- hence the
# mapped valence -- to, see `affect_production_organ.py`'s own `set_transmission_gate("affect_out", 0.0)`)
# collapses this to an EXACT no-op regardless of arousal, which is what makes `BRAIN_AFFECT_LESION=1` genuinely
# lesion this coupling's effect, not merely dampen it.
#
# HONEST SCOPE: this is HOST decode-territory arithmetic over an already-neurally-sourced signal (the valence/
# arousal floats themselves ARE the real organ's read, not invented here) -- the SAME category the module
# docstring already claims for `_apply_fact_boost`. It is a TRACKED SHORTCUT toward a genuinely neural version
# (e.g. a neuromodulatory gain/threshold term inside `FewSpikeWordRead`'s own Izhikevich population, so arousal
# and valence act on the SPIKING read mechanism itself rather than the logits feeding it) -- named, not resolved,
# here; see the finding's "Honest residual" section. It does NOT touch `render_fact_sentence`'s closed-class
# fact-clause path (facts stay tone-neutral by construction, matching Gate-B's own honesty floor: affect colors
# manner, never the certainty band) -- only `_free_gen`/`_free_gen_linattn`'s free generation.
_AFFECT_ENV = "BRAIN_WKV_MOUTH_AFFECT"


def wkv_mouth_affect_enabled() -> bool:
    """Default-ON (this closes a named production gap, not an opt-in trial). `BRAIN_WKV_MOUTH_AFFECT` in
    {0,false,no,off} -> an explicit OFF, reverting `generate()`'s affect coupling to a hard no-op -- combined
    with the parameter default `valence=0.0`, `arousal=0.0`, this is BYTE-IDENTICAL to `generate()`'s behavior
    before this coupling existed (no import of the affect lexicon, no bias ever computed)."""
    v = os.environ.get(_AFFECT_ENV)
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


_AFFECT_BIAS_LOCK = threading.Lock()
_AFFECT_BIAS_CACHE: dict[int, dict] = {}


def _affect_bias_ids(seed: int) -> dict:
    """{token_id: signed_word_valence in [-1,1]} for every word in the checkpoint's OWN vocabulary that is a
    strongly affect-bearing word per `research.runners.affect_production_organ`'s existing salience-gated
    lexicon (the Warriner-norm gate + the DR-2 LEARNED per-word value when `affect_production_organ.
    dr2_enabled()`) -- REUSE of the shipped Gate-B appraisal artifact (the SAME map that already appraises the
    live user message for the strict/rich-path mood coupling), not a fresh host sentiment formula invented for
    this module. Pure dict build over the checkpoint's ~V words (no RNG effect); cached per seed, mirroring
    `_get_readout`'s own cache discipline. Lazy-imports `affect_production_organ` (only when this coupling is
    actually enabled and a non-neutral valence is in play, see `generate()`'s `_run()`) so the default-OFF /
    neutral-valence path never even imports it."""
    hit = _AFFECT_BIAS_CACHE.get(seed)
    if hit is not None:
        return hit
    with _AFFECT_BIAS_LOCK:
        hit = _AFFECT_BIAS_CACHE.get(seed)
        if hit is not None:
            return hit
        import research.runners.affect_production_organ as _AO
        ro, _vocab, _word_to_id = _get_readout(seed)
        learned = _AO._get_learned_valence() if _AO.dr2_enabled() else {}
        out: dict = {}
        for tid, w in enumerate(ro.words):
            raw = str(w).lower()
            # BPE word-boundary normalization (2026-09-03): a BPE-vocabulary checkpoint (`tokenizer_mode()==
            # "bpe"`) spells a whole word that survived merging intact as e.g. "happy</w>"/"angry</w>", NOT
            # "happy"/"angry" -- confirmed empirically against the shipped linattn checkpoint's own vocabulary
            # (`happy</w>`, `angry</w>`, `good</w>`, `bad</w>`, `love</w>` all present; a naive un-stripped
            # lookup would silently match ZERO of them, the exact BPE-vs-word-level mismatch class
            # `fact_grounding_ids`'s own docstring already names for the fact-boost lever). A word-level
            # checkpoint's vocabulary never contains this marker, so the strip is a no-op there -- one lookup
            # path serves both recurrence families. A content word split across MULTIPLE subwords (e.g. a
            # hypothetical "af" + "raid</w>") still misses -- an honest partial-coverage residual, the same
            # shape as the fact-boost lever's own named limitation, not resolved here.
            lookup = raw[:-4] if raw.endswith("</w>") else raw
            if not lookup or lookup in _AO.STOP or lookup not in _AO.WARRINER:
                continue
            v9, a9 = _AO.WARRINER[lookup]
            if abs(v9 - 5.0) < _AO._STRONG_MARGIN:      # not strongly affective -> ignore (mirrors appraise_text)
                continue
            v9l, _a9l = learned.get(lookup, (v9, a9))
            out[tid] = float(max(-1.0, min(1.0, (v9l - 5.0) / 4.0)))
        hit = out
        _AFFECT_BIAS_CACHE[seed] = hit
        return hit


# ── MARGIN-TO-TOP1, SATURATING scaling (2026-09-04, closes the linattn flip-gate FAIL named by
# `research/findings/raw/_affect_wkv_mouth_verify_phase4_linattn_flip_confirmation.json`: through the real
# `webapp.server.brain_chat`, a genuine +mood (organ differential +0.040, appraisal 'thrilled/overjoyed/
# wonderful') left the linattn mouth's raw output BYTE-IDENTICAL lesion0-vs-lesion1, while the SAME mechanism
# was already load-bearing for ssm). DIAGNOSIS (measured, not assumed -- see the finding this rung shipped
# with for the full numbers + the two rejected intermediate designs):
#   (a) the valence magnitude that actually reaches this function LIVE is neither the raw organ differential
#       (~0.04) nor the raw appraisal (`appraisal_valence`, ~0.475) -- it is `_valence_from_differential(diff)
#       = clip(4*diff, -1, 1)` (`webapp/open_ended_chat.py::valence_from_affect`, `research.runners.
#       _open_ended_state_driven_generation_derisk._valence_from_differential`), i.e. ~0.16 for this scenario.
#       That is ~5.6x smaller than the `valence=+-0.9` sweep the pre-existing `affect_boost=5.0` calibration
#       (see `generate()`'s own docstring) was tuned against -- the fixed boost was calibrated at a magnitude
#       the live pipeline never actually presents.
#   (b) the actual obstacle is NOT generic "sharpness" (full-vocab logit std and the plain top1-vs-top2 gap
#       were measured comparable between the checkpoints, ratio 0.8-1.1x) and NOT merely "is an affect word
#       inside the top-K candidate window" (a first design scaled the bias by the gap to the top-K CUTOFF --
#       rejected: on a templated "tell me about <encyclopedic topic>" continuation the linattn checkpoint's
#       TOP-1 pick routinely out-scores the top-64 cutoff by 4-11 raw logit units, e.g. measured top1=9.3 vs
#       cutoff=5.3 at step 0 of the `frank_lincoln_wright` trajectory -- a candidate can clear the cutoff and
#       still lose to top1 by a wide margin, which is what a realistic-magnitude bias against the cutoff alone
#       measurably failed to move). The quantity that determines whether mood can plausibly change the WORD
#       actually selected is the gap to the CURRENT TOP-1 logit, `margin(t) = top1 - lg[t]` -- what `t` must
#       close to become a genuine rival, not merely a candidate.
#   (c) a second rejected design applied a `gain*wv*deficit`-shaped assist UNIFORMLY to every affect id at
#       once: at the realistic magnitude the assist was correspondingly too small everywhere (no measured
#       diff); scaling it up to compensate caused an unbounded, UNconcentrated pileup across dozens of
#       simultaneously-boosted affect words at the pre-existing calibration's own `valence=+-0.9` sweep
#       (salad_frac up to 0.48 on this prompt, worse than the pre-existing `affect_boost>=8` failure it was
#       supposed to avoid re-creating) -- an absolute, un-saturating multiplier on `margin` has no ceiling, so
#       any boost large enough to matter at realistic magnitude overshoots wildly at the sweep's extreme.
#   (d) even CONCENTRATING that assist onto one candidate (dropping the "every affect id at once" flaw in (c))
#       still cascaded once `boost` was raised enough to make the realistic magnitude cross the margin at all
#       (measured: `boost=15` produced a genuine diff at `valence=0.16`, but salad_frac 0.60 -- WORSE than the
#       `boost=5` extreme-sweep collapse this rung is meant to avoid). Root cause: selecting ONE affect word
#       shifts the CONTEXT the next step conditions on toward affect-adjacent continuations, so later steps
#       need progressively LESS help from this bias to also select an affect word -- an autoregressive
#       positive-feedback cascade, not a per-step calibration problem, so no fixed `boost` alone threads the
#       needle between "never fires" and "runs away".
# THE FIX, three changes together: (1) SATURATE the per-word congruence strength at +-1 -- `strength(t) =
# clip(boost*valence*clip(arousal,0,1)*wv(t), -1, 1)` -- so mood can, AT MOST, close a candidate's ENTIRE
# margin to the current top-1 (full parity, never a forced override past it), regardless of how large `boost`
# is tuned; this is what makes the extreme `valence=+-0.9` sweep safe by construction rather than by finding
# another fragile constant. (2) CONCENTRATE the margin-closing assist on the single mood-CONGRUENT candidate
# already closest to top1 (smallest `margin`) -- spreading it over every affect id at once is what caused (c)'s
# pileup; one contextually-plausible word contending for the win reads as mood coloring word choice, not
# flooding the sequence with disconnected affect vocabulary. A small, UNSATURATED, UNCONCENTRATED floor
# (`strength(t)` alone, capped at +-1 per word) still colors every matched word mildly -- this term alone is
# already bounded, so applying it broadly (as the pre-existing 2026-09-03 formula did) stays safe. (3)
# HABITUATE: scale `strength` by `(1 - recent_affect_frac)`, where `recent_affect_frac` is the fraction of the
# last `_HABIT_WINDOW` generated tokens (see `recent_ids`) that were ALREADY an affect-lexicon word --
# short-term synaptic depression's own shape (Tsodyks & Markram 1997, PNAS 94:719-723: a synapse driven
# repeatedly by the same input transmits progressively less) applied to this decode-time population so a mood
# can tip ONE word choice without the resulting context perpetually re-triggering itself -- this is what closes
# (d)'s cascade. `recent_ids=None` (every pre-existing call site before this rung) leaves habituation at its
# neutral value (no damping) -- additive, not a behavior change for a caller that does not opt in.
_HABIT_WINDOW = 8


def _apply_affect_bias(lg: np.ndarray, affect_ids, valence: float, arousal: float, boost: float,
                        topk: int = 64, recent_ids=None) -> np.ndarray:
    """Additive decode-time logit bias toward mood-congruent in-vocab affect-bearing words, saturating,
    margin-to-top1-aware, and habituating -- see the module comment block directly above this function for the
    full mechanism + the two rejected intermediate designs. Returns `lg` UNCHANGED (same object, no allocation)
    when `affect_ids` is empty, OR `boost == 0.0`, OR `valence == 0.0` -- the last condition is what makes
    `BRAIN_AFFECT_LESION=1` (which clamps the real organ's differential, hence the mapped valence, to exactly
    0.0) collapse this coupling to an EXACT no-op regardless of the host-appraised arousal that turn (arousal
    alone supplies no direction, see the mechanism comment above). `topk` is accepted for call-shape continuity
    with callers that also apply a top-`topk` candidate cut right after this function returns, but this
    function's own scaling no longer depends on it (see (b) in the mechanism comment for why the top-K cutoff
    was the wrong reference point) -- unused, kept so neither call site needs touching if a future design
    reintroduces it. `recent_ids` (default None): the driving loop's own `gen` list-so-far (prompt ids +
    generated ids); when given, the last `_HABIT_WINDOW` entries are checked against `affect_ids` to compute a
    habituation multiplier -- `None` (or an empty/short history) leaves it at 1.0 (no damping).

    THREE TERMS: (1) a broad, SATURATING floor -- `strength(t) = clip(boost*valence*arousal*wv(t), -1, 1)`
    added to every affect id, capped at +-1 raw logit unit per word regardless of `boost`; (2) a CONCENTRATED
    assist -- `strength(best) * margin(best)` where `best` is the single mood-congruent affect id already
    closest to the current step's top-1 logit and `margin(best) = top1 - lg[best]` -- added ONLY to that one
    candidate, so at `strength==+-1` (full saturation) it exactly ties top1 (parity, never an override), and at
    partial saturation it closes only that fraction of the gap; (3) HABITUATION -- both (1) and (2) are scaled
    by `(1 - recent_affect_frac)` BEFORE the saturation clip, damping the coupling once recent output is already
    affect-saturated (breaks the autoregressive cascade named in the mechanism comment's (d), without capping
    how strongly an ISOLATED, in-context nudge can act when the recent context is still neutral). `boost` sets
    how much realistic-magnitude valence*arousal (~0.1, see (a) above) it takes to approach saturation; it can
    no longer cause unbounded overshoot at the sweep's `valence=+-0.9` extreme, because `strength` is clipped
    before it ever multiplies `margin`."""
    if not affect_ids or boost == 0.0 or valence == 0.0:
        return lg
    habit = 1.0
    if recent_ids:
        recent = recent_ids[-_HABIT_WINDOW:]
        if recent:
            habit = 1.0 - (sum(1 for t in recent if t in affect_ids) / len(recent))
    gain = float(boost) * float(valence) * max(0.0, min(1.0, float(arousal))) * habit
    if gain == 0.0:
        return lg
    ids = np.fromiter(affect_ids.keys(), dtype=np.intp, count=len(affect_ids))
    wv = np.fromiter((affect_ids[i] for i in affect_ids), dtype=np.float64, count=len(affect_ids))
    strength = np.clip(gain * wv, -1.0, 1.0)              # saturating: mood can tip a close call, never force
    orig = lg[ids]                                        # pre-bias reference for the margin measurement
    out = lg.copy()
    out[ids] += strength                                  # (1) broad, saturating floor -- +-1 logit unit max
    congruent = strength > 0.0                            # only a word AGREEING in sign with this turn's mood
    if recent_ids:
        # the CONCENTRATED assist (below) must not repeatedly re-select the SAME just-used affect word -- the
        # pre-existing `_apply_repetition_controls` runs BEFORE this function and already damped that word's
        # logit, but a large concentrated assist can re-inflate it past that penalty every step (measured: this
        # is what produced literal "love love love" 3-in-a-row at the START of a sequence, before habituation
        # (3) has enough history to have damped anything -- no earlier occurrence exists yet for the no-repeat-
        # ngram guard to have banned). Excluding the last `_HABIT_WINDOW` tokens from the CONCENTRATED pool
        # (not the broad floor, which stays a small, harmless per-word nudge either way) forces each assisted
        # pick to be a genuinely DIFFERENT affect word -- thematically consistent, lexically varied.
        just_used = set(recent_ids[-_HABIT_WINDOW:])
        if just_used:
            congruent = congruent & ~np.isin(ids, np.fromiter(just_used, dtype=np.intp, count=len(just_used)))
    if congruent.any():
        top1 = float(lg.max())
        cong_ids = ids[congruent]
        cong_orig = orig[congruent]
        cong_strength = strength[congruent]
        best_local = int(np.argmax(cong_orig))            # smallest margin-to-top1 among congruent candidates
        best_id = int(cong_ids[best_local])
        margin = max(0.0, top1 - cong_orig[best_local])   # what full parity with the current top-1 would need
        out[best_id] += cong_strength[best_local] * margin      # (2) concentrated, saturating assist
    return out


# ── generation (reuses WKVReadout + FewSpikeWordRead verbatim; only the driving loop is new) ───────────────────────
def _free_gen(ro, vocab_ids_by_word, reader, prompt: str, seed: int, max_new_tokens: int, topk: int, gen_temp: float,
              repetition_penalty: float = 1.0, no_repeat_ngram_size: int = 0,
              fact_boost_ids=None, fact_boost: float = 0.0,
              affect_ids=None, valence: float = 0.0, arousal: float = 0.0, affect_boost: float = 0.0, bpe=None):
    """`bpe` (default `None`, added 2026-09-03): a `sim.bpe_tokenizer.BPETokenizer` instance (see
    `_get_bpe_tokenizer`) for BPE-vocabulary checkpoints. `bpe=None` (the default, and every pre-existing call
    site) runs the EXACT SAME two lines that existed before this parameter did -- the word-level prompt-id
    lookup and the final space-joined decode -- so the default path is byte-identical BY CONSTRUCTION, not
    merely by equivalent behavior. When `bpe` is given, both the prompt encode and the final decode instead go
    through that tokenizer's own `.encode`/`.decode` (Sennrich-BPE subword id space, matching what produced the
    checkpoint's `words` vocabulary in the first place -- see the module's BPE-mode block above). The prompt
    encode goes through `_bpe_encode_prompt` (2026-09-04 BPE-caps fix -- see that function's own docstring and
    the module comment block above `bpe_lowercase_enabled`), which lowercases by default before calling
    `bpe.encode`."""
    if bpe is not None:
        pid = _bpe_encode_prompt(bpe, prompt)
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
        # affect bias (2026-09-03, closes the affect-hollow gap -- see the module comment block above
        # `_apply_affect_bias`): applied AFTER fact-boost/repetition, same full-vocab logits, same "additive,
        # cannot un-ban a repetition-suppressed token" composition rule. No-op when `affect_ids` is empty,
        # `affect_boost == 0.0`, or `valence == 0.0` (the lesioned-organ / neutral-mood case).
        lg = _apply_affect_bias(lg, affect_ids, valence, arousal, affect_boost, topk=topk, recent_ids=gen)
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


# ── linattn generation (2026-09-03, additive twin of `_free_gen` for `LinAttnReadout` -- see `recurrence_mode`'s
# docstring for why this is a SEPARATE function rather than a branch inside `_free_gen`). Every decode control
# below is copied VERBATIM from `_free_gen` (top-K cut, `_apply_repetition_controls`, `_apply_fact_boost`, the
# genuine few-spike `reader.read(p)` Izhikevich soft-WTA) -- ONLY the state object and the two calls that produce
# it/read it differ (`state = ro.advance(state, t)` / `ro.logits(state, gen[-1])`, vs `WKVReadout`'s `(ap, an)`
# array-pair calls). `_free_gen` itself is completely untouched by this function's existence.
def _free_gen_linattn(ro, vocab_ids_by_word, reader, prompt: str, seed: int, max_new_tokens: int, topk: int,
                       gen_temp: float, repetition_penalty: float = 1.0, no_repeat_ngram_size: int = 0,
                       fact_boost_ids=None, fact_boost: float = 0.0,
                       affect_ids=None, valence: float = 0.0, arousal: float = 0.0, affect_boost: float = 0.0,
                       bpe=None):
    """`LinAttnReadout`'s own driving loop -- see the module docstring block above `recurrence_mode` for why this
    exists as a twin of `_free_gen` rather than a generalization of it. `bpe`/prompt-encode/final-decode, and the
    affect-bias coupling (2026-09-03, see `_apply_affect_bias`), mirror `_free_gen`'s own handling exactly (see
    that function's docstring) -- this is what closes the affect-hollow gap for BOTH recurrence families from
    ONE shared mechanism, not just the `ssm` default. The prompt encode goes through `_bpe_encode_prompt`
    (2026-09-04 BPE-caps fix), same as `_free_gen` -- see that function's docstring."""
    if bpe is not None:
        pid = _bpe_encode_prompt(bpe, prompt)
    else:
        pid = [vocab_ids_by_word[w] for w in (t.lower() for t in _WORD_RE.findall(prompt or "")) if w in vocab_ids_by_word]
    if not pid:
        pid = [0]
    state = ro.init_state()
    for t in pid:
        state = ro.advance(state, t)
    gen = list(pid)
    self_nll = 0.0
    steps = 0
    for _ in range(max_new_tokens):
        lg = ro.logits(state, gen[-1])
        if ro.unk_idx >= 0:
            lg = lg.copy()
            lg[ro.unk_idx] = -1e30
        lg = _apply_repetition_controls(lg, gen, repetition_penalty, no_repeat_ngram_size)
        lg = _apply_fact_boost(lg, fact_boost_ids, fact_boost)
        lg = _apply_affect_bias(lg, affect_ids, valence, arousal, affect_boost, topk=topk, recent_ids=gen)
        cand = np.argpartition(-lg, topk - 1)[:topk]
        cand = cand[np.argsort(-lg[cand])]
        p = _softmax(lg[cand] / gen_temp)
        win, _, _ = reader.read(p)
        nxt = int(cand[win]) if win >= 0 else int(cand[0])
        pfull = _softmax(lg)
        self_nll += -float(np.log(max(pfull[nxt], 1e-12)))
        steps += 1
        gen.append(nxt)
        state = ro.advance(state, nxt)
        if ro.words[nxt] == "endoftext":
            break
    if bpe is not None:
        kept = [i for i in gen if 0 <= i < len(ro.words) and ro.words[i] != "endoftext"]
        text = bpe.decode(kept)
    else:
        text = " ".join(ro.words[i] if 0 <= i < len(ro.words) else "<unk>" for i in gen if ro.words[i] != "endoftext")
    return text, (self_nll / max(1, steps)), gen


# ── BPE-CAPS FIX, OUTPUT half (2026-09-04, same finding as `_bpe_encode_prompt` above -- readability, not the
# perplexity-recovery mechanism). BOTH checkpoint families ship an ENTIRELY lowercase vocabulary -- verified
# directly by loading the checkpoints: 0/1000 `wkv_ssmU6_v1000_d128_seed42.npz` word entries and the BPE
# `wkv_bpe8k.json` merge table's symbols alike contain a single uppercase character, because BOTH training
# corpora (TinyStories via `_train_bpe_bounded`'s `raw.lower()`; the pre-lowercased-on-disk `data/corpus/
# simplewiki.txt`) were lowercased before training. The mouth therefore CANNOT emit a capital letter --
# structurally, not as a decode-time failure -- regardless of `tokenizer_mode()`/`recurrence_mode()`. This is a
# lightweight, EXPLICITLY-TRACKED host articulation scaffold restoring readable casing on the way OUT, the same
# category the task that added it named as precedent: `_apply_affect_bias`'s host decode-time arithmetic over an
# already-neurally-sourced signal. It is pure string post-processing on the ALREADY-CHOSEN word sequence --
# never touches token selection or the genuine few-spike spiking read (`reader.read(p)`) -- and it is NOT a
# claim of general proper-noun/NER capability; see `_KNOWN_CAPITALIZED_WORDS`'s own comment for that scope.
_TRUECASE_ENV = "BRAIN_WKV_MOUTH_TRUECASE"


def truecase_enabled() -> bool:
    """Default-ON (this closes a named readability gap -- the checkpoint's structural inability to emit a
    capital letter, see the module comment block above -- not an opt-in trial). `BRAIN_WKV_MOUTH_TRUECASE` in
    {0,false,no,off} -> an explicit OFF, `generate()`'s return value stays the checkpoint's raw all-lowercase
    text, BYTE-IDENTICAL to before this function existed."""
    v = os.environ.get(_TRUECASE_ENV)
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


# A SMALL, EXPLICIT allowlist -- NOT a general NER/proper-noun detector. Hand-curated directly from the V=1000
# `wkv_ssmU6_*` checkpoint's own vocabulary (dumped and read by hand), keeping only names/titles with negligible
# collision risk against an ordinary English word. Deliberately EXCLUDES several genuine TinyStories character-
# name candidates that ALSO sit in the same vocabulary as ordinary words -- "will", "rose", "mark", "hope",
# "joy", "grace" -- because capitalizing those on sight would be WRONG most of the time (a modal verb, a past
# tense of "rise", ordinary nouns); left out on purpose, an honest bounded residual rather than an oversight.
# Matched case-insensitively per whitespace token, `'s`-suffix-aware (`_truecase_word` below) -- e.g. "tom's" ->
# "Tom's". This list is TinyStories-specific and will rarely fire on the very different BPE/simplewiki
# checkpoint family's vocabulary; a cross-domain gazetteer (e.g. sourced from the live `wikidata_core_15k`
# store's own agent names) was considered and NOT built here -- named as the natural next step in the finding's
# honest-residuals section, not attempted in this lightweight pass.
_KNOWN_CAPITALIZED_WORDS = frozenset("""
tim timmy tom tommy max sue sam ben mia lucy bob bobo bobby sara sarah anna amy jack jane
daisy john joe lila billy molly remy bella leo emma ella mimi polly lisa jen jenny momo
buddy spot mr mrs dr
""".split())

_SENT_BOUNDARY_RE = re.compile(r"([.!?]\s+)")
_I_WORD_RE = re.compile(r"\bi\b")


def _cap_first_alpha(s: str) -> str:
    """Uppercase the first alphabetic character in `s`, leaving any leading whitespace/punctuation untouched.
    A no-op (returns `s` unchanged) when `s` has no alphabetic character at all."""
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1:]
    return s


def _truecase_word(w: str) -> str:
    """Capitalize `w` in place IF its `'s`-stripped, lowercased form is a known name/title (see
    `_KNOWN_CAPITALIZED_WORDS`) -- returns `w` UNCHANGED otherwise (including when it is already correctly
    cased, e.g. a `render_fact_sentence` proper-noun NP already capitalized by `slug_to_np` -- re-uppercasing an
    already-uppercase first letter is a no-op, so this never fights that path)."""
    core = w[:-2] if w.lower().endswith("'s") else w
    if core and core.lower() in _KNOWN_CAPITALIZED_WORDS:
        return core[:1].upper() + core[1:] + w[len(core):]
    return w


def _truecase(text: str) -> str:
    """The full truecasing pass (see the module comment block above `truecase_enabled`): capitalizes the first
    letter of each sentence (split on '.'/'!'/'?' followed by whitespace -- today's checkpoints structurally
    never emit punctuation at all, per the module comment block, so in practice this fires once, on the whole
    string, but the split is written generally rather than assuming that stays true forever), the standalone
    pronoun "i" (and its contractions "i'm"/"i've"/"i'll"/"i'd" -- the apostrophe is itself a `\\b` word
    boundary, so `\\bi\\b` already matches the bare "i" in all of these), and any token matching
    `_KNOWN_CAPITALIZED_WORDS`. Pure string manipulation, no RNG draw -- safe to call OUTSIDE `_RngIsolation.run`
    (see `generate()`'s own call site, after `_RNG.run` returns) without touching that function's RNG-isolation
    contract. Empty input returned unchanged."""
    if not text:
        return text
    parts = _SENT_BOUNDARY_RE.split(text)
    parts = [_cap_first_alpha(p) if (i % 2 == 0) else p for i, p in enumerate(parts)]
    text = "".join(parts)
    text = _I_WORD_RE.sub("I", text)
    return " ".join(_truecase_word(w) for w in text.split(" "))


def generate(prompt: str, seed: int = 42, max_new_tokens: int = 60, topk: int = 64, read_window: int = 40,
             pop: int = 8, gen_temp: float = 0.8, repetition_penalty: float = 1.0,
             no_repeat_ngram_size: int = 0, facts=None, fact_boost: float = 6.0,
             sentence_facts=None, valence: float = 0.0, arousal: float = 0.0,
             affect_boost: float = 10.0, trace: dict | None = None) -> tuple[str, float]:
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
    of that checkpoint).

    RECURRENCE MODE (`BRAIN_WKV_MOUTH_RECURRENCE`, default 'ssm', see `recurrence_mode`'s docstring): 'ssm' (the
    default) is BYTE-IDENTICAL to before this env var existed -- `_get_readout` builds `WKVReadout` and `_run()`
    drives it through `_free_gen`, unchanged. 'linattn' builds `LinAttnReadout` instead and drives it through
    `_free_gen_linattn` -- a different, mutually-exclusive recurrence family (never silently mixed with 'ssm')
    reading a `--recurrence linattn --save-ssm` checkpoint (point `BRAIN_WKV_MOUTH_CKPT` at it).

    AFFECT (`valence`/`arousal`, default 0.0/0.0, `affect_boost` default 10.0 but INERT while `valence == 0.0`
    -- 2026-09-03, closes the affect-hollow gap `research/findings/2026-09-03-linattn-mouth-live-brain-grounded-
    honest-verification-PARTIAL-affect-gap.md` (ii-c) measured: `_free_gen`/`_free_gen_linattn` took NO affect
    parameter at all before this triple existed, so the real spiking affect organ's live valence/arousal read,
    already assembled by `answer_turn` into `state`/`system`/`user`, never reached this generator). See
    `_apply_affect_bias`'s own docstring for the mechanism (a mood-congruent additive logit bias, saturating and
    margin-to-top1-aware, over a Warriner-gated, DR-2-learned-value word lexicon, gain-scaled by arousal and
    habituated against its own recent output, applied in the SAME decode-control category as `fact_boost`/the
    repetition guards -- the spiking `reader.read(p)` selection itself is untouched). `valence=0.0` (the
    default, and what `BRAIN_AFFECT_LESION=1` clamps the real organ's read to) is an EXACT no-op -- `generate()`
    is byte-identical to before this triple existed. Gated additionally by `wkv_mouth_affect_enabled()`
    (`BRAIN_WKV_MOUTH_AFFECT`, default-ON, an independent kill switch).

    `affect_boost=10.0` is an EMPIRICAL CALIBRATION (2026-09-04, re-measured against the REBUILT mechanism --
    see `_apply_affect_bias`'s own mechanism comment for why the ORIGINAL `5.0`/2026-09-03 calibration, which
    scaled a fixed absolute bias against a `valence=+-0.9` sweep, was replaced rather than reused: that
    magnitude never occurs on the LIVE pipeline, see `webapp.open_ended_chat.valence_from_affect` -- a real
    'thrilled/overjoyed/wonderful' priming turn measured `valence~0.16` live, ~5.6x smaller). `10.0` is
    calibrated at THAT realistic magnitude directly (`valence=0.16, arousal=0.65`, `research/findings/raw/
    _affect_wkv_mouth_verify_phase5_boost_and_prompt_sweep.json`), on the HARDER of the two checkpoint families
    (`linattn`/BPE V=8001 general-vocabulary, whose confidently-templated continuations put the checkpoint's
    top-1 pick 4-11 raw logit units above ANY affect-tagged candidate on a typical "tell me about <topic>"
    prompt -- the `ssm`/TinyStories V=1000 checkpoint moves cleanly at this same value, its continuations
    already lean toward affect vocabulary). Below ~8, one of the two mood directions sometimes failed to
    measurably move the realistic-magnitude output on the tested prompts (undershoot, not a safety concern);
    `10.0` reliably differed from neutral in BOTH mood directions on both prompts tested. Because the
    saturating cap in `_apply_affect_bias` bounds a single word's assist to at most FULL PARITY with the
    current top-1 regardless of `boost`, this value does not reproduce the OLD mechanism's `>=8` word-salad
    collapse at the sweep's `valence=+-0.9` extreme (measured salad-fraction <=0.14 across both families, both
    mood directions, both prompts tested, vs a neutral baseline of ~0.09-0.10 on the same prompts -- see the
    finding). HONEST RESIDUAL: this is ONE constant shared by both recurrence families and not independently
    re-tuned per checkpoint/topic; a production deployment that finds it too weak or too strong on real traffic
    should re-run that sweep rather than assume this value transfers unconditionally.

    `trace` (default `None`, additive -- 2026-09-04, closes the generator-trace mislabel the 2026-09-03 linattn
    live verification found: see research/findings/2026-09-04-generator-trace-mislabel-fix.md): when the caller
    passes a dict, `_run()` records `trace["sentence_fact_used"] = True` right before returning a
    `render_fact_sentence` clause (the `sentence_facts` branch above fired), or `= False` right before falling
    through to genuine `_free_gen`/`_free_gen_linattn` decode -- so the caller can tell WHICH mechanism actually
    produced the returned text, independent of which of `generate()`'s own internal branches reached it. ROOT
    CAUSE this closes: `webapp/open_ended_chat.py::answer_turn` previously inferred the producer purely from
    WHICH CODE PATH called `generate()` (its own WKV-mouth try-block vs the separate fact-clause-fallback
    branch) rather than from what `generate()` itself actually did -- so whenever `sentence_facts` found a
    lexicon-covered relation INSIDE this call (the common case once `BRAIN_WKV_MOUTH_SCOPE=broad` routes nearly
    every prompt through the WKV-mouth try-block first, since the outer fact-clause-fallback branch is then
    never reached), the reply was genuinely written by `render_fact_sentence` but the caller labelled it
    `generator="wkv_mouth"` regardless. `trace=None` (the default and every pre-existing call site) is an exact
    no-op -- both `if trace is not None:` guards below never fire, so `generate()`'s behavior and return value
    are BYTE-IDENTICAL to before this parameter existed.

    BPE-CAPS FIX (2026-09-04, research/findings/2026-09-03-linattn-mouth-broad-scope-coverage-threshold.md):
    two independent, independently-guarded pieces. INPUT -- `bpe_lowercase_enabled()` (default ON): when `bpe`
    mode is active, `_free_gen`/`_free_gen_linattn` lowercase the prompt before BPE-encoding it (via
    `_bpe_encode_prompt`), recovering a measured ~5.6x teacher-forced-perplexity hit the raw-case encode paid
    (every capital letter fell outside the checkpoint's trained, lowercase-only character alphabet and
    BPE-encoded to `<UNK>`). Inert whenever `tokenizer_mode() != 'bpe'` (the shipped default). OUTPUT --
    `truecase_enabled()` (default ON): both checkpoint families ship an entirely lowercase vocabulary and so
    structurally cannot emit a capital letter at all; `_truecase` (a lightweight sentence-initial + pronoun-"I"
    + small known-name-allowlist heuristic, NOT retrained, NOT a general NER) is applied to this function's
    final return text regardless of which internal path produced it (free-gen or `render_fact_sentence`).
    BOTH default ON but are independently `BRAIN_WKV_MOUTH_BPE_LOWERCASE=0` / `BRAIN_WKV_MOUTH_TRUECASE=0`
    revertible to the exact pre-fix behavior (byte-identical text) -- see each flag function's own docstring."""
    t0 = time.time()

    def _run():
        if sentence_facts:
            sentence = render_fact_sentence(sentence_facts, seed=seed)
            if sentence is not None:
                if trace is not None:
                    trace["sentence_fact_used"] = True
                return sentence
        if trace is not None:
            trace["sentence_fact_used"] = False
        ro, _vocab, word_to_id = _get_readout(seed)
        reader = FewSpikeWordRead(topk, pop, seed, read_window=read_window)
        fact_ids = fact_grounding_ids(facts, seed=seed) if facts else None
        aff_ids = _affect_bias_ids(seed) if (valence != 0.0 and wkv_mouth_affect_enabled()) else None
        bpe = _get_bpe_tokenizer() if tokenizer_mode() == "bpe" else None
        gen_fn = _free_gen_linattn if recurrence_mode() == "linattn" else _free_gen
        text, _self_nll, _gen = gen_fn(ro, word_to_id, reader, prompt, seed, max_new_tokens, topk, gen_temp,
                                       repetition_penalty, no_repeat_ngram_size,
                                       fact_boost_ids=fact_ids, fact_boost=fact_boost,
                                       affect_ids=aff_ids, valence=valence, arousal=arousal,
                                       affect_boost=affect_boost, bpe=bpe)
        return text

    text = _RNG.run(seed, _run)
    if truecase_enabled():
        text = _truecase(text)
    return text, round(time.time() - t0, 3)
