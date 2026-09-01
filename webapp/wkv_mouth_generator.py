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

# ── opt-in e-prop LEARNED read-out head (honest residual #1, see the module docstring). Default OFF: the native
# checkpoint head.weight is used, byte-identical to before this flag existed. `_wkv_mouth_readout_eprop_batched_
# substrate_derisk.py --save-w-hat <path>` is the ONLY producer of this file's shape/basis; see `_apply_learned_
# head` for the compatibility check performed before it is ever substituted in.
_LEARNED_HEAD_ENV = "BRAIN_WKV_MOUTH_LEARNED_HEAD"
_LEARNED_HEAD_PATH_TEMPLATE = os.environ.get(
    "BRAIN_WKV_MOUTH_LEARNED_HEAD_PATH",
    # Default points at the 6/6-GO persisted heads (ratio 0.9273, min 0.8906; finding
    # 2026-08-28-mouth-better-head-persist-6seed-GO-plus-wander-production-partial). Still behind the
    # default-OFF BRAIN_WKV_MOUTH_LEARNED_HEAD flag; override with BRAIN_WKV_MOUTH_LEARNED_HEAD_PATH.
    str(_REPO_ROOT / "research/findings/raw/_persist_eprop_head_scope/wkv_eprop_learned_head_0p94_s{seed}.npz"),
)


def learned_head_enabled() -> bool:
    return os.environ.get(_LEARNED_HEAD_ENV, "0").strip().lower() in ("1", "true", "on", "yes")


def _learned_head_path(seed: int) -> str:
    if "{seed}" in _LEARNED_HEAD_PATH_TEMPLATE:
        return _LEARNED_HEAD_PATH_TEMPLATE.format(seed=seed)
    return _LEARNED_HEAD_PATH_TEMPLATE


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
    dominated by. The caller (`answer_turn`) MUST fall back to Qwen when this is False -- this module never
    attempts an out-of-scope prompt itself. Three conditions, all required: (1) `min_hits` total vocab-word
    matches (guards very short prompts), (2) `min_frac` overall word-overlap fraction, (3) `min_content_hits`
    matches that are NOT in `_FUNCTION_WORDS` -- closes the stopword-only / function-word-dominated false-positive
    an adversarial verify-go pass found in the first version of this gate (2026-08-28)."""
    _, vocab, _ = _get_readout(seed)
    words = [w.lower() for w in _WORD_RE.findall(text or "")]
    if not words:
        return False
    hits = [w for w in words if w in vocab]
    content_hits = [w for w in hits if w not in _FUNCTION_WORDS]
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
              fact_boost_ids=None, fact_boost: float = 0.0):
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
    text = " ".join(ro.words[i] if 0 <= i < len(ro.words) else "<unk>" for i in gen if ro.words[i] != "endoftext")
    return text, (self_nll / max(1, steps)), gen


def generate(prompt: str, seed: int = 42, max_new_tokens: int = 60, topk: int = 64, read_window: int = 40,
             pop: int = 8, gen_temp: float = 0.8, repetition_penalty: float = 1.0,
             no_repeat_ngram_size: int = 0, facts=None, fact_boost: float = 6.0) -> tuple[str, float]:
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
    `facts` is None (the pre-existing default and every pre-existing call site)."""
    t0 = time.time()

    def _run():
        ro, _vocab, word_to_id = _get_readout(seed)
        reader = FewSpikeWordRead(topk, pop, seed, read_window=read_window)
        fact_ids = fact_grounding_ids(facts, seed=seed) if facts else None
        text, _self_nll, _gen = _free_gen(ro, word_to_id, reader, prompt, seed, max_new_tokens, topk, gen_temp,
                                          repetition_penalty, no_repeat_ngram_size,
                                          fact_boost_ids=fact_ids, fact_boost=fact_boost)
        return text

    text = _RNG.run(seed, _run)
    return text, round(time.time() - t0, 3)
