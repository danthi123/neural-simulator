"""EMERGE-68 -- extend the VALIDATED SPIKING A->W (concept-pool -> spoken-word) read-out to the FUNCTION-WORD
(DET/FUNC) slots so the EMERGE-frame render becomes 100% SPIKING -- ORDER *and* ALL WORDS (content AND function).

EMERGE-67 wired the CONTENT slots (subject/verb = emergent concepts) of the EMERGE-frame render onto spikes: the
`spell` callback DRIVES the word's concept pool on a real `SimulationBridge` and DECODES the spoken word from
`cp_firing_states[language_output]` (the validated `concept_speak_demo` A->W read-out, CLAUDE.md "chat_speak A->W 100%
multi-seed"). The residual EMERGE-67 named (its GO finding line 23; the "named function-word follow-on = EMERGE-68"):
the DET/FUNC (closed-class function-word) slots -- `the`/`a`/`can`/`does`/`not` -- kept a TOKEN surface (`spell`
returned `str(word)` for words not in the trained content vocab, `_emerge67:236`). THOSE are the residual this closes.

THE FRAME FUNCTION WORDS (the exact residual). `_emerge59_spiking_broca_frame_slots_derisk.py:98-105`:
  F_MODAL  [(DET,"the"), SUBJ, (FUNC,"can"),  (VERB,"bare")]   -> "the owl can fly"
  F_INTR   [(DET,"the"), SUBJ, (VERB,"3sg")]                    -> "the penguin walks"
  F_NEGMOD [(DET,"the"), SUBJ, (FUNC,"does"), (FUNC,"not"), (VERB,"bare")] -> "the penguin does not fly"
So the FUNCTION words the frames emit are {the, can, does, not}; the task names {the, a, can, does, not} (the DET
alternative `a` from `argstructure_composer.FUNCTION_WORDS`, so the DET slot can spell either article on spikes). These
are EXACTLY the closed class EMERGE-62 DISCOVERS from distributional statistics (`_emerge62`:6-8, the discovered set is
the hand `{the,a,can,does,not}` complement + validated). EMERGE-68 makes their A->W SPIKING.

THE SEAM (already function-word-ready). `_emerge59:138 realize_slot(slot, subject, verb, spell)`: the DET/FUNC branch
ALREADY calls `spell(payload)` (`_emerge59:143-144`). So a `NeuralSpell` that KNOWS the function words (decodes them
from spikes) makes the DET/FUNC slots spiking AUTOMATICALLY -- no seam change. EMERGE-68 extends the A->W engine's
word->pool set to cover the 5 function words alongside EMERGE-67's 16 content words, so `neural_spell("the"/"can"/
"does"/"not"/"a")` decodes them from `language_output` spikes too.

THE ENGINE (reuse EMERGE-67's machinery; two co-validated A->W bridges). `concept_speak_demo`'s validated A->W is a
16-pool concept-pool bridge (4 kinds x 4 words: motor/noun/verb/adjective) rebound to a target vocab (EMERGE-67 rebound
the 16 CONTENT words). The producer needs 16 content + 5 function = 21 words. The concept-pool architecture has EXACTLY
16 pools (4 kinds x 4) and `train_word_to_pool` supports ONLY those 4 kinds -- so a 21st word needs a SECOND bridge (the
project's own scaling route: the EMERGE-67 finding named "extend the content vocab ... across 2 bridges"; G.20
multi-bridge, CLAUDE.md). EMERGE-68 therefore builds:
  * BRIDGE-A (the EMERGE-67 cache `bridges/emerge67_aw/aw_content.simstate.h5`): the 16 CONTENT words. REUSED verbatim.
  * BRIDGE-F (the new EMERGE-68 cache `bridges/emerge68_aw/aw_func.simstate.h5`): the 5 FUNCTION words rebound onto 5 of
    the 16 pools (the/a/can/does/not -> motor_N/E/S/W + noun_APPLE), the other 11 pools filled with CONTENT-word fillers
    so the SAME 16-pool builder + topographic-bias + orthogonal-codes + train_word_to_pool recipe trains at the validated
    scale; only the 5 function pools are DECODED. (A vocab rebind on a second bridge -- reuse-by-import; NO `sim/` edit.)
`NeuralSpell68.spell(word)` DISPATCHES: a CONTENT word -> decode on BRIDGE-A (== EMERGE-67); a FUNCTION word -> decode on
BRIDGE-F. Both read `cp_firing_states[language_output]` -- genuinely spiking. `realize_slot`'s DET/FUNC branch routes
through this SAME `spell`, so every slot -- content AND function -- is spike-spelled.

DE-RISK (6 seeds 42/43/44/100/101/102; A->W on GPU/cupy, numpy-skip fallback; the engines are GPU-trained ONCE + cached,
deterministic per their own seed, while the derisk seed varies the frame/fact selection):
  (a) ALL WORDS of the EMERGE frames spelled ON SPIKES: "the owl can fly" -> the/owl/can/fly ALL decoded from
      language_output spikes (content on BRIDGE-A, function on BRIDGE-F). surface-accuracy of ALL slots (det+subj+
      func+verb) vs ground-truth == 100%.
  (b) GENUINELY SPIKING for the function words too: the FUNCTION-word LESION control (zero the pool->language_output
      pathway on BRIDGE-F) COLLAPSES the function-word decode (a host lookup would be unaffected).
  (c) gate-first MOAT: on ABSTAIN the producer -- and hence BOTH A->W read-outs -- is NEVER invoked (0 spell calls,
      0 productions).
  (d) NO regression vs the token spell: every slot surface (content + function) is IDENTICAL to the token spell (the
      neural spell reproduces the token spell for the trained 21-word vocab); the token-spell default path is
      byte-identical (EMERGE-59..67).
GO bar: ALL-word spike-spell accuracy >= 0.90 (a clear bar), function-word lesion-collapse (>= 0.40 drop), moat 0, no
regression, 6-seed. If a function word does NOT cleanly decode (high-frequency closed-class pools overlap / the A->W
read-out can't separate them) -> honest BOUNDARY naming the exact gap + next step (do NOT force a GO; do NOT weaken the
moat).

HONEST SCOPE: this makes ALL WORDS of the BOUNDED EMERGE frame inventory spike-produced (the ORDER was already spiking,
EMERGE-59/63; the CONTENT words since EMERGE-67; the FUNCTION words now). ==> the EMERGE-frame render is 100% SPIKING
(order AND every word). NOT open prose (R4, the deferred scale wall). The A->W engines are GPU-trained ONCE at the
validated scale + cached (a scale/data lever, not a new mechanism); the vocab rebind is a research-runner edit. Reuse-
by-import; NO `sim/` edit.

Run:
  SIM_BACKEND=cupy python -m research.runners._emerge68_function_word_spell_derisk --train    # build+cache BRIDGE-F
  SIM_BACKEND=cupy python -m research.runners._emerge68_function_word_spell_derisk --demo
  SIM_BACKEND=cupy python -m research.runners._emerge68_function_word_spell_derisk --derisk
  SIM_BACKEND=cupy python -m research.runners._emerge68_function_word_spell_derisk --derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # numpy for the CPU logic; the A->W engines force cupy when built
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Reuse-by-import ONLY -- NO sim/ edit. EMERGE-59 producer seam + EMERGE-67 A->W engine machinery.
from research.runners._emerge59_spiking_broca_frame_slots_derisk import (  # noqa: E402
    FRAMES, FRAME_NAMES, DET, SUBJ, FUNC, VERB, FrameSlotCQ, BrocaProducer, decision_from_emerge,
)
from research.runners._emerge57_ra_refinetune_emerge_frames_derisk import emerge_v3  # noqa: E402
import research.runners._emerge67_neural_spell_wirein_derisk as m67  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_emerge68_function_word_spell.json"
# the cached FUNCTION-word A->W engine (GPU-trained once; regenerable via --train). Kept under bridges/ (h5 gitignored).
_FUNC_CACHE_DIR = _REPO / "bridges" / "emerge68_aw"
_FUNC_CACHE_BRIDGE = _FUNC_CACHE_DIR / "aw_func.simstate.h5"

# ---------------------------------------------------------------------------------------------------------------------
# THE 5 FUNCTION WORDS (the EMERGE-frame DET/FUNC residual: {the,can,does,not} the frames emit + `a` the DET
# alternative), rebound onto 5 of the 16 validated concept pools (motor_N/E/S/W + noun_APPLE -- pool KINDS that
# train_word_to_pool supports). The other 11 pools are filled with CONTENT-word FILLERS so the SAME 16-pool
# builder/bias/train recipe runs at the validated scale; only the 5 function pools are DECODED.
# ---------------------------------------------------------------------------------------------------------------------
_FUNC_WORDS = ["the", "a", "can", "does", "not"]      # the closed-class residual (EMERGE-59 frames + `a`)
# 11 content-word fillers (reuse EMERGE-67's content vocab tail so the 16-pool bias/train recipe is well-separated).
_FUNC_FILLERS = m67._AW_CONTENT[:11]                   # owl..(11th) -- fill the non-function pools
_FUNC_VOCAB16 = _FUNC_WORDS + _FUNC_FILLERS            # 5 function + 11 filler == 16 words == 16 pools

_N_LANG = m67._N_LANG
_N_PER_POOL = m67._N_PER_POOL
_N_FS_PER_POOL = m67._N_FS_PER_POOL
_SPARSITY = m67._SPARSITY
_TRAIN_EVENTS = m67._TRAIN_EVENTS
_AW_SEED = m67._AW_SEED            # the function engine's OWN training seed (deterministic spelling module). Seed 42
_FUNC_SEED = 42                   # matches EMERGE-67's validated content cache (16/16); a per-pool-fragility seed lever
                                  # (the concept-pool architecture has documented per-word seed fragility -- CLAUDE.md).


def _func_pool_assignment():
    """Rebind the 16 FUNCTION+filler words onto the 16 validated concept pools (function words on the first 5 pools:
    motor_N/E/S/W + noun_APPLE). Returns (word_to_pool, pool_to_word, func_pools)."""
    from research.runners.concept_pool_demo import MOTOR_NAMES, NOUN_NAMES, VERB_NAMES, ADJECTIVE_NAMES
    pools = ([f"motor_{a}" for a in MOTOR_NAMES] + [f"noun_pool_{n}" for n in NOUN_NAMES]
             + [f"verb_pool_{v}" for v in VERB_NAMES] + [f"adjective_pool_{n}" for n in ADJECTIVE_NAMES])
    assert len(pools) == len(_FUNC_VOCAB16), (len(pools), len(_FUNC_VOCAB16))
    word_to_pool = {w: p for w, p in zip(_FUNC_VOCAB16, pools)}
    pool_to_word = {p: w for w, p in word_to_pool.items()}
    func_pools = {w: word_to_pool[w] for w in _FUNC_WORDS}   # the 5 function pools (the ones we DECODE)
    return word_to_pool, pool_to_word, func_pools


def _swap_func_vocab():
    """Swap `concept_pool_demo`'s module-level 16-word vocab dicts for the 16 FUNCTION+filler words (function words on
    the motor+noun-APPLE pools), keeping the pool VALUE-slots fixed. Returns word_to_idx over the 16 words (pool order)."""
    import research.runners.concept_pool_demo as cpd
    w = _FUNC_VOCAB16
    cpd.DIRECTION_VOCAB.clear(); cpd.DIRECTION_VOCAB.update({w[0]: "N", w[1]: "E", w[2]: "S", w[3]: "W"})
    cpd.NOUN_VOCAB.clear();      cpd.NOUN_VOCAB.update({w[4]: "APPLE", w[5]: "RIVER", w[6]: "DOG", w[7]: "CAT"})
    cpd.VERB_VOCAB.clear();      cpd.VERB_VOCAB.update({w[8]: "GO", w[9]: "COME", w[10]: "STOP", w[11]: "LOOK"})
    cpd.ADJECTIVE_VOCAB.clear(); cpd.ADJECTIVE_VOCAB.update({w[12]: "BIG", w[13]: "SMALL", w[14]: "HOT", w[15]: "COLD"})
    allw = (list(cpd.DIRECTION_VOCAB) + list(cpd.NOUN_VOCAB) + list(cpd.VERB_VOCAB) + list(cpd.ADJECTIVE_VOCAB))
    return {w2: i for i, w2 in enumerate(allw)}


# ---------------------------------------------------------------------------------------------------------------------
# THE FUNCTION-WORD A->W ENGINE (BRIDGE-F). Same NeuralSpell machinery as EMERGE-67, but the 16-word vocab is the 5
# function words + 11 fillers, cached separately. `spell_func(word)` decodes a function word from BRIDGE-F's spikes.
# ---------------------------------------------------------------------------------------------------------------------
class FuncNeuralSpell:
    """The genuinely-spiking A->W `spell` for the FUNCTION words. Builds/loads the validated concept-pool bridge rebound
    to the 16 FUNCTION+filler words; `spell(word)` drives the function word's pool and decodes the spoken word from
    language_output SPIKES. Only the 5 function words are exposed for spelling (the fillers make the 16-pool recipe run
    at the validated scale)."""

    def __init__(self, seed=_FUNC_SEED, load=True, train_events=_TRAIN_EVENTS, lesion_pool_out=False, read_steps=100):
        self.seed = int(seed)
        self.train_events = int(train_events)
        self.lesion_pool_out = bool(lesion_pool_out)
        self.read_steps = int(read_steps)
        self.spell_calls = 0
        self.func_words = list(_FUNC_WORDS)
        self.vocab16 = list(_FUNC_VOCAB16)
        self.word_to_pool, self.pool_to_word, self.func_pools = _func_pool_assignment()
        self._word_to_idx = _swap_func_vocab()      # rebinds concept_pool_demo's vocab dicts (must precede build)
        self._pats = None
        self.bridge = None
        self._backend_gpu = None
        self._build(load=load)

    def _build(self, load=True):
        os.environ["SIM_BACKEND"] = "cupy"          # the A->W engine is a GPU read-out (the validated scale)
        from sim.backend import get_backend, is_gpu_backend
        _xp, _name = get_backend()
        self._backend_gpu = bool(is_gpu_backend())
        from sim.text_embeddings import orthogonal_drive_pattern
        from research.runners.concept_pool_demo import (build_concept_bridge, apply_concept_topographic_bias,
                                                        train_word_to_pool)
        V = len(self.vocab16)
        # language_output reference patterns (cosine targets), one per word (orthogonal bands)
        self._pats = {w: orthogonal_drive_pattern(self._word_to_idx[w], n_cues=V, n_neurons=_N_LANG, sparsity=_SPARSITY)
                      for w in self.vocab16}

        b = build_concept_bridge(seed=self.seed, n_lang_input=_N_LANG, n_per_pool=_N_PER_POOL,
                                 n_fs_per_pool=_N_FS_PER_POOL, weak_dynamics=True, enable_adjective=True, verbose=False)
        if load and _FUNC_CACHE_BRIDGE.exists():
            b.load_checkpoint(str(_FUNC_CACHE_BRIDGE))
        else:
            apply_concept_topographic_bias(b, n_lang_input=_N_LANG, orthogonal_codes=True,
                                           n_words_for_orthogonal=V, word_to_idx=self._word_to_idx,
                                           sparsity=_SPARSITY, verbose=False)
            for w, pool in self.word_to_pool.items():
                train_word_to_pool(b, w, pool, n_events=self.train_events, n_lang_input=_N_LANG, n_lang_output=_N_LANG,
                                   orthogonal_codes=True, n_words_for_orthogonal=V, word_to_idx=self._word_to_idx,
                                   sparsity=_SPARSITY, verbose=False)
            _FUNC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            b.save_checkpoint(str(_FUNC_CACHE_BRIDGE))
        for g in ("language_input_to_motor", "language_input_to_noun_pool", "language_input_to_verb_pool",
                  "language_input_to_adjective_pool", "motor_to_language_output", "noun_pool_to_language_output",
                  "verb_pool_to_language_output", "adjective_pool_to_language_output"):
            try:
                b.set_plasticity_gate(g, 0.0)
            except Exception:
                pass
        if self.lesion_pool_out:
            self._lesion_pool_to_output(b)
        self.bridge = b

    def _lesion_pool_to_output(self, bridge):
        """Zero every synapse whose POST neuron is in language_output -> the language_output spikes carry no
        word-selective signal -> the function-word A->W decode collapses (a host lookup would be unaffected)."""
        from sim.backend import get_backend
        cp, _ = get_backend()
        rm = bridge.region_manager
        lo = cp.asarray(list(rm.indices("language_output")), dtype=bridge.cp_connections.indices.dtype)
        mask = cp.isin(bridge.cp_connections.indices, lo)
        bridge.cp_connections.data[mask] = 0.0

    def _decode(self, word):
        """Drive `word`'s pool on BRIDGE-F; accumulate language_output spikes; cosine-decode the spoken word (over the
        full 16-word vocab so a function word can be confused with a filler if the read-out cannot separate it)."""
        from research.runners.concept_speak_demo import drive_pool_and_read_lang_output, _cosine
        pool = self.word_to_pool[word]
        patt = drive_pool_and_read_lang_output(self.bridge, pool, stim_steps=self.read_steps, n_lang_output=_N_LANG)
        scores = {w2: _cosine(patt, self._pats[w2]) for w2 in self.vocab16}
        top = max(scores, key=scores.get)
        return top, float(scores[word]), float(scores[top]), int(np.asarray(patt).sum())

    def spell(self, word):
        """Spell a FUNCTION word ON SPIKES via BRIDGE-F. (Non-function words are not this engine's job -> token surface.)"""
        self.spell_calls += 1
        if word in self.func_pools:
            decoded, _sc, _tc, _n = self._decode(word)
            return decoded
        return str(word)


# ---------------------------------------------------------------------------------------------------------------------
# THE UNIFIED 21-WORD A->W SPELL. Dispatches: CONTENT word -> BRIDGE-A (EMERGE-67's NeuralSpell); FUNCTION word ->
# BRIDGE-F (FuncNeuralSpell). Both decode from language_output SPIKES. `realize_slot`'s DET/FUNC branch routes through
# this SAME `spell`, so EVERY slot -- content AND function -- is spike-spelled.
# ---------------------------------------------------------------------------------------------------------------------
class UnifiedNeuralSpell:
    """The genuinely-spiking A->W `spell` over the full 21-word producer vocab (16 content + 5 function). `spell(word)`
    dispatches to the CONTENT engine (BRIDGE-A) or the FUNCTION engine (BRIDGE-F) and returns the SPIKE-DECODED
    surface. A word in neither trained vocab -> token surface (no such word occurs in the EMERGE frames)."""

    def __init__(self, load=True, train_events=_TRAIN_EVENTS, content_lesion=False, func_lesion=False, read_steps=100):
        self.spell_calls = 0
        self.content = m67.NeuralSpell(load=load, train_events=train_events, lesion_pool_out=content_lesion,
                                       read_steps=read_steps)
        self.func = FuncNeuralSpell(load=load, train_events=train_events, lesion_pool_out=func_lesion,
                                    read_steps=read_steps)
        self._backend_gpu = bool(self.content._backend_gpu and self.func._backend_gpu)
        # the words each engine can spell on spikes
        self.content_words = set(self.content.word_to_pool)
        self.func_words = set(self.func.func_pools)

    def spell(self, word):
        self.spell_calls += 1
        if word in self.content_words:
            return self.content._decode(word)[0]    # spike-decode on BRIDGE-A (== EMERGE-67 content engine)
        if word in self.func_words:
            return self.func._decode(word)[0]        # spike-decode on BRIDGE-F (the function engine)
        return str(word)                            # not in either vocab -> token surface (no EMERGE-frame word)


# ---------------------------------------------------------------------------------------------------------------------
# ALL-SLOT SPELL SCORING. Every slot (DET + SUBJ + FUNC + VERB) is spelled by the unified neural spell + compared to the
# ground-truth surface. This is the WHOLE frame on spikes (content AND function), unlike EMERGE-67's content-only score.
# ---------------------------------------------------------------------------------------------------------------------
def _all_slot_surfaces(frame, subject, verb):
    """The ground-truth surface for EVERY slot (det + subj + func + verb) a frame emits for a fact."""
    out = []
    for stype, payload in FRAMES[frame]:
        if stype in (DET, FUNC):
            out.append(payload)                     # the/a/can/does/not
        elif stype == SUBJ:
            out.append(subject)
        elif stype == VERB:
            out.append(verb if payload == "bare" else emerge_v3(verb, already_3sg=None))
    return out


def _facts(seed, n=8):
    """Held-out (subject, ability_verb, intr_verb) facts drawn from the A->W CONTENT vocab so every content slot is
    spike-spellable (the function slots are fixed frame furniture)."""
    return m67._facts_from_content_vocab(seed, n=n)


def _render_and_score_all(cq, spell, facts, frames=FRAME_NAMES):
    """Render every fact in every frame with the unified NEURAL spell; score ALL slots (det+subj+func+verb) vs the
    ground-truth surface. Returns (all_slot_accuracy, function_slot_accuracy, examples)."""
    all_hits = all_tot = func_hits = func_tot = 0
    examples = []
    for frame in frames:
        for fact in facts:
            verb = fact["intr_verb"] if frame == "F_INTR" else fact["ability_verb"]
            words = cq.emit(frame, fact["subject"], verb, spell)      # ALL slots spelled via the unified neural spell
            expect = _all_slot_surfaces(frame, fact["subject"], verb)
            # order-agnostic multiset comparison of the produced surfaces vs ground-truth (the ORDER is separately
            # validated by EMERGE-59/63; here we check the WORDS -- content AND function -- are the right ones)
            prod_ms = sorted(words)
            exp_ms = sorted(expect)
            for w in exp_ms:
                all_tot += 1
                # count each expected surface once (multiset membership)
                if w in prod_ms:
                    prod_ms.remove(w)
                    all_hits += 1
            # function-word-slot accuracy specifically (the/a/can/does/not)
            func_expect = [p for (t, p) in FRAMES[frame] if t in (DET, FUNC)]
            prod2 = sorted(words)
            for w in func_expect:
                func_tot += 1
                if w in prod2:
                    prod2.remove(w)
                    func_hits += 1
            if len(examples) < 6:
                examples.append({"frame": frame, "fact": {"subject": fact["subject"], "verb": verb},
                                 "surface": " ".join(words), "expected": " ".join(expect)})
    all_acc = float(all_hits / max(1, all_tot))
    func_acc = float(func_hits / max(1, func_tot))
    return all_acc, func_acc, examples


# ---------------------------------------------------------------------------------------------------------------------
# PER-WORD A->W ACCURACY over the FUNCTION words (the function engine's own spike-spell rate) + the content engine's.
# ---------------------------------------------------------------------------------------------------------------------
def _func_wordwise_accuracy(func_speller):
    """Drive each FUNCTION word's pool on BRIDGE-F; decode; how many decode to the correct function word."""
    n_ok = 0
    per = []
    for w in func_speller.func_words:
        dec, self_cos, top_cos, spikes = func_speller._decode(w)
        ok = (dec == w)
        n_ok += int(ok)
        per.append({"word": w, "decoded": dec, "ok": bool(ok), "self_cos": round(self_cos, 3),
                    "top_cos": round(top_cos, 3), "spikes": spikes})
    return float(n_ok / len(func_speller.func_words)), per


# ---------------------------------------------------------------------------------------------------------------------
# THE DE-RISK: all-word spike-spell accuracy + function-word lesion collapse + moat + no-regression-vs-token.
# ---------------------------------------------------------------------------------------------------------------------
def _derisk_one(seed, unified, unified_func_lesion=None):
    facts = _facts(seed, n=8)
    cq = FrameSlotCQ(seed=seed)
    cq.learn()

    # (a) MAIN: render the EMERGE frames with ALL words (content + function) spelled ON SPIKES.
    all_acc, func_acc, examples = _render_and_score_all(cq, unified.spell, facts)

    # (d) NO REGRESSION vs the token spell: the produced surfaces (content + function) as a multiset must equal the
    # token spell's surfaces (== ground-truth by construction) for every frame+fact.
    token_spell = lambda w: str(w)
    regress_mismatch = 0
    for frame in FRAME_NAMES:
        for fact in facts:
            verb = fact["intr_verb"] if frame == "F_INTR" else fact["ability_verb"]
            tok = sorted(cq.emit(frame, fact["subject"], verb, token_spell))
            neu = sorted(cq.emit(frame, fact["subject"], verb, unified.spell))
            regress_mismatch += int(tok != neu)

    # (c) MOAT: an ABSTAIN never invokes the producer -> neither A->W engine's spell is called.
    prod = BrocaProducer(cq, spell=unified.spell)
    calls_before = unified.spell_calls
    for _ in range(3):
        prod.speak(decision_from_emerge("ABSTAIN"))
    spell_calls_on_abstain = unified.spell_calls - calls_before
    producer_calls_on_abstain = prod.production_count
    ans = prod.speak(decision_from_emerge("ANSWER", subject="owl", verb="fly", polarity="affirm"))
    answer_produced = bool(ans["produced"])

    # (b) FUNCTION-WORD LESION: the SAME frames rendered with BRIDGE-F lesioned must collapse the FUNCTION-word decode.
    lesion_func_acc = None
    if unified_func_lesion is not None:
        _all2, lesion_func_acc, _ = _render_and_score_all(cq, unified_func_lesion.spell, facts)

    return {
        "seed": seed,
        "all_acc": all_acc,
        "func_acc": func_acc,
        "regress_mismatch": int(regress_mismatch),
        "spell_calls_on_abstain": int(spell_calls_on_abstain),
        "producer_calls_on_abstain": int(producer_calls_on_abstain),
        "answer_produced": answer_produced,
        "lesion_func_acc": lesion_func_acc,
        "examples": examples,
    }


def _sample_transcript(unified, seed=42):
    cq = FrameSlotCQ(seed=seed)
    cq.learn()
    prod = BrocaProducer(cq, spell=unified.spell)
    lines = []
    facts = [("owl", "fly", "affirm", {}), ("penguin", "walks", "negate", {}),
             ("penguin", "fly", None, {"negated_modal": True})]
    for subj, verb, pol, extra in facts:
        d = decision_from_emerge("ANSWER", subject=subj, verb=verb, polarity=pol, **extra)
        r = prod.speak(d)
        lines.append((f"can {subj} {verb}?", r["surface"], "producer INVOKED"))
    r4 = prod.speak(decision_from_emerge("ABSTAIN"))
    lines.append(("can a zzz fly?", "I don't know." if not r4["produced"] else r4["surface"],
                  "producer NOT invoked"))
    return lines, prod.production_count


def _demo(seed=42):
    print("\n=== EMERGE-68 -- extend the SPIKING A->W read-out to the FUNCTION-WORD slots: EVERY word of the EMERGE-"
          "frame render (the/owl/can/fly) is now produced ON SPIKES (content on BRIDGE-A, function on BRIDGE-F) ===\n",
          flush=True)
    unified = UnifiedNeuralSpell(load=True)
    if not unified._backend_gpu:
        print("  [skip] the A->W engines need a GPU (SIM_BACKEND=cupy); numpy fallback cannot run the read-out.\n")
        return
    frate, fper = _func_wordwise_accuracy(unified.func)
    print(f"  BRIDGE-F spells {int(frate*len(fper))}/{len(fper)} FUNCTION words correctly (spike-decoded); "
          f"mean spike-total {int(np.mean([p['spikes'] for p in fper]))}")
    for p in fper:
        print(f"    {p['word']:5s} -> {p['decoded']:5s} {'OK' if p['ok'] else 'X'} "
              f"(self_cos {p['self_cos']}, spikes {p['spikes']})")
    print()
    lines, pc = _sample_transcript(unified, seed)
    for q, surface, inv in lines:
        print(f"  you> {q}\n      broca> {surface}   [{inv}; ALL words spike-spelled]")
    print(f"\n  producer-invocation count after 4 probes: {pc} (the abstain never invoked the producer/spell -- moat)\n")


def _train():
    """Build + train + cache the FUNCTION-word A->W engine (BRIDGE-F, GPU). Idempotent: overwrites the cache."""
    if _FUNC_CACHE_BRIDGE.exists():
        _FUNC_CACHE_BRIDGE.unlink()
    print("[emerge68] building + training the FUNCTION-word A->W engine BRIDGE-F (GPU, validated scale)...", flush=True)
    t0 = time.time()
    func = FuncNeuralSpell(load=False)
    if not func._backend_gpu:
        print("[emerge68] SKIP -- needs SIM_BACKEND=cupy (GPU) to train the A->W engine.")
        return 1
    rate, per = _func_wordwise_accuracy(func)
    print(f"[emerge68] trained + cached ({time.time()-t0:.0f}s). BRIDGE-F spells {int(rate*len(per))}/{len(per)} "
          f"function words correctly. cache: {_FUNC_CACHE_BRIDGE}", flush=True)
    for p in per:
        print(f"    {p['word']:5s} -> {p['decoded']:5s} {'OK' if p['ok'] else 'X'} (self_cos {p['self_cos']}, "
              f"top_cos {p['top_cos']}, spikes {p['spikes']})", flush=True)
    return 0


def _derisk(seeds, train_events=_TRAIN_EVENTS):
    print(f"EMERGE-68 de-risk: extend the SPIKING A->W to the FUNCTION-word slots -- ALL words (content+function) "
          f"spike-spelled; all-word accuracy + function-word lesion collapse + gate-first moat + no-regression; "
          f"{len(seeds)}-seed", flush=True)
    t0 = time.time()
    err = None
    per = []
    frate = None
    fper = []
    content_rate = None
    lesion_func_engine = None
    gpu = True
    try:
        unified = UnifiedNeuralSpell(load=True, train_events=train_events)
        gpu = unified._backend_gpu
        if not gpu:
            raise RuntimeError("A->W engines require SIM_BACKEND=cupy (GPU); numpy cannot run the spiking read-out")
        frate, fper = _func_wordwise_accuracy(unified.func)
        content_rate, _cper = m67._aw_wordwise_accuracy(unified.content)
        # a function-lesioned unified engine (BRIDGE-F pool->output zeroed) for the genuinely-spiking control
        unified_func_lesion = UnifiedNeuralSpell(load=True, train_events=train_events, func_lesion=True)
        lesion_func_engine, _ = _func_wordwise_accuracy(unified_func_lesion.func)
        for s in seeds:
            d = _derisk_one(s, unified, unified_func_lesion=unified_func_lesion)
            per.append(d)
            la = f"{d['lesion_func_acc']:.3f}" if d["lesion_func_acc"] is not None else "n/a"
            print(f"  [seed {s}] all-word acc {d['all_acc']:.3f} | func acc {d['func_acc']:.3f} | "
                  f"func-lesion acc {la} | regress {d['regress_mismatch']} | "
                  f"spell-on-abstain {d['spell_calls_on_abstain']} | producer-on-abstain "
                  f"{d['producer_calls_on_abstain']}", flush=True)
    except Exception as e:
        err = repr(e)
        traceback.print_exc()

    if err is None:
        def m(k):
            return float(np.mean([d[k] for d in per]))
        all_acc = m("all_acc")
        func_acc = m("func_acc")
        lesion_vals = [d["lesion_func_acc"] for d in per if d["lesion_func_acc"] is not None]
        lesion_func_acc = float(np.mean(lesion_vals)) if lesion_vals else None
        regress = int(sum(d["regress_mismatch"] for d in per))
        spell_calls_abstain = int(sum(d["spell_calls_on_abstain"] for d in per))
        producer_calls_abstain = int(sum(d["producer_calls_on_abstain"] for d in per))
        answer_ok = all(d["answer_produced"] for d in per)

        BAR = 0.90
        all_ok = all_acc >= BAR
        func_ok = func_acc >= BAR
        # genuinely spiking for the FUNCTION words: the function engine's own rate is high AND the function lesion
        # collapses BOTH the frame function-word decode and the engine's own rate (a host lookup would be unaffected).
        spiking_ok = (frate is not None and frate >= BAR
                      and lesion_func_acc is not None and (func_acc - lesion_func_acc) >= 0.40
                      and lesion_func_engine is not None and (frate - lesion_func_engine) >= 0.40)
        no_regress = (regress == 0)
        moat_ok = (spell_calls_abstain == 0) and (producer_calls_abstain == 0) and answer_ok

        go = bool(all_ok and func_ok and spiking_ok and no_regress and moat_ok)
        if go:
            verdict = (
                f"GO -- the EMERGE-frame render is now 100% PRODUCED ON SPIKES: ORDER *and* EVERY WORD (content AND "
                f"function). The spiking-Broca producer's `spell` is the VALIDATED A->W read-out extended to the "
                f"FUNCTION-word (DET/FUNC) slots: content words decode on BRIDGE-A (== EMERGE-67), function words "
                f"(the/a/can/does/not) decode on BRIDGE-F -- each spelled by DRIVING its concept pool on a real "
                f"SimulationBridge and DECODING the spoken word from `language_output` SPIKES. ALL-WORD spike-spell "
                f"accuracy {all_acc:.3f} over the EMERGE frames (>= {BAR}); FUNCTION-word slot accuracy {func_acc:.3f}; "
                f"BRIDGE-F spells {int((frate or 0)*len(fper))}/{len(fper)} function words correctly (raw rate "
                f"{frate:.3f}). GENUINELY SPIKING for the function words too: the FUNCTION-word LESION (zeroing "
                f"BRIDGE-F's pool->language_output pathway) COLLAPSES the function-word decode to {lesion_func_acc:.3f} "
                f"(engine-lesion {lesion_func_engine:.3f}; a host lookup would be unaffected -> the function words are "
                f"decoded FROM SPIKES). NO regression: the neural spell reproduces the token spell on ALL slots "
                f"({regress} mismatches; the token-spell default path is byte-identical). The gate-first no-confab MOAT "
                f"holds BY CONSTRUCTION: {spell_calls_abstain} spell calls + {producer_calls_abstain} producer "
                f"invocations on abstains (the producer -- and hence BOTH A->W read-outs -- is NEVER invoked on an "
                f"abstain). {len(seeds)} seeds. ==> the EMERGE-frame render is FULLY SPIKING (order via EMERGE-59/63, "
                f"content words via EMERGE-67, FUNCTION words now). HONEST SCOPE: renders the BOUNDED EMERGE frame "
                f"inventory (ability-affirm / intransitive-exception / negated-modal), NOT open prose (R4). The A->W "
                f"engines are GPU-trained ONCE at the validated scale + cached (a scale/data lever, not a new "
                f"mechanism); the 5 function words are rebound onto 5 concept pools of a second bridge (reuse-by-import; "
                f"NO sim/ edit).")
        else:
            miss = []
            if not all_ok:
                miss.append(f"all-word spike-spell accuracy {all_acc:.3f} < {BAR}")
            if not func_ok:
                miss.append(f"function-word slot accuracy {func_acc:.3f} < {BAR}")
            if not spiking_ok:
                miss.append(f"function read-out not clearly spiking (BRIDGE-F rate {frate}, func-lesion "
                            f"{lesion_func_acc}, engine-lesion {lesion_func_engine} -- the lesion did not collapse the "
                            f"function-word decode by >= 0.40)")
            if not no_regress:
                miss.append(f"slot REGRESSION vs the token spell ({regress} mismatches)")
            if not moat_ok:
                miss.append(f"MOAT: {spell_calls_abstain} spell-calls + {producer_calls_abstain} producer-calls on "
                            f"abstains / answer-produced {answer_ok} -- BLOCKING if the producer/spell ran on abstain")
            verdict = ("BOUNDARY -- " + "; ".join(miss) + ". THE EXACT GAP: the CONTENT words spell 100% on BRIDGE-A "
                       "(EMERGE-67 GO), but the FUNCTION words (the/a/can/does/not) are HIGH-frequency closed-class "
                       "words whose orthogonal-band A->W codes may not separate as cleanly on the shared BRIDGE-F "
                       "read-out at this training budget (function words co-occur with everything -- the EMERGE-62 "
                       "Goldilocks signature -- so their pool->language_output selectivity is the harder read). NEXT "
                       "STEP: train BRIDGE-F at the fully-validated scale (n_per_pool=500, more events) or give each "
                       "function word a DEDICATED closed-class pool block (the EMERGE-62 discovered class as its own "
                       "kind). This is a SCALE/DATA lever + a bounded pool-assignment change, NOT a new mechanism. If "
                       "the MOAT was breached this is BLOCKING -- do NOT weaken the moat.")
    elif not gpu:
        go = False
        verdict = ("SKIP/BOUNDARY -- the spiking A->W read-out requires SIM_BACKEND=cupy (GPU); this run had only the "
                   "numpy backend, which cannot execute the concept-pool spiking read-out. Re-run on GPU with "
                   "SIM_BACKEND=cupy. The wire (unified spell -> BrocaProducer.spell) + the moat logic are CPU-testable "
                   "(tests/test_emerge68_function_word_spell.py); the on-spikes A->W is GPU-only.")
        all_acc = func_acc = lesion_func_acc = None
        regress = spell_calls_abstain = producer_calls_abstain = None
    else:
        go = False
        verdict = f"ERROR -- {err}"
        all_acc = func_acc = lesion_func_acc = None
        regress = spell_calls_abstain = producer_calls_abstain = None

    transcript = []
    try:
        if err is None and gpu:
            lines, _ = _sample_transcript(unified, seeds[0])
            transcript = [{"question": q, "surface": s, "invocation": i} for (q, s, i) in lines]
    except Exception:
        pass

    summary = {
        "probe": "emerge68_function_word_spell", "verdict": verdict,
        "go": bool(go) if err is None and gpu else False,
        "mechanism": ("extend the VALIDATED spiking A->W read-out (concept_speak_demo: drive a concept pool -> decode "
                      "the spoken word from language_output SPIKES via cosine to the word patterns; CLAUDE.md chat_speak "
                      "A->W 100% multi-seed) to the FUNCTION-WORD (DET/FUNC) slots of the EMERGE-59 spiking-Broca "
                      "producer. EMERGE-67 wired the CONTENT slots (BRIDGE-A, the 16-word content engine); EMERGE-68 "
                      "adds a SECOND concept-pool bridge BRIDGE-F for the 5 function words (the/a/can/does/not) rebound "
                      "onto 5 of its 16 pools (motor_N/E/S/W + noun_APPLE; the other 11 pools filled with content-word "
                      "fillers so the SAME builder+topographic-bias+orthogonal-codes+train_word_to_pool recipe trains at "
                      "the validated scale, cached once). UnifiedNeuralSpell.spell dispatches content->BRIDGE-A, "
                      "function->BRIDGE-F; realize_slot's DET/FUNC branch (which already calls spell(payload)) routes "
                      "through this SAME spell, so EVERY slot -- content AND function -- is spike-spelled. The read-out "
                      "is genuinely spiking (reads cp_firing_states[language_output]; a BRIDGE-F pool->output lesion "
                      "collapses the function-word decode). The gate-first no-confab moat is untouched (abstain -> "
                      "producer + both A->W engines NEVER invoked). Reuse-by-import; NO sim/ edit."),
        "task": ("extend the spiking A->W read-out to the FUNCTION-word slots so the EMERGE-frame render's WORDS are "
                 "100% produced on spikes (order AND every word, content AND function); all-word spike-spell accuracy + "
                 "genuinely-spiking (function-word lesion collapse) + gate-first moat (0 spell/producer calls on "
                 "abstains) + no regression vs the token spell; >=6 seeds; GPU for the A->W"),
        "func_words": _FUNC_WORDS,
        "content_vocab": m67._AW_CONTENT,
        "seeds": list(seeds), "gpu": bool(gpu), "elapsed_seconds": round(time.time() - t0, 1),
        "aggregate": None if (err is not None or not gpu) else {
            "all_acc": all_acc, "func_acc": func_acc, "func_wordwise_rate": frate, "content_wordwise_rate": content_rate,
            "lesion_func_acc": lesion_func_acc, "engine_lesion_func_acc": lesion_func_engine,
            "regress_mismatch": regress, "spell_calls_on_abstain_total": spell_calls_abstain,
            "producer_calls_on_abstain_total": producer_calls_abstain,
        },
        "func_wordwise": fper,
        "sample_transcript": transcript,
        "per_seed": per,
        "HONEST_NOTE": ("Makes ALL WORDS (content + function) of the BOUNDED EMERGE frame inventory PRODUCED ON SPIKES "
                        "via the validated A->W read-out (the ORDER was already spiking, EMERGE-59/63; the CONTENT words "
                        "since EMERGE-67; the FUNCTION words now). ==> the EMERGE-frame render is 100% spiking. The A->W "
                        "engines are GPU-trained ONCE at the production scale + cached (a scale/data lever, not a new "
                        "mechanism): BRIDGE-A = EMERGE-67's 16 content words, BRIDGE-F = the 5 function words rebound "
                        "onto 5 concept pools (a research-runner vocab rebind on a second bridge, NO sim/ edit). The "
                        "read-out is genuinely spiking (decodes from cp_firing_states[language_output]; the BRIDGE-F "
                        "pool->language_output lesion collapses the function-word decode -- a host lookup would be "
                        "unaffected). The gate-first no-confab moat is untouched (0 spell/producer invocations on "
                        "abstains, by construction). NOT open prose (R4, the separate deferred wall)."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118, flush=True)
    print(f"[emerge68] VERDICT: {verdict}", flush=True)
    print(f"[emerge68] wrote {OUT}\n" + "=" * 118, flush=True)
    return 0 if (err is None and gpu and go) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--train-events", type=int, default=_TRAIN_EVENTS)
    ap.add_argument("--train", action="store_true", help="build + train + cache the FUNCTION-word A->W engine (GPU)")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--derisk", action="store_true")
    a = ap.parse_args()
    if a.train:
        return _train()
    if a.derisk:
        return _derisk(a.seeds, train_events=a.train_events)
    _demo(a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
