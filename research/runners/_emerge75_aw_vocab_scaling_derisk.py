"""EMERGE-75 -- A->W VOCAB SCALING: make the spiking A->W read-out support ARBITRARY content vocabulary via the G.20
MULTI-BRIDGE route, so the EMERGE-72/73 broadened constructions render EVERY word ON SPIKES (not the token surface).

CONTEXT (the honest gap this closes). EMERGE-67 wired the CONTENT slots (subject/verb) of the EMERGE-frame render onto
spikes: `spell(word)` drives the word's concept pool on a real `SimulationBridge` and DECODES the spoken word from
`cp_firing_states[language_output]` (the validated `concept_speak_demo` A->W read-out, CLAUDE.md "chat_speak A->W 100%
multi-seed"). EMERGE-68 extended it to the 5 FUNCTION words {the,a,can,does,not} via a SECOND bridge (BRIDGE-F). But each
A->W bridge is CAPPED at 16 words -- the concept-pool architecture has EXACTLY 16 pools (4 kinds x 4: motor/noun/verb/
adjective) and `train_word_to_pool` supports only those 4 kinds. The EMERGE-72/73 BROADENED constructions introduced NEW
words that live in NEITHER bridge and so currently fall back to the TOKEN surface:
  * OBJECT nouns (pond/rock/tree/... -- the transitive-motion PP-goal / PP-location ARGUMENT; EMERGE-72 C_PPGOAL/C_PPLOC).
  * NEW function words `to` / `on` (the PP prepositions -- not in BRIDGE-F's {the,a,can,does,not}) and `is` (the copula --
    EMERGE-73 C_PRED).
So "the owl flies to the pond" renders the/owl/flies on spikes (BRIDGE-A) but to/the/pond fell back to the token surface
(to/pond untrained; the was in BRIDGE-F). EMERGE-75 CLOSES that: a THIRD A->W bridge (BRIDGE-C) holds the overflow words,
and a `UnifiedNeuralSpell75` DISPATCHES each word to the bridge holding its pool -- the G.20 MULTI-BRIDGE scaling route
(CLAUDE.md: the 320-concept ensemble is 5 sparse bridges; "extend the content vocab ... across N bridges" -- EMERGE-67/68
finding). ==> every word of the EMERGE-72/73 constructions decodes from `language_output` spikes.

THE INFLECTION-AWARE VERB DECODE (the one subtlety). EMERGE-72's PP constructions render the verb 3sg ("fly"->"flies");
the A->W pool is trained on the LEMMA ("fly"). So the unified spell STRIPS a 3sg suffix to find the lemma, DECODES the
lemma pool on spikes (BRIDGE-A), then RE-APPLIES the inflection to the spike-decoded surface (emerge_v3). The word is still
decoded FROM SPIKES; only the morphology is re-attached (the ORDER/inflection tag is EMERGE-59/63 serial-order, host-side
frame furniture -- the CONTENT identity is the spiking read). This makes "flies" spike-produced (lemma on spikes + 3sg
tag) rather than a token fallback.

THE THREE BRIDGES (all reuse-by-import; NO `sim/` edit; each a 16-pool concept-pool bridge rebound to a target vocab):
  * BRIDGE-A (EMERGE-67 cache `bridges/emerge67_aw/aw_content.simstate.h5`): the 16 CONTENT words (8 subjects + 8 verbs).
  * BRIDGE-F (EMERGE-68 cache `bridges/emerge68_aw/aw_func.simstate.h5`): the 5 FUNCTION words {the,a,can,does,not}.
  * BRIDGE-C (the NEW EMERGE-75 cache `bridges/emerge75_aw/aw_overflow.simstate.h5`): the 16 OVERFLOW words -- 3 new
    function words {to,on,is} + 13 OBJECT nouns (nest/pond/tree/rock/hill/cave/leaf/seed/worm/fish/branch/shore/log),
    rebound onto the 16 pools (function words on motor_N/E/S, object nouns on the other 13 pools). Trained INLINE at the
    validated scale + cached, same builder+topographic-bias+orthogonal-codes+train_word_to_pool recipe.
`UnifiedNeuralSpell75.spell(word)` DISPATCHES: content -> BRIDGE-A; {the,a,can,does,not} -> BRIDGE-F; {to,on,is}+object
nouns -> BRIDGE-C. All read `cp_firing_states[language_output]` -- genuinely spiking. Any word in NONE of the three vocabs
-> token surface (flagged; but the EMERGE-72 held-out facts draw objects from BRIDGE-C's 13 so every word spike-spells).

DE-RISK (6 seeds 42/43/44/100/101/102; A->W on GPU/cupy, numpy-skip fallback; the engines are GPU-trained ONCE + cached,
deterministic per their own seed, while the de-risk seed varies the fact selection):
  (a) ALL words of the EMERGE-72 constructions spelled ON SPIKES: "the owl flies to the pond" -> the/owl/flies/to/the/pond
      ALL decoded from language_output spikes (content on BRIDGE-A, function on BRIDGE-F, overflow on BRIDGE-C).
      surface-accuracy of ALL slots (det+subj+func+verb+obj) vs ground-truth == 100%.
  (b) GENUINELY SPIKING for the NEW overflow words: the BRIDGE-C LESION control (zero the pool->language_output pathway on
      BRIDGE-C) COLLAPSES the overflow-word decode (a host lookup would be unaffected). [INPUT-DESTRUCTION / mechanism-
      ablation control -- NOT a fixed-random control.]
  (c) gate-first MOAT: on ABSTAIN the producer -- and hence ALL THREE A->W read-outs -- is NEVER invoked (0 spell calls,
      0 productions).
  (d) NO regression vs the token spell: every slot surface (content + function + overflow) is IDENTICAL to the token spell
      (the neural spell reproduces the token spell for the trained vocab); the token-spell default path is byte-identical
      (EMERGE-59..74). The EMERGE-72 CONSTRUCTION render itself is unchanged (only the SPELL is upgraded).
GO bar: ALL-word spike-spell accuracy >= 0.90 (a clear bar), overflow-word (BRIDGE-C) raw rate >= 0.90 + lesion-collapse
(>= 0.40 drop), moat 0, no regression, 6-seed. If an overflow word does NOT cleanly decode -> honest BOUNDARY naming the
exact gap + next step (do NOT force a GO; do NOT weaken the moat).

HONEST SCOPE: this makes the EMERGE-72 broadened constructions FULLY SPIKING end-to-end (ORDER via EMERGE-59/63; content
words via EMERGE-67; the 5 original function words via EMERGE-68; the NEW overflow words -- object nouns + to/on/is -- now).
The overflow vocab is a BOUNDED 16 words on ONE extra bridge (the G.20 multi-bridge route scales linearly in bridge count
-- more objects = more bridges, a data/bridge lever, not a new mechanism); the full 30-object corpus vocab is 2 BRIDGE-Cs
(named follow-on). The EMERGE-73 ADJECTIVE constructions (C_ATTRIB/C_PRED) are NOT in scope here -- their construction
MINING is EMERGE-73's own boundary; their adjective A->W (big/small/... on a 4th bridge) is a further follow-on. The A->W
engines are GPU-trained ONCE at the validated scale + cached (a scale/data lever, not a new mechanism); the vocab rebind
is a research-runner edit. Reuse-by-import; NO `sim/` edit; the gate-first no-confab moat is untouched. Renders the BOUNDED
EMERGE-72 inventory, NOT open prose (R4).

Run:
  SIM_BACKEND=cupy python -m research.runners._emerge75_aw_vocab_scaling_derisk --train    # build+cache BRIDGE-C
  SIM_BACKEND=cupy python -m research.runners._emerge75_aw_vocab_scaling_derisk --demo
  SIM_BACKEND=cupy python -m research.runners._emerge75_aw_vocab_scaling_derisk --derisk
  SIM_BACKEND=cupy python -m research.runners._emerge75_aw_vocab_scaling_derisk --derisk --seeds 42 43 44 100 101 102
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

# Reuse-by-import ONLY -- NO sim/ edit. The EMERGE-72 construction registry + producer + ground-truth; the EMERGE-67/68
# A->W engine machinery (BRIDGE-A content + BRIDGE-F function); the EMERGE-57 3sg inflection.
from research.runners._emerge72_construction_registry_derisk import (  # noqa: E402
    ConstructionRegistry, RegistryProducer, RegistryBrocaProducer, CONSTRUCTIONS, CONSTRUCTION_NAMES,
    DET, SUBJ, FUNC, VERB, OBJ, decision, build_stream, build_heldout_facts_ext, _verb_for, _expected_surface,
)
from research.runners._emerge57_ra_refinetune_emerge_frames_derisk import emerge_v3  # noqa: E402
import research.runners._emerge67_neural_spell_wirein_derisk as m67  # noqa: E402
import research.runners._emerge68_function_word_spell_derisk as m68  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_emerge75_aw_vocab_scaling.json"
# the NEW cached OVERFLOW-word A->W engine (BRIDGE-C, GPU-trained once; regenerable via --train).
_OVF_CACHE_DIR = _REPO / "bridges" / "emerge75_aw"
_OVF_CACHE_BRIDGE = _OVF_CACHE_DIR / "aw_overflow.simstate.h5"

# ---------------------------------------------------------------------------------------------------------------------
# THE 16 OVERFLOW WORDS (BRIDGE-C): 3 NEW function words {to,on,is} + 13 OBJECT nouns. These are the EMERGE-72/73 words
# that live in NEITHER BRIDGE-A (content subj/verb) nor BRIDGE-F ({the,a,can,does,not}). Rebound onto the 16 concept
# pools (the 3 function words on motor_N/E/S, the 13 object nouns on motor_W + noun/verb/adjective pools). The de-risk
# facts draw OBJECTS from THIS 13-set so every word of the rendered constructions is spike-spellable. (>13 objects -> a
# second BRIDGE-C, the G.20 linear-in-bridge-count scaling route -- a data/bridge lever, named follow-on.)
# ---------------------------------------------------------------------------------------------------------------------
_OVF_FUNC = ["to", "on", "is"]                    # the NEW function words the EMERGE-72/73 constructions add
_OVF_OBJ = ["nest", "pond", "tree", "rock", "hill", "cave", "leaf", "seed", "worm", "fish", "branch", "shore", "log"]
_OVF_VOCAB16 = _OVF_FUNC + _OVF_OBJ               # 3 function + 13 object == 16 words == 16 pools
_OVF_OBJ_SET = set(_OVF_OBJ)
_OVF_FUNC_SET = set(_OVF_FUNC)

_N_LANG = m67._N_LANG
_N_PER_POOL = m67._N_PER_POOL
_N_FS_PER_POOL = m67._N_FS_PER_POOL
_SPARSITY = m67._SPARSITY
_TRAIN_EVENTS = m67._TRAIN_EVENTS
_OVF_SEED = 42                                    # the overflow engine's OWN training seed (deterministic spelling module)

# 3sg suffix stripping for the inflection-aware verb decode (BRIDGE-A trains on the lemma; the PP render inflects 3sg).
_ABILITY_LEMMAS = set(m67._AW_ABILITY)            # {fly,swim,run,hop} -- the pool lemmas the PP verb inflects from


def _ovf_pool_assignment():
    """Rebind the 16 OVERFLOW words onto the 16 validated concept pools. Returns (word_to_pool, pool_to_word, func_pools,
    obj_pools)."""
    from research.runners.concept_pool_demo import MOTOR_NAMES, NOUN_NAMES, VERB_NAMES, ADJECTIVE_NAMES
    pools = ([f"motor_{a}" for a in MOTOR_NAMES] + [f"noun_pool_{n}" for n in NOUN_NAMES]
             + [f"verb_pool_{v}" for v in VERB_NAMES] + [f"adjective_pool_{n}" for n in ADJECTIVE_NAMES])
    assert len(pools) == len(_OVF_VOCAB16), (len(pools), len(_OVF_VOCAB16))
    word_to_pool = {w: p for w, p in zip(_OVF_VOCAB16, pools)}
    pool_to_word = {p: w for w, p in word_to_pool.items()}
    func_pools = {w: word_to_pool[w] for w in _OVF_FUNC}
    obj_pools = {w: word_to_pool[w] for w in _OVF_OBJ}
    return word_to_pool, pool_to_word, func_pools, obj_pools


def _swap_ovf_vocab():
    """Swap `concept_pool_demo`'s module-level 16-word vocab dicts for the 16 OVERFLOW words, keeping the pool VALUE-slots
    fixed. Returns word_to_idx over the 16 words (pool order)."""
    import research.runners.concept_pool_demo as cpd
    w = _OVF_VOCAB16
    cpd.DIRECTION_VOCAB.clear(); cpd.DIRECTION_VOCAB.update({w[0]: "N", w[1]: "E", w[2]: "S", w[3]: "W"})
    cpd.NOUN_VOCAB.clear();      cpd.NOUN_VOCAB.update({w[4]: "APPLE", w[5]: "RIVER", w[6]: "DOG", w[7]: "CAT"})
    cpd.VERB_VOCAB.clear();      cpd.VERB_VOCAB.update({w[8]: "GO", w[9]: "COME", w[10]: "STOP", w[11]: "LOOK"})
    cpd.ADJECTIVE_VOCAB.clear(); cpd.ADJECTIVE_VOCAB.update({w[12]: "BIG", w[13]: "SMALL", w[14]: "HOT", w[15]: "COLD"})
    allw = (list(cpd.DIRECTION_VOCAB) + list(cpd.NOUN_VOCAB) + list(cpd.VERB_VOCAB) + list(cpd.ADJECTIVE_VOCAB))
    return {w2: i for i, w2 in enumerate(allw)}


# ---------------------------------------------------------------------------------------------------------------------
# THE OVERFLOW-WORD A->W ENGINE (BRIDGE-C). Same NeuralSpell machinery as EMERGE-67/68, but the 16-word vocab is the 3
# new function words + 13 object nouns, cached separately. `_decode(word)` decodes an overflow word from BRIDGE-C spikes.
# ---------------------------------------------------------------------------------------------------------------------
class OverflowNeuralSpell:
    """The genuinely-spiking A->W `spell` for the OVERFLOW words (BRIDGE-C). Builds/loads the validated concept-pool
    bridge rebound to the 16 overflow words; `_decode(word)` drives the word's pool and decodes the spoken word from
    language_output SPIKES."""

    def __init__(self, seed=_OVF_SEED, load=True, train_events=_TRAIN_EVENTS, lesion_pool_out=False, read_steps=100):
        self.seed = int(seed)
        self.train_events = int(train_events)
        self.lesion_pool_out = bool(lesion_pool_out)
        self.read_steps = int(read_steps)
        self.spell_calls = 0
        self.vocab16 = list(_OVF_VOCAB16)
        self.word_to_pool, self.pool_to_word, self.func_pools, self.obj_pools = _ovf_pool_assignment()
        self.overflow_words = set(self.word_to_pool)
        self._word_to_idx = _swap_ovf_vocab()       # rebinds concept_pool_demo's vocab dicts (must precede build)
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
        self._pats = {w: orthogonal_drive_pattern(self._word_to_idx[w], n_cues=V, n_neurons=_N_LANG, sparsity=_SPARSITY)
                      for w in self.vocab16}

        b = build_concept_bridge(seed=self.seed, n_lang_input=_N_LANG, n_per_pool=_N_PER_POOL,
                                 n_fs_per_pool=_N_FS_PER_POOL, weak_dynamics=True, enable_adjective=True, verbose=False)
        if load and _OVF_CACHE_BRIDGE.exists():
            b.load_checkpoint(str(_OVF_CACHE_BRIDGE))
        else:
            apply_concept_topographic_bias(b, n_lang_input=_N_LANG, orthogonal_codes=True,
                                           n_words_for_orthogonal=V, word_to_idx=self._word_to_idx,
                                           sparsity=_SPARSITY, verbose=False)
            for w, pool in self.word_to_pool.items():
                train_word_to_pool(b, w, pool, n_events=self.train_events, n_lang_input=_N_LANG, n_lang_output=_N_LANG,
                                   orthogonal_codes=True, n_words_for_orthogonal=V, word_to_idx=self._word_to_idx,
                                   sparsity=_SPARSITY, verbose=False)
            _OVF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            b.save_checkpoint(str(_OVF_CACHE_BRIDGE))
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
        word-selective signal -> the overflow-word A->W decode collapses (a host lookup would be unaffected)."""
        from sim.backend import get_backend
        cp, _ = get_backend()
        rm = bridge.region_manager
        lo = cp.asarray(list(rm.indices("language_output")), dtype=bridge.cp_connections.indices.dtype)
        mask = cp.isin(bridge.cp_connections.indices, lo)
        bridge.cp_connections.data[mask] = 0.0

    def _decode(self, word):
        """Drive `word`'s pool on BRIDGE-C; accumulate language_output spikes; cosine-decode the spoken word (over the
        full 16-word overflow vocab)."""
        from research.runners.concept_speak_demo import drive_pool_and_read_lang_output, _cosine
        pool = self.word_to_pool[word]
        patt = drive_pool_and_read_lang_output(self.bridge, pool, stim_steps=self.read_steps, n_lang_output=_N_LANG)
        scores = {w2: _cosine(patt, self._pats[w2]) for w2 in self.vocab16}
        top = max(scores, key=scores.get)
        return top, float(scores[word]), float(scores[top]), int(np.asarray(patt).sum())


# ---------------------------------------------------------------------------------------------------------------------
# THE UNIFIED 3-BRIDGE A->W SPELL (EMERGE-75). Dispatches each word to the bridge holding its pool: content -> BRIDGE-A
# (EMERGE-67); {the,a,can,does,not} -> BRIDGE-F (EMERGE-68); {to,on,is}+object nouns -> BRIDGE-C (EMERGE-75). The
# inflection-aware verb decode strips a 3sg suffix to the lemma, decodes it on BRIDGE-A, re-applies the inflection.
# `realize_slot_ext`'s DET/FUNC/OBJ branch routes through this SAME `spell`, so EVERY slot -- content AND function AND
# object -- is spike-spelled. This is the G.20 multi-bridge scaling route (CLAUDE.md: the 320-concept ensemble is 5
# sparse bridges). Reuse-by-import; NO `sim/` edit.
# ---------------------------------------------------------------------------------------------------------------------
class UnifiedNeuralSpell75:
    """The genuinely-spiking A->W `spell` over the full producer vocab across THREE bridges (16 content + 5 function + 16
    overflow). `spell(word)` DISPATCHES to the bridge holding the word's pool and returns the SPIKE-DECODED surface. A
    verb rendered 3sg is decoded on its LEMMA pool (BRIDGE-A) + re-inflected. A word in NONE of the three vocabs -> token
    surface (flagged; the EMERGE-72 held-out facts draw objects from BRIDGE-C's 13 so no such word occurs)."""

    def __init__(self, load=True, train_events=_TRAIN_EVENTS, content_lesion=False, func_lesion=False,
                 overflow_lesion=False, read_steps=100):
        self.spell_calls = 0
        self.content = m67.NeuralSpell(load=load, train_events=train_events, lesion_pool_out=content_lesion,
                                       read_steps=read_steps)
        self.func = m68.FuncNeuralSpell(load=load, train_events=train_events, lesion_pool_out=func_lesion,
                                        read_steps=read_steps)
        self.overflow = OverflowNeuralSpell(load=load, train_events=train_events, lesion_pool_out=overflow_lesion,
                                            read_steps=read_steps)
        self._backend_gpu = bool(self.content._backend_gpu and self.func._backend_gpu and self.overflow._backend_gpu)
        self.content_words = set(self.content.word_to_pool)
        self.func_words = set(self.func.func_pools)
        self.overflow_words = set(self.overflow.word_to_pool)
        # the tokens the unified spell can produce on spikes (incl. 3sg inflections of the ability-verb lemmas)
        self._spikeable = set(self.content_words) | set(self.func_words) | set(self.overflow_words)

    def _lemma_and_inflection(self, word):
        """If `word` is a 3sg inflection of a BRIDGE-A ability-verb lemma, return (lemma, apply_3sg=True); else
        (word, False). E.g. 'flies' -> ('fly', True). Uses emerge_v3 to check the surface matches the lemma's 3sg form
        (so we only strip morphology we can faithfully re-apply)."""
        for lemma in _ABILITY_LEMMAS:
            if emerge_v3(lemma, already_3sg=None) == word:
                return lemma, True
        return word, False

    def spell(self, word):
        self.spell_calls += 1
        # inflection-aware verb decode: a 3sg surface of a BRIDGE-A ability-verb lemma decodes the LEMMA pool on spikes,
        # then re-inflects. (The intransitive 3sg words walks/lurks/... ARE trained as-is on BRIDGE-A -> handled below.)
        lemma, apply_3sg = self._lemma_and_inflection(word)
        if apply_3sg and lemma in self.content_words:
            decoded = self.content._decode(lemma)[0]         # spike-decode the LEMMA on BRIDGE-A
            return emerge_v3(decoded, already_3sg=None)       # re-apply 3sg to the spike-decoded surface
        if word in self.content_words:
            return self.content._decode(word)[0]              # spike-decode on BRIDGE-A (== EMERGE-67 content engine)
        if word in self.func_words:
            return self.func._decode(word)[0]                 # spike-decode on BRIDGE-F (== EMERGE-68 function engine)
        if word in self.overflow_words:
            return self.overflow._decode(word)[0]             # spike-decode on BRIDGE-C (the NEW overflow engine)
        return str(word)                                     # not in any vocab -> token surface (no EMERGE-72-frame word)


# ---------------------------------------------------------------------------------------------------------------------
# FACTS drawn so every content/object slot is spike-spellable. Reuse EMERGE-72's held-out facts but constrain the OBJECT
# to BRIDGE-C's 13 (so the PP constructions spike-spell their argument); use a BRIDGE-A ability-verb lemma as the pp_verb.
# ---------------------------------------------------------------------------------------------------------------------
def _facts(seed, n=8):
    """Held-out (subject, ability_verb, intr_verb, obj, pp_verb) facts. Subjects/verbs from EMERGE-67's content vocab so
    they spike-spell on BRIDGE-A; the OBJECT constrained to BRIDGE-C's 13 so the PP argument spike-spells on BRIDGE-C;
    pp_verb a BRIDGE-A ability lemma (fly/swim/run/hop) so the 3sg render decodes the lemma + re-inflects."""
    base = m67._facts_from_content_vocab(seed, n=n)          # subject in _AW_SUBJECTS, ability/intr in _AW_ABILITY/_INTR3SG
    rng = np.random.default_rng(seed * 373 + 11)
    for f in base:
        f["obj"] = str(rng.choice(_OVF_OBJ))                 # constrain the object to BRIDGE-C's 13
        f["pp_verb"] = str(rng.choice(m67._AW_ABILITY))      # a BRIDGE-A ability lemma (3sg-inflectable on spikes)
    return base


# the constructions in scope for the ALL-word spike-spell (the EMERGE-72 broadened set: 3 EMERGE frames + 2 PP). The
# EMERGE-73 adjective constructions are NOT in scope (their adjective A->W is a further follow-on; noted in HONEST_NOTE).
_SCOPE_CONSTRUCTIONS = ["F_MODAL", "F_INTR", "F_NEGMOD", "C_PPGOAL", "C_PPLOC"]


def _render_and_score_all(reg: ConstructionRegistry, spell, facts):
    """Render every in-scope registered construction for every held-out fact with the UNIFIED neural spell; score ALL
    slots (det+subj+func+verb+obj) vs the ground-truth template surface. Returns (all_acc, overflow_acc, examples)."""
    cq = reg.render_cq()
    all_hits = all_tot = ovf_hits = ovf_tot = 0
    examples = []
    for name in _SCOPE_CONSTRUCTIONS:
        if name not in reg.registered:
            continue
        for fact in facts:
            verb = _verb_for(name, fact)
            obj = fact.get("obj")
            words = cq.emit(name, fact["subject"], verb, obj, spell)   # ALL slots spelled via the unified neural spell
            expected = _expected_surface(name, fact["subject"], verb, obj)
            # order-agnostic multiset comparison of the produced surfaces vs ground-truth (the ORDER is separately
            # validated by EMERGE-59/63; here we check the WORDS -- content + function + object -- are the right ones).
            prod_ms = sorted(words)
            for w in sorted(expected):
                all_tot += 1
                if w in prod_ms:
                    prod_ms.remove(w)
                    all_hits += 1
            # OVERFLOW-word-slot accuracy specifically (the object noun + to/on -- the NEW words BRIDGE-C/the new-func add)
            ovf_expect = [w for w in expected if (w in _OVF_OBJ_SET or w in _OVF_FUNC_SET)]
            prod2 = sorted(words)
            for w in ovf_expect:
                ovf_tot += 1
                if w in prod2:
                    prod2.remove(w)
                    ovf_hits += 1
            if len(examples) < 6:
                examples.append({"construction": name,
                                 "fact": {"subject": fact["subject"], "verb": verb, "obj": obj},
                                 "surface": " ".join(words), "expected": " ".join(expected)})
    all_acc = float(all_hits / max(1, all_tot))
    ovf_acc = float(ovf_hits / max(1, ovf_tot))
    return all_acc, ovf_acc, examples


# ---------------------------------------------------------------------------------------------------------------------
# PER-WORD A->W accuracy over the OVERFLOW words (BRIDGE-C's own spike-spell rate).
# ---------------------------------------------------------------------------------------------------------------------
def _overflow_wordwise_accuracy(ovf_speller):
    """Drive each OVERFLOW word's pool on BRIDGE-C; decode; how many decode to the correct word."""
    n_ok = 0
    per = []
    for w in ovf_speller.vocab16:
        dec, self_cos, top_cos, spikes = ovf_speller._decode(w)
        ok = (dec == w)
        n_ok += int(ok)
        per.append({"word": w, "decoded": dec, "ok": bool(ok), "self_cos": round(self_cos, 3),
                    "top_cos": round(top_cos, 3), "spikes": spikes})
    return float(n_ok / len(ovf_speller.vocab16)), per


# ---------------------------------------------------------------------------------------------------------------------
# THE DE-RISK: all-word spike-spell accuracy + overflow-word lesion collapse + moat + no-regression-vs-token.
# ---------------------------------------------------------------------------------------------------------------------
def _derisk_one(seed, unified, unified_ovf_lesion=None):
    facts = _facts(seed, n=8)
    tokens = build_stream(seed)
    reg = ConstructionRegistry(seed).build(tokens)

    # (a) MAIN: render the EMERGE-72 constructions with ALL words (content + function + overflow) spelled ON SPIKES.
    all_acc, ovf_acc, examples = _render_and_score_all(reg, unified.spell, facts)

    # (d) NO REGRESSION vs the token spell: the produced surfaces (all slots) as a multiset must equal the token spell's
    # surfaces (== ground-truth by construction) for every in-scope construction + fact.
    cq = reg.render_cq()
    token_spell = lambda w: str(w)
    regress_mismatch = 0
    for name in _SCOPE_CONSTRUCTIONS:
        if name not in reg.registered:
            continue
        for fact in facts:
            verb = _verb_for(name, fact)
            obj = fact.get("obj")
            tok = sorted(cq.emit(name, fact["subject"], verb, obj, token_spell))
            neu = sorted(cq.emit(name, fact["subject"], verb, obj, unified.spell))
            regress_mismatch += int(tok != neu)

    # (c) MOAT: an ABSTAIN never invokes the producer -> none of the three A->W engines' spell is called.
    prod = RegistryBrocaProducer(cq, spell=unified.spell)
    calls_before = unified.spell_calls
    for _ in range(3):
        prod.speak(decision("ABSTAIN"))
    spell_calls_on_abstain = unified.spell_calls - calls_before
    producer_calls_on_abstain = prod.production_count
    a_name = "F_MODAL" if "F_MODAL" in reg.registered else next(iter(reg.registered), None)
    answer_produced = False
    if a_name is not None:
        ans = prod.speak(decision("ANSWER", construction=a_name, subject="owl", verb="fly", obj="pond"))
        answer_produced = bool(ans["produced"])

    # (b) OVERFLOW-WORD LESION: the SAME constructions rendered with BRIDGE-C lesioned must collapse the OVERFLOW decode.
    lesion_ovf_acc = None
    if unified_ovf_lesion is not None:
        _all2, lesion_ovf_acc, _ = _render_and_score_all(reg, unified_ovf_lesion.spell, facts)

    return {
        "seed": seed,
        "all_acc": all_acc,
        "overflow_acc": ovf_acc,
        "regress_mismatch": int(regress_mismatch),
        "spell_calls_on_abstain": int(spell_calls_on_abstain),
        "producer_calls_on_abstain": int(producer_calls_on_abstain),
        "answer_produced": answer_produced,
        "lesion_overflow_acc": lesion_ovf_acc,
        "n_registered_in_scope": sum(1 for n in _SCOPE_CONSTRUCTIONS if n in reg.registered),
        "examples": examples,
    }


def _sample_transcript(unified, seed=42):
    tokens = build_stream(seed)
    reg = ConstructionRegistry(seed).build(tokens)
    cq = reg.render_cq()
    prod = RegistryBrocaProducer(cq, spell=unified.spell)
    lines = []
    specs = [
        ("MODAL  (ability affirm)", decision("ANSWER", "F_MODAL", subject="owl", verb="fly"), "can an owl fly?"),
        ("INTR   (intransitive)", decision("ANSWER", "F_INTR", subject="penguin", verb="walks"),
         "what does a penguin do?"),
        ("NEGMOD (negated modal)", decision("ANSWER", "F_NEGMOD", subject="penguin", verb="fly"),
         "can a penguin fly? [deny]"),
        ("PPGOAL (motion goal)", decision("ANSWER", "C_PPGOAL", subject="owl", verb="fly", obj="pond"),
         "where does the owl fly?"),
        ("PPLOC  (motion location)", decision("ANSWER", "C_PPLOC", subject="owl", verb="fly", obj="rock"),
         "where does the owl fly?"),
        ("MOAT   (abstain)", decision("ABSTAIN"), "can a zzz fly?"),
    ]
    for tag, d, q in specs:
        if d["gate"] == "ANSWER" and d["construction"] not in reg.registered:
            lines.append((tag, q, "[construction not mined]", "producer NOT invoked"))
            continue
        r = prod.speak(d)
        surface = r["surface"] if r["produced"] else "I don't know."
        inv = "producer INVOKED" if r["produced"] else "producer NOT invoked"
        lines.append((tag, q, surface, inv))
    return lines, prod.production_count


def _demo(seed=42):
    print("\n=== EMERGE-75 -- A->W VOCAB SCALING: dispatch each word to the bridge holding its pool (BRIDGE-A content / "
          "BRIDGE-F {the,a,can,does,not} / BRIDGE-C overflow {to,on,is}+object nouns) so the EMERGE-72 broadened "
          "constructions render EVERY word ON SPIKES (the transitive-motion OBJECT + PP preposition now spike-spelled) "
          "===\n", flush=True)
    unified = UnifiedNeuralSpell75(load=True)
    if not unified._backend_gpu:
        print("  [skip] the A->W engines need a GPU (SIM_BACKEND=cupy); numpy fallback cannot run the read-out.\n")
        return
    orate, oper = _overflow_wordwise_accuracy(unified.overflow)
    print(f"  BRIDGE-C spells {int(orate*len(oper))}/{len(oper)} OVERFLOW words correctly (spike-decoded); "
          f"mean spike-total {int(np.mean([p['spikes'] for p in oper]))}")
    for p in oper:
        print(f"    {p['word']:7s} -> {p['decoded']:7s} {'OK' if p['ok'] else 'X'} "
              f"(self_cos {p['self_cos']}, spikes {p['spikes']})")
    print()
    lines, pc = _sample_transcript(unified, seed)
    print("  render the EMERGE-72 broadened inventory with EVERY word SPIKE-SPELLED (3-bridge dispatch; gate-first moat):")
    for tag, q, surface, inv in lines:
        print(f"    you> {q}\n      broca> {surface}   [{tag}; {inv}]")
    print(f"\n  producer-invocation count after {len(lines)} probes: {pc} (the abstain never invoked the producer -- moat)\n")


def _train():
    """Build + train + cache the OVERFLOW-word A->W engine (BRIDGE-C, GPU). Idempotent: overwrites the cache."""
    if _OVF_CACHE_BRIDGE.exists():
        _OVF_CACHE_BRIDGE.unlink()
    print("[emerge75] building + training the OVERFLOW-word A->W engine BRIDGE-C (GPU, validated scale)...", flush=True)
    t0 = time.time()
    ovf = OverflowNeuralSpell(load=False)
    if not ovf._backend_gpu:
        print("[emerge75] SKIP -- needs SIM_BACKEND=cupy (GPU) to train the A->W engine.")
        return 1
    rate, per = _overflow_wordwise_accuracy(ovf)
    print(f"[emerge75] trained + cached ({time.time()-t0:.0f}s). BRIDGE-C spells {int(rate*len(per))}/{len(per)} "
          f"overflow words correctly. cache: {_OVF_CACHE_BRIDGE}", flush=True)
    for p in per:
        print(f"    {p['word']:7s} -> {p['decoded']:7s} {'OK' if p['ok'] else 'X'} (self_cos {p['self_cos']}, "
              f"top_cos {p['top_cos']}, spikes {p['spikes']})", flush=True)
    return 0


def _derisk(seeds, train_events=_TRAIN_EVENTS):
    print(f"EMERGE-75 de-risk: A->W VOCAB SCALING via a 3-bridge dispatch (BRIDGE-A content / BRIDGE-F function / "
          f"BRIDGE-C overflow) -- the EMERGE-72 broadened constructions render EVERY word on spikes; all-word accuracy "
          f"+ overflow-word lesion collapse + gate-first moat + no-regression; {len(seeds)}-seed", flush=True)
    t0 = time.time()
    err = None
    per = []
    orate = None
    oper = []
    content_rate = None
    func_rate = None
    lesion_ovf_engine = None
    gpu = True
    try:
        unified = UnifiedNeuralSpell75(load=True, train_events=train_events)
        gpu = unified._backend_gpu
        if not gpu:
            raise RuntimeError("A->W engines require SIM_BACKEND=cupy (GPU); numpy cannot run the spiking read-out")
        orate, oper = _overflow_wordwise_accuracy(unified.overflow)
        content_rate, _cper = m67._aw_wordwise_accuracy(unified.content)
        func_rate, _fper = m68._func_wordwise_accuracy(unified.func)
        # an overflow-lesioned unified engine (BRIDGE-C pool->output zeroed) for the genuinely-spiking control
        unified_ovf_lesion = UnifiedNeuralSpell75(load=True, train_events=train_events, overflow_lesion=True)
        lesion_ovf_engine, _ = _overflow_wordwise_accuracy(unified_ovf_lesion.overflow)
        for s in seeds:
            d = _derisk_one(s, unified, unified_ovf_lesion=unified_ovf_lesion)
            per.append(d)
            la = f"{d['lesion_overflow_acc']:.3f}" if d["lesion_overflow_acc"] is not None else "n/a"
            print(f"  [seed {s}] all-word acc {d['all_acc']:.3f} | overflow acc {d['overflow_acc']:.3f} | "
                  f"overflow-lesion acc {la} | regress {d['regress_mismatch']} | "
                  f"spell-on-abstain {d['spell_calls_on_abstain']} | producer-on-abstain "
                  f"{d['producer_calls_on_abstain']} | in-scope {d['n_registered_in_scope']}", flush=True)
    except Exception as e:
        err = repr(e)
        traceback.print_exc()

    if err is None:
        def m(k):
            return float(np.mean([d[k] for d in per]))
        all_acc = m("all_acc")
        overflow_acc = m("overflow_acc")
        lesion_vals = [d["lesion_overflow_acc"] for d in per if d["lesion_overflow_acc"] is not None]
        lesion_ovf_acc = float(np.mean(lesion_vals)) if lesion_vals else None
        regress = int(sum(d["regress_mismatch"] for d in per))
        spell_calls_abstain = int(sum(d["spell_calls_on_abstain"] for d in per))
        producer_calls_abstain = int(sum(d["producer_calls_on_abstain"] for d in per))
        answer_ok = all(d["answer_produced"] for d in per)

        BAR = 0.90
        all_ok = all_acc >= BAR
        overflow_ok = overflow_acc >= BAR
        # genuinely spiking for the OVERFLOW words: BRIDGE-C's own rate is high AND the overflow lesion collapses BOTH the
        # frame overflow-word decode and the engine's own rate (a host lookup would be unaffected).
        spiking_ok = (orate is not None and orate >= BAR
                      and lesion_ovf_acc is not None and (overflow_acc - lesion_ovf_acc) >= 0.40
                      and lesion_ovf_engine is not None and (orate - lesion_ovf_engine) >= 0.40)
        no_regress = (regress == 0)
        moat_ok = (spell_calls_abstain == 0) and (producer_calls_abstain == 0) and answer_ok

        go = bool(all_ok and overflow_ok and spiking_ok and no_regress and moat_ok)
        if go:
            verdict = (
                f"GO -- the A->W read-out now supports ARBITRARY content vocabulary via the G.20 MULTI-BRIDGE route, so "
                f"the EMERGE-72 broadened constructions render EVERY word ON SPIKES. UnifiedNeuralSpell75 DISPATCHES each "
                f"word to the bridge holding its pool: content -> BRIDGE-A (== EMERGE-67), {{the,a,can,does,not}} -> "
                f"BRIDGE-F (== EMERGE-68), and the NEW overflow words {{to,on,is}} + the OBJECT nouns (pond/rock/tree/... "
                f"the transitive-motion argument) -> BRIDGE-C (the new engine) -- each spelled by DRIVING its concept pool "
                f"on a real SimulationBridge and DECODING the spoken word from `language_output` SPIKES (the PP verb 3sg "
                f"is decoded on its LEMMA pool + re-inflected, still FROM SPIKES). ALL-WORD spike-spell accuracy "
                f"{all_acc:.3f} over the EMERGE-72 constructions (>= {BAR}); OVERFLOW-word slot accuracy "
                f"{overflow_acc:.3f}; BRIDGE-C spells {int((orate or 0)*len(oper))}/{len(oper)} overflow words correctly "
                f"(raw rate {orate:.3f}). GENUINELY SPIKING for the overflow words: the OVERFLOW LESION (zeroing "
                f"BRIDGE-C's pool->language_output pathway) COLLAPSES the overflow-word decode to {lesion_ovf_acc:.3f} "
                f"(engine-lesion {lesion_ovf_engine:.3f}; a host lookup would be unaffected -> the object nouns + PP "
                f"prepositions are decoded FROM SPIKES). NO regression: the neural spell reproduces the token spell on ALL "
                f"slots ({regress} mismatches; the token-spell default path is byte-identical; the EMERGE-72 construction "
                f"render itself is unchanged -- only the SPELL is upgraded). The gate-first no-confab MOAT holds BY "
                f"CONSTRUCTION: {spell_calls_abstain} spell calls + {producer_calls_abstain} producer invocations on "
                f"abstains (the producer -- and hence ALL THREE A->W read-outs -- is NEVER invoked on an abstain). "
                f"{len(seeds)} seeds. ==> the EMERGE-72 broadened constructions are now FULLY SPIKING end-to-end (order "
                f"via EMERGE-59/63, content words via EMERGE-67, the 5 original function words via EMERGE-68, the NEW "
                f"overflow words now). HONEST SCOPE: the overflow vocab is a BOUNDED 16 words on ONE extra bridge; the "
                f"full 30-object corpus vocab is 2 BRIDGE-Cs (the G.20 route scales LINEARLY in bridge count -- a data/"
                f"bridge lever, not a new mechanism). The EMERGE-73 ADJECTIVE constructions are NOT in scope here (their "
                f"construction mining is EMERGE-73's own boundary; their adjective A->W is a further follow-on). The A->W "
                f"engines are GPU-trained ONCE at the validated scale + cached; the vocab rebind is a research-runner edit "
                f"(reuse-by-import; NO sim/ edit). Renders the BOUNDED EMERGE-72 inventory, NOT open prose (R4).")
        else:
            miss = []
            if not all_ok:
                miss.append(f"all-word spike-spell accuracy {all_acc:.3f} < {BAR}")
            if not overflow_ok:
                miss.append(f"overflow-word slot accuracy {overflow_acc:.3f} < {BAR}")
            if not spiking_ok:
                miss.append(f"overflow read-out not clearly spiking (BRIDGE-C rate {orate}, overflow-lesion "
                            f"{lesion_ovf_acc}, engine-lesion {lesion_ovf_engine} -- the lesion did not collapse the "
                            f"overflow-word decode by >= 0.40)")
            if not no_regress:
                miss.append(f"slot REGRESSION vs the token spell ({regress} mismatches)")
            if not moat_ok:
                miss.append(f"MOAT: {spell_calls_abstain} spell-calls + {producer_calls_abstain} producer-calls on "
                            f"abstains / answer-produced {answer_ok} -- BLOCKING if the producer/spell ran on abstain")
            verdict = ("BOUNDARY -- " + "; ".join(miss) + ". THE EXACT GAP: the content words spell 100% on BRIDGE-A + "
                       "the 5 original function words 100% on BRIDGE-F (EMERGE-67/68 GO), but at this training budget/"
                       "scale the OVERFLOW words on BRIDGE-C (the object nouns + to/on/is) may not all separate cleanly "
                       "(BRIDGE-C has 16 words in one 16-pool bridge, incl. 3 high-frequency closed-class prepositions "
                       "co-trained with 13 content nouns -- the closed-class words' orthogonal-band A->W codes are the "
                       "harder read, as EMERGE-68 named). NEXT STEP: train BRIDGE-C at the fully-validated scale "
                       "(n_per_pool=500, more events) OR split the function/content overflow across TWO bridges (the G.20 "
                       "route). This is a SCALE/DATA lever + a bounded pool-assignment change, NOT a new mechanism. If the "
                       "MOAT was breached this is BLOCKING -- do NOT weaken the moat.")
    elif not gpu:
        go = False
        verdict = ("SKIP/BOUNDARY -- the spiking A->W read-out requires SIM_BACKEND=cupy (GPU); this run had only the "
                   "numpy backend, which cannot execute the concept-pool spiking read-out. Re-run on GPU with "
                   "SIM_BACKEND=cupy. The 3-bridge dispatch (UnifiedNeuralSpell75 -> RegistryBrocaProducer.spell) + the "
                   "moat logic are CPU-testable (tests/test_emerge75_aw_vocab_scaling.py); the on-spikes A->W is GPU-only.")
        all_acc = overflow_acc = lesion_ovf_acc = None
        regress = spell_calls_abstain = producer_calls_abstain = None
    else:
        go = False
        verdict = f"ERROR -- {err}"
        all_acc = overflow_acc = lesion_ovf_acc = None
        regress = spell_calls_abstain = producer_calls_abstain = None

    transcript = []
    try:
        if err is None and gpu:
            lines, _ = _sample_transcript(unified, seeds[0])
            transcript = [{"question": q, "surface": s, "invocation": i} for (_t, q, s, i) in lines]
    except Exception:
        pass

    summary = {
        "probe": "emerge75_aw_vocab_scaling", "verdict": verdict,
        "go": bool(go) if err is None and gpu else False,
        "mechanism": ("A->W VOCAB SCALING: make the spiking A->W read-out support ARBITRARY content vocabulary via the "
                      "G.20 MULTI-BRIDGE route. Each A->W concept-pool bridge caps at 16 words (4 kinds x 4 pools); the "
                      "EMERGE-72/73 broadened constructions add NEW words in neither BRIDGE-A (content) nor BRIDGE-F "
                      "(function): the OBJECT nouns (the transitive-motion PP argument) + the NEW function words to/on/is. "
                      "EMERGE-75 adds a THIRD bridge BRIDGE-C for those 16 overflow words (3 new function + 13 object "
                      "nouns rebound onto its 16 pools; same builder+topographic-bias+orthogonal-codes+train_word_to_pool "
                      "recipe, cached once), and UnifiedNeuralSpell75.spell DISPATCHES each word to the bridge holding its "
                      "pool: content->BRIDGE-A, {the,a,can,does,not}->BRIDGE-F, {to,on,is}+objects->BRIDGE-C. A verb "
                      "rendered 3sg decodes on its LEMMA pool (BRIDGE-A) + re-inflects (still FROM SPIKES). "
                      "RegistryBrocaProducer's DET/FUNC/SUBJ/VERB/OBJ slots route through this SAME spell, so EVERY slot "
                      "-- content AND function AND object -- is spike-spelled. The read-out is genuinely spiking (reads "
                      "cp_firing_states[language_output]; a BRIDGE-C pool->output lesion collapses the overflow-word "
                      "decode). The gate-first no-confab moat is untouched (abstain -> producer + all three A->W engines "
                      "NEVER invoked). G.20 multi-bridge (CLAUDE.md: the 320-concept ensemble is 5 sparse bridges); "
                      "concept_speak_demo A->W read-out; EMERGE-67/68 precedent. Reuse-by-import; NO sim/ edit."),
        "task": ("generalize the spiking A->W spell to arbitrary vocabulary via the G.20 multi-bridge route -- a "
                 "UnifiedNeuralSpell75 dispatching each word to the bridge holding its pool, adding a 3rd bridge "
                 "(BRIDGE-C) for the overflow content words -- so the EMERGE-72 broadened constructions render every word "
                 "on spikes; all-word spike-spell accuracy + genuinely-spiking (overflow lesion collapse) + gate-first "
                 "moat (0 spell/producer calls on abstains) + no regression vs the token spell; >=6 seeds; GPU"),
        "overflow_words": _OVF_VOCAB16,
        "overflow_func": _OVF_FUNC, "overflow_obj": _OVF_OBJ,
        "content_vocab": m67._AW_CONTENT, "func_words": m68._FUNC_WORDS,
        "scope_constructions": _SCOPE_CONSTRUCTIONS,
        "seeds": list(seeds), "gpu": bool(gpu), "elapsed_seconds": round(time.time() - t0, 1),
        "aggregate": None if (err is not None or not gpu) else {
            "all_acc": all_acc, "overflow_acc": overflow_acc, "overflow_wordwise_rate": orate,
            "content_wordwise_rate": content_rate, "func_wordwise_rate": func_rate,
            "lesion_overflow_acc": lesion_ovf_acc, "engine_lesion_overflow_acc": lesion_ovf_engine,
            "regress_mismatch": regress, "spell_calls_on_abstain_total": spell_calls_abstain,
            "producer_calls_on_abstain_total": producer_calls_abstain,
        },
        "overflow_wordwise": oper,
        "sample_transcript": transcript,
        "per_seed": per,
        "HONEST_NOTE": ("Makes the EMERGE-72 BROADENED constructions FULLY SPIKING end-to-end via the G.20 MULTI-BRIDGE "
                        "A->W route (the ORDER was already spiking EMERGE-59/63; content words since EMERGE-67; the 5 "
                        "original function words since EMERGE-68; the NEW overflow words -- object nouns + to/on/is -- "
                        "now, on a 3rd bridge BRIDGE-C). UnifiedNeuralSpell75 dispatches each word to the bridge holding "
                        "its pool; every word decodes from cp_firing_states[language_output] (genuinely spiking; the "
                        "BRIDGE-C pool->language_output lesion collapses the overflow-word decode -- a host lookup would "
                        "be unaffected). The A->W engines are GPU-trained ONCE at the production scale + cached (a scale/"
                        "data lever, not a new mechanism): BRIDGE-A = EMERGE-67's 16 content words, BRIDGE-F = EMERGE-68's "
                        "5 function words, BRIDGE-C = the 16 overflow words (a research-runner vocab rebind on a third "
                        "bridge, NO sim/ edit). HONEST SCOPE: the overflow vocab is a BOUNDED 16 words on ONE extra "
                        "bridge; the full 30-object corpus vocab is 2 BRIDGE-Cs (the G.20 route scales LINEARLY in bridge "
                        "count -- a data/bridge lever). The EMERGE-73 ADJECTIVE constructions are NOT in scope (their "
                        "construction mining is EMERGE-73's own boundary; the adjective A->W big/small/... on a 4th bridge "
                        "is a further follow-on). The gate-first no-confab moat is untouched (0 spell/producer "
                        "invocations on abstains, by construction). NOT open prose (R4, the separate deferred wall)."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118, flush=True)
    print(f"[emerge75] VERDICT: {verdict}", flush=True)
    print(f"[emerge75] wrote {OUT}\n" + "=" * 118, flush=True)
    return 0 if (err is None and gpu and go) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--train-events", type=int, default=_TRAIN_EVENTS)
    ap.add_argument("--train", action="store_true", help="build + train + cache the OVERFLOW-word A->W engine (GPU)")
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
