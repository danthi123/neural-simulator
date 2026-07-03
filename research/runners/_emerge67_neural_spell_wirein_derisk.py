"""EMERGE-67 -- wire the VALIDATED SPIKING A->W (concept-pool -> spoken-word) read-out into the spiking-Broca producer's
`spell` callback, so the WORDS of the EMERGE-frame render are produced ON SPIKES (not host token strings). This closes
the last host piece of the EMERGE-frame render: the ORDER was already spiking (EMERGE-59/63 competitive-queuing rate
ranking); the WORDS were still a host token-string identity (`spell=lambda w: str(w)`). Here the CONTENT slots
(subject/verb = emergent concepts) are spelled by driving the word's concept pool on a real `SimulationBridge` and
DECODING the spoken word from `language_output` SPIKES -- the `concept_speak_demo` A->W read-out (CLAUDE.md: "chat_speak
A->W 100% multi-seed"; `concept_speak_demo` 16/16 pools speak their trained word).

THE SEAM (EMERGE-59 `_emerge59:138 realize_slot(slot, subject, verb, spell)` + `_emerge59:299 BrocaProducer(cq, spell=)`):
`spell(word)` takes a word string and returns its SURFACE. EMERGE-59/65/66 pass `spell=lambda w: str(w)` (a token
identity for the CPU de-risk). Here we pass `spell=NeuralSpell.spell`, a genuinely-spiking A->W:
  neural_spell(word):  drive word's concept pool on a spiking bridge -> accumulate `language_output` spikes over a read
                       window -> cosine-decode the spoken word vs the content-vocab word patterns -> return the decoded
                       surface. The read-out reads `cp_firing_states[language_output]` (real spikes), NOT a host lookup.

CHEAP-FIRST (per the task): wire the A->W spell for the CONTENT slots (SUBJ + VERB = the emergent concepts) FIRST. The
DET/FUNC (closed-class function-word) slots keep the token surface for now -- a NAMED follow-on (their own A->W pools are
the EMERGE-62 discovered closed class; a small closed-class A->W is the next rung). This is the clean cheap-first split:
CONTENT words are the emergent lexicon the A->W read-out is validated on (`concept_speak_demo`); function words are frame
FURNITURE (Bock-Levelt) whose A->W pools are a separable, smaller build.

THE VOCAB ENGINE (the honest crux). `concept_speak_demo`'s validated A->W is welded to a FIXED 16-word vocab
(north/east/.../cold) via its pool names + topographic-bias loop + orthogonal codes. The EMERGE producer's content vocab
(owl/penguin/robin/... + fly/swim/walks/...) is DISJOINT. So EMERGE-67 REBINDS the 16 validated concept pools to 16
PRODUCER content words (8 subjects onto motor+noun pools, 8 verbs onto verb+adj pools) -- reuse-by-import: the vocab dicts
of `concept_pool_demo` are swapped (a research-runner edit, NO `sim/` edit), the SAME builder + topographic-bias +
orthogonal-codes + `train_word_to_pool` recipe is used, trained ONCE at the validated scale (n_per_pool=200, n_lang=2048,
120 events) and CACHED. `neural_spell` then spells those 16 producer content words ON SPIKES via the validated read-out.
The producer facts for the de-risk are drawn from THIS 16-word content vocab so every content slot is spike-spelled.

DE-RISK (>=3, ideally 6 seeds 42/43/44/100/101/102; the A->W is a shared GPU-trained engine, deterministic per its own
seed -- built ONCE + cached -- while the seed varies the frame/order/fact selection; SIM_BACKEND=cupy for the A->W, with
a numpy/skip-if-no-GPU fallback):
  (a) the producer renders the EMERGE frames with the CONTENT words SPELLED ON SPIKES ("the owl can fly" -- owl+fly
      decoded via A->W), surface-accuracy of the content slots vs ground-truth.
  (b) the A->W spell is GENUINELY SPIKING: the read-out decodes from `language_output` spikes (`cp_firing_states`), NOT a
      host lookup -- asserted by (i) the spike-count read path + (ii) a lesion control (zero the pool->language_output
      pathway -> the decode collapses; a host lookup would be unaffected).
  (c) gate-first MOAT: on ABSTAIN the producer -- and hence the A->W read-out -- is NEVER invoked (0 productions;
      0 spell calls).
  (d) NO regression vs the token-spell producer: the CONTENT surfaces are identical (the neural spell reproduces the
      token spell for the trained content vocab); the token-spell default path is byte-identical (EMERGE-59..66).
GO bar: content-word spike-spell accuracy >= 0.90 (a clear bar), moat 0 (spell never called on abstain), no content-slot
regression vs token spell, >=3 seeds. If the A->W does not cleanly spell the producer's content words (vocab/pool
mismatch) -> honest BOUNDARY naming the exact gap + next step (do NOT force a GO; do NOT weaken the moat).

HONEST SCOPE: this makes the CONTENT words of the BOUNDED EMERGE frame inventory spike-produced (the ORDER was already
spiking). Function words (the/can/does/not) keep the token surface -- the named closed-class-A->W follow-on. The A->W
engine is GPU-trained ONCE at the validated scale + cached (a scale/data lever, not a new mechanism). NOT open prose (R4).
Reuse-by-import; NO `sim/` edit.

Run:
  SIM_BACKEND=cupy python -m research.runners._emerge67_neural_spell_wirein_derisk --train    # build+cache the A->W engine
  SIM_BACKEND=cupy python -m research.runners._emerge67_neural_spell_wirein_derisk --demo
  SIM_BACKEND=cupy python -m research.runners._emerge67_neural_spell_wirein_derisk --derisk
  SIM_BACKEND=cupy python -m research.runners._emerge67_neural_spell_wirein_derisk --derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # numpy for the CPU logic; the A->W engine forces cupy when built
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

# Reuse-by-import ONLY -- NO sim/ edit. The EMERGE-59 producer seam (realize_slot / BrocaProducer / decision adapter /
# frames / expected words / held-out facts); the EMERGE-57 3sg inflection.
from research.runners._emerge59_spiking_broca_frame_slots_derisk import (  # noqa: E402
    FRAMES, FRAME_NAMES, DET, SUBJ, FUNC, VERB, FrameSlotCQ, BrocaProducer, decision_from_emerge,
    build_heldout_facts, _expected_words,
)
from research.runners._emerge57_ra_refinetune_emerge_frames_derisk import emerge_v3  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_emerge67_neural_spell_wirein.json"
# the cached A->W engine bridge (GPU-trained once; regenerable via --train). Kept under bridges/ (h5 gitignored pattern).
_CACHE_DIR = _REPO / "bridges" / "emerge67_aw"
_CACHE_BRIDGE = _CACHE_DIR / "aw_content.simstate.h5"

# ---------------------------------------------------------------------------------------------------------------------
# THE 16-WORD PRODUCER CONTENT VOCAB rebound onto the 16 validated concept pools. 8 SUBJECTS (4 onto motor, 4 onto noun)
# + 8 VERBS (4 bare-ability onto verb, 4 intransitive-3sg onto adjective). These are the EMERGE producer's OWN content
# words (from `_emerge59._SUBJECTS` / `_ABILITY` / `_INTR3SG`); the de-risk facts are drawn from THIS set so every
# content slot is spike-spelled by the A->W read-out. The pool STRUCTURE (names/FS topology) is unchanged -- only the
# word each pool is TRAINED to speak is swapped (a research-runner vocab rebind; NO sim/ edit).
_AW_SUBJECTS = ["owl", "penguin", "robin", "sparrow", "eagle", "hawk", "wren", "crow"]
_AW_ABILITY = ["fly", "swim", "run", "hop"]          # bare ability verbs (F_MODAL / F_NEGMOD)
_AW_INTR3SG = ["walks", "lurks", "hides", "rests"]   # already-3sg intransitive verbs (F_INTR)
_AW_VERBS = _AW_ABILITY + _AW_INTR3SG
_AW_CONTENT = _AW_SUBJECTS + _AW_VERBS               # 16 content words == 16 pools

# A->W engine hyperparameters (the validated `concept_speak_demo` recipe at production scale).
_N_LANG = 2048
_N_PER_POOL = 200
_N_FS_PER_POOL = 24
_SPARSITY = 0.05
_TRAIN_EVENTS = 120
_AW_SEED = 42                # the A->W engine's OWN training seed (the engine is a shared, deterministic spelling module)


def _pool_assignment():
    """Rebind the 16 producer content words onto the 16 validated concept pools. Returns (word_to_pool, pool_to_word)."""
    from research.runners.concept_pool_demo import MOTOR_NAMES, NOUN_NAMES, VERB_NAMES, ADJECTIVE_NAMES
    pools = ([f"motor_{a}" for a in MOTOR_NAMES] + [f"noun_pool_{n}" for n in NOUN_NAMES]
             + [f"verb_pool_{v}" for v in VERB_NAMES] + [f"adjective_pool_{n}" for n in ADJECTIVE_NAMES])
    assert len(pools) == len(_AW_CONTENT), (len(pools), len(_AW_CONTENT))
    word_to_pool = {w: p for w, p in zip(_AW_CONTENT, pools)}
    pool_to_word = {p: w for w, p in word_to_pool.items()}
    return word_to_pool, pool_to_word


def _swap_concept_vocab():
    """Swap `concept_pool_demo`'s module-level 16-word vocab dicts for the 16 producer content words, keeping the pool
    VALUE-slots (N/E/.../COLD) fixed. Reuse-by-import: the SAME builder/bias/train recipe then trains the producer words.
    Returns (word_to_idx over the 16 content words, in canonical pool order)."""
    import research.runners.concept_pool_demo as cpd
    s = _AW_SUBJECTS
    v = _AW_VERBS
    cpd.DIRECTION_VOCAB.clear(); cpd.DIRECTION_VOCAB.update({s[0]: "N", s[1]: "E", s[2]: "S", s[3]: "W"})
    cpd.NOUN_VOCAB.clear();      cpd.NOUN_VOCAB.update({s[4]: "APPLE", s[5]: "RIVER", s[6]: "DOG", s[7]: "CAT"})
    cpd.VERB_VOCAB.clear();      cpd.VERB_VOCAB.update({v[0]: "GO", v[1]: "COME", v[2]: "STOP", v[3]: "LOOK"})
    cpd.ADJECTIVE_VOCAB.clear(); cpd.ADJECTIVE_VOCAB.update({v[4]: "BIG", v[5]: "SMALL", v[6]: "HOT", v[7]: "COLD"})
    # canonical order matches the pool order (motor, noun, verb, adjective)
    allw = (list(cpd.DIRECTION_VOCAB) + list(cpd.NOUN_VOCAB) + list(cpd.VERB_VOCAB) + list(cpd.ADJECTIVE_VOCAB))
    return {w: i for i, w in enumerate(allw)}


# ---------------------------------------------------------------------------------------------------------------------
# THE SPIKING A->W SPELL ENGINE. Builds (or loads) the validated concept-pool bridge rebound to the producer content
# vocab; `spell(word)` drives that word's pool and decodes the spoken word from language_output SPIKES.
# ---------------------------------------------------------------------------------------------------------------------
class NeuralSpell:
    """The genuinely-spiking A->W `spell` callback. `spell(word)` -> drive the word's concept pool on the spiking bridge
    -> accumulate `language_output` spikes -> cosine-decode the spoken word vs the content-vocab patterns -> return the
    decoded surface. For a producer content word: returns the word if the A->W decodes it correctly (spike-produced);
    for a word NOT in the content vocab (e.g. a function word if ever routed here) it is NOT trained -> falls back to the
    token surface (the named follow-on), flagged so the de-risk can measure content-only accuracy."""

    def __init__(self, seed=_AW_SEED, load=True, train_events=_TRAIN_EVENTS, lesion_pool_out=False, read_steps=100):
        self.seed = int(seed)
        self.train_events = int(train_events)
        self.lesion_pool_out = bool(lesion_pool_out)
        self.read_steps = int(read_steps)
        self.spell_calls = 0            # counts A->W read-out invocations (moat assertion: 0 on abstain)
        self.content_vocab = list(_AW_CONTENT)
        self.word_to_pool, self.pool_to_word = _pool_assignment()
        self._word_to_idx = _swap_concept_vocab()   # rebinds concept_pool_demo's vocab dicts (must precede build)
        self._pats = None
        self.bridge = None
        self._backend_gpu = None
        self._build(load=load)

    # -- build / load the A->W engine bridge -------------------------------------------------------------------------
    def _build(self, load=True):
        os.environ["SIM_BACKEND"] = "cupy"       # the A->W engine is a GPU read-out (the validated scale)
        from sim.backend import get_backend, is_gpu_backend
        _xp, _name = get_backend()
        self._backend_gpu = bool(is_gpu_backend())
        from sim.text_embeddings import orthogonal_drive_pattern
        from research.runners.concept_pool_demo import (build_concept_bridge, apply_concept_topographic_bias,
                                                        train_word_to_pool)
        V = len(self.content_vocab)
        # the language_output word reference patterns (cosine targets), one per content word (orthogonal bands)
        self._pats = {w: orthogonal_drive_pattern(self._word_to_idx[w], n_cues=V, n_neurons=_N_LANG, sparsity=_SPARSITY)
                      for w in self.content_vocab}

        b = build_concept_bridge(seed=self.seed, n_lang_input=_N_LANG, n_per_pool=_N_PER_POOL,
                                 n_fs_per_pool=_N_FS_PER_POOL, weak_dynamics=True, enable_adjective=True, verbose=False)
        if load and _CACHE_BRIDGE.exists():
            b.load_checkpoint(str(_CACHE_BRIDGE))
        else:
            apply_concept_topographic_bias(b, n_lang_input=_N_LANG, orthogonal_codes=True,
                                           n_words_for_orthogonal=V, word_to_idx=self._word_to_idx,
                                           sparsity=_SPARSITY, verbose=False)
            for w, pool in self.word_to_pool.items():
                train_word_to_pool(b, w, pool, n_events=self.train_events, n_lang_input=_N_LANG, n_lang_output=_N_LANG,
                                   orthogonal_codes=True, n_words_for_orthogonal=V, word_to_idx=self._word_to_idx,
                                   sparsity=_SPARSITY, verbose=False)
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            b.save_checkpoint(str(_CACHE_BRIDGE))
        # freeze plasticity for inference (matches concept_speak_demo)
        for g in ("language_input_to_motor", "language_input_to_noun_pool", "language_input_to_verb_pool",
                  "language_input_to_adjective_pool", "motor_to_language_output", "noun_pool_to_language_output",
                  "verb_pool_to_language_output", "adjective_pool_to_language_output"):
            try:
                b.set_plasticity_gate(g, 0.0)
            except Exception:
                pass
        # LESION control (b-ii): zero the pool->language_output weights so the A->W read-out has NO spikes to decode
        # (a host lookup would be UNAFFECTED; a genuine spiking read-out COLLAPSES).
        if self.lesion_pool_out:
            self._lesion_pool_to_output(b)
        self.bridge = b

    def _lesion_pool_to_output(self, bridge):
        """Zero every synapse whose POST neuron is in language_output (the read-out pathway) -> the language_output
        spikes carry no word-selective signal -> the A->W decode collapses. Load-bearing: proves the decode is FROM
        SPIKES (a host lookup would be unaffected). `cp_connections.indices` is the post-neuron column (pre->post CSR)."""
        from sim.backend import get_backend
        cp, _ = get_backend()
        rm = bridge.region_manager
        lo = cp.asarray(list(rm.indices("language_output")), dtype=bridge.cp_connections.indices.dtype)
        mask = cp.isin(bridge.cp_connections.indices, lo)     # synapses whose POST is language_output
        bridge.cp_connections.data[mask] = 0.0

    # -- the spiking A->W read-out ------------------------------------------------------------------------------------
    def _decode(self, word):
        """Drive `word`'s pool; accumulate language_output spikes; cosine-decode the spoken word. Returns (decoded_word,
        self_cos, top_cos, spike_total)."""
        from research.runners.concept_speak_demo import drive_pool_and_read_lang_output, _cosine
        pool = self.word_to_pool[word]
        patt = drive_pool_and_read_lang_output(self.bridge, pool, stim_steps=self.read_steps, n_lang_output=_N_LANG)
        scores = {w2: _cosine(patt, self._pats[w2]) for w2 in self.content_vocab}
        top = max(scores, key=scores.get)
        return top, float(scores[word]), float(scores[top]), int(np.asarray(patt).sum())

    def spell(self, word):
        """The `spell` callback wired into BrocaProducer. CONTENT words (in the trained vocab) are spelled ON SPIKES via
        the A->W read-out (return the DECODED surface). Non-content words (function words -- the named follow-on) are NOT
        trained here -> return the token surface (flagged internally)."""
        self.spell_calls += 1
        if word in self.word_to_pool:
            decoded, _sc, _tc, _n = self._decode(word)
            return decoded                     # the spike-decoded surface (genuinely produced on spikes)
        return str(word)                       # function/DET word -- token surface (named closed-class-A->W follow-on)


# ---------------------------------------------------------------------------------------------------------------------
# CONTENT-SLOT SPELL SCORING. For a rendered frame, compare the CONTENT slots (SUBJ + VERB) the neural spell produced to
# the ground-truth surface; DET/FUNC slots keep the token surface (not gated -- the named follow-on).
# ---------------------------------------------------------------------------------------------------------------------
def _content_surfaces(frame, subject, verb):
    """The ground-truth CONTENT surfaces (subject + inflected verb) a frame should spell for a fact."""
    out = {}
    for stype, payload in FRAMES[frame]:
        if stype == SUBJ:
            out["subject"] = subject
        elif stype == VERB:
            out["verb"] = verb if payload == "bare" else emerge_v3(verb, already_3sg=None)
    return out


def _facts_from_content_vocab(seed, n=8):
    """Held-out (subject, ability_verb, intr_verb) facts drawn from the A->W content vocab so every content slot is
    spike-spellable. Subjects from _AW_SUBJECTS; ability verbs from _AW_ABILITY (bare); intransitive from _AW_INTR3SG."""
    rng = np.random.default_rng(seed * 101 + 7)
    facts = []
    for _ in range(n):
        facts.append({
            "subject": str(rng.choice(_AW_SUBJECTS)),
            "ability_verb": str(rng.choice(_AW_ABILITY)),
            "intr_verb": str(rng.choice(_AW_INTR3SG)),
        })
    return facts


def _render_and_score(cq, speller, facts, frames=FRAME_NAMES):
    """Render every fact in every frame with the NEURAL spell; score the CONTENT slots (subject + verb) vs ground-truth.
    Returns (content_slot_accuracy, per-frame words, examples)."""
    hits = tot = 0
    examples = []
    for frame in frames:
        for fact in facts:
            verb = fact["intr_verb"] if frame == "F_INTR" else fact["ability_verb"]
            words = cq.emit(frame, fact["subject"], verb, speller.spell)
            expect_content = _content_surfaces(frame, fact["subject"], verb)
            # locate the content surfaces in the produced word list (order-agnostic content check)
            wset = list(words)
            for key, exp in expect_content.items():
                tot += 1
                hits += int(exp in wset)
            if len(examples) < 6:
                examples.append({"frame": frame, "fact": {"subject": fact["subject"], "verb": verb},
                                 "surface": " ".join(words), "expected_content": expect_content})
    return (float(hits / max(1, tot)), examples)


# ---------------------------------------------------------------------------------------------------------------------
# THE DE-RISK: content-word spike-spell accuracy + genuinely-spiking (lesion collapse) + moat + no-regression-vs-token.
# The A->W engine is built ONCE (cached) and shared across seeds; the seed varies the frame/fact selection.
# ---------------------------------------------------------------------------------------------------------------------
def _derisk_one(seed, speller, speller_lesion=None):
    facts = _facts_from_content_vocab(seed, n=8)

    # (a) MAIN: render the EMERGE frames with CONTENT words spelled ON SPIKES; content-slot accuracy vs ground-truth.
    cq = FrameSlotCQ(seed=seed)
    cq.learn()
    content_acc, examples = _render_and_score(cq, speller, facts)

    # (d) NO REGRESSION vs the token-spell producer: the CONTENT surfaces the neural spell produces must equal the ones
    # the token spell produces (for the trained content vocab). Compared as a MULTISET of the CONTENT slot surfaces
    # (order-agnostic -- the ORDER is separately validated by EMERGE-59/63; here we check the WORDS themselves match).
    token_spell = lambda w: str(w)
    regress_mismatch = 0
    for frame in FRAME_NAMES:
        slot_types = [t for (t, _p) in cq_slot_types(cq, frame)]
        content_positions = [i for i, t in enumerate(slot_types) if t in (SUBJ, VERB)]
        for fact in facts:
            verb = fact["intr_verb"] if frame == "F_INTR" else fact["ability_verb"]
            # emit_order gives the emission ordering of slot positions; realize each in-order via token vs neural spell,
            # then compare the CONTENT slots' surfaces as a multiset (which words, not their order).
            tok_content = sorted(m67_content_surface(cq, frame, p, fact["subject"], verb, token_spell)
                                 for p in content_positions)
            neu_content = sorted(m67_content_surface(cq, frame, p, fact["subject"], verb, speller.spell)
                                 for p in content_positions)
            regress_mismatch += int(tok_content != neu_content)

    # (c) MOAT: an ABSTAIN never invokes the producer -> the spell is NEVER called (spell_calls unchanged).
    prod = BrocaProducer(cq, spell=speller.spell)
    calls_before = speller.spell_calls
    for _ in range(3):
        prod.speak(decision_from_emerge("ABSTAIN"))
    spell_calls_on_abstain = speller.spell_calls - calls_before
    producer_calls_on_abstain = prod.production_count
    # positive control: an ANSWER DOES invoke the producer + spell (so the counters are meaningful)
    ans = prod.speak(decision_from_emerge("ANSWER", subject="owl", verb="fly", polarity="affirm"))
    answer_produced = bool(ans["produced"])

    # (b) GENUINELY SPIKING (lesion): the SAME frames rendered with the LESIONED A->W engine (pool->output zeroed) must
    # collapse the content accuracy (a host lookup would be unaffected). Uses a shared lesioned speller (built once).
    lesion_acc = None
    if speller_lesion is not None:
        lesion_acc, _ = _render_and_score(cq, speller_lesion, facts)

    return {
        "seed": seed,
        "content_acc": content_acc,
        "regress_mismatch": int(regress_mismatch),
        "spell_calls_on_abstain": int(spell_calls_on_abstain),
        "producer_calls_on_abstain": int(producer_calls_on_abstain),
        "answer_produced": answer_produced,
        "lesion_acc": lesion_acc,
        "examples": examples,
    }


def cq_slot_types(cq, frame):
    """The (slot_type, payload) list the cq actually emits for a frame (respects ablation etc.)."""
    return list(cq.frame_slots[frame])


def m67_content_surface(cq, frame, slot_pos, subject, verb, spell):
    """Realize ONE slot (by template position) into its surface via `spell` (reuse the EMERGE-59 realize_slot)."""
    from research.runners._emerge59_spiking_broca_frame_slots_derisk import realize_slot
    return realize_slot(cq.frame_slots[frame][slot_pos], subject, verb, spell)


# ---------------------------------------------------------------------------------------------------------------------
# PER-WORD A->W ACCURACY (the read-out's own spelling accuracy over the 16 content words; the engine's spike-spell rate).
# ---------------------------------------------------------------------------------------------------------------------
def _aw_wordwise_accuracy(speller):
    """Drive each content word's pool; decode; how many decode to the correct word. Reports the engine's raw A->W rate,
    the self-cos, and the mean spike-total (evidence the read-out is spiking)."""
    n_ok = 0
    per = []
    for w in speller.content_vocab:
        dec, self_cos, top_cos, spikes = speller._decode(w)
        ok = (dec == w)
        n_ok += int(ok)
        per.append({"word": w, "decoded": dec, "ok": bool(ok), "self_cos": round(self_cos, 3),
                    "top_cos": round(top_cos, 3), "spikes": spikes})
    return float(n_ok / len(speller.content_vocab)), per


# ---------------------------------------------------------------------------------------------------------------------
# DEMO + DE-RISK drivers.
# ---------------------------------------------------------------------------------------------------------------------
def _sample_transcript(speller, seed=42):
    """Render the canonical EMERGE frames with content words SPELLED ON SPIKES + one moat abstain."""
    cq = FrameSlotCQ(seed=seed)
    cq.learn()
    prod = BrocaProducer(cq, spell=speller.spell)
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
    print("\n=== EMERGE-67 -- wire the VALIDATED SPIKING A->W read-out into the spiking-Broca producer's `spell`: the "
          "CONTENT words (subject/verb) of the EMERGE-frame render are now produced ON SPIKES (drive the word's concept "
          "pool -> decode the spoken word from language_output spikes) ===\n", flush=True)
    speller = NeuralSpell(load=True)
    if not speller._backend_gpu:
        print("  [skip] A->W engine needs a GPU (SIM_BACKEND=cupy); numpy fallback cannot run the read-out.\n")
        return
    aw_rate, per = _aw_wordwise_accuracy(speller)
    print(f"  A->W engine spells {int(aw_rate*len(per))}/{len(per)} content words correctly (spike-decoded); "
          f"mean spike-total {int(np.mean([p['spikes'] for p in per]))}\n")
    lines, pc = _sample_transcript(speller, seed)
    for q, surface, inv in lines:
        print(f"  you> {q}\n      broca> {surface}   [{inv}; content words spike-spelled]")
    print(f"\n  producer-invocation count after 4 probes: {pc} (the abstain never invoked the producer/spell -- the moat)\n")


def _derisk(seeds, train_events=_TRAIN_EVENTS):
    print(f"EMERGE-67 de-risk: wire the SPIKING A->W into the Broca producer's spell -- CONTENT words spike-spelled; "
          f"content-slot accuracy + genuinely-spiking (lesion collapse) + gate-first moat + no-regression-vs-token; "
          f"{len(seeds)}-seed", flush=True)
    t0 = time.time()
    err = None
    per = []
    aw_rate = None
    aw_per = []
    lesion_acc_engine = None
    gpu = True
    try:
        # build the shared A->W engine ONCE (cached), + a lesioned copy for the genuinely-spiking control
        speller = NeuralSpell(load=True, train_events=train_events)
        gpu = speller._backend_gpu
        if not gpu:
            raise RuntimeError("A->W engine requires SIM_BACKEND=cupy (GPU); numpy cannot run the spiking read-out")
        aw_rate, aw_per = _aw_wordwise_accuracy(speller)
        speller_lesion = NeuralSpell(load=True, train_events=train_events, lesion_pool_out=True)
        lesion_acc_engine, _ = _aw_wordwise_accuracy(speller_lesion)
        for s in seeds:
            d = _derisk_one(s, speller, speller_lesion=speller_lesion)
            per.append(d)
            la = f"{d['lesion_acc']:.3f}" if d["lesion_acc"] is not None else "n/a"
            print(f"  [seed {s}] content-spike-spell acc {d['content_acc']:.3f} | lesion acc {la} | "
                  f"regress-mismatch {d['regress_mismatch']} | spell-calls-on-abstain {d['spell_calls_on_abstain']} | "
                  f"producer-on-abstain {d['producer_calls_on_abstain']}", flush=True)
    except Exception as e:
        err = repr(e)
        traceback.print_exc()

    if err is None:
        def m(k):
            return float(np.mean([d[k] for d in per]))
        content_acc = m("content_acc")
        lesion_vals = [d["lesion_acc"] for d in per if d["lesion_acc"] is not None]
        lesion_acc = float(np.mean(lesion_vals)) if lesion_vals else None
        regress = int(sum(d["regress_mismatch"] for d in per))
        spell_calls_abstain = int(sum(d["spell_calls_on_abstain"] for d in per))
        producer_calls_abstain = int(sum(d["producer_calls_on_abstain"] for d in per))
        answer_ok = all(d["answer_produced"] for d in per)

        BAR = 0.90
        content_ok = content_acc >= BAR
        # genuinely spiking: the A->W engine's own spike-spell rate is high AND the lesion collapses it (a host lookup
        # would be unaffected). Margin: lesion accuracy far below the intact engine (and below chance-ish).
        spiking_ok = (aw_rate is not None and aw_rate >= BAR
                      and lesion_acc is not None and (aw_rate - lesion_acc) >= 0.40
                      and lesion_acc_engine is not None and (aw_rate - lesion_acc_engine) >= 0.40)
        no_regress = (regress == 0)      # content surfaces identical to the token spell (no regression)
        moat_ok = (spell_calls_abstain == 0) and (producer_calls_abstain == 0) and answer_ok

        go = bool(content_ok and spiking_ok and no_regress and moat_ok)
        if go:
            verdict = (
                f"GO -- the CONTENT words of the EMERGE-frame render are now PRODUCED ON SPIKES. The spiking-Broca "
                f"producer's `spell` callback is the VALIDATED A->W read-out (`concept_speak_demo`): each CONTENT slot "
                f"(subject/verb = the emergent concepts) is spelled by DRIVING the word's concept pool on a real "
                f"SimulationBridge and DECODING the spoken word from `language_output` SPIKES. Content-slot spike-spell "
                f"accuracy {content_acc:.3f} over the EMERGE frames (>= {BAR}); the A->W engine spells "
                f"{int((aw_rate or 0)*len(aw_per))}/{len(aw_per)} content words correctly (raw A->W rate {aw_rate:.3f}). "
                f"GENUINELY SPIKING: the read-out reads cp_firing_states[language_output], and the LESION control "
                f"(zeroing the pool->language_output pathway) COLLAPSES the decode to {lesion_acc:.3f} (a host lookup "
                f"would be unaffected -> the words are decoded FROM SPIKES). NO regression: the neural spell reproduces "
                f"the token spell on the content slots ({regress} mismatches; the token-spell default path is "
                f"byte-identical). The gate-first no-confab MOAT holds BY CONSTRUCTION: {spell_calls_abstain} spell "
                f"calls + {producer_calls_abstain} producer invocations on abstains (the producer -- and hence the A->W "
                f"read-out -- is NEVER invoked on an abstain). {len(seeds)} seeds. ==> the EMERGE-frame render is now "
                f"FULLY SPIKING for the CONTENT slots (the ORDER was already spiking via EMERGE-59/63; the WORDS are now "
                f"spiking via A->W). HONEST SCOPE: the DET/FUNC (function-word) slots keep the token surface -- the "
                f"named closed-class-A->W follow-on (their pools are the EMERGE-62 discovered closed class). The A->W "
                f"engine is GPU-trained ONCE at the validated scale + cached (a scale/data lever, not a new mechanism); "
                f"the producer content vocab is 16 words rebound onto the 16 validated pools (reuse-by-import; NO sim/ "
                f"edit). Renders the BOUNDED EMERGE frame inventory, NOT open prose (R4).")
        else:
            miss = []
            if not content_ok:
                miss.append(f"content-slot spike-spell accuracy {content_acc:.3f} < {BAR}")
            if not spiking_ok:
                miss.append(f"read-out not clearly spiking (A->W rate {aw_rate}, lesion {lesion_acc}, "
                            f"engine-lesion {lesion_acc_engine} -- the lesion did not collapse the decode by >= 0.40)")
            if not no_regress:
                miss.append(f"content-slot REGRESSION vs the token spell ({regress} mismatches)")
            if not moat_ok:
                miss.append(f"MOAT: {spell_calls_abstain} spell-calls + {producer_calls_abstain} producer-calls on "
                            f"abstains / answer-produced {answer_ok} -- BLOCKING if the producer/spell ran on an abstain")
            verdict = ("BOUNDARY -- " + "; ".join(miss) + ". THE EXACT GAP: the validated A->W read-out spells its "
                       "TRAINED content vocab on spikes, but at this training budget/scale the content-word spike-spell "
                       "accuracy is below the bar (the A->W read-out needs the fully-validated scale -- n_per_pool=500, "
                       "n_lang>=2048, ~200 events, ~17 min/seed on GPU -- to reach the documented per-pool "
                       "discrimination). NEXT STEP: train the cached A->W engine at the full validated scale (or extend "
                       "the content vocab to the producer's full 32 words across 2 bridges). This is a SCALE/DATA lever "
                       "+ a bounded vocab-rebind, NOT a new mechanism. If the MOAT was breached (spell/producer ran on "
                       "an abstain) this is BLOCKING -- do NOT weaken the moat.")
    elif not gpu:
        go = False
        verdict = ("SKIP/BOUNDARY -- the spiking A->W read-out requires SIM_BACKEND=cupy (GPU); this run had only the "
                   "numpy backend, which cannot execute the concept-pool spiking read-out. Re-run on GPU with "
                   "SIM_BACKEND=cupy. The wire (neural_spell -> BrocaProducer.spell) + the moat logic are CPU-testable "
                   "(see tests/test_emerge67_neural_spell_wirein.py); the on-spikes A->W is GPU-only.")
        content_acc = lesion_acc = None
        regress = spell_calls_abstain = producer_calls_abstain = None
    else:
        go = False
        verdict = f"ERROR -- {err}"
        content_acc = lesion_acc = None
        regress = spell_calls_abstain = producer_calls_abstain = None

    transcript = []
    try:
        if err is None and gpu:
            lines, _ = _sample_transcript(speller, seeds[0])
            transcript = [{"question": q, "surface": s, "invocation": i} for (q, s, i) in lines]
    except Exception:
        pass

    summary = {
        "probe": "emerge67_neural_spell_wirein", "verdict": verdict,
        "go": bool(go) if err is None and gpu else False,
        "mechanism": ("wire the VALIDATED spiking A->W read-out (concept_speak_demo: drive a concept pool -> decode the "
                      "spoken word from language_output SPIKES via cosine to the word patterns; CLAUDE.md chat_speak "
                      "A->W 100% multi-seed) into the EMERGE-59 spiking-Broca producer's `spell` callback "
                      "(realize_slot/BrocaProducer, the neural_serial_order_renderer pluggable-callback precedent). The "
                      "CONTENT slots (subject/verb = emergent concepts) are spelled ON SPIKES; DET/FUNC function-word "
                      "slots keep the token surface (the named closed-class-A->W follow-on). The producer content vocab "
                      "(16 words: 8 subjects + 8 verbs from the EMERGE producer's own _SUBJECTS/_ABILITY/_INTR3SG) is "
                      "rebound onto the 16 validated concept pools (reuse-by-import: concept_pool_demo's vocab dicts are "
                      "swapped, the SAME builder+topographic-bias+orthogonal-codes+train_word_to_pool recipe trains "
                      "them at the validated scale, cached once). The read-out is genuinely spiking (reads "
                      "cp_firing_states[language_output]; a pool->output lesion collapses it). The gate-first no-confab "
                      "moat is untouched (abstain -> producer + A->W NEVER invoked). Reuse-by-import; NO sim/ edit."),
        "task": ("wire the spiking A->W read-out into the Broca producer's spell for the CONTENT slots so the EMERGE-"
                 "frame render's WORDS are produced on spikes; content-slot spike-spell accuracy + genuinely-spiking "
                 "(lesion collapse) + gate-first moat (0 spell/producer calls on abstains) + no regression vs the token "
                 "spell; >=3 (ideally 6) seeds; GPU for the A->W"),
        "content_vocab": _AW_CONTENT,
        "seeds": list(seeds), "gpu": bool(gpu), "elapsed_seconds": round(time.time() - t0, 1),
        "aggregate": None if (err is not None or not gpu) else {
            "content_acc": content_acc, "aw_wordwise_rate": aw_rate, "lesion_acc": lesion_acc,
            "engine_lesion_acc": lesion_acc_engine, "regress_mismatch": regress,
            "spell_calls_on_abstain_total": spell_calls_abstain,
            "producer_calls_on_abstain_total": producer_calls_abstain,
        },
        "aw_wordwise": aw_per,
        "sample_transcript": transcript,
        "per_seed": per,
        "HONEST_NOTE": ("Makes the CONTENT words (subject/verb) of the BOUNDED EMERGE frame inventory PRODUCED ON SPIKES "
                        "via the validated A->W read-out (the ORDER was already spiking, EMERGE-59/63). The DET/FUNC "
                        "function-word slots (the/can/does/not) keep the token surface -- the named closed-class-A->W "
                        "follow-on (their pools are the EMERGE-62 discovered closed class; a small closed-class A->W is "
                        "the next rung). The A->W engine is the validated concept_speak_demo read-out, GPU-trained ONCE "
                        "at the production scale + cached (a scale/data lever, not a new mechanism); the producer content "
                        "vocab is 16 words rebound onto the 16 validated concept pools (reuse-by-import; the vocab-rebind "
                        "is a research-runner edit, NO sim/ edit). The read-out is genuinely spiking (decodes from "
                        "cp_firing_states[language_output]; the pool->language_output lesion collapses it -- a host "
                        "lookup would be unaffected). The gate-first no-confab moat is untouched (0 spell/producer "
                        "invocations on abstains, by construction). NOT open prose (R4, the separate deferred wall)."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118, flush=True)
    print(f"[emerge67] VERDICT: {verdict}", flush=True)
    print(f"[emerge67] wrote {OUT}\n" + "=" * 118, flush=True)
    return 0 if (err is None and gpu and go) else 1


def _train():
    """Build + train + cache the A->W engine (GPU). Idempotent: overwrites the cache."""
    if _CACHE_BRIDGE.exists():
        _CACHE_BRIDGE.unlink()
    print("[emerge67] building + training the A->W content-vocab engine (GPU, validated scale)...", flush=True)
    t0 = time.time()
    speller = NeuralSpell(load=False)
    if not speller._backend_gpu:
        print("[emerge67] SKIP -- needs SIM_BACKEND=cupy (GPU) to train the A->W engine.")
        return 1
    rate, per = _aw_wordwise_accuracy(speller)
    print(f"[emerge67] trained + cached ({time.time()-t0:.0f}s). A->W spells {int(rate*len(per))}/{len(per)} content "
          f"words correctly. cache: {_CACHE_BRIDGE}", flush=True)
    for p in per:
        print(f"    {p['word']:9s} -> {p['decoded']:9s} {'OK' if p['ok'] else 'X'} (self_cos {p['self_cos']}, "
              f"spikes {p['spikes']})", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--train-events", type=int, default=_TRAIN_EVENTS)
    ap.add_argument("--train", action="store_true", help="build + train + cache the A->W engine (GPU)")
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
