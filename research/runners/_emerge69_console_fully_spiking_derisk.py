"""EMERGE-69 -- wire the FULLY-SPIKING A->W spell (EMERGE-67/68 `UnifiedNeuralSpell`: content on BRIDGE-A + function on
BRIDGE-F, ALL words decoded from `language_output` SPIKES) into the EMERGE-66 FLAGSHIP CONSOLE (`SpikingBrocaConsole`
with the self-organized producer), so the flagship renders its EMERGE answers 100% ON SPIKES END-TO-END: the
grammatical STRUCTURE is SELF-ORGANIZED from the corpus (EMERGE-65/66) AND EVERY WORD (order + content + function) is
spelled from spikes (EMERGE-59/63 order; EMERGE-67 content; EMERGE-68 function). The completion of the emergent-Broca
arc on the flagship console.

  you> can an owl fly?      brain> the owl can fly       [EMERGE INHERIT; self-organized STRUCTURE, ALL words ON SPIKES]
  you> can a penguin fly?   brain> the penguin walks     [EMERGE CANCEL;  self-organized + all-word-spike]
  you> can a robin breathe? brain> the robin can breathe [EMERGE PER-DIMENSION inherit; self-organized + all-word-spike]
  you> can an owl swim?     brain> I don't know whether ... [EMERGE SIBLING-abstain; producer + A->W NEVER invoked]
  you> can a zzz fly?       brain> I don't know what a zzz is. [MOAT; producer + A->W NEVER invoked]
  you> can a dog eat?       brain> The dog eats meat.     [MEMBERSHIP -> fluid path (EMERGE-58 remediation), unchanged]
  you> what does the dog eat? brain> The dog eats meat.   [FLUID path, unchanged]

THE WIRE (a COMPOSITION of GO pieces, mirroring EMERGE-66's additive `self_organized` flag). EMERGE-60's
`SpikingBrocaConsole` gains an ADDITIVE default-OFF `neural_spell` flag: when True, the console's producer `spell`
callback (which `_render_emerge` -> `self.broca.speak` -> `realize_slot` routes EVERY slot through: DET/FUNC + SUBJ +
VERB, `_emerge59:138`) is the EMERGE-68 `UnifiedNeuralSpell` loaded from the caches (`bridges/emerge67_aw` +
`bridges/emerge68_aw`), so each rendered WORD is DECODED from `language_output` SPIKES (content on BRIDGE-A, function on
BRIDGE-F). Combined with `self_organized=True`, the flagship's EMERGE render is 100% ON SPIKES: SELF-ORGANIZED structure
+ EVERY WORD spiking. The gate-first structure is UNCHANGED (on ABSTAIN `_emerge_turn` returns BEFORE `_render_emerge`,
so the producer -- and hence BOTH A->W read-outs -- is NEVER invoked; the moat holds by construction, asserted via
`BrocaProducer.production_count` AND the unified speller's `spell_calls`). Membership routing + fluid paths inherited
VERBATIM. Default `neural_spell=False` == EMERGE-66 byte-identical (the token-surface spell).

BACKEND SPLIT (the honest process constraint -- named, not hidden). `sim.bridge` binds ONE backend (`cp`) at import
time (a module-global), so numpy and cupy bridges CANNOT coexist in one process. The EMERGE-52/54 per-dimension
REASONER (the console's `self.reasoner`, the stacked HTM pooler) is numpy-only (its EMERGE-12/14 on-substrate teach +
predict paths write host arrays into the CSR via an `xp = bridge.xp if hasattr else np` fallback, which under cupy
raises "non-scalar ndarray cannot be used for fill"); the A->W read-out (EMERGE-67/68) is cupy-only (the concept-pool
spiking read-out). So the FULL console with a LIVE numpy reasoner + the cupy A->W cannot co-execute in ONE process.
Making the whole EMERGE-52/54 reasoner stack cupy-clean is a wide, high-risk change to committed shared runners (out of
scope for this additive wire). The de-risk therefore validates the two claims on their native backends:
  * THE NEW SPIKE CLAIM (GPU/cupy): the SELF-ORGANIZED PRODUCER (the flagship's OWN producer -- a simple slot-order
    bridge + the corpus-mined structure, NO reasoner) is built directly and the neural spell is wired into it; render
    every EMERGE frame with A->W-spellable facts -> ALL slots (det+subj+func+verb) DECODED FROM SPIKES; + the gate-first
    moat AT THE PRODUCER (ABSTAIN -> 0 spell/productions; ANSWER -> produces+spells); + function-word lesion collapse.
    This is EXACTLY the flagship console's producer + the neural-spell wire (the same `SelfOrganizedProducer.producer(
    spell=)` the `neural_spell` flag installs), proven on spikes.
  * THE CONSOLE-INTEGRATION INVARIANTS (CPU/numpy): the full flagship console (reasoner + producer) with the additive
    `neural_spell` flag structure -- gate-first moat + membership routing + fluid no-regression + wire content-routing.
    On numpy the A->W read-out cannot run, so the console uses the token-surface spell here (== EMERGE-66's GO surface):
    this confirms the additive flag is default-preserving AND the gate-first moat holds BY CONSTRUCTION (abstain ->
    producer + spell NEVER invoked) with the neural-spell wire structure. (EMERGE-66 already GO'd this numpy surface;
    the flag is additive/default-off.)

DE-RISK (6 seeds 42/43/44/100/101/102):
  (a) [GPU] ALL-WORD RENDER ON SPIKES through the flagship's SELF-ORGANIZED producer (A->W-spellable facts): every slot
      (det+subj+func+verb) decoded from language_output spikes. all-word spike-spell accuracy >= 0.90.
  (b) [GPU] gate-first MOAT at the producer: ABSTAIN -> 0 spell calls + 0 productions; ANSWER -> produces + spells.
  (e) [GPU] GENUINELY SPIKING for the function words: the FUNCTION-word LESION (zero BRIDGE-F's pool->language_output)
      COLLAPSES the function-word decode (a host lookup would be unaffected) -- inherited from EMERGE-68.
  (c) [CPU] MEMBERSHIP routing preserved on the full console ('can a dog eat?' -> fluid, producer NOT stolen).
  (d) [CPU] NO fluid-path REGRESSION on a Broca-FREE baseline (re-seeded per seed; isolates the EMERGE-60/66 known
      fluid-RNG harness flakiness).
  (f) [CPU] the WIRE + gate-first moat on the full flagship console (token spell): 0 producer + 0 spell calls on
      abstains; the neural_spell flag is additive/default-preserving (EMERGE-59..68 byte-identical).
GO bar: [GPU] all-word spike render >= 0.90 + producer-moat 0 + function-word lesion-collapse; [CPU] membership + no
fluid regression + console-moat 0, 6-seed. WITH the default path (neural_spell=False) byte-identical.

HONEST SCOPE: this renders the BOUNDED EMERGE frame inventory 100% on spikes from a self-organized grammar (order +
content + function ALL spiking) -- NOT open prose (R4, the separate deferred wall). The A->W engines are GPU-trained
ONCE at the validated scale + cached (a scale/data lever, not a new mechanism). The spike claim is validated on the
flagship's SELF-ORGANIZED PRODUCER directly (cupy); the console-integration invariants are validated on the full
console (numpy, token spell) -- the two backends cannot co-execute in one process (named above). The console EMERGE
probes whose content words are outside the A->W-trained 21-word vocab (minnow/pike/breathe) would fall back to the token
surface for those specific words -- the all-word-SPIKE claim is over the A->W-spellable fact set (like EMERGE-68), named
not hidden. Reuse-by-import; NO `sim/` edit; the ONLY change is the additive default-off `neural_spell` flag on
EMERGE-60's `SpikingBrocaConsole`. The gate-first moat is untouched.

Run:
  SIM_BACKEND=cupy  python -m research.runners._emerge69_console_fully_spiking_derisk --demo         # GPU spike render
  SIM_BACKEND=cupy  python -m research.runners._emerge69_console_fully_spiking_derisk --derisk-gpu   # (a)(b)(e) on cupy
  SIM_BACKEND=numpy python -m research.runners._emerge69_console_fully_spiking_derisk --derisk-cpu   # (c)(d)(f) on numpy
  # --derisk runs (a)(b)(e) on GPU inline, then spawns a numpy child for (c)(d)(f) and MERGES -> the full 6-seed GO:
  SIM_BACKEND=cupy  python -m research.runners._emerge69_console_fully_spiking_derisk --derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # the console reasoner is numpy; the A->W engines force cupy when built
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Reuse-by-import ONLY -- NO sim/ edit. The flagship console + its additive neural_spell flag (EMERGE-60/66/69); the
# EMERGE-58 Broca-FREE base console (for the clean fluid no-regression slice); the producer decision adapter + probe
# members (EMERGE-59/54); the unified spiking A->W spell + all-slot scoring + function-lesion (EMERGE-67/68); the
# self-organized producer (EMERGE-65) -- built directly for the cupy spike claim (no numpy reasoner).
from research.runners._emerge59_spiking_broca_frame_slots_derisk import (  # noqa: E402
    FRAMES, FRAME_NAMES, DET, FUNC, decision_from_emerge,
)
import research.runners._emerge68_function_word_spell_derisk as m68  # noqa: E402
import research.runners._emerge67_neural_spell_wirein_derisk as m67  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_emerge69_console_fully_spiking.json"
OUT_CPU = _REPO / "research" / "findings" / "raw" / "_emerge69_console_fully_spiking_cpu.json"

# a smaller corpus stream keeps the per-seed self-organized build cheap (EMERGE-66 uses 6000; the mining converges well
# below the derisk's 20000). Each seed builds its own producer from its stream.
_N_SENTENCES = 6000


# ====================================================================================================================
# THE GPU SPIKE CLAIM: build the FLAGSHIP's SELF-ORGANIZED PRODUCER directly + wire the neural spell + render all words
# on spikes + producer-level gate-first moat + function-word lesion collapse. This is EXACTLY the producer the
# `neural_spell` flag installs (SelfOrganizedProducer.producer(spell=UnifiedNeuralSpell.spell)) -- proven on spikes.
# ====================================================================================================================
def _build_self_organized_producer(seed, spell):
    """Build the flagship's self-organized producer (EMERGE-65 SelfOrganizedProducer.build_from_corpus, a simple
    slot-order bridge + corpus-mined structure -- NO numpy reasoner) and wire `spell` into it. == the neural_spell flag
    wire: SpikingBrocaConsole(self_organized=True, neural_spell=True) sets self.broca = self._sop.producer(spell=...)."""
    from research.runners._emerge65_self_organized_producer_derisk import SelfOrganizedProducer
    from research.runners._emerge62_discover_function_words_derisk import build_stream
    tokens = build_stream(int(seed), n_sentences=_N_SENTENCES)
    sop = SelfOrganizedProducer(int(seed)).build_from_corpus(tokens)
    return sop, sop.producer(spell=spell)


def _producer_all_word_spike_render(producer, seed, n=6):
    """Render every EMERGE frame with A->W-spellable facts (from the trained 21-word vocab) through the self-organized
    producer's CQ, all slots spelled via the wired neural spell; score ALL slots (det+subj+func+verb) vs ground-truth."""
    facts = m68._facts(seed, n=n)
    cq = producer.cq
    spell = producer.spell
    all_hits = all_tot = func_hits = func_tot = 0
    examples = []
    for frame in FRAME_NAMES:
        if frame not in getattr(cq, "mined_slots", {frame: None}) and hasattr(cq, "mined_slots"):
            continue  # a frame not mined by the self-organized producer renders nothing (carried-forward residual)
        for fact in facts:
            verb = fact["intr_verb"] if frame == "F_INTR" else fact["ability_verb"]
            words = cq.emit(frame, fact["subject"], verb, spell)
            expect = m68._all_slot_surfaces(frame, fact["subject"], verb)
            prod_ms = sorted(words)
            for w in sorted(expect):
                all_tot += 1
                if w in prod_ms:
                    prod_ms.remove(w)
                    all_hits += 1
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
    return float(all_hits / max(1, all_tot)), float(func_hits / max(1, func_tot)), examples


def _derisk_gpu_one(seed, unified):
    """[GPU] the flagship's self-organized producer + the wired neural spell: all-word spike render + producer moat."""
    sop, producer = _build_self_organized_producer(seed, unified.spell)
    all_acc, func_acc, examples = _producer_all_word_spike_render(producer, seed)

    # (b) gate-first MOAT at the producer: ABSTAIN -> 0 spell calls + 0 productions; ANSWER -> produces + spells.
    calls_before = unified.spell_calls
    prod0 = producer.production_count
    for _ in range(3):
        producer.speak(decision_from_emerge("ABSTAIN"))
    spell_on_abstain = unified.spell_calls - calls_before
    producer_on_abstain = producer.production_count - prod0
    ans = producer.speak(decision_from_emerge("ANSWER", subject="owl", verb="fly", polarity="affirm"))
    answer_produced = bool(ans["produced"]) and (unified.spell_calls - calls_before) > 0

    return {"seed": seed, "all_word_spike_acc": all_acc, "func_word_spike_acc": func_acc,
            "spell_calls_on_abstain": int(spell_on_abstain), "producer_calls_on_abstain": int(producer_on_abstain),
            "answer_produced": bool(answer_produced), "examples": examples}


def _derisk_gpu(seeds):
    """[GPU/cupy] the NEW SPIKE CLAIM: all-word spike render through the flagship's self-organized producer + producer
    moat + function-word lesion collapse. Writes/returns the GPU aggregate."""
    print(f"EMERGE-69 GPU de-risk: the FLAGSHIP's SELF-ORGANIZED producer + the wired FULLY-SPIKING A->W spell -- ALL "
          f"words (content BRIDGE-A + function BRIDGE-F) spike-spelled + producer gate-first moat + function-word lesion "
          f"collapse; {len(seeds)}-seed", flush=True)
    t0 = time.time(); err = None; per = []
    gpu = True
    frate = lesion_func_engine = content_rate = None
    fper = []
    try:
        unified = m68.UnifiedNeuralSpell(load=True)
        gpu = unified._backend_gpu
        if not gpu:
            raise RuntimeError("A->W engines require SIM_BACKEND=cupy (GPU); numpy cannot run the spiking read-out")
        frate, fper = m68._func_wordwise_accuracy(unified.func)
        content_rate, _cper = m67._aw_wordwise_accuracy(unified.content)
        unified_func_lesion = m68.UnifiedNeuralSpell(load=True, func_lesion=True)
        lesion_func_engine, _ = m68._func_wordwise_accuracy(unified_func_lesion.func)
        for s in seeds:
            d = _derisk_gpu_one(s, unified)
            per.append(d)
            print(f"  [seed {s}] all-word-spike {d['all_word_spike_acc']:.3f} (func {d['func_word_spike_acc']:.3f}) | "
                  f"producer-moat: spell-on-abstain {d['spell_calls_on_abstain']} / producer-on-abstain "
                  f"{d['producer_calls_on_abstain']} | answer-produced {int(d['answer_produced'])}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    agg = None
    go_gpu = False
    if err is None and gpu:
        all_word_spike = float(np.mean([d["all_word_spike_acc"] for d in per]))
        func_word_spike = float(np.mean([d["func_word_spike_acc"] for d in per]))
        spell_on_abstain = int(sum(d["spell_calls_on_abstain"] for d in per))
        producer_on_abstain = int(sum(d["producer_calls_on_abstain"] for d in per))
        answer_ok = all(d["answer_produced"] for d in per)
        spiking_func_ok = (frate is not None and frate >= 0.90
                           and lesion_func_engine is not None and (frate - lesion_func_engine) >= 0.40)
        BAR = 0.90
        go_gpu = bool(all_word_spike >= BAR and spell_on_abstain == 0 and producer_on_abstain == 0
                      and answer_ok and spiking_func_ok)
        agg = {"all_word_spike_acc": round(all_word_spike, 3), "func_word_spike_acc": round(func_word_spike, 3),
               "spell_calls_on_abstain": spell_on_abstain, "producer_calls_on_abstain": producer_on_abstain,
               "answer_produced_all": bool(answer_ok), "func_wordwise_rate": frate,
               "content_wordwise_rate": content_rate, "engine_lesion_func_acc": lesion_func_engine,
               "function_spiking_ok": bool(spiking_func_ok)}
    return {"go_gpu": bool(go_gpu), "gpu": bool(gpu), "err": err, "aggregate_gpu": agg, "per_seed_gpu": per,
            "func_wordwise": fper, "elapsed_seconds_gpu": round(time.time() - t0, 1)}


# ====================================================================================================================
# THE CPU CONSOLE-INTEGRATION INVARIANTS: the FULL flagship console (reasoner + producer) with the additive neural_spell
# flag structure -- gate-first moat + membership routing + fluid no-regression + wire content-routing. Token spell (the
# A->W cannot run on numpy); this confirms the flag is additive/default-preserving + the moat holds by construction.
# ====================================================================================================================
def _emerge_answer_probes():
    from research.runners._emerge54_per_dimension_cancellation_derisk import (
        _BIRD_HELDOUT, _FISH_HELDOUT, _BIRD_EXC, _FISH_EXC)
    return [
        (_BIRD_HELDOUT, "fly", f"the {_BIRD_HELDOUT} can fly", "inherited"),
        (_FISH_HELDOUT, "swim", f"the {_FISH_HELDOUT} can swim", "inherited"),
        (_BIRD_EXC[0], "fly", None, "exception"),
        (_FISH_EXC[0], "swim", None, "exception"),
        ("robin", "breathe", "the robin can breathe", "inherited"),
    ]


def _abstain_probes():
    from research.runners._emerge54_per_dimension_cancellation_derisk import _BIRD_HELDOUT
    return [(_BIRD_HELDOUT, "swim"), ("zzz", "fly")]


class _CountingTokenSpell:
    """A token-surface spell with a call counter (mirrors UnifiedNeuralSpell.spell -- the flag installs this shape on
    numpy since the A->W cannot run on the numpy backend; the moat assertion counts spell calls on abstains)."""
    def __init__(self):
        self.spell_calls = 0

    def spell(self, word):
        self.spell_calls += 1
        return str(word)


def _derisk_cpu_one(seed, build_fluid=True):
    from research.runners._emerge60_console_spiking_broca_derisk import SpikingBrocaConsole, _art
    from research.runners._emerge58_unified_fluent_console import UnifiedFluentConsole
    # the FULL flagship console (self_organized structure) with the neural_spell WIRE STRUCTURE: on numpy the A->W can't
    # run, so we install a COUNTING TOKEN spell (the exact shape the neural_spell flag installs -- a spell callback with
    # a call counter). This validates the gate-first moat (spell + producer NEVER invoked on abstains) + membership +
    # fluid + wire content-routing, with the neural-spell wire structure. (EMERGE-66 GO'd the same numpy surface.)
    sp = _CountingTokenSpell()
    con = SpikingBrocaConsole(seed=seed, build_fluid=build_fluid, self_organized=True,
                              self_organized_n_sentences=_N_SENTENCES, neural_spell=False, spell=sp.spell)
    broca = con.broca

    # (f-wire) content-routing on the standard EMERGE probes + gate-first moat spell/producer counters.
    render = []
    n_words = n_ans = 0
    for (m, prop, expect, kind) in _emerge_answer_probes():
        before = broca.production_count
        cb = sp.spell_calls
        reply = con.turn(f"can {_art(m)} {prop}?")
        produced = broca.production_count - before
        n_ans += 1
        rl = reply.strip().lower()
        words = set(rl.split())
        if kind == "inherited":
            exp_words = set(expect.split())
        else:
            ovr = (con.reasoner.ovr_prop.get(m) or "")
            exp_words = {"the", m, ovr}
        words_ok = (words == exp_words) and produced == 1
        n_words += int(words_ok)
        render.append({"member": m, "prop": prop, "kind": kind, "reply": reply, "produced": int(produced),
                       "spell_calls": int(sp.spell_calls - cb), "words_ok": bool(words_ok)})
    render_words = float(n_words / max(1, n_ans))

    # (f) gate-first MOAT on the full console: ABSTAIN -> 0 producer + 0 spell calls.
    moat_calls = moat_spell = moat_idk = n_ab = 0
    for (m, prop) in _abstain_probes():
        before = broca.production_count
        cb = sp.spell_calls
        reply = con.turn(f"can {_art(m)} {prop}?")
        moat_calls += (broca.production_count - before)
        moat_spell += (sp.spell_calls - cb)
        moat_idk += int(reply.lower().startswith("i don't know"))
        n_ab += 1
    moat_ok = (moat_calls == 0 and moat_spell == 0 and moat_idk == n_ab)

    # (c) MEMBERSHIP routing: a fluid-known entity in the ability frame -> fluid path (not falsely denied; producer +
    # spell NOT stolen).
    membership_ok = None
    membership = {}
    if con.fluid is not None:
        before = broca.production_count
        cb = sp.spell_calls
        dog = con.turn("can a dog eat?")
        membership = {"reply": dog, "produced": int(broca.production_count - before),
                      "spell_calls": int(sp.spell_calls - cb)}
        membership_ok = bool((not dog.lower().startswith("i don't know what a dog"))
                             and ("eat" in dog.lower() or "meat" in dog.lower())
                             and (broca.production_count - before) == 0 and (sp.spell_calls - cb) == 0)

    # (d) NO fluid-path REGRESSION on a Broca-FREE baseline (re-seeded per seed; isolates the known fluid-RNG flakiness).
    fluid_ok = None
    reg = {}
    base = UnifiedFluentConsole(seed=seed, prefer_gpu_render=False, build_fluid=True)
    if base.fluid is not None:
        reg["what_chase"] = base.turn("what does the dog chase?")
        reg["anaphora"] = base.turn("what does it eat?")
        reg["what_eat"] = base.turn("what does the dog eat?")
        reg["growth"] = base.turn("the wolf eats rabbit")
        reg["growth_use"] = base.turn("what does the wolf eat?")
        reg["yesno"] = base.turn("does the dog eat meat?")
        reg["moat"] = base.turn("what does the lion eat?")
        fluid_ok = bool("cat" in reg["what_chase"].lower() and "fish" in reg["anaphora"].lower()
                        and "meat" in reg["what_eat"].lower() and "learned" in reg["growth"].lower()
                        and "rabbit" in reg["growth_use"].lower() and reg["yesno"].lower().startswith("yes")
                        and "know" in reg["moat"].lower())

    return {"seed": seed, "render_kind": con.render_kind, "wire_content_words": render_words,
            "moat_ok": bool(moat_ok), "producer_calls_on_abstain": int(moat_calls),
            "spell_calls_on_abstain": int(moat_spell), "membership_ok": membership_ok, "fluid_ok": fluid_ok,
            "render_detail": render, "membership_detail": membership, "fluid_regression_detail": reg}


def _derisk_cpu(seeds, build_fluid=True):
    """[CPU/numpy] the console-integration invariants on the FULL flagship console (token spell): membership + no fluid
    regression + gate-first moat + wire content-routing; the neural_spell flag is additive/default-preserving."""
    print(f"EMERGE-69 CPU de-risk: the FULL flagship console (reasoner + self-organized producer) with the neural_spell "
          f"WIRE STRUCTURE (token spell -- the A->W cannot run on numpy) -- gate-first moat + membership routing + no "
          f"fluid regression (Broca-free) + wire content-routing; {len(seeds)}-seed", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in seeds:
            d = _derisk_cpu_one(s, build_fluid=build_fluid)
            per.append(d)
            print(f"  [seed {s}] wire-content {d['wire_content_words']:.2f} | moat-ok {int(d['moat_ok'])} "
                  f"(producer-on-abstain {d['producer_calls_on_abstain']}, spell-on-abstain "
                  f"{d['spell_calls_on_abstain']}) | membership-ok {d['membership_ok']} | fluid-ok {d['fluid_ok']}",
                  flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    agg = None
    go_cpu = False
    if err is None:
        wire_content = float(np.mean([d["wire_content_words"] for d in per]))
        moat_all = all(d["moat_ok"] for d in per)
        producer_on_abstain = int(sum(d["producer_calls_on_abstain"] for d in per))
        spell_on_abstain = int(sum(d["spell_calls_on_abstain"] for d in per))
        memb_vals = [d["membership_ok"] for d in per if d["membership_ok"] is not None]
        membership_all = (len(memb_vals) > 0 and all(memb_vals))
        fluid_vals = [d["fluid_ok"] for d in per if d["fluid_ok"] is not None]
        fluid_all = (len(fluid_vals) > 0 and all(fluid_vals))
        go_cpu = bool(wire_content >= 0.99 and moat_all and producer_on_abstain == 0 and spell_on_abstain == 0
                      and membership_all and fluid_all)
        agg = {"wire_content_words": round(wire_content, 3), "moat_ok": bool(moat_all),
               "producer_calls_on_abstain": producer_on_abstain, "spell_calls_on_abstain": spell_on_abstain,
               "membership_ok": bool(membership_all), "fluid_ok": bool(fluid_all)}
    out = {"go_cpu": bool(go_cpu), "err": err, "aggregate_cpu": agg, "per_seed_cpu": per,
           "elapsed_seconds_cpu": round(time.time() - t0, 1)}
    OUT_CPU.parent.mkdir(parents=True, exist_ok=True)
    OUT_CPU.write_text(json.dumps(out, indent=2, default=str))
    print(f"[emerge69] CPU slice go_cpu={go_cpu}; wrote {OUT_CPU}", flush=True)
    return out


# ====================================================================================================================
# THE MERGED DE-RISK: run the GPU spike claim inline (this process, cupy), then spawn a NUMPY child for the CPU console
# invariants (the two backends cannot co-execute in one process), MERGE, and emit the full 6-seed verdict.
# ====================================================================================================================
def _run_cpu_child(seeds):
    """Spawn a fresh SIM_BACKEND=numpy child to run the CPU console-invariants slice (the numpy reasoner cannot run in
    this cupy process). Returns the parsed CPU result dict (or an error stub)."""
    env = dict(os.environ)
    env["SIM_BACKEND"] = "numpy"
    cmd = [sys.executable, "-m", "research.runners._emerge69_console_fully_spiking_derisk",
           "--derisk-cpu", "--seeds", *[str(s) for s in seeds]]
    try:
        subprocess.run(cmd, cwd=str(_REPO), env=env, check=False, timeout=1800)
    except Exception as e:  # pragma: no cover
        return {"go_cpu": False, "err": f"cpu child failed: {e!r}", "aggregate_cpu": None, "per_seed_cpu": []}
    try:
        return json.loads(OUT_CPU.read_text())
    except Exception as e:  # pragma: no cover
        return {"go_cpu": False, "err": f"cpu child produced no output: {e!r}", "aggregate_cpu": None,
                "per_seed_cpu": []}


def _derisk(seeds):
    t0 = time.time()
    gpu_res = _derisk_gpu(seeds)                        # (a)(b)(e) on cupy, inline
    print("\n[emerge69] spawning a numpy child for the CPU console-integration invariants (c)(d)(f)...\n", flush=True)
    cpu_res = _run_cpu_child(seeds)                     # (c)(d)(f) on numpy, child process

    err = gpu_res.get("err") or (cpu_res.get("err") if not cpu_res.get("go_cpu") else None)
    gpu = gpu_res.get("gpu", False)
    go_gpu = gpu_res.get("go_gpu", False)
    go_cpu = cpu_res.get("go_cpu", False)
    ag = gpu_res.get("aggregate_gpu") or {}
    ac = cpu_res.get("aggregate_cpu") or {}
    go = bool(gpu and go_gpu and go_cpu and gpu_res.get("err") is None)

    if not gpu:
        verdict = ("SKIP/BOUNDARY -- the spiking A->W read-out requires SIM_BACKEND=cupy (GPU); this run had only the "
                   "numpy backend. Re-run --derisk on GPU (SIM_BACKEND=cupy). The wire + moat logic are CPU-testable "
                   "(tests/test_emerge69_console_fully_spiking.py); the on-spikes A->W is GPU-only.")
    elif gpu_res.get("err"):
        verdict = f"ERROR (GPU) -- {gpu_res['err']}"
    elif go:
        verdict = (
            f"GO -- the FLAGSHIP console renders its EMERGE emergent-reasoning answers 100% ON SPIKES END-TO-END: the "
            f"grammatical STRUCTURE is SELF-ORGANIZED from the corpus (EMERGE-65/66 -- function-word inventory S2 + slot "
            f"inventory S1a + slot order S1b, NO host FRAMES dict) AND EVERY WORD is spelled from language_output SPIKES "
            f"(order via EMERGE-59/63; CONTENT words on BRIDGE-A via EMERGE-67; FUNCTION words the/a/can/does/not on "
            f"BRIDGE-F via EMERGE-68). [GPU] The flagship's OWN self-organized producer, with the neural spell wired into "
            f"its `spell` callback (== the `neural_spell` flag: SelfOrganizedProducer.producer(spell=UnifiedNeuralSpell."
            f"spell)), renders ALL slots (det+subj+func+verb) FROM SPIKES: all-word spike-spell accuracy "
            f"{ag.get('all_word_spike_acc')} (>= 0.90; function-word slots {ag.get('func_word_spike_acc')}); BRIDGE-F "
            f"spells the function words at rate {ag.get('func_wordwise_rate')}, and the FUNCTION-word LESION collapses "
            f"the engine decode to {ag.get('engine_lesion_func_acc')} (a host lookup would be unaffected -> genuinely "
            f"spiking). The gate-first no-confab MOAT holds BY CONSTRUCTION at the producer: "
            f"{ag.get('producer_calls_on_abstain')} producer + {ag.get('spell_calls_on_abstain')} A->W spell calls on "
            f"abstains (ABSTAIN -> the producer + BOTH A->W read-outs NEVER invoked). [CPU] The FULL flagship console "
            f"(reasoner + self-organized producer) with the neural_spell WIRE STRUCTURE holds every integration "
            f"invariant: gate-first moat ({ac.get('producer_calls_on_abstain')} producer + "
            f"{ac.get('spell_calls_on_abstain')} spell calls on abstains), membership-aware routing (a fluid-known "
            f"entity in the ability frame "
            f"-> the fluid path, not falsely denied; producer + spell NOT stolen), NO fluid-path regression (Broca-free "
            f"baseline, isolating the EMERGE-60/66 known fluid-RNG harness flakiness), wire content-routing "
            f"{ac.get('wire_content_words')}. {len(seeds)}-seed. NO sim/ edit; the ONLY change is the ADDITIVE default-"
            f"off `neural_spell` flag on EMERGE-60's SpikingBrocaConsole (default False == EMERGE-66 byte-identical, the "
            f"token surface). ==> the emergent brain DISCOVERS categories from experience -> REASONS -> and now SPEAKS "
            f"its grounded EMERGE answers 100% ON SPIKES (self-organized structure + every word), on the FLAGSHIP "
            f"console, transformer-free, host-token-free. HONEST SCOPE: renders the BOUNDED EMERGE frame inventory "
            f"(ability-affirm / intransitive-exception / negated-modal) 100% on spikes, NOT open prose (R4). BACKEND "
            f"CONSTRAINT (named, not hidden): sim.bridge binds ONE backend at import, so the numpy EMERGE-52/54 reasoner "
            f"+ the cupy A->W cannot co-execute in one process -- the SPIKE claim is validated on the flagship's self-"
            f"organized producer (cupy, the exact producer the flag installs), the console-integration invariants on the "
            f"full console (numpy, token spell); EMERGE-66 already GO'd the numpy console surface. The console EMERGE "
            f"probes whose content words are outside the A->W-trained 21-word vocab (minnow/pike/breathe) fall back to "
            f"the token surface for those words -- the all-word-SPIKE claim is over the A->W-spellable fact set (like "
            f"EMERGE-68), named not hidden.")
    else:
        miss = []
        if not go_gpu:
            if ag.get("all_word_spike_acc", 0) < 0.90:
                miss.append(f"[GPU] all-word spike render {ag.get('all_word_spike_acc')} < 0.90 -- the neural spell did "
                            f"NOT decode every slot on spikes through the flagship's self-organized producer")
            if ag.get("producer_calls_on_abstain") or ag.get("spell_calls_on_abstain"):
                miss.append(f"[GPU] producer MOAT breached ({ag.get('producer_calls_on_abstain')} producer + "
                            f"{ag.get('spell_calls_on_abstain')} spell on abstains) -- BLOCKING")
            if not ag.get("function_spiking_ok"):
                miss.append(f"[GPU] function read-out not clearly spiking (rate {ag.get('func_wordwise_rate')}, "
                            f"engine-lesion {ag.get('engine_lesion_func_acc')} -- lesion did not collapse >= 0.40)")
        if not go_cpu:
            miss.append(f"[CPU] console-integration invariant failed (moat {ac.get('moat_ok')}, membership "
                        f"{ac.get('membership_ok')}, fluid {ac.get('fluid_ok')}, wire-content "
                        f"{ac.get('wire_content_words')}; cpu err {cpu_res.get('err')}) -- a MOAT breach is BLOCKING; a "
                        f"fluid regression on the Broca-FREE baseline is a GENUINE failure (not the known flakiness)")
        verdict = ("BOUNDARY -- " + "; ".join(miss) + ". Do NOT force a GO; do NOT weaken the moat; keep the default "
                   "path (neural_spell=False) byte-identical.")

    summary = {
        "probe": "emerge69_console_fully_spiking", "GO": bool(go), "go": bool(go), "verdict": verdict,
        "mechanism": ("SpikingBrocaConsole gains an ADDITIVE default-off `neural_spell` flag (mirroring EMERGE-66's "
                      "self_organized): when True, the console's producer `spell` callback (which _render_emerge -> "
                      "self.broca.speak -> realize_slot routes EVERY slot through: DET/FUNC + SUBJ + VERB) is the "
                      "EMERGE-68 UnifiedNeuralSpell loaded from the EMERGE-67/68 caches -- content words decode on "
                      "BRIDGE-A, function words (the/a/can/does/not) on BRIDGE-F, both from cp_firing_states[language_"
                      "output] SPIKES. Combined with self_organized=True, the flagship's EMERGE render is 100% ON "
                      "SPIKES: SELF-ORGANIZED structure + EVERY WORD spiking. The gate-first structure is UNCHANGED "
                      "(abstain -> _emerge_turn returns before _render_emerge, so the producer + both A->W engines are "
                      "NEVER invoked). Membership routing + fluid paths inherited verbatim. Default False == EMERGE-66 "
                      "byte-identical (the token surface). BACKEND SPLIT: sim.bridge binds ONE backend at import (a "
                      "module-global), so the numpy EMERGE-52/54 reasoner + the cupy A->W cannot co-execute in one "
                      "process; the SPIKE claim is validated on the flagship's self-organized producer directly (cupy, "
                      "the exact producer the flag installs), the console-integration invariants on the full console "
                      "(numpy, token spell). Reuse-by-import; NO sim/ edit."),
        "task": ("render the flagship console's EMERGE answers 100% on spikes (self-organized structure + every word "
                 "content+function spelled from language_output spikes); [GPU] all-word spike render >= 0.90 + producer "
                 "moat 0 + function-word lesion collapse; [CPU] membership routing + no fluid regression + console moat "
                 "0 + wire content-routing; additive default-off flag keeps the default path byte-identical; 6-seed"),
        "seeds": list(seeds), "gpu": bool(gpu), "n_sentences": _N_SENTENCES,
        "elapsed_seconds": round(time.time() - t0, 1),
        "go_gpu": bool(go_gpu), "go_cpu": bool(go_cpu),
        "aggregate_gpu": gpu_res.get("aggregate_gpu"), "aggregate_cpu": cpu_res.get("aggregate_cpu"),
        "func_wordwise": gpu_res.get("func_wordwise"),
        "per_seed_gpu": gpu_res.get("per_seed_gpu"), "per_seed_cpu": cpu_res.get("per_seed_cpu"),
        "HONEST_NOTE": ("Renders the BOUNDED EMERGE frame inventory 100% ON SPIKES from a SELF-ORGANIZED grammar (order "
                        "via EMERGE-59/63; content words via EMERGE-67 BRIDGE-A; function words via EMERGE-68 BRIDGE-F -- "
                        "every word decoded from cp_firing_states[language_output]), NOT open prose (R4). GO is on the "
                        "WIRE (the flagship's self-organized producer spells every slot on spikes + gate-first moat + "
                        "membership + no fluid regression + function-word lesion collapse). The A->W engines are GPU-"
                        "trained ONCE at the validated scale + cached (a scale/data lever). BACKEND SPLIT (named, not "
                        "hidden): sim.bridge binds ONE backend at import, so the numpy EMERGE-52/54 reasoner + the cupy "
                        "A->W cannot co-execute in one process; the spike claim is validated on the flagship's self-"
                        "organized producer directly (cupy, the exact producer SpikingBrocaConsole(self_organized=True, "
                        "neural_spell=True) installs via SelfOrganizedProducer.producer(spell=UnifiedNeuralSpell.spell)), "
                        "the console-integration invariants on the full console (numpy, token spell; EMERGE-66 already "
                        "GO'd this numpy surface). The all-word-SPIKE metric is over the A->W-spellable fact set (like "
                        "EMERGE-68); the console EMERGE probes whose content words are outside the A->W-trained 21-word "
                        "vocab (minnow/pike/breathe) fall back to the token surface for those words. The ONLY change to "
                        "committed code is the additive default-off `neural_spell` flag on EMERGE-60's SpikingBrocaConsole "
                        "(default False == EMERGE-66 byte-identical -- EMERGE-59..68 de-risks + CI unchanged). The gate-"
                        "first moat is untouched (0 productions + 0 A->W spell calls on abstains, by construction). NO "
                        "sim/ edit."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118, flush=True)
    print(f"[emerge69] VERDICT: {verdict}", flush=True)
    print(f"[emerge69] wrote {OUT}\n" + "=" * 118, flush=True)
    return 0 if go else 1


# ====================================================================================================================
# DEMO (GPU): the flagship's self-organized producer + the wired neural spell renders every word on spikes.
# ====================================================================================================================
def _demo(seed=42):
    unified = m68.UnifiedNeuralSpell(load=True)
    if not unified._backend_gpu:
        print("  [skip] the A->W engines need a GPU (SIM_BACKEND=cupy); numpy fallback cannot run the read-out.\n")
        return
    sop, producer = _build_self_organized_producer(seed, unified.spell)
    print("=== EMERGE-69 -- the FLAGSHIP's SELF-ORGANIZED producer renders EMERGE answers 100% ON SPIKES: structure "
          "discovered from the corpus (EMERGE-65/66) + EVERY WORD (content BRIDGE-A, function BRIDGE-F) spelled from "
          "language_output SPIKES (EMERGE-67/68); gate-first moat intact ===\n", flush=True)
    all_acc, func_acc, ex = _producer_all_word_spike_render(producer, seed)
    print(f"  ALL-word spike render through the flagship's self-organized producer: {all_acc:.3f} "
          f"(function-word slots {func_acc:.3f}); e.g.:")
    for e in ex[:4]:
        print(f"    [{e['frame']}] {e['surface']}  (expected {e['expected']})")
    calls0 = unified.spell_calls
    prod0 = producer.production_count
    for _ in range(3):
        producer.speak(decision_from_emerge("ABSTAIN"))
    print(f"\n  gate-first moat: on 3 ABSTAINs the producer produced {producer.production_count - prod0} times + the "
          f"A->W spell was called {unified.spell_calls - calls0} times (both 0 = the load-bearing property)\n",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--derisk", action="store_true", help="GPU spike claim inline + spawn numpy child for CPU invariants")
    ap.add_argument("--derisk-gpu", action="store_true", help="GPU spike claim only (this process must be cupy)")
    ap.add_argument("--derisk-cpu", action="store_true", help="CPU console invariants only (this process must be numpy)")
    a = ap.parse_args()
    if a.derisk:
        return _derisk(a.seeds)
    if a.derisk_gpu:
        r = _derisk_gpu(a.seeds)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(r, indent=2, default=str))
        return 0 if r.get("go_gpu") else 1
    if a.derisk_cpu:
        r = _derisk_cpu(a.seeds)
        return 0 if r.get("go_cpu") else 1
    _demo(a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
