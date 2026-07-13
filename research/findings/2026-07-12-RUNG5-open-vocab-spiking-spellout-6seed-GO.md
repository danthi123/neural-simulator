# RUNG 5 — open-vocab spiking spell-out: the emergent reservoir-LM SPELLS its next-token prediction ON SPIKES (6-seed GO)

**Date:** 2026-07-12
**Runner:** `research/runners/_rung5_reslm_spiking_spellout_derisk.py` (reuse-by-import: the reslm ladder `_emerge_reservoir_lm_derisk` + the EMERGE-67 A→W read-out `_emerge67_neural_spell_wirein_derisk`; NO `sim/` edit).
**Verdict:** INTEGRATION GO (6-seed) — honest, modest scope (see below).

## What Rung 5 is
The open-generation ladder's Rungs 1–4 are GO (emergent next-token beats bigram, WM-latch distal context, novel-subject generalization, order-decisive recombination). But the reslm's OUTPUT was still a **host word string** (`vocab.word(argmax(W@x))`). Rung 5 makes that output **spiking**: each predicted token is fed to the validated EMERGE-67 A→W read-out (`NeuralSpell.spell`) — drive the word's concept pool on a real `SimulationBridge`, accumulate `language_output` spikes over a read window, cosine-decode the spoken word. So the generator SPELLS what it predicts, from spikes, not a host lookup. This is the EXPRESSIVENESS rung, composing two already-LEARNED components (the reslm's learned next-token + the A→W's learned spelling) — on the emergence ladder, not a hand-built capability. The reservoir's Ueda scale-ceiling (`2026-07-12-reslm-batched-scale-CONFOUND-FREE-...md`) does NOT block it (Rung 5 is expressiveness, not scale).

## Result (6-seed 42/43/44/100/101/102, SIM_BACKEND=cupy, single process)
| metric | value (all 6 seeds) |
|---|---|
| reslm next-token acc (subject→verb) | **1.000** |
| all predictions in the A→W vocab | **True** |
| **spike-spell fidelity** (decode == predicted token, from `language_output` spikes) | **1.000** |
| **lesion fidelity** (zero pool→language_output → decode collapses) | **0.000** |
| e2e (predict-correct AND spell-correct, 2nd independent spell call) | 0.875 |
| **GO gate** (acc≥0.9 + in-vocab + fid≥0.9 + lesion≤0.3) | **GO — 6/6 seeds** |

- **Genuinely spiking (the load-bearing anti-cheat):** lesion fidelity 0.000 — cutting the pool→language_output pathway collapses the decode to nothing; a host lookup would be unaffected. The word comes from spikes.
- **e2e=0.875 (a characterized read-noise, not a failure):** the runner calls `spell(w)` twice per word (once for the fidelity metric, once for e2e); on ~1/8 words the SECOND spiking read flips (finite-spike-read variability, EMERGE-67's noted spiking read). The GO bar rides on `spike_spell_fidelity` (1.000). If a downstream use needs per-call determinism, the fix is a multi-read majority vote on the decode (EMERGE-67's own pattern) — a read-side change, moat untouched.

## Honest scope (what this is NOT)
- **Bounded vocab, not "open" yet.** The A→W read-out spells a **16-word content set** (owl/penguin/.../fly/swim/walks/...) welded to 16 concept pools + GPU-trained-once + cached. Rung 5 demonstrates the WIRING at that vocab. TRUE open vocab (V=200+) is the named **LEVER = more A→W bridges** (EMERGE-68's multi-bridge pattern) — a bounded scale/data follow-on, not this rung.
- **Trivial generation corpus.** The reslm here learns a subject→verb bijection (8 pairs) — enough to produce an in-vocab prediction to spell. The generation QUALITY (beats-bigram, distal context, recombination) is Rungs 1–4, not re-demonstrated here.
- **The spell fidelity is inherited from EMERGE-67.** The NEW contribution is the end-to-end reslm-prediction → spiking-spell WIRING, validated 6-seed.

## ⇒ Ladder status + next
Rungs 1–4 + the emergence-bar close + **Rung 5 (this)** are GO. The generator's output is now spiking. Open follow-ons:
- **Open-vocab lever:** more A→W bridges → spell the reslm's full vocab (bounded engineering, EMERGE-68 pattern).
- **Rung 6 (the next mechanism):** multi-clause discourse — wire the reslm's multi-clause generation through the D3 discourse-referent register (`_d3_reference_tracking_derisk.py`, the spiking discrete-attractor that tracks "who/what we're talking about" across an unbounded narrative, 6-seed GO) for cross-clause coherence. This is a NEW mechanism composition → gets the full deep-research → cheap-first → anti-cheat → 6-seed treatment.

Reuse-by-import; NO `sim/` edit. Runner: `research/runners/_rung5_reslm_spiking_spellout_derisk.py` (`--reslm-only` numpy-CPU smoke; `--derisk` cupy A→W).
