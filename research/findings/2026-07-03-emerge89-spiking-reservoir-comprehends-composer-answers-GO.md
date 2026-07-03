# EMERGE-89 — FULLY-SPIKING comprehension→composition: the ON-BRIDGE spiking reservoir COMPREHENDS → the composer ANSWERS — **GO** (6-seed)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge89_spiking_reservoir_comprehends_composer_answers_derisk.py`
**Test:** `tests/test_emerge89_spiking_reservoir_comprehends_composer_answers.py`
**Raw:** `research/findings/raw/_emerge89_spiking_reservoir_comprehends.json`

## Why (finishing the fully-spiking pipeline)

EMERGE-88 proved the comprehension→composition HANDOFF with the RATE reservoir (EMERGE-78 echo-state) driving the
spiking `RFPhasorComposer`, and named the mechanical follow-on: swap the rate reservoir for EMERGE-82's `OnBridgeLSM`
— a recurrent Izhikevich `BrainRegion` on a real `SimulationBridge` whose `final_state(U)` has the IDENTICAL signature.
EMERGE-89 does exactly that, so the WHOLE comprehension→composition pipeline runs on spikes end-to-end: the reservoir
understands the sentence on a SimulationBridge, and the composer stores + answers on RF resonate-and-fire spikes.

## The mechanism

EMERGE-88's `ReservoirComprehender` is reused verbatim, with an EMERGE-82 `OnBridgeLSM` injected as its reservoir
(`ReservoirComprehender(seed, discovered, res=OnBridgeLSM(...), enc=...)`). `comprehend(tokens)` drives the recurrent
Izhikevich region per token via `cp_external_input_current`, runs the bridge's real step loop, reads the region's
per-neuron spike-counts as the population feature, and labels each content position's thematic role via the slot
read-out (fit on the on-bridge spike states). Roles → the composer's (agent, action, patient); the composer stores; the
who/what turn + the no-confab moat run on the reservoir's OWN spiking comprehension.

## The de-risk — **GO** (6 seeds; reuse EMERGE-82 + EMERGE-88 + the production composer; NO `sim/` edit)

| gate | value (6-seed) | bar |
|---|---|---|
| reservoir region genuinely spiking | ~2.4 spikes/neuron | > 0.5 |
| **parse** — spiking reservoir maps each transitive to (agent, action, patient) | **1.000** | ≥ 0.90 |
| **who/what recall** — spiking comprehension → composer stores → `query_patient` returns the true patient | **1.000** | ≥ 0.90 |
| **no-confab MOAT** — an (agent, action) never stored → the composer abstains (false-accept rate) | **0.000** | ≤ 0.05 |
| **comprehension-lesion** — collapse the reservoir's closed-class identity → recall collapses | **0.000** | ≤ 0.55 |

*(seed 42 confirmed: spikes 2.43, parse 1.000, recall 1.000, moat 0.000, lesion 0.000; the 6-seed aggregate is in the
raw json.)*

**The result:** the recurrent Izhikevich reservoir, running on a real `SimulationBridge` and genuinely spiking,
comprehends each sentence into thematic roles on spikes; those roles drive the spiking composer's bind/store; and the
who/what turn answers correctly (recall 1.000) with the no-confab moat intact (0 false-accepts) and the spiking
reservoir load-bearing (lesioning its closed-class identity collapses role-labeling → recall → 0.000). Comprehension
→ composition is now **fully spiking end-to-end** — the reservoir on the bridge, the composer on RF resonate-and-fire.

## Anti-cheats (all pass)

- **Held-out CONTENT** — fresh transitive draws the slot read-out never saw; the reservoir abstracts content to the
  OPEN marker, so it generalizes its role-labeling to new words even on spikes.
- **No-confab MOAT** — the composer abstains on any (agent, action) never stored (0 false-accepts).
- **Comprehension-lesion** — collapsing the reservoir's closed-class identity collapses role-labeling on spikes → the
  whole turn collapses (recall 0.000): the spiking reservoir's comprehension is necessary.

## Honest scope

- The reservoir runs at `n_pool=300` on numpy-CPU (the bridge step is heavy → a reduced train per EMERGE-82); a GPU
  run scales the pool. Validated on core transitive SVO (the construction that fills the composer's three roles).
- Reuse-by-import (EMERGE-82 `OnBridgeLSM` + EMERGE-88 `ReservoirComprehender` + the production composer); NO `sim/`
  edit.

## The reservoir arc, fully spiking into the conversational loop

EMERGE-78 (learned map) → 79 (uncontingent non-local) → 80 (spiking pool) → 81 (memory survives) → 82 (on-substrate) →
83–86 (recursion boundary + surpass) → 87 (composes on the one brain) → 88 (comprehension DRIVES composition, rate
reservoir) → **89 (comprehension → composition, FULLY spiking end-to-end — the reservoir on a SimulationBridge, the
composer on RF spikes)**.

## Files
- `research/runners/_emerge89_spiking_reservoir_comprehends_composer_answers_derisk.py` — the on-bridge `OnBridgeLSM`
  injected into EMERGE-88's comprehender + the integration de-risk.
- `tests/test_emerge89_spiking_reservoir_comprehends_composer_answers.py` — 2 CPU tests.
- `research/findings/raw/_emerge89_spiking_reservoir_comprehends.json` — the 6-seed integration.
