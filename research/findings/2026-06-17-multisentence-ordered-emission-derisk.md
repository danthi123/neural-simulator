# Multi-sentence fluency via ORDERED TOPIC-SEQUENCING on the spiking substrate — cheap-first de-risk (GO, 6/6)

**Date:** 2026-06-17
**Runner:** `research/runners/_phaseB_multisentence_ordered_emission_derisk.py`
**Raw:** `research/findings/raw/_phaseB_multisentence_ordered_emission.json`
**Backend:** CPU / numpy (`SIM_BACKEND=numpy`). The spiking RF composer + ordered-WM run there (each op is a small resonate-and-fire bridge).
**Verdict:** **GO** — unanimous 6/6 seeds, 1.000 on every test.

## The question

The agent can recall ONE fact and render a single word-ordered sentence (`BrainConversationalAgent.describe` →
`RFPhasorComposer.render_fact`, whose intra-sentence word ORDER is the de-risked spiking competitive-queuing
serial-order generator). It can separately hold an ORDERED sequence of concepts in an order-encoded working memory
(the CYCLE-135 GO `OrderedPositionWM`: items bound to gamma-slot POSITION phasors on the resonate-and-fire
substrate; ordered recall 1.000 @ loads {2,3,5}). **Does composing those two give a coherent ORDERED
MULTI-SENTENCE output — the core of multi-sentence fluency?**

## The mechanism (deliberately topic-sequencing, NOT nested-binding)

```
hold topics in ordered-WM slots          emit one sentence per slot, IN SLOT ORDER
[dog, cat, bird]  --encode_sequence-->    for k in 0..K-1:
  slot0 = bind(pos0, dog)                     topic = wm.read_slot(C, posk)   # spiking unbind + familiarity gate
  slot1 = bind(pos1, cat)                      sentence = agent.describe(topic) # composer recall + serial-order render
  slot2 = bind(pos2, bird)                  concatenate sentences in slot order
  C = bundle(slot0, slot1, slot2)
```

- **Order** comes from the ordered-WM SLOTS (each topic bound to a successive gamma-slot position phasor).
- **Content** comes from the COMPOSER's own validated flat fact memory (each topic's stored SVO fact).
- **Word order WITHIN each sentence** comes from the existing neural serial-order renderer (`enable_neural_render=True`).

**Why topic-sequencing and not nested-binding:** each slot holds a SINGLE concept/topic phasor — the *validated*
regime of `OrderedPositionWM`. It does **not** bind a whole composed SVO fact into a slot. Binding an
already-bound composite into a position phasor is nested binding (`role⊗(role⊗filler)`), the project's documented
SNR wall (the hierarchical-320 nesting null, `2026-06-02-full-320-flat-distinct…`). By keeping slots = single
topics and letting the composer hold the facts, we stay entirely inside two separately-validated regimes and only
ask whether they COMPOSE. **They do.**

Codes are byte-shared: the `OrderedPositionWM` is built with the same `seed` / `D=128` / sorted vocab as the
agent's composer, so a word read out of a WM slot IS a genuine composer concept the recall path uses directly.
The WM cleans up slot reads against the topic subset only (a slot read resolves to a topic, never an action/object
word). NO `sim/` edit; reuse-by-import only.

## Tests + per-seed results (6 seeds: 42 43 44 100 101 102)

All four tests scored **1.000 on every seed**. Frozen bars: emission ≥0.80 at K∈{2,3}, order-control ≥0.80,
unknown-abstain ≥0.80 + both ≥0.80, single-sentence regression ≥0.999.

| seed | K2 emission | K3 emission | order-control | no-confab abstain | no-confab both | K=1 regression | full |
|------|:-----------:|:-----------:|:-------------:|:-----------------:|:--------------:|:--------------:|:----:|
| 42   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | ✅ |
| 43   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | ✅ |
| 44   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | ✅ |
| 100  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | ✅ |
| 101  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | ✅ |
| 102  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | ✅ |
| **mean** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **6/6** |

**Per-seed passes:** emission 6/6 · order 6/6 · no-confab 6/6 · regression 6/6 · **full 6/6**.
**K-split:** K=2 pass 6/6 · K=3 pass 6/6.

### Test 1 — ORDERED EMISSION (the capability)
For K∈{2,3}: pick a random ORDERED subset of topics, emit the multi-sentence output, score CORRECT iff the
emitted sentence sequence is exactly the K stored-fact sentences in the SAME ORDER as the topic sequence
(every slot's sentence == that topic's stored fact, in order, no abstain). **1.000 at both loads, all seeds.**

### Test 2 — ORDER-CONTROL (load-bearing)
Emit for a sequence and for a PERMUTATION of it; require `emitted(perm) == perm(emitted)` — the multi-sentence
output of the permuted topics is the permuted multi-sentence output. **1.000, all seeds** (24–30/40 trials per
seed were non-trivial permutations). An explicit anti-cheat confirms this is **not vacuous**: a storage-order-dump
baseline (emit the matching facts in fixed storage order, ignoring the WM slots) FAILS the permutation test
(`perm(dump) ≠ dump(perm)`), while the real slot-driven emission PASSES — so the inter-sentence order genuinely
comes from the ordered-WM slots, not a fixed storage/recall order.

### Test 3 — CONTENT-FIDELITY / NO-CONFAB (load-bearing)
Emit a length-3 sequence with one UNKNOWN topic (`owl` — a referent the WM can hold, but no fact has it as agent),
placed at a random slot. That slot must ABSTAIN (emit `None`, no sentence) rather than confabulate, while the other
slots emit their correct stored sentences in order. **abstain 1.000, both(abstain+knowns-correct) 1.000, all
seeds.** The abstention is the composer's own no-confab moat: the WM recovers the unknown topic faithfully (its
match is well above the familiarity threshold ≈0.30), then `describe()` returns `None` because no stored fact has
it as agent. Zero moat breaches. (Per the owner's 2026-06-17 moat relaxation the moat is not a hard gate; here it
is free, kept, and perfect.)

### Test 4 — SINGLE-SENTENCE REGRESSION (K=1)
The existing single-fact `describe()` still renders each of the 6 stored facts correctly. **1.000, all seeds** —
the multi-sentence machinery does not regress the base path.

## Example emitted multi-sentence transcript (K=3, seed 42)

```
topic sequence (slot order): ['bird', 'hawk', 'frog']
emitted:                     "bird ate worm. hawk chased mouse. frog crossed road."
```

The three sentences are emitted in the discourse order held in the spiking ordered-WM; permuting the topic
sequence permutes the output (Test 2); the intra-sentence word order is the neural serial-order renderer.

## Honest boundary — the validated ceiling is K≈4 at D=128

The de-risk's pre-registered loads (K∈{2,3}) are comfortably inside the clean regime. A stress sweep past them
(same harness, 3 seeds) maps where the topic-sequencing fidelity degrades — the ordered-WM's bundle-cross-talk
wall as more slots are bundled into one composite at the agent's `D=128`:

| K | seed 42 | seed 43 | seed 44 |
|---|:-------:|:-------:|:-------:|
| 4 | 1.000 | 0.975 | 1.000 |
| 5 | 0.975 | 0.625 | 0.825 |
| 6 | 0.700 | 0.200 | 0.525 |

So multi-sentence turns of **2–4 sentences are clean** at D=128; **5+** erodes (seed-variable) and would need a
larger `D` or fewer concurrent slots (the foundation's `OrderedPositionWM` is validated to load 5 at its native
D=256). This is a property of the ordered-WM bundle capacity, not of the topic-sequencing composition — exactly
as expected, and reported here so the operating envelope is auditable.

## Verdict

**GO.** The agent produces a coherent ORDERED MULTI-SENTENCE output by holding a sequence of topics in the
spiking ordered-WM and emitting one correct sentence per slot IN SLOT ORDER. The output order is order-encoded
(permuting the topic sequence permutes the sentences, and a storage-dump baseline cannot fake this), each
sentence renders the correct stored fact via the validated recall + neural serial-order path, an unknown topic
abstains (no confabulation), and the single-sentence path is un-regressed — all multi-seed, all 1.000.
Multi-sentence fluency via topic-sequencing **composes** the validated ordered-WM and the validated
fact-recall/serial-order renderer, staying clear of the nested-binding wall. The clean multi-sentence ceiling at
D=128 is K≈4 (5+ needs a larger D).

## Reproduce

```bash
SIM_BACKEND=numpy python -u -m research.runners._phaseB_multisentence_ordered_emission_derisk \
    --seeds 42 43 44 100 101 102
```

NO `sim/` edit; reuse-by-import only (`OrderedPositionWM` for the ordered topic buffer, the composer's stored-fact
recall for content, `BrainConversationalAgent(enable_neural_render=True)` / `describe()` for each sentence's neural
word order). Pure runner; raw JSON to `research/findings/raw/`.
