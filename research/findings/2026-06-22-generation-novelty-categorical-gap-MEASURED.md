# Generation-novelty ceiling-probe — the categorical LLM gap is MEASURED: the composer generates ZERO novel content (2026-06-22)

**Scope:** the decisive empirical probe from the conversational-scaling-vs-dendritic scoping (§6) — converts "the
fixed-algebra composer RETRIEVES stored facts, never GENERATES novel language" from ASSERTED to MEASURED.
`research/runners/generation_novelty_probe.py`, GPU, V=320 / D=128 / N=64 facts, seed 42. NO `sim/` edit, NO retrain
(the RF composer self-generates phasor codes from the seed). On `main`.

## Result — CATEGORICAL_GAP_CONFIRMED
| metric | value |
|---|---|
| novel sentences generated | **0** (`n_novel_generated=0`) |
| distinct-generated / stored ratio | **1.0** (every emission is one of the 64 stored facts) |
| novel-composition score | **0.0** |
| held-out novel triples produced (of 16) | **0** — all 16 ABSTAINED (what_does/who_does/describe → None; is_it_true → unknown) |
| no-confab moat (abstention floor) | **20/20, 0 confabulations** |
| shuffled-fact control | **0 false hits / 132** |
| K-capacity (recall + moat) | **1.0 at K=8/16/32/64** (clean; K=128 skipped — vocab too small for the third pool) |

⇒ The composer emits ONLY stored facts (ratio 1.0, 0 novel), and on 16 in-vocabulary SVO triples it was never told —
structurally identical to stored ones, only never stored — it produces NONE and abstains on all 16. Meanwhile the
**scale axis is clean** (K-capacity recall + moat 1.0 to 64 facts). So the gap to a small LLM is **empirically
CATEGORICAL (free generation), not scale** — exactly as the scoping predicted. The moat held throughout (0
confabulations), so the "0 novel" is a TRUE generation-absence, not a suppressed confabulation.

## What this settles
"The composer can't freely generate" is no longer an assertion — it is **0.0, measured, anti-cheated**. Closing this
gap requires a learned GENERATIVE-SEQUENCE model (the benched BPTT-spiking net, `sim/bptt_snn*.py`), which per the
SOTA (SpikingBrain / SpikeGPT / SpikeLLM = backprop-trained transformers on ~150B tokens, then spiked) needs
large-corpus backprop OFF the biology-faithful local-learning path — a strategic decision, NOT a scaling step, and
NOT the already-ruled-out dendrite. The structured-retrieval agent sits at a clean, characterized ceiling.

## Multi-seed note
Single-seed (42) suffices for a STRUCTURAL result: "0 novel" is a property of the fixed exact-inverse algebra (it has
no generative path — the held-out triples are in-vocabulary + structurally identical to stored ones, so no seed makes
the algebra emit one). Cheaply multi-seed-confirmable if wanted, but the result is mechanism-level, not a
seed-variable measurement.
