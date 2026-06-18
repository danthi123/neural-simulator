# OneBrainComposer validated at 320-concept PRODUCTION scale → the production-default flip (2026-06-18, CYCLE 190)

## Headline

The integrated **one-brain composer** (`OneBrainComposer` — the whole who/what conversational
pipeline on ONE persistent co-resident spiking `SimulationBridge`, no host round-trips between
ops) **converses end-to-end at the full 320-concept production scale on the codes it LEARNED
FROM CONVERSATION, 3/3 seeds GO**. This gates and triggers the **production-default flip**: the
flagship production conversation demo now defaults to `--composer onebrain`; the legacy
`RFPhasorComposer` (rf) is retained as the **test oracle** and the **numpy-CPU path**.

This is the capstone of the A5 cleanup arc (CYCLE 185–190) that brought onebrain to full
feature + scale parity with rf, per the owner's standing directive: *make fully-spiking the
default once competitive → retire the legacy numpy production runtime, keeping numpy as the
test oracle* (memory `project_one_brain_integrated_pipeline_and_cleanup`).

## The 320-scale result

`research/runners/consolidated_320_conversation_demo.py --composer onebrain --readout neural`
on the stream-learned 320-concept cortex codes (`_phaseB_stream_codes_320_neural_seed{42,43,44}`):

| seed | recall | abstain (false-accepts) | within-cat-err | yes/no | describe | elaborate | verdict |
|------|--------|-------------------------|----------------|--------|----------|-----------|---------|
| 42   | 1.00   | 1.00 (0)                | 0              | ok     | 'dog eat apple' (ok) | 'apple' (ok) | **GO** |
| 43   | 1.00   | 1.00 (0)                | 0              | ok     | 'dog eat apple' (ok) | 'eat' (ok)   | **GO** |
| 44   | 1.00   | 1.00 (0)                | 0              | ok     | 'dog eat apple' (ok) | 'apple' (ok) | **GO** |

**3/3 GO.** recall 1.00, abstain 1.00 with **0 false-accepts** (the no-confab moat holds at
320 concepts), yes/no via the bound polarity tag, neural-ordered `describe` (the spiking
competitive-queuing serial-order renderer), and on-topic `elaborate` (the dlPFC dialogue
planner) — the WHOLE conversational turn on ONE spiking brain.

The loop closes: **learn word meanings from a conversation stream → converse using them through
the fully-integrated one-brain agent.** The grounded codes (the learned-from-conversation
cortex) pass through `OneBrainComposer(grounded_codes=...)` so the cleanup codebook + binding
both use the SAME codes the production conversation depends on (== the rf grounded path).

Scale: onebrain at V=320 is ≈ 54K neurons (parser slice + RF work registers + the K=32
fact-store + per-block + batched cleanup over the 320-word codebook). The 8-fact demo runs
well under `k_max=32`; construction + the conversation completed in ~1–2 min/seed on the
RTX 3090 (the cleanup-list-construction-at-scale concern was unfounded at this fact count).

## The A5 cleanup arc (CYCLE 185–190) — what made this possible

| cycle | piece | gate |
|-------|-------|------|
| 185 | A5 lever 3: masked RF resonate megakernel (`sim/` edit, bit-identity) → onebrain 4.3× faster than rf | `test_rf_megakernel` 4/4 |
| 186 | recursive embedded **clauses** (2-level register→register unbind, re-kick-per-hop == oracle) | clause parity 2/2 |
| 187 | **reconsolidation** (`update_on_mismatch`/`count_facts`: phase-PE-gated in-place block rewrite) | parity + refactor 2/2 |
| 188 | **agent-level** validation (parser-agnostic `agent.parse`; `composer_kind` through both multi-turn agents) | rf 18/18 + onebrain 2/2 |
| 189 | production **drop-in** (`grounded_codes` passthrough; `--composer` opt-in on both demos) | grounded passthrough + suite 11/11 |
| 190 | **320-scale GO 3/3** → the production-default flip (this doc) | demo 3/3 GO |

CI guard `tests/test_one_brain_composer_agent.py` is **11 tests**, all GREEN with the masked
megakernel default-on. No `sim/` edit anywhere in 186–190 (reuse-by-import).

## The production-default flip (what changed)

- **`consolidated_320_conversation_demo.py`** (the flagship "converse on learned codes"
  production showcase): `--composer` default flipped **rf → onebrain**. The production
  conversation now runs fully-spiking-one-brain by default (needs `SIM_BACKEND=cupy`).
  `--composer rf` still reaches the oracle / numpy-CPU path.
- **rf is retained as the TEST ORACLE** (the parity tests assert onebrain == rf) **and the
  numpy-CPU portability path**. It is NOT deleted.
- **NOT flipped (deliberate, safe):** the library constructor defaults
  (`BrainConversationalAgent` / `MultiTurnAgent` `composer_kind="rf"`) and the lightweight
  CPU transcript demo (`multi_turn_conversation_demo`, which defaults to the numpy backend).
  Flipping the library default would force GPU on every default agent and break the
  numpy-CPU CI/portability (a documented project feature); onebrain stays the explicit choice
  there (with the `--composer onebrain` opt-in surfaced).

## Honest scope

- This is a FUNCTIONAL integration of the EXISTING who/what capabilities. The bind stays the
  exact-inverse FHRR (Fourier Holographic Reduced Representation) idealization — the genuine
  learned-cortex bind is the separate step-3 frontier, unchanged by this arc.
- The 320 neural readout codes exist for 3 seeds (42/43/44); 3/3 is the available neural-code
  ceiling. rf already passes this demo multi-seed and onebrain is answer-identical to rf at the
  unit-test scale (11/11), so 3/3 at production scale is a strong parity result, not a thin one.
- Recall errors under noise are near-random, not semantically biased (the codes' category margin
  is real but thin) — see `2026-06-17-within-category-error-signature-NEGATIVE.md`. This is a
  CODE property, independent of the composer; the moat is structurally safe (the relational
  abstention is code-independent).
- Optional lower-priority composer-parity follow-ons (NOT on the agent's critical path):
  attributed entities (adj+noun — the parser feeds flat SVO), A4 (fully-spiking WTA selection;
  the host argmax read-out is brain-based-compliant).

## Reproduce

```bash
SIM_BACKEND=cupy python -m research.runners.consolidated_320_conversation_demo \
    --composer onebrain --seeds 42 43 44 --readout neural \
    --out research/findings/raw/_consolidated_320_onebrain.json
```
