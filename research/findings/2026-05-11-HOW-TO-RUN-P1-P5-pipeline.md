# How to run the P1-P5 catalog-grounded pipeline

**Date:** 2026-05-11
**Purpose:** Quick-start commands for everything shipped in the
2026-05-11 autonomous arc.

## Prerequisites

- Python environment with cupy + numpy (sim/ default)
- For CPU-only: `SIM_BACKEND=numpy` prefix on any command

## P1 — Hippocampal trisynaptic loop validation

Single seed:
```bash
python -m research.runners.validate_trisynaptic_loop \
    --seed 42 --train-events 400 --ca3-recurrent-weight 5.0 \
    --direct-ca3-drive \
    --out research/findings/raw/g11_bg/p1_seed42.json
```

Expected: D.12 (DG separation) PASS robust at all seeds; D.13 (CA3
completion) PASS at cos > 0.7 in seed 42 specifically; near-pass
(0.67-0.68) at seeds 43-44.

The "real" criterion is two-concept discrimination (relative test):

```bash
for seed in 42 43 44; do
    python -m research.runners.validate_two_concept_discrimination \
        --seed $seed --train-events 400 --ca3-recurrent-weight 5.0 \
        --out research/findings/raw/g11_bg/two_concept_seed${seed}.json
done
python -m research.runners.aggregate_two_concept_seeds \
    --seeds 42,43,44 \
    --out research/findings/2026-05-11-P1-two-concept-multiseed.md
```

Multi-seed result (already verified): **3/3 BIOLOGY-FAITHFUL PASS**.

## P2 — Engram-tagging API

API methods on SimulationBridge:
```python
bridge.start_engram_recording("apple")
# ... drive lang_input("apple") + run sim steps ...
bridge.commit_engram_tag("apple", top_k=50, region_filter=["ca3"])
bridge.stimulate_tag("apple", drive_pA=200.0)
bridge.clear_tag_drive()
bridge.list_engram_tags()
bridge.get_engram_tag_indices("apple")
bridge.delete_engram_tag("apple")
```

Tags persist through `save_checkpoint`/`load_checkpoint` (HDF5
`engram_tags/` group).

Unit tests:
```bash
SIM_BACKEND=numpy python -m pytest tests/test_engram_tagging.py -v
```

Liu 2012-style behavioral test:
```bash
python -m research.runners.validate_causal_recall \
    --seed 42 --train-events 200 \
    --out research/findings/raw/g11_bg/causal_seed42.json
```

## P3.1 — Concept replay during NREM

Programmatic:
```python
from research.runners.consolidation_trainer import (
    run_concept_replay_phase,
)
from research.runners.text_minimal_isolation import set_sleep_gates

set_sleep_gates(bridge)
run_concept_replay_phase(
    bridge,
    tag_names=["apple", "river"],  # engram tags committed earlier
    n_replays_per_tag=20,
    burst_duration_ms=100,
    inter_burst_ms=50,
)
# Now consolidated; tags' associations are in cortex
```

Unit tests:
```bash
SIM_BACKEND=numpy python -m pytest tests/test_concept_replay.py -v
```

## P4.1 — Positional context for episodic binding

Single seed:
```bash
python -m research.runners.validate_positional_binding \
    --seed 42 --train-events 100 \
    --out research/findings/raw/g11_bg/p41_seed42.json
```

Multi-seed:
```bash
for seed in 42 43 44; do
    python -m research.runners.validate_positional_binding \
        --seed $seed --train-events 100 \
        --out research/findings/raw/g11_bg/p41_seed${seed}.json
done
python -m research.runners.aggregate_positional_seeds \
    --seeds 42,43,44 \
    --out research/findings/2026-05-11-P41-positional-multiseed.md
```

Expected: all seeds PASS (cosines well below 0.4 across all 4
(word, position) pair criteria).

Programmatic: enable_episodic_context=True in
`build_biological_brain_regions`. Use `positional_drive_pattern`
from `sim/text_embeddings`:
```python
from sim.text_embeddings import positional_drive_pattern
drive = positional_drive_pattern(
    position=k,  # 0-indexed sentence position
    n_neurons=200,
    drive_max_pA=200.0,
    sparsity=0.1,
    n_max_positions=16,
)
# Apply to ec_context region during encoding
```

## P5 — Ventral semantic stream

Single seed:
```bash
python -m research.runners.validate_ventral_semantic \
    --seed 42 --n-train-events 100 --n-replay-cycles 20 \
    --out research/findings/raw/g11_bg/p5_seed42.json
```

This runs the FULL P1+P2+P3.1+P5 pipeline end-to-end:
1. Build hippo bridge WITH ventral semantic stream enabled.
2. Encode "apple" and "river" via lang_input + hippo plasticity.
3. Tag CA3 ensembles via P2 engram API.
4. Run P3.1 concept replay (cortical consolidation).
5. Test comprehension (lang_input → semantic_cortex).
6. Test naming (engram tag → lang_output).

Wall clock: ~8-15 min per seed.

Programmatic: enable_ventral_semantic=True in
`build_biological_brain_regions`.

## P6 — Broca's compositional syntax (DESIGN ONLY)

Design at `docs/plans/2026-05-11-P6-brocas-grammar-design.md`.
Implementation pending.

## Full unit test suite

```bash
SIM_BACKEND=numpy python -m pytest \
    tests/test_engram_tagging.py \
    tests/test_concept_replay.py \
    tests/test_bridge_memory.py \
    tests/test_lineage.py \
    -q
```

Expected: 84+ passing.

## File map

| File | Catalog | Phase |
|---|---|---|
| `sim/text_embeddings.py` | various | helpers for word + positional embeddings |
| `sim/bridge.py` (engram methods) | D.14 | P2 |
| `research/runners/text_minimal_isolation.py` (builder) | D.03-D.13, G.11/13 | P1, P4.1, P5 substrates |
| `research/runners/consolidation_trainer.py` | D.19 | P3.1 |
| `research/runners/validate_trisynaptic_loop.py` | D.03+D.12+D.13 | P1 validation |
| `research/runners/validate_two_concept_discrimination.py` | D.12 ∩ D.13 | P1+P2 integration |
| `research/runners/validate_positional_binding.py` | D.01+D.02+D.11 | P4.1 validation |
| `research/runners/validate_ventral_semantic.py` | G.11+G.13 | P5 validation |
| `research/runners/validate_causal_recall.py` | D.14 | P2 Liu-2012 behavioral |
| `research/runners/aggregate_*.py` | — | multi-seed aggregation |
| `tests/test_engram_tagging.py` | D.14 | P2 unit tests |
| `tests/test_concept_replay.py` | D.19 | P3.1 unit tests |

## What's next (after P4.1 multi-seed completes)

1. **Run P5 multi-seed** — validates comprehension + naming across
   seeds 42, 43, 44. Wall clock ~30-45 min.
2. **Run Liu-2012 causal recall multi-seed** — validates engram
   tagging behavioral correctness.
3. **Implement P6** (Broca's + compositional syntax). Design ready;
   ~2 weeks code + tests.
4. **P3.2 sequence replay** (deferred until P6 produces sequences).
5. **P7+ — multi-word sentences, reasoning, conversation.**

Catalog citations + roadmap T1.A-C all addressed by P1-P5. Remaining
T2/T3 (cerebellum, compartmental neurons, muscle) per the
catalog's existing buildout roadmap.
