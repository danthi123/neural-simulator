# Direction 3 V=32 PRODUCTION DECISIVE = DIRECTION_3_V32_PASS multi-seed (3 of 3 seeds, all 3 loads, both readouts; 18/18 cells clear 0.80 bar; L=5 OI 0.993 mean); pillar n=105 candidate pending adversarial reviewer CLEAR

**Date:** 2026-05-25 ~07:10 EDT
**Status:** DIRECTION_3_V32_PASS at PRODUCTION scale; pre-staged adversarial reviewer dispatch is the immediate next step; if reviewer CLEAR -> record pillar n=105 + update capability_status.json headline

## What was tested

Production-scale decisive multi-seed run of Direction 3 V=32
(vocab scaling on bio_brain_regions from V=16 to V=32). Launched
04:41 EDT; completed 07:08 EDT; total wall 146.8 min (~2.45 hr;
faster than 6-hr design estimate).

Production config (vs smoke):
- n_lang_input = 2048 (vs 1024)
- n_per_pool = 200 (vs 100)
- n_fs_per_pool = 24 (vs 12)
- n_events_per_word = 200 (vs 100)
- M_OBS = 16 (vs 8)
- 3 seeds [42, 43, 44]
- Loads [L=2, L=3, L=5]
- Bridge: 11264 neurons (vs 5632 smoke); 9.1M synapses; CuPy GPU

## Result: DIRECTION_3_V32_PASS

Multi-seed mean accuracy:

| Load | OB (order-bearing) | OI (order-invariant) |
|---|---|---|
| L=2 | 1.000 | 1.000 |
| L=3 | 1.000 | 1.000 |
| L=5 | 1.000 | 0.993 |

Per-seed L=5 (the strictest load):
- seed 42: OB 1.000 / OI 1.000
- seed 43: OB 1.000 / OI 0.990
- seed 44: OB 1.000 / OI 0.990

All 18 cells (3 seeds × 2 readouts × 3 loads) clear the 0.80 bar
by ≥0.19 margin. Verdict computed by frozen
`direction_3_verdict.compute_verdict` from recorded JSON:
**DIRECTION_3_V32_PASS** (pre-registered tag).

Seed wall clock: seed 42 = 57.8 min (with R-v3 GPU contention);
seeds 43, 44 = 32-43 min each (after R-v3 freed GPU).

## Confirmation of smoke result

Smoke scale (commit 9a09576) also produced 1.000/1.000/1.000/1.000/
1.000/0.993 (essentially identical multi-seed means). Production
confirms the smoke wasn't an artifact of reduced-scale parameters.

The bio_brain_regions substrate's parallel-matching mode-unification
mechanism SCALES from V=16 (pillars n=96/n=97/n=98) to V=32
WITHOUT precision degradation at any of the 3 tested loads.

## Biology-translatable insight (now production-validated)

The bio_brain_regions concept-pool architecture has substantial
vocab-capacity headroom beyond V=16. Doubling concept-pool count
(16 -> 32 distinct pools = 4 motor + 12 noun + 12 verb + 4 adjective)
produces NO accuracy degradation at gamma-slot loads L=2/3/5 on the
production-trained substrate (200 events/word; 2048 lang_input;
200 neurons/pool).

This aligns with the FHRR algebra capacity envelope prediction
(pillar n=87): capacity ~ N_dim/V. V=32 should have ~100 load
capacity (far above the L=7 gamma ceiling). The substrate-grounded
geometry preserves this algebra-level prediction at production scale.

Concrete claim now production-validated: **cortical concept-pool
substrates (each pool = 200-neuron attractor with FS inhibition +
lang_input/lang_output pathways) support compositional encoding/
decoding at vocabularies of at least 32 concepts on the 6-7
gamma-slot capacity envelope.** Extends the project's biology-
faithful conversational substrate from 16 to 32 unique concepts
single-substrate (no cross-bridge composition required).

## Pillar n=105 candidacy

The result clears all pre-registered criteria:
- Multi-seed (3/3 seeds at every cell)
- Multi-load (3 loads pre-registered: 2, 3, 5)
- Both readouts (OB + OI)
- Bar UNCHANGED at 0.80 (cleared by 0.19+ margin)
- Verdict computed by frozen module from recorded JSON
- Per-seed reproducibility (all 3 seeds well above bar)
- Production scale (consistent with smoke at reduced scale)

**Pillar n=105 candidate**, pending adversarial reviewer CLEAR per
the pre-registered post-verdict chain.

## Pre-registered next concrete action

Dispatch adversarial reviewer subagent with the prompt at:
`docs/plans/2026-05-25-direction-3-v32-production-adversarial-reviewer-prompt.md`

Reviewer scrutinizes:
1. Multi-seed reproducibility (3/3 at every cell)
2. Smell-test recomputation from raw per-seed data
3. V=16 vs V=32 genuine extension (not artifact)
4. Frozen verdict module output verification
5. Anti-cheat: byte-unchanged parallel-matching primitive
6. Score-tampering check (no threshold changes during run)
7. Load-ceiling map V=16 reference applicability

Pre-registered reviewer verdict:
- CLEAR -> record pillar n=105 + update capability_status.json
  headline + commit findings doc as the pillar record
- BLOCK -> document strengthening fix; do NOT promote pillar

## What is preserved unconditionally

- bio_brain_regions substrate (build_biological_brain_regions
  byte-unchanged)
- Pillars n=96/n=97/n=98 unchanged (this is an extension of those)
- Frozen verdict module unchanged from Task 3 declaration
- No-confab moat 7/7 byte-identical
- Bar UNCHANGED at 0.80 throughout
- Direction Q + Direction R-v3 prior findings preserved
- Both remotes propagated

## Discipline preserved

- Multi-seed [42, 43, 44] decisive (not just one seed)
- Production scale matches design doc spec
- Mandatory characterization: per-seed per-load values reported
- Reviewer dispatch is pre-registered (not improvised)
- Smell-test: production verdict matches smoke verdict (sanity check
  on parameter scale)
- The 18/18 cells PASS at production scale rules out smoke artifact

## Files

- Runner: `research/findings/raw/direction_3_v32_runner.py`
- Vocab spec: `research/findings/raw/direction_3_vocab_spec.py`
- Bridge builder: `research/findings/raw/direction_3_bridge_builder.py`
- Verdict module (frozen): `research/findings/raw/direction_3_verdict.py`
- Production result JSON: `research/findings/raw/direction_3_v32_production.json`
- Production log: `research/findings/raw/direction_3_v32_production.log`
- Smoke counterpart findings: `research/findings/2026-05-25-DIRECTION-3-V32-SMOKE-PASS-...md`
- Design doc: `docs/plans/2026-05-25-direction-3-vocab-scaling-bio_brain_regions-design.md`
- Reviewer prompt (pre-staged): `docs/plans/2026-05-25-direction-3-v32-production-adversarial-reviewer-prompt.md`
- Mechanism-class audit guide: `docs/plans/2026-05-25-prior-mechanism-class-audit-direction-selection-guide.md`
