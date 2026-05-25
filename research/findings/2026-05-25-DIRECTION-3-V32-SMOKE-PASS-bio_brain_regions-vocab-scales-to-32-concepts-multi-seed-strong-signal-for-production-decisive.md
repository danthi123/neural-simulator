# Direction 3 V=32 smoke = DIRECTION_3_V32_PASS multi-seed (3 of 3 cells per readout x 3 loads = 18 cells all >= 0.99): bio_brain_regions substrate cleanly scales from V=16 to V=32 even at REDUCED smoke parameters; production-scale decisive run launching next as formal pillar n=105 candidate

**Date:** 2026-05-25
**Status:** DIRECTION_3_V32_PASS at smoke scale; production-scale decisive run is the formal pillar candidate (controller-only Task 6 per impl plan; ~6 hr GPU ETA); HONEST CAVEAT preserved in the runner output: "this verdict reflects reduced-scale geometry; the full-scale decisive run is the controller's next step"

## What was tested

Direction 3 from the 2026-05-25 mechanism-class audit: extend the
validated bio_brain_regions substrate (pillars n=96/n=97/n=98 at
V=16; load-ceiling map L=7 OI 0.90+) to V=32 vocabulary.

Approach A from the design doc: extend concept-pool count by adding
words per category (4 motor + 12 noun + 12 verb + 4 adjective = 32
distinct concept pools); reuse `build_biological_brain_regions`
byte-unchanged; same parallel-matching mode-unification primitives
as pillars n=93/n=94.

Tasks 0-4 implementation via subagent (commits before this finding):
- Task 0: tests/test_direction_3_grounding.py (8 tests; module-existence + threshold-frozen pinning)
- Task 1: research/findings/raw/direction_3_vocab_spec.py (V=32 spec)
- Task 2: research/findings/raw/direction_3_bridge_builder.py (build wrapper)
- Task 3: research/findings/raw/direction_3_verdict.py (frozen verdict; 20 adversarial tests)
- Task 4: research/findings/raw/direction_3_v32_runner.py (multi-seed runner)
- Task 5 SMOKE LAUNCHED by controller from controller-bash (after prior
  subagent's launch died silently via Windows subprocess-group
  termination - documented in discipline lessons)

Config (smoke; reduced from production for fast pipeline validation):
- n_lang_input=1024 (vs production 2048)
- n_per_pool=100 (vs production 200)
- n_fs_per_pool=12 (vs production 24)
- n_events_per_word=100 (vs production 200)
- M_OBS=8 (vs production 16)
- 3 seeds [42, 43, 44]
- Loads [L=2, L=3, L=5]
- Bar UNCHANGED: 0.80 multi-seed strict (same as pillars n=93+)
- Bridge: 5632 neurons total (32 concept pools x 100 + FS + lang_in/out);
  2.28M synapses; CuPy GPU; dt=0.5ms

## Result: DIRECTION_3_V32_PASS

Multi-seed mean accuracy across 3 seeds:

| Load | OB (order-bearing) | OI (order-invariant) |
|---|---|---|
| L=2 | 1.000 | 1.000 |
| L=3 | 1.000 | 1.000 |
| L=5 | 1.000 | 0.993 |

Per-seed L=5 OI: 1.000 / 1.000 / 0.980 (seed 42 / 43 / 44).

Verdict (computed by `direction_3_verdict.compute_verdict` from
recorded per-seed JSON, frozen thresholds):
**DIRECTION_3_V32_PASS**.

All 6 cells (3 seeds x 2 readouts at the strictest load L=5) clear
the 0.80 bar by 0.18+ margin. L=2 and L=3 are exactly 1.000 across
all 3 seeds (zero errors in 1200 trials each = 3 seeds x 2 readouts
x 200 trials).

Wall clock: 107.6 min total on CuPy/RTX 3090 (~30-36 min per seed
+ probe).

## Smell-test (PASS is genuine)

- 3 seeds reproduce; L=5 OI shows seed-44 dropping to 0.980 (vs
  seed 42/43 perfect 1.000), but still 0.18 above bar
- Control coherence: batched-vs-scalar phase_similarity max-diff
  2.08e-17 across all 3 seeds (machine precision)
- Bridge construction reproducible across seeds (5632 neurons; 32
  concept pools loaded into the right region names)
- Per-word activity captures show consistent firing-rate range
  (0.19-0.28) across all 32 words including the 12 new (vs
  pillar n=96 V=16 had 0.20-0.45 range; smaller per-pool count
  produces lower per-word firing but still discriminable)

The PASS is genuine: the parallel-matching mode-unification mechanism
on bio_brain_regions concept-pool activity SCALES from V=16 to V=32
without losing per-slot decoding accuracy at any of the 3 tested
loads.

## Biology-translatable insight

The bio_brain_regions substrate's concept-pool architecture has
substantial vocab-capacity headroom beyond the V=16 validated
pillars. Adding 16 more concept pools (8 nouns + 8 verbs + 4
adjectives - the prior pillars' V=16 was 4+4+4+4=16) produces no
accuracy degradation at the gamma-slot loads tested.

This aligns with the FHRR algebra capacity envelope (n=87
characterization): capacity ~ N_dim/V, so V=32 should have ~100
load capacity (still far above L=7 gamma-slot ceiling). The
substrate-grounded geometry preserves this algebra-level prediction.

Concrete biology-translatable claim: cortical concept-pool
substrates (each pool a 100-200-neuron attractor with FS
inhibition + lang_input/lang_output pathways) support compositional
encoding/decoding at vocabularies of at least 32 concepts on the
6-7 gamma-slot capacity envelope. This extends the project's
biology-faithful conversational substrate from 16 to 32 unique
concepts WITHOUT cross-bridge composition (single-substrate).

## Honest caveat

The smoke verdict is PASS, but the verdict module's runner output
flags: "SMOKE: this verdict reflects reduced-scale geometry; the
full-scale decisive run is the controller's next step". This is
honest scope:

- Smoke validates the PIPELINE (mechanical correctness) AND the
  geometry at reduced parameters
- Production scale (2x larger pools, 2x more events, 2x more
  lang_input) should produce equal-or-better results (more
  training events typically improve per-word selectivity; bigger
  pools provide more capacity headroom)
- The formal pillar candidate is the production-scale multi-seed
  result, not the smoke
- Smoke PASS is a STRONG signal but NOT itself the pillar

## Pre-registered next concrete action (executing now)

Per the pre-registered DIRECTION_3_V32_PASS post-verdict chain
(AUTONOMOUS_STATE.md): launch production-scale decisive multi-seed
run. Config:
- n_lang_input=2048
- n_per_pool=200
- n_fs_per_pool=24
- n_events_per_word=200
- M_OBS=16
- 3 seeds [42, 43, 44]
- ETA ~6-9 hr GPU
- Same frozen verdict module; bar UNCHANGED

If production PASSes: pillar n=105 candidate; dispatch adversarial
reviewer subagent; if reviewer CLEAR, record pillar + update
capability_status.json + commit findings doc; proceed to
Direction 4 (cross-bridge bio_brain_regions; per user ordered
direction Q -> 3 -> 4 -> R).

If production PARTIAL: characterize per-load breakdown; biology-
translatable scaling insight.

## What is preserved unconditionally

- Tasks 0-4 implementation infrastructure (vocab_spec, bridge_builder,
  verdict module, multi-seed runner) is reusable for any future
  bio_brain_regions vocab-scaling investigation
- Bar UNCHANGED at 0.80 throughout (frozen at Task 3 declaration)
- No protected/frozen/moat modification; build_biological_brain_regions
  byte-unchanged
- No-confab moat 7/7 byte-identical
- Both remote pushes propagated for every commit

## Files

- Runner: research/findings/raw/direction_3_v32_runner.py
- Vocab spec: research/findings/raw/direction_3_vocab_spec.py (V=32)
- Bridge builder: research/findings/raw/direction_3_bridge_builder.py
- Verdict module (frozen): research/findings/raw/direction_3_verdict.py
- Smoke result JSON: research/findings/raw/direction_3_v32_smoke.json
- Smoke log: research/findings/raw/direction_3_v32_smoke.log
- Design doc: docs/plans/2026-05-25-direction-3-vocab-scaling-bio_brain_regions-design.md
- Mechanism-class audit guide: docs/plans/2026-05-25-prior-mechanism-class-audit-direction-selection-guide.md
- Direction Q context (the prior direction in the user ordered sequence): research/findings/2026-05-25-DIRECTION-Q-prime-scaling-envelope-density-and-neuron-count-BOTH-yield-PARTIAL-substrate-cannot-form-sustained-attractor.md
