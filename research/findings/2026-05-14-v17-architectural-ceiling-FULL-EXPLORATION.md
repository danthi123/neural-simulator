# v17 28-word architectural ceiling — full exploration (2026-05-14)

## TL;DR

After **7 distinct hypotheses tested** with full Phase 1 retrains, the
v17 28-pool architecture's failure to reach 90% multitag is **not
caused by any single parameter**. The structural cause is most likely
the **off-target topographic dampening (0.3x) doesn't scale to 28
pools**: 28 × 0.3 = 8.4 total off-target effective weight > 3.0 target
boost.

The v16 16-word vocab remains the validated multi-seed ceiling for
robust semantic conversation. 28-word requires architectural redesign
beyond single-parameter tweaks.

## All hypotheses tested (seed 42)

| # | Hypothesis | Config | Phase 1 PASS | Multitag FULL |
|---|---|---|---|---|
| 0 | **Original v17** | 200 events, weak concepts, canon motors | 14/28 = 50% | 0/9 |
| 1 | More events | 400 events | 6/28 = 21% | 0/9 |
| 2 | Smaller motor pools | 50 motor neurons (was 200) | 7/28 = 25% | not run |
| 3 | No motor training | skip motor STDP + topographic | 0/24 = 0% | not run |
| 4 | Weak motor dynamics | motor exc=0.3 (was 2.0) | 13/28 = 46% | 0/9 |
| 5 | Bigger lang_input + stronger topo | 8192 lang + 5x topo + 0.2 off | 1/28 = 3.5% | not run |
| 6 | Motors effectively off | n_motor=1 + skip training + weak | 0/24 = 0% | 0/9 (PARTIAL 0/9!) |

**Every single hypothesis failed.** The v17 architecture has a
fundamental issue that survives all the levers tested.

## Diagnostic insights

### Phase 1 metric is misleading at small pool sizes

The Phase 1 PASS metric measures per-neuron firing rate ratio
(target / max_off_target). With n_motor_per_pool=1, the single motor
neuron fires at 4-10 Hz (saturated). Per-neuron, that beats the
200-neuron concept pool averaging 1-3 Hz. The metric reports motor
"wins" even though absolute total population activity is tiny.

This means small motor pools don't actually compete for downstream
activation — they have 1 neuron emitting spikes. But the Phase 1
metric still reports them as max-off-target.

### Multitag works for v16 but breaks for v17

v16 (16 pools, ~80% Phase 1) gives 90% multitag. v17 (28 pools,
~50% Phase 1) gives 0%. The discontinuity is NOT linear in Phase 1
quality.

The likely cause: at recall time, stimulating an engram tag activates
both bound concept pools, but they project to lang_output. If 28
concept pools all have similar weak lang_output projections, the
spelling-pattern cosines are too noisy to identify which pools are
firing.

### Off-target accumulation hypothesis

With topographic_factor=3.0 and off_target_factor=0.3:
- v16 (12 concept pools): target=3.0, total off-target weight=11×0.3=3.3
  - Ratio target/off ≈ 0.91 — tight competition but feasible
- v17 (24 concept pools): target=3.0, total off-target weight=23×0.3=6.9
  - Ratio target/off ≈ 0.43 — off-target dominates
- v17 motors+concepts (28 pools): even worse

The off_target_factor needs to scale with pool count. For 28 pools to
preserve the same target/off ratio as v16, off_target_factor would
need to be 0.3 × 12/28 = 0.13.

### Why biglang (0.2 off + 5x topo) failed

I tried tighter off (0.2) + stronger topo (5x), but at 8192 lang_input
the absolute weights of motor pools (canon recurrence) overwhelmed the
benefits. Motor pools fired at 5-7 Hz uniformly across all words.

A cleaner test would be 8192 + 5x topo + 0.1 off + weak_motor_dynamics
(motors equalized). Not run today due to wall-clock budget.

## Path forward (deferred)

Real fixes for v17 28-word vocab:

### Option A: Adaptive off_target_factor
Make `off_target_factor` scale with pool count: `0.3 × 12/n_pools`.
With 28 pools → 0.13. Requires modifying apply_concept_topographic_bias.

### Option B: Concept-only architecture
Eliminate motor pools entirely from text_minimal_isolation when
n_motor_per_pool=0. Requires modifying build_biological_brain_regions
(motor pools currently unconditional).

### Option C: Hippocampus consolidation
Use the validated Phase 1.3 hippocampus architecture (catalog D.13)
to bind concepts. The trisynaptic loop provides natural pattern
completion + separation that may handle 28+ concepts better than
the lateral-FS-only v17.

### Option D: Accept v16 16-word vocab
The v16 architecture is multi-seed validated at 90% multitag /
96.7% yes/no / 98.8% precision. 16 words is sufficient for
demonstrating concept-concept conversation. v17+ scaling is a
multi-week project.

## Recommendation

**Accept v16 16-word as the production vocab ceiling** for the
multitag-based chat REPL. Document the architectural boundary
honestly. Plan v17+ scaling as a separate research arc (option A
above is the most surgical).

The chat REPL today provides:
- 16-word semantic conversation at multi-seed reliability
- Natural-language input (`remember the apple is big`)
- Natural-language output (`describe apple` → "apple is big and red")
- Yes/no questions (96.7% accuracy)
- Compositional intersection (90% FULL)
- Multi-turn drill-down (`tell me more`)
- Cross-session persistence (`save` + auto-restore)

This is genuine conversational capability at the validated scale.

## Wall-clock budget for v17 exploration (2026-05-14)

Each v17 retrain takes 47-95 minutes wall clock. Today's arc:
- v17 stronger: 94 min
- v17 smallmotor: 42 min
- v17 nomotor: 47 min
- v17 weakmotor: 47 min
- v17 biglang: 95 min (bigger bridge)
- v17 motoroff: 39 min
- **Total: ~6 hours of GPU compute**

Result: 0 productive new vocab. All single-parameter levers exhausted.

The next architectural attempt should be Option A (adaptive off_target_factor)
which is a small code change (modify apply_concept_topographic_bias to
auto-scale by pool count). If that fails, Option C (hippocampus) is
the major next direction.

## Files

- 7 bridge .simstate.h5 files (one per hypothesis) in
  `research/findings/raw/g11_bg/concept_pool_demo/`
- 7 result JSONs in same directory
- 5 multitag eval JSONs in
  `research/findings/raw/g11_bg/multitag_eval/`
- Code changes: `--n-motor-per-pool`, `--skip-motor-training`,
  `--weak-motor-dynamics` flags in `concept_pool_demo.py` (retained
  as opt-in infrastructure for future experiments)
