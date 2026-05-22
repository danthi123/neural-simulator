# ca1 -> concept-pool variant = honest NEGATIVE: adding the missing consolidation wire is NECESSARY BUT NOT SUFFICIENT; the deeper cause is that the concept pools are built with deliberately WEAK internal dynamics (a v14/v16 design choice for Phase-1 multi-concept training stability) and those weak dynamics prevent the pools from igniting into a readable consolidated attractor from hippocampal drive -- the same property that makes the concept pools trainable makes them non-consolidatable

## Status

The experimental substrate variant arc, executed end-to-end
(design: `docs/plans/2026-05-21-ca1-to-concept-pool-consolidation-pathway-variant-design.md`,
commit `fb19b8a`). Controller-only; single seed 42; net-new code that
touches no protected module. The variant builder calls
`build_biological_brain_regions` byte-unchanged, then appends 12
`ca1 -> concept-pool` consolidation pathways (4 noun + 4 verb + 4
adjective) mirroring the existing `ca1 -> motor` pathway exactly
(density 0.20, weight 2.0, jitter 0.3, plastic, gated). Phase-1
trained fresh on the variant (44.5 min), then the pre-registered
compositional consolidation test.

## Result (pre-registered decision rule; no bar tuned)

```
Variant substrate: 4,883,176 synapses (base 4,825,651; +57,525 =
  12 ca1->concept-pool pathways, confirmed installed).
Phase-1 training: 44.5 min, seed 42, standard 200-event recipe.

Direct-binding sanity: 11/16 = 68.8%  (base 200ev seed 42 ~ 68.8%)
  -> IDENTICAL to base. The added pathway did not disturb Phase-1.
     The variant substrate is clean; the negative below is genuine.

Compositional consolidation test (tag-stim concept-pool firing):
| cumulative replay | bound-adj pool rate | lifted >0.02 | selective | permuted-control |
|-------------------|--------------------:|-------------:|----------:|-----------------:|
|  0 cycles         | 0.0073              | 0/4          | 0/4       | 2/4              |
| 20 cycles         | 0.0066              | 0/4          | 1/4       | 0/4              |
| 60 cycles         | 0.0077              | 0/4          | 1/4       | 0/4              |

Pre-registered verdict -> NEGATIVE.
  pre-consolidation selective:  0/4   (not VOID -- selectivity is not
                                       present from the prior)
  post-consolidation selective: 1/4   (chance level for top-of-4)
  post-consolidation permuted-control: 0/4   (fails)
  post-consolidation lifted off noise floor: 0/4
```

## The diagnosis: the wire is necessary but not sufficient

The `ca1 -> concept-pool` wire IS present and IS functional. The
bound-adjective pool firing rate during tag stimulation rose from
0.0015 (the base substrate, no wire -- consolidation probe finding)
to 0.0073 (the variant, with the wire) -- the wire transmits roughly
5x more drive. But 0.0073 is still ~3x below the pre-registered 0.02
noise-floor threshold and 30-100x below the readable direct-binding
range (0.2-0.8). Replay-driven consolidation across 0, 20, 60 cycles
is dead flat (0.0073 -> 0.0066 -> 0.0077) -- replay moves it nowhere.

The wire was the structurally-missing piece the consolidation probe
identified. Adding it did not work. The deeper cause:

**The concept pools are built with deliberately WEAK internal
dynamics**, and weak pools cannot ignite into a readable consolidated
attractor from hippocampal drive.

The substrate builds the concept pools (noun / verb / adjective) with
`concept_pool_internal_density=0.05` and
`concept_pool_exc_weight_mean=0.3`. The motor pools -- which DO
consolidate, the validated Phase 1.3 result -- use canon dynamics:
`motor_internal_density=0.10` and `motor_exc_weight_mean=2.0`. The
concept pools' recurrent excitation is roughly 7x weaker than the
motor pools'.

This weakness is not an accident. It is a v14/v16 design choice. The
project's own findings (CLAUDE.md, the v10 / iter-KK results)
established that strong concept-pool dynamics cause a "canon
amplifies bias" failure: with canon dynamics, the concept pools'
recurrent activity overwhelms the topographic-prior-driven word
training and Phase-1 multi-concept binding collapses. Weak dynamics
are required for stable multi-concept Phase-1 training.

So the substrate faces a genuine architectural tension:

- **Weak concept-pool dynamics** -> stable Phase-1 multi-concept
  training (validated v14/v16, 88.75% multi-seed) BUT the pools
  cannot host a consolidated attractor driven from ca1; the
  `ca1 -> concept-pool` wire's drive does not ignite them.
- **Strong (canon) concept-pool dynamics** -> the pools could host a
  consolidated attractor (as the motor pools do) BUT Phase-1
  multi-concept training collapses ("canon amplifies bias").

The same property that makes the concept pools trainable makes them
non-consolidatable. The motor pools consolidate (Phase 1.3) precisely
because they have canon dynamics AND only ever host four mutually
exclusive directions -- they never needed weak dynamics.

## The full causal chain (renewed-focus compositional investigation)

The renewed-focus investigation -- one design plus three cheap-first
probes plus this variant arc -- drove the eight-architecture
convergent ceiling to a precise, multi-level root cause:

1. **Difference-readout probe** -- the blocker is not the readout
   computation.
2. **Storage-locus probe** -- the compositional engram is
   hippocampal-only; tag stimulation drives the cortical concept
   pools at the noise floor.
3. **Consolidation probe** -- the validated replay consolidation does
   not bridge it; there is no `ca1 -> concept-pool` wire.
4. **This variant arc** -- adding the `ca1 -> concept-pool` wire is
   necessary but NOT sufficient: the concept pools' deliberately-weak
   internal dynamics prevent them from igniting into a consolidated
   attractor. The blocker is a property tension between Phase-1
   trainability and consolidatability.

This is the deepest the compositional investigation has reached. It
is a genuine, sharp, biology-translatable result: compositional
capability on this substrate is blocked not by a missing feature that
can be added, but by an architectural tension -- the concept-pool
dynamics that the validated direct-binding capability depends on are
incompatible with hosting a consolidated compositional attractor.

## Honest status and the architectural decision this surfaces

Compositional / conversational capability is NOT achieved and is NOT
claimed. The renewed-focus investigation's deliverable is the
multi-level root cause and the architectural tension it exposes.

Resolving the tension is genuinely an architectural decision, and it
pits compositional capability against the validated direct-binding
capability. Candidate routes (none performed here -- surfaced for the
owner):

1. **Dedicated compositional-attractor region** (most promising;
   net-new, no protected modification). Rather than consolidating
   into the weak Phase-1 concept pools, add a SEPARATE region with
   canon dynamics that receives from both ca1 and the concept pools
   and hosts the consolidated compositional attractor. This sidesteps
   the tension: the weak concept pools stay weak (Phase-1 unaffected),
   and a distinct strong region hosts composition. This is testable
   as a further experimental variant without touching any protected
   module. (A design doc for a "dedicated compositional readout
   region" was sketched for the eighth arc but never built as a
   consolidation target -- the eighth arc tested pool-readout
   substitution instead.)

2. **Staged / developmental concept-pool dynamics**: weak during
   Phase-1 training, strengthened afterward for the consolidation
   phase. Biologically grounded (developmental excitability changes)
   but requires runtime dynamics modulation and risks Phase-1
   retention.

3. **Strong-dynamics concept pools with a different Phase-1 training
   regime** that tolerates canon dynamics. Highest risk to the
   validated v14/v16 capability.

Route 1 is the autonomous next step: it is net-new, touches no
protected module, and directly tests whether a strong dedicated
region resolves the tension.

## Discipline check

NO bar tuned. The variant builder calls `build_biological_brain_regions`
byte-unchanged and augments its returned pathway list -- no protected,
frozen, or moat module modified. The direct-binding sanity check
(68.8% = base) confirms the variant substrate is correctly built and
Phase-1 is intact. Reuse-by-import for encode, replay, gate, and
measurement helpers. No autograd. The protected set is byte-unchanged;
the no-confabulation moat is 7/7 byte-identical.

## Files / evidence

- Variant runner: `research/findings/raw/ca1_concept_pool_variant.py`
- Variant result JSON: `research/findings/raw/ca1_concept_pool_variant.json`
- Variant log: `research/findings/raw/ca1_concept_pool_variant.log`
- Variant Phase-1 cache: `research/findings/raw/unified_per_regime/phase1_ca1variant/seed42.simstate.h5`
- Design: `docs/plans/2026-05-21-ca1-to-concept-pool-consolidation-pathway-variant-design.md`

## Pre-registered next step

Route 1 -- the dedicated compositional-attractor region. Build a
further experimental variant (net-new, no protected modification)
that adds a region with canon (strong) dynamics receiving from ca1
and the concept pools, and test whether replay consolidates the
compositional binding into THAT region (read it out there, not at the
weak concept pools). Pre-registered decision rule and anti-cheat
(permuted-tag control; selectivity must emerge from consolidation) to
be pinned in its design doc before the run. If route 1 also fails,
the architectural tension is confirmed as fundamental and the honest
finding is that compositional capability requires resolving the
Phase-1-trainability-vs-consolidatability tension at the substrate
level -- a decision for the owner with the full evidence in hand.
