# Consolidation probe = TERMINAL localization: replay-driven consolidation does NOT lift the compositional engram's cortical readout through 60 cycles; the compositional blocker is a MISSING SUBSTRATE PATHWAY -- the hippocampo-cortical consolidation wiring (ca1 -> motor, ca1 -> language-output) was built for direct word-to-motor bindings and there is NO ca1 -> concept-pool consolidation pathway; compositional bindings cannot be consolidated into the cortical concept representation because the wire that would carry them does not exist

## Status

Third and terminal probe of the renewed-focus compositional
investigation. The storage-locus probe (commit `ddd714d`) found the
compositional engram tag is hippocampal-only and stranded -- tag
stimulation drives the cortical concept pools only at the noise floor.
It hypothesized replay-driven consolidation (the project's validated
Phase 1.3 mechanism) as the missing ingredient. This probe runs that
mechanism and measures whether it works. Controller-only; single seed
42; cached 200-event unified substrate; reused `run_concept_replay_phase`,
`set_sleep_gates`, `freeze_all_gates`, `lang_output_pattern_during_stim`
all byte-unchanged. ~2 min wall-clock.

## Result (diagnostic; pre-registered routing rule)

The four compositional engram tags were measured at 0, 20, and 60
cumulative replay cycles per tag (20 = the sixth arc's count; 60 = 3x):

```
| cumulative replay | bound-adj POOL rate | bound-adj top-among-adj | langout bound-adj rate | langout-top correct |
|-------------------|--------------------:|------------------------:|-----------------------:|--------------------:|
|  0 cycles         | 0.0015              | 0/4                     | 0.18                   | 1/4                 |
| 20 cycles         | 0.0017              | 1/4                     | 0.16                   | 0/4                 |
| 60 cycles         | 0.0015              | 2/4                     | 0.15                   | 0/4                 |

bound-adjective pool rate:    0.0015 -> 0.0015  (FLAT at the noise floor)
language-output bound-adj:    0.18   -> 0.15    (drifts slightly DOWN)
```

**Pre-registered routing rule -> CONSOLIDATION_DOES_NOT_LIFT.**
Replay-driven consolidation does not lift the tag-to-cortex readout
through 60 cycles. The bound-adjective pool firing is flat at the
0.0015 noise floor (direct binding, for calibration, runs 0.2-0.8 --
two to three orders of magnitude higher). The language-output
bound-attribute signal does not rise; it drifts slightly down. The
"top-among-adj 0/4 -> 2/4" wobble is at noise-floor magnitudes
(0.0015) -- it is not a real signal, it is which of four
indistinguishable near-zero rates happens to be marginally largest.

## The terminal diagnosis: a missing substrate pathway

The validated `run_concept_replay_phase` consolidation transfers a
hippocampal binding into cortex by driving ca3 -> ca1 and letting STDP
strengthen ca1's projections to cortex. It is 3/3 strict anti-cheat
multi-seed validated -- **for direct word-to-motor bindings.** It does
not consolidate the compositional engram tags.

The reason is structural and visible in the substrate's gate
inventory. The hippocampo-cortical consolidation pathways the
substrate provides are:

- `ca1_to_motor` -- ca1 projects to the motor pools
- `ca1_to_lang_out` -- ca1 projects to the language-output population
- `ca1_to_semantic`, `ca1_to_lang_pool_*` -- ca1 to the semantic /
  per-concept language-output pools

There is **no `ca1_to_noun_pool` and no `ca1_to_adjective_pool`.** Ca1
does not project to the concept pools at all.

This consolidation wiring was built for the project's validated
direct-binding task: consolidate a word-to-motor association by
strengthening ca1 -> motor. It works, and it is validated. But a
compositional (noun, adjective) binding would need to be consolidated
into the cortical concept representation -- the noun and adjective
POOLS -- and there is no ca1 -> concept-pool pathway for replay to
strengthen. Replay drives ca3 -> ca1, ca1 fires, and its drive goes to
motor and language-output, not to the adjective pools where
compositional readout must land. The compositional binding cannot be
consolidated because the wire that would carry it does not exist.

## The full three-probe causal chain

The renewed-focus investigation drilled the eight-architecture
convergent ceiling down to a single structural cause through three
cheap-first probes (~5 minutes of compute total):

1. **Difference-readout probe** -- the blocker is not the readout
   computation. Removing the cue's contribution by subtraction
   surfaces no tag signal; the failure is upstream of the readout.

2. **Storage-locus probe** -- the compositional engram tag is
   hippocampal-only by construction (`region_filter=["dg","ca3","ca1"]`);
   tag stimulation drives the cortical concept pools at the noise
   floor. The binding is stored but stranded.

3. **Consolidation probe (this finding)** -- replay-driven
   consolidation, the validated mechanism that bridges hippocampus to
   cortex for direct bindings, does not bridge it for compositional
   bindings, because the ca1 -> concept-pool pathway it would need to
   strengthen does not exist.

The eight prior architectures (gating, theta-multiplexing,
disinhibition, per-regime monitoring, cue-suppression, generative
replay, aggressive consolidation, pool-readout) all operated
downstream of this missing pathway. None could have worked. The sixth
arc's modest abstention-inclusive 0.46 was never genuine compositional
retrieval -- the localization diagnostic already put raw compositional
top-1 near 1/5, and this probe confirms even the sixth arc's 20-cycle
replay left the tag-to-cortex readout at the noise floor.

## What this means -- and what it requires

This is the terminal finding of the compositional retrieval
investigation under the current substrate, and it is a genuine,
sharp, biology-translatable result. It converts "eight architectures
mysteriously plateau at ~0.46" into a single precise structural
statement: **the unified substrate has a hippocampo-cortical
consolidation pathway for direct word-to-motor bindings but no
consolidation pathway from the hippocampal engram to the cortical
concept pools; compositional bindings therefore cannot be
consolidated and cannot be read out.**

This cannot be fixed by any runner-side overlay, readout computation,
gating, rhythm, or replay tuning -- the entire eight-arc design space.
It requires a SUBSTRATE refinement: a ca1 -> concept-pool consolidation
pathway, so that replay can transfer a compositional binding into the
cortical concept representation.

Biologically this is coherent and correct. The hippocampus binds
arbitrary conjunctions in one shot; cortical consolidation requires a
pathway from the hippocampal output (ca1 / subiculum) to the cortical
region that will hold the consolidated trace. The substrate has that
pathway for the motor-output cortex (the validated direct-binding
task) but not for the concept-pool cortex. Adding it is the
biology-faithful next step -- it is the same projection class
(ca1 -> cortex), extended to the concept pools.

## Honest ceiling (unchanged)

Compositional / conversational capability is NOT achieved and is NOT
claimed. The investigation's deliverable is the terminal localization:
the blocker is a missing substrate pathway, precisely named. The
protected module set is byte-unchanged throughout; the
no-confabulation moat is 7/7 byte-identical; no bar was tuned; all
three probes were reuse-by-import only with no autograd.

## Files / evidence

- Probe script: `research/findings/raw/consolidation_probe.py`
- Probe result JSON: `research/findings/raw/consolidation_probe.json`
- Probe log: `research/findings/raw/consolidation_probe.log`
- Prior two probes: `difference_readout_probe.*`, `engram_storage_locus_probe.*`

## Pre-registered next step (decision point flagged for the owner)

The fix is a substrate refinement: add a ca1 -> concept-pool
consolidation pathway. There are two routes, and the choice has a
genuine governance dimension:

1. **Autonomous, protected-discipline-preserving**: build a NET-NEW
   experimental substrate variant in a new runner -- using the
   brain-region framework's own pathway-declaration mechanism to
   construct a substrate that includes ca1 -> concept-pool
   consolidation pathways -- WITHOUT modifying `build_biological_brain_regions`
   or `text_minimal_isolation.py` (both protected, byte-unchanged
   through the whole 8-arc series). Encode compositional bindings
   into this variant, run replay consolidation, and test whether the
   tag-to-cortex readout now lifts off the noise floor and the bound
   attribute becomes selectively retrievable. This is a genuine
   pre-registered arc and it touches no protected module. It is the
   autonomous next step.

2. **Requires owner decision**: if the experimental variant validates
   the fix, rolling the ca1 -> concept-pool pathway into the main
   `build_biological_brain_regions` would modify a protected,
   validated module. That carries real risk (it could perturb the
   validated direct-binding capability) and is an architectural
   decision the owner should make, with the experimental variant's
   evidence in hand.

The autonomous next step is route 1: the experimental substrate
variant. Route 2 is surfaced now so the owner can weigh in before the
variant's evidence forces the question.
