# Storage-locus probe = ROOT CAUSE found: the compositional engram tag is captured hippocampal-only by construction, and engram-tag stimulation drives the cortical concept pools only at the noise floor (0.001-0.004 vs the 0.2-0.8 of direct binding); the compositional binding is stored but STRANDED -- the hippocampo-cortical pathway for one-shot compositional bindings is not consolidated, so no readout (pool or language-output) can reach it; this explains the eight-architecture convergent ceiling at one stroke

## Status

Second probe of the renewed-focus compositional investigation. The
difference-readout probe (commit `8cb90bf`) relocated the blocker from
readout to storage-and-reactivation. This probe localizes it to the
precise mechanism. Controller-only; single seed 42; cached 200-event
unified substrate; reused `stimulate_tag` + `region_manager` +
`_encode_facts` byte-unchanged; measures concept-pool firing directly,
bypassing the language-output readout entirely. ~1 min wall-clock.

## Result (diagnostic; pre-registered routing rule)

Each of 4 (noun, adjective) bindings was encoded as an engram tag,
then the tag was stimulated and the firing rate of every concept pool
was measured directly:

```
| tag (binding)    | top pool fired     | bound-adj pool rate | cued-noun pool rate |
|------------------|--------------------|--------------------:|--------------------:|
| ep_0 (apple,big) | motor_E    (0.003) | BIG    0.001        | apple  0.002        |
| ep_1 (river,small)| motor_S   (0.004) | SMALL  0.003        | river  0.003        |
| ep_2 (dog,hot)   | noun_APPLE (0.002) | HOT    0.001        | dog    0.002        |
| ep_3 (cat,cold)  | motor_W    (0.002) | COLD   0.001        | cat    0.001        |

bound adjective pool fires strongest among adjective pools: 1/4
cued noun pool outranks bound adjective pool:                 1/4
```

## The decisive observation: every rate is at the noise floor

The headline is not the ranking -- it is the **magnitude**. Every
concept pool, on every tag stimulation, fires at a rate of
**0.001-0.004**. The "top" pool that fires on each tag stim is a
different, essentially-random pool (motor_E, motor_S, noun_APPLE,
motor_W) at a rate indistinguishable from noise.

For calibration: the direct-binding diagnostic (the validated
capability, 85.4% multi-seed) measures concept-pool firing rates of
**0.2-0.8** when a word drives its pool through the trained
language-input pathway. Engram-tag stimulation produces pool firing
**100-400x lower** -- it does not activate the cortical concept
representation at all.

The probe's routing rule classified this "ENCODING_GAP_DIFFUSE", but
the rates show something more fundamental than diffuse encoding: the
engram-tag-to-cortical-pool pathway is **functionally silent**.

## Root cause: the compositional engram is hippocampal-only and stranded

The mechanism is visible in the encoding call itself. Compositional
bindings are encoded by `encode_concept_pair(...)` with
`region_filter=["dg", "ca3", "ca1"]` -- the hippocampal regions. The
engram tag is, by construction, a top-K co-firing ensemble of
**hippocampal neurons only**. It contains no cortical concept-pool
neurons.

Stimulating that tag therefore drives hippocampal neurons. For the
recalled attribute to appear at a cortical readout (concept pool or
language-output population), the hippocampal engram must drive the
cortex through a hippocampo-cortical pathway. For these one-shot-
encoded compositional bindings, that pathway is **not established** --
it has not been consolidated. The binding is stored in the
hippocampus but stranded: it cannot reach the cortical structures that
every readout measures.

This is exactly the complementary-learning-systems division of labor
(McClelland-McNaughton-O'Reilly 1995): the hippocampus does fast
one-shot binding; the cortex holds the slow consolidated representation;
the transfer between them requires replay-driven consolidation. A
one-shot engram tag is a hippocampal trace with no cortical
counterpart until consolidation builds one.

## This explains the eight-architecture convergent ceiling at one stroke

Every one of the eight prior architectures encoded the compositional
tag the same way -- hippocampal-only -- and then tried to read the
binding from a cortical structure (the language-output population, or
in the eighth arc the concept pools). The tag and the readout were in
different structures with no consolidated pathway between them.

- Gating, theta-multiplexing, disinhibition, per-regime monitoring
  (arcs 1, 2, 3, 4, 5): overlays on the dynamics around a readout that
  was reading a structure the tag could not reach. Could not work.
- The readout-computation fix (the difference-readout probe): could
  not recover a signal that never arrived at the readout structure.
- Pool-readout substitution (arc 8): read the concept pools directly,
  but as this probe shows, the pools are at the noise floor on tag
  stim -- so it could not work either.

The single arc that did better -- the sixth, generative replay plus a
prefrontal frame, the local optimum at the abstention-inclusive 0.46
metric -- is the one that **ran replay**. Replay is precisely the
hippocampo-cortical consolidation mechanism. The sixth arc's partial
lift is the corroborating evidence: the missing ingredient is
consolidation of the compositional binding from hippocampus to cortex,
and the sixth arc's 20-cycle replay supplied it partially.

## Honest scientific status

This is a genuine biology-translatable result and the sharpest the
compositional investigation has produced. The eight-architecture-plus-
two-probe convergent ceiling now has a single precise mechanistic
cause: **compositional bindings are encoded as hippocampal-only
engram tags and are not consolidated into cortex, so no cortical
readout can reach them.** The prior arcs' gating/rhythm/readout
variations were all downstream of this and could not address it.

This does not yet achieve compositional capability. It identifies,
with direct measurement, what must change: the compositional binding
must be consolidated from the hippocampal engram into the cortical
concept representation before any readout can succeed.

The protected module set is byte-unchanged; the no-confabulation moat
is 7/7 byte-identical; no bar was tuned; reuse-by-import only; no
autograd.

## Files / evidence

- Probe script: `research/findings/raw/engram_storage_locus_probe.py`
- Probe result JSON: `research/findings/raw/engram_storage_locus_probe.json`
- Probe log: `research/findings/raw/engram_storage_locus_probe.log`

## Pre-registered next step: the consolidation probe

The root cause is the un-consolidated hippocampo-cortical pathway for
one-shot compositional bindings. The decisive cheap-first probe: take
the same four engram tags, run the project's VALIDATED
`run_concept_replay_phase` consolidation (Phase 1.3 mechanism --
replay-driven hippocampus-to-cortex transfer, validated 3/3 strict
anti-cheat multi-seed for direct bindings), then re-run the
storage-locus probe.

Pre-registered routing rule (fixed; diagnostic, no PASS/FAIL bar):
- If consolidation lifts the tag-stimulated concept-pool firing from
  the 0.001-0.004 noise floor into a readable range AND the bound
  adjective's pool becomes selectively strongest: consolidation is the
  missing ingredient. The forward path is clear -- consolidate the
  compositional binding, then read the concept pools. This becomes a
  full pre-registered arc.
- If consolidation does NOT lift the tag-stimulated pool firing: the
  hippocampo-cortical pathway is structurally inadequate for
  compositional (noun+adjective co-firing) bindings even though it
  works for direct (single-word) bindings. That is a deeper substrate
  finding and would route to a substrate-level consolidation-pathway
  question.

This reuses `run_concept_replay_phase` byte-unchanged; single seed 42;
cached substrate; the only new code is the probe driver. It directly
tests the project's own validated consolidation mechanism against the
root cause this probe identified.
