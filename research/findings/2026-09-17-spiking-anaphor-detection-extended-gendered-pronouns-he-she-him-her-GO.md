---
type: finding
status: verified
date: 2026-09-17
mechanism: extend the spiking CA3 pattern-completion anaphor detector to the gendered pronouns he/she/him/her via an
  additive, opt-in extra_anaphors kwarg — four more CA3 attractor assemblies recruited through the byte-identical
  existing mechanism (same SpikingLoopContextBuffer, same ATTRACTOR_WEIGHT Hebbian install, same decide_pronoun
  threshold read), the original five (it/that/they/them/this) left byte-identical
integration_faculty: language (anaphora / referent detection)
lane: language (comprehension — anaphor detection)
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO (6/6 seed-GO). The gendered pronouns are detected by the same spiking attractor mechanism as the original
  five, learning is load-bearing (an untrained-attractor lesion collapses detection with full attribution to the
  attractor weight), specificity holds (no false positives on content words through the public is_anaphor surface), and
  the original five stay byte-identical (recruiting concepts after them never shifts their disjoint neuron slices). This
  is a DETECTION-only extension and is deliberately NOT wired to production — flipping it on before the resolution path
  does gender/number agreement would let "he" pass the anaphor gate and then resolve to whatever the WTA already holds
  regardless of gender, a discourse-coherence gap. Additive, opt-in (extra_anaphors kwarg, default None), byte-identical
  when unused.
runner: research/runners/_spiking_anaphor_gendered_extension_derisk.py
artifacts:
  - research/findings/raw/_spiking_anaphor_gendered_extension/decisive_6seed.json
external: NO-EXTERNAL-NEEDED — an additive extension of the already-committed, already-biology-bound spiking CA3
  pattern-completion anaphor mechanism (2026-09-09 6-seed GO; biology binding spiking-closed-class-pattern-completion);
  no new mechanism, no wall.
builds_on:
  - research/findings/raw/_spiking_anaphor_gendered_extension/decisive_6seed.json
---

# Spiking anaphor detection extended to gendered pronouns (he/she/him/her) — GO 6/6, additive/default-OFF

The spiking CA3 pattern-completion anaphor detector (`SpikingAnaphorDetectorOrgan`) recognized five closed-class
anaphors (it/that/they/them/this) at 6-seed GO (2026-09-09). This extends it to the gendered pronouns he/she/him/her
without touching the original mechanism: an additive, opt-in `extra_anaphors` kwarg (default `None`) appends the new
tokens after the frozen five, recruiting four more CA3 attractor assemblies through the identical
`SpikingLoopContextBuffer` + `ATTRACTOR_WEIGHT` Hebbian install + `decide_pronoun` threshold read.

## Why the original five stay byte-identical (verified, not argued)

Pattern allocation is `perm = rng.permutation(n)` sliced by LIST POSITION, so appending new concepts after the original
five never shifts their neuron slices; each concept's Hebbian edges live entirely within its own disjoint slice-pair;
`set_pathway_weights` is a plain per-(pre,post) CSR write with no cross-edge normalization; and the buffer is built with
Hebbian/structural plasticity disabled, so nothing reshapes weights mid-probe. The de-risk's G0 gate confirms this
empirically (exact peak-rate equality of a baseline-5 vs extended-9 organ on the original probes).

## Verify (from research/findings/raw/_spiking_anaphor_gendered_extension/decisive_6seed.json)
<!--derived-->
- GO 6/6 seed-GO; both preconditions ok (all six project-standard seeds present; floor seed-GO count met).
- G0 byte-identical: mismatches between the baseline-5 and extended-9 organ on the original probes = zero across all six
  seeds.
- G1 clean-cue recall on he/she/him/her = perfect across all six seeds.
- G2 noisy/partial-cue (80% of the cue corrupted) surpasses its bar on all six seeds (worst-case still well above bar).
- G3 specificity: zero false positives on content-word probes through the public is_anaphor() surface, all six seeds.
- G4 untrained-attractor lesion: detection collapses to zero, fully attributable to the attractor weight, all six seeds
  (learning is load-bearing, not a wired prior).
- Byte-identical-off confirmed directly: the DEFAULT organ (no extra_anaphors) still REJECTS he/she/him/her and still
  detects the original five — production is unchanged.

## What it means, and the next rung

This retires nothing on its own and flips no default — it is a de-risk that the same spiking attractor mechanism scales
to gendered pronouns. It is deliberately DETECTION-only and NOT wired to the three production call sites
(multi_turn_agent, multi_turn_agent_v2, brain_chat_tui), because nothing downstream (the WTA biased-competition /
held-referent resolution) yet checks gender or number agreement — turning detection on first would let "he" resolve to
whatever the WTA holds regardless of gender. The next rung (not attempted here): add a gender/number agreement check in
the resolution path, THEN wire `EXTRA_ANAPHORS` into the three call sites with the same byte-identical-differential +
no-regression soak used for the original five's 2026-09-16 flip, and re-verify against the anaphora/discourse battery.
A wall defers a METHOD, never the capability — here there is no wall, only an ordering (agreement-check before flip).
