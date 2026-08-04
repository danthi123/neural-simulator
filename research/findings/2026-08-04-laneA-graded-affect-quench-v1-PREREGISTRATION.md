---
type: preregistration
status: locked
date: 2026-08-04
mechanism: laneA-graded-affect-quench-v1
spec: research/specs/lanea_graded_affect_quench_v1.json
---

# Lane A graded affect plus active clearing

## Question

Can the existing opponent affect circuit hold a continuously varying positive or negative state, change smoothly
with appraisal history, and still be actively cleared and restarted by the existing spiking `quench_fs` circuit?

This is deliberately narrower than “emotion.” It does not test learned appraisal, interoception, arousal, emotion
concepts, speech tone, or whole-brain integration. A positive result promotes only a graded, persistent, evictable
valence state on one spiking bridge.

## Prior boundary and retained mechanism

The P0.3 circuit established persistent NMDA-dependent affect and causal downstream bias, but its robust
cross-inhibitory operating point saturated into a positive/negative latch. The later active-clear work established
that a dedicated spiking FS pool can terminate that state through GABA-A and leave the circuit able to re-ignite.
Those were separate questions. This experiment lowers only recurrent NMDA gain and requires graded state behavior
and the validated `quench_fs` clear limb to work together on the same `EvictionAffectBrain` bridge.

No `sim/` change is authorized by this preregistration. GABA-B, short-term depression, and host negative-current
quenching are excluded because the record already found them unsuitable for this role.

## Locked phases and seeds

The machine-readable authority is
`research/specs/lanea_graded_affect_quench_v1.json`. Seeds are derived from SHA-256 of the locked namespace, source
anchor, role, and index using the formula in that file. A repository-wide exact-number search found no prior use
before this preregistration.

- Diagnostic, no promotion value: `6158765`, `7695139`.
- Formal, sealed until diagnostic aggregation chooses an operating point: `8981258`, `3822995`, `7318565`,
  `7957896`, `8575633`, `8803404`.

The diagnostic phase evaluates the fixed recurrent-weight ladder `10, 12, 14, 16, 18, 20, 22`. A weight is
eligible only if every required gate passes on both diagnostic seeds. Selection maximizes the worst normalized gate
margin; an exact tie selects the lower weight. If no weight is eligible, there is no selection and formal seeds
remain sealed. Diagnostic values may select an operating point but cannot count as replication or promotion
evidence.

## Required behavior on one bridge

Each candidate contains the opponent valence pools and `quench_fs` limb before any scored step. Resets may separate
protocol arms, but they reinitialize the same bridge object and fixed topology.

1. Persistence: both positive and negative states survive input removal. Each sign retains at least `0.50` of its
   driven displacement and their mean retention is at least `0.62`. A matched NMDA-off bridge must retain at most
   `0.10`.
2. Magnitude: retained absolute mood tracks eight locked appraisal magnitudes with Pearson `r >= 0.60`, spans at
   least `0.02` rate units, and has correct polarity for at least seven of eight levels.
3. Smooth sign crossing: the fixed down/up/down bipolar schedule must correlate with the signed state at
   `r >= 0.60`, express both directions with at least two crossings, match sign at least `75%` outside zero, remain
   near neutral around zero, and avoid a single latch-like jump spanning more than `60%` of its full range.
4. Active clearing: after a held positive state, intact `quench_fs` output reduces residual mood below `0.60` of
   the pre-clear value. A subsequent negative appraisal re-ignites to at least `0.60` of the original magnitude.
5. Lesions and anti-cheats: with only the same bridge's `quench_out` transmission gate closed, the clear command
   and FS firing remain present but residual mood is at least `0.90`; the lesion/intact residual gap is at least
   `0.30`. The FS pool fires during clear, its drive is zero at read, and its read-window rate is below the locked
   quiet bound.

The formal verdict requires at least five of six formal seeds to pass every per-seed gate. All per-seed artifacts
and the phase aggregate are create-only. Missing seeds, duplicate seeds, altered specs, output reuse, absent
diagnostic selection, or a dirty source tree make the phase undefined rather than negative.

## Explicit scaffolds and interpretation

Appraisal values and clear timing remain host-issued experimental inputs. Recurrent gain is selected from a
host-defined ladder. Valence is not yet grounded in a learned appraisal or interoceptive body model, and this package
does not integrate arousal or speech. These are visible scaffolds, not hidden claims.

A formal pass would justify using this state in the next integration experiment. It would not establish complex
emotion, biological appraisal, felt experience, consciousness, or conversational behavior. A failure would localize
which member of the persistence/gradability/clearing tradeoff remains incompatible at this operating scale.
