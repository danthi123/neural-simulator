---
type: finding
status: verified
date: 2026-09-17
mechanism: onebrain organ-merge Wave-3 — wiring d6_multiref_wm onto the SINGLE shared spiking pool (prospective_memory
  scope-reduced: its read organ is validated in the pool but its production wrapper wiring is a separate rung), bringing
  the shared pool to ALL 11 cortical organs; the FINAL wave of the one-brain organ consolidation
integration_faculty: onebrain-merge-organs
lane: one-brain substrate consolidation
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO 6/6 — the shared spiking pool validates at ALL 11 co-resident cortical organs (surprise, worldmodel,
  metacog, pragmatic, comprehension, source_provenance, self_schema, curiosity, causal_whatif, prospective_memory,
  d6_multiref_wm). Wave-3 organread all_go 6/6: n_gate_a/b/c 6, n_gain0_freeze 6, n_legacy_diverges 6, n_go 6, and
  n_wave2_carried_read_byte_identical 6 (Wave-2's 9 organs stay byte-identical under the 11-organ pool). Every one of
  the 11 organs: n_alive 6, n_answer_same 6, n_coresidence_byte_identical 6. Completes the Wave-1(6)/Wave-2(9)/Wave-3(11)
  one-brain organ-merge de-risk. The determinism prerequisite (megakernel-v2 N>4968) does NOT apply on the numpy
  backend (megakernel is GPU-only). d6_multiref_wm's get_organ() wiring is committed default-off (5d29d6a70); the
  DEFAULT-ON FLIP of the merged pool (retiring the per-organ co-resident bridges for the pooled path) is the reviewed
  follow-on, gated on the integrated /api/brain-chat no-regression battery.
runner: research/runners/_onebrain_wave3_organread_verify.py
artifacts:
  - research/findings/raw/_onebrain_wave3/organread_6seed.json
external: NO-EXTERNAL-NEEDED — consolidates already-de-risked organs onto the already-validated single_pool framework
  (shipped default-on 2026-09-05); the organread gate is an internal byte-identity/answer/faculty-alive check.
builds_on:
  - research/findings/2026-09-17-onebrain-wave1-wave2-organ-merge-9-organs-one-pool-GO.md
---

# One-brain organ merge COMPLETE — ALL 11 cortical organs on ONE shared spiking pool (Wave-3 GO 6/6)

Wave-1 (6 organs) and Wave-2 (9 organs) validated the growing shared spiking pool 6/6. Wave-3 is the final rung:
d6_multiref_wm wired onto the same pool (prospective_memory's read organ validated in-pool; its production wrapper
wiring is a separate rung), bringing the pool to all 11 cortical organs the onebrain-merge-organs row tracks.

## Result (from research/findings/raw/_onebrain_wave3/organread_6seed.json)

all_go 6/6. Aggregate: n_gate_a 6, n_gate_b 6, n_gate_c 6, n_gain0_freeze 6, n_legacy_diverges 6, n_go 6,
n_wave2_carried_read_byte_identical 6. Per-organ (all 11): n_alive 6, n_answer_same 6, n_coresidence_byte_identical 6
— every organ is byte-identical merged-vs-coresident, answer-preserving, and faculty-alive across all 6 seeds. The
megakernel-v2 N>4968 determinism prerequisite (research/FAILURE_LOG.md 2026-09-02) does NOT apply: it gates only the
GPU megakernel path, and this runs numpy/CPU.

## Scope + the one-brain milestone

Committed to main default-off (5d29d6a70). The one-brain organ-merge is now de-risk-VALIDATED end to end: 11 cortical
organs — surprise, world-model, metacognition, pragmatics, comprehension, source-honesty, self-schema, curiosity,
cause-and-effect, prospective-memory, multi-referent working-memory — computing on a SINGLE shared spiking substrate,
answer-preserving. This is the substrate side of the one-brain goal. The remaining step is the production DEFAULT-ON
flip (retire the per-organ co-resident bridges for the pooled path), gated on the integrated /api/brain-chat battery —
a reviewed follow-on, not a defer: the capability (one shared substrate for the cortical organs) is de-risked GO.
