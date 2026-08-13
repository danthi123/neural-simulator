---
type: finding
status: wired
date: 2026-08-13
lane: prospective
integration_faculty: prospective-memory
mechanism: spiking intention-LATCH + BA10 NMDA-plateau cue-MONITOR wired into the default /api/brain-chat turn
runner: research/runners/_prospective_memory_production_verify.py
seed-waiver: production-INTEGRATION verify of an already-6-seed GO faculty (the intention-latch + BA10 NMDA-plateau cue-monitor: fire_on_cue 6/6 + every silence clause 6/6 in `2026-08-13-prospective-sfa-nmda-amplifier-GO.md`, seeds 42/43/44/100/101/102). This doc verifies the deterministic WIRING glue on the REAL handler (single process, one seed=42 co-resident organ) — it is not a new scientific GO claim. The fires-on-cue / silent-before / wrong-cue / lesion / byte-identical-off arms are decisive on the single wired seed.
artifacts:
  - research/findings/raw/_prospective_memory/production_verify.json
verification: >
  Verified through the REAL webapp.server.brain_chat handler on the production tiny-demo ChatBrain (rf recall,
  SIM_BACKEND=numpy) across a multi-turn conversation: turn 1 FORMS the intention ("remind me to water the plants
  when I mention the garden") -> a disjoint acknowledgement (prospective.kind=formation, held=True); three
  INTERVENING distractor turns HOLD it SILENT (fired=False, the held assembly stays 0.3438 on every turn incl. the
  one that hits the D4 comprehension-repair short-circuit); the CUE turn ("the garden is blooming nicely") FIRES
  (prospective.fired=True, rel>=FIRE_THR=0.20) and the reminder is PREPENDED to the normal turn's answer. Wrong-cue
  silent (a different topic + a DIRECT spiking-specificity read: rel_A=0.0 driving the unlatched slot-B cue while
  slot-A is held). BRAIN_PMEM_LESION=1 zeroes the latch at formation (held_after_lesion=0.0) -> the SAME cue does
  NOT fire (intact fires vs lesion silent). BRAIN_PMEM=0 byte-identical on a 4-turn recall/abstain panel (no
  prospective key) + a "remind me..." turn falls through to the normal path. Canonical brain_chat_tui --smoke
  byte-identical (the smoke calls ChatBrain.gate/answer directly, not the handler; verdict unchanged with the edits
  reverted).
---

# Prospective memory is PRODUCTION-WIRED: a spiking intention-LATCH + BA10 NMDA-plateau cue-MONITOR holds a deferred intention across turns and fires it on the cue, on the default /api/brain-chat turn

**Verdict: WIRED, default-ON, real-handler verified (numpy-CPU) — `wired: YES / on_by_default: YES /
scaffold_retired: NO`.** The de-risked GO prospective-memory faculty
(`2026-08-13-prospective-sfa-nmda-amplifier-GO.md`: fire_on_cue 6/6, every silence clause 6/6) is now a co-resident
spiking organ on the DEFAULT chat turn. It reuses the de-risk substrate VERBATIM (reuse-by-import; NO `sim/` edit) and
was verified END-TO-END through the REAL `webapp.server.brain_chat` handler, not a runner.

## What is wired

A PER-SESSION persistent co-resident spiking bridge (`research/runners/prospective_memory_production_organ.py`,
reuse-by-import of `_pmem_sfa_nmda_amplifier_derisk.SFANmdaProspectiveMemory` — the validated PFC persistent-attractor
intention LATCH + the BA10 NMDA-recurrent cue-MONITOR whose release amplitude is closed by a per-pool
intrinsic-plasticity homeostat + a supralinear POOL-GATED NMDA/dendritic-plateau COINCIDENCE amplifier). Two
behaviours on `brain_chat`, placed right after the AFFECT read (so a "remind me..." formation is not mis-read as a
recall/assertion by the episodic/comprehension/surprise gates):

- **FORMATION (a disjoint turn class).** A host language scaffold detects "remind me to X when Y" / "when Y, remind me
  to X" / "when Y, do X"; the intention assembly is LATCHED (`encode_intention` — a self-sustaining cortex<->dlpfc
  attractor) and the cue-word set is stored host-side. The turn short-circuits with an acknowledgement.
- **MONITOR (a prefix on the normal turn).** On each later turn, when an intention is HELD, the cue-monitor is READ: a
  cue turn drives the cue assembly and reads the SPIKING held×cue coincidence off `cp_firing_states` (rel >= the frozen
  FIRE_THR); on a fire the reminder is PREPENDED to whatever the normal turn produces (the turn still answers what the
  user actually said). A non-cue turn ADVANCES the hold (a distractor write = real competing WM load) and reads
  persistence + silence. The intention is CONSUMED only when the reminder is DELIVERED on a main answer path, so a cue
  turn that hits a disjoint short-circuit keeps the intention held to fire on the next main-path cue mention (never
  silently lost).

Flags: default-ON; `BRAIN_PMEM=0` -> the whole block is skipped and no `prospective` key is added (byte-identical
oracle); `BRAIN_PMEM_LESION=1` -> the latch is zeroed at formation (the held assembly collapses -> the same cue does
NOT fire — load-bearing).

## The real-handler verification (all GO)

Runner `research/runners/_prospective_memory_production_verify.py`, through `webapp.server.brain_chat` on the
production tiny-demo ChatBrain (rf recall, numpy-CPU); artifact
`research/findings/raw/_prospective_memory/production_verify.json`.

| check | result |
|---|---|
| (A) FIRES-ON-CUE + SILENT-BEFORE | formation acknowledged + held; 3 intervening turns silent, held stays 0.3438 (incl. a short-circuit turn); the cue FIRES (rel>=0.20), reminder prepended |
| (B) SILENT-ON-WRONG-CUE | a different topic stays silent; DIRECT spiking cue-specificity: rel_A=0.0 driving the unlatched slot-B cue while slot-A is held |
| (C) LESION-LOAD-BEARING | `BRAIN_PMEM_LESION=1` -> held_after_lesion=0.0 -> the SAME cue does NOT fire; intact fires vs lesion silent |
| (D) BYTE-IDENTICAL-WHEN-OFF | 4-turn recall/abstain panel ON==OFF with NO prospective key; a "remind me..." turn falls through when off |

The fire is caused by the SPIKING latch (the coincidence), not the host cue-match: the lesion — which zeroes only the
attractor edges — collapses the fire while the host cue-detection is unchanged.

## Brain-based scope + the HONEST residual (declared, not hidden)

**Brain-based (load-bearing spiking):** the HOLD (attractor persistence), the cue-monitoring (NMDA-plateau coincidence
integration) and the RELEASE (the accumulator crossing threshold) are all spiking neurons + synapses; every fire/hold
read is `cp_firing_states`, the plateau reads `cp_conductance_g_nmda`.

**HOST-SCAFFOLD, FLAGGED (so `scaffold_retired: NO`):** the cue->action CONTENT binding is installed SYNAPTICALLY at
build (the fixed slot-A cue/act outer-product edges, exactly like every SpikingLoopContextBuffer attractor), and the
host maps the arbitrary intention/cue TEXT onto the fixed slot-A assembly + derives cue-presence from the turn text (a
language/sensory boundary, like the surprise organ's assertion extraction and curiosity's wh-frame). The named
follow-on LEARNS the binding via one-shot Hebbian potentiation at intention-formation (Gollwitzer
implementation-intentions), replacing the synaptic install — that is the retirement rung. Also declared: the organ
runs on its own co-resident latch bridge (rides the one-brain merge, burn-down #1), and the reminder is surfaced on the
DEFAULT recall/abstain/rich turn classes (a cue turn that ALSO hits a disjoint short-circuit class defers the reminder
to the next main-path cue mention).

**Functional correlate, NOT phenomenal:** this measures + reports a prospective-memory CORRELATE (a held-intention ×
cue coincidence release); it makes NO claim of subjective intending.

## Reproduce

```bash
SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf python -u -m research.runners._prospective_memory_production_verify
```
