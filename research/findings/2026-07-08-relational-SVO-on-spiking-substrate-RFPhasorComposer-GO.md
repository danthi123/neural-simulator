# Relational SVO on the SPIKING substrate (GO, 3-seed): what- AND who-questions over the brain's OWN real-corpus codes run on the validated spiking `RFPhasorComposer` (resonate-and-fire + complex-synapse store), no-confab moat intact. The fully-spiking realization of the relational dimension — matching the property dimension's HTM realization. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_svo_spiking_derisk.py` (reuse-by-import: the validated `RFPhasorComposer` + the breadth discovery; `--substrate` = the RF complex-synapse store). `SIM_BACKEND=cupy` for the substrate. NO `sim/` edit.
**Verdict:** GO — relational SVO on the spiking substrate over real-corpus codes, 3-seed, moat intact.

## Why this ran (fully-spiking both dimensions)
The relational SVO Q&A (CYCLE 988) was rate-level (numpy FHRR). The property dimension has a spiking realization (`CancellingPoolerProbe`, the committed HTM coincidence kernel). For consistency + the fully-spiking directive, the relational dimension should run on the project's validated spiking FHRR — the `RFPhasorComposer` (resonate-and-fire phasor neurons + complex synapses). This feeds the brain's OWN real-corpus co-occurrence codes as the composer's grounded concept phases and runs SVO store/query on the RF substrate.

## The mechanism
- Each concept's real-corpus code → a D=64 phase vector in [0,1] (fixed random complex projection → angle) = the composer's `grounded_codes`.
- `RFPhasorComposer(vocab=words, grounded_codes=..., enable_substrate_store=True)` — the RF complex-synapse store (spiking).
- `store(agent, action, patient)` binds the fact through complex synapses; `query_patient(agent, action)` recovers the object; `query_agent(action, patient)` recovers the subject; the no-confab moat abstains when no stored fact matches the cue.

## The result — 3-seed (K=64, D=64, RF complex-synapse store, cupy)
```
seed 42/43/44: what_acc=1.000 | who_acc=1.000 | MOAT abstain=1.000 | permuted=0.000
AGGREGATE: what=1.000 who=1.000 MOAT=1.000 permuted=0.000  -> GO
```
- **what** (object recovery) + **who** (subject recovery) both 1.000 — the full relational algebra runs on the RF substrate over real-corpus codes.
- **MOAT abstain 1.000** — an unstored (subject, verb) cue → abstain (no confabulation).
- **permuted 0.000** — a wrong-verb cue recovers nothing (the verb binding is load-bearing).
Also confirmed on the numpy-KB store path (what/who/moat 1.0), so the algebra holds independent of the store backend; the `--substrate` run is the genuinely-spiking RF complex-synapse store.

## What this establishes
BOTH of the talkable brain's knowledge dimensions now have spiking realizations over the brain's own real-corpus codes: **property** (inheritance + cancellation) on the committed HTM coincidence kernel (`CancellingPoolerProbe`), and **relational** (SVO what/who) on the RF resonate-and-fire + complex-synapse composer (`RFPhasorComposer`) — both with the no-confab moat. The relational dimension is fully-spiking-realizable on the validated substrate. Follow-on: co-execute both spiking reasoners + the spiking A→W in one process (the one-backend consolidation); scale the substrate store.

## Files
`research/runners/_realcorpus_svo_spiking_derisk.py`; `research/findings/raw/_svo_spiking.json`. Prior: the rate relational Q&A `2026-07-08-relational-SVO-QA-over-real-corpus-codes-GO.md`; the RFPhasorComposer (the project's spiking FHRR-on-bridge composer); the step-3 grounded-codes-into-composer pattern.
