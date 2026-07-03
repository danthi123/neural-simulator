# EMERGE-67 — the EMERGE-frame render's CONTENT WORDS are now produced ON SPIKES: the validated A→W read-out wired into the spiking-Broca producer's `spell` — **GO** (6-seed)

**2026-07-03 (autonomous).** The remaining host piece of the spiking-Broca render was the `spell` callback (a token-surface identity for the CPU de-risk). EMERGE-67 replaces it with the **validated spiking A→W** (concept-pool → spoken-word) read-out (`concept_speak_demo`, CLAUDE.md "chat_speak A→W 100% multi-seed"), so the CONTENT words (subject/verb = the emergent concepts) are **spelled on spikes**. The ORDER was already spiking (EMERGE-59/63 competitive queuing) → the EMERGE-frame render is now **fully spiking for the content slots**. Reuse-by-import; **NO `sim/` edit**.

## Mechanism

`NeuralSpell.spell(word)` DRIVES the word's concept pool on a real `SimulationBridge` and DECODES the spoken word from `cp_firing_states[language_output]` (cosine to the word patterns) — the validated A→W read-out. It is wired into the EMERGE-59 producer as `BrocaProducer(cq, spell=neural_spell)`; each CONTENT slot (SUBJ, VERB) spells via this spiking read-out. The A→W engine is **GPU-trained ONCE at the validated scale + cached** (`bridges/emerge67_aw/aw_content.simstate.h5`, regenerable via `--train`; a scale/data lever, not a new mechanism); the producer's 16 content words (8 subjects + 8 verbs) are rebound onto the 16 validated concept pools.

## De-risk — **GO** (6-seed 42/43/44/100/101/102; A→W on GPU/cupy, wire+moat CPU-safe)

| gate | value | bar |
|---|---|---|
| CONTENT-slot spike-spell accuracy (the content words decoded from spikes, vs ground-truth) | **1.000** every seed | ≥ 0.9 |
| raw A→W word-wise rate (the engine spells the 16 content words) | **1.000** (16/16) | — |
| **genuinely SPIKING** — LESION control (zero the pool→`language_output` pathway) collapses the decode | **0.097** (engine-lesion **0.000**) | ≪ main |
| NO regression vs the token spell (content surfaces identical) | **0 mismatches** | 0 |
| gate-first MOAT — spell-calls + producer-invocations on abstains | **0 / 0** | 0 |

**Genuinely spiking, not a host lookup:** the read-out reads real `cp_firing_states[language_output]` spikes; **lesioning the pool→`language_output` pathway collapses the decode to ~0.10** (a host dict lookup would be unaffected) — the load-bearing proof the words are decoded FROM SPIKES. The gate-first moat holds by construction: on an abstain the producer — and hence the A→W read-out — is NEVER invoked (0 spell-calls, 0 productions).

## Verdict

**GO.** The CONTENT words of the EMERGE-frame render are now produced ON SPIKES via the validated A→W read-out — "the owl can **fly**" has *owl* + *fly* decoded from `language_output` spikes. Combined with EMERGE-59/63's spiking ORDER, the EMERGE-frame render is **fully spiking except the named function-word follow-on**: the DET/FUNC slots (`the`/`can`/`does`/`not`) keep the token surface — their A→W pools are the closed class EMERGE-62 discovers, so wiring their spiking spell is the direct next step (EMERGE-68). Renders the BOUNDED EMERGE frame inventory, NOT open prose (R4, the deferred scale wall). NO `sim/` edit; the token-spell default path is byte-identical (EMERGE-59..66 all still pass).

## Files
- `research/runners/_emerge67_neural_spell_wirein_derisk.py` — `NeuralSpell` (the spiking A→W spell) + the wire + `--train`/`--demo`/`--derisk`.
- `tests/test_emerge67_neural_spell_wirein.py` — 6 CPU-safe (wire/moat/content-scoring/no-regression) + 1 GPU smoke (skip unless the process is `SIM_BACKEND=cupy` — the backend is process-sticky, so the GPU read-out can't run in the numpy CI process; the full GPU path is validated by the 6-seed `--derisk`).
- `research/findings/raw/_emerge67_neural_spell_wirein.json` — the 6-seed de-risk.
- Cached A→W engine `bridges/emerge67_aw/aw_content.simstate.h5` (local, gitignored `.h5`; regenerable via `--train`).

## Process note (orphaned-run recovery)
The building subagent GPU-trained the A→W engine but then **ended its turn "waiting" for the training** rather than running the de-risk — the recurring orphaned-detached-run failure mode (see `feedback_proactively_monitor_long_runs`). The training had in fact completed (the cache was written); the controller recovered by running `--derisk` against the cache to completion, writing this finding + fixing the GPU test's process-sticky skip-guard. Lesson reinforced: a GPU-training subagent must run the train+de-risk INLINE in its turn (or the controller does the GPU step), never detach-and-wait.
