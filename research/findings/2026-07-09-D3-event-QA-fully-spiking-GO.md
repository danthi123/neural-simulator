# D3 EVENT QA → the FULLY-SPIKING read-out (6-seed GO): RANK-3 reasoning/QA over the composed running event, end-to-end on the spiking substrate

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_event_qa_spiking_derisk.py` (reuse-by-import: `factored_event_rnn` + `build_fswta_score_bridge`/`fswta_drive` + `build_fact_store`/`recency_resolved`; numpy backend, small Izhikevich bridges; NO `sim/` edit).
**Verdict:** GO (6-seed: dev 42/43/44 + blind 100/101/102).

## What this closes
The rate QA (`2026-07-09-D3-event-QA-unification-GO.md`) proved the RANK-3 payoff at rate. The non-negotiable is **fully-spiking-on-one-brain**, so this moves BOTH QA stages onto the project's spiking FS-WTA substrate — no host argmax anywhere in the QA path:

- **STAGE 1 — resolve the referent, SPIKING:** the running FACTORED event is re-discretized by two co-evolving FS-WTA Izhikevich attractor slots (the `_d3_event_spiking_derisk` port) → the resolved coref-DEEP agent, on spikes.
- **STAGE 2 — key the fact store, SPIKING:** the resolved entity's code drives a THIRD FS-WTA property bridge through a learned associative map (emb[entity]→property) → the answer property spikes out. This is the composer's associative-recall role realized as a spiking read-out (NEF/engram-style).

⇒ *"what does HE eat?"* is answered by SPIKES end-to-end: spiking-composed running agent → spiking associative recall of its stored property.

## Result (6-seed; NO `sim/` edit)
| | dev 42/43/44 | blind 100/101/102 |
|---|---|---|
| **SPIKING deep-agent QA** | 0.967 / 0.975 / 0.967 | 0.967 / 0.992 / 0.983 |
| RECENCY-QA (resolve→last-mentioned) | 0.367 / 0.333 / 0.442 | 0.350 / 0.325 / 0.375 |
| host-agree — STAGE-1 resolve | 0.998 / 0.998 / 0.999 | 0.995 / 0.995 / 0.998 |
| host-agree — STAGE-2 property | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |

Mean spiking QA **≈0.975** vs recency **≈0.365** (chance 1/K=0.167). Both read-out FS-WTAs are **faithful** to the host argmax (resolve ≥0.995, property 1.000) — the spiking read-outs ARE the QA, not a host shortcut checked against spikes.

## Anti-cheats (all pass)
- **(a) SPIKING QA ≫ chance** (0.975 vs 0.167) and == the rate QA (0.982).
- **(b) RECENCY FAILS** (0.365): a recency resolver (even on the spiking-composed event) keys the store with the wrong referent — the running agent is coref-deep.
- **(c) both read-out FS-WTAs FAITHFUL** (resolve host-agree ≥0.995; property host-agree 1.000): the spiking winners equal the host argmax, so the QA genuinely runs on the substrate.
- **(d) the fact store is keyed ON SPIKES** (STAGE-2 property FS-WTA): the property answer is produced by the spiking associative-recall read-out, not a host lookup.

## ⇒ the anti-RAG payoff, on the substrate
The situation-model × fact-store QA — resolve a coref-deep pronoun via the D3-composed running event, then recall the referent's stored property — now runs end-to-end on the project's spiking substrate, no host argmax in the QA path. This is the simulated recurrent sequence/language cortex **reasoning over a running meaning**.

## Honest scope + next
- **resolve + recall (the first RANK-3 rung), realized fully-spiking** — not yet multi-hop inference. The associative map (emb→property) is a learned linear read-out standing in for the composer's fact store; wiring the ACTUAL composer KB (bind/unbind or engram) is the deployment step.
- **Next:** multi-turn QA / connectives (chain two composed events across turns); wire the QA into the live `MultiTurnAgent` (the `D3EventRegister` already tracks the running event — add a QA method that resolves the pronoun via the register → queries the composer KB); the self-supervised TEM δ.

## Files
`research/runners/_d3_event_qa_spiking_derisk.py`; the rate rung `2026-07-09-D3-event-QA-unification-GO.md`; the event arc `2026-07-09-D3-event-*.md`.
