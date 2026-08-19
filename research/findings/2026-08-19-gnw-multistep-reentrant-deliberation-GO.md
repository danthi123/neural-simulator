---
status: go
type: finding
lane: T1-1
date: 2026-08-19
---
# THE KEYSTONE'S DEFERRED RUNG, WIRED LIVE (T1-1 rung d): a MULTI-STEP re-entrant deliberation loop on the LIVE brain-chat recall path whose re-entrant CYCLE COUNT emerges from the substrate's OWN spiking ignition read — DEFAULT-OFF de-risk, 6/6-seed GO, adversarially verified

**Date:** 2026-08-19 · **Runners:** [`webapp/gnw_multistep_deliberation.py`](../../webapp/gnw_multistep_deliberation.py) (the live wiring) + [`research/runners/_gnw_multistep_reentrant_deliberation_derisk.py`](runners/_gnw_multistep_reentrant_deliberation_derisk.py) (the GO-gate verify) · **Artifact:** `research/findings/raw/_gnw_multistep_reentrant/summary.json` (+ `.prov.json`) · **Scope:** additive DEFAULT-OFF de-risk (`BRAIN_GNW_MULTISTEP`), reuse-by-import, `NO sim/ edit` (`git diff sim/` empty); one additive default-off install hook in `webapp/server.py`. FUNCTIONAL correlate only; NO phenomenal claim.

## Verdict: GO (6/6 seeds; gates A/B/C/D; the two "it's not really substrate control" anti-cheats dissociate)

The single-hop deliberation gate ([`2026-08-18-gnw-deliberation-wired-brain-chat-GO.md`](2026-08-18-gnw-deliberation-wired-brain-chat-GO.md)) decides ONCE (halt-if-unsure) and EXPLICITLY named the multi-hop "deliberation-until-sure over a CHAIN" as the DEFERRED rung — it "stays the de-risk fixture". This closes that rung on the LIVE path: on an explicit chase-form question the P1.2 GNW workspace cycles the partial answer back through itself (re-entrant broadcast), re-igniting, and the substrate's OWN spiking read (`n_ignited` off `cp_firing_states`) decides how many cycles to run — replacing the host `query_chain(cue, actions)` counter — halting when the leaf collapses ignition. The brain works through a multi-step transitive inference whose DEPTH it discovers itself, LIVE.

## Why it is the keystone's missing half

`RFPhasorComposer.query_chain(cue, actions)` (GO 2026-06-17) already follows a chain — but for a HOST-FIXED number of hops `len(actions)`. The keystone de-risk ([`2026-08-18-gnw-reentrant-metacog-gated-deliberation-GO-caveat.md`](2026-08-18-gnw-reentrant-metacog-gated-deliberation-GO-caveat.md), 6/6 GO) moved that count to a spiking `n_ignited` read — but only OFFLINE, on its own synthetic composer. Neither ran on the LIVE production recall path. This landing wires the OFFLINE GO's `confidence_gated_chase` onto the production ChatBrain + the real `/api/brain-chat` handler, verified end-to-end.

**Biology grounding (external).** Serial multi-step cognition is assembled by routing individual parallel neural steps through the global workspace as a SEQUENCE of content activations, with re-entrant broadcast between steps (Zylberberg, Dehaene, Roelfsema & Sigman 2011, "The human Turing machine: a neural framework for mental programs", Trends Cogn Sci 15(7):293-300, PubMed 21696998). Cycling the partial answer back through the workspace until the substrate's own ignition read signals convergence is exactly that step-sequence; here the number of steps is set by the spiking read, not a host program counter.

## Mechanism (reuse-by-import; the ONLY structural change is host-count -> substrate-read, now LIVE)

`webapp/gnw_multistep_deliberation.py` wraps `chat.gate` (already the bus + single-hop-deliberation gate). Per turn:
- **DETECT** (comprehension of the teacher/world utterance — the declared boundary the SVO parser already occupies): a small set of explicit "follow the relation to the end" markers ("all the way", "to the end", "eventually", ...). No marker -> pure pass-through (byte-identical). The cognition (the inference + when to stop) is 100% the substrate.
- **EXTRACT** (agent, action): strip the marker, run the UNCHANGED inner gate on the clean "what does <agent> <action>?" so the production pipeline extracts (agent, action) + its first hop + runs its side effects EXACTLY ONCE; reuse that (agent, action) as the chase (cue, relation).
- **CHASE** (the substrate): the imported keystone `confidence_gated_chase` drives `composer.query_patient(x, action)` (the declared modular-processor boundary) into the P1.2 workspace, WTA-ignites one winner, reads `n_ignited` off the spikes, and the keystone `acc_conflict_gate` decides ADVANCE (broadcast the winner back, next hop) vs COMMIT (`n_ignited==0` at the leaf -> the terminal reached) vs ABSTAIN. The cycle count EMERGES; H_cap is a pure safety budget correct answers never reach.

The warm P1.2 workspace + self-calibrated theta are SHARED with the single-hop deliberation gate (`webapp.gnw_deliberation._get_bridge`/`_get_theta`). `git diff sim/` is empty.

## GO task — variable-depth transitive chase on the LIVE composer, depth NEVER told

16 chains of mixed depth L in {1,2,3,4} under one relation (chase), each edge taught via the PRODUCTION `chat.inner.hear` acquisition path; cue = ch[0], answer = the terminal leaf. L=1 is the one-step control (single-pass correct); L>=2 needs re-entry. The loop keeps re-entering while a single slot ignites and HALTS when `query_patient` misses at the leaf -> the workspace reads `n_ignited==0`.

## RESULT — 6/6 seeds (rule >=5/6), pooled + per-seed identical

| metric | gate | mean | per-seed 42/43/44/100/101/102 |
|---|---|---|---|
| reentrant_acc (variable depth, depth not told) | (B) >=0.90 | 1.00 | 1.00 all six |
| single-pass acc (k=1, the wired single-hop bus) | (C2) <=0.30 | 0.25 | 0.25 all six |
| reentrant per-depth (L1/L2/L3/L4) | all >=0.90 | 1.00 | 1.00/1.00/1.00/1.00 all six |
| single-pass per-depth (L1 kept / L>=2 failed) | L1>=0.99, L>=2<=0.10 | — | L1=1.00, L2=L3=L4=0.00 all six |
| spearman(resolved_hops, true depth) | (C2) >=0.9 | 1.00 | 1.00 all six; halt_at_H_cap=False all |
| workspace-silence LESION acc | (C1) <=0.10 | 0.00 | 0.00 all six |
| 1-hop reflex acc (survives the lesion) | (C1) >=0.85 | 1.00 | 1.00 all six |
| moat (unstored cue + over-run -> abstain) | (B) True | True | True all six |

**(A) LIVE end-to-end.** Gate level: OFF (single-hop bus) reaches only the FIRST hop (`[zorp, chase, blib]`); ON reaches the chain TERMINAL (`[zorp, chase, munt]`, resolved_hops=3, cycles=4). Through the REAL `/api/brain-chat` handler: ON answer "The zorp chases munt" (recalled_svo terminal); OFF "The zorp chases blib" (first hop); LESION "I don't know" (abstains — the recurrence-zeroed workspace collapses convergence).

## Anti-cheats — the two dissociations that ARE the deliverable

- **(C1) EMERGENT stopping, not host-counted.** Lesioning the substrate read that gates convergence (recurrence-zeroed workspace) collapses the multi-step chase to 0.00 on every seed while the 1-hop reflex (a workspace-independent `query_patient` read) survives at 1.00 — a full dissociation. If a host counter drove the cycle count, the lesion could not remove the multi-step answer while leaving the reflex intact. Verdict `control`: treatment (intact reent 1.00) vs control (lesion 0.00), |sep|=1.00 > 0.5.
- **(C2) RE-ENTRY is load-bearing + DIFFICULTY-GRADED.** Ablating re-entry (force single-pass k=1) leaves the one-step (L=1) answer UNCHANGED (1.00) but DEGRADES every multi-step depth to 0.00; and the emergent cycle count grows with difficulty (resolved_hops == depth, spearman=1.00, no halt at the safety cap). A genuinely multi-step problem takes MORE cycles than a one-step one, and removing the re-entry provably hurts only the multi-step problems.

## Adversarial verify (tools.verdict) — 3 lenses

- **Same-quantity (SURVIVES):** reentrant, single-pass, and lesion accuracies are the SAME 16 chains, SAME scoring (`== ch[-1]`), SAME `confidence_gated_chase` ignition, SAME per-hop distractor RNG; the ONLY difference is the stop rule (`while n_ignited` vs `range(1)` vs recurrence-zeroed). Single-pass IS the wired single-hop bus baseline.
- **No-host-orchestration (SURVIVES):** the chase is never passed L/the chain/the depth; the stop is `acc_conflict_gate(conf, n_ignited, ...)` reading spikes. Determinism: `cfg.seed` seeds the workspace substrate (build-twice thresholds identical). `git diff sim/` empty.
- **Byte-identical + moat (SURVIVES):** DEFAULT-OFF the gate is not installed -> byte-identical; even ON it is INERT on every non-chase-form turn (the reactive recall/abstain/learn/anaphora panel is byte-identical in-process AND through the real handler md5, BRAIN_GNW_MULTISTEP=1 vs =0). It only ever ADDS a terminal on a chase-form question; never un-abstains, never invents a fact; an unstored cue / over-run past a leaf -> abstain.

## Honest residuals (declared, not faked)

- **DEFAULT-OFF landing.** This is wired onto the live gate as an additive default-off de-risk (`BRAIN_GNW_MULTISTEP`); it is NOT on-by-default and the single-hop bus stays the default. The flip to on-by-default (a production-integration ledger move + a behavioural lesion probe) is the named next step, NOT claimed here.
- **PROPOSE is a declared modular-processor boundary** (`composer.query_patient`), same as P1.2 / the keystone / the coincidence integrator. The terminal is upstream-caused by its miss at the leaf; the substrate's independent work — and the whole novelty — is the CYCLE COUNT / when-to-halt moving from a host counter to a spiking `n_ignited` read, LIVE. Per the keystone caveat, the DECISIVE spiking read is `n_ignited` (the ignition/CONFLICT count), NOT the graded NMDA-balance `conf` (a binary redundant on this fixture).
- **Per-hop-reset form only** (snapshot-restore wash-out; the continuous no-reset train-of-thought is gated on the unbuilt Rung-2b async attractor).
- **Chase-form DETECT is host comprehension** of the teacher/world utterance (the declared boundary the SVO question parser already occupies), not a new shortcut; the surface phrasing "zorp chases munt" is a rendering of "the chase-chain from zorp ends at munt" — the substrate claim is the terminal + the emergent hop count in the trace.
- **Ceiling on reentrant_acc / reflex_acc (advisory).** Both sit at 1.00 on every seed (a ceiling). The DISCRIMINATING evidence is NOT the absolute 1.00 but the contrasts — lesion 0.00, single-pass 0.25 — and the Verdict `control` records treatment 1.00 vs control 0.00 (|sep|=1.00). `attributable_to(reent, lesion)` = 1.00: the whole multi-step answer is owed to the workspace ignition read (the recurrence-zeroed lesion removes all of it). The discriminating-power gate flags the ceiling as advisory; the verdict rests on the separation, not the ceiling.
- This is re-entrant multi-step deliberation with the MEASURED improvement, NOT "reasoning to a true conclusion".

## Backend / determinism (measured)

Decisive 6-seed on `SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf` (the fast production recall composer; `Pool1BoundComposer.query_patient` composes taught chains across hops). The workspace is the P1.2 390-neuron Izhikevich build; determinism from `cfg.seed`.

Cites: single-hop deliberation wired [`2026-08-18-gnw-deliberation-wired-brain-chat-GO.md`](2026-08-18-gnw-deliberation-wired-brain-chat-GO.md); keystone offline [`2026-08-18-gnw-reentrant-metacog-gated-deliberation-GO-caveat.md`](2026-08-18-gnw-reentrant-metacog-gated-deliberation-GO-caveat.md); P1.2 GO [`2026-07-24-P1.2-GNW-workspace-deliberation-6seed-GO-adversarially-verified.md`](2026-07-24-P1.2-GNW-workspace-deliberation-6seed-GO-adversarially-verified.md); multihop query-chain (host-counted) 2026-06-17. External: Zylberberg, Dehaene, Roelfsema & Sigman 2011, Trends Cogn Sci 15(7):293-300 (PubMed 21696998).
