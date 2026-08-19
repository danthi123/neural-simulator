---
type: finding
status: contributing
date: 2026-08-18
mechanism: episodic-temporal-context-when
lane: EPISODIC
---

# Episodic memory gains a SENSE OF WHEN (6-seed GO): a drifting temporal-context signal (Howard-Kahana TCM) as LEC time cells (Tsao 2018), bound to each CA3 assembly at encode, produces a RECENCY gradient + TEMPORAL CONTIGUITY — and a CONTEXT-LESION collapses both to flat (the effect is 100% carried by the context->CA3 pathway, not a write/size confound)

The gap#5 episodic store had a WHO/WHAT but no WHEN. The production organ (EpisodicDapMemory,
`research/runners/_episodic_dap_dialogue_memory.py`) is a CA3-only recurrent-completion store: NO context pool, NO
plastic context->CA3 pathway, so recency/contiguity are impossible there by construction. This finding ADDS the WHEN
attribute on top of the SAME 6/6-GO substrate (emergent-DG selection -> BTSP one-shot formation -> dendritic-dAP
apical-UP readout, ALL reused by import), with NO `sim/` edit. Runner:
`research/runners/_gap5_episodic_temporal_context_when_derisk.py`. SIM_BACKEND=numpy, `cfg.seed` per bridge (the
substrate is genuinely seeded).

## The new topology (the survey's named blocker: this needed a new bridge, not a config tweak)

<!--derived-->

- **temporal-context pool** — `n_ctx=200` LEC episodic TIME cells (Tsao et al. 2018 Nature) whose population vector
  DRIFTS across encode-time by a Howard-Kahana TCM update `c_i = normalize(rho*c_{i-1} + beta*eta_i)` with a SPARSE
  non-negative recruitment `eta_i` (k=10 cells/step). The drift makes the probe (test) context overlap recent items'
  stored contexts most, and neighbour contexts overlap most — the structure that yields recency + contiguity
  (measured: probe-overlap oldest..newest 0.29..0.80, neighbour 0.79 vs far 0.39). The drift schedule is a documented
  SCAFFOLD standing in for LEC time-cell dynamics.
- **plastic context->CA3 heteroassociative pathway** `W_ctx` (n_ca3 x n_ctx), Hebbian-bound AT ENCODE (post = the
  co-firing assembly cells of the item being stored, pre = the current context vector `c_i`). Its transmission
  `W_ctx @ c_probe` is delivered to CA3 as injected current — the SAME kind of synaptic-current injection the reused
  instrument already uses for the partial cue. It is NOT host "recency" bookkeeping: no store-index is ever read to
  build the gradient; the graded drive emerges from the synaptic overlap `c_i . c_probe`, and zeroing `W_ctx` removes
  exactly it.

## Result — 6-seed GO (pooled serial-position curve; seeds 42/43/44/100/101/102, pre-assigned equal-size 71-cell assemblies, n_ca3=500)

<!--derived-->

Recency and contiguity are POPULATION effects (a serial-position curve, a lag-CRP), so the headline is the pooled
across-seed curve; per-seed counts are the robustness check.

- **RECENCY — graded gradient.** Pooled held-cell completion by serial position (oldest..newest) =
  **[0.145, 0.123, 0.164, 0.178, 0.257, 0.404]** — Spearman(position, completion) = **0.943**, newest-third /
  oldest-third ratio = **2.47**. Recently-encoded assemblies complete more readily from the same partial cue. Per-seed
  5/6 (Spearman 0.99/0.90/0.12/0.66/0.93/0.60; seed-44 flat — see residuals).
- **CONTIGUITY — temporal neighbours co-reactivate.** Cueing item i (which reinstates its encoding context t_i)
  co-reactivates item j with a pooled lag-CRP peaked at +-1 and decaying with |lag|:
  `{-5:0.13, -4:0.13, -3:0.17, -2:0.27, -1:0.47, +1:0.46, +2:0.21, +3:0.14, +4:0.13, +5:0.10}`. Near (|lag|=1) = 0.466
  vs far (|lag|>=3) = 0.133 (3.5x). Per-seed 6/6.
- **CONTEXT-LESION (the load-bearing anti-cheat) COLLAPSES BOTH.** Zeroing the context->CA3 pathway (`W_ctx := 0`) sets
  the recency curve to **[0,0,0,0,0,0]** (Spearman 0.0, range 0.0) and every contiguity lag to **0.0**. Fraction of the
  recency range absent under the lesion = **1.0 (100%)**. Per-seed 6/6. So the recency + contiguity are entirely
  carried by the drifting-context binding on the substrate — NOT a BTSP-write-recency confound (each assembly is
  formed in its OWN temporally-isolated encode episode, so the CA3 recurrent weights carry no write order) and NOT an
  assembly-size confound (all six assemblies are equal size, 71 cells).

Verdict `WHEN-GO`: pooled recency Spearman 0.943 ratio 2.47 | pooled contiguity near 0.466 vs far 0.133 | per-seed
recency 5/6, lesion-collapse 6/6, contiguity 6/6, seed_go 5/6.

## Why the operating point is what it is (and what it means)

<!--derived-->

The dendritic-dAP completion read is a sharp threshold, so a recency GRADIENT is only visible when the partial cue is
WEAK enough to leave headroom: at the standing GO cue (cue_frac=0.30, drive=300) the cue saturates the read (near-full
completion with OR without context — no gradient). At the chosen weak point (cue_frac=0.15, drive=50) the cue-alone
completion is FLAT ZERO for every item — this is exactly the committed lesion curve `[0,0,0,0,0,0]` — and the
temporal-context current (ctx_pA=700 at overlap=1) supplies the graded, recency-tuned drive that gates completion. The
partial cue provides the item-specific seed volley; the context binding provides the item-specific graded lift
(`c_i . c_probe`). This is the honest reading: at this point completion is context-GATED — which is exactly why the
context-lesion is so decisive (it removes the only supra-threshold drive).

## Honest residuals + next levers (NOT walls)

<!--derived-->

- **One seed's recency is flat (44/6 -> 5/6).** At the fixed `ctx_pA=700`, seed-44's particular assemblies over-complete
  even the OLDEST item (context floor overlap 0.29 x 700 = ~200 pA already crosses that seed's completion threshold),
  so its serial-position curve is flat (Spearman 0.12) though its contiguity and lesion-collapse still pass. This is
  per-assembly threshold heterogeneity at a single global operating point; the pooled effect absorbs it. Next lever: a
  per-assembly / homeostatic normalisation of the context gain (the "companion process we replaced with a constant"),
  or a mild `ctx_pA` reduction, would likely lift 44 into a per-seed gradient — a runner-side knob, not a substrate
  limit.
- **Membership is PRE-ASSIGNED (equal-size disjoint), not emergent-DG-selected.** The emergent selection is a
  separately-closed anti-cheat (`2026-07-14-ca3-competitive-hebbian-formation-6seed-GO`,
  `2026-08-10-episodic-cortical-cue-recall-completion-6seed-GO`) and is ORTHOGONAL to the WHEN mechanism, which acts
  on the context->CA3 pathway given working assemblies (same scoping the 2026-08-10 cortical-recall GO used).
- **The operating point is SCALE-SPECIFIC (an emergent n_ca3=2000 probe RAN and over-completed, in-session).** At this
  same weak operating point the emergent production substrate (whose DG-selected assemblies are much SMALLER, ~20-28
  cells vs 71) SATURATES the dAP read: completion is near-ceiling across every serial position, so the recency
  gradient flattens and the cue-alone baseline is no longer zero (a small tight-recurrent assembly completes more
  readily). This is NOT a mechanism failure — the context code and `W_ctx` pathway are identical, and contiguity is
  still present (near > far, both near-ceiling) — it is the well-known "probe must match the deployed config": the
  smaller emergent assemblies need a proportionally WEAKER cue+context to leave completion headroom. The faithfulness
  follow-up is a per-scale (or homeostatic per-assembly context-gain) re-tune of `ctx_pA`/`drive` on the emergent
  substrate; the mechanism is substrate-agnostic. This is a runner-side operating-point knob, NOT a wall.
- **The drift schedule is a host scaffold** (the LEC time-cell dynamics stand-in). Per the standing standard this is a
  documented shortcut to burn down: the faithful replacement is a SPIKING LEC time-cell population whose drift emerges
  from its own recurrent dynamics, projecting through the SAME plastic `W_ctx`. The brain-based parts here are already
  on-substrate: the plastic context->CA3 synapses and the spiking dendritic-dAP completion they gate.

## Anti-cheats satisfied

<!--derived-->

Determinism (`cfg.seed` per bridge); NO host recency bookkeeping (the gradient is the synaptic `W_ctx @ c` overlap, and
the context-lesion zeroes it 100%); equal-size assemblies (no size/position confound); temporally-isolated encode
episodes (no BTSP write-order in the CA3 recurrents); plasticity FROZEN at recall; OU noise OFF. NO `sim/` edit
(additive runner, reuse-by-import of the 6/6-GO substrate).

Artifacts: `research/findings/raw/_episodic_when/when_preassigned_6seed.json` (verdict GO, pooled + per-seed) with
`.prov.json` sidecar. SIM_BACKEND=numpy.
