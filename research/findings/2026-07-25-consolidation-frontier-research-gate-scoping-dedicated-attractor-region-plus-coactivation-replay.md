# Consolidation frontier — research-gate scoping: getting a COMPOSED fact from a hippocampal engram to a self-sustaining cortical concept assembly. Recommended surpass = a DEDICATED compositional-attractor region (strong Wang-2002 NMDA + WTA + SFA) + CO-ACTIVATION replay (2026-07-25)

**Deep-research gate output** (read-only, local corpus + RAG). Fired because compositional consolidation hit a confirmed
boundary (the compositional engram strands in hippocampus) and the next move is a mechanism to push past it.

## 1. Diagnosis — the two-part blocker, quantified (naive fixes REFUTED)
The composed engram (an SVO / adj+noun binding) is **stored but stranded** in hippocampus; it never reaches a lasting
cortical readout.
- **Part A — missing wire (built).** No `ca1→noun_pool` / `ca1→adjective_pool` existed; only `ca1→motor`/`ca1→lang_out`
  (3/3-validated for direct word→motor, Phase 1.3). Tag-stim concept-pool rate = **0.0015** (noise floor); direct
  bindings = 0.2–0.8 (2–3 orders higher); replay 0/20/60 cycles dead flat.
  (`2026-05-21-consolidation-probe-TERMINAL-...-no-ca1-to-concept-pool-consolidation-wire.md`)
- **Part B — the wire is NECESSARY BUT NOT SUFFICIENT (load-bearing NEGATIVE).** Building 12 `ca1→concept` pathways
  (density 0.20, w 2.0, plastic, mirroring `ca1→motor`) lifted drive 0.0015→**0.0073** but stayed ~3× below the 0.02
  bar and 30–100× below readable (0.2–0.8); replay moved it nowhere; selective 1/4, lifted 0/4.
  (`2026-05-22-ca1-concept-pool-variant-NEGATIVE-...-weak-dynamics-prevent-consolidation.md`)
- **The genuine residual (quantified).** Concept pools are built `density=0.05, exc_w=0.3`; motor pools (which DO
  consolidate) are `density=0.10, exc_w=2.0` — concept recurrence is **~7× weaker.** This is DELIBERATE: strong concept
  pools cause the documented "canon amplifies bias" failure (recurrence overwhelms topographic word-training; Phase-1
  multi-concept 88.75% depends on weak dynamics). Real tension: *weak → Phase-1-trainable but can't host a consolidated
  attractor; strong → holds the attractor but Phase-1 collapses.* Motor pools escape only because they host 4
  mutually-exclusive directions.
- **The A1 NMDA attempt already failed the naive way** (`2026-07-24-A1-consolidation-regression-nmda-slow-self-loops-break-direct-binding-...`): `nmda_slow` self-loops on the weak pools → **runaway single winner** (adjective_pool_SMALL
  captured every readout, direct binding 1/8 vs 6/8), compositional readout stayed at floor in BOTH arms, and the plastic
  `ca1→concept` wires **stayed at w=0.01 — never potentiated during replay.**
- **Root cause of non-potentiation (code read):** `consolidation_trainer.run_concept_replay_phase` drives ONLY the CA3
  tag (`stimulate_tag`); the weak `ca1→concept` (w≈0.01) can't fire the pools → no post-spike → STDP has no post to
  potentiate the wire → frozen at floor. Circular: wire won't potentiate until pools fire; pools won't fire until the
  wire potentiates / an attractor holds them.
- **⇒ any go-forward must simultaneously (1) co-activate ca1 + target during replay (fix potentiation) and (2) give the
  target a SELECTIVE, self-sustaining NMDA attractor that neither collapses Phase-1 nor becomes a single global winner.**

## 2. Reusable machinery
- **Phase 1.3 CLS consolidation** (hippo-OFF retention 94%, 3/3 strict) — the **hippo-lesion-after-consolidation**
  anti-cheat ("proves it's in cortex now"). `2026-05-08-Phase1.3-Tier2.1-strict-anti-cheat-3seed-CONFIRMED`.
- **Direction-Q Wang-2002 NMDA attractor op-point** — self-sustaining on this substrate at `nmda_ratio≥0.6, n=1000,
  density=0.20, inh=2.0` (3000 ms delay, ~650-750 Hz; off-control silent). The exact "strong target region" recipe.
  `2026-05-26-DIRECTION-Q-tertiary-NMDA-AMPA-ratio-sweep-PASS-Wang-2002-bistability-closed-at-nmda-ratio-0.6`.
- **P0.3 slow-NMDA attractor** — reusable BUT saturates on point neurons → a bistable LATCH / single winner (the same
  runaway A1 hit); must be paired with **lateral WTA + SFA-eviction** for one-of-N selectivity. `2026-07-24-P0.3-...-6seed-GO`.
- **The ready harness:** `research/runners/nmda_compositional_consolidation.py` — `build_substrate` (appends `ca1→concept`
  RegionPathways + optional `nmda_slow` self-loops, NO sim/ edit), `encode_facts_with_reinstatement`, `consolidate`,
  `hippo_lesioned`, `recall_adj_rates`, `diag_ca1_concept_selectivity`, `--skip-nmda`. Extend THIS.
- `consolidation_trainer.py` replay (must add target co-activation); gap#5 replay+phase-switch (`_gap5_onebrain_capstone.py`,
  `_gap5_wake_sleep_phase_switch.py`) as the SWR generator; `2026-07-20-single-shared-substrate-consolidation-coresidence-de-risk-GO`.

## 3. Biology
CLS (McClelland-McNaughton-O'Reilly 1995): hippocampus = fast one-shot sparse binder; neocortex = slow interleaved
schema learner (weak cortical learning rate = the biology's version of the weak pools). SWR replay (~20× compressed,
coupled to cortical slow-osc + thalamic spindles) is the ripple-window hand-off that potentiates cortical targets;
repeated replay trains the assembly until reinstatable WITHOUT hippocampus (Tse 2007 schema). A schema is only useful if
the cortical assembly completes + self-sustains (Wang-2002 recurrent NMDA), and avoids the single-global-winner pathology
via lateral inhibition (WTA) + SFA. ⇒ biology prescribes exactly: ca1→cortex + a strong WTA/SFA-shaped NMDA attractor +
ripple-coupled co-activation replay.

## 4. Recommended de-risk (Option 1 + Option 2 folded in) — cheapest-first
**Do NOT consolidate into the weak Phase-1 pools.** Add a SEPARATE `compositional_attractor` region (Direction-Q strong:
`nmda_ratio=0.6, density=0.20, inh=2.0`) + lateral WTA + SFA-eviction; wire `ca1→region` + `concept→region` strong; read
out THERE (weak pools + Phase-1 untouched). Drive it with **co-activation replay** (drive the CA3 tag AND reinstate its
noun+adj pools together so the wires potentiate). Offline on a cached Phase-1 substrate (no retrain), 1-seed smoke first.
Measure (a) SELECTIVE ignition (tag A vs B → distinct region sub-assemblies), (b) HOLD after drive-off (NMDA-on vs off).

**GO-gate (per-seed 42/43/44/100/101/102):** region readout **>0.02** (order above the 0.0073 weak ceiling); selectivity
EMERGES from replay (top-among-tags rises 0→20→60 cycles, ≥⌈N/2⌉-of-N); HOLD (NMDA-on ≥50% peak displacement ≥300 ms
post-drive, off <10%); survives **hippo-lesion-after-consolidation**; `ca1→region` weight rose off init.

**Anti-cheats (the result IS the anti-cheats):** no-region (→weak-pool floor) · no-NMDA (→no hold) · no-replay (→no
selectivity) · no-co-activation (→wire frozen at floor, reproduces A1) · hippo-lesion-after (→cortical) · permuted-tag
(→selectivity collapses) · control-outperforms-real guard.

**Honest expected bounded negative (still a deliverable):** single-global-winner (P0.3 saturation + A1 runaway) — a strong
point-neuron NMDA attractor may latch to one dominant sub-assembly, not N selective ones. If so, that names the
**SFA-eviction / dendritic line-attractor** surpass (the exact P0.3 open-risk) and is worth committing either way.

## Provenance
Deep-research gate (read-only), 2026-07-25. Findings cited inline. Reuse-by-import (extend
`nmda_compositional_consolidation.py`); no protected/moat module touched. Next: build the Option-1 de-risk.
