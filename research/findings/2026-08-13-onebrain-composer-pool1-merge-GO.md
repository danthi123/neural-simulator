---
type: finding
status: live
date: 2026-08-13
mechanism: onebrain-composer-pool1-merge
---

# One-brain merge — the COMPOSER (+ transducer cleanup) joins POOL #1 (SURPRISE + WORLD-MODEL) on ONE pool (GO)

**Date:** 2026-08-13 · **Status:** GO (6-seed 42/43/44/100/101/102). The recall COMPOSER + its phase→spike
TRANSDUCER cleanup region + the D2 SURPRISE organ + the E2 affective WORLD-MODEL organ all run on ONE shared
`SimulationBridge` (one `cp_membrane_potential_v`, N=2248). All THREE reads are byte-identical merged-vs-
co-resident (composer recall + moat, surprise, world-model — **max delta 0.0 Hz**), both spiking faculties still
SEPARATE, determinism holds, and the composer's RF-phasor recall drives a load-bearing `cleanup→surprise`
cross-organ synapse WITH the world-model also co-resident in the pool. A **DE-RISK, NOT a production flip.**

## What this de-risks (the mission)

Production pool #1 (`onebrain_merge_production.py`, `MergedSubstrate`) already puts the surprise + world-model
organs on one bridge. Separately, the composer was merged with the surprise organ
(`2026-08-13-onebrain-composer-merge-GO.md`) and its recall was made to drive a cross-organ synapse via a
phase→spike transducer (`2026-08-13-onebrain-composer-transducer-GO.md`). The natural next step this lane
de-risks: the composer (+ its transducer cleanup region) JOINING pool #1 — the composer + surprise + world-model
all on ONE shared pool. Can three organs' reads stay byte-identical on one pool, moat intact, with the
recall→surprise interaction still load-bearing next to the world-model? This runner measures it.

## The four codes on ONE pool

- **COMPOSER (recall):** the production `RFPhasorComposer` on a masked SLICE (`SharedBridgeComposer`). RF-phasor
  resonate-and-fire ops bypass `_run_one_simulation_step`; masked writes touch only the composer slice; the
  no-confab MOAT abstains on unstored cues.
- **CLEANUP (phase→spike transducer):** V=9 word-blocks of 24 Izhikevich WTA neurons on the shared pool. Driven
  by the recall's input-normalized matched-filter scores; the winner block SPIKES → the same-code
  `cleanup→surprise` synapse.
- **SURPRISE:** the D2 expectation-violation organ (cue → patient_expected(FS,GABA_A) → surprise ←
  patient_asserted(exc)), Izhikevich + Hebbian + homeostasis + the two merge flags.
- **WORLD-MODEL:** the E2 affective forward model (state --learned-transition--> pred_{pos,neg}(FS,GABA_A);
  obs_{pos,neg}(exc) → surprise_{pos,neg} ← pred_{pos,neg}(inh)), same config, DISJOINT region names.

## Result (`_onebrain_composer_pool1_merge_derisk.py`, 6-seed; `--seeds 42,43,44,100,101,102`)

Facts `[dog→chase→cat, owl→eat→mouse, wolf→hunt→deer]`; unstored cue `lion roar` → the moat must abstain.

| Axis | Verdict | Detail |
|---|---|---|
| one shared neuron pool (surprise+world-model+composer+cleanup) | 6/6 | N=2248 = 1056 surprise + 528 world-model + 448 composer + 216 cleanup, one `cp_membrane_potential_v`, composer + cleanup spans contiguous |
| determinism (`cfg.seed` incl. thresholds) | 6/6 | two fresh full-pool builds at one seed → identical v / connections / thresholds |
| SURPRISE read byte-identical (merged vs co-resident) | 6/6 | max err 0.0 Hz over confirm/contradict/novel per-fact rates |
| surprise faculty alive (contradict ≫ confirm) | 6/6 | separation 5.4–61.3× (byte-identical of a LIVE organ); confirm 0.09–0.98 Hz vs contradict 5.11–5.32 Hz |
| COMPOSER recall byte-identical + CORRECT | 6/6 | `['cat','mouse','deer']` == standalone `RFPhasorComposer`, every seed; == stored patients |
| no-confab MOAT preserved (unstored → abstain) | 6/6 | shared `query_patient('lion','roar')` == None == isolated |
| composer op byte-ISOLATED from surprise slice | 6/6 | a composer store+query leaves the surprise slice v/u/thresholds byte-identical (interleave max err 0.0) |
| WORLD-MODEL read byte-identical (merged vs co-resident) | 6/6 | max err 0.0 Hz over expected/violated per-state rates; pred_acc identical |
| world-model faculty alive (violated ≫ expected) | 6/6 | expected 0.00 Hz (clean cancel) vs violated 28.2–43.3 Hz; predicted-valence sign accuracy 6/6 states, all seeds |
| **THREE-WAY BYTE-IDENTITY (composer + surprise + world-model)** | **6/6** | all three reads max delta 0.0 on one pool |
| **RECALL DRIVES the cross-organ synapse (world-model co-resident)** | **6/6** | recall→surprise interaction **+37.5…+78.7 Hz**, lesion `cleanup→surprise` −0.93…+0.00 Hz, attribution frac 1.00–1.02, cleanup winner 180–199 Hz |
| **POOL #1 JOIN GO** | **6/6** | one pool + determinism + three-way byte-identity + both faculties alive + recall-drives-edge |

## Why byte-identity holds (the mechanism, inherited from pool #1 + the composer merge)

The four organs read through DIFFERENT machinery on the SAME `cp_membrane_potential_v`. The composer's RF ops
never call `_run_one_simulation_step` and (masked) write only the composer slice → recall + moat invariant to
the three Izhikevich organs. The surprise and world-model organs have DISJOINT region names, NO cross synapse in
the byte-identity config, and both merge flags ON: `per_region_threshold_heterogeneity` makes each slice's
per-neuron init NAME-keyed (invariant to co-residents / build order), and `per_region_homeostasis_isolation`
freezes an idle co-resident's neurons so they do not drift during the active organ's read. Each organ trains +
reads ONLY its own regions, so every read reproduces the co-resident-with-flags (single-organ) read
bit-for-bit. The world-model byte-identity is checked on a SEPARATE fresh full-pool build trained ONLY on the
world-model, so surprise-training Hebbian cannot drift the world-model's plastic `state→pred` edges (a
training-order confound, not a merge failure — the clean claim is per-organ isolation, exactly pool #1's
protocol).

## The recall-driven cross-organ synapse (with the world-model in the pool)

A CONFIRM read of surprise fact i (surprise ~0 when the prediction cancels the assertion), with the `cleanup`
region driven by the composer's RF-phasor RECALL of fact i, raises surprise BLOCK i by +37.5…+78.7 Hz. The
recall's matched-filter WTA winner (== the recalled patient word, 6/6) SPIKES on the shared pool (180–199 Hz)
and the topographic `cleanup→surprise` synapse carries it. LESIONING `cleanup→surprise` (weight→0,
`plastic=False`) collapses the interaction to −0.93…+0.00 Hz (attribution frac 1.00–1.02 via
`tools.lab.attributable_to`). So the recall drives the cross-organ edge on the shared substrate WITH the
world-model organ also co-resident — the transducer GO reproduced inside the three-organ pool.

## Anti-cheats

- **Genuinely one pool:** one `cp_membrane_potential_v` holds all four organs' neurons (N=2248 = 1056+528+448+
  216), composer + cleanup spans contiguous, every region index < N (6/6).
- **Byte-identity in the DATA:** surprise + world-model per-condition max err 0.0 Hz (exact per-fact/per-state
  compare), composer recall exact string compare vs a standalone `RFPhasorComposer`, determinism by hashing v /
  connections / thresholds. `tools.lab.void_if` VOIDs the byte-id arms if any read produced an empty
  per-condition list (an empty list would make the max-delta 0.0 spuriously — UNDEFINED, not a pass).
- **Both faculties ALIVE, not exact-of-dead:** surprise contradict ≫ confirm (5.4–61.3×) and world-model
  violated ≫ expected (28–43 Hz vs 0 Hz, predicted-valence sign 6/6) — the byte-identity is of LIVE organs.
- **Moat preserved AND op-isolated:** unstored cue abstains (None == standalone); a composer store+query leaves
  the surprise slice byte-identical (interleave max err 0.0).
- **Load-bearing cross synapse:** the recall→surprise rise collapses to ~0 on lesion (frac 1.00–1.02), measured
  on the block the recall's winner topographically targets (the like-for-like read).
- Determinism via `cfg.seed` (not `actual_seed_used`). 6 seeds; smoke first.

## Honest scope

1. **This is a DE-RISK, not a production flip.** The composer stays its own bridge in the live chat turn; this
   proves the composer can share one pool with pool #1's two organs byte-safely. Per `docs/TERMS.md`, "closed"
   requires production integration (reachable + on-by-default + scaffold-retired) — this is a 6-seed GO de-risk,
   the next rung toward the composer joining the production pool.
2. **Per-organ read isolation is the claim** (each organ trained/read alone on the pool, co-residents restored),
   exactly pool #1's protocol — not a claim that all four are trained simultaneously with cross-plasticity. A
   single-config shared pool where three organs' reads are byte-identical is the de-risked object.
3. **The world-model byte-identity is measured merged-vs-co-resident-with-flags** (both on the same
   construction path), which isolates the merge itself. The numeric world-model Hz differs from a raw
   `build_world_model_circuit` standalone (the characterized cost of a shared pool's one global RNG draw, per
   pool #1's flip finding); the classification (violated ≫ expected, predicted sign) is preserved.
4. **The composer's fact-store here is the numpy-kb idealization** (its documented principled idealization; the
   RF resonate/unbind ops themselves ARE on the shared bridge), and the transducer inherits the validated
   spiking-cleanup's documented Stage1→Stage2 host read+normalize residual.
5. **The cross weight (8.0) and WTA drive scale (win_pa=600) are synaptic-gain parameters** (within the working
   range; `hebbian_max_weight=45`). The interaction is linear in the weight and collapses to 0 on lesion.

## Read-out

⇒ the recall COMPOSER (+ its phase→spike transducer cleanup region), the SURPRISE organ, and the affective
WORLD-MODEL organ all run on ONE shared spiking pool, with all three reads byte-identical merged-vs-co-resident
(max delta 0.0), the moat preserved, both faculties alive, determinism intact, and the composer's recall
driving a load-bearing `cleanup→surprise` synapse next to the world-model. The composer can join pool #1 on one
substrate. The next rung is the production flip of the composer into pool #1 (a de-risk here, not yet flipped).

CI/repro: `SIM_BACKEND=numpy python -m research.runners._onebrain_composer_pool1_merge_derisk --seeds
42,43,44,100,101,102 --out research/findings/raw/_onebrain_composer_pool1_merge_6seed.json`. Runner:
`research/runners/_onebrain_composer_pool1_merge_derisk.py` (`--seed`, `--seeds`, `--D-cmp`, `--cblk`,
`--cross-weight`, `--win-pa`). De-risk chain: composer+surprise MERGE
(`2026-08-13-onebrain-composer-merge-GO.md`) → the recall drives the cross synapse via the phase→spike
TRANSDUCER (`2026-08-13-onebrain-composer-transducer-GO.md`) → THIS (the composer + transducer join pool #1's
surprise + world-model on one pool). Pool #1: `onebrain_merge_production.py` (`MergedSubstrate`).
