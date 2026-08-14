---
type: finding
status: live
date: 2026-08-13
mechanism: onebrain-composer-pool1-production-wire
---

# The RECALL COMPOSER can JOIN production POOL #1 (surprise + world-model) — opt-in, DEFAULT-OFF (GO)

**Date:** 2026-08-13 · **Status:** GO (opt-in join; DEFAULT-OFF this rung).
`seed-waiver: this is a byte-IDENTITY + no-regression WIRE verification — exact off-vs-on equality (max delta
0.0) of the recall/moat/surprise/world-model reads through the real handler, a DETERMINISM claim, not a seeded
statistical effect; the statistical robustness of the underlying merge is the cited 6-seed de-risk
(2026-08-13-onebrain-composer-pool1-merge-GO). No-regression is determinism 9/9.`

The production RF-phasor RECALL COMPOSER (+ its phase→spike TRANSDUCER cleanup region) can now JOIN production
pool #1 on ONE shared `SimulationBridge` — one `cp_membrane_potential_v` with the D2 SURPRISE organ + the E2
affective WORLD-MODEL organ — behind a NEW **default-OFF** flag `BRAIN_COMPOSER_MERGE`. Verified BYTE-IDENTICAL
through the REAL brain-chat handler path (flag OFF vs ON): the composer recall + the no-confab MOAT, the surprise
reads, and the world-model reads are all byte-identical, both spiking faculties stay ALIVE, it is genuinely ONE
pool when ON, and there is NO regression with the flag off (determinism 9/9, smoke). This is a FIRST
production-integration rung for the composer; DEFAULT stays OFF because the composer is the core no-confab MOAT
organ — the default flip is the NEXT rung (its blocker is mapped below).

## What this enables (the mission)

The 6-seed de-risk (`2026-08-13-onebrain-composer-pool1-merge-GO.md`,
`research/runners/_onebrain_composer_pool1_merge_derisk.py`) proved the composer + surprise + world-model can
share one pool byte-identically. This lane makes that REACHABLE from the production `/api/brain-chat` recall
path behind a default-off flag (opt-in; the default turn is unchanged):

- **`onebrain_merge_production.py`** — `composer_merge_enabled()` (`BRAIN_COMPOSER_MERGE`,
  `_COMPOSER_IN_POOL1_DEFAULT_ON=False`). When ON, `get_merged_substrate` builds the pool with organs
  `("surprise","worldmodel","composer","cleanup")` — the composer + transducer-cleanup regions APPENDED after
  the two organs (name-keyed per-region init ⇒ the surprise/world-model slices are index- and byte-identical to
  the 2-organ pool). `Pool1BoundComposer(RFPhasorComposer)` runs the recall's RF resonate ops on a masked SLICE
  of the shared bridge (the de-risk `SharedBridgeComposer` index-shift mechanism), with a graceful private-bridge
  fallback for an op larger than the region. `make_pool1_composer(**rf_kwargs)` builds it bound to the pool.
- **`brain_conversational_agent.py`** — the `composer_kind='rf'` construction builds `make_pool1_composer(...)`
  when the flag is ON, else `RFPhasorComposer(...)` exactly as today (default-off ⇒ byte-identical construction).

## Result (`_onebrain_composer_pool1_production_verify.py --compare`, real handler, flag OFF vs ON)

The verify builds the brain the SAME way the server does (`_build_tiny_demo(composer_kind='rf')`, the brain
`/api/brain-chat` builds when `BRAIN_COMPOSER_KIND=rf`) and drives the REAL production reads: recall
(`what_does`→`composer.query_patient`) + moat, `SurpriseProductionOrgan.judge`, and
`WorldModelProductionOrgan.expectation`/`read_surprise`. The flag is read at first-build of a process-global
singleton, so OFF and ON run in SEPARATE subprocesses and are diffed.

| Axis | Verdict | Detail |
|---|---|---|
| composer recall byte-identical (OFF vs ON) | GO | who/what answers identical: `{dog|chase→cat, cat|eat→fish, brain|use→spikes, brain|learn→words, brain|store→memory}` |
| recall CORRECT (== stored) | GO | every stored cue returns its stored patient |
| no-confab MOAT abstains (unstored) | GO | `lion|roar → None`, `owl|eat → None`, both OFF and ON |
| SURPRISE reads byte-identical | GO | per-case surprise_hz max delta **0.0 Hz**, `surprised` bool identical |
| surprise faculty ALIVE | GO | confirm 0.00 Hz vs contradict 4.98 Hz on the ON pool |
| WORLD-MODEL reads byte-identical | GO | pred_sign + pool rates + surprise_hz max delta **0.0 Hz** |
| world-model faculty ALIVE | GO | expected 0.00 Hz vs violated 44.44 Hz; predicted-valence sign correct |
| **GENUINELY ONE POOL when ON** | GO | composer `_pool1.bridge` **IS** surprise `_shared.bridge` **IS** world-model `_st['bridge']`; N=6064 = surprise + world-model + composer(4096) + cleanup(384), one `cp_membrane_potential_v` |
| **PRODUCTION WIRE GO** | **GO** | byte-identical recall + moat + surprise + world-model, both alive, one pool |

**NO regression (flag OFF, the production default):** `pytest tests/test_determinism.py -q` → **9 passed**;
`brain_chat_tui --smoke` runs; pool #1 (surprise + world-model) stays DEFAULT-ON and unchanged (the flag-off
`get_merged_substrate` builds the exact 2-organ pool as before).

## Why byte-identity holds

Inherited from the de-risk: the composer's RF ops (masked, they bypass `_run_one_simulation_step` and write only
the composer slice) are invariant to the three Izhikevich organs, and a masked shared-slice RF op reproduces a
dedicated per-op RF bridge bit-for-bit — so recall + moat equal a standalone `RFPhasorComposer`. The surprise +
world-model organs have disjoint region names, NO cross synapse in the byte-identity config, and both merge
flags ON (`per_region_threshold_heterogeneity` name-keys each slice's init; `per_region_homeostasis_isolation`
freezes idle co-residents) — so appending the composer + cleanup regions leaves their reads byte-identical to
the 2-organ pool. The cleanup (transducer) region is present but IDLE in the byte-identical production turn (the
recall→surprise cross synapse is not wired here — see residual 3).

## Honest scope / residual (what blocks the DEFAULT flip)

1. **The production-DEFAULT composer is `OneBrainComposer` (`composer_kind='onebrain'`), NOT the RF-phasor path
   wired here.** OneBrainComposer builds its OWN large co-resident `SimulationBridge` (parser + a big RF region +
   the persistent store + Q registers + batched cleanup, N≈P+…), and its recall runs on THAT bridge via
   `_read_blocks`, not through the inner `RFPhasorComposer._resonate`. This rung wires + verifies the **RF-phasor
   composer** (`composer_kind='rf'`) onto pool #1 — the exact object the de-risk validated. The default flip needs
   EITHER (a) the OneBrainComposer's own bridge joining pool #1 (a larger, un-de-risked layout merge — a NEXT
   rung), OR (b) production defaulting to the rf composer on pool #1. Kept default-off; the moat is not touched.
2. **Fixed composer-region SIZING.** The composer region is `max(7, 2·kmax)·D` (kmax=16, D=128 = 4096 neurons).
   A store whose K-fact batched moat scan (`2·K·D`) exceeds that FALLS BACK to a private per-op RF bridge —
   byte-identical recall, but off-pool for that op. A fixed shared region cannot hold an unbounded batched scan;
   dynamic region growth (recruit-an-assembly on the shared bridge) is the named next step. The verification
   panel (≤5 facts) fits entirely on the shared slice ⇒ genuinely one pool.
3. **The recall→surprise cross synapse is NOT wired in the byte-identical production turn.** The transducer
   cleanup region is on the pool (the four-code object), but wiring the `cleanup→surprise` edge live would change
   surprise (not byte-identical), so it is a separate BEHAVIOURAL rung (de-risked: recall drives the edge, GO
   6/6). Byte-identity here is of the recall + moat + the two faculties' reads.

## Read-out

⇒ the production RF-phasor recall composer (+ its transducer cleanup region) can run on pool #1's shared spiking
bridge alongside surprise + world-model, byte-identically through the real handler, moat intact, both faculties
alive, one pool, no regression — behind `BRAIN_COMPOSER_MERGE` (default-off). The composer is WIRED to pool #1
(opt-in); on_by_default = NO (first rung, core organ). The default flip is gated on residual 1 (the
OneBrainComposer's own bridge joining, or defaulting to the rf composer) + residual 2 (dynamic sizing).

CI/repro: `SIM_BACKEND=numpy python -m research.runners._onebrain_composer_pool1_production_verify --compare
--out research/findings/raw/_onebrain_composer_pool1_production_verify.json`. Runner:
`research/runners/_onebrain_composer_pool1_production_verify.py`. Wire:
`research/runners/onebrain_merge_production.py` (`composer_merge_enabled`, `Pool1BoundComposer`,
`make_pool1_composer`) + `research/runners/brain_conversational_agent.py`. De-risk:
`2026-08-13-onebrain-composer-pool1-merge-GO.md`.
