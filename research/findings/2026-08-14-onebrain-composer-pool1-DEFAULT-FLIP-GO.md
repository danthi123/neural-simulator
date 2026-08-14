---
type: finding
status: live
lane: track1
date: 2026-08-14
mechanism: onebrain-composer-pool1-DEFAULT-FLIP
---

# The PRODUCTION-DEFAULT composer JOINS pool #1 by DEFAULT — the DEFAULT FLIP (GO)

**Date:** 2026-08-14 · **Status:** GO — the flip is earned; `_COMPOSER_IN_POOL1_DEFAULT_ON=True`, so the SHIPPED
`OneBrainComposer` now joins pool #1 on the real production turn.
`seed-waiver: this is a byte-IDENTITY + no-regression DEFAULT-FLIP verification — exact DEFAULT-vs-ESCAPE equality
(max delta 0.0) of the recall/moat/surprise/world-model reads through the real handler, a DETERMINISM claim run over
6 seeds (42/43/44/100/101/102), not a seeded statistical effect. No-regression is determinism 9/9 + the smoke
byte-identical to the pre-flip baseline.`

The PRODUCTION-DEFAULT recall composer (`composer_kind='onebrain'`, `OneBrainComposer`) now JOINS production pool #1
— ONE shared `SimulationBridge` (`cp_membrane_potential_v`) with the D2 SURPRISE organ + the E2 affective
WORLD-MODEL organ — **by default**. This closes the residual the opt-in wire finding named
(`2026-08-13-onebrain-composer-pool1-production-wire-GO.md` §Honest scope, item 1): the bare flag was a NO-OP on the
real turn because the shipped composer is `OneBrainComposer`, NOT the RF-phasor path the wire wired. The b-closer
`Pool1BoundOneBrainComposer` routes the shipped composer's RF recall/store onto pool #1 (its parser on a private
bridge), so the DEFAULT turn now exercises a genuinely-shared pool — a Track-1 one-substrate spine step for the core
moat organ.

## The closer (why the shipped composer, not just the RF path)

`OneBrainComposer` runs TWO substrates on one bridge: (a) the RF who/what pipeline (config-INDEPENDENT
resonate-and-fire — `_rf_advance_one` reads only the per-op `_rf_omega/_rf_lambda/_rf_floor`, never the cfg's
Hebbian/homeostasis), and (b) a Hebbian PARSER (`BridgeParser`, Izhikevich `_run_one_simulation_step` over the WHOLE
bridge, trained at `hebbian_max_weight=400`). The RF pipeline ports onto pool #1's slice byte-identically (the
`CoResidentOneBrainComposer` index-shift, Probe-1 GO at atol 1e-9). The parser CANNOT: pool #1's config differs
(`hebbian_max_weight=45`, `per_region_homeostasis_isolation=True`) AND `_run_one_simulation_step` steps ALL neurons —
the parser would train differently (broken recall byte-identity) and advance + corrupt surprise/world-model (broken
criterion-4). So `Pool1BoundOneBrainComposer` builds a full standalone `OneBrainComposer` (parser trained on its
private big bridge; the complete layout incl. `vocab_headroom` + recruit slots), then REBASES only the RF layout
(`P/store_base/q_base/c_base/bat_q_base/bat_c_base += rf_base`, `n_total = pool_N`, `rf_mask`/`_rf_reset_mask` = the
composer's span on the pool, `self.b = pool.bridge`). The parser handle keeps pointing at the private bridge;
`hear()` comprehends there; the RF store/read run on the pool slice. The pool reserves an "onebrain_composer" region
of EXACTLY the standalone `n_total` (45856 at the tiny-demo vocab: D=128, V=144 with headroom, k_max=32, attribute
role → rf_base=1584, pool N=47440); every RF op fits by construction (a mis-size RAISES — never silently truncates).

Files:
- `onebrain_merge_production.py` — `_COMPOSER_IN_POOL1_DEFAULT_ON=True`; `Pool1BoundOneBrainComposer`
  (`_pool1_onebrain_class`/`_pool1_onebrain_init`); `_onebrain_layout_span`; `make_pool1_onebrain_composer`;
  `MergedSubstrate` "onebrain_composer" region + `onebrain_composer_idx()`; `get_merged_substrate` onebrain branch.
- `brain_conversational_agent.py` — the `composer_kind='onebrain'` construction routes through
  `make_pool1_onebrain_composer` when `composer_merge_enabled()`, else `OneBrainComposer(...)` exactly as today
  (default-on ⇒ the pool path; escape ⇒ byte-identical private construction).
- `_onebrain_composer_pool1_production_verify.py` — `--default-flip` mode (DEFAULT-no-env vs ESCAPE MERGE=0, the
  shipped onebrain path), 6 seeds.

## The verify (pre-registered, 6/6)

DEFAULT (no `BRAIN_COMPOSER_MERGE` env → reads the flag, ON → pool) vs ESCAPE (`BRAIN_COMPOSER_MERGE=0` → private),
two subprocesses per seed (the flag is read at first-build of a process-global singleton), through the REAL
brain-chat handler (`_build_tiny_demo(composer_kind='onebrain')`) + the real production reads (`what_does` →
`query_patient`, `SurpriseProductionOrgan.judge`, `WorldModelProductionOrgan.expectation`/`read_surprise`), 6 seeds
42/43/44/100/101/102, `SIM_BACKEND=numpy`. Artifact:
`research/findings/raw/_onebrain_composer_pool1_default_flip_6seed.json`.

| seed | (1) recall+qp byte-id + correct | (2) moat abstains | (3) one pool (N==v_len) | (4) surprise byte-id (maxerr Hz) / alive | (4) world-model byte-id (maxerr Hz) / alive | GO |
|---|---|---|---|---|---|---|
| 42  | True | True | True (47440) | 0.00 / 0.00→4.98 | 0.00 / 0.00→44.44 | GO |
| 43  | True | True | True (47440) | 0.00 / 0.17→5.50 | 0.00 / 0.00→47.57 | GO |
| 44  | True | True | True (47440) | 0.00 / 0.00→5.27 | 0.00 / 0.00→10.07 | GO |
| 100 | True | True | True (47440) | 0.00 / 0.00→5.15 | 0.00 / 0.00→34.72 | GO |
| 101 | True | True | True (47440) | 0.00 / 0.00→4.80 | 0.00 / 0.00→32.64 | GO |
| 102 | True | True | True (47440) | 0.00 / 0.35→5.38 | 0.00 / 0.00→34.38 | GO |

- **(5) determinism 9/9** with the flip default-ON: `pytest tests/test_determinism.py -q` → 9 passed (includes
  `TestSubstrateActuallySeeded`).
- **(6) no-regression** on `brain_chat_tui --smoke --stub-renderer`: JSON byte-identical to the pre-flip baseline
  (the smoke uses the RF path; its pool-join is the 6/6-GO RF wire, so byte-identity holds).

**GO-gate: 6/6 seeds pass criteria 1-4 AND criteria 5-6 pass ⇒ flip earned.**

**Residual: 0.0.** The composer read is answer-byte-identical (the FHRR phase read is magnitude-invariant + the
decode argmax winner is identical on the rebased slice, Probe-1 atol 1e-9); surprise/world-model reads are
numerically byte-identical (max err 0.00 Hz on every seed — name-keyed per-region init makes the 2-organ and 3-organ
pools byte-identical for the two organs' slices, and the composer's masked RF writes leave their v/u untouched). No
shared-pool threshold re-draw occurs for the composer slice (its neurons are RF, not thresholded), so criterion 1
holds at exact byte-identity, not merely answer-preservation.

## What stays a host shortcut / the next lever

- The composer's PARSER stays on a PRIVATE bridge (not on pool #1) — a genuine config conflict
  (`hebbian_max_weight` 45-vs-400 + `per_region_homeostasis_isolation`) plus the whole-bridge Izhikevich step.
  Putting the parser on the pool is a distinct, owner-scoped merge (a config-superset like the surprise+comprehension
  merge, OR a per-region-config engine feature), not a defeat: the parser is a comprehension front-end analogous to
  the RF path's separate agent parser, so binding the RECALL substrate to the pool is the same "one pool" level the
  RF wire achieved. This is the named next lever.
- No cross-organ synapse (recall→surprise) is wired here (the next behavioural rung), so surprise/world-model stay
  byte-identical.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._onebrain_composer_pool1_production_verify --default-flip \
    --seeds 42,43,44,100,101,102 \
    --out research/findings/raw/_onebrain_composer_pool1_default_flip_6seed.json
SIM_BACKEND=numpy python -m pytest tests/test_determinism.py -q          # 9/9
SIM_BACKEND=numpy python -m research.runners.brain_chat_tui --smoke --stub-renderer   # byte-id to pre-flip
```
