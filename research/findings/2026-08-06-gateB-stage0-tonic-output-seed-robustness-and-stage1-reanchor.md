---
type: finding
status: qualified
date: 2026-08-06
mechanism: gateB-v13-continuous-selector-stage0-tonic-output-substrate
backend: numpy+cupy
runner: research/runners/_vocal_gateb_stage0_seed_robustness.py
artifacts:
  - research/findings/raw/gateb_stage0_seed_robustness/numpy.json
  - research/findings/raw/gateb_stage0_seed_robustness/cupy.json
---

# Gate B Stage-0 tonic-output substrate is seed-robust and cross-backend; re-anchor to Stage 1

## Why this finding exists (state correction)

The mission board and my hand-off described Gate B as "v2 still in calibration".
That is stale. The vocal action-credit line ran through **v1..v14** on 2026-08-03/04.
The mission-relevant thread reached a correct, biology-grounded pivot at the
**v13 continuous-basal-ganglia-selector research gate**
(`2026-08-04-neural-vocal-credit-gateB-v13-continuous-bg-selector-RESEARCH-GATE.md`),
then **drifted** into byte-determinism debugging and finally the parked v14
ion-channel kinetics arc. This finding maps the real state, confirms the
Stage-0 foundation, and re-anchors the exact next mechanism.

## The drift map (so it is not re-derived)

- **v9** established a learned, dendritically expressed pre-reward expectation
  (`QUALIFIED-GO` engagement), but its **output was UNDEFINED**: after training,
  the neural selector never re-expressed the rewarded action.
- **v10** correctly moved plasticity onto the real corticostriatal policy routes,
  but its engagement smoke was `UNDEFINED`: under a sustained unchanged drive the
  losing motor channel also crossed, so no clean exclusive action existed.
- **v11/v12** added downstream neural action-boundary circuits (self-recurrent
  corollary; disinhibitory guard). Both are `CONSTRUCTION_NO_GO`, sharing one
  root cause: startup/warmup transients fire the boundary/motor before any
  action, and sustained competition still admits the loser. Two dead ends on the
  "add a boundary" decomposition.
- **v13 research gate** drew the right conclusion: **repair the selector, not its
  boundary.** Gate A's "clean action" was defined inside a host-controlled trial
  (Python stopped at first crossing, drove `selector_reset` at 1200 pA, imposed a
  100 ms washout, and fed 1000 pA tonic current to GPi). The fix is a
  **continuously-operating center-surround BG selector** with autonomous GPi/SNr
  output plus the missing hyperdirect (cortex/proposal to STN) and GPe to GPi/SNr
  pathways (Nambu 2002; Schmidt 2013; Mallet 2016; Nakanishi 1987; Kandel 6e ch38).
- **v13 Stage 0** (autonomous output substrate) was then implemented via the
  sanctioned reduced scaffold: a region-scoped constant `intrinsic_current_pA`
  on GPi/SNr Izhikevich neurons (`sim/regions.py`, `sim/bridge.py`;
  `docs/SCAFFOLD-LEDGER.md`). **Its physiology passed on both backends** (V6
  calibration/replication/held-out `HELD_OUT_GO`, `HELD_OUT_GO` cupy). It was left
  **unpromoted only by (a) a performance-overhead ceiling (`1.059` vs `1.02`, per `docs/SCAFFOLD-LEDGER.md`) <!--derived--> and (b) NumPy/CuPy byte-determinism** — both *engineering* gates, not science.
- **v13 backend spiral + v14** chased those two engineering gates: exact izh
  initialization, state transplant, backend arithmetic, then fitting real Na/Kv3
  channel kinetics to make the pacemaker biophysically faithful. That is the
  **parked P3 drift** (substrate-detail tunnel vision; speed is explicitly
  secondary in this project).

## What this run establishes

Artifacts: `research/findings/raw/gateb_stage0_seed_robustness/numpy.json` and
`research/findings/raw/gateb_stage0_seed_robustness/cupy.json` (each with a
`.prov.json` provenance sidecar).

`research/runners/_vocal_gateb_stage0_seed_robustness.py` reuses the locked v13
runner's own validated primitives (`build_tonic_bridge`, `_run_steps`,
`_physiology_metrics`) unchanged, and sweeps 12 fresh non-sealed seeds
(810001..810012) at the selected 100 pA drive, 1000 steps, both backends. This is
an engineering robustness smoke, not a formal capability partition, and consumes
no sealed seed.

| Backend | Seeds passing | Pop-rate range (Hz) | Only failing seed | Its failed check |
|---|---:|---|---:|---|
| numpy | `11/12` | `60.33-65.72` | `810002` | `same_step_fraction` |
| cupy | `11/12` | `61.92-65.78` | `810002` | `same_step_fraction` |

- The autonomous tonic-output phenotype (40-80 Hz all bins, all 40 cells fire,
  zero external current, intrinsic + weights immutable) holds at **11/12 fresh
  seeds on both backends**, with rates tightly clustered at 60-66 Hz.
- The single miss (seed 810002) fails **only** `max_same_step_fraction <= 0.25`
  (a brief synchronized volley), while still firing 40/40 cells tonically at
  ~65 Hz in-band. It is a mild synchrony transient, **not** a loss of autonomous
  tonic output.
- **NumPy and CuPy agree exactly** on the pass/fail partition (same seed, same
  failing check) across all 12 seeds. The prior `REPLICATION_NO_GO` /
  `COMPATIBILITY_NO_GO` were about *byte-identical* numerical determinism, not
  phenotype disagreement — the phenotype is cross-backend consistent.

## Verdict and decision

**QUALIFIED.** The Stage-0 autonomous-output substrate is functionally sound and
seed-robust for the mission's purpose: a continuously-firing, causally
suppressible GPi/SNr output that needs no per-step host current. The two gates
that held it unpromoted (a 4% overhead ceiling and byte-determinism) are
**secondary** under this project's standing rules (speed secondary; NumPy is a
valid backend and phenotype agreement holds). They should be recorded as
caveats, not treated as walls, and the ion-channel faithfulness arc stays parked.

Do **not** add a third boundary topology (v11/v12 already retired that
decomposition) and do **not** resume byte-determinism/perf grinding.

## Exact next mechanism (Stage 1)

Build the **continuous center-surround selector** per the v13 gate on the working
`intrinsic_current_pA` scaffold:

1. Start from Gate A v2 populations; **remove runtime `selector_reset` and the
   host GPi tonic current**; GPi/SNr runs on intrinsic drive from step 0.
2. Add, in mechanism order: proposal/cortex to shared **STN (hyperdirect hold)**,
   then **GPe to same-channel GPi/SNr**; add arkypallidal/pallidostriatal feedback
   only if initiation is clean but autonomous termination fails.
3. Keep direct/indirect/hyperdirect symmetric before learning; no pathway opened
   or closed after observing a winner; the circuit decides when an action begins
   and ends (no host stop-on-winner).
4. Construction gate (score from step 0): tonic GPi/SNr + inhibited thalamus at
   baseline; early STN/GPi rise on proposal onset; one focused GPi pause then one
   clean motor action; competitor suppressed for the full window; autonomous
   return to tonic output; **>=2 clean actions from one uninterrupted brain**;
   immutable weights; zero reset current; NumPy and CuPy both pass before seeds.
5. Only a continuous-selector construction pass reopens the **v10** local
   reward-credit question (Stage 2), which is the actual Gate B goal.
