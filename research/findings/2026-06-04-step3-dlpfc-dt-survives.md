---
type: finding
status: live
date: 2026-06-04
mechanism: dlpfc-wm
---

# One-bridge unification step 3 — Task 1 DECISION de-risk: dlPFC working-memory bistability SURVIVES dt=1.0 → MERGE — 2026-06-04

**Verdict: MERGE.** The dlPFC dialogue-planning working-memory loop's PERSISTENT ACTIVITY (its working-memory
"latch") survives at **dt=1.0 ms** — the parser+composer timestep — when the loop is operated in its
**genuinely NMDA-dependent bistability regime**. The dlPFC therefore CAN merge onto the unified
`SimulationBridge` at dt=1.0 (step-3 Task 2 — merging the dlPFC `cortex_ctx`/`dlpfc_wm` regions onto the
unified bridge — is available). This is the opposite of the boundary that was a real possible outcome; the
test was built to the falsifiable NMDA-dependent regime and a collapse would have failed it honestly.

## The one question

A single `SimulationBridge` has ONE integration timestep `dt`. The parser + composer (steps 1–2) run at
**dt=1.0 ms** (NMDA OFF). The dlPFC working memory (`content_selection_spiking.build_loop_wm_bridge`: a
`cortex_ctx ↔ dlpfc_wm` reverberatory loop, NMDA ON) is tuned to **dt=0.5 ms** — its persistent activity (the
driven neurons keep firing AFTER the input drive is removed, holding a concept active for spreading-activation
dialogue planning) is sustained by NMDA + loop reverberation, which *might* need the finer timestep.
**Does the latch survive dt=1.0?** If yes → MERGE. If no → honest BOUNDARY (working-memory timescale ≠ binding
timescale; the dlPFC stays a separate-timing region).

## Result — multi-seed (42/43/44), the genuinely NMDA-dependent regime (attractor weight 30)

Per-dt rates are per-neuron firing fractions of the driven concept assembly (50 neurons), averaged over the
window. DRIVE window = 60 steps holding the concept's `cortex_ctx` pattern at 2500 pA; POST-DRIVE window = 80
steps with NO input (the persistence). The no-drive baseline is the same assembly on a fresh identical bridge
that is never driven.

| seed | dt=0.5 NMDA-on: during / **post (PERSIST)** | dt=1.0 NMDA-on: during / **post (PERSIST)** | dt=0.5 NMDA-OFF post | no-drive baseline | dt=1.0 retains |
|---|---|---|---|---|---|
| 42 | 0.409 / **0.098** | 0.449 / **0.324** | 0.000 | 0.000 | **330%** |
| 43 | (during ~0.40) / **0.059** | (during ~0.41) / **0.304** | 0.000 | 0.000 | **513%** |
| 44 | (during ~0.40) / **0.106** | (during ~0.40) / **0.278** | 0.000 | 0.000 | **263%** |

- **Non-vacuity — the dt=0.5 latch is GENUINELY NMDA-DEPENDENT** (not trivial AMPA recurrence): with NMDA ON
  the latch persists (post 0.059–0.106); with NMDA OFF it collapses to **0.000** at all three seeds. This is
  the real working-memory mechanism (NMDA slow current + loop reverberation), the thing the dlPFC dialogue
  planning relies on.
- **Specificity:** an un-driven control assembly's post-drive rate is **0.000** at both timesteps — the
  persistence is concept-specific, not a global excitation.
- **THE DECISION:** at dt=1.0 the SAME NMDA-dependent latch persists — post-drive 0.278–0.324, which is
  **263–513% of the dt=0.5 rate**, decisively clearing the MERGE bar (dt=1.0 post-drive ≥ 70% of dt=0.5 AND
  clearly above the 0.000 baseline). The latch at dt=1.0 is **still NMDA-dependent** (dt=1.0 NMDA-on/off post
  ratio 6.9× (seed 42), 608× (43), ~2.8e5× (44); NMDA-off post 0.000–0.047) — so it is the real WM latch at
  dt=1.0, not AMPA recurrence.

`run_dlpfc_dt_probe(seed=42)` console (the committed numbers):

```
          config   driven_during  driven_post(PERSIST)   control_post   no_drive_base
-------------------------------------------------------------------------------------
  dt=0.5 NMDA-on          0.4090                0.0982         0.0000          0.0000
  dt=1.0 NMDA-on          0.4490                0.3238         0.0000          0.0000
 dt=0.5 NMDA-OFF          0.3387                0.0000         0.0000          0.0000

Non-vacuity (NMDA-dependence) check at dt=0.5: NMDA-on post=0.0982 vs NMDA-OFF post=0.0000 -> the latch is GENUINELY NMDA-DEPENDENT.
MERGE bar (>=70% of dt=0.5 persist AND > baseline): 0.0688
dt=1.0 retains 329.5% of the dt=0.5 post-drive persistence.
DECISION: MERGE
```

## The load-bearing methodology call — operate in the NMDA-DEPENDENT regime, not the module's weight-50 attractor

The first cut of the probe used the module's installed-attractor weight (50.0 — the `SpikingLoopContextBuffer`
value). At weight 50 the probe reported MERGE with dt=1.0 retaining 100% of dt=0.5 — but a control sweep
exposed that result as **the wrong question**: at weight 50 the post-drive "persistence" SURVIVES even with
**NMDA OFF** (post ≈ 0.329), i.e. it is trivial saturated AMPA ping-pong around the strong reciprocal
attractor, not the NMDA-dependent latch the dlPFC actually uses. A weight-50 probe would have reported MERGE
for an AMPA recurrence that has nothing to do with the dlPFC's bistability, and the dt sweep would be
meaningless.

A weight sweep located the clean transition into the genuinely NMDA-dependent regime:

| attractor weight | dt=0.5 NMDA-ON post | dt=0.5 NMDA-OFF post | regime |
|---|---|---|---|
| 5–15 | ~0.000 | ~0.000 | sub-threshold (no latch even with NMDA) |
| 20 | 0.018 | 0.000 | NMDA-dependent, weak |
| **30** | **0.098** | **0.000** | **NMDA-dependent latch — the real WM mechanism (probe operating point)** |
| 50 (module value) | 0.333 | 0.329 | trivial AMPA recurrence (NMDA-independent — WRONG question) |

The probe is pinned at weight **30** and asserts the NMDA-OFF collapse as a non-vacuity guard, so the test can
only pass on the genuine WM mechanism. This is the faithful regime: it is exactly where "does NMDA + loop
reverberation hold a concept after the drive is gone?" is the question, and exactly where the dt sweep
decides something real.

## Why dt=1.0 persistence is STRONGER than dt=0.5 (not just surviving)

The NMDA conductance per-step decay is dt-scaled correctly: `decay_nmda = exp(-dt_ms / nmda_tau_decay)` (see
`sim/bridge.py` `_cached_decay_nmda`), so per unit *time* the NMDA conductance decays identically at both
timesteps. What differs is the neuron integration + spike machinery, which advances once per step: at dt=1.0
each step covers 2× the time, so the membrane integrates more depolarization per step and the loop ping-pongs
more readily — the coarser timestep FAVORS sustained firing here. The honest read is therefore even stronger
than "survives": the dlPFC's reverberatory WM latch is at least as robust at dt=1.0 as at dt=0.5. (This also
means dt=1.0 is comfortably away from the sub-threshold edge — the merge is not knife-edge.)

## What this de-risks (and what it does NOT)

- **De-risked:** the timestep conflict that made step 3 "the last + hardest." The dlPFC's working-memory
  bistability does NOT require dt=0.5; it holds at the parser/composer's dt=1.0. The merge is timing-feasible.
- **NOT claimed here:** the actual dlPFC merge (step-3 Task 2 — adding the `cortex_ctx`/`dlpfc_wm` regions as
  persistent index slices on the unified bridge and routing `elaborate` through them) is separate work. This
  task only answers the decision question. The validated dialogue planning
  (`BrainConversationalAgent.elaborate` via `SpikingSpreadingController`) is untouched and remains the
  regression oracle for Task 2.
- **Scope of the regime:** the merge should wire the dlPFC loop's installed concept attractors at the
  NMDA-dependent magnitude (≈30 in this probe's units), not at a saturated weight, so the merged region holds
  concepts via the genuine NMDA mechanism rather than AMPA ping-pong. Task 2 should re-confirm `elaborate`
  parity at dt=1.0 on the unified bridge (the spreading-activation Control already runs `turn_latency`, which
  is a fresh-probe latency read — its dt sensitivity should be checked end-to-end there).

## Artifacts

- Probe: `research/findings/raw/_step3_dlpfc_dt_probe.py` — `run_dlpfc_dt_probe(seed)` returns
  `{0.5: {...}, 1.0: {...}, "0.5_nmda_off": {...}}`; `python -m research.findings.raw._step3_dlpfc_dt_probe
  --seed 42` prints the table + decision. Carries a runner-side `dt`+`nmda`-parameterized copy of
  `build_loop_wm_bridge` (the shipped one hardcodes dt_ms=0.5, NMDA on) — **no `sim/` edit and no
  `content_selection_spiking.py` edit**.
- Test: `tests/test_unified_brain_bridge.py::test_step3_dlpfc_bistability_survives_dt1` — asserts the measured
  MERGE truth (drive lands; dt=0.5 NMDA-dependent persistence; specificity; dt=1.0 clears the MERGE bar). A
  dt=1.0 collapse would have flipped the assert to fail and the decision to BOUNDARY. SPIKING; runs on the
  CuPy/GPU backend; skips gracefully without a GPU.
- Backend: CuPy / RTX 3090 (the substrate's persistence dynamics are GPU-bound; NumPy diverges).

## Decision

**MERGE.** Proceed to step-3 Task 2 (merge the dlPFC onto the unified bridge at dt=1.0). The "working-memory
timescale ≠ binding timescale" boundary did NOT materialize — within this substrate, the dlPFC's
NMDA-dependent reverberatory working memory and the parser/composer's binding share dt=1.0.
