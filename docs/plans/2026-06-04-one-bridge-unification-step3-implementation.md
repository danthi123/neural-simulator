# One-Bridge Unification — Step 3 (dlPFC merge) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Bring the third functional region — the dlPFC (dialogue-planning) working-memory loop — onto the SAME
`SimulationBridge` as the parser + composer, so all three conversational regions are one interacting brain.

**The hard constraint (why this is step 3, the last + hardest):** one bridge has ONE `dt` (integration timestep).
The parser + composer (steps 1–2) run at **dt=1.0 ms** with NMDA OFF. The dlPFC's bistable working memory
(`content_selection_spiking.build_loop_wm_bridge`: a `cortex_ctx ↔ dlpfc_wm` reverberatory loop, NMDA ON) is tuned
to **dt=0.5 ms** — its persistent activity (the "latch" that holds a concept active for spreading-activation
dialogue planning) depends on NMDA + loop reverberation, which may need the finer timestep. So step 3 has TWO honest
outcomes, and Task 1 decides which:
- **Merge:** if the dlPFC's bistability survives a dt=1.0 re-tune, the dlPFC regions join the unified bridge.
- **Honest boundary:** if the bistability cannot survive dt=1.0, the finding is *"working-memory timescale ≠ binding
  timescale"* — the dlPFC stays a separate-timing region (the validated dialogue planning is NOT broken to force a
  merge). This is a real biology-translatable result, not a failure.

**Tech Stack:** Python, `sim.bridge.SimulationBridge`, the brain-region framework (`BrainRegion`/`RegionPathway` —
the dlPFC loop uses it), `content_selection_spiking` (the dlPFC + spreading-activation Control), CuPy/NumPy, pytest.
New code in `research/runners/`; no `sim/` (protected) edits unless a task proves it strictly necessary (flag + stop).

**Standing gate:** every task ends green (the 10 on-brain + the unified tests). The validated dialogue planning
(`elaborate`, the content-selection controlled-coherence result) is NEVER weakened to force a merge. Honest outcomes
to both remotes. Plain professional language. GPU/CuPy for real runs (numpy only tiny smoke).

**Terms:** *bistability / persistent activity* = the loop keeps a concept's neurons firing AFTER the input drive is
removed (working memory); *dt* = the simulation timestep (ms per step); *NMDA* = a slow voltage-dependent synaptic
current that sustains reverberation; *dlPFC loop* = `cortex_ctx ↔ dlpfc_wm` mutually-exciting regions.

---

## Task 1: De-risk — does the dlPFC loop's bistability survive at dt=1.0? (THE decision)

**This decides step 3. Do it before any merge code.**

**Files:** Create probe `research/findings/raw/_step3_dlpfc_dt_probe.py`; Test `tests/test_unified_brain_bridge.py`
(append).

**Step 1 — Write the failing test.** Build the dlPFC working-memory loop (`content_selection_spiking.build_loop_wm_bridge`)
at **dt=0.5** (its tuned baseline) AND at **dt=1.0**. For each: drive a concept pattern into the loop for a window,
then REMOVE the drive and run a post-drive window; measure whether the loop sustains PERSISTENT activity (the driven
neurons keep firing well above baseline after the drive is gone). Assert: (a) at dt=0.5 it persists (the baseline
must work, else the probe is wrong); (b) the dt=1.0 persistence is the QUANTITY being decided — record its
post-drive sustained rate vs the dt=0.5 rate. The PASS criterion for a MERGE: dt=1.0 sustains persistence comparable
to dt=0.5 (e.g. post-drive rate ≥ ~70% of the dt=0.5 post-drive rate, and clearly above the no-drive baseline). A
FAIL (boundary) is dt=1.0 collapsing to baseline (no persistence).

**Step 2 — Run; expect FAIL** (probe not built).

**Step 3 — Implement** the probe: `build_loop_wm_bridge(dt=...)` — read the function; it currently hardcodes
`cfg.dt_ms=0.5`, so parameterize a `dt` (runner-side, or set `cfg.dt_ms` before init). Drive a concept (a set of
loop neurons) for ~N steps, remove drive, measure the sustained firing over the next ~M steps. Compare dt=1.0 vs
dt=0.5. (If NMDA kinetics in the bridge are dt-scaled correctly, dt=1.0 may persist; if the loop needs the finer
step for stable reverberation, it will decay.)

**Step 4 — Run + DECIDE:**
- **dt=1.0 persists (MERGE path)** → write `research/findings/2026-06-04-step3-dlpfc-dt-survives.md`; proceed to
  Task 2 (merge the dlPFC onto the unified bridge at dt=1.0).
- **dt=1.0 collapses (BOUNDARY path)** → write `research/findings/2026-06-04-step3-dlpfc-dt-BOUNDARY.md` honestly:
  the dlPFC's working-memory bistability requires dt=0.5; it cannot share the parser/composer's dt=1.0 bridge; the
  dlPFC stays a separate-timing region. This is the biology-translatable result (working-memory vs binding
  timescales). **STOP** — do NOT force a merge; surface to the controller + owner. Step 3 concludes as an honest
  boundary (steps 1+2 — the core conversational loop on one bridge — stand).

**Step 5 — Commit** (probe + test + the finding); push both remotes.

---

## Task 2 (ONLY if Task 1 = MERGE): dlPFC regions onto the unified bridge

**Files:** Modify `research/runners/unified_brain_bridge.py`; Test `tests/test_unified_brain_bridge.py`.

**Step 1 — Failing test:** a `UnifiedBrainBridge(enable_dlpfc=True)` holds the parser + composer + the dlPFC loop
regions on ONE bridge; `u.elaborate(topic)` returns an on-topic associate (the dlPFC spreading-activation Control
runs on the shared bridge), and it matches the separate-dlPFC `BrainConversationalAgent.elaborate` behavior
(on-topic associate from the agent's own facts; abstains on an unconnected topic). The parser + composer capability
is unaffected (re-assert the end-to-end test).

**Step 2 — Run; expect FAIL.**

**Step 3 — Implement:** add the dlPFC `cortex_ctx`/`dlpfc_wm` regions as further index slices on the unified bridge
(via the brain-region framework or explicit wiring at dt=1.0), wire the loop, and route `elaborate` through them.
This is involved — the dlPFC currently builds its own bridge per call; on the unified bridge it is persistent
regions. Keep `elaborate`'s association-graph construction as-is for now (the graph is built from the agent's facts;
making it substrate-native is out of scope). Keep the separate-dlPFC path as the regression oracle.

**Step 4 — Run; expect PASS.** **Step 5 — Commit + push both remotes.**

---

## Task 3 (ONLY if Task 2 done): no-regression confirm + B complete

**Files:** probe + Test (append, skip-by-default heavy if needed). Confirm `elaborate` on the unified bridge matches
the separate path (multi-seed if cheap), and the parser+composer capability matrix is unchanged. PASS → write
`research/findings/2026-06-04-one-bridge-unification-COMPLETE.md` (all three functional regions on ONE interacting
bridge — B done). Commit + push.

---

## Final review + handoff
After Task 1's decision (merge-complete OR honest boundary): dispatch a code-quality review of the step-3 diff;
confirm no `sim/` edits (or the one flagged); tests green; both remotes carry every outcome. Surface the outcome to
the owner — either "B complete (all three regions one bridge)" or "B core complete (steps 1+2); dlPFC is a
documented separate-timing boundary." Do not start new work without the owner's steer.
