# Phase B — Spiking similarity-matching cortex on the bridge (64-concept proof) — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute this plan
> task-by-task (one fresh subagent per task, strict failing-test → minimal-impl → run → commit, controller
> trust-but-verify each diff). Owner approved the build (option A, 2026-06-15).

**Goal:** Build the simplified learned cortex (subtractive-inhibition centering + bounded Hebbian
feedforward, **no recurrent lateral**, PPMI-shaped spiking input) on the `SimulationBridge` and show it
recovers the real 64-concept category structure — capability-equivalent to the Phase-A numpy result
(+0.545) — multi-seed, full anti-cheat battery.

**Architecture:** A 2-region bridge — `hub` (H context-hub neurons, the "sensory" context) → `cortex`
(k output neurons, plastic). Each concept is presented by driving the hubs with its PPMI-shaped pattern
(log-compression + the Phase-1 dendritic divisive gain for the /marginal + threshold), common mode removed
by an explicit subtractive-inhibition front-end; the plastic hub→cortex synapses learn a structure-
preserving compression via the bridge's bounded (soft-bound STDP + homeostasis) Hebbian rule; the concept
code = the cortex spike-count vector over the readout window. Reuse the `_build_cortex_bridge` / `_present`
pattern from `research/runners/dendritic_cortex_forward_codes_derisk.py`, but make hub→cortex **plastic**
and **read after learning**.

**Tech Stack:** Python, `sim.bridge.SimulationBridge` + the brain-region framework, `SIM_BACKEND=cupy` (GPU)
for real-corpus runs (numpy only for tiny smoke). Metrics/anti-cheat reuse-by-import from this cycle's L1
runners (`_cos_sim`, `_pearson_vs_Strue`, `heldout_generalization`, `ppmi_matrix`, `center_cols`,
`build_real_corpus`, `ppmi_svd_sim`).

**The two load-bearing risks (front-loaded as cheap-first HARD GATES — Tasks 2 and 3):** (a) can the
**centering** (common-mode removal) be realized neurally on the bridge? (b) does the bridge's **spike-timing
STDP** realize the bounded-Hebbian structure extraction the numpy rule did? A NEGATIVE at either gate is the
honest Phase-B deliverable (it maps the rate→spike wall precisely) — propagate it and STOP rather than build
on a broken foundation.

**New file:** `research/runners/spiking_sm_cortex.py` (the builder + trainer + reader + validation).
**New test file:** `tests/test_spiking_sm_cortex.py`. **No sim/ edits expected** (the bounded Hebbian maps
to existing soft-bound STDP + homeostasis; subtractive inhibition to existing inhibitory machinery; the
/marginal to the already-shipped Phase-1 gain). If any sim/ edit IS needed: default-off guard + byte-review.

---

### Task 1: The cortex-bridge builder + PPMI-shaped input encoder

**Files:**
- Create: `research/runners/spiking_sm_cortex.py`
- Test: `tests/test_spiking_sm_cortex.py`

**Step 1 — failing test.** `test_build_and_encode`: `build_sm_cortex_bridge(n_hub=200, n_cortex=64, seed=42)`
returns `(bridge, hub_idx, cortex_idx)` with `len(hub_idx)==200`, `len(cortex_idx)==64`, the hub→cortex
pathway present and `plastic=True`; and `encode_drive(C_row, log=True)` returns a non-negative vector of
length H equal to `log1p(max(C_row,0))` (the Weber-Fechner input; the /marginal + threshold are applied by
the bridge's dendritic gain + rheobase, not here).

**Step 2 — run, expect fail** (`pytest tests/test_spiking_sm_cortex.py::test_build_and_encode -v`; ImportError).

**Step 3 — implement.** Adapt `_build_cortex_bridge`: regions `hub` + `cortex`; pathway
`RegionPathway(from_region="hub", to_region="cortex", density=…, weight_mean=small, weight_jitter, plastic=True, plasticity_gate="hub_to_cortex")`;
`cfg.enable_dendritic_divisive_gain=True` (the /marginal); `cfg.enable_stdp=True` with `cfg.stdp_w_max`
set above the design weight (the soft-bound gotcha); `cfg.enable_homeostasis=True`; OU off. `encode_drive`
returns `log1p(max(C_row,0))`.

**Step 4 — run, expect pass.**

**Step 5 — commit** (`feat(phaseB): cortex-bridge builder + PPMI-shaped input encoder`).

---

### Task 2: HARD GATE — neural common-mode removal (the centering)

**Files:** modify `research/runners/spiking_sm_cortex.py`; test `tests/test_spiking_sm_cortex.py`.

**Step 1 — failing test.** `test_centering_recovers_structure_forward` (numpy smoke, synthetic 64-concept
counts with a strong common mode): with a **fixed random** hub→cortex projection (learning OFF, isolating
the input path), driving the hubs WITH the subtractive-inhibition front-end enabled yields cortex codes whose
`Pearson(cos(codes), S_true)` clears a bar (e.g. ≥ +0.30) AND clearly exceeds the same pipeline with
centering OFF. (This mirrors Phase-A: centering is the load-bearing op; verify a *neural* mechanism realizes
it.)

**Step 2 — run, expect fail.**

**Step 3 — implement** `subtractive_inhibition`: a global (or per-hub) inhibitory mechanism subtracting the
slow population-mean hub drive — first try the bridge's existing **homeostatic threshold adaptation** (each
hub's running-activity baseline), warmed up over an all-concepts pass; if that under-performs the numpy
`center_cols`, add an explicit inhibitory interneuron pool (existing inhibitory region) that pools the hubs
and subtracts the mean. No host mean-subtraction in the code path (that would be a shortcut — it must be
neural).

**Step 4 — run, expect pass.** **HARD GATE:** if no neural mechanism clears the bar, STOP — write the
finding (the centering cannot be realized neurally on the bridge cheaply), propagate to both remotes, and
surface to the owner before proceeding.

**Step 5 — commit.**

---

### Task 3: HARD GATE — bridge STDP realizes bounded-Hebbian structure extraction

**Files:** modify `research/runners/spiking_sm_cortex.py`; test `tests/test_spiking_sm_cortex.py`.

**Step 1 — failing test.** `test_learned_codes_recover_structure` (numpy smoke, 64-concept synthetic):
`train_sm_cortex(bridge, C, n_epochs)` presents each concept (centered, log+gain+threshold drive) so the
plastic hub→cortex STDP grows; then `read_codes(bridge, C)` returns cortex spike-count codes whose
`Pearson(cos(codes), S_true)` clears a bar (≥ +0.30) AND beats an untrained random-projection control on the
identical pipeline by ≥ +0.10 (learning load-bearing).

**Step 2 — run, expect fail.**

**Step 3 — implement** `train_sm_cortex` (present-and-learn loop: drive hubs, run the integration window so
cortex fires, STDP potentiates co-active hub→cortex pairs; homeostasis + `stdp_w_max` bound the weights =
the firing-rate ceiling) + `read_codes` (freeze plasticity via the `hub_to_cortex` gate, present each
concept, accumulate cortex spike counts over the readout window). Tune drive_scale / window / epochs so
cortex fires in a workable band (not silent, not saturated).

**Step 4 — run, expect pass.** **HARD GATE:** if the bridge STDP cannot extract the structure (codes ≈
random), STOP — write the finding (the bridge rate→spike learning gap is the wall; localize: STDP timing vs
the bounded-Hebbian rate rule), propagate, surface to the owner. This is the riskiest gate (proposal risk #2).

**Step 5 — commit.**

---

### Task 4: Full anti-cheat battery + multi-seed on the REAL corpus (GPU)

**Files:** modify `research/runners/spiking_sm_cortex.py`; test `tests/test_spiking_sm_cortex.py`.

**Step 1 — failing test.** `test_validation_harness_shape` (numpy smoke): `validate(seeds=[42])` returns a
dict with keys `learned_pearson`, `random_proj_pearson`, `permuted_pearson`, `host_ceiling`, `gen`, `gates`
and the gates dict has `host_carries`, `learning_load_bearing`, `permuted_collapses`, `reaches_structure`,
`generalizes`.

**Step 2 — run, expect fail. Step 3 — implement** `validate`: builds the real 64-concept corpus
(`build_real_corpus`), trains the cortex, reads codes, computes the learned Pearson + the controls
(random-projection, permuted-similarity), the host PPMI+SVD ceiling, held-out generalization, and the gates.
**Step 4 — run, expect pass. Step 5 — commit.**

**Step 6 — the real run (GPU, controller-launched, not a unit test):**
`SIM_BACKEND=cupy python -u -m research.runners.spiking_sm_cortex --real-corpus --seeds 42,43,44 --out research/findings/raw/_phaseB_spiking_sm_cortex_multiseed.json`.
**Gate (GO):** learned Pearson ≥ 0.70 × host ceiling AND ≥ +0.30, multi-seed; learning load-bearing (≥ +0.10
over random projection); permuted ~0; generalizes above chance. Record wall-clock + GPU use.

---

### Task 5 (CONTROLLER-ONLY — bring back to the controller, not a subagent): the Phase-B gate + finding

Review the Task-4 real-run result against the Phase-A numpy ceiling (+0.545) and the gates. Write the
finding `research/findings/2026-06-15-phaseB-spiking-sm-cortex-<verdict>.md` (GO / BOUNDARY / NEGATIVE — all
three are the deliverable), update `AUTONOMOUS_STATE.md` + the build proposal, mark task #24, commit+push
**both remotes**, and surface to the owner: GO → proceed to Phase C (scale); NEGATIVE → the rate→spike wall
is mapped (the honest finding the de-risk could not reach without the bridge). Do NOT proceed to Phase C
without the gate passing + the owner informed.

---

## Standing rules for every task
GPU/CuPy for the real runs (numpy only for tiny smoke); commit+push BOTH remotes each task; never weaken the
no-confab moat (Phase D concern, not touched here); honest propagation of GO/BOUNDARY/NEGATIVE; the protected
set stays byte-empty unless an edit is strictly required (then default-off guard + byte-review); the two HARD
GATES (Tasks 2, 3) STOP the build on a NEGATIVE rather than building on a broken foundation.
