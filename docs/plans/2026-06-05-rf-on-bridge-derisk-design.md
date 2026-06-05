# Resonate-and-fire ON the SimulationBridge — minimal de-risk design + TDD plan — 2026-06-05

> **For Claude:** this is the owner-FUNDED Option-A first step (FHRR phasor pivot). Scope = the **minimal contained
> de-risk PROOF** that the bridge can natively host resonate-and-fire (RF) phasor neurons, NOT the full FHRR feature.
> First protected `sim/` edit of the arc — keep it minimal, contained, guarded, and FLAG the diff for owner review.

**Goal:** prove a `RESONATE_AND_FIRE` neuron model on the `SimulationBridge` (the bridge's own neurons, in its own
step) reads out phase as spike timing, composes (bind/unbind/bundle via complex kicks), and clears the project's
frozen 0.80 compositional bar at loads 2/3/5 — at parity with the numpy reference `research/runners/resonate_fire_fhrr.py`.

**Why:** the opponency is a confirmed rate-coded SNR wall (`2026-06-05-B-opponency-rate-coded-SNR-wall-CONFIRMED.md`).
FHRR avoids it structurally (representation GO, perfect at all loads). On-bridge realization needs an RF substrate the
bridge lacks (`2026-06-05-FHRR-pivot-derisk.md`). This de-risk proves the substrate is feasible before the months-scale
full feature (RF in every code path + complex-synapse bind + capability-matrix re-validation).

**Mechanism (ported verbatim from `resonate_fire_fhrr.py`):** an RF neuron holds complex state `Z = re + i·im`; each
step `Z *= exp(λ + iω)` (ω=2π/T, λ=−3e-4 damping); it spikes at the first **upward zero-crossing of `im`** (`im<0 →
im≥0`) with `|Z|>floor`; the spike step encodes the kick's phase as `(T − raw) mod T`. bind=`Z=phasor_a·phasor_b`,
unbind=`Z=phasor_c·conj(phasor_a)`, bundle=`Z=Σ phasor` — all just the initial kick, then resonate + read the phase.

## Architecture (minimal, contained, guarded)
- **`sim/enums.py`:** add `RESONATE_AND_FIRE = "RESONATE_AND_FIRE"` to `NeuronModel` (1 line).
- **`sim/bridge.py`:** ONE new `elif cfg.neuron_model_type == NeuronModel.RESONATE_AND_FIRE.name:` branch in
  `_run_one_simulation_step` (after the AdEx branch, ~line 5305), reusing `cp_membrane_potential_v` as `re` and
  `cp_recovery_variable_u` as `im`. The branch: rotate (re,im) by the fixed complex rotation, detect the upward
  `im` zero-crossing → `fired_this_step`, and record each neuron's FIRST spike step this resonate window into a
  lazily-allocated `cp_rf_spike_step` (with `cp_rf_prev_im`, `cp_rf_fired`, `cp_rf_step_counter`). The RF branch
  bypasses the Izhikevich v-threshold spike/reset path entirely (guarded by the model check) — **zero impact on the
  Izhikevich / HH / AdEx paths** (the existing branches are untouched).
  Plus two small methods: `rf_kick(self, kick_complex)` (set re,im = kick.real/imag; zero the RF trackers) and
  `rf_read_phases(self)` (return `(T − spike_step) mod T` per neuron, the recovered phases). Synaptic/OU drive is
  zero in the de-risk bridge (the kick is the input), so the existing pre-dynamics code runs harmlessly.
- **`research/findings/raw/_rf_on_bridge_probe.py`** (NOT protected): builds a `RESONATE_AND_FIRE` bridge, runs the
  three TDD gates below via `rf_kick`/step/`rf_read_phases`, reuses `resonate_fire_fhrr.py` phase helpers + the
  composer task structure by import.
- **`tests/test_rf_on_bridge.py`:** pins the gates.

**Constraint:** frozen bars / no-confab moat NEVER weakened. Protected edits limited to the RF substrate ONLY
(enum value + the guarded RF branch + the two RF methods). The diff is surfaced to the owner before/at landing.

## TDD task sequence (each: failing test → minimal impl → run → commit)

### Task 1 — RF phase readout on the bridge (foundational gate)
- **Test:** build an N-neuron `RESONATE_AND_FIRE` bridge; for phases φ ∈ {0,0.1,…,0.9}, `rf_kick(exp(2πiφ))`, step
  the bridge `T+8` times, `rf_read_phases()` → recovered ≈ φ (circular distance < tol, e.g. 0.02). Expected FAIL
  first (enum/branch absent).
- **Impl:** the enum value + the RF branch + `rf_kick`/`rf_read_phases`.
- **Gate:** mean circular phase error < 0.02 across all φ.

### Task 2 — bind / unbind / bundle via the bridge RF readout
- **Test:** random phasors a,b; bind kick = `a·b` → recovered phase ≈ phase(a)+phase(b); unbind `c·conj(a)` ≈
  phase(c)−phase(a); bundle `Σ` ≈ phase of the complex sum. (Kick math reused from `resonate_fire_fhrr.py`; the
  bridge does the resonate + readout.)
- **Gate:** circular error < 0.03 for bind/unbind; bundle matches `np.angle(Σ)/2π`.

### Task 3 — the composer task on the bridge RF (the DE-RISK GATE)
- **Test:** the `resonate_fire_fhrr.py` `ResonateFireFHRR` task (vocab 8×8, loads 2/3/5, bar 0.80) but with EVERY
  `rf_resonate` call routed through the bridge's RF step; compositional accuracy ≥ 0.80 at all loads AND abstention
  separates (groundable sim min > ungroundable max) — matching the numpy reference's perfect loads 2/3/5.
- **Gate (de-risk verdict):** GO if the bridge RF clears the bar at all loads with clean abstention → the RF
  substrate is realized on the bridge → proceed to the full FHRR-on-bridge feature (separate design/plan).
  NEGATIVE if a bridge-specific wall (numerical drift, backend mismatch, spike-timing resolution) breaks it → report.

## Honest scope / boundaries
- This proves RF **dynamics + phase readout + composition** on the bridge's neurons. It does NOT yet do the
  **complex-synapse bind** (the kick is computed from phasors and injected; the synapse carrying the operand phasor
  is the next feature layer) NOR the rate→timing recoding of the production composer. Those are the full-feature arc.
- GPU + CPU: the RF rotation is backend-agnostic (`xp` ops); validate on both where cheap.
- A bridge-RF NEGATIVE is a real finding (the bridge's neuron machinery can't host clean phase timing) → would
  reopen the Option-A-vs-D decision with the owner.
