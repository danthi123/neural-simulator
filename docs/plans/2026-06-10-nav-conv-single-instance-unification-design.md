---
type: plan
status: live
date: 2026-06-10
---

# Navigation + Conversational single-instance unification — design (roadmap step 2)

> **Status:** DESIGN — read-only research deliverable. Awaiting controller review + cheapest-first de-risk
> before any build. Per the project's standing "deep research + design FIRST at a new direction" practice.
> **Date:** 2026-06-10.
> **Scope:** consolidate the navigation brain and the conversational brain into ONE `SimulationBridge`
> instance, capability-equivalent to the two separate brains. (Step 3 — replacing the composer's exact-inverse
> VSA algebra with a learned cortex — is a SEPARATE later arc and is explicitly OUT OF SCOPE here.)

---

## 0. Terms (defined once; no undefined acronyms after this)

- **bridge** — one `sim.bridge.SimulationBridge`: a network of simulated neurons that all share ONE set of GPU
  state arrays, ONE per-step update loop (`_run_one_simulation_step`, `sim/bridge.py:5410`), ONE timestep
  `dt`, and ONE global config object (`CoreSimConfig`).
- **region / slice** — a contiguous block of neuron indices that performs one function. The project supports
  two ways to lay regions onto a bridge: (a) the **brain-region framework** (`BrainRegion` + `RegionPathway`
  + `RegionManager`, `sim/regions.py`), where you declare regions and the manager assigns index slices and
  builds the wiring; and (b) **`inject_explicit_wiring(plan)`** (`sim/bridge.py`), where you hand the bridge an
  explicit synapse list addressed by raw neuron indices. These two are mutually exclusive on a given bridge:
  the framework path sets `region_manager`; the explicit path leaves it `None`.
- **Izhikevich** — the standard spiking point-neuron model (`NeuronModel.IZHIKEVICH`) the navigation brain and
  the conversational parser use. State = membrane voltage `v` + recovery `u`.
- **resonate-and-fire (RF)** — `NeuronModel.RESONATE_AND_FIRE` (`sim/enums.py:15`): a complex-valued "phasor"
  neuron whose state is `Z = re + i·im`. **It REUSES the same two arrays Izhikevich uses** — `re` is stored in
  `cp_membrane_potential_v`, `im` in `cp_recovery_variable_u` (`sim/bridge.py:5380-5381`, comment 6058). Each
  step it rotates `Z` by `exp(λ + iω)` and fires on the upward zero-crossing of `im`; the spike step encodes a
  PHASE. Used by the conversational composer because phase codes have no "common mode" and so escape the
  rate-coded composer's opponency SNR wall (CLAUDE.md "OPPONENCY ESCAPED").
- **FHRR phasor algebra** — the vector-binding scheme the RF composer realizes: each concept/role is a vector
  of phases; bind = complex multiply through a diagonal complex synapse; unbind = multiply by the conjugate;
  bundle = sum of unit phasors. Realized on the bridge via complex synaptic weights (`rf_set_complex_weights`,
  `sim/bridge.py:5351`) held in `cp_rf_w_re` / `cp_rf_w_im` (SEPARATE CSR matrices from the real-valued
  `cp_connections`).
- **plastic / frozen** — a plastic synapse's weight changes with learning; a frozen one's never does.
- **plasticity gate** (`cp_plasticity_rate_gain`, `sim/bridge.py:356`) — a per-synapse multiplier in [0,1] on
  weight UPDATES (gates STDP, eligibility, Hebbian, synaptic scaling). 0.0 = frozen weights. Does NOT gate
  synaptic CURRENT.
- **transmission gate** (`cp_transmission_gain`, complement of the above) — a per-synapse multiplier in [0,1]
  on synaptic CURRENT at runtime (`set_transmission_gate`, `cp_transmission_gain`). Pre-wire a route, hold it
  closed (no current), open it on command.
- **per-region NMDA mask** (`cp_nmda_neuron_mask`, `sim/bridge.py:269`) — a per-neuron 0/1 array; the slow
  NMDA current is multiplied by it (`sim/bridge.py:5701-5702`), so NMDA reaches only masked neurons. The
  brain-region framework auto-builds it from any `BrainRegion(enable_nmda=True)` (`sim/bridge.py:1180-1189`).
- **the two brains:**
  - **navigation brain** — built by `research/runners/g11_bg_runner.py`: `build_bg_brain_regions()` returns the
    region/pathway lists; `run_moving_goal_episode()` (config at `g11_bg_runner.py:4068`) constructs the
    bridge. A basal-ganglia action-selection cascade + visual cortex + a fully-spiking superior colliculus +
    a dopamine / SNc reward loop. Learns by reward-modulated STDP. Izhikevich, dt=1.0, brain-region framework.
  - **conversational brain** — the production agent is `research/runners/brain_conversational_agent.py`
    (`BrainConversationalAgent`). It is a learned syntactic **parser** + a **composer** + a **dlPFC** dialogue
    planner. NOTE the important nuance in §2 below about how many bridges it actually uses today.

---

## 1. The single most important finding up front (read this first)

There are TWO distinct "one-bridge" conversational artifacts in the codebase, and the prompt conflates them:

1. **`UnifiedBrainBridge`** (`research/runners/unified_brain_bridge.py`) — merges the parser + the **rate-coded**
   composer (`CoreSimComposer`, the ±1 Hadamard coincidence circuit, all-Izhikevich) + the dlPFC onto ONE
   bridge via `inject_explicit_wiring`. This is the "ONE-BRIDGE UNIFICATION COMPLETE" arc. It is **all
   Izhikevich** — no RF neurons anywhere. The dlPFC NMDA latch is isolated with the per-neuron NMDA mask set by
   hand (`unified_brain_bridge.py:664-667`).

2. **`BrainConversationalAgent` with `composer_kind="rf"`** (the PRODUCTION DEFAULT since the 2026-06-05
   opponency switch, `brain_conversational_agent.py:151,172-180`) — uses the **RF phasor composer**
   (`RFPhasorComposer`). This is the agent the production tests (`tests/test_brain_conversational_agent.py`)
   build. **It does NOT use `UnifiedBrainBridge` and it is NOT one bridge.** It is several bridges:
   - the parser: its own ~126-neuron Izhikevich bridge (`brain_conversational_agent.py:167` →
     `BridgeParser.__init__`, `brain_conversational_agent.py:80-83`);
   - the composer: a CACHE of EPHEMERAL per-op RF bridges keyed by neuron count
     (`rf_phasor_composer.py:98,104-107`), each built Izhikevich-then-flipped-to-RF
     (`rf_phasor_composer.py:39-58`), driven by the dedicated RF fast loop `rf_resonate_steps`
     (`rf_phasor_composer.py:110`, `sim/bridge.py:5399`) — NOT the global step loop;
   - the spiking cleanup (opt-in): a SEPARATE Izhikevich "bank" bridge `_izh_bank`
     (`rf_phasor_composer.py:169-195`);
   - the dlPFC (`elaborate`): builds its OWN bridge on demand (`brain_conversational_agent.py:254`).

**Consequence for this design:** "consolidate nav + conv into one brain" is really two sub-questions, because
the RF composer's binding op is, in production, NOT run as a region of a persistent shared bridge at all — it
runs on throwaway per-op bridges using a dedicated RF loop that bypasses `_run_one_simulation_step`. The merge
therefore splits cleanly into:

- **(A) the parts that ARE persistent-bridge regions** — the navigation cascade + the parser + the dlPFC loop
  (all Izhikevich, dt=1.0) — which CAN live as disjoint slices on one persistent bridge. This is the merge the
  brain-region framework / the `UnifiedBrainBridge` precedent already make mechanical.
- **(B) the RF composer's binding op** — which today does not need to be a region of the persistent bridge,
  and (because of the crux in §2) should NOT be forced to time-share the persistent bridge's `v`/`u` arrays
  with the running navigation Izhikevich dynamics.

The honest, lowest-risk unification keeps (B) on its own RF substrate (the per-op or one persistent RF bridge it
already uses) and merges (A) into one persistent navigation+conversational bridge. Whether (B) must also become
a slice of the SAME bridge to satisfy the owner's "ONE brain" bar is the key scope question for the controller
(§7).

---

## 2. The crux, trust-but-verified: can RESONATE_AND_FIRE and IZHIKEVICH coexist on one bridge?

**Verdict: NO — not within a single `_run_one_simulation_step` call, as the code stands. A true co-resident
RF-slice + Izhikevich-slice bridge needs a protected `sim/` edit. BUT the production RF composer never asks for
that, so the merge does not require it.** Evidence:

1. **The neuron-model dynamics dispatch is a single GLOBAL `if/elif` on `cfg.neuron_model_type`, one branch per
   step for the ENTIRE bridge.** `sim/bridge.py:5870` `if cfg.neuron_model_type == IZHIKEVICH … elif … HH …
   elif … ADEX … elif RESONATE_AND_FIRE (6056)`. There is no per-neuron or per-region model selector anywhere
   in the bridge — every `neuron_model_type` reference is the one global config string (verified across all 40+
   call sites; none is per-slice). So on any given step, EITHER the Izhikevich branch runs for all neurons OR
   the RF branch runs for all neurons — never both.

2. **RF and Izhikevich alias the SAME state arrays, so running one corrupts the other's state.** The Izhikevich
   branch computes `v_new, u_new = fused_izhikevich2007_dynamics_update(cp_membrane_potential_v,
   cp_recovery_variable_u, …)` over the FULL arrays and writes the full arrays back (`sim/bridge.py:5871-5876,
   and the spike-reset writes at 5903-5905 / 6052-6053`). The RF branch stores its phasor `re`/`im` in those
   very same arrays (`sim/bridge.py:5380-5381`). So if an RF slice's phase state lived in
   `cp_membrane_potential_v`/`cp_recovery_variable_u` on a bridge that also steps Izhikevich, the Izhikevich
   update would overwrite the RF slice's phase every step. The two models cannot time-share `v`/`u`.

3. **What WOULD be needed for true co-residence (a byte-review-gated `sim/` edit, not a blocker):** a per-neuron
   model mask + (critically) SEPARATE state arrays for the RF slice (its own `re`/`im`, not aliased onto
   `v`/`u`), and a step that runs both `fused_izhikevich2007_dynamics_update` on the Izhikevich slice and
   `_rf_advance_one` on the RF slice, merged by the mask. This is feasible and additive (the RF branch already
   factors its per-step math into `_rf_advance_one`, `sim/bridge.py:5370`), but it is a genuine protected-module
   change and should be gated on the owner's byte-level diff review per the standing directive.

4. **Why the production path does NOT need any of that.** The RF composer's binding op runs via
   `rf_resonate_steps` (`sim/bridge.py:5399-5408`), a dedicated loop that calls `_rf_advance_one` directly and
   **"skips the full `_run_one_simulation_step` machinery"** (its docstring). It runs on its OWN bridge(s) whose
   `cfg.neuron_model_type` is RESONATE_AND_FIRE for their whole (short) lifetime. The navigation bridge's
   `_run_one_simulation_step` is never invoked on the RF state. And the RF complex synapses (`cp_rf_w_re` /
   `cp_rf_w_im`, `sim/bridge.py:5367-5368`) are array-disjoint from the navigation real-valued synapses
   (`cp_connections`): `_rf_advance_one` touches only the former (`sim/bridge.py:5384-5387`), the Izhikevich /
   navigation path touches only the latter. So the two computations are already isolated by living on different
   bridges with different model strings.

**Bottom line on the crux:** the prompt's prior ("does the per-region dispatch keep RF and Izhikevich isolated
on shared v/u?") resolves to: *there is no per-region dispatch; a shared-v/u RF+Izhikevich step is impossible
as-is and would corrupt state; but the production composer deliberately runs RF off the global step loop on its
own RF bridge, so the merge does not depend on co-residence.* The de-risk in §5 nails this down concretely.

---

## 3. The merge architecture

### 3.1 What is mechanical (the brain-region framework already does multi-region)

The navigation brain is ALREADY a ~30-region, ~30-pathway multi-region bridge built with the brain-region
framework (`g11_bg_runner.py:3836` `regions, pathways = build_bg_brain_regions(...)`; config
`g11_bg_runner.py:4068-4078`: `enable_brain_region_framework=True`, `cfg.brain_regions = regions`,
`cfg.region_pathways = pathways`). Adding more disjoint regions to that list is the framework's core job:
`RegionManager.initialize` assigns each region a contiguous index slice and `total_neurons()` sets
`num_neurons` automatically (`sim/regions.py`). So the *mechanical* merge of any Izhikevich, dt=1.0 region-set
into the navigation bridge is: append its `BrainRegion`s to `regions` and its `RegionPathway`s to `pathways`.

Both brains are already mutually compatible on the three settings that matter for coexistence:
- **timestep:** nav dt=1.0 (`g11_bg_runner.py:4070`); parser dt=1.0 (`brain_conversational_agent.py:70`);
  composer (rate and RF init) dt=1.0 (`rf_phasor_composer.py:46`); the `UnifiedBrainBridge` ran the dlPFC at
  dt=1.0 and the step-3 de-risk proved the dlPFC NMDA working-memory latch SURVIVES dt=1.0
  (`unified_brain_bridge.py:89-96`, finding `2026-06-04-step3-dlpfc-dt-survives.md`). ✔ no dt reconciliation
  needed for the Izhikevich regions.
- **neuron model:** nav + parser + dlPFC are all Izhikevich. ✔
- **NMDA isolation:** the framework auto-builds the per-neuron NMDA mask for `enable_nmda=True` regions
  (`sim/bridge.py:1180-1189`). The nav already relies on this (the critic / `sel_X` accumulator / PFC slices
  carry `enable_nmda=True`, `g11_bg_runner.py:842,1229,4141,4162`). A merged dlPFC region simply sets
  `enable_nmda=True` and inherits the same mask machinery — **cleaner than the `UnifiedBrainBridge`, which had
  to set the mask by hand because its `inject_explicit_wiring` bridge has no `region_manager`**
  (`unified_brain_bridge.py:99-104,664-667`).

### 3.2 Recommended structure

Build ONE persistent `SimulationBridge` via the brain-region framework whose region list is:

```
regions  =  nav_regions              # build_bg_brain_regions(...) output, ~30 regions
          ++ parser_regions           # 6 conjunction + 3 role ensembles  (Izhikevich, dt=1.0)
          ++ dlpfc_regions            # cortex_ctx + dlpfc_wm loop          (Izhikevich, NMDA-on slice)
pathways =  nav_pathways ++ parser_internal ++ dlpfc_internal ++ dlpfc_graph_slots
```

`RegionManager` lays each on its own slice; `num_neurons` is the sum. The parser's learned (position×voice)→role
wiring and the dlPFC's loop+graph edges are expressed as `RegionPathway`s (or, where the framework cannot
express the exact topology, via a post-init `inject_explicit_wiring`-equivalent on the framework slices — but
note the framework and `inject_explicit_wiring` are mutually exclusive at the *cfg* level, so the parser/dlPFC
wiring on a framework bridge must go through the framework's plan, or through the bridge's
`set_pathway_weights` / a CSR-safe pre-allocation as `UnifiedBrainBridge._wire_dlpfc` does,
`unified_brain_bridge.py:595-667`).

The **RF composer's binding op** (the part that can't share `v`/`u`) stays on its own RF bridge(s) — exactly as
production does today — and is invoked by the agent when it needs to bind/unbind/recall, reading concept codes
that are the substrate's own. This keeps the merge entirely within the all-Izhikevich, dt=1.0, framework world
and side-steps the crux. (If the owner wants the RF op ALSO co-resident as a slice of the one bridge, that is
the §2.3 protected `sim/` edit + the §5b extended de-risk.)

### 3.3 The two cross-region seams

- **parser → composer.** Today a Python `{role: word}` hand-off (`brain_conversational_agent.py:195-196`). The
  `UnifiedBrainBridge` step-2 already demonstrated a SYNAPTIC version on a shared Izhikevich bridge: per-role
  `role_src` pools drive the composer's role bank through a parser-coupled transmission gate
  (`unified_brain_bridge.py:396-446`). For the RF composer this seam stays Python in the first merge (the RF
  op is on a separate bridge); converting it to synaptic is a step-2/step-3-conversational concern, not a
  step-2-unification (nav+conv) requirement.
- **agent facts → dlPFC graph.** Python-built association graph from stored facts
  (`brain_conversational_agent.py:222-240`); the dlPFC spreads over it. On a shared bridge the dlPFC loop is a
  region and the graph edges are pre-allocated CSR slots overwritten in place (the `UnifiedBrainBridge`
  pattern, `unified_brain_bridge.py:108-120,637-662`) so installing graph edges never triggers a CSR rebuild
  that would invalidate other gate→synapse index maps.

### 3.4 What is genuinely hard (ranked in §4)

The mechanical part is the region/pathway append. The hard parts are: (i) the RF/Izhikevich coexistence IF the
owner insists the RF op be co-resident (§4.1); (ii) keeping the navigation's global reward-STDP + dopamine
neuromodulator from drifting the frozen conversational populations (§4.3); (iii) the two stepping disciplines
coexisting (§4.5).

---

## 4. Ranked integration risks (each: evidence + mitigation)

### 4.1 (CRUX, highest) RESONATE_AND_FIRE vs IZHIKEVICH coexistence

- **Evidence:** §2. Global single-branch dispatch (`sim/bridge.py:5870-6056`); RF aliases `v`/`u`
  (`sim/bridge.py:5380-5381`). They cannot time-share state in one step loop.
- **Severity:** would be fatal IF the design required the RF binding op to run as a slice of the persistent
  navigation bridge stepped by `_run_one_simulation_step`.
- **Mitigation (recommended, zero `sim/` edit):** do NOT co-resident the RF op. Keep it on its own RF bridge(s)
  (production already does), invoked via `rf_resonate_steps` which bypasses the global step loop. The RF
  complex synapses are array-disjoint from `cp_connections`, so there is no interference. The merge is then
  purely the all-Izhikevich regions.
- **Mitigation (IF owner requires co-residence):** a byte-review-gated `sim/` edit — add a per-neuron model
  mask + separate RF state arrays (un-alias `re`/`im` from `v`/`u`) + a masked dual-dynamics step. Additive
  and guarded (Izhikevich/HH/AdEx paths byte-unchanged when no RF slice). De-risk per §5b BEFORE building.

### 4.2 dt reconciliation

- **Evidence:** all merge candidates are dt=1.0 (§3.1). The dlPFC NMDA latch survives dt=1.0 (step-3 finding).
- **Severity:** LOW. No per-region dt is needed.
- **Mitigation:** none required; reuse the validated dt=1.0 dlPFC operating point (self-attractor weight ≈30,
  the genuinely NMDA-dependent regime, `unified_brain_bridge.py:117-118`).

### 4.3 Plasticity isolation (nav reward-STDP + dopamine must NOT drift the frozen conversational populations)

- **The concern:** nav runs `enable_stdp=True`, `enable_reward_modulation=True`, and the neuromodulator
  subsystem with a `dopamine` modulator targeting `plasticity_rate` at `scope="all"`
  (`g11_bg_runner.py:4079-4080,4234-4238`). The parser bridge runs `enable_hebbian_learning=True`. On a SHARED
  bridge these are GLOBAL flags — so the nav's reward-STDP and the parser's Hebbian would both be "on" for the
  whole bridge, and the dopamine `scope="all"` plasticity-rate multiplier is global
  (`compute_plasticity_rate_multiplier()`, applied at `sim/bridge.py:6416-6417`).
- **Trust-but-verified mitigation — the per-synapse plasticity gate covers BOTH learning paths:**
  - STDP weight updates are multiplied by `cp_plasticity_rate_gain` per synapse (`sim/bridge.py:6268-6271`):
    gain=0 ⇒ no STDP change AND no eligibility accumulation.
  - The reward-modulation eligibility→weight conversion is ALSO multiplied by `cp_plasticity_rate_gain` per
    synapse (`sim/bridge.py:6456-6457`): gain=0 ⇒ reward-driven updates are zeroed for those synapses, EVEN
    THOUGH `compute_plasticity_rate_multiplier()` is a global scalar applied before it. So the dopamine
    `scope="all"` target cannot drift a gated-0 synapse — the per-synapse gain wins.
  - Hebbian potentiation AND the Hebbian weight-DECAY term are both gated by `cp_plasticity_rate_gain`
    (`sim/bridge.py:6156-6157` and `6170-6171`). This is the exact isolation the `UnifiedBrainBridge` relied on
    (`unified_brain_bridge.py:19-26`): a `plastic=False` population still drifts under global Hebbian decay
    unless its synapses are ALSO plasticity-gated to 0.0.
- **Therefore:** every fixed conversational population merged onto the nav bridge (the composer's bind
  population if co-resident; the dlPFC loop+graph edges; the parser's role-route fixed weights) must carry a
  `plasticity_gate` held at 0.0 — exactly the `UnifiedBrainBridge` recipe (`unified_brain_bridge.py:436,662`).
  The parser's plastic "parse" population is the one population that SHOULD learn; it must be trained with the
  nav's reward-STDP either OFF for its slice or harmless. **Two residual subtleties to verify in the de-risk:**
  1. **The parser learns by Hebbian; nav learns by reward-STDP.** On a shared bridge both `enable_stdp` and
     `enable_hebbian_learning` would be global. The parser slice must get Hebbian (it is trained at construction
     with `enable_hebbian_learning=True`) and the nav slice must get reward-STDP. Because STDP/reward/Hebbian
     only touch synapses whose pre- AND post-neuron fire, and the parser and nav slices fire only when THEIR
     inputs are driven, cross-contamination is limited to co-firing within a slice. But the GLOBAL flags being
     on for both means the parser slice would ALSO see reward-STDP if it fires while reward is non-zero, and the
     nav slice would see Hebbian if its neurons co-fire. **De-risk must confirm** that (a) freezing all
     non-trained conversational synapses with the plasticity gate, and (b) sequencing parser training as a
     dedicated phase (drive parser only, nav idle, reward=0) reproduces the parser's standalone learning, and
     that (c) running nav afterward with the parser slice's "parse" synapses plasticity-gated to 0.0 (post
     training) leaves the parser intact while nav learns.
  2. **The dopamine `plasticity_rate` `scope="all"` multiplier** scales `reward_learning_rate` globally. During
     nav, this only matters for reward-modulation updates, which are per-synapse gated — so frozen
     conversational synapses are safe. Confirm the multiplier does not also touch the Hebbian rate (it does not:
     Hebbian uses `cfg.hebbian_learning_rate`, the reward multiplier only scales `effective_reward_lr`,
     `sim/bridge.py:6413-6418`).

### 4.4 Scale / memory (fits the 24 GB RTX 3090?)

- **Navigation flagship:** prints `len(regions)` regions, `cfg.num_neurons` neurons, `cp_connections.nnz`
  synapses at `g11_bg_runner.py:4789-4790`. Per the BG-cascade docstring the per-action cascade is ~14.5K
  synapses at the base config; with visual cortex + SC + hippocampus + critic the flagship is on the order of
  a few thousand neurons and ≲ a few×10⁵–10⁶ synapses (the controller should read the printed line from a
  flagship run to pin exact numbers — it is environment/flag dependent). CSR sparse storage scales with nnz,
  not N².
- **Conversational regions added:** parser ≈126 neurons; dlPFC ≈ 2×max(600, 60·V) neurons (V≈16 at the probe
  vocab → ~1200; V≈320 production → ~38K) plus the dlPFC pre-allocated V² c2d graph slots, which is the real
  memory item: V=16 → 256 pairs × (50×50) ≈ 6.4×10⁵ slot synapses; V=320 → 10⁵ pairs × 2500 ≈ 2.5×10⁸ — that
  is the documented step-4 scale concern (`unified_brain_bridge.py:615-617`), to be sparsified (only realizable
  association pairs) at production vocab. The RF composer's per-op bridges are tiny (2·D ≈ 256 neurons at D=128)
  and ephemeral.
- **Severity:** LOW at probe scale (V=16): combined is single-digit thousands of neurons + ≲10⁶ synapses —
  trivial for 24 GB. MODERATE only if the dlPFC graph is pre-allocated dense at production V=320 (a known,
  separately-scoped sparsification, not a step-2 blocker).
- **Mitigation:** do the first merge at the validated probe vocab (V=16) where every conversational capability
  is already validated; defer dense-graph sparsification to its own step. Confirm the exact flagship nnz from
  the printed line and budget headroom.

### 4.5 The two step loops coexisting (nav sensorimotor windows vs conversational encode/bind/read)

- **Evidence:** the nav runs a per-step sensorimotor loop — set sensory current, step, read motor firing over
  STIMULUS/READOUT windows (`g11_bg_runner.py:4780-4782`, `run_moving_goal_episode`). The conversational ops
  set `cp_external_input_current` on a slice, run N steps, read firing, then RESET by zeroing the input and
  free-running (e.g. the composer's per-op `RESET_STEPS`, and `UnifiedBrainBridge._op_synaptic`'s explicit
  reset loop `unified_brain_bridge.py:535-538`).
- **The risk:** these are DIFFERENT stepping disciplines on ONE step loop. If nav and conversation are stepped
  in the same wall-clock interval, one's drive/reset perturbs the other's slice. Concretely: a conversational
  op that zeros `cp_external_input_current[:]` (the WHOLE array, e.g. `unified_brain_bridge.py:535,589`) would
  wipe the nav's sensory drive; and nav's continuous stepping would advance the conversational slice's state
  between an encode and a read.
- **Mitigation (TIME-MULTIPLEX, the `UnifiedBrainBridge` model):** the agent ORCHESTRATES which capability is
  active. The merge does NOT run nav and conversation concurrently in the same step; it runs nav for a
  navigation episode, then (when the agent converses) runs the conversational ops. Two concrete requirements:
  1. **Per-slice drive/reset, not whole-array.** Replace any `cp_external_input_current[:] = 0` with zeroing
     only the active capability's slice, so an inactive capability's drive is preserved. (The conversational
     ops currently zero the whole array because on their standalone bridges that IS their slice; on the merged
     bridge they must zero only their indices. Small, runner-side change.)
  2. **Quiescence of the idle capability.** When nav steps, the conversational slices receive no input and
     (being plasticity-gated where fixed, and Izhikevich at rest) stay quiescent — but verify they do not
     accumulate spurious state that corrupts the next conversational read. The nav already runs OU noise OFF
     (`g11_bg_runner.py:4094`) and the conversational composer runs OU OFF (`rf_phasor_composer.py:53`); the
     dlPFC explicitly toggles OU off for its read (`unified_brain_bridge.py:727-732`). So at-rest drift is
     small, but the de-risk should confirm a nav episode does not disturb a subsequent conversational answer
     and vice versa.
- **Severity:** MODERATE. It is the most likely source of a subtle behavioral regression and the thing the
  no-regression gate (§6) must catch.

### 4.6 (lower) CSR-rebuild invalidating gate→synapse index maps

- **Evidence:** `inject_explicit_wiring` rebuilds `cp_connections` wholesale and resets gate maps
  (`unified_brain_bridge.py:28-34`); a `set_pathway_weights(add_missing=True)` that adds NEW edges resorts the
  CSR and invalidates the plasticity-gate→synapse index map (`unified_brain_bridge.py:108-116`).
- **Severity:** LOW–MODERATE; well-understood, with a known pattern.
- **Mitigation:** pre-allocate all dynamic edges (e.g. dlPFC graph slots) at construction (weight 0, gated 0.0)
  and only overwrite `.data` in place at runtime — the `UnifiedBrainBridge._wire_dlpfc` pattern
  (`unified_brain_bridge.py:108-120`). On a framework bridge, build the nav + conversational wiring through the
  framework plan once; do not add new synapses after init.

---

## 5. The cheapest-first de-risk (smallest experiment that proves-or-kills the crux BEFORE the big merge)

There are two distinct de-risks depending on the §7 scope decision. Do **5a first** — it is the one the
recommended (zero-`sim/`-edit) merge actually depends on. Do **5b** only if the owner requires RF co-residence.

### 5a (REQUIRED) — Plasticity-isolation + step-coexistence on a tiny all-Izhikevich shared bridge

This is the real crux of the RECOMMENDED merge: does the navigation's global reward-STDP + dopamine
neuromodulator leave a frozen, plasticity-gated conversational slice byte-stable, AND do the two stepping
disciplines not corrupt each other?

- **Setup (read-only-designed; the controller builds it):** one Izhikevich, dt=1.0, brain-region-framework
  bridge with TWO trivial region groups:
  - a "nav-like" group: 2–3 small regions wired like a miniature BG cascade with `enable_stdp=True`,
    `enable_reward_modulation=True`, and the same `dopamine` neuromodulator (`scope="all"` plasticity_rate,
    `g11_bg_runner.py:4227-4250`) the flagship uses;
  - a "conv-like" group: a small fixed population (e.g. 50 neurons, a few hundred synapses) tagged with a
    `plasticity_gate` held at 0.0, plus a tiny plastic "parser-like" population trained by Hebbian first.
- **Procedure:**
  1. Snapshot the conv-like fixed population's weights (`cp_connections.data` at its synapse indices).
  2. Run the nav-like group for a navigation-length burst (e.g. 1000+ steps) with reward injected so STDP +
     dopamine actively update the nav synapses, zeroing ONLY the nav slice's input each "trial" (per §4.5.1).
  3. Re-read the conv-like fixed population's weights.
- **PASS:** the conv-like fixed weights are BYTE-IDENTICAL before and after the nav burst (the plasticity gate
  + reward-gate path fully isolate them); the nav synapses DID change (proving the learning was live, not a
  no-op); and a conversational read on the conv-like slice AFTER the nav burst returns the same answer as
  before it (step-coexistence: nav stepping did not corrupt the conv slice's read).
- **FAIL (and what it means):** any drift in the conv-like fixed weights ⇒ the global dopamine/STDP path is
  reaching them despite the gate — re-examine whether `compute_plasticity_rate_multiplier()` or a path other
  than the two gated sites (`sim/bridge.py:6268-6271,6456-6457`) is at play; OR the conv read changes after the
  nav burst ⇒ step-coexistence needs per-slice reset / quiescence work (§4.5) before the big merge.
- **Cost:** one tiny bridge, minutes of GPU. This proves the merge's load-bearing isolation claim cheaply.

### 5b (CONDITIONAL — only if RF co-residence is required) — RF-slice + Izhikevich-slice state non-corruption

This is the de-risk for the prompt's literal crux. It will FAIL as-is (proving the §2 verdict) and thereby
scope the protected `sim/` edit precisely.

- **Setup:** one bridge, `num_neurons` = a small Izhikevich slice + a small RF slice. Drive the Izhikevich
  slice with a suprathreshold current; `rf_kick` the RF slice with a known phasor.
- **Procedure (as-is, no edit):** set `cfg.neuron_model_type = IZHIKEVICH`, step `_run_one_simulation_step`
  once; read the RF slice's `cp_membrane_potential_v`/`cp_recovery_variable_u`.
- **EXPECTED (kill) RESULT:** the RF slice's phasor state is overwritten by the Izhikevich update (its `re`/`im`
  no longer match the kick) — i.e. the Izhikevich step corrupted the RF slice. This is the documented §2
  outcome and CONFIRMS that co-residence needs the protected edit.
- **PASS criterion for the EDITED version (the thing to build only if the owner wants co-residence):** after
  the `sim/` edit (per-neuron model mask + separate RF arrays + masked dual-dynamics step), one step advances
  the Izhikevich slice by Izhikevich dynamics AND the RF slice by RF dynamics, and the RF slice's read-back
  phase (`rf_read_phases` over the period) matches a control RF-only bridge byte/behaviorally, while the
  Izhikevich slice matches a control Izhikevich-only bridge. Plus: a `RESONATE_AND_FIRE`-only bridge and an
  `IZHIKEVICH`-only bridge are byte-unchanged vs today (the guard).
- **Cost:** the as-is run is minutes (it just demonstrates the corruption). The edited version is a real
  protected-module change, byte-review-gated, and is the bigger investment — which is exactly why 5a (the
  zero-edit path) is the recommended first move and 5b is conditional.

---

## 6. Anti-cheat / no-regression acceptance gates

The unified bridge is accepted ONLY if BOTH hold:

- **(a) Navigation not regressed.** The merged bridge, run on the flagship navigation benchmark with the same
  flags + seeds, scores within noise of the standalone navigation flagship (the documented cheat-5 multi-goal
  metric; e.g. the G v2.5 + K v2 32×32 result, and/or the current N9 nav A/B score). Use the SAME 6-seed
  protocol the project requires for any generalization claim. A regression beyond the seed-to-seed noise floor
  is a FAIL (or a measured, reported cost — not hidden).
- **(b) Conversational capability matrix preserved + no-confab moat intact.** The production conversational
  tests pass VERBATIM on the unified path: `tests/test_brain_conversational_agent.py` (comprehend/store/QA,
  voice-invariant comprehension, negation/yes-no, embedded clause, dialogue-planning `elaborate` + abstention,
  generation `describe` + abstention, and the cache-invalidation guard) AND `tests/test_core_sim_composition.py`
  (who/what + abstention, negation, one-attribute, clause, recovery ≥ 0.80) for the rate composer if it is in
  the merged path. The abstention/no-confab assertions (`what_does("river","look") is None`,
  `describe("river") is None`, `elaborate("river") is None`) are the load-bearing moat — they MUST pass
  unchanged.
- **Anti-cheat sanity:** the merge must not make either gate pass for a trivial reason — e.g. the nav score
  must come from the merged bridge actually stepping the nav regions (not a fallback to a standalone nav
  bridge), and the conversational tests must build the unified artifact (not silently fall back to separate
  bridges). State both explicitly in the acceptance harness.

---

## 7. Honest risks / what could make this a NEGATIVE — and the open scope question for the controller

- **The scope question that decides everything:** does "ONE brain" require the RF composer's binding op to be a
  slice of the SAME persistent bridge, or is it acceptable for the RF binding op to run on its own RF substrate
  (as production does) while the navigation cascade + parser + dlPFC are one persistent bridge?
  - If the **looser** bar is acceptable (recommended): the merge is mostly mechanical (region/pathway append +
    plasticity-gate freezing + per-slice reset), the crux is side-stepped (no `sim/` edit), and the de-risk is
    the cheap 5a. This is the lowest-risk path to a capability-equivalent single navigation+conversational
    instance.
  - If the **strict** bar is required (RF co-resident): it needs the §2.3 / §5b protected `sim/` edit
    (per-neuron model mask + un-aliased RF state arrays + masked dual-dynamics step). That is feasible and
    additive but is a byte-review-gated change with real risk of perturbing the Izhikevich path if done
    carelessly. Flag it as a gated edit, not a blocker.
- **Could-be-NEGATIVE outcomes (each a real finding, per the project's "honest negative IS the deliverable"
  standard):**
  1. **Step-coexistence corruption (§4.5) proves un-isolable cheaply.** If a navigation episode measurably
     perturbs a subsequent conversational answer even after per-slice reset, the honest finding is that the two
     functions need a stronger separation (e.g. a "mode" that quiesces one capability's slice) — a
     biology-translatable point about time-sharing a substrate between sensorimotor control and
     language/working-memory.
  2. **Plasticity isolation leaks (§4.3) despite the gate.** If 5a shows any drift in the frozen conversational
     weights under the nav's global dopamine/STDP, that is a concrete bug to fix (or a boundary of the
     per-synapse gate) — and it is exactly the load-bearing claim to verify before the big build.
  3. **The dlPFC dense graph (§4.4) does not scale to production vocab** without sparsification — already a
     documented step-4 concern, out of step-2 scope, but a real ceiling for "one brain at production V=320".
  4. **The RF co-residence edit (§2.3), if attempted, perturbs the Izhikevich navigation path** — the byte-diff
     review + the §5b guard ("RESONATE_AND_FIRE-only and IZHIKEVICH-only bridges byte-unchanged") is the
     control that catches this; if it can't be made byte-safe, keep the RF op on its own bridge (the looser
     bar) and document why true co-residence is not warranted.

---

## 8. Reusable machinery (exact file:line the merge builds on)

- **Brain-region framework (multi-region on one bridge, auto slices, auto NMDA mask):**
  `sim/regions.py` (`BrainRegion`, `RegionPathway`, `RegionManager`); the nav builder
  `research/runners/g11_bg_runner.py:306` (`build_bg_brain_regions`), its bridge config
  `g11_bg_runner.py:4068-4078`; auto NMDA mask build `sim/bridge.py:1180-1189`; NMDA mask applied
  `sim/bridge.py:5701-5702`.
- **Conversational one-bridge precedent (the recipe to copy):**
  `research/runners/unified_brain_bridge.py` — wiring accumulation `:151-187`
  (`merge_population_into_shared_bridge`); plasticity-gate isolation rationale `:19-34`; per-slice synaptic
  hand-off `:396-446`; CSR-safe pre-allocated dlPFC edges + by-hand NMDA mask `:595-667`; OU-off-for-dlPFC-read
  `:727-732`.
- **Plasticity gate (freezes weight updates incl. reward path):** `set_plasticity_gate` `sim/bridge.py:2906`;
  STDP gating `sim/bridge.py:6268-6271`; reward-modulation gating `sim/bridge.py:6456-6457`; Hebbian
  potentiation + decay gating `sim/bridge.py:6156-6157,6170-6171`.
- **Transmission gate (gates current; the parser→composer route + thalamocortical gating):**
  `set_transmission_gate` + `_apply_gate_couplings` (`sim/bridge.py`, gate couplings invoked at
  `:6080`); the route operating point `unified_brain_bridge.py:58-61`.
- **Per-region NMDA (one global flag, NMDA confined to a slice):** nav usage
  `g11_bg_runner.py:842,1229,4141,4162`; mask build `sim/bridge.py:1180-1189`.
- **RF substrate (separate-bridge phasor composer + the dedicated RF loop that bypasses the global step):**
  `NeuronModel.RESONATE_AND_FIRE` `sim/enums.py:15`; `rf_kick` `sim/bridge.py:5321`; `rf_set_complex_weights`
  (separate `cp_rf_w_re`/`cp_rf_w_im` CSRs) `sim/bridge.py:5351-5368`; `_rf_advance_one` (RF-only state +
  RF-only synapses) `sim/bridge.py:5370-5397`; `rf_resonate_steps` (bypasses `_run_one_simulation_step`)
  `sim/bridge.py:5399-5408`; RF branch in the global dispatch `sim/bridge.py:6056-6070`; the production RF
  composer `research/runners/rf_phasor_composer.py:39-58,98-111`.
- **The capability surface that must not regress:** `research/runners/brain_conversational_agent.py`
  (`BrainConversationalAgent`, composer-default `:151,172-180`); `research/runners/rf_phasor_composer.py`
  (`RFPhasorComposer`); tests `tests/test_brain_conversational_agent.py`, `tests/test_core_sim_composition.py`.
- **The conversational design history (what was already solved):**
  `docs/plans/2026-06-04-one-bridge-unification-design.md` (the §2 dt+NMDA constraint analysis);
  `-step1/2/3-implementation.md`;
  `docs/plans/2026-06-04-consolidate-conversational-pipeline-onto-core-sim-design.md`.

---

## 9. Recommended sequence (so every step is independently valuable and pausable)

1. **De-risk 5a** (plasticity isolation + step coexistence on a tiny all-Izhikevich shared bridge). Gate: PASS
   per §5a. This proves the load-bearing isolation claim for ~minutes of work.
2. **Merge the all-Izhikevich regions** (nav cascade + parser + dlPFC) onto ONE framework bridge at probe vocab
   V=16. Freeze every fixed conversational population with a plasticity gate (0.0). Keep the RF composer op on
   its own RF bridge (looser bar). Gate: BOTH acceptance gates in §6 (nav flagship 6-seed within noise +
   `tests/test_brain_conversational_agent.py` verbatim).
3. **(Decision point)** Present to the owner the §7 scope question. If the strict bar (RF co-resident) is
   required, run de-risk 5b and, only on a green edited-version PASS, do the byte-review-gated `sim/` edit.
4. **(Later, separate arcs)** parser→composer synaptic seam for the RF composer; dlPFC dense-graph
   sparsification for production V=320. Both are out of step-2 scope.
