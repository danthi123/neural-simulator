---
type: plan
status: live
date: 2026-06-04
---

# One-Bridge Unification — Step 1 (parser + composer merge) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Merge the parser (`BridgeParser`) and composer (`CoreSimComposer`) onto ONE `SimulationBridge` — their
neurons as disjoint index slices — so the two regions live on one bridge while remaining capability-equivalent to
the two separate bridges.

**Architecture:** Parameterize `BridgeParser` and `CoreSimComposer` so each can OPTIONALLY wire its synapses into a
caller-provided shared `SimulationBridge` at a given index offset, instead of building its own (default behavior
unchanged → existing tests pass untouched). A new `UnifiedBrainBridge` (in `research/runners/`) builds ONE bridge,
wires both regions into it at non-overlapping offsets, and exposes the same `parse` + `store/query/...` API the
`BrainConversationalAgent` already uses. The cross-region hand-off stays Python-orchestrated in step 1 (the gated
synaptic route is step 2; the dlPFC merge is step 3 — both out of scope here).

**Tech Stack:** Python, `sim.bridge.SimulationBridge` (Izhikevich neurons, `inject_explicit_wiring`), CuPy/NumPy
backend (`sim.backend`), pytest. All new code in `research/runners/`; `sim/` (protected) is not edited unless Task 1
proves it strictly necessary (flag if so).

**The standing gate (every task ends green):** the 10 on-brain tests
(`tests/test_core_sim_composition.py` 5 + `tests/test_brain_conversational_agent.py` 5) pass, and the capability
matrix (flat / one-attribute / two-attribute / negation) on the unified bridge does not regress versus the
separate-bridge baseline. A regression is a reportable finding (the measured cost of merging), committed honestly to
both git remotes (origin + gitea) — never hidden, never papered over.

**Plain-language note (owner standing requirement):** define each term once, no undefined acronyms. Key terms:
*bridge* = one `SimulationBridge` (a network of simulated neurons). *Region/slice* = a contiguous block of neuron
indices used for one function. *Plastic* = synapses whose weights change with learning; *fixed* = weights never
change. *Hebbian learning* = a co-activation weight-update rule (here, the parser's only learning). *Coincidence
wiring* = the composer's fixed bind/unbind circuit. *Capability matrix* = the per-category recall test
(flat fact / one-attribute / two-attribute / negation).

---

## Background the implementer must read first (zero-context onboarding)

Before Task 1, read these so the wiring is exact:
- `research/runners/brain_conversational_agent.py` → `class BridgeParser` (lines ~28–108): builds a bridge of
  `6 + 3*R` neurons (R=40 → 126): conjunction units 0–5, role ensembles 6–125. Wiring population `"parse"`:
  every conjunction → every role-ensemble neuron, initial weight 0.5, `plastic=True`, `conn_type="E_TO_E"`.
  Config: Izhikevich, `GENERIC_UNSTRUCTURED`, `dt_ms=1.0`, `enable_hebbian_learning=True`,
  `hebbian_max_weight=400`, `hebbian_learning_rate=0.005`, STDP/STP/structural/homeostasis/reward/Watts-Strogatz all
  off, `ou_std_current_pA=20`. `_train()` runs 30 epochs × 6 conjunctions × 120 steps.
- `research/runners/core_sim_composition.py` → `build_bind_bridge(seed, D)` (lines ~82–117): builds `8*D` neurons
  (role_ON/OFF, fill_ON/OFF sources, 4 AND banks A/B/C/D). Wiring population `"bind"`: source→AND-bank pairs,
  weight `W_COINC=320`, `plastic=False`, `conn_type="E_TO_E"`. Config: Izhikevich, `GENERIC_UNSTRUCTURED`,
  `dt_ms=1.0`, ALL plasticity off (including `enable_hebbian_learning=False`), `ou_std_current_pA=20`.
  `class CoreSimComposer.__init__` calls `build_bind_bridge` and stores `self.bridge, self.idx, self.D`.
- The **conflict** this plan resolves: the parser needs `enable_hebbian_learning=True`; the composer's bridge sets
  it `False`. One bridge has ONE global flag. Step 1 sets it `True` (for the parser) and relies on the composer's
  `"bind"` population being `plastic=False` to keep the composer's fixed weights from drifting. **Task 1 verifies
  exactly that** — it is the load-bearing assumption; everything else depends on it.
- `inject_explicit_wiring(plan)`: `plan` is a dict of named populations, each
  `{"pre_indices": [...], "post_indices": [...], "initial_weights": np.array, "plastic": bool, "conn_type": str,
  "count": int}`. Multiple populations on one bridge are allowed (the composer already uses one; the parser one).
- Backend: `from sim.backend import get_backend, to_host; xp, _ = get_backend()`. Read the composer's
  `hadamard_spiking` for how weights live on the bridge (the implementer must find the exact array name for
  synaptic weights — likely `bridge.cp_connections` data or a CSR; confirm before Task 1's assertion).

---

## Task 1: Per-population plasticity isolation (THE load-bearing de-risk)

**Why first:** if global Hebbian learning corrupts the composer's fixed coincidence weights despite
`plastic=False`, the whole merge approach changes (fallback: zero `cp_plasticity_rate_gain` on the composer slice).
Prove this cheaply before building anything else.

**Files:**
- Create test: `tests/test_unified_brain_bridge.py`
- (No implementation file yet — Task 1 may only need a tiny helper.)

**Step 1 — Write the failing test.** A merged bridge: parser `"parse"` population (plastic) at offset 0 + a small
composer-style fixed population (plastic=False) at an offset; global `enable_hebbian_learning=True`; train the
parser-style population; assert the fixed population's weights are byte-identical before/after.

```python
import numpy as np
import pytest
from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.backend import get_backend, to_host

def _weights_of(bridge, population_name):
    # IMPLEMENTER: return a host copy of the named population's synaptic weights.
    # Find the exact accessor by reading inject_explicit_wiring + how the composer reads/uses weights.
    raise NotImplementedError

def test_fixed_population_survives_global_hebbian():
    D = 64
    cfg = CoreSimConfig()
    cfg.num_neurons = 6 + 3*40 + 8*D          # parser slice + composer slice
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = 42; cfg.dt_ms = 1.0; cfg.connections_per_neuron = 0; cfg.num_traits = 1
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True        # ON for the parser
    cfg.hebbian_max_weight = 400.0; cfg.hebbian_learning_rate = 0.005
    for f in ("enable_short_term_plasticity","enable_structural_plasticity","enable_homeostasis",
              "enable_reward_modulation","enable_watts_strogatz"):
        setattr(cfg, f, False)
    cfg.ou_std_current_pA = 20.0
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    # plastic parser-style pop + FIXED composer-style pop (offset past the parser slice)
    off = 6 + 3*40
    plan = {
        "parse": {"pre_indices":[0], "post_indices":[6], "initial_weights":np.array([0.5],np.float32),
                  "plastic":True, "conn_type":"E_TO_E", "count":1},
        "bind":  {"pre_indices":[off], "post_indices":[off+1], "initial_weights":np.array([320.0],np.float32),
                  "plastic":False, "conn_type":"E_TO_E", "count":1},
    }
    bridge.inject_explicit_wiring(plan)
    before = _weights_of(bridge, "bind")
    # drive co-activation of the plastic pair for many steps (Hebbian would change a plastic synapse)
    xp,_ = get_backend()
    for _ in range(300):
        cur = xp.zeros(cfg.num_neurons, dtype=xp.float32); cur[0] = 2500.0; cur[6] = 2500.0
        bridge.cp_external_input_current[:] = cur
        bridge.runtime_state.current_time_ms += cfg.dt_ms   # advance clock (see CLAUDE.md gotcha)
        bridge._run_one_simulation_step()
    after = _weights_of(bridge, "bind")
    assert np.array_equal(before, after), "FIXED composer weights drifted under global Hebbian -> isolation failed"
```

**Step 2 — Run it; expect FAIL** (`_weights_of` raises `NotImplementedError`). `pytest tests/test_unified_brain_bridge.py::test_fixed_population_survives_global_hebbian -v`

**Step 3 — Implement `_weights_of`** by reading the real weight storage (the same array the composer reads). Keep it
a test helper for now.

**Step 4 — Run it.**
- **PASS** → isolation holds; the merge approach is sound; proceed to Task 2.
- **FAIL** → STOP and surface honestly: write finding
  `research/findings/2026-06-04-unified-bridge-plasticity-isolation.md` (per-population `plastic=False` does NOT
  isolate under global Hebbian), then implement the fallback as Task 1b: after wiring, zero
  `bridge.cp_plasticity_rate_gain` on the composer's synapses (per-synapse plasticity gate — see CLAUDE.md
  "plasticity gate" section) and re-run until the assertion passes. Do NOT proceed to Task 2 until green.

**Step 5 — Commit** (`git add` test + helper; message `de-risk: per-population plasticity isolation on a merged bridge`;
push origin + gitea).

---

## Task 2: `UnifiedBrainBridge` skeleton — build ONE bridge sized for both regions

**Files:** Create `research/runners/unified_brain_bridge.py`; Test `tests/test_unified_brain_bridge.py`.

**Step 1 — Failing test:** constructing `UnifiedBrainBridge(seed=42, proj_dim=64)` yields one bridge with
`num_neurons == (6 + 3*40) + 8*64`, exposes `parser_slice` (0..125) and `composer_offset` (126), and the two slices
do not overlap.

**Step 2 — Run; expect FAIL** (module missing).

**Step 3 — Implement** the skeleton: a class holding one `SimulationBridge` (config from the Background section,
`enable_hebbian_learning=True`), `num_neurons = 126 + 8*proj_dim`, with `parser_slice` and `composer_offset`
attributes. No wiring yet.

**Step 4 — Run; expect PASS. Step 5 — Commit + push both remotes.**

---

## Task 3: Parameterize `BridgeParser` to wire into a shared bridge at an offset

**Files:** Modify `research/runners/brain_conversational_agent.py` (`BridgeParser`); Test `tests/test_unified_brain_bridge.py`.

**Step 1 — Failing test:** `BridgeParser(seed=42, shared_bridge=<bridge>, index_offset=0)` does NOT build its own
bridge (uses the provided one), wires its `"parse"` population at the given offset, trains, and `role_of`/`parse`
return correct roles — identical to a standalone `BridgeParser(seed=42)` on the same seed (voice-invariant 6/6).
Also: `BridgeParser(seed=42)` with no `shared_bridge` is byte-identical in behavior to before (regression guard).

**Step 2 — Run; expect FAIL.**

**Step 3 — Implement:** add `shared_bridge=None, index_offset=0` params. When `shared_bridge` is provided: skip
building/initializing a bridge, use it; add `index_offset` to every conjunction/role index in the `"parse"` plan and
in the drive/readout index arrays; `inject_explicit_wiring` adds the population to the shared bridge. When `None`:
unchanged. Keep `_train`/`role_of`/`parse` working through the offset (the index arrays already abstract this).

**Step 4 — Run; expect PASS** (new shared-bridge path + the regression guard). **Step 5 — Commit + push.**

---

## Task 4: Parameterize `CoreSimComposer` to wire into a shared bridge at an offset

**Files:** Modify `research/runners/core_sim_composition.py` (`build_bind_bridge` + `CoreSimComposer.__init__`);
Test `tests/test_unified_brain_bridge.py`.

**Step 1 — Failing test:** `CoreSimComposer(seed=42, proj_dim=64, shared_bridge=<bridge>, index_offset=126)` wires
its `"bind"` population at the offset and `store`/`query_patient`/`query_agent`/`ask_yes_no` work on the shared
bridge (a small flat fact recovers). `CoreSimComposer(seed=42, proj_dim=64)` with no `shared_bridge` is unchanged
(regression guard — the 5 composer tests still pass).

**Step 2 — Run; expect FAIL.**

**Step 3 — Implement:** add `shared_bridge=None, index_offset=0`. Factor the index ranges in `build_bind_bridge` to
add `index_offset`; when `shared_bridge` is given, inject the `"bind"` population into it instead of building a new
bridge; store `self.bridge = shared_bridge`, `self.idx` shifted by offset, `self.D`. Every spiking op
(`_op`/`hadamard_spiking`) already addresses neurons via `self.idx`, so offsetting `self.idx` is sufficient. Default
path unchanged.

**Step 4 — Run; expect PASS. Step 5 — Commit + push.**

---

## Task 5: `UnifiedBrainBridge` wires both regions + exposes the agent API

**Files:** Modify `research/runners/unified_brain_bridge.py`; Test `tests/test_unified_brain_bridge.py`.

**Step 1 — Failing test:** a `UnifiedBrainBridge` builds the shared bridge, constructs `BridgeParser(shared_bridge,
offset=0)` + `CoreSimComposer(shared_bridge, offset=126)` into it, and the END-TO-END loop works on ONE bridge:
`unified.parse("dog go north")` → roles; `unified.store(...)`; `unified.query_patient("dog","go") == "north"`;
abstention holds (`query_patient("river","look") is None`). The composer's fixed weights are unchanged after the
parser trained (re-assert Task 1's isolation at full scale).

**Step 2 — Run; expect FAIL.**

**Step 3 — Implement:** in `UnifiedBrainBridge.__init__`, build the shared bridge (Task 2), then
`self.parser = BridgeParser(seed, shared_bridge=self.bridge, index_offset=0)` and
`self.composer = CoreSimComposer(seed, proj_dim, shared_bridge=self.bridge, index_offset=self.composer_offset)`.
Expose `parse`, `store`, `query_patient`, `query_agent`, `ask_yes_no`, `describe`, `render_fact` by delegation, plus
the `kb` and `words`/`concepts` the agent reads. (Dialogue planning `elaborate` stays on its own dlPFC bridge for
now — out of scope until step 3.)

**Step 4 — Run; expect PASS. Step 5 — Commit + push.**

---

## Task 6: The capability gate — no regression on the unified bridge

**Files:** Create `research/findings/raw/_unified_bridge_capability_probe.py`; Test `tests/test_unified_brain_bridge.py`.

**Step 1 — Failing test:** a multi-seed (42/43/44) capability check — build a `UnifiedBrainBridge` at production
`proj_dim=800`, run the capability matrix (flat / one-attribute / two-attribute / negation, 6 trials each) and
assert each category matches the separate-bridge `CoreSimComposer` baseline (allow ±1 trial tolerance for spiking
noise; a real drop is a regression to report).

**Step 2 — Run; expect FAIL** (probe missing). **Step 3 — Implement** the probe (reuse
`_decorrelate_v16_probe.run_matrix` against a `UnifiedBrainBridge`; GPU). **Step 4 — Run:**
- PASS → step 1 of B is DONE; the parser + composer are one bridge, capability-equivalent. Write finding
  `research/findings/2026-06-04-one-bridge-unification-step1-DONE.md`.
- Any regression → write the finding honestly with the measured per-category numbers (the cost of the shared step
  loop / OU noise), and decide with the controller whether it is within tolerance or needs a fix (e.g. per-region OU,
  per-region reset). Do NOT weaken the test to pass.

**Step 5 — Commit + push both remotes.**

---

## Final review (after Task 6)

Dispatch a code-quality review over the whole step-1 diff (the parameterization of `BridgeParser` /
`CoreSimComposer` + `UnifiedBrainBridge`), confirm `sim/` protected modules are untouched (or the one strictly-needed
edit is flagged + justified), confirm the 10 on-brain tests + the new unified tests are all green, and confirm
every outcome (incl. any regression) is on both remotes. Then surface step-1 completion to the owner and STOP for the
step-2 (gated synaptic route) plan — do not auto-start step 2.
