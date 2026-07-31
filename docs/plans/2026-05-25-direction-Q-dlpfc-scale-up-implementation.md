---
type: plan
status: live
date: 2026-05-25
---

# Direction Q Implementation Plan — dlpfc_wm scale-up Wang 2002 NMDA persistence test

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Build a standalone test bridge with dlpfc_wm at n=1000 neurons, dense recurrent NMDA-rich connectivity, and run Wang 2002 delayed-response protocol to measure NMDA-driven persistent activity. Multi-seed [42,43,44]. Pre-registered verdict; bar frozen.

**Architecture:** Standalone test bridge using existing `BrainRegion` + `RegionPathway` framework. Single dlpfc_wm region at n=1000 + stimulus input region. Reuses validated NMDA kernel (`fused_nmda_update_and_current`) byte-unchanged. No modification of `build_biological_brain_regions` or any other validated builder. Net-new = the standalone bridge constructor + stimulus protocol + verdict module.

**Tech Stack:** CuPy GPU (real runs), Izhikevich neuron model (consistent with Direction I baseline; HH variant reserved for Approach C if Approach B PASSes), NMDA voltage-dependent Mg block (Wang 2002 calibration), pre-registered fixed-threshold verdict module.

---

### Task 0: Grounding pin (intentionally RED until Task 4 lands)

**Files:**
- Create: `tests/test_direction_Q_grounding.py`

**Step 1: Write the failing tests**

```python
# tests/test_direction_Q_grounding.py
"""Direction Q grounding pin - intentionally RED until later tasks land.

These tests pin the contracts the Direction Q standalone test bridge
runner MUST satisfy. They are RED on commit; turn GREEN as Tasks 1-4
land. Final test (Task 5) keeps the contract.
"""
import importlib.util
import os
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_direction_Q_runner_module_exists():
    """Task 4: the standalone test bridge runner module imports."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_Q_dlpfc_scale_up_standalone.py",
    )
    assert os.path.exists(path), (
        "Task 4 not yet landed: " + path + " does not exist"
    )


def test_direction_Q_verdict_module_exists():
    """Task 3: the verdict module imports."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_Q_verdict.py",
    )
    assert os.path.exists(path), (
        "Task 3 not yet landed: " + path + " does not exist"
    )


def test_direction_Q_verdict_thresholds_frozen():
    """Task 3: pre-registered thresholds are present, not modifiable
    by results."""
    spec = importlib.util.spec_from_file_location(
        "direction_Q_verdict",
        os.path.join(
            REPO_ROOT,
            "research/findings/raw/direction_Q_verdict.py",
        ),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Frozen thresholds (must be present)
    assert hasattr(mod, "_Q_RATE_RATIO_MIN")
    assert hasattr(mod, "_Q_DELAY_MIN_SEC")
    assert hasattr(mod, "_Q_MIN_SEEDS_PASS")
    # Pre-registered values (must equal design doc)
    assert mod._Q_RATE_RATIO_MIN == 2.0
    assert mod._Q_DELAY_MIN_SEC == 3.0
    assert mod._Q_MIN_SEEDS_PASS == 3


def test_direction_Q_runner_has_nmda_off_control():
    """Task 5: runner imports the NMDA-off control runner."""
    path = os.path.join(
        REPO_ROOT,
        "research/findings/raw/direction_Q_dlpfc_scale_up_standalone.py",
    )
    if not os.path.exists(path):
        pytest.skip("Task 4 not landed yet")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "ENABLE_NMDA_OFF_CONTROL" in src or "nmda_off" in src.lower(), (
        "Task 5 control runner not yet integrated"
    )
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_direction_Q_grounding.py -v`
Expected: 4 FAILED (none of the target files exist yet)

**Step 3: Commit**

```bash
git add tests/test_direction_Q_grounding.py
git commit -m "Direction Q Task 0: grounding pin (intentionally RED until Tasks 3-5 land)"
```

---

### Task 1: Standalone test bridge constructor

**Files:**
- Create: `research/findings/raw/direction_Q_bridge_builder.py`
- Test: existing `tests/test_direction_Q_grounding.py` (verified GREEN partially)

**Goal**: pure constructor function that builds a SimulationBridge with ONLY dlpfc_wm at n=1000 + stimulus input region; NMDA-enabled; dense recurrent.

**Step 1: Write the failing test first (TDD)**

Add to `tests/test_direction_Q_bridge_builder.py`:

```python
# tests/test_direction_Q_bridge_builder.py
"""Tests for direction_Q_bridge_builder - the standalone Q test bridge."""
import os
import sys
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def test_build_q_test_bridge_returns_bridge():
    from research.findings.raw.direction_Q_bridge_builder import (
        build_q_test_bridge,
    )
    bridge = build_q_test_bridge(
        seed=42, n_dlpfc=1000, dlpfc_density=0.10,
        enable_nmda=True, verbose=False,
    )
    assert bridge is not None
    # dlpfc_wm region exists
    rm = bridge.region_manager
    idx = rm.indices("dlpfc_wm")
    assert idx.size == 1000


def test_build_q_test_bridge_has_stim_region():
    from research.findings.raw.direction_Q_bridge_builder import (
        build_q_test_bridge,
    )
    bridge = build_q_test_bridge(
        seed=42, n_dlpfc=1000, dlpfc_density=0.10,
        enable_nmda=True, verbose=False,
    )
    rm = bridge.region_manager
    stim_idx = rm.indices("q_stim_input")
    assert stim_idx.size >= 100  # at least 100 stim neurons


def test_build_q_test_bridge_nmda_off_control():
    from research.findings.raw.direction_Q_bridge_builder import (
        build_q_test_bridge,
    )
    bridge = build_q_test_bridge(
        seed=42, n_dlpfc=1000, dlpfc_density=0.10,
        enable_nmda=False, verbose=False,
    )
    # Should construct successfully with NMDA off
    assert bridge is not None
```

**Step 2: Run to verify FAIL**

Run: `pytest tests/test_direction_Q_bridge_builder.py -v`
Expected: 3 FAILED (`ModuleNotFoundError: direction_Q_bridge_builder`)

**Step 3: Implement minimal `build_q_test_bridge`**

```python
# research/findings/raw/direction_Q_bridge_builder.py
"""Direction Q standalone test bridge builder.

Builds a fresh SimulationBridge with ONLY dlpfc_wm at the
target scale + a small stimulus input region. Isolates the
Wang 2002 NMDA persistence mechanism from any other substrate
component.

Reuses validated infrastructure byte-unchanged:
- BrainRegion + RegionPathway framework
- NMDA kernel (fused_nmda_update_and_current)
- IZH2007_HIPPO_PYRAMIDAL preset (Direction I baseline; PFC-style)
"""
from __future__ import annotations
import os
import sys


def build_q_test_bridge(seed: int, n_dlpfc: int = 1000,
                          dlpfc_density: float = 0.10,
                          n_stim: int = 200,
                          enable_nmda: bool = True,
                          verbose: bool = False):
    """Construct a standalone Direction Q test bridge.

    Args:
        seed: RNG seed
        n_dlpfc: dlpfc_wm region size (Direction I used 60; Q uses 1000+)
        dlpfc_density: internal recurrent density (Wang 2002 ~0.20;
                       0.10 is conservative starting point)
        n_stim: stimulus input region size
        enable_nmda: NMDA on/off (False = AMPA-only control)
        verbose: print build info
    """
    from sim.config import (CoreSimConfig, VisualizationConfig,
                              RuntimeState, GPUConfig)
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronType

    regions = [
        BrainRegion(
            name="dlpfc_wm",
            n_neurons=n_dlpfc,
            exc_fraction=0.8,
            internal_density=dlpfc_density,
            exc_weight_mean=2.0,
            inh_weight_mean=4.0,
            weight_jitter=0.2,
            plastic_internal=False,  # frozen for test (no learning)
            izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
            enable_nmda=enable_nmda,
        ),
        BrainRegion(
            name="q_stim_input",
            n_neurons=n_stim,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0,
            inh_weight_mean=0.0,
            weight_jitter=0.0,
            plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ),
    ]

    pathways = [
        RegionPathway(
            from_region="q_stim_input",
            to_region="dlpfc_wm",
            density=0.10,
            weight_mean=3.0,
            weight_jitter=0.3,
            plastic=False,
        ),
    ]

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = seed
    cfg.enable_nmda = enable_nmda
    cfg.nmda_tau_decay = 100.0  # Wang 2002 calibration
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.fast_spike_reset = True

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)

    if verbose:
        print("[BUILD-Q] dlpfc_wm n=" + str(n_dlpfc)
              + " density=" + str(dlpfc_density)
              + " NMDA=" + str(enable_nmda)
              + " stim_input n=" + str(n_stim), flush=True)
    return bridge
```

**Step 4: Run tests to verify PASS**

Run: `pytest tests/test_direction_Q_bridge_builder.py -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add research/findings/raw/direction_Q_bridge_builder.py tests/test_direction_Q_bridge_builder.py
git commit -m "Direction Q Task 1: standalone test bridge builder (3/3 tests pass)"
```

---

### Task 2: Stimulus + delay protocol functions

**Files:**
- Create: `research/findings/raw/direction_Q_protocol.py`
- Test: `tests/test_direction_Q_protocol.py`

**Goal**: pure functions that implement Wang 2002 delayed-response protocol on a Q test bridge: baseline → cue → delay → measure.

**Step 1: Write failing tests**

```python
# tests/test_direction_Q_protocol.py
import os, sys
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def test_run_baseline_returns_rate():
    from research.findings.raw.direction_Q_bridge_builder import (
        build_q_test_bridge,
    )
    from research.findings.raw.direction_Q_protocol import (
        run_baseline_period,
    )
    bridge = build_q_test_bridge(
        seed=42, n_dlpfc=200, dlpfc_density=0.10,
        enable_nmda=True, verbose=False,
    )
    rate = run_baseline_period(bridge, duration_ms=200.0)
    assert isinstance(rate, float)
    assert rate >= 0.0


def test_apply_cue_stimulates_dlpfc():
    from research.findings.raw.direction_Q_bridge_builder import (
        build_q_test_bridge,
    )
    from research.findings.raw.direction_Q_protocol import (
        apply_cue_stimulus,
    )
    bridge = build_q_test_bridge(
        seed=42, n_dlpfc=200, dlpfc_density=0.10,
        enable_nmda=True, verbose=False,
    )
    rate = apply_cue_stimulus(
        bridge, cue_amplitude_pA=1500.0, duration_ms=200.0,
        cue_fraction=0.5,
    )
    assert rate > 0.0  # stim should drive at least some firing


def test_measure_delay_period_returns_rate_trajectory():
    from research.findings.raw.direction_Q_bridge_builder import (
        build_q_test_bridge,
    )
    from research.findings.raw.direction_Q_protocol import (
        measure_delay_period,
    )
    bridge = build_q_test_bridge(
        seed=42, n_dlpfc=200, dlpfc_density=0.10,
        enable_nmda=True, verbose=False,
    )
    rates = measure_delay_period(
        bridge, duration_ms=500.0, bin_ms=50.0,
    )
    assert len(rates) == 10  # 500 / 50 = 10 bins
    assert all(r >= 0.0 for r in rates)
```

**Step 2: Run to verify FAIL**

Run: `pytest tests/test_direction_Q_protocol.py -v`
Expected: 3 FAILED

**Step 3: Implement minimal `direction_Q_protocol.py`**

The implementation should:
- `run_baseline_period(bridge, duration_ms)`: step bridge with no stim; capture dlpfc_wm spike counts; return mean population rate
- `apply_cue_stimulus(bridge, cue_amplitude_pA, duration_ms, cue_fraction)`: inject current into cue_fraction of dlpfc_wm exc neurons; step bridge; return cue-period rate
- `measure_delay_period(bridge, duration_ms, bin_ms)`: step bridge with no stim; capture dlpfc_wm spike counts per bin; return list of bin rates

**Step 4: Run to verify PASS**

```bash
pytest tests/test_direction_Q_protocol.py -v
```

**Step 5: Commit**

---

### Task 3: Verdict module (pre-registered, frozen)

**Files:**
- Create: `research/findings/raw/direction_Q_verdict.py`
- Test: extend `tests/test_direction_Q_grounding.py` (Task 0 test should now PASS for verdict portions)

**Goal**: pure verdict function that takes per-seed delay-rate trajectories + baseline rates and returns a frozen-threshold verdict tag.

**Implementation** (`research/findings/raw/direction_Q_verdict.py`):
- `_Q_RATE_RATIO_MIN = 2.0` (delay rate / baseline rate)
- `_Q_DELAY_MIN_SEC = 3.0`
- `_Q_MIN_SEEDS_PASS = 3`
- `compute_verdict(per_seed_data, control_per_seed_data)`: returns tag in {Q_BISTABILITY_PASS, Q_BISTABILITY_PARTIAL, Q_BISTABILITY_NEGATIVE, Q_VOID_CONTROL_ALSO_PASSED}

The verdict is computed ONLY from recorded per-seed JSON (no re-running). If the AMPA-only control also passes the bistability bar, verdict is VOID (the persistence is not NMDA-driven).

**Tests**: Verify thresholds are frozen at design-doc values; verify per-seed data conforms to expected shape; verify control-also-PASSes correctly yields VOID.

---

### Task 4: Multi-seed runner

**Files:**
- Create: `research/findings/raw/direction_Q_dlpfc_scale_up_standalone.py`

**Goal**: orchestrate the full Q test across 3 seeds + NMDA-off control across 3 seeds; write results to JSON; compute verdict.

Structure:
1. For each seed in [42, 43, 44]:
   a. Build bridge with NMDA on
   b. Run baseline period
   c. Apply cue stimulus
   d. Measure delay period
   e. Record per-seed result
2. Repeat for control (NMDA off)
3. Compute verdict via direction_Q_verdict.compute_verdict
4. Write `direction_Q_dlpfc_scale_up_standalone.json` with verdict + per-seed data

---

### Task 5: NMDA-off control runner integration

Already covered in Task 4 (control loop). Verify Task 0's
`test_direction_Q_runner_has_nmda_off_control` PASSes after Task 4.

---

### Task 6: CONTROLLER-ONLY decisive run

NOT a subagent task. Controller (Claude in main session, or watchdog
spawn) launches the decisive multi-seed run:

```bash
python -u -m research.findings.raw.direction_Q_dlpfc_scale_up_standalone \
    > research/findings/raw/direction_Q_dlpfc_scale_up_standalone.log 2>&1 &
```

Background; monitor via Bash run_in_background with until-loop for
"verdict:" or "FATAL". Expected wall ~1-2 hr per seed at n=1000 Izh
= ~3-6 hr total. Smell-test the result (recompute verdict from
recorded JSON; confirm control didn't also pass; per-seed values
reproduce aggregate). Honest propagation both remotes.

---

### Post-Task chain (per verdict)

- **Q_BISTABILITY_PASS**: write findings doc; dispatch adversarial
  reviewer subagent; if CLEAR record pillar n=105; update
  AUTONOMOUS_STATE + capability_status.json; consider Approach C
  (Wang 2002 published-parameter HH replication) as next.
- **Q_BISTABILITY_PARTIAL**: characterize scaling envelope at
  n=200/500/2000; identify the threshold.
- **Q_BISTABILITY_NEGATIVE**: deeper structural diagnosis required;
  localize what's missing beyond scale; pivot to Direction 3 or 4
  per the mechanism-class audit guide.
- **Q_VOID_CONTROL_ALSO_PASSED**: the persistence is not NMDA-driven;
  diagnose what's actually driving it; potentially a substrate
  configuration bug; do not propagate as a pillar.

---

## Discipline (binding throughout)

- Bar UNCHANGED: `_Q_RATE_RATIO_MIN=2.0`, `_Q_DELAY_MIN_SEC=3.0`,
  `_Q_MIN_SEEDS_PASS=3`. Set ONCE in direction_Q_verdict.py at Task 3;
  never tuned by results.
- No protected/frozen/moat modification. The validated bridge
  builders (build_biological_brain_regions, _build_bridge_with_hippo)
  remain byte-unchanged. Direction Q uses its OWN constructor
  (`build_q_test_bridge`).
- No autograd.
- GPU/CuPy for real runs; numpy only for cheap probes.
- Honest propagation EVERY outcome both remotes.
- Pre-launch grep confirmed: no prior Direction Q n=1000 work; this
  is genuinely net-new (Direction I tested n=60 across 4 probes).
- Reviewer-style scrutiny applied at the time of result, not deferred.
