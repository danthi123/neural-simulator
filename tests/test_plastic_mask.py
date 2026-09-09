"""Test that SimulationBridge.cp_synapse_plastic_mask gates STDP writes.

Strategy: build two small networks that are identical except one has all
synapses plastic, the other has all fixed. Drive both with identical Poisson
stimulus so STDP fires in both. Compare weight deltas. If the mask works,
the plastic net moves weights while the fixed net keeps them stable.

2026-09-08 (Vikunja #203 follow-up): the drive-and-measure logic used to live
inline here and build its two `SimulationBridge`s IN-PROCESS. That is fragile
in a way that produced a real, reproducible false failure: `sim/bridge.py`
resolves numpy-vs-cupy ONCE at module level, on its first-ever import in the
process (`cp, _backend_name = get_backend()`, sim/bridge.py:44) -- so whichever
test file happens to import `sim.bridge` first in a pytest session (often
`test_enforce_plastic_mask.py`, which is collected alphabetically before this
file and forces `SIM_BACKEND=numpy` for its own unrelated purpose) permanently
pins the WHOLE process to that backend, no matter what this file's own
`import cupy as cp` or any later env-var change asks for. Under the numpy
backend this exact tiny seeded network produces ZERO spikes over 500 steps
(verified), so the old in-process test silently degraded into "did the bridge
fire at all" and failed with a "plastic weights didn't move" message that read
like an STDP plastic-mask regression but was actually a cross-backend
firing-activity difference (same class as the 2026-06-09 N9 CuPy-vs-numpy
divergence finding) -- FAILURE_LOG.md's row citing this test as evidence the
"STDP-update path is NOT covered [by the mask]" was a mischaracterization; the
STDP kernel already applies `cp_synapse_plastic_mask` unconditionally
(sim/bridge.py:10241, matching BDSP/BTSP), confirmed by direct code read.

The fix: run the dynamics in `tests/_plastic_mask_stdp_scenario.py`, in its OWN
subprocess with `SIM_BACKEND=cupy` set in that subprocess's environment before
`sim.bridge` is ever imported there -- immune to whatever any sibling test did
earlier in the parent process.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run_scenario(seed, all_plastic):
    env = dict(os.environ)
    env["SIM_BACKEND"] = "cupy"
    args = [sys.executable, "-m", "tests._plastic_mask_stdp_scenario", "--seed=%d" % seed]
    if all_plastic:
        args.append("--all-plastic")
    out = subprocess.check_output(args, cwd=_REPO_ROOT, env=env, text=True)
    line = [ln for ln in out.strip().splitlines() if ln.strip().startswith("{")][-1]
    return json.loads(line)


def test_plastic_mask_freezes_fixed_synapses():
    pytest.importorskip("cupy")

    fixed = _run_scenario(seed=7, all_plastic=False)
    plastic = _run_scenario(seed=7, all_plastic=True)

    assert fixed["w0_allclose_0p5"] and plastic["w0_allclose_0p5"]

    # Mask presence
    assert fixed["mask_present"], "expected a plastic mask on the all-fixed net"
    assert not plastic["mask_present"], "expected no plastic mask on the all-plastic net"

    # Both nets must actually have fired -- otherwise this test proves nothing
    # about STDP (the exact failure mode this file's docstring documents).
    assert fixed["fired_any"], "fixed-net scenario had zero spikes; test setup is wrong"
    assert plastic["fired_any"], "plastic-net scenario had zero spikes; test setup is wrong"

    # Fixed: every weight exactly unchanged.
    assert fixed["w1_allclose_w0"], (
        f"Fixed weights changed: max diff = {fixed['max_change']:.6f}"
    )

    # Plastic: at least some weight moved meaningfully.
    assert plastic["max_change"] > 0.01, (
        f"Plastic weights didn't move (max |dW|={plastic['max_change']:.6f}); "
        f"test setup is wrong"
    )


def test_no_mask_means_all_plastic():
    """When inject_explicit_wiring is called with no plastic=False populations,
    cp_synapse_plastic_mask should stay None (back-compat with existing paths)."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
    from sim.config import CoreSimConfig
    from sim.enums import NeuronModel

    cfg = CoreSimConfig()
    cfg.num_neurons = 3
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = 5
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.enable_stdp = True
    cfg.enable_watts_strogatz = False

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)

    wiring_plan = {
        "pop_a": {"pre_indices": [0], "post_indices": [1],
                  "initial_weights": np.array([0.3], dtype=np.float32),
                  "plastic": True, "count": 1},
        "pop_b": {"pre_indices": [0], "post_indices": [2],
                  "initial_weights": np.array([0.3], dtype=np.float32),
                  "plastic": True, "count": 1},
    }
    bridge.inject_explicit_wiring(wiring_plan)
    assert bridge.cp_synapse_plastic_mask is None


def test_mask_aligned_with_csr_order():
    """When populations have a mix of plastic and non-plastic synapses, the
    mask must align with cp_connections.data's internal CSR order."""
    pytest.importorskip("cupy")
    import cupy as cp

    from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
    from sim.config import CoreSimConfig
    from sim.enums import NeuronModel

    cfg = CoreSimConfig()
    cfg.num_neurons = 6
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = 11
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.enable_stdp = True
    cfg.enable_watts_strogatz = False

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)

    # Plastic: (0,3), (0,4). Fixed: (1,3), (1,4). Verify CSR-aligned mask
    # labels each (pre, post) correctly.
    plan = {
        "plastic": {
            "pre_indices": [0, 0],
            "post_indices": [3, 4],
            "initial_weights": np.array([0.2, 0.3], dtype=np.float32),
            "plastic": True,
            "count": 2,
        },
        "fixed": {
            "pre_indices": [1, 1],
            "post_indices": [3, 4],
            "initial_weights": np.array([0.4, 0.5], dtype=np.float32),
            "plastic": False,
            "count": 2,
        },
    }
    bridge.inject_explicit_wiring(plan)

    coo = bridge.cp_connections.tocoo(copy=False)
    pre_h = cp.asnumpy(coo.row)
    post_h = cp.asnumpy(coo.col)
    mask_h = cp.asnumpy(bridge.cp_synapse_plastic_mask)

    # For every synapse: if pre=0 then plastic, if pre=1 then fixed.
    for pre, post, is_plastic in zip(pre_h, post_h, mask_h):
        if pre == 0:
            assert is_plastic, f"Synapse (0,{post}) should be plastic"
        elif pre == 1:
            assert not is_plastic, f"Synapse (1,{post}) should be fixed"
