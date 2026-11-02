"""GUARD-COMPLETENESS TEST for the general-step megakernel dispatch (found 2026-10-31, #203 follow-up).

THE FAILURE CLASS. `enable_step_megakernel_v2` is DEFAULT-ON. On the GPU backend the Python step is replaced
wholesale by the fused step whenever `_step_megakernel_can_dispatch()` returns True. The guard's READ-ONLY list
checked hebbian / STP / homeostasis / STDP / structural / reward / inhibitory-STDP -- but NOT synaptic scaling,
BTSP, BDSP or the shunt-norm pool. A bridge armed with synaptic scaling and nothing else therefore dispatched
the fused "read-only" step and silently lost the mechanism. Instrumented proof (`sys.settrace` over the step):
numpy executes 500 line-hits in the homeostasis block (12320-12440), cupy executes ZERO and the maximum bridge
line traced is 9615 -- the dispatch `return`. The scaling rule moved weights 15.0 on numpy, 0.0 on cupy.

WHAT THIS TEST DOES. It cannot run CUDA on CI, so it drives the guard's attribute checks directly: patch
`sim.bridge.is_gpu_backend` to report GPU and call `_step_megakernel_can_dispatch()` on a real bridge. Every
mechanism that writes weights (or a synaptic-current denominator) inside the Python step must make the guard
False; a genuinely read-only bridge must still dispatch, otherwise the perf path is dead.

Run: PYTHONPATH=. .venv/bin/python -m pytest tests/test_step_megakernel_guard_completeness.py -q
"""
from __future__ import annotations

import os
import re
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")   # the guard is exercised on CPU by faking is_gpu_backend

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.config import CoreSimConfig  # noqa: E402
from sim.enums import NeuronModel  # noqa: E402
import sim.bridge as bridge_mod  # noqa: E402


def _bridge():
    cfg = CoreSimConfig()
    cfg.num_neurons = 6
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = 42
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 2
    cfg.num_traits = 1
    cfg.inhibitory_trait_indices = []
    cfg.ou_std_current_pA = 0.0
    for flag in ("enable_hebbian_learning", "enable_stdp", "enable_short_term_plasticity",
                 "enable_structural_plasticity", "enable_homeostasis", "enable_synaptic_scaling",
                 "enable_reward_modulation", "enable_btsp", "enable_bdsp", "enable_shunt_norm_pool"):
        setattr(cfg, flag, False)
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def _can_dispatch(b):
    """Evaluate the GPU dispatch guard on CPU by faking the backend predicate."""
    original = bridge_mod.is_gpu_backend
    bridge_mod.is_gpu_backend = lambda: True
    try:
        return b._step_megakernel_can_dispatch()
    finally:
        bridge_mod.is_gpu_backend = original


def test_a_genuinely_read_only_gpu_bridge_still_dispatches():
    """The perf path must not be killed: an inert bridge still dispatches."""
    b = _bridge()
    assert _can_dispatch(b), "a read-only bridge must still take the fused step, or the megakernel is dead"


@pytest.mark.parametrize("mechanism", ["enable_synaptic_scaling", "enable_btsp", "enable_bdsp",
                                       "enable_hebbian_learning", "enable_stdp",
                                       "enable_structural_plasticity", "enable_reward_modulation",
                                       "enable_homeostasis"])
def test_every_weight_writing_mechanism_refuses_the_fused_step(mechanism):
    b = _bridge()
    assert _can_dispatch(b), "baseline must dispatch before arming the mechanism"
    setattr(b.core_config, mechanism, True)
    assert not _can_dispatch(b), (
        f"{mechanism} writes weights (or a synaptic-current denominator) inside the Python step; the fused step "
        f"does not implement it, so the guard must refuse the dispatch"
    )


def test_shunt_norm_pool_refuses_when_its_masks_are_live():
    """The guard tests masks, not flags, for the divisive terms: no masks means no contribution."""
    b = _bridge()
    assert _can_dispatch(b)
    b.core_config.enable_shunt_norm_pool = True
    assert _can_dispatch(b), "no shunt masks allocated -> nothing to skip, dispatch is still faithful"
    b.cp_shunt_norm_source_mask = np.ones(6, dtype=np.float32)
    b.cp_shunt_norm_read_mask = np.ones(6, dtype=np.float32)
    assert not _can_dispatch(b), "live shunt masks contribute to synaptic current; the fused path ignores them"


def test_the_guard_set_covers_every_flag_that_gates_a_weight_write_in_the_python_step():
    """Static cross-check, derived from the source: every `enable_*` flag whose guarded block writes
    `cp_connections.data` inside the step body must appear in the dispatch guard. A flag that writes weights and
    is absent from the guard is a mechanism the fused GPU step silently drops -- the exact shape that made the
    scaling rule inert on cupy while it moved 15.0 on numpy."""
    src = open(os.path.abspath(bridge_mod.__file__), encoding="utf-8").read()
    lines = src.splitlines()
    step_start = next(i for i, l in enumerate(lines) if l.strip() == "def _run_one_simulation_step(self):")
    step = lines[step_start:step_start + 3000]
    guard_start = next(i for i, l in enumerate(lines) if l.strip() == "def _step_megakernel_can_dispatch(self):")
    guard = "\n".join(lines[guard_start:guard_start + 140])

    WRITE = re.compile(r"cp_connections\.data\s*(\[[^\]]*\])?\s*([*+]?=)")
    writer_flags = set()
    for i, line in enumerate(step):
        if "enable_" not in line or not line.rstrip().endswith(":"):
            continue
        indent = len(line) - len(line.lstrip())
        block = []
        for nxt in step[i + 1:]:
            if not nxt.strip():
                continue
            if (len(nxt) - len(nxt.lstrip())) <= indent:
                break
            block.append(nxt)
        if any(WRITE.search(b) for b in block):
            for tok in re.findall(r'cfg\.enable_([a-z0-9_]+)|"enable_([a-z0-9_]+)"', line):
                writer_flags.add(tok[0] or tok[1])
    missing = sorted(f for f in writer_flags if f not in guard)
    assert not missing, (
        f"flags gating a cp_connections.data write inside the step but absent from the megakernel dispatch guard: {missing}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
