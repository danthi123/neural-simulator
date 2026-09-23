#!/usr/bin/env python3
"""brain_build_sanity.py — build the SAME tiny-demo brain the production webapp/battery builds
(`webapp.server._build_chat_brain('tiny-demo', ...)` -> `research.runners.brain_chat_tui._build_tiny_demo`) and
report its ACTUAL substrate size (neuron + synapse counts), as JSON on stdout.

WHY THIS EXISTS (2026-09-23, part B of the AWS-provisioner fix). Board note 2026-09-22: "AWS Phase 2 ABANDONED
-- AWS env built a DEGENERATE 2-neuron/0-synapse brain (even with h5py/hdf5plugin) -> exercised=0". `nvidia-smi`
proving a GPU exists (aws_provision.sh's own existing check) and a clean pip install say nothing about whether
the ACTUAL substrate the battery builds came out the right size -- a silently-broken import chain, a wrong
CoreSimConfig default, or a missing on-disk dependency can all produce a brain that "builds" (no exception) but
is functionally empty. This is the same "the last check that matters is the one nobody runs" lesson
aws_provision.sh's own cupy-device check already encodes, one layer up the stack: PROVING THE GPU IS REACHABLE
is not proving THE BRAIN BUILT CORRECTLY.

USAGE (both sides of a provision get run with the IDENTICAL command, so the comparison is like-for-like):
    SIM_BACKEND=numpy .venv/bin/python -m tools.brain_build_sanity          # CPU lane (aws_cpu_provision.sh)
    SIM_BACKEND=cupy  .venv/bin/python -m tools.brain_build_sanity          # GPU lane (aws_provision.sh)

Output: {"ok": true, "n_neurons": N, "n_synapses": M, "backend": "numpy"|"cupy", "source": "tiny-demo"}
        or {"ok": false, "error": "<ExceptionType>: <message>"} on any failure (never a bare traceback -- the
        caller compares JSON, not stderr).

HONEST SCOPE: this proves the tiny-demo substrate builds to the SAME size on both ends -- it does NOT prove the
substrate is functionally correct (see `research.runners.brain_chat_tui.run_smoke` for that, the scripted
multi-turn conversation + no-confab-moat check `tools/aws_gate_check.sh` also runs post-provision) or that a
DEVELOPED brain bundle (bridges/developed/.../brain.json) loads correctly (out of scope: the tiny-demo path is
in-code, no bundle needed, by design -- see `_build_tiny_demo`'s own docstring)."""
from __future__ import annotations

import json
import os
import sys


def build_and_measure():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")

    # The tiny-demo brain is NOT one bridge -- MultiTurnAgent composes the main parser/composer substrate with
    # SEVERAL independently-built organs (discourse-event register, biased-competition, world-model/surprise
    # monitor, ...), each its own `sim.bridge.SimulationBridge`. A single `.bridge` attribute undercounts the
    # brain (a live check here first tried that and measured only ONE of three bridges the local build actually
    # made -- 126 neurons instead of the true multi-thousand total). So this records EVERY SimulationBridge this
    # process constructs (a lightweight monkeypatch of `_initialize_simulation_data`, done HERE in tools/, not
    # in sim/ -- no sim/ edit needed) and sums across all of them: a total of ~2 neurons / 0 synapses across the
    # WHOLE process is exactly the AWS degenerate-build symptom regardless of which organ measured it, so
    # summing is the right instrument for "did the substrate build at all", not a precision requirement.
    import sim.bridge as _bridge_mod
    _built = []
    _orig_init_sim_data = _bridge_mod.SimulationBridge._initialize_simulation_data

    def _recording_init(self, *a, **kw):
        r = _orig_init_sim_data(self, *a, **kw)
        _built.append(self)
        return r

    _bridge_mod.SimulationBridge._initialize_simulation_data = _recording_init
    try:
        from research.runners.brain_chat_tui import _build_tiny_demo
        _agent, _aliases, n_facts = _build_tiny_demo(seed=42, use_multiturn=True, enable_neural_render=False,
                                                     composer_kind="rf", integrated_loop=False)
    finally:
        _bridge_mod.SimulationBridge._initialize_simulation_data = _orig_init_sim_data

    seen = set()
    n_neurons = n_synapses = 0
    for b in _built:
        if id(b) in seen:
            continue
        seen.add(id(b))
        n_neurons += int(b.core_config.num_neurons)
        n_synapses += int(getattr(b, "_synapse_count", 0))
    return {
        "ok": True,
        "n_neurons": n_neurons,
        "n_synapses": n_synapses,
        "n_bridges": len(seen),
        "n_facts": int(n_facts),
        "backend": os.environ.get("SIM_BACKEND", "numpy"),
        "source": "tiny-demo",
    }


def main():
    try:
        result = build_and_measure()
    except Exception as e:                                     # noqa: BLE001 -- report, never crash the caller
        result = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
    print(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
