#!/usr/bin/env python
"""Diagnose why the homeostatic synaptic-scaling WRITE is inert on the cupy backend.

WHY THIS RUN EXISTS (2026-10-31, Vikunja #203 follow-up). The cupy confirmation battery
(research/findings/raw/_pi203_frozen_synapse_residual/backend_confirmation_cupy.json) reproduced every rule's
drift EXCEPT `scaling`: on numpy the scaling rule moves BOTH populations by 15.0 (20.0 -> 5.0, the homeostatic
weight clip inside the same block), on cupy it moves NEITHER, in BOTH arms, from the very first read
(first_delta = 0.0). A rule that never fires is not a pass, it is an instrument blind spot: the flag cannot be
said to gate a site the backend never reaches. No step error was logged, so the block is being skipped, not
crashing (sim/bridge.py:12384 guard chain, and the swallow at :12522 that only raises when
bridge.strict_step_errors is set).

This runner builds the SAME 6-neuron scaling bridge as tests/_plastic_mask_beyond_hebbian_scenario.py, sets
strict_step_errors=True so any swallowed exception surfaces, and prints the guard chain's actual values on the
backend it is asked for: enable_synaptic_scaling, _homeostasis_gated equivalent (experiment_engine), nnz, the
type and host contents of cp_connections.data, the cached COO and whether its nnz matches, the activity EMA,
and cp_plasticity_rate_gain. Then it runs 20 steps and reports the weight delta.

Run:  SIM_BACKEND=cupy PYTHONPATH=. python -m research.runners.pi203_scaling_backend_diagnostic \
        --out research/findings/raw/_pi203_frozen_synapse_residual/scaling_backend_diagnostic.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402

from tools.lab import assert_backend, attributable_to, lever, project_cost  # noqa: E402
from tests._plastic_mask_beyond_hebbian_scenario import (  # noqa: E402
    DRIVE_PA, STEPS_PER_READ, _base_cfg, _host, _pairs,
)


def _probe(backend: str, seed: int, scaling_on: bool) -> dict:
    from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
    from sim.backend import get_backend

    _, resolved = get_backend()
    cfg = _base_cfg("scaling", seed)
    cfg.enable_synaptic_scaling = scaling_on      # the control arm: the block's own guard is the manipulation
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    bridge.strict_step_errors = True          # surface the swallowed step error, if there is one

    fp, fq = _pairs((0, 4))
    cp_, cq = _pairs((4, 6))
    bridge.inject_explicit_wiring({
        "frozen_recurrent": {"pre_indices": fp, "post_indices": fq,
                             "initial_weights": np.full(len(fp), 20.0, dtype=np.float32),
                             "plastic": False, "count": len(fp)},
        "control_recurrent": {"pre_indices": cp_, "post_indices": cq,
                              "initial_weights": np.full(len(cp_), 20.0, dtype=np.float32),
                              "plastic": True, "count": len(cp_)},
    })

    before = _host(bridge.cp_connections.data).copy()
    ema_before = bridge.cp_neuron_activity_ema
    coo = bridge._get_cached_coo()

    # Decisive instrument, not inference: which lines of the homeostasis block ACTUALLY execute on this
    # backend. sys.settrace over 20 steps of a 6-neuron bridge is cheap.
    line_hits = {}
    import sys as _sys
    import sim.bridge as _bmod
    _bridge_file = os.path.abspath(_bmod.__file__)
    _gated = {"homeostasis_gated": None, "plasticity_gated": None}

    def _trace(frame, event, arg):
        if event == "line" and frame.f_code.co_filename == _bridge_file:
            if 10903 <= frame.f_lineno <= 10913:
                _gated["homeostasis_gated_seen"] = frame.f_lineno
            if 12320 <= frame.f_lineno <= 12440:
                line_hits[frame.f_lineno] = line_hits.get(frame.f_lineno, 0) + 1
        return _trace

    _sys.settrace(_trace)
    error = None
    fired_steps = 0
    for _ in range(STEPS_PER_READ):
        if bridge.cp_external_input_current is not None:
            bridge.cp_external_input_current[:] = DRIVE_PA
        try:
            bridge._run_one_simulation_step()
        except Exception as exc:  # strict mode: this is the loud path the default UI swallows
            error = f"{type(exc).__name__}: {exc}"
            break
        bridge.runtime_state.current_time_step += 1
        fired_steps += int(_host(bridge.cp_prev_firing_states).sum())
    _sys.settrace(None)

    after = _host(bridge.cp_connections.data).copy()
    coo_after = bridge._get_cached_coo()
    gain = bridge.cp_plasticity_rate_gain
    return {
        "backend_resolved": resolved,
        "seed": seed,
        "enable_synaptic_scaling": bool(cfg.enable_synaptic_scaling),
        "experiment_engine_present": bridge.experiment_engine is not None,
        "connections_type": type(bridge.cp_connections).__module__ + "." + type(bridge.cp_connections).__name__,
        "data_type": type(bridge.cp_connections.data).__module__ + "." + type(bridge.cp_connections.data).__name__,
        "nnz": int(bridge.cp_connections.nnz),
        "coo_cache_at_build": None if coo is None else int(coo.nnz),
        "coo_cache_after_run": None if coo_after is None else int(coo_after.nnz),
        "ema_before_is_none": ema_before is None,
        "homeostasis_block_lines_hit": len(line_hits),
        "homeostasis_block_line_count": sum(line_hits.values()),
        "scaling_guard_line_hit": 12384 in line_hits,
        "ema_update_line_hit": 12391 in line_hits or 12392 in line_hits,
        "scaling_write_line_hit": any(l in line_hits for l in (12418, 12419, 12420, 12421, 12422)),
        "homeostasis_line_hits": dict(sorted(line_hits.items())),
        "ema_before_max": None if ema_before is None else float(np.asarray(_host(ema_before)).max()),
        "ema_after_max": None if bridge.cp_neuron_activity_ema is None
                        else float(np.asarray(_host(bridge.cp_neuron_activity_ema)).max()),
        "rate_gain_is_none": gain is None,
        "rate_gain_sum": None if gain is None else float(np.asarray(_host(gain)).sum()),
        "fired_steps_sum": fired_steps,
        "weight_min_before": float(before.min()), "weight_max_before": float(before.max()),
        "weight_min_after": float(after.min()), "weight_max_after": float(after.max()),
        "max_abs_delta_20_steps": float(np.abs(after - before).max()),
        "hebbian_min_weight": cfg.hebbian_min_weight,
        "hebbian_max_weight": cfg.hebbian_max_weight,
        "homeostasis_target_rate": cfg.homeostasis_target_rate,
        "synaptic_scaling_rate": cfg.synaptic_scaling_rate,
        "dt_ms": cfg.dt_ms,
        "strict_step_error": error,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cupy")
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--prereg", default="research/findings/2026-09-24-plastic-mask-flip-PREREGISTRATION.md",
                    help="the committed pre-registration these seeds belong to (gate guardrail requires it)")
    ap.add_argument("--out", default="research/findings/raw/_pi203_frozen_synapse_residual/scaling_backend_diagnostic.json")
    args = ap.parse_args()
    assert_backend(args.backend, "this diagnostic is about the GPU write path; a numpy run of it proves nothing")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    t0 = time.time()
    rows, first_unit_s = [], None
    for seed in seeds:
        for scaling_on in (False, True):
            t1 = time.time()
            rows.append(_probe(args.backend, seed, scaling_on))
            if first_unit_s is None:
                first_unit_s = round(time.time() - t1, 2)
                project_cost(f"{args.backend} 6-neuron 20-step scaling probe",
                            1, 2 * len(seeds), first_unit_s)

    on = max(r["max_abs_delta_20_steps"] for r in rows if r["enable_synaptic_scaling"])
    off = max(r["max_abs_delta_20_steps"] for r in rows if not r["enable_synaptic_scaling"])
    lever("cfg.enable_synaptic_scaling (ON vs OFF) on the " + args.backend + " backend",
          off, on, required=False)
    attributable_to("weight movement in the scaling rule (scaling ON vs OFF)", on, off)
    inert = sum(1 for r in rows if r["enable_synaptic_scaling"] and r["max_abs_delta_20_steps"] == 0.0)
    record = {
        "runner": "research.runners.pi203_scaling_backend_diagnostic",
        "script": "research/runners/pi203_scaling_backend_diagnostic.py",
        "cmd": " ".join(sys.argv),
        "backend_requested": args.backend,
        "prereg": args.prereg,
        "elapsed_seconds": round(time.time() - t0, 2),
        "question": "On the cupy backend the synaptic-scaling block writes nothing at all. Which guard, or which "
                    "exception, is the reason?",
        "rows": rows,
        "scaling_on_max_delta": on,
        "scaling_off_max_delta": off,
        "summary": f"{inert}/{sum(1 for r in rows if r['enable_synaptic_scaling'])} scaling-ON probes on "
                   f"{args.backend} show zero weight movement in 20 steps (numpy shows 15.0 for the same rule "
                   "and seed).",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
